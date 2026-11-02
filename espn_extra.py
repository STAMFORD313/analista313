"""Competiciones de ESPN con cuotas DraftKings que el bot aún no usa: college football (FBS), college basketball (D-I)
y Champions / Europa / Conference League. history(league) descarga y guarda el historial en data/espn/<liga>/ (mismo
formato que multisport) reutilizando multisport.parse/get. `python espn_extra.py` descarga y hace el backtest:
Elo binario (NCAA, con la misma función multisport.elo) y Elo con empate para el fútbol europeo (prob. de empate
por logística ordinal ajustada con los 2 años previos al test)."""
import itertools, json, time

import numpy as np
import pandas as pd
from scipy import stats

import multisport as M

EXTRA = {  # liga: (ruta ESPN, nombre, deporte, parámetros extra del scoreboard, paso de descarga)
    "cfb": ("football/college-football", "🏈 NCAA Football", "fútbol americano", {"groups": 80}, "M"),
    "cbb": ("basketball/mens-college-basketball", "🏀 NCAA Basketball", "baloncesto", {"groups": 50}, "D"),
    "ucl": ("soccer/uefa.champions", "🏆 Champions League", "fútbol", {}, "M"),
    "uel": ("soccer/uefa.europa", "🏆 Europa League", "fútbol", {}, "M"),
    "uecl": ("soccer/uefa.europa.conf", "🏆 Conference League", "fútbol", {}, "M"),
}
PARAMS = M.A.DATA / "espn_extra_params.json"
SOCCER = ("ucl", "uel", "uecl")


def history(league, today=None, years=M.YEARS):
    """Como multisport.update_history pero con parámetros del scoreboard. ESPN recorta a 25 partidos si limit=1000 en
    college football reciente: se pide limit=300 y, si un mes llega a 250 partidos (o el paso es diario, college
    basketball), se descarga día a día. Meses cerrados se descargan una vez."""
    path, _, _, extra, step = EXTRA[league]
    today = today or pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    d = M.DIR / league
    d.mkdir(parents=True, exist_ok=True)
    def fetch(day):
        time.sleep(0.2)  # respeta la API pública
        try: return M.parse(M.get(f"{path}/scoreboard", dates=day, limit=300, **extra))
        except (M.requests.RequestException, ValueError, KeyError) as e: print("ESPN", league, day, e); return []
    months = pd.period_range(today - pd.DateOffset(years=years), today, freq="M")
    for i, m in enumerate(months):
        f = d / f"{m.strftime('%Y%m')}.csv"
        if f.exists() and i < len(months) - 2: continue
        rows = fetch(m.strftime("%Y%m")) if step == "M" else []
        if step == "D" or len(rows) >= 250:
            rows = [r for x in pd.date_range(m.start_time, m.end_time, freq="D") for r in fetch(x.strftime("%Y%m%d"))]
        pd.DataFrame([r for r in rows if r["res"] is not None], columns=M.COLS).drop_duplicates("id").to_csv(f, index=False)
    return load(league)


def load(league):
    d = M.DIR / league
    df = pd.concat([x for f in sorted(d.glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)


def moneyline3(odds):
    """Cuotas decimales (local, empate, visitante) del 1X2 de DraftKings en fútbol, o None."""
    try:
        ml = odds[0]["moneyline"]
        return tuple(M.american(ml[s]["close"]["odds"]) for s in ("home", "draw", "away"))
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def soccer_probs(p_elo, c):
    """Prob. del Elo binario (empate = medio punto) → (H, D, A) con logística ordinal: z = a·logit(p) y cortes ±cut."""
    z = c["a"] * np.log(np.clip(p_elo, 1e-6, 1 - 1e-6) / (1 - np.clip(p_elo, 1e-6, 1 - 1e-6)))
    pa = 1 / (1 + np.exp(z - c["lo"]))
    ph = 1 / (1 + np.exp(-(z - c["hi"])))
    return np.column_stack([ph, 1 - ph - pa, pa])


def fit_ordinal(p, y):
    """Ajusta a, lo, hi (lo < hi) por máxima verosimilitud; y: 0=H, 1=D, 2=A."""
    from scipy.optimize import minimize
    def nll(v):
        c = {"a": v[0], "lo": v[1], "hi": v[1] + np.exp(v[2])}
        P = soccer_probs(p, c)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).sum()
    v = minimize(nll, [1.0, -0.6, 0.0], method="Nelder-Mead").x
    return {"a": float(v[0]), "lo": float(v[1]), "hi": float(v[1] + np.exp(v[2]))}


def main():
    out = {}
    for league in EXTRA: history(league, years=5 if league == "cbb" else M.YEARS)  # cbb: descarga diaria, 5 años bastan
    soc = pd.concat([load(l).assign(lg=l) for l in SOCCER]).sort_values("date", ignore_index=True)  # un Elo común: los equipos cruzan torneos
    for league in EXTRA:
        df = soc if league in SOCCER else load(league)
        end = df.date.max()
        y, ev = df.res.to_numpy(float), (df.date >= end - pd.DateOffset(years=2)).to_numpy()
        val = ~ev & (df.date >= end - pd.DateOffset(years=4)).to_numpy()
        lg = (df.lg == league).to_numpy() if league in SOCCER else np.ones(len(df), bool)
        ll = lambda p, m: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))[m]
        res = {}
        for k, hfa, keep, mov in itertools.product([8, 16, 32, 48, 64, 96, 128], [0, 25, 50, 75, 100, 130], [0.5, 0.75, 1.0], [False, True]):
            pred = []
            try: M.elo(df, hfa, k, pred, keep, mov=mov)
            except OverflowError: continue  # K alto con margen: los ratings divergen
            res[k, hfa, keep, mov] = (ll(np.array(pred), val & lg).mean(), np.array(pred))
        k, hfa, keep, mov = min(res, key=lambda c: res[c][0])
        p = res[k, hfa, keep, mov][1]
        base = ll(np.full(len(y), y[val & lg].mean()), ev & lg)  # referencia: frecuencia de victoria local
        d = ll(p, ev & lg) - base
        o = {"K": k, "hfa": hfa, "keep": keep, "mov": mov, "logloss": round(float(ll(p, ev & lg).mean()), 4),
             "logloss_naive": round(float(base.mean()), 4), "n_test": int((ev & lg).sum())}
        print(f"{league}: Elo K {k}, ventaja {hfa}, keep {keep}, margen {mov}: log-loss {o['logloss']:.4f} vs ingenuo {o['logloss_naive']:.4f} "
              f"({d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}, {o['n_test']} partidos)")
        if league in SOCCER:
            y3 = np.select([df.hs > df["as"], df.hs == df["as"]], [0, 1], 2)
            # empates: logística ordinal con las 3 competiciones juntas y los 365 días previos, reajustada cada mes
            # (con 2 años de una sola competición sobreestimaba los empates: 20 % previsto vs 15 % real en Champions)
            idx, mth = np.where(ev & lg)[0], df.date.dt.to_period("M").to_numpy()
            P = np.empty((len(idx), 3))
            for m in np.unique(mth[idx]):
                sel = mth[idx] == m
                t0 = df.date[idx[sel][0]]
                tr = ((df.date < t0) & (df.date >= t0 - pd.DateOffset(days=365))).to_numpy()
                P[sel] = soccer_probs(p[idx[sel]], fit_ordinal(p[tr], y3[tr]))
            c = fit_ordinal(p[(df.date >= end - pd.DateOffset(days=365)).to_numpy()],
                            y3[(df.date >= end - pd.DateOffset(days=365)).to_numpy()])  # para el bot: último año
            l3 = -np.log(np.clip(P[np.arange(len(P)), y3[ev & lg]], 1e-9, 1))
            fr = np.bincount(y3[val & lg], minlength=3) / (val & lg).sum()
            n3 = -np.log(fr[y3[ev & lg]])
            o.update(c, logloss_1x2=round(float(l3.mean()), 4), logloss_1x2_naive=round(float(n3.mean()), 4))
            print(f"   1X2: log-loss {l3.mean():.4f} vs ingenuo {n3.mean():.4f} ({(l3 - n3).mean():+.4f} ± {(l3 - n3).std() / np.sqrt(len(l3)):.4f}); "
                  f"empates previstos {P[:, 1].mean():.3f} reales {(y3[ev & lg] == 1).mean():.3f}")
        out[league] = o
    PARAMS.write_text(json.dumps(out))


if __name__ == "__main__":
    P = soccer_probs(np.array([0.5]), {"a": 1.0, "lo": -0.5, "hi": 0.5})
    assert np.allclose(P.sum(), 1) and np.isclose(P[0, 0], P[0, 2])
    main()
