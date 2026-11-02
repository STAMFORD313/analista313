"""Entrena el Elo de ESPN (NBA, NFL, NHL, MLB): K, ventaja local, cuánto rating se conserva entre temporadas y si el K
depende del margen, elegidos en los 2 años anteriores a la prueba y medidos (log-loss, prob. calculada ANTES de cada
partido) en las 2 últimas temporadas; recalibra el margen del hándicap. Guarda data/espn_params.json (lo lee multisport.py).
`python train_espn.py informe`: log-loss, mezcla y ROI frente a las cuotas previas de DraftKings guardadas."""
import itertools, json, sys

import numpy as np
import pandas as pd
from scipy import stats

import multisport as M
import train_football as TF

BASE = {"nba": (100, 20), "nfl": (55, 20), "nhl": (50, 8), "mlb": (24, 4)}  # (ventaja, K) de partida, para comparar
GRID = {"K": [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80], "hfa": [0, 15, 25, 35, 50, 65, 80, 100, 120],
        "keep": [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]}


def load(league):
    df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)


def logloss(df, hfa, k, val, test, keep=0.75, mov=False):
    """Log-loss (validación, prueba) con la prob. calculada ANTES de cada partido; empates (0.5) cuentan medio."""
    pred = []
    M.elo(df, hfa, k, pred, keep, mov=mov)
    p, y = np.clip(np.array(pred), 1e-6, 1 - 1e-6), df.res.to_numpy(float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(ll[val].mean()), float(ll[test].mean())


def crps(mu, sd, y):
    u = (y - mu) / sd
    return sd * (u * (2 * stats.norm.cdf(u) - 1) + 2 * stats.norm.pdf(u) - 1 / np.sqrt(np.pi))


def spread_fit(df, c, val, test):
    """Margen del local ~ normal(a·Φ⁻¹(p_elo) + b, sd) con el Elo del bot (para totals_model.p_cover): ajuste en validación,
    CRPS en prueba frente a los coeficientes que había."""
    pred = []
    M.elo(df, c["hfa"], c["K"], pred, c["keep"], mov=c["mov"])
    z = stats.norm.ppf(np.clip(np.array(pred), 1e-6, 1 - 1e-6))
    y = (pd.to_numeric(df.hs, errors="coerce") - pd.to_numeric(df["as"], errors="coerce")).to_numpy()
    ok = np.isfinite(y)
    a, b = np.polyfit(z[val & ok], y[val & ok], 1)
    sd = float((y[val & ok] - a * z[val & ok] - b).std())
    T, old = test & ok, M.SPREAD.get(c["league"])
    new = crps(a * z[T] + b, sd, y[T]).mean()
    was = crps(old["a"] * z[T] + old["b"], old["sd"], y[T]).mean() if old else float("nan")
    print(f"  hándicap: CRPS del margen en prueba {was:.3f} → {new:.3f}")
    return {"a": float(a), "b": float(b), "sd": sd, "crps": round(float(new), 3), "crps_old": round(float(was), 3)}


def measure(name, P, odds, y):
    """log-loss modelo vs mercado sin margen, peso óptimo del modelo y ROI (mezcla 0.5, EDGE de analyst)."""
    m = np.isfinite(P).all(1) & np.isfinite(odds).all(1) & np.isfinite(y)
    if m.sum() < 10: return print(f"{name}: {m.sum()} partidos terminados con cuota, aún pocos")
    P, odds, y = P[m], odds[m], y[m].astype(int)
    mk = (1 / odds) / (1 / odds).sum(1, keepdims=True)
    G = np.where(np.arange(P.shape[1])[None, :] == y[:, None], odds - 1, -1.0)
    lls = {w: TF.logloss(w * P + (1 - w) * mk, y) for w in np.round(np.arange(0, 1.01, 0.1), 1)}
    r = TF.roi_g(0.5 * P + 0.5 * mk, odds, G, M.A.EDGE)
    print(f"{name}: {m.sum()} partidos · log-loss modelo {lls[1.0]:.4f} · mercado {lls[0.0]:.4f} · "
          f"mejor peso {min(lls, key=lls.get)} ({min(lls.values()):.4f}) · ROI {r[0]:+.1%} ± {r[2]:.1%} en {r[1]} apuestas")


def market_report():
    """Con las cuotas previas guardadas por multisport.save_odds y los partidos ya terminados: log-loss del modelo y del
    mercado sin margen, peso óptimo del modelo en la mezcla y ROI por liga y mercado."""
    for f in sorted(M.DIR.glob("odds_*.csv")):
        league = f.stem[5:]
        hist = pd.concat([load(league)] + ([load(l) for l in ("ucl", "uel", "uecl") if l != league] if league in M.SOCCER else []))
        o = pd.read_csv(f, dtype={"id": str}).merge(hist[["id", "hs", "as", "res"]].drop_duplicates("id"), on="id")
        col = lambda *c: np.column_stack([pd.to_numeric(o[x], errors="coerce") if x in o else np.full(len(o), np.nan) for x in c])
        hs, as_, res = col("hs")[:, 0], col("as")[:, 0], col("res")[:, 0]
        nan = lambda cond, v: np.where(cond, np.nan, v)
        if "od" in o:  # fútbol: 1X2 con empate
            p = col("p", "pd")
            measure(f"{league} 1X2", np.column_stack([p, 1 - p.sum(1)]), col("oh", "od", "oa"), np.select([res == 1, res == 0.5], [0, 1], 2).astype(float))
        else: measure(f"{league} ganador", np.column_stack([col("p"), 1 - col("p")]), col("oh", "oa"), nan(res == 0.5, (res == 0).astype(float)))
        tl, sl = col("tl")[:, 0], col("sl")[:, 0]
        measure(f"{league} total", np.column_stack([col("p_over"), 1 - col("p_over")]), col("to", "tu"), nan(hs + as_ == tl, (hs + as_ < tl).astype(float)))
        measure(f"{league} hándicap", np.column_stack([col("p_cover"), 1 - col("p_cover")]), col("sh", "sa"), nan(hs - as_ + sl == 0, (hs - as_ + sl < 0).astype(float)))


def main():
    out = json.loads((M.A.DATA / "espn_params.json").read_text())
    for league in BASE:
        df = load(league)
        end = df.date.max()
        test = (df.date >= end - pd.DateOffset(years=2)).to_numpy()
        val = ~test & (df.date >= end - pd.DateOffset(years=4)).to_numpy()  # se elige aquí y se mide en test
        old = logloss(df, BASE[league][0], BASE[league][1], val, test)
        res = {(k, h, kp, mv): logloss(df, h, k, val, test, kp, mv)
               for k, h, kp, mv in itertools.product(GRID["K"], GRID["hfa"], GRID["keep"], (False, True))}
        k, h, kp, mv = min(res, key=lambda c: res[c][0])
        if res[k, h, kp, mv][1] >= old[1]: k, h, kp, mv = BASE[league][1], BASE[league][0], 0.75, False  # no mejora fuera de muestra
        out[league] = c = {"K": k, "hfa": h, "keep": kp, "mov": mv, "logloss": round(res.get((k, h, kp, mv), old)[1], 4), "logloss_old": round(old[1], 4)}
        print(league, c, f"({test.sum()} partidos de prueba; el mejor en prueba sería {min(res, key=lambda c: res[c][1])})")
        if league in ("nba", "nfl"): c["spread"] = spread_fit(df, {**c, "league": league}, val, test)
    (M.A.DATA / "espn_params.json").write_text(json.dumps(out))


if __name__ == "__main__":
    market_report() if sys.argv[1:] == ["informe"] else main()
