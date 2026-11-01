"""Distribución del total de puntos en NBA/NFL/NHL/MLB. p_over(league, exp, line) usa data/totals_params.json, que
escribe `python totals_model.py`: backtest sin mirar el futuro (fuerzas recalculadas cada semana con
train_football.lambdas; sd/dispersión ajustada con el año anterior al test; test = 2 últimas temporadas) que elige
vida media, encogimiento k y forma (normal con sd de residuos o binomial negativa) por log-verosimilitud y CRPS."""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PARAMS = Path(__file__).parent / "data" / "totals_params.json"


def cdf(kind, mu, par, t):
    """P(total <= t) con t entero (array) para media mu. normal: par = sd (corrección de continuidad);
    nbinom: par = r (varianza mu + mu²/r)."""
    if kind == "normal": return stats.norm.cdf(t + 0.5, mu, par)
    return stats.nbinom.cdf(t, par, par / (par + mu))


def p_over(league, exp, line, params=None):
    """Prob. de que el total supere la línea (x.5 o entera: en entera el empate se reparte fuera, push)."""
    c = (params or json.loads(PARAMS.read_text()))[league]
    lo = math.floor(line - 1e-9) if line != int(line) else int(line)
    p_le = float(cdf(c["dist"], exp, c["par"], np.array(lo)))
    if line == int(line):  # línea entera: P(over | no push)
        p_push = p_le - float(cdf(c["dist"], exp, c["par"], np.array(lo - 1)))
        return (1 - p_le) / max(1 - p_push, 1e-9)
    return 1 - p_le


def p_cover(league, p_elo, spread, params=None):
    """Prob. de que el LOCAL cubra el hándicap `spread` (formato DraftKings: -5.5 = local da 5.5 puntos).
    Margen ~ normal(a·Φ⁻¹(p_elo) + b, sd), ajustado en data/totals_params.json["spread"] (solo nba y nfl)."""
    c = (params or json.loads(PARAMS.read_text()))["spread"][league]
    mu = c["a"] * stats.norm.ppf(min(max(p_elo, 1e-6), 1 - 1e-6)) + c["b"]
    return float(1 - stats.norm.cdf(-spread, mu, c["sd"]))


def spread_fit(league, L, d, since, end):
    """Margen derivado del Elo (multisport) frente al modelo de puntos: CRPS en las 2 últimas temporadas."""
    import multisport as M
    df = d.rename(columns={"HomeTeam": "home", "AwayTeam": "away"})
    df["res"] = np.where(d.FTHG > d.FTAG, 1.0, np.where(d.FTHG < d.FTAG, 0.0, 0.5))
    pred, P = [], M._P[league]
    M.elo(df.rename(columns={"Date": "date"}), P["hfa"], P["K"], pred, P["keep"])
    m = (d.Date >= since).to_numpy()
    y, z = (d.FTHG - d.FTAG).to_numpy()[m], stats.norm.ppf(np.clip(np.array(pred)[m], 1e-6, 1 - 1e-6))
    test = (d.Date[m] >= end - pd.DateOffset(years=2)).to_numpy()
    mg = L[120, 2][0] - L[120, 2][1]
    ok = mg != 0; fr, T = ~test & ok, test & ok
    a, b = np.polyfit(z[fr], y[fr], 1)
    me = a * z + b; sde, sdp = (y[fr] - me[fr]).std(), (y[fr] - mg[fr]).std()
    def crps(mu, sd, yy):
        u = (yy - mu) / sd
        return sd * (u * (2 * stats.norm.cdf(u) - 1) + 2 * stats.norm.pdf(u) - 1 / np.sqrt(np.pi))
    dd = crps(me[T], sde, y[T]) - crps(mg[T], sdp, y[T])
    print(f"{league} hándicap: margen desde Elo (sd {sde:.1f}) vs modelo de puntos, CRPS {dd.mean():+.3f} ± {dd.std() / np.sqrt(T.sum()):.3f}")
    return {"a": float(a), "b": float(b), "sd": float(sde), "crps_vs_points": round(float(dd.mean()), 3)}


def score(kind, mu, par, y, line):
    """Por partido: (−log-verosimilitud del total exacto, CRPS discreto, log-loss de P(total > line))."""
    t = np.arange(0, int(max(y.max(), mu.max()) * 2 + 20))
    F = cdf(kind, mu[:, None], np.asarray(par)[..., None] if np.ndim(par) else par, t[None, :])
    pm = np.diff(np.concatenate([np.zeros((len(y), 1)), F], 1), axis=1)
    ll = -np.log(np.clip(pm[np.arange(len(y)), y.astype(int)], 1e-12, 1))
    crps = ((F - (t[None, :] >= y[:, None])) ** 2).sum(1)
    po = 1 - F[np.arange(len(y)), np.floor(line).astype(int)]
    po = np.clip(po, 1e-9, 1 - 1e-9); o = (y > line)
    return ll, crps, -(o * np.log(po) + (1 - o) * np.log(1 - po))


def fit_nb(mu, y):
    """r de la binomial negativa por máxima verosimilitud (rejilla); r enorme ≈ Poisson."""
    rs = np.exp(np.linspace(np.log(0.5), np.log(5000), 120))
    return float(max(rs, key=lambda r: stats.nbinom.logpmf(y, r, r / (r + mu)).sum()))


def main():
    import multisport as M, train_football as TF
    hls, ks = [30, 60, 120, 180, 365], [2, 5, 10, 20]
    out = {}
    for league in ("nba", "nfl", "nhl", "mlb"):
        df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
        df = df.dropna(subset=["hs", "as"]).drop_duplicates("id")
        d = pd.DataFrame({"Div": league, "HomeTeam": df.home, "AwayTeam": df.away, "FTHG": df.hs.astype(float), "FTAG": df["as"].astype(float),
                          "Date": pd.to_datetime(df.date, utc=True).dt.tz_localize(None)}).sort_values("Date", ignore_index=True)
        end = d.Date.max()
        since = end - pd.DateOffset(years=3)
        L = TF.lambdas(d, since, hls, ks)
        e = d[d.Date >= since].reset_index(drop=True)
        y = (e.FTHG + e.FTAG).to_numpy()
        test = (e.Date >= end - pd.DateOffset(years=2)).to_numpy()
        fitp = ~test
        # línea independiente del modelo: mediana de los 200 totales anteriores de la liga, a .5
        allt = (d.FTHG + d.FTAG)
        med = allt.rolling(200, min_periods=50).median().shift(1)[d.Date >= since].to_numpy()
        line = np.floor(med) + 0.5
        # base (como multisport hoy): vida media 365, k=5, normal con sd del total bruto de los 400 días previos
        prev = allt[(d.Date < end - pd.DateOffset(years=2)) & (d.Date >= end - pd.DateOffset(years=2) - pd.Timedelta(days=400))]
        mu0 = L[365, 5][0] + L[365, 5][1]
        ok = (mu0 > 0) & np.isfinite(line)
        T = test & ok
        base_a = score("normal", mu0[T], float(prev.std()), y[T], line[T])
        base = [a.mean() for a in base_a]
        print(f"\n{league} ({T.sum()} partidos de test) BASE normal sd bruta {prev.std():.1f}: "
              f"log-verosim. {base[0]:.4f}, CRPS {base[1]:.3f}, log-loss over/under {base[2]:.4f}")
        best = None
        for hl in hls:
            for k in ks:
                mu = L[hl, k][0] + L[hl, k][1]
                F_ = fitp & ok
                cand = [("normal", float((y[F_] - mu[F_]).std()))]
                if league in ("nhl", "mlb"): cand.append(("nbinom", fit_nb(mu[F_], y[F_])))
                for kind, par in cand:
                    s = [a.mean() for a in score(kind, mu[T], par, y[T], line[T])]
                    if best is None or s[0] < best[0][0]: best = (s, hl, k, kind, par)
        s, hl, k, kind, par = best
        se = [(a - b).std() / np.sqrt(T.sum()) for a, b in zip(score(kind, (L[hl, k][0] + L[hl, k][1])[T], par, y[T], line[T]), base_a)]
        print(f"{league} MEJOR vida media {hl} d, k={k}, {kind} par={par:.2f}: log-verosim. {s[0]:.4f}, CRPS {s[1]:.3f}, "
              f"log-loss over/under {s[2]:.4f} (errores típicos pareados: {se[0]:.4f}, {se[1]:.3f}, {se[2]:.4f})")
        if league in ("nba", "nfl"): out.setdefault("spread", {})[league] = spread_fit(league, L, d, since, end)
        out[league] = {"half_life": hl, "k": k, "dist": kind, "par": par, "ll": round(s[0], 4), "ll_old": round(base[0], 4),
                       "crps": round(s[1], 3), "crps_old": round(base[1], 3), "ou": round(s[2], 4), "ou_old": round(base[2], 4),
                       "ou_se": round(float(se[2]), 4)}
    PARAMS.write_text(json.dumps(out))


if __name__ == "__main__":
    P = {"x": {"dist": "normal", "par": 10.0}, "y": {"dist": "nbinom", "par": 1e6}}
    assert abs(p_over("x", 200, 200.5, P) - (1 - stats.norm.cdf(200.5, 200, 10))) < 1e-9
    assert abs(p_over("y", 3.0, 2.5, P) - (1 - stats.poisson.cdf(2, 3.0))) < 1e-3  # r enorme ≈ Poisson
    assert abs(p_cover("x", 0.5, -0.0, {"spread": {"x": {"a": 10, "b": 0, "sd": 12}}}) - 0.5) < 1e-9
    main()
