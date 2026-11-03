"""Cuotas previas como feature de las estadísticas de partido (mejora n.º 1 de INVESTIGACION_GITHUB.md).

Esperado de cada equipo (football_stats + forma) × exp(a·s + b·g), con s = supremacía del equipo según el 1X2 medio
sin margen (P(gana) − P(pierde)) y g = logit P(más de 2.5) sin margen − su media. a, b por estadística por máxima
verosimilitud binomial negativa en el año −3; medido en las 2 últimas temporadas (más/menos a la línea mediana, de
equipo y total). Cuotas de apertura medias (AvgH… o BbAvH…), que el bot tiene antes del partido en fixtures.csv.
Guarda data/football_stats.json["market"][stat] = {a, b} si mejora > 2 errores típicos (lo usa football_stats).
Uso: PYTHONIOENCODING=utf-8 python train_market_stats.py"""
import json, math

import numpy as np
import pandas as pd
from scipy import optimize, stats

import analyst as A
import football_stats as FS
import train_form as TFm
import train_sgp as S
import totals_model as T

ODDS = {"H": ["AvgH", "BbAvH"], "D": ["AvgD", "BbAvD"], "A": ["AvgA", "BbAvA"], "O": ["Avg>2.5", "BbAv>2.5"], "U": ["Avg<2.5", "BbAv<2.5"]}


def market_features(df):
    """(s_local, g) por fila: supremacía del local sin margen y logit de P(más de 2.5) sin margen; NaN sin cuotas."""
    o = {k: pd.concat([pd.to_numeric(df.get(c), errors="coerce") for c in cs if c in df], axis=1).bfill(axis=1).iloc[:, 0]
         if any(c in df for c in cs) else pd.Series(np.nan, index=df.index) for k, cs in ODDS.items()}
    ih, idr, ia = 1 / o["H"], 1 / o["D"], 1 / o["A"]; t = ih + idr + ia
    io, iu = 1 / o["O"], 1 / o["U"]
    po = (io / (io + iu)).clip(0.02, 0.98)
    return ((ih - ia) / t).to_numpy(), np.log(po / (1 - po)).to_numpy()


def main():
    V, test, fit, e = S.load(frame=True)
    cols = sorted({c for cs in ODDS.values() for c in cs})
    raw = pd.concat([A.load_csv(f.read_bytes(), ["Div", "Date", "HomeTeam", "AwayTeam", *cols]) for f in sorted(A.HIST.glob("*.csv"))],
                    ignore_index=True).drop_duplicates(["Div", "Date", "HomeTeam", "AwayTeam"])
    m = e[["Div", "Date", "HomeTeam", "AwayTeam"]].merge(raw, how="left", on=["Div", "Date", "HomeTeam", "AwayTeam"])
    s_h, g = market_features(m)
    g = g - np.nanmean(g[fit])
    has = np.isfinite(s_h) & np.isfinite(g)
    print(f"partidos con cuotas: ajuste {(fit & has).sum()}/{fit.sum()}, test {(test & has).sum()}/{test.sum()}")
    P = json.loads(FS.PARAMS.read_text()); P.setdefault("market", {})
    keep = (e.Date >= e.Date.min()).to_numpy()
    d_all = FS.prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + FS.COLS) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True))
    d_all = d_all.dropna(subset=["FTHG", "FTAG", *FS.COLS[:4], "HF", "AF", "HST", "AST"]).sort_values("Date", ignore_index=True)
    k3 = (d_all.Date >= d_all.Date.max() - pd.DateOffset(years=3)).to_numpy()
    for st, (h, a, name) in FS.STATS.items():
        (mh, rt, yh, lh), (ma, _, ya, la), (_, r, yt, lt) = V[f"{st}_h"], V[f"{st}_a"], V[f"{st}_t"]
        c = P.get("form", {}).get(st, {}).get("c")  # la forma ya está en producción: se parte de ella
        if c:
            fh, fa = (x[k3] for x in TFm.form_ratio(d_all, h, a)); mh, ma = mh * np.exp(c * fh), ma * np.exp(c * fa)
        F_, T_ = fit & has, test & has
        x_h, x_a = np.c_[s_h, g], np.c_[-s_h, g]
        nll = lambda w, M: -(stats.nbinom.logpmf(yh[M], rt, rt / (rt + mh[M] * np.exp(x_h[M] @ w)))
                             + stats.nbinom.logpmf(ya[M], rt, rt / (rt + ma[M] * np.exp(x_a[M] @ w)))).sum()
        w = optimize.minimize(lambda w: nll(w, F_), [0.0, 0.0], method="Nelder-Mead").x
        mh1, ma1 = mh * np.exp(np.nan_to_num(x_h) @ w), ma * np.exp(np.nan_to_num(x_a) @ w)
        team0 = np.r_[TFm.ou_ll(mh, rt, yh, lh)[T_], TFm.ou_ll(ma, rt, ya, la)[T_]]
        team1 = np.r_[TFm.ou_ll(mh1, rt, yh, lh)[T_], TFm.ou_ll(ma1, rt, ya, la)[T_]]
        tot0, tot1 = TFm.ou_ll(mh + ma, r, yt, lt)[T_], TFm.ou_ll(mh1 + ma1, r, yt, lt)[T_]
        se = lambda x, y: float((x - y).std() / np.sqrt(len(x)))
        res = {"a": round(float(w[0]), 4), "b": round(float(w[1]), 4), "g_mean": round(float(np.nanmean(market_features(m)[1][fit])), 4),
               "team": [round(float(team0.mean()), 4), round(float(team1.mean()), 4), round(se(team1, team0), 4)],
               "total": [round(float(tot0.mean()), 4), round(float(tot1.mean()), 4), round(se(tot1, tot0), 4)]}
        ok = team1.mean() < team0.mean() - 2 * res["team"][2] and tot1.mean() <= tot0.mean() + res["total"][2]
        print(f"{name}: a={w[0]:+.3f} (supremacía) b={w[1]:+.3f} (goles) · de equipo {team0.mean():.4f} → {team1.mean():.4f} "
              f"({team1.mean() - team0.mean():+.4f} ± {res['team'][2]}) · total {tot0.mean():.4f} → {tot1.mean():.4f} "
              f"({tot1.mean() - tot0.mean():+.4f} ± {res['total'][2]}){' → se usa' if ok else ''}")
        if ok: P["market"][st] = res
        else: P["market"].pop(st, None)
    FS.PARAMS.write_text(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
