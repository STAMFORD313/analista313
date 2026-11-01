"""Backtest de features extra (features_extra.py) sobre el Poisson actual y sobre el mercado, sin mirar el futuro:
una regresión logística multinomial se ajusta con el primer año y se evalúa en los 2 siguientes (log-loss 1X2 y ROI
a la cuota media, misma regla que train_football.roi). Uso: PYTHONIOENCODING=utf-8 python backtest_datos.py"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import analyst as A
import features_extra as FX
import train_football as TF

TF.ODDS = TF.ODDS + ["HST", "AST"]  # load() las pide junto a las cuotas (solo ligas de data/hist las tienen)


def lr(Xtr, ytr, Xte):
    return LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)


def main():
    df = TF.load()
    F = FX.features(df)
    since = df.Date.max() - pd.DateOffset(years=3)
    L = TF.lambdas(df, since, [A.HALF_LIFE], [A.SHRINK])[A.HALF_LIFE, A.SHRINK]
    e = df[df.Date >= since].reset_index(drop=True)
    F = F[df.Date >= since].reset_index(drop=True)
    P = TF.probs_1x2(L[0], L[1], A.RHO)
    mk = 1 / e[["mH", "mD", "mA"]].to_numpy(float); mk /= mk.sum(1, keepdims=True)
    odds = e[["oH", "oD", "oA"]].to_numpy(float)
    y = np.select([e.FTHG > e.FTAG, e.FTHG == e.FTAG], [0, 1], 2)
    ok = (L[0] > 0) & np.isfinite(mk).all(1) & np.isfinite(odds).all(1) & F[["elo_diff", "rest_diff"]].notna().all(1).to_numpy()
    tr = ok & (e.Date < since + pd.DateOffset(years=1)).to_numpy()
    te = ok & ~tr
    lp, lm = np.log(np.clip(P, 1e-9, 1)), np.log(mk)
    has_sot = F.sot_diff.notna().to_numpy()
    std = {c: float(F[c].std()) for c in FX.FEATS}
    D = FX.design(P, F, std)
    feats = {c: D[:, [3 + i]] for i, c in enumerate([*FX.FEATS, "sot_disp"])}  # sot_disp: distingue "sin datos de tiros"
    print(f"entreno {tr.sum()} partidos, test {te.sum()} ({(te & has_sot).sum()} con tiros a puerta)")
    sets = {"poisson": [lp], "poisson+elo": [lp, feats["elo_diff"]], "poisson+tiros": [lp, feats["sot_diff"], feats["sot_disp"]],
            "poisson+descanso": [lp, feats["rest_diff"]], "poisson+todo": [lp, *feats.values()],
            "mercado": [lm], "mercado+poisson": [lm, lp], "mercado+poisson+elo": [lm, lp, feats["elo_diff"]],
            "mercado+poisson+tiros": [lm, lp, feats["sot_diff"], feats["sot_disp"]],
            "mercado+poisson+descanso": [lm, lp, feats["rest_diff"]], "mercado+poisson+todo": [lm, lp, *feats.values()]}
    res, Qs = {}, {}
    for name, cols in sets.items():
        X = np.hstack(cols)
        Q = Qs[name] = lr(X[tr], y[tr], X[te])
        ll = TF.logloss(Q, y[te]); ll_s = TF.logloss(Q[has_sot[te]], y[te][has_sot[te]])
        r = [TF.roi(Q, mk[te], odds[te], y[te], 1.0, ed) for ed in (0.03, 0.05)]
        res[name] = ll
        print(f"{name:26s} log-loss {ll:.4f} (ligas con tiros {ll_s:.4f}) | ROI edge3% {r[0][0]:+.2%} en {r[0][1]} "
              f"(±{r[0][2]:.1%}), edge5% {r[1][0]:+.2%} en {r[1][1]}")
    nll = lambda Q: -np.log(np.clip(Q[np.arange(te.sum()), y[te]], 1e-9, 1))
    for name, Q in (("poisson crudo", P[te]), ("poisson+todo", Qs["poisson+todo"])):  # como lo usa el bot: mezcla con el mercado
        M = A.BLEND * Q + (1 - A.BLEND) * mk[te]
        r = TF.roi(Q, mk[te], odds[te], y[te], A.BLEND, A.EDGE)
        print(f"mezcla {A.BLEND} {name:14s} log-loss {nll(M).mean():.4f} | ROI edge {A.EDGE} {r[0]:+.2%} en {r[1]} (±{r[2]:.1%})")
    d = nll(Qs["poisson+todo"]) - nll(P[te])
    print(f"mejora poisson+todo vs poisson crudo: {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f} (error típico pareado)")
    m = LogisticRegression(C=10, max_iter=2000).fit(D[ok], y[ok])  # coeficientes para el bot: todo el periodo
    coef = {"std": std, "coef": m.coef_.tolist(), "intercept": m.intercept_.tolist(), "n": int(ok.sum())}
    assert np.allclose(FX.adjust(P[ok][:50], F[ok][:50], coef), m.predict_proba(D[ok][:50]))
    FX.COEF.write_text(json.dumps(coef))
    return res


if __name__ == "__main__":
    main()
