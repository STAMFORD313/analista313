"""Quitar el margen de la casa: proporcional (lo de siempre) o método de Shin (INVESTIGACION_GITHUB.md, mejora n.º 2;
penaltyblog / mberk/shin, MIT): el margen se carga más en las sorpresas (favorite-longshot bias).
`python devig.py` → log-loss del 1X2 en las 2 últimas temporadas de data/hist con cuotas medias (Avg) y de Pinnacle."""
import numpy as np


def proportional(odds):
    q = 1 / np.asarray(odds, float)
    return q / q.sum(-1, keepdims=True)


def shin(odds, iters=60):
    """Probabilidades de Shin (vectorizado por filas): p_i = (√(z² + 4(1−z)·q_i²/S) − z) / (2(1−z)), z por bisección
    para que sumen 1."""
    q = 1 / np.atleast_2d(np.asarray(odds, float)); S = q.sum(1, keepdims=True)
    lo, hi = np.zeros((len(q), 1)), np.full((len(q), 1), 0.4)
    f = lambda z: ((np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / S) - z) / (2 * (1 - z))).sum(1, keepdims=True) - 1
    for _ in range(iters):
        mid = (lo + hi) / 2
        pos = f(mid) > 0  # con margen, la suma baja al subir z
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    z = (lo + hi) / 2
    p = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / S) - z) / (2 * (1 - z))
    p = p / p.sum(1, keepdims=True)
    return p[0] if np.ndim(odds) == 1 else p


if __name__ == "__main__":
    import math
    import pandas as pd
    import analyst as A
    cols = ["AvgH", "AvgD", "AvgA", "PSH", "PSD", "PSA", "BbAvH", "BbAvD", "BbAvA"]
    df = pd.concat([A.load_csv(f.read_bytes(), ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *cols]) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True)
    df = df[df.Date >= df.Date.max() - pd.DateOffset(years=2)].dropna(subset=["FTHG", "FTAG"])
    y = np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], [0, 1], 2)
    for name, c in (("medias (Avg)", ["AvgH", "AvgD", "AvgA"]), ("Pinnacle", ["PSH", "PSD", "PSA"])):
        o = df[c].apply(pd.to_numeric, errors="coerce").to_numpy(); ok = np.isfinite(o).all(1) & (o > 1).all(1)
        l = lambda P: -np.log(np.clip(P[np.arange(ok.sum()), y[ok]], 1e-9, 1))
        lp, ls = l(proportional(o[ok])), l(shin(o[ok]))
        print(f"{name}: {ok.sum()} partidos · log-loss proporcional {lp.mean():.5f} → Shin {ls.mean():.5f} "
              f"({ls.mean() - lp.mean():+.5f} ± {(ls - lp).std() / math.sqrt(ok.sum()):.5f})")
