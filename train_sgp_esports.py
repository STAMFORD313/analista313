"""Combinada del mismo partido en esports BO3 (CS2, Dota 2, LoL): «X gana + Más de 2.5 mapas» = marcador exacto 2-1.

En un BO3 todas las combinadas del mismo partido salen del marcador (2-0 = hándicap −1.5; ganador + más de 2.5 = 2-1;
ganador + menos de 2.5 = 2-0), así que la correlación es exacta si el marcador está bien calibrado. Aquí se mide,
walk-forward, el log-loss del marcador (4 casillas) en BO3:
  (a) mapas independientes con la prob. por mapa del Elo de producción,
  (b) producción: prob. de ir al 3er mapa recalibrada (esports.bo3_scores, MAPS3 reajustado en el año −2),
  (c) (b) + tendencia de cada equipo a ir al 3er mapa (exceso sobre lo esperado, encogido, calculado solo con series
      anteriores), coeficiente ajustado en el año −2 y medido en el año −1.
Guarda las cifras en data/<juego>/params.json["sgp"] (informativo: bo3_scores sigue con MAPS3).
Uso: PYTHONIOENCODING=utf-8 python train_sgp_esports.py [cs2|dota2|lol]"""
import json, math, sys

import numpy as np
import pandas as pd
from scipy import optimize

import esports as E

KS = [5, 10, 20, 40]  # encogimiento de la tendencia (series de un equipo)
FORGET = 0.98         # olvido por serie del equipo


def logit(x): return np.log(x / (1 - x))


def cells(p, q3):
    """[2-0, 2-1, 1-2, 0-2] desde prob. por mapa p y prob. de 3er mapa q3 (como esports.bo3_scores)."""
    two = 1 - q3; s = p * p + (1 - p) ** 2
    return np.c_[two * p * p / s, q3 * p, q3 * (1 - p), two * (1 - p) ** 2 / s]


def dataset(slug):
    E.configure(slug)
    d = pd.read_csv(E.HIST); d["date"] = pd.to_datetime(d["date"], utc=True); d = d.sort_values("date", ignore_index=True)
    end = d.date.max(); since = end - pd.DateOffset(years=2)
    preds = {}
    E.map_elo(d, since, E.load_games(), preds)
    b = d[(d.bo == 3) & d.id.isin(preds) & d.s1.notna()].copy()
    b = b[((b.s1 == 2) & b.s2.isin([0, 1])) | ((b.s2 == 2) & b.s1.isin([0, 1]))].reset_index(drop=True)
    b["pm"] = [preds[i][1] for i in b.id]
    b["y"] = np.select([(b.s1 == 2) & (b.s2 == 0), (b.s1 == 2) & (b.s2 == 1), (b.s1 == 1) & (b.s2 == 2)], [0, 1, 2], 3)
    b["test"] = b.date >= end - pd.DateOffset(years=1)
    return b


def tendency(b, q3base, k):
    """Por serie: tendencia previa de t1 + t2 a ir al 3er mapa (exceso sobre q3base encogido con k), sin mirar la serie."""
    S, N, out = {}, {}, np.zeros(len(b))
    y3 = b.y.isin([1, 2]).to_numpy(float)
    for i, (t1, t2) in enumerate(zip(b.t1_id, b.t2_id)):
        out[i] = sum(S.get(t, 0.0) / (N.get(t, 0.0) + k) for t in (t1, t2))
        for t in (t1, t2):
            S[t] = S.get(t, 0.0) * FORGET + (y3[i] - q3base[i]); N[t] = N.get(t, 0.0) * FORGET + 1
    return out


def nll(C, y): return -np.log(np.clip(C[np.arange(len(y)), y], 1e-9, 1))


def main(slug):
    b = dataset(slug)
    p, y, te = b.pm.to_numpy(), b.y.to_numpy(), b.test.to_numpy(); fi = ~te
    x = logit(np.clip(2 * p * (1 - p), 1e-6, 1 - 1e-6))
    y3 = np.isin(y, [1, 2]).astype(float)
    print(f"{slug}: {len(b)} series BO3 (ajuste año −2: {fi.sum()}, test año −1: {te.sum()}); 3er mapa real {y3[te].mean():.3f}")
    L_ind = nll(cells(p, 2 * p * (1 - p)), y)

    def fit(X):  # logística de y3 sobre X (columnas), en el año −2
        f = lambda w: -(y3[fi] * np.log(1 / (1 + np.exp(-X[fi] @ w))) + (1 - y3[fi]) * np.log(1 - 1 / (1 + np.exp(-X[fi] @ w)))).sum()
        return optimize.minimize(f, np.r_[0, 1, np.zeros(X.shape[1] - 2)]).x

    X0 = np.c_[np.ones(len(b)), x]; w0 = fit(X0)
    q0 = 1 / (1 + np.exp(-X0 @ w0)); L_b = nll(cells(p, q0), y)
    best = None
    for k in KS:  # k elegido en el año −2
        t = tendency(b, q0, k); X1 = np.c_[X0, t]; w1 = fit(X1)
        q1 = 1 / (1 + np.exp(-X1 @ w1)); L_c = nll(cells(p, q1), y)
        if best is None or L_c[fi].mean() < best[0]: best = (L_c[fi].mean(), k, w1, L_c)
    _, k, w1, L_c = best
    se = lambda a, c: float((a[te] - c[te]).std() / math.sqrt(te.sum()))
    r = {"ind": round(float(L_ind[te].mean()), 4), "prod": round(float(L_b[te].mean()), 4), "team": round(float(L_c[te].mean()), 4),
         "se_team_prod": round(se(L_c, L_b), 4), "se_prod_ind": round(se(L_b, L_ind), 4), "n": int(te.sum())}
    print(f"  log-loss marcador año −1: mapas independientes {r['ind']} · producción {r['prod']} "
          f"({r['prod'] - r['ind']:+.4f} ± {r['se_prod_ind']}) · + tendencia del equipo (k={k}) {r['team']} "
          f"({r['team'] - r['prod']:+.4f} ± {r['se_team_prod']}), coef. {w1[2]:+.3f}")
    # combinada «favorito gana + más de 2.5» (2-1 del favorito) frente al producto ingenuo de las dos probabilidades
    fav = p >= 0.5
    P = cells(p, q0); pj = np.where(fav, P[:, 1], P[:, 2])
    pw = np.where(fav, P[:, 0] + P[:, 1], P[:, 2] + P[:, 3]); po = P[:, 1] + P[:, 2]
    hit = np.where(fav, y == 1, y == 2)
    print(f"  «favorito gana + más de 2.5 mapas» (año −1): prob. conjunta media {pj[te].mean():.3f}, producto ingenuo "
          f"{(pw * po)[te].mean():.3f}, acierto real {hit[te].mean():.3f}")
    r.update(combo_p=round(float(pj[te].mean()), 4), combo_naive=round(float((pw * po)[te].mean()), 4), combo_hit=round(float(hit[te].mean()), 4))
    f = E.DIR / "params.json"; P_ = json.loads(f.read_text())
    P_["sgp"] = r
    r["team_k"], r["team_w"] = k, [round(float(v), 4) for v in w1]  # candidata: aún no la usa esports.bo3_scores
    f.write_text(json.dumps(P_))
    return r


if __name__ == "__main__":
    for s in [a for a in sys.argv[1:] if a in E.DISCIPLINES] or ["cs2", "dota2", "lol"]: main(s)
