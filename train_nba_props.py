"""Props de jugador NBA: puntos, rebotes, asistencias y triples (más/menos), walk-forward.

Datos: data/nba/box.jsonl (boxscores ESPN con REB, AST y triples desde nba_players.EXTRA). Modelo por jugador, solo
con partidos anteriores: minutos esperados = media con olvido de sus minutos (partidos jugados) y producción por
minuto con olvido encogida hacia la media de la liga con K minutos; esperado = ritmo × minutos. Binomial negativa con
r ajustada. Referencia (lo que haría un apostador sin modelo): media de sus últimos 10 partidos.
Olvido, K y r elegidos en la temporada anterior; medido en la última (log-loss más/menos a una línea tipo casa =
mediana de sus 10 anteriores, a .5). Solo jugadores con ≥ 10 partidos previos y ≥ 15 minutos ese día.
Uso: PYTHONIOENCODING=utf-8 python train_nba_props.py → data/nba/props_params.json"""
import itertools, json, math

import numpy as np
import pandas as pd
from scipy import optimize, stats

import analyst as A
import nba_players as N

OUT = A.DATA / "nba" / "props_params.json"
STATS = {"puntos": 3, "rebotes": 7, "asistencias": 8, "triples": 9}


def rows():
    out = []
    for g in N.load_box("nba"):
        for t in g.get("teams", []):
            for p in t["players"]:
                if len(p) >= N.N_COLS and p[2] > 0:
                    out.append((g["date"], g["id"], p[0], p[1], t["team"], p[2], *[p[i] for i in STATS.values()]))
    d = pd.DataFrame(out, columns=["date", "game", "pid", "name", "team", "min", *STATS])
    d["date"] = pd.to_datetime(d.date, utc=True)
    return d.sort_values(["date", "game"], ignore_index=True)


def features(d, a_min, a_rate, k):
    """Por fila (antes del partido): minutos esperados, ritmo por minuto de cada estadística, nº de partidos previos."""
    S, out = {}, {c: np.zeros(len(d)) for c in ["e_min", "n", *[f"r_{s}" for s in STATS]]}
    glob = {s: [0.0, 0.0] for s in STATS}
    for i, row in enumerate(d.itertuples(index=False)):
        st = S.get(row.pid)
        out["n"][i] = st["n"] if st else 0
        out["e_min"][i] = st["min"] if st else row.min
        for s in STATS:
            g = glob[s][0] / glob[s][1] if glob[s][1] else 0.0
            x, m = (st[s] if st else (0.0, 0.0))
            out[f"r_{s}"][i] = (x + k * g) / (m + k)
        if st is None: st = S[row.pid] = {"n": 0, "min": row.min, **{s: (0.0, 0.0) for s in STATS}}
        st["n"] += 1; st["min"] += a_min * (row.min - st["min"])
        for s in STATS:
            v = getattr(row, s); x, m = st[s]
            st[s] = (x * (1 - a_rate) + v, m * (1 - a_rate) + row.min)
            glob[s][0] += v; glob[s][1] += row.min
    return pd.DataFrame(out, index=d.index)


def nb_ll(y, mu, r): return stats.nbinom.logpmf(y, r, r / (r + np.maximum(mu, 1e-6)))


def ou(y, mu, r, line):
    p = np.clip(1 - stats.nbinom.cdf(np.floor(line), r, r / (r + np.maximum(mu, 1e-6))), 1e-9, 1 - 1e-9); o = y > line
    return -(o * np.log(p) + (1 - o) * np.log(1 - p)), p


def main():
    d = rows()
    end = d.date.max()
    te = (d.date >= end - pd.DateOffset(months=11)).to_numpy()
    fi = ((d.date < end - pd.DateOffset(months=11)) & (d.date >= end - pd.DateOffset(months=23))).to_numpy()
    grp = d.groupby("pid")
    base = {s: grp[s].transform(lambda x: x.shift(1).rolling(10, min_periods=10).mean()).to_numpy() for s in STATS}
    line = {s: (np.floor(grp[s].transform(lambda x: x.shift(1).rolling(10, min_periods=10).median())) + 0.5).to_numpy() for s in STATS}
    best = None
    for a_min, a_rate, k in itertools.product([0.1, 0.2], [0.02, 0.05, 0.1], [50, 200]):
        f = features(d, a_min, a_rate, k)
        ok = (f.n >= 10).to_numpy() & (d["min"] >= 15).to_numpy()
        score = 0.0
        for s in STATS:
            mu = (f[f"r_{s}"] * f.e_min).to_numpy(); m = fi & ok
            r = math.exp(optimize.minimize_scalar(lambda lr: -nb_ll(d[s].to_numpy()[m], mu[m], math.exp(lr)).sum(), bounds=(0, 6), method="bounded").x)
            score += -nb_ll(d[s].to_numpy()[m], mu[m], r).mean()
        if best is None or score < best[0]: best = (score, a_min, a_rate, k, f, ok)
    _, a_min, a_rate, k, f, ok = best
    res = {"a_min": a_min, "a_rate": a_rate, "k": k, "stats": {}}
    print(f"NBA props: olvido minutos {a_min}, ritmo {a_rate}, K {k} min; test última temporada (≥15 min, ≥10 partidos previos)")
    for s in STATS:
        y = d[s].to_numpy(); mu = (f[f"r_{s}"] * f.e_min).to_numpy()
        F_, T_ = fi & ok & np.isfinite(base[s]), te & ok & np.isfinite(base[s]) & np.isfinite(line[s])
        r = math.exp(optimize.minimize_scalar(lambda lr: -nb_ll(y[F_], mu[F_], math.exp(lr)).sum(), bounds=(0, 6), method="bounded").x)
        r0 = math.exp(optimize.minimize_scalar(lambda lr: -nb_ll(y[F_], base[s][F_], math.exp(lr)).sum(), bounds=(0, 6), method="bounded").x)
        l1, p1 = ou(y[T_], mu[T_], r, line[s][T_]); l0, _ = ou(y[T_], base[s][T_], r0, line[s][T_])
        se = float((l1 - l0).std() / math.sqrt(T_.sum()))
        conf = np.maximum(p1, 1 - p1); hit = (y[T_] > line[s][T_]) == (p1 >= 0.5); sel = conf >= 0.6
        res["stats"][s] = {"r": round(r, 2), "n_test": int(T_.sum()), "ou": round(float(l1.mean()), 4), "ou_ref": round(float(l0.mean()), 4),
                           "se": round(se, 4), "picks60": [int(sel.sum()), round(float(conf[sel].mean()), 3), round(float(hit[sel].mean()), 3)]}
        print(f"  {s}: más/menos a la línea tipo casa: media 10 {l0.mean():.4f} → modelo {l1.mean():.4f} ({l1.mean() - l0.mean():+.4f} ± {se:.4f}); "
              f"picks ≥60 %: {sel.sum()} · {conf[sel].mean():.3f} → {hit[sel].mean():.3f}")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
