"""Entrena la regla de las combinadas (bankroll.py) sin mirar el futuro.

Datos: fútbol (27 ligas, 3 últimas temporadas, prob. del modelo walk-forward mezclada con el mercado como en vivo y cuota
media real) + todas las señales liquidadas de todos los deportes en data/senales.csv (NBA, NFL, NHL, MLB, esports...).
Regla: 1 combinada al día con las `legs` selecciones de más prob. de partidos distintos entre las que cumplen:
  · origen "valor": la condición de las simples (EV >= EDGE + Z_UNC·|p − p_mercado|·cuota y <= MAX_EDGE)
  · origen "ev0":   EV >= 0 (sin margen), para dejar entrar favoritos seguros
  y prob. >= min_p. Ganancia por unidad = producto de cuotas − 1 si aciertan todas, si no −1.
Se elige en el año −3/−2 (con >= 100 combinadas) y se mide en el año −1. Solo se cambia la regla actual si el año −1
no es peor; si la mejor regla es claramente negativa (ROI + 2·e.e. < 0) se apaga el stake a la mitad.
Guarda data/combo_params.json, que lee bankroll.py. Uso: PYTHONIOENCODING=utf-8 python train_combos.py"""
import itertools, json, math

import numpy as np
import pandas as pd

import analyst as A
import train_football as TF

OUT = A.DATA / "combo_params.json"
Z_UNC = 0.25  # igual que bankroll.Z_UNC
DEFAULT = {"source": "valor", "legs": 3, "min_p": 0.55, "stake": 0.02}
GRID = {"source": ["valor", "ev0"], "legs": [2, 3, 4], "min_p": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]}


def football_legs(w=None):
    """w: peso del modelo (None = el de football_params.json, hoy 0 → solo mercado, sin valor; 0.5 = diagnóstico). Una fila por resultado 1X2 de cada partido de las 3 últimas temporadas: fecha, partido, p, p_mercado, cuota, acierto."""
    df = TF.load()
    since = df.Date.max() - pd.DateOffset(years=3)
    L = TF.lambdas(df, since, [A.HALF_LIFE], [A.SHRINK])[A.HALF_LIFE, A.SHRINK]
    e = df[df.Date >= since].reset_index(drop=True)
    mk = 1 / e[["mH", "mD", "mA"]].to_numpy(float); mk /= mk.sum(1, keepdims=True)
    odds = e[["oH", "oD", "oA"]].to_numpy(float)
    w = A.W["1x2"] if w is None else w
    P = w * TF.probs_1x2(L[0], L[1], A.RHO) + (1 - w) * mk
    y = np.select([e.FTHG > e.FTAG, e.FTHG == e.FTAG], [0, 1], 2)
    ok = (L[0] > 0) & np.isfinite(mk).all(1) & np.isfinite(odds).all(1)
    rows = []
    for j in range(3):
        rows.append(pd.DataFrame({"fecha": e.Date.dt.normalize(), "partido": np.arange(len(e)), "p": P[:, j], "implied": mk[:, j],
                                  "odds": odds[:, j], "gano": (y == j).astype(int), "deporte": "⚽"})[ok])
    return pd.concat(rows, ignore_index=True), since


def live_legs():
    d = pd.read_csv(A.DATA / "senales.csv")
    d = d[d.retorno.isin([0.0]) | (d.gano == 1)]  # fuera nulas y medias (hándicap) para no inventar el cobro
    d = d.assign(fecha=pd.to_datetime(d.fecha), partido=d.index)  # sin clave de partido: cada señal cuenta como distinta
    return d[["fecha", "partido", "p", "implied", "odds", "gano", "deporte"]]


def candidates(d, source, min_p):
    ev = d.p * d.odds - 1
    marg = A.EDGE + Z_UNC * (d.p - d.implied).abs() * d.odds if source == "valor" else 0.0
    return d[(ev >= marg) & (ev <= A.MAX_EDGE) & (d.p >= min_p)]


def combos(d, rule):
    """Ganancia por unidad de cada combinada diaria de la regla."""
    c = candidates(d, rule["source"], rule["min_p"]).sort_values("p", ascending=False)
    out = []
    for _, g in c.groupby("fecha"):
        g = g.drop_duplicates("partido").head(rule["legs"])
        if len(g) < rule["legs"]: continue
        out.append(g.odds.prod() - 1 if g.gano.all() else -1.0)
    return np.array(out)


def stat(g):
    if not len(g): return {"roi": 0.0, "se": 1.0, "n": 0, "hit": 0.0}
    return {"roi": round(float(g.mean()), 4), "se": round(float(g.std() / math.sqrt(len(g))) if len(g) > 1 else 1.0, 4),
            "n": int(len(g)), "hit": round(float((g > -1).mean()), 3)}


def main():
    fb, since = football_legs(None if A.W["1x2"] > 0 else 0.5)
    print(f"fútbol: peso del modelo {A.W['1x2'] if A.W['1x2'] > 0 else '0 en vivo → diagnóstico con 0.5'}")
    test0 = since + pd.DateOffset(years=2)
    tr, te = fb[fb.fecha < test0], fb[fb.fecha >= test0]
    live = live_legs()
    cur = {**DEFAULT, **(json.loads(OUT.read_text()) if OUT.exists() else {})}
    rules = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = []
    for r in rules:
        a, b = stat(combos(tr, r)), stat(combos(te, r))
        res.append((r, a, b))
        print(f"{r['source']:5s} {r['legs']} piernas p>={r['min_p']:.2f} | años -3/-2 {a['roi']:+.1%} ±{a['se']:.1%} n={a['n']:4d} "
              f"acierto {a['hit']:.0%} | año -1 {b['roi']:+.1%} ±{b['se']:.1%} n={b['n']}")
    best = max((x for x in res if x[1]["n"] >= 100), key=lambda x: x[1]["roi"] - x[1]["se"], default=None)  # prudente: ROI − 1 e.e.
    key = lambda r: {k: r[k] for k in GRID}
    now = next(x for x in res if key(x[0]) == key(cur))
    pick = best if best and best[2]["roi"] >= now[2]["roi"] else now
    stake = DEFAULT["stake"] if pick[2]["roi"] + 2 * pick[2]["se"] >= 0 else DEFAULT["stake"] / 2
    lv = stat(combos(live, pick[0]))
    print(f"\nactual {key(cur)}: año -1 {now[2]['roi']:+.1%} ±{now[2]['se']:.1%} (n={now[2]['n']})")
    print(f"elegida {key(pick[0])}: año -1 {pick[2]['roi']:+.1%} ±{pick[2]['se']:.1%} (n={pick[2]['n']}) · stake {stake:.1%}")
    print(f"señales reales de todos los deportes con la regla elegida: {lv['roi']:+.1%} ±{lv['se']:.1%} en {lv['n']} combinadas")
    OUT.write_text(json.dumps({**key(pick[0]), "stake": stake, "train": pick[1], "test": pick[2], "test_actual": now[2],
                               "en_vivo": lv, "fecha": f"{pd.Timestamp.now():%Y-%m-%d}"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
