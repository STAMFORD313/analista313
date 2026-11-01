"""Elo de individuales (UFC y, con superficies, tenis: ver tennis_model.py) con K dinámico kt/(n+5)^pw, ventaja de
esquina/"local" (hfa), regresión a 1500 por inactividad y, opcionalmente, ratings por grupo (superficie) mezclados con
el general. `python ufc_model.py` hace el backtest de UFC (log-loss de las 2 últimas temporadas, prob. calculada antes
de cada pelea, como train_espn.py) y guarda los mejores en data/ufc_params.json."""
import itertools, json
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = Path(__file__).parent / "data" / "ufc_params.json"


def elo(df, kt=250, pw=0.4, hfa=0, idle=365, keep=1.0, group=None, mix=0.0, kmult=None):
    """df ordenado (home, away, res, date) → (prob. local antes de cada fila, ratings {(clave, jugador): R}).
    group: Serie con el grupo de cada fila (superficie); mix: peso del rating de grupo en la predicción.
    kmult: Serie con multiplicador de K por fila (nivel de torneo). Filas con res NaN predicen sin actualizar."""
    R, n, last, pred = {}, {}, {}, np.empty(len(df))
    g = group.to_numpy() if group is not None else np.full(len(df), None)
    km = kmult.to_numpy(float) if kmult is not None else np.ones(len(df))
    for i, (h, a, res, d) in enumerate(zip(df.home, df.away, df.res, df.date)):
        for t in (h, a):
            if t in last and (d - last[t]).days > idle:  # vuelve tras parón: menos confianza en su rating
                for key in ((None, t), (g[i], t)):
                    if key in R: R[key] = 1500 + keep * (R[key] - 1500)
        diff = lambda key: R.get((key, h), 1500) - R.get((key, a), 1500)
        dd = (1 - mix) * diff(None) + (mix * diff(g[i]) if group is not None else 0)
        e = pred[i] = 1 / (1 + 10 ** (-(dd + hfa) / 400))
        if pd.isna(res): continue
        for key in {None, g[i]}:
            eh = 1 / (1 + 10 ** (-(diff(key) + hfa) / 400))
            for t, s in ((h, res - eh), (a, eh - res)):
                kk = km[i] * kt / (n.get((key, t), 0) + 5) ** pw
                R[key, t] = R.get((key, t), 1500) + kk * s
                n[key, t] = n.get((key, t), 0) + 1
        last[h] = last[a] = d
    return pred, R


def load(league):
    import multisport as M
    df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)


def logloss(p, y, ev):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))[ev]


def main():
    import multisport as M
    df = load("ufc")
    end = df.date.max()
    y, ev = df.res.to_numpy(float), (df.date >= end - pd.DateOffset(years=2)).to_numpy()
    val = ~ev & (df.date >= end - pd.DateOffset(years=4)).to_numpy()  # se elige aquí y se mide en ev: sin sesgo de selección
    old = []
    M.elo(df, 0, None, old, kt=M._P.get("ufc", {}).get("kt", 250))
    base = logloss(np.array(old), y, ev)
    grid = itertools.product([200, 300, 400, 550, 700], [0.4, 0.6, 0.8], [-60, -40, -20, 0, 20], [365, 10 ** 6], [0.6, 0.8])
    res = {}
    for kt, pw, hfa, idle, keep in grid:
        if idle == 10 ** 6 and keep != 0.8: continue  # sin regresión: keep no importa
        res[kt, pw, hfa, idle, keep] = logloss(elo(df, kt, pw, hfa, idle, keep)[0], y, val).mean()
    best = min(res, key=res.get)
    d = logloss(elo(df, *best)[0], y, ev) - base
    print(f"UFC base (kt {M._P.get('ufc', {}).get('kt', 250)}): log-loss {base.mean():.4f} ({ev.sum()} peleas)")
    test = base.mean() + d.mean()
    print(f"UFC mejor en validación kt/pw/hfa/idle/keep {best}: test {test:.4f} ({d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}); "
          f"frecuencia real de victoria 'local': {y[ev].mean():.3f}")
    kt, pw, hfa, idle, keep = best
    PARAMS.write_text(json.dumps({"kt": kt, "pw": pw, "hfa": hfa, "idle": idle, "keep": keep, "logloss": round(float(test), 4),
                                  "logloss_old": round(float(base.mean()), 4), "se": round(float(d.std() / np.sqrt(len(d))), 4)}))


if __name__ == "__main__":
    t = pd.DataFrame({"home": ["A", "A"], "away": ["B", "B"], "res": [1.0, np.nan], "date": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    p, R = elo(t)
    assert p[0] == 0.5 and p[1] > 0.5 and R[None, "A"] > 1500
    main()
