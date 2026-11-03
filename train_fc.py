"""Entrena el modelo de FC 27 (esoccer.py) con walk-forward: elige lr/lr_league/shrink en el tramo A (días -30 a -15)
y mide fuera de muestra en el B (últimos 15 días); compara con la frecuencia 1X2 de cada liga. Guarda data/fc/params.json
solo si el log-loss del tramo B no empeora frente a los parámetros anteriores. Uso: python train_fc.py [fetch DÍAS]"""
import itertools, json, math, sys

import pandas as pd

import esoccer as F


def losses(df, a, b, **kw):
    """Log-loss 1X2 en los tramos A=[a, b) y B=[b, fin) con predicción previa a cada partido."""
    m, la, lb, na, nb = F.Model(**kw), 0.0, 0.0, 0, 0
    for r in df.itertuples():
        if r.date >= a:
            p, _, _ = F.markets(m.lam(r.league, r.p1, r.p2), m.lam(r.league, r.p2, r.p1))
            l = -math.log(max(p["1" if r.s1 > r.s2 else "2" if r.s2 > r.s1 else "X"], 1e-9))
            if r.date < b: la += l; na += 1
            else: lb += l; nb += 1
        m.update(r.league, r.p1, r.p2, r.s1, r.s2)
    return la / max(na, 1), lb / max(nb, 1), nb


def league_freq(df, a, b):
    """Referencia: frecuencia 1X2 de cada liga aprendida antes del tramo B."""
    o = df.apply(lambda r: "1" if r.s1 > r.s2 else "2" if r.s2 > r.s1 else "X", axis=1)
    tr, te = df[df.date < b], df[df.date >= b]
    fr = pd.crosstab(tr.league, o[tr.index], normalize="index")
    return float(-sum(math.log(max(fr.loc[l, k] if l in fr.index else 1 / 3, 1e-9)) for l, k in zip(te.league, o[te.index])) / len(te))


def main():
    df = pd.read_csv(F.HIST); df["date"] = pd.to_datetime(df["date"], utc=True); df = df.sort_values(["date", "id"])
    end = df.date.max()
    a, b = end - pd.Timedelta(days=30), end - pd.Timedelta(days=15)
    res = []
    for lr, lrl, sh in itertools.product([0.002, 0.003, 0.005, 0.01], [0.0005, 0.002], [0, 0.0005]):
        la, lb, nb = losses(df, a, b, lr=lr, lr_league=lrl, shrink=sh)
        res.append((la, lb, {"lr": lr, "lr_league": lrl, "shrink": sh}))
        print(f"{res[-1][2]}: tramo A {la:.4f} · B {lb:.4f}", flush=True)
    la, lb, best = min(res, key=lambda r: r[0])
    old = losses(df, a, b)[1]
    print(f"elegido en A {best} → B {lb:.4f} ({nb} partidos) · antes {old:.4f} · frecuencia por liga {league_freq(df, a, b):.4f} · azar {math.log(3):.4f}")
    if lb <= old:  # conserva la recalibración de train_fc_calib.py (se rehace justo después en entrenar.yml)
        prev = json.loads((F.DIR / "params.json").read_text()) if (F.DIR / "params.json").exists() else {}
        (F.DIR / "params.json").write_text(json.dumps({**{k: prev[k] for k in ("calib", "calib_test") if k in prev},
                                                       **best, "logloss_oos": round(lb, 4), "n_oos": nb}))
    else: print("empeora: se mantienen los parámetros anteriores")


if __name__ == "__main__":
    if "fetch" in sys.argv: F.update_history(days=int(sys.argv[-1]))
    else: main()
