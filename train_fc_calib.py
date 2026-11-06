"""Recalibración del 1X2 de FC 27 (la simulación real mostraba favoritos al 72 % que acertaban el 62 %).

p_i ∝ p_i^a · exp(b·[i = empate]): `a` < 1 encoge los favoritos, `b` > 0 sube el empate. Predicciones walk-forward del
modelo de producción (esoccer.fit, cada partido predicho antes de jugarse); a y b por máxima verosimilitud en los días
−30 a −15, medido en los últimos 15 días: log-loss y calibración de los picks seguros (prob. máx. ≥ 0.6, como los que
manda el bot). Guarda "calib": [a, b] en data/fc/params.json solo si mejora el log-loss fuera de muestra en más de
2 errores típicos (lo usa esoccer.calibrate). Igual con el cara a cara («h2h», esoccer.h2h_adjust). Uso: PYTHONIOENCODING=utf-8 python train_fc_calib.py"""
import json, math

import numpy as np
import pandas as pd
from scipy import optimize

import esoccer as E


def calibrate_arr(P, a, b):
    L = a * np.log(np.clip(P, 1e-9, 1)); L[:, 1] += b
    L = np.exp(L - L.max(1, keepdims=True)); return L / L.sum(1, keepdims=True)


CLUB_K, CLUB_DAYS = 50, 60  # sorpresa por club encogida con 50 partidos; se acumula desde 60 días antes del tramo


def predictions(df, since):
    """Predicciones walk-forward + cara a cara + «club»: lo que rinde el equipo que usa cada jugador por encima de lo que
    esperaban los jugadores (resultado − prob. del modelo, acumulado solo con partidos anteriores, encogido)."""
    m, rows, H, C = E.Model(), [], {}, {}
    cl = lambda c: c if isinstance(c, str) else None
    for r in df.itertuples():
        c1, c2 = cl(getattr(r, "club1", None)), cl(getattr(r, "club2", None))
        p, _, _ = E.markets(m.lam(r.league, r.p1, r.p2), m.lam(r.league, r.p2, r.p1)) if (r.date >= since or (c1 and c2 and r.date >= since - pd.Timedelta(days=CLUB_DAYS))) else (None, 0, 0)
        if r.date >= since:
            cv = (lambda a, b: a[0] / (a[1] + CLUB_K) - b[0] / (b[1] + CLUB_K))(C.get(c1, [0, 0]), C.get(c2, [0, 0])) if c1 and c2 else 0.0
            rows.append((r.date, p["1"], p["X"], p["2"], 0 if r.s1 > r.s2 else 1 if r.s1 == r.s2 else 2, E.h2h_value(H, r.p1, r.p2), r.league, cv))
        if p and c1 and c2:
            res = (1 if r.s1 > r.s2 else 0) - p["1"]
            for c, v in ((c1, res), (c2, -res)): t = C.setdefault(c, [0, 0]); t[0] += v; t[1] += 1
        m.update(r.league, r.p1, r.p2, r.s1, r.s2)
        k = tuple(sorted((r.p1, r.p2))); s = H.setdefault(k, [0, 0])  # cara a cara, solo con partidos anteriores
        s[0] += (1 if r.s1 > r.s2 else -1 if r.s2 > r.s1 else 0) * (1 if k[0] == r.p1 else -1); s[1] += 1
    d = pd.DataFrame(rows, columns=["date", "p1", "pX", "p2", "y", "h2h", "league", "club"])
    d.attrs["club_table"] = {c: round(v[0] / (v[1] + CLUB_K), 4) for c, v in C.items() if v[1] >= 20}
    return d


def main():
    df = pd.read_csv(E.HIST).dropna(subset=["s1", "s2"])
    if (E.DIR / "players_teams.csv").exists():  # club que usa cada jugador (fc_players.py)
        df = df.merge(pd.read_csv(E.DIR / "players_teams.csv")[["id", "club1", "club2"]], on="id", how="left")
    df["date"] = pd.to_datetime(df.date, utc=True); df = df.sort_values(["date", "id"], ignore_index=True)
    end = df.date.max()
    d = predictions(df, end - pd.Timedelta(days=30))
    P, y = d[["p1", "pX", "p2"]].to_numpy(), d.y.to_numpy()
    te = (d.date >= end - pd.Timedelta(days=15)).to_numpy(); fi = ~te
    nll = lambda Q, m: -np.log(np.clip(Q[m][np.arange(m.sum()), y[m]], 1e-9, 1))
    w = optimize.minimize(lambda w: nll(calibrate_arr(P, *w), fi).sum(), [1.0, 0.0], method="Nelder-Mead").x
    Q = calibrate_arr(P, *w)
    l0, l1 = nll(P, te), nll(Q, te)
    se = float((l1 - l0).std() / math.sqrt(te.sum()))
    print(f"FC 27: ajuste {fi.sum()} partidos, test {te.sum()}; a={w[0]:.3f}, b={w[1]:+.3f}")
    print(f"  log-loss 1X2 últimos 15 días: modelo {l0.mean():.4f} → recalibrado {l1.mean():.4f} ({l1.mean() - l0.mean():+.4f} ± {se:.4f})")
    for name, R in (("modelo", P), ("recalibrado", Q)):
        mx = R[te].max(1); pick = R[te].argmax(1); hit = pick == y[te]; s = mx >= 0.6
        print(f"  picks ≥ 60 % ({name}): {s.sum()} · prob. media {mx[s].mean():.3f} → acierto {hit[s].mean():.3f}; "
              f"empates previstos {R[te][:, 1].mean():.3f} vs reales {(y[te] == 1).mean():.3f}")
    f = E.DIR / "params.json"; P_ = json.loads(f.read_text())
    P_["calib_test"] = {"ll": round(float(l0.mean()), 4), "ll_cal": round(float(l1.mean()), 4), "se": round(se, 4), "n": int(te.sum())}
    if l1.mean() < l0.mean() - 2 * se: P_["calib"] = [round(float(x), 4) for x in w]; print("  guardado calib")
    else: P_.pop("calib", None)
    # empate por liga (sobre el 1X2 recalibrado que se use): factor real/previsto en el ajuste, encogido con K partidos
    B = Q if "calib" in P_ else P
    K = 100; st = pd.DataFrame({"liga": d.league[fi], "pX": B[fi][:, 1], "X": (y[fi] == 1)}).groupby("liga").agg(n=("X", "size"), pred=("pX", "mean"), real=("X", "mean"))
    st["f"] = ((st.real * st.n + st.pred * K) / (st.n + K)) / st.pred
    fl = d.league.map(st.f).fillna(1.0).to_numpy()
    D = B.copy(); D[:, 1] *= fl; D /= D.sum(1, keepdims=True)
    ld0, ld1 = nll(B, te), nll(D, te); sed = float((ld1 - ld0).std() / math.sqrt(te.sum()))
    print(f"  empate por liga: log-loss {ld0.mean():.4f} → {ld1.mean():.4f} ({ld1.mean() - ld0.mean():+.4f} ± {sed:.4f})")
    P_["draw_test"] = {"ll": round(float(ld0.mean()), 4), "ll_draw": round(float(ld1.mean()), 4), "se": round(sed, 4)}
    if ld1.mean() < ld0.mean() - 2 * sed:
        # para producción: factores con los 30 días completos (lo medido arriba usa solo la 1.ª mitad)
        al = pd.DataFrame({"liga": d.league, "pX": B[:, 1], "X": (y == 1)}).groupby("liga").agg(n=("X", "size"), pred=("pX", "mean"), real=("X", "mean"))
        P_["draw_liga"] = {k: round(float(((r.real * r.n + r.pred * K) / (r.n + K)) / r.pred), 4) for k, r in al.iterrows()}
        B = D; print("  guardado draw_liga")
    else: P_.pop("draw_liga", None)
    # cara a cara entre los dos jugadores (sobre el 1X2 recalibrado que se use)
    x = d.h2h.to_numpy()
    shift = lambda w: (lambda Z: Z / Z.sum(1, keepdims=True))(B * np.exp(np.c_[w * x, 0 * x, -w * x]))
    wh = float(optimize.minimize_scalar(lambda w: nll(shift(w), fi).sum(), bounds=(-1, 1), method="bounded").x)
    lb, lh = nll(B, te), nll(shift(wh), te); seh = float((lh - lb).std() / math.sqrt(te.sum()))
    print(f"  cara a cara: w={wh:+.3f}, log-loss {lb.mean():.4f} → {lh.mean():.4f} ({lh.mean() - lb.mean():+.4f} ± {seh:.4f})")
    P_["h2h_test"] = {"ll": round(float(lb.mean()), 4), "ll_h2h": round(float(lh.mean()), 4), "se": round(seh, 4)}
    if lh.mean() < lb.mean() - 2 * seh: P_["h2h"] = round(wh, 4); print("  guardado h2h")
    else: P_.pop("h2h", None)
    # club que usa cada jugador (sobre lo anterior)
    B2 = shift(wh) if "h2h" in P_ else B; xc = d.club.to_numpy()
    cshift = lambda w: (lambda Z: Z / Z.sum(1, keepdims=True))(B2 * np.exp(np.c_[w * xc, 0 * xc, -w * xc]))
    wc = float(optimize.minimize_scalar(lambda w: nll(cshift(w), fi).sum(), bounds=(-3, 3), method="bounded").x)
    lb2, lc = nll(B2, te), nll(cshift(wc), te); sec = float((lc - lb2).std() / math.sqrt(te.sum()))
    print(f"  club: w={wc:+.3f}, log-loss {lb2.mean():.4f} → {lc.mean():.4f} ({lc.mean() - lb2.mean():+.4f} ± {sec:.4f})")
    P_["club_test"] = {"ll": round(float(lb2.mean()), 4), "ll_club": round(float(lc.mean()), 4), "se": round(sec, 4)}
    if lc.mean() < lb2.mean() - 2 * sec: P_["club_w"] = round(wc, 4); P_["club"] = d.attrs["club_table"]; print("  guardado club")
    else: P_.pop("club_w", None); P_.pop("club", None)
    f.write_text(json.dumps(P_))


if __name__ == "__main__":
    main()
