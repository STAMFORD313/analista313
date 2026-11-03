"""Efecto del árbitro en tarjetas (y faltas): factor por árbitro = (tarjetas reales + k·esperadas) / (esperadas + k·esperadas)
con olvido, calculado SOLO con partidos anteriores; esperado de football_stats × factor^c. football-data solo trae el
árbitro en Premier, Championship y Escocia. c y k elegidos en el año −3, medido en las 2 últimas temporadas
(más/menos a la línea mediana, total). Guarda data/football_stats.json["referee"][stat] si mejora > 2 errores típicos.
Uso: PYTHONIOENCODING=utf-8 python train_referee.py"""
import json, math

import numpy as np
import pandas as pd
from scipy import optimize, stats

import football_stats as FS
import train_sgp as S
import totals_model as T

KS = [2, 5, 10, 20]


def ref_factor(refs, y, mu, k, forget=0.99):
    """Por partido: factor previo del árbitro (sus tarjetas / las esperadas, encogido con k partidos)."""
    Y, M, out = {}, {}, np.ones(len(refs))
    for i, (r, yy, m) in enumerate(zip(refs, y, mu)):
        if isinstance(r, str) and r.strip():
            r = r.strip()
            out[i] = (Y.get(r, 0.0) + k * m) / (M.get(r, 0.0) + k * m) if r in Y else 1.0
            Y[r] = Y.get(r, 0.0) * forget + yy; M[r] = M.get(r, 0.0) * forget + m
    return out


def main():
    V, test, fit, e = S.load(frame=True)
    has = e.Referee.notna().to_numpy() & e.Div.isin(["E0", "E1", "SC0"]).to_numpy()
    P = json.loads(FS.PARAMS.read_text()); P.setdefault("referee", {})
    for st in ("cards", "fouls"):
        mu, r, y, line = V[f"{st}_t"]
        best = None
        for k in KS:
            f = ref_factor(e.Referee.to_numpy(), y, mu, k)
            F_ = fit & has
            c = optimize.minimize_scalar(lambda c: -stats.nbinom.logpmf(y[F_], r, r / (r + mu[F_] * f[F_] ** c)).sum(),
                                         bounds=(0, 2), method="bounded").x
            ll = -stats.nbinom.logpmf(y[F_], r, r / (r + mu[F_] * f[F_] ** c)).mean()
            if best is None or ll < best[0]: best = (ll, k, c, f)
        _, k, c, f = best
        T_ = test & has
        ou = lambda m: (lambda p, o: -(o * np.log(p) + (1 - o) * np.log(1 - p)))(
            np.clip(1 - T.cdf("nbinom", m, r, np.floor(line[T_])), 1e-9, 1 - 1e-9), y[T_] > line[T_])
        l0, l1 = ou(mu[T_]), ou(mu[T_] * f[T_] ** c)
        se = float((l1 - l0).std() / math.sqrt(T_.sum()))
        print(f"{st}: árbitro k={k}, c={c:.2f}; test {T_.sum()} partidos (E0/E1/SC0): más/menos total {l0.mean():.4f} → {l1.mean():.4f} "
              f"({l1.mean() - l0.mean():+.4f} ± {se:.4f}); factor árbitro p10-p90 {np.percentile(f[T_], 10):.2f}-{np.percentile(f[T_], 90):.2f}")
        if l1.mean() < l0.mean() - 2 * se:
            P["referee"][st] = {"k": k, "c": round(float(c), 3), "ou": [round(float(l0.mean()), 4), round(float(l1.mean()), 4), round(se, 4)]}
        else: P["referee"].pop(st, None)
    FS.PARAMS.write_text(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
