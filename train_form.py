"""Forma reciente dentro del modelo de estadísticas de football_stats (córners, tarjetas, faltas, tiros a puerta).

Esperado del equipo × exp(c · ln((media últimos 10 + 0.5) / (media últimos 76 + 0.5))) de su estadística a favor
(cualquier condición, solo partidos anteriores). c por estadística, por máxima verosimilitud binomial negativa en el
año −3; medido en las 2 últimas temporadas (log-loss más/menos a la línea mediana, de equipo y total).
Guarda c en data/football_stats.json["form"][stat] solo si mejora en test más de 2 errores típicos (la lee
football_stats.expected). Uso: PYTHONIOENCODING=utf-8 python train_form.py"""
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

import analyst as A
import football_stats as FS
import totals_model as T

SHORT, LONG = 10, 76


def form_ratio(d, h, a):
    """(fr_local, fr_visitante) por partido de d (ordenado por fecha): ln de la media reciente / media larga del equipo."""
    x = pd.DataFrame({"team": np.r_[d.HomeTeam, d.AwayTeam], "v": np.r_[d[h], d[a]], "i": np.r_[d.index, d.index],
                      "home": np.r_[np.ones(len(d)), np.zeros(len(d))]}).sort_values(["i", "home"], kind="stable")
    g = x.groupby("team").v
    s = g.transform(lambda v: v.shift(1).rolling(SHORT, min_periods=5).mean())
    l = g.transform(lambda v: v.shift(1).rolling(LONG, min_periods=20).mean())
    x["fr"] = np.log((s + 0.5) / (l + 0.5))
    f = lambda hm: x[x.home == hm].set_index("i").fr.reindex(d.index).fillna(0).to_numpy()
    return f(1), f(0)


def ou_ll(mu, r, y, line):
    p = np.clip(1 - T.cdf("nbinom", mu, r, np.floor(line)), 1e-9, 1 - 1e-9); o = y > line
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def main():
    import train_sgp as S
    V, test, fit = S.load()
    d = FS.prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + FS.COLS) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True))
    d = d.dropna(subset=["FTHG", "FTAG", *FS.COLS[:4], "HF", "AF", "HST", "AST"]).sort_values("Date", ignore_index=True)
    keep = (d.Date >= d.Date.max() - pd.DateOffset(years=3)).to_numpy()
    P = json.loads(FS.PARAMS.read_text()); P.setdefault("form", {})
    for st, (h, a, name) in FS.STATS.items():
        fh, fa = (x[keep] for x in form_ratio(d, h, a))
        (mh, rt, yh, lh), (ma, _, ya, la), (_, r, yt, lt) = V[f"{st}_h"], V[f"{st}_a"], V[f"{st}_t"]
        nll = lambda c, m: -(stats.nbinom.logpmf(yh[m], rt, rt / (rt + mh[m] * np.exp(c * fh[m])))
                             + stats.nbinom.logpmf(ya[m], rt, rt / (rt + ma[m] * np.exp(c * fa[m])))).sum()
        c = float(optimize.minimize_scalar(lambda c: nll(c, fit), bounds=(-1, 2), method="bounded").x)
        mh1, ma1 = mh * np.exp(c * fh), ma * np.exp(c * fa)
        T_ = test
        team0 = np.r_[ou_ll(mh, rt, yh, lh)[T_], ou_ll(ma, rt, ya, la)[T_]]
        team1 = np.r_[ou_ll(mh1, rt, yh, lh)[T_], ou_ll(ma1, rt, ya, la)[T_]]
        tot0, tot1 = ou_ll(mh + ma, r, yt, lt)[T_], ou_ll(mh1 + ma1, r, yt, lt)[T_]
        se = lambda x, y: float((x - y).std() / np.sqrt(len(x)))
        res = {"c": round(c, 3), "team": [round(float(team0.mean()), 4), round(float(team1.mean()), 4), round(se(team1, team0), 4)],
               "total": [round(float(tot0.mean()), 4), round(float(tot1.mean()), 4), round(se(tot1, tot0), 4)]}
        ok = team1.mean() < team0.mean() - 2 * res["team"][2] and tot1.mean() <= tot0.mean() + res["total"][2]
        print(f"{name}: c={c:+.3f} · más/menos de equipo {team0.mean():.4f} → {team1.mean():.4f} ({team1.mean() - team0.mean():+.4f} ± "
              f"{res['team'][2]}) · total {tot0.mean():.4f} → {tot1.mean():.4f} ({tot1.mean() - tot0.mean():+.4f} ± {res['total'][2]})"
              f"{' → se usa' if ok else ''}")
        if ok: P["form"][st] = res
        else: P["form"].pop(st, None)
    FS.PARAMS.write_text(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
