"""Combinadas del mismo partido en fútbol («crear apuesta»): dependencia entre mercados con una cópula gaussiana.

Variables: goles, córners, tarjetas, faltas y tiros a puerta, cada una total (t), del local (h) y del visitante (a).
Marginales = las de football_stats (binomial negativa con las fuerzas de cada equipo; goles Poisson). La correlación
latente ρ de cada par se estima por máxima verosimilitud de las 4 casillas más/menos a la línea mediana, solo con el
año −3; se mide en las 2 últimas temporadas frente a (a) independencia y (b) la corrección de Bahadur con covarianzas
que usa ahora football_stats.joint, también a líneas desplazadas (±1) para ver que vale a cualquier línea.
Guarda la matriz en data/football_stats.json["rho"] (la lee football_stats.joint).
Uso: PYTHONIOENCODING=utf-8 python train_sgp.py"""
import itertools, json

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import analyst as A
import football_stats as FS
import totals_model as T

SCOPES = ("t", "h", "a")


def phi2(h, k, rho):
    """Φ2(h, k; ρ) vectorizado con la T de Owen."""
    h = np.where(np.abs(h) < 1e-9, 1e-9, h); k = np.where(np.abs(k) < 1e-9, 1e-9, k)
    s = np.sqrt(1 - rho ** 2)
    beta = np.where(h * k > 0, 0.0, 0.5)
    return np.clip(0.5 * stats.norm.cdf(h) + 0.5 * stats.norm.cdf(k) - special.owens_t(h, (k - rho * h) / (h * s))
                   - special.owens_t(k, (h - rho * k) / (k * s)) - beta, 0, 1)


def cells_copula(pa, pb, rho):
    """[P(oo), P(ou), P(uo), P(uu)] con P(más) marginales pa, pb y correlación latente rho."""
    za, zb = stats.norm.ppf(np.clip(1 - pa, 1e-9, 1 - 1e-9)), stats.norm.ppf(np.clip(1 - pb, 1e-9, 1 - 1e-9))
    uu = phi2(za, zb, rho)
    ou, uo = (1 - pa) - uu, (1 - pb) - uu
    return np.clip(np.c_[1 - uu - ou - uo, uo, ou, uu], 1e-9, 1)  # oo, ou (a más, b menos), uo, uu


def cells_cov(pa, pb, c):
    return np.clip(np.c_[pa * pb + c, pa * (1 - pb) - c, (1 - pa) * pb - c, (1 - pa) * (1 - pb) + c], 1e-9, 1)


def cell(ya, yb): return np.where(ya, np.where(yb, 0, 1), np.where(yb, 2, 3))


def load(frame=False):
    """Dict {var: (mu, r, y, línea)}, máscaras test y ajuste (y con frame=True, las filas del periodo de 3 años)."""
    import train_football as TF
    P = json.loads(FS.PARAMS.read_text())
    df = FS.prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + FS.COLS + ["Referee"]) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True))
    d = df.dropna(subset=["FTHG", "FTAG", *FS.COLS[:4], "HF", "AF", "HST", "AST"]).sort_values("Date", ignore_index=True)
    end = d.Date.max(); since = end - pd.DateOffset(years=3)
    e = d[d.Date >= since].reset_index(drop=True)
    test = (e.Date >= end - pd.DateOffset(years=2)).to_numpy()
    V = {}
    specs = {"goals": ("FTHG", "FTAG", A.HALF_LIFE, A.SHRINK, 1e6, 1e6),
             **{s: (h, a, P[s]["half_life"], P[s]["k"], P[s]["r"], P[s]["r_team"]) for s, (h, a, _) in FS.STATS.items()}}
    for st, (h, a, hl, k, r, rt) in specs.items():
        s = d.assign(FTHG=d[h], FTAG=d[a])
        mh, ma = TF.lambdas(s, since, [hl], [k])[hl, k]
        tot = s.FTHG + s.FTAG
        lines = {sc: x.groupby(s.Div).transform(FS.median_line)[s.Date >= since].to_numpy() for sc, x in
                 (("t", tot), ("h", s.FTHG), ("a", s.FTAG))}
        if st == "goals": lines["t"] = np.full(len(e), 2.5)
        for sc, mu, y, rr in (("t", mh + ma, e[h] + e[a], r), ("h", mh, e[h], rt), ("a", ma, e[a], rt)):
            V[f"{st}_{sc}"] = (mu, rr, y.to_numpy(float), lines[sc])
    ok = np.logical_and.reduce([np.isfinite(l) & (mu > 0) for mu, _, _, l in V.values()])
    return (V, test & ok, ~test & ok, e) if frame else (V, test & ok, ~test & ok)


def p_over(v, shift=0):
    mu, r, _, line = v
    return 1 - T.cdf("nbinom", mu, r, np.floor(np.nan_to_num(line)) + shift)


def fit_rho(pa, pb, ya, yb):
    c = cell(ya, yb)
    nll = lambda rho: -np.log(cells_copula(pa, pb, rho)[np.arange(len(c)), c]).sum()
    return float(optimize.minimize_scalar(nll, bounds=(-0.95, 0.95), method="bounded").x)


def nearest_psd(R):
    w, Q = np.linalg.eigh(R)
    R = Q @ np.diag(np.clip(w, 1e-4, None)) @ Q.T
    s = np.sqrt(np.diag(R)); return R / np.outer(s, s)


def main():
    V, test, fit = load()
    names = list(V)
    print(f"{fit.sum()} partidos de ajuste (año −3), {test.sum()} de test (2 últimas temporadas), {len(names)} variables")
    cov = json.loads(FS.PARAMS.read_text()).get("cov", {})
    po = {n: p_over(V[n]) for n in names}
    yo = {n: V[n][2] > V[n][3] for n in names}
    R = np.eye(len(names))
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        if a.split("_")[0] == b.split("_")[0]: continue  # mismo estadístico (total vs equipo): no es combinable en la casa
        R[i, j] = R[j, i] = fit_rho(po[a][fit], po[b][fit], yo[a][fit], yo[b][fit])
    R = nearest_psd(R)
    res = {}
    for shift in (0, -1, 1):
        L = {k: [] for k in ("ind", "cov", "cop")}
        for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
            if a.split("_")[0] == b.split("_")[0]: continue
            pa, pb = p_over(V[a], shift)[test], p_over(V[b], shift)[test]
            c = cell(V[a][2][test] > V[a][3][test] + shift, V[b][2][test] > V[b][3][test] + shift)
            ix = np.arange(len(c))
            cv = cov.get(f"{a[:-2]}|{b[:-2]}", cov.get(f"{b[:-2]}|{a[:-2]}", 0.0)) if a.endswith("_t") and b.endswith("_t") else 0.0
            for k, C in (("ind", cells_cov(pa, pb, 0.0)), ("cov", cells_cov(pa, pb, cv)), ("cop", cells_copula(pa, pb, R[i, j]))):
                L[k].append(-np.log(C[ix, c]))
        L = {k: np.concatenate(v) for k, v in L.items()}
        # error típico por partido (los pares del mismo partido no son independientes): agrupar por fila
        n = test.sum(); per = {k: v.reshape(-1, n).sum(0) for k, v in L.items()}
        se = lambda x, y: float((x - y).std() / np.sqrt(n))
        res[shift] = {k: round(float(v.mean()), 5) for k, v in L.items()}
        res[shift]["se_cop_ind"] = round(se(per["cop"], per["ind"]) / (len(L["ind"]) / n), 5)
        res[shift]["se_cop_cov"] = round(se(per["cop"], per["cov"]) / (len(L["ind"]) / n), 5)
        print(f"línea {'mediana' if not shift else f'mediana {shift:+d}'}: log-loss 4 casillas por par independencia "
              f"{res[shift]['ind']:.5f} · covarianzas actuales {res[shift]['cov']:.5f} · cópula {res[shift]['cop']:.5f} "
              f"(cópula − indep. {res[shift]['cop'] - res[shift]['ind']:+.5f} ± {res[shift]['se_cop_ind']}, "
              f"− covarianzas {res[shift]['cop'] - res[shift]['cov']:+.5f} ± {res[shift]['se_cop_cov']})")
    # combinadas de 3 piernas totales como las de la tarjeta (lado más probable de cada pierna), con la prob. conjunta
    tot = [n for n in names if n.endswith("_t")]
    rng = np.random.default_rng(0); rows = np.flatnonzero(test); rows = rng.choice(rows, min(3000, len(rows)), replace=False)
    out = {k: [] for k in ("ind", "cov", "cop")}
    for tri in itertools.combinations(tot, 3):
        idx = [names.index(t) for t in tri]
        p = np.array([po[t][rows] for t in tri]).T
        sg = np.where(p >= 0.5, 1, -1); pp = np.where(sg > 0, p, 1 - p)
        hit = np.all([(yo[t][rows] == (sg[:, q] > 0)) for q, t in enumerate(tri)], axis=0)
        Rs = R[np.ix_(idx, idx)]
        for r_ in range(len(rows)):
            legs = [{"stat": t[:-2], "sign": int(sg[r_, q]), "p": float(pp[r_, q])} for q, t in enumerate(tri)]
            pj = {"ind": float(np.prod(pp[r_])), "cov": FS.joint(legs, cov)}
            S = Rs * np.outer(sg[r_], sg[r_])
            z = stats.norm.ppf(pp[r_])
            pj["cop"] = float(stats.multivariate_normal(np.zeros(3), S).cdf(z))
            for k in out: out[k].append((pj[k], hit[r_]))
    for k in out:
        a = np.array(out[k]); p = np.clip(a[:, 0], 1e-6, 1 - 1e-6)
        out[k] = (float(-(a[:, 1] * np.log(p) + (1 - a[:, 1]) * np.log(1 - p)).mean()), float(p.mean()), float(a[:, 1].mean()))
    print("combinadas de 3 totales (10 tríos × 3000 partidos de test): " + " · ".join(
        f"{k} log-loss {v[0]:.5f}, prob. media {v[1]:.3f} (acierto {v[2]:.3f})" for k, v in out.items()))
    if res[0]["cop"] >= res[0]["ind"]:
        print("la cópula no mejora a la independencia fuera de muestra: no se guarda"); return
    P = json.loads(FS.PARAMS.read_text())
    P["rho"] = {"names": names, "R": np.round(R, 4).tolist(), "test": {str(k): v for k, v in res.items()},
                "trios": {k: [round(x, 5) for x in v] for k, v in out.items()}}
    FS.PARAMS.write_text(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
