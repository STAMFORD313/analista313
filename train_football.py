"""Entrena el modelo de fútbol con backtest sin mirar el futuro: cada semana de las 2 últimas temporadas se predice
solo con los partidos anteriores. Elige vida media, encogimiento k y RHO por log-loss 1X2, y el peso del modelo
frente al mercado y la ventaja mínima por ROI apostando a la cuota media (Avg, o B365) del CSV.
Guarda los mejores en data/football_params.json (lo lee analyst.py)."""
import itertools, json, math

import numpy as np
import pandas as pd

import analyst as A

ODDS = [f"{b}{m}" for b in ("PS", "Avg", "B365", "PSC", "AvgC", "B365C") for m in "HDA"]
N = 11
I, J = np.indices((N, N))
FACT = np.array([math.factorial(i) for i in range(N)], float)


def load():
    frames = [A.load_csv(f.read_bytes(), A.COLS + ODDS) for f in sorted(A.HIST.glob("*.csv"))]
    for f in sorted(A.NEW.glob("*.csv")):
        frames.append(A.load_csv(f.read_bytes(), ["Date", "Home", "Away", "HG", "AG", *ODDS]).assign(Div=f.stem))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["FTHG", "FTAG"]).sort_values("Date", ignore_index=True)
    df[ODDS] = df.reindex(columns=ODDS).apply(pd.to_numeric, errors="coerce")  # hay celdas basura ('x')
    for m in "HDA":  # mercado: Pinnacle si hay, si no la media (de apertura; ligas nuevas solo traen cierre)
        df[f"m{m}"] = df[f"PS{m}"].fillna(df[f"Avg{m}"]).fillna(df[f"PSC{m}"]).fillna(df[f"AvgC{m}"])
        df[f"o{m}"] = df[f"Avg{m}"].fillna(df[f"B365{m}"]).fillna(df[f"AvgC{m}"]).fillna(df[f"B365C{m}"])
    return df


def probs_1x2(lh, la, rho):
    """H, D, A como analyst.probs pero vectorizado (N partidos a la vez)."""
    ph = np.exp(-lh)[:, None] * lh[:, None] ** np.arange(N) / FACT
    pa = np.exp(-la)[:, None] * la[:, None] ** np.arange(N) / FACT
    M = ph[:, :, None] * pa[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho; M[:, 1, 0] *= 1 + la * rho; M[:, 0, 1] *= 1 + lh * rho; M[:, 1, 1] *= 1 - rho
    return np.stack([M[:, I > J].sum(1), M[:, I == J].sum(1), M[:, I < J].sum(1)], 1)


def lambdas(df, since, hls, ks):
    """{(vida media, k): (lh, la)} para los partidos desde `since`, con fuerzas recalculadas cada semana."""
    ev = df.index[df.Date >= since]
    out = {c: (np.zeros(len(ev)), np.zeros(len(ev))) for c in itertools.product(hls, ks)}
    pos = pd.Series(np.arange(len(ev)), index=ev)
    for div, d in df.groupby("Div"):
        teams = {t: i for i, t in enumerate(pd.unique(pd.concat([d.HomeTeam, d.AwayTeam])))}
        hi, ai = d.HomeTeam.map(teams).to_numpy(), d.AwayTeam.map(teams).to_numpy()
        hg, ag, dt = d.FTHG.to_numpy(float), d.FTAG.to_numpy(float), d.Date.to_numpy()
        e = d[d.Date >= since]
        week = since + pd.to_timedelta((e.Date - since).dt.days // 7 * 7, unit="D")
        for T, g in e.groupby(week):
            m = (dt < np.datetime64(T)) & (dt >= np.datetime64(T - pd.DateOffset(years=12)))
            if not m.any(): continue
            age = (np.datetime64(T) - dt[m]) / np.timedelta64(1, "D")
            gh, ga = g.HomeTeam.map(teams).to_numpy(), g.AwayTeam.map(teams).to_numpy()
            for hl in hls:
                w = 0.5 ** (age / hl)
                mu_h, mu_a = (w * hg[m]).sum() / w.sum(), (w * ag[m]).sum() / w.sum()
                bc = lambda idx, x: np.bincount(idx[m], w * x if x is not None else w, minlength=len(teams))
                nh, na, sh, ch, sa, ca = bc(hi, None), bc(ai, None), bc(hi, hg[m]), bc(hi, ag[m]), bc(ai, ag[m]), bc(ai, hg[m])
                for k in ks:
                    att_h, def_h = (sh + k * mu_h) / (nh + k) / mu_h, (ch + k * mu_a) / (nh + k) / mu_a
                    att_a, def_a = (sa + k * mu_a) / (na + k) / mu_a, (ca + k * mu_h) / (na + k) / mu_h
                    p = pos[g.index].to_numpy()
                    out[hl, k][0][p] = mu_h * att_h[gh] * def_a[ga]
                    out[hl, k][1][p] = mu_a * att_a[ga] * def_h[gh]
    return out


def logloss(P, y): return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())


def roi(P, mk, odds, y, w, edge):
    """Apuesta 1 u. a cada resultado con ventaja entre `edge` y MAX_EDGE y prob >= MIN_P, a la cuota media."""
    p = w * P + (1 - w) * mk
    ev = p * odds - 1
    bet = (ev >= edge) & (ev <= A.MAX_EDGE) & (p >= A.MIN_P) & np.isfinite(odds)
    win = np.zeros_like(bet); win[np.arange(len(y)), y] = True
    gain = np.where(win, odds - 1, -1.0)[bet]
    return (float(gain.mean()) if len(gain) else 0.0), len(gain), (float(gain.std() / math.sqrt(len(gain))) if len(gain) else 1.0)


def main():
    df = load()
    since = df.Date.max() - pd.DateOffset(years=2)
    hls, ks, rhos = [180, 270, 365, 540, 730], [2, 5, 10, 20], [-0.2, -0.15, -0.1, -0.05, 0.0]
    L = lambdas(df, since, hls, ks)
    e = df[df.Date >= since].reset_index(drop=True)
    y = np.select([e.FTHG > e.FTAG, e.FTHG == e.FTAG], [0, 1], 2)
    ok = L[hls[0], ks[0]][0] > 0  # sin historial previo: fuera
    # comprobación: el vectorizado coincide con analyst.probs
    p0 = A.probs(1.4, 0.9); assert np.allclose(probs_1x2(np.array([1.4]), np.array([0.9]), A.RHO)[0], [p0["H"], p0["D"], p0["A"]])
    res = {}
    for (hl, k), rho in itertools.product(L, rhos):
        res[hl, k, rho] = logloss(probs_1x2(L[hl, k][0][ok], L[hl, k][1][ok], rho), y[ok])
        print(f"vida media {hl} d, k={k}, RHO={rho}: log-loss {res[hl, k, rho]:.4f}")
    hl, k, rho = min(res, key=res.get)
    old = res[365, 5, -0.1]
    P = probs_1x2(L[hl, k][0], L[hl, k][1], rho)
    P0 = probs_1x2(L[365, 5][0], L[365, 5][1], -0.1)
    mk = 1 / e[["mH", "mD", "mA"]].to_numpy(float); mk /= mk.sum(1, keepdims=True)
    odds = e[["oH", "oD", "oA"]].to_numpy(float)
    ok &= np.isfinite(mk).all(1) & np.isfinite(odds).all(1)
    P, P0, mk, odds, y = P[ok], P0[ok], mk[ok], odds[ok], y[ok]
    print(f"\nMercado solo: log-loss {logloss(mk, y):.4f} ({ok.sum()} partidos con cuotas)")
    grid = {}
    for w, edge in itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], [0.02, 0.03, 0.05, 0.07, 0.10]):
        r, n, _ = grid[w, edge] = roi(P, mk, odds, y, w, edge)
        print(f"peso modelo {w}, EDGE {edge}: log-loss {logloss(w * P + (1 - w) * mk, y):.4f}, ROI {r:+.2%} en {n} apuestas")
    w, edge = max((c for c in grid if grid[c][1] >= 300), key=lambda c: grid[c][0])  # mínimo de apuestas: evita suerte
    if grid[w, edge][0] - grid[A.BLEND, A.EDGE][0] < 2 * grid[w, edge][2]:  # mejora dentro del ruido: se queda lo actual
        print(f"(mejor ROI {grid[w, edge][0]:+.2%} con peso {w}, EDGE {edge}: no supera el ruido, se mantiene {A.BLEND}/{A.EDGE})")
        w, edge = A.BLEND, A.EDGE
    before, after = roi(P0, mk, odds, y, 0.5, 0.05), grid[w, edge]
    print(f"\nANTES  (365 d, k=5, RHO=-0.1, peso 0.5, EDGE 0.05): log-loss modelo {old:.4f}, "
          f"mezcla {logloss(0.5 * P0 + 0.5 * mk, y):.4f}, ROI {before[0]:+.2%} en {before[1]}")
    print(f"MEJOR  ({hl} d, k={k}, RHO={rho}, peso {w}, EDGE {edge}): log-loss modelo {res[hl, k, rho]:.4f}, "
          f"mezcla {logloss(w * P + (1 - w) * mk, y):.4f}, ROI {after[0]:+.2%} en {after[1]}")
    (A.DATA / "football_params.json").write_text(json.dumps(
        {"half_life": hl, "k": k, "rho": rho, "blend": w, "edge": edge, "logloss": round(res[hl, k, rho], 4),
         "logloss_old": round(old, 4), "roi": round(after[0], 4), "bets": after[1], "roi_old": round(before[0], 4)}))


if __name__ == "__main__":
    main()
