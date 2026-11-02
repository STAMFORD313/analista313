"""Entrena el modelo de fútbol con backtest sin mirar el futuro: cada semana de las 2 últimas temporadas se predice
solo con los partidos anteriores. Elige vida media, encogimiento k y RHO por log-loss 1X2, y el peso del modelo
frente al mercado y la ventaja mínima por ROI apostando a la cuota media (Avg, o B365) del CSV.
Guarda los mejores en data/football_params.json (lo lee analyst.py)."""
import itertools, json, math

import numpy as np
import pandas as pd

import analyst as A

ODDS = [f"{b}{m}" for b in ("PS", "Avg", "B365", "PSC", "AvgC", "B365C") for m in "HDA"] +        [f"{b}{m}" for b in ("P", "Avg", "B365") for m in (">2.5", "<2.5")] + ["AHh", "AvgAHH", "AvgAHA", "B365AHH", "B365AHA"]
N = 11
I, J = np.indices((N, N))
DV = np.arange(-(N - 1), N)  # diferencias de goles posibles
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


def score_matrix(lh, la, rho):
    """Matriz de marcadores (N partidos, goles local, goles visitante) como analyst.probs, vectorizada."""
    ph = np.exp(-lh)[:, None] * lh[:, None] ** np.arange(N) / FACT
    pa = np.exp(-la)[:, None] * la[:, None] ** np.arange(N) / FACT
    M = ph[:, :, None] * pa[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho; M[:, 1, 0] *= 1 + la * rho; M[:, 0, 1] *= 1 + lh * rho; M[:, 1, 1] *= 1 - rho
    return M


def probs_1x2(lh, la, rho):
    """H, D, A como analyst.probs pero vectorizado (N partidos a la vez)."""
    M = score_matrix(lh, la, rho)
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
    win = np.zeros(odds.shape, bool); win[np.arange(len(y)), y] = True
    return roi_g(w * P + (1 - w) * mk, odds, np.where(win, odds - 1, -1.0), edge)


def roi_g(p, odds, G, edge):
    """Como roi() con la ganancia por unidad G de cada opción ya calculada (sirve para hándicap con nulas y medias)."""
    ev = p * odds - 1
    bet = (ev >= edge) & (ev <= A.MAX_EDGE) & (p >= A.MIN_P) & np.isfinite(odds)
    gain = G[bet]
    return (float(gain.mean()) if len(gain) else 0.0), len(gain), (float(gain.std() / math.sqrt(len(gain))) if len(gain) else 1.0)


def ah_side(D, sign, line, odds, diff):
    """Hándicap asiático de un lado (sign 1 local, -1 visitante): (prob. equivalente = cobro esperado / cuota, ganancia real
    por unidad). Las líneas de cuarto se parten en dos medias apuestas, como analyst.ah_factor."""
    q = (line * 4) % 2 == 1
    exp = real = 0
    for h in (np.where(q, line - 0.25, line), np.where(q, line + 0.25, line)):
        m = sign * DV[None, :] + h[:, None]
        exp = exp + ((D * (m > 0)).sum(1) * odds + (D * (m == 0)).sum(1)) / 2
        r = sign * diff + h
        real = real + np.where(r > 0, odds, np.where(r == 0, 1.0, 0.0)) / 2
    return exp / odds, real - 1


def market(name, Pm, mk, odds, G, y, yr, ok, cur=None):
    """Elige en el año -2 el peso del modelo (log-loss) y el EDGE (ROI a cuota media; si el ROI es claramente negativo
    o no hay apuestas suficientes, mercado apagado: EDGE 1) y mide en el año -1. Compara con una mezcla logística (Platt sobre log-probs de ambos)."""
    from sklearn.linear_model import LogisticRegression
    B, C = ok & (yr == 1), ok & (yr == 2)
    mix = lambda w: w * Pm + (1 - w) * mk
    lls = {w: logloss(mix(w)[B], y[B]) for w in np.round(np.arange(0, 0.51, 0.05), 2)}
    w = min(lls, key=lls.get)
    if cur is not None and logloss(mix(cur)[C], y[C]) <= logloss(mix(w)[C], y[C]): w = cur  # reentreno: no empeorar lo actual
    X = np.log(np.clip(np.column_stack([Pm, mk]), 1e-9, 1))
    platt = LogisticRegression(C=10, max_iter=2000).fit(X[B], y[B]).predict_proba(X[C])
    print(f"{name}: log-loss año -1 mercado {logloss(mk[C], y[C]):.4f} · modelo {logloss(Pm[C], y[C]):.4f} · "
          f"mezcla 0.5 {logloss(mix(0.5)[C], y[C]):.4f} · mezcla {w} (elegida en año -2) {logloss(mix(w)[C], y[C]):.4f} · "
          f"Platt {logloss(platt, y[C]):.4f} ({C.sum()} partidos)")
    rB = {ed: roi_g(mix(w)[B], odds[B], G[B], ed) for ed in (A.EDGE, 0.07, 0.10, 0.15)}  # bankroll.py nunca apuesta por debajo de A.EDGE
    ed = max((x for x in rB if rB[x][1] >= 100), key=lambda x: rB[x][0], default=1.0)
    if ed < 1 and rB[ed][0] + 2 * rB[ed][2] < 0: ed = 1.0  # claramente negativo incluso en su mejor EDGE: no se apuesta
    rC, r0 = roi_g(mix(w)[C], odds[C], G[C], ed), roi_g(mix(0.5)[C], odds[C], G[C], 0.05)
    print(f"  EDGE {ed}: ROI año -2 {rB.get(ed, (0, 0))[0]:+.2%} · año -1 {rC[0]:+.2%} ± {rC[2]:.2%} en {rC[1]} apuestas "
          f"(antes, 0.5/0.05: {r0[0]:+.2%} en {r0[1]})")
    return {"w": float(w), "edge": ed, "logloss": round(logloss(mix(w)[C], y[C]), 4), "logloss_old": round(logloss(mix(0.5)[C], y[C]), 4),
            "roi": round(rC[0], 4), "bets": rC[1], "roi_old": round(r0[0], 4), "bets_old": r0[1]}


def main():
    import features_extra as FX
    from sklearn.linear_model import LogisticRegression
    df = load()
    end = df.Date.max()
    since = end - pd.DateOffset(years=3)  # año -3 ajusta features, -2 elige mezcla/EDGE, -1 solo mide
    hls, ks, rhos = [180, 270, 365, 540, 730], [2, 5, 10, 20], [-0.2, -0.15, -0.1, -0.05, 0.0]
    L = lambdas(df, since, hls, ks)
    e = df[df.Date >= since].reset_index(drop=True)
    yr = (e.Date >= since + pd.DateOffset(years=1)).to_numpy(int) + (e.Date >= since + pd.DateOffset(years=2)).to_numpy(int)
    y = np.select([e.FTHG > e.FTAG, e.FTHG == e.FTAG], [0, 1], 2)
    ok = L[hls[0], ks[0]][0] > 0  # sin historial previo: fuera
    # comprobación: el vectorizado coincide con analyst.probs
    p0 = A.probs(1.4, 0.9); assert np.allclose(probs_1x2(np.array([1.4]), np.array([0.9]), A.RHO)[0], [p0["H"], p0["D"], p0["A"]])
    res, l2 = {}, ok & (yr > 0)  # vida media, k y RHO: log-loss del Poisson en las 2 últimas temporadas
    for (hl, k), rho in itertools.product(L, rhos):
        res[hl, k, rho] = logloss(probs_1x2(L[hl, k][0][l2], L[hl, k][1][l2], rho), y[l2])
        print(f"vida media {hl} d, k={k}, RHO={rho}: log-loss {res[hl, k, rho]:.4f}")
    hl, k, rho = min(res, key=res.get)
    print(f"ANTES (365 d, k=5, RHO=-0.1) {res[365, 5, -0.1]:.4f} → MEJOR ({hl} d, k={k}, RHO={rho}) {res[hl, k, rho]:.4f}\n")
    out = {"half_life": hl, "k": k, "rho": rho, "logloss": round(res[hl, k, rho], 4), "logloss_old": round(res[365, 5, -0.1], 4)}
    M = score_matrix(L[hl, k][0], L[hl, k][1], rho)
    num = lambda *cols: np.column_stack([e[c].to_numpy(float) for c in cols])
    fill = lambda *alts: np.where(np.isfinite(alts[0]), alts[0], fill(*alts[1:]) if len(alts) > 1 else np.nan)
    norm = lambda o: (1 / o) / (1 / o).sum(1, keepdims=True)
    # 1X2: Poisson corregido con features_extra (regresión ajustada solo en el año -3), como en el bot
    F = FX.features(df)[(df.Date >= since).to_numpy()].reset_index(drop=True)
    P = np.stack([M[:, I > J].sum(1), M[:, I == J].sum(1), M[:, I < J].sum(1)], 1)
    X = FX.design(P, F, {c: float(F[c].std()) or 1.0 for c in FX.FEATS})
    mk, odds = norm(num("mH", "mD", "mA")), num("oH", "oD", "oA")
    o1 = ok & np.isfinite(mk).all(1) & np.isfinite(odds).all(1) & F[["elo_diff", "rest_diff"]].notna().all(1).to_numpy()
    Q = LogisticRegression(C=10, max_iter=2000).fit(X[o1 & (yr == 0)], y[o1 & (yr == 0)]).predict_proba(X)
    win = np.zeros(odds.shape, bool); win[np.arange(len(y)), y] = True
    out["1x2"] = market("1X2", Q, mk, odds, np.where(win, odds - 1, -1.0), y, yr, o1, A.W["1x2"])
    # Más/Menos 2.5
    po = M[:, I + J > 2].sum(1)
    mk = norm(fill(num("P>2.5", "P<2.5"), num("Avg>2.5", "Avg<2.5")))
    odds = fill(num("Avg>2.5", "Avg<2.5"), num("B365>2.5", "B365<2.5"))
    yo = (e.FTHG + e.FTAG <= 2).to_numpy(int)
    G = np.where(np.arange(2)[None, :] == yo[:, None], odds - 1, -1.0)
    out["ou"] = market("Más/Menos 2.5", np.column_stack([po, 1 - po]), mk, odds, G, yo, yr, ok & np.isfinite(mk).all(1) & np.isfinite(odds).all(1), A.W["ou"])
    # Hándicap asiático (prob. equivalente, como analyst.ah_bets); log-loss solo con líneas de medio gol (sin nulas)
    D = np.stack([M[:, I - J == d].sum(1) for d in DV], 1)
    line, diff = e.AHh.to_numpy(float), (e.FTHG - e.FTAG).to_numpy(float)
    odds = fill(num("AvgAHH", "AvgAHA"), num("B365AHH", "B365AHA"))
    (ph, gh), (pa, ga) = ah_side(D, 1, line, odds[:, 0], diff), ah_side(D, -1, -line, odds[:, 1], diff)
    half = (line * 2) % 2 == 1
    oa = ok & np.isfinite(line) & np.isfinite(odds).all(1) & half
    out["ah"] = market("Hándicap asiático", np.column_stack([ph, pa]), norm(odds), odds, np.column_stack([gh, ga]),
                       (diff + line < 0).astype(int), yr, oa, A.W["ah"])
    (A.DATA / "football_params.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
