"""Estadísticas de partido en fútbol (córners, tarjetas, faltas, tiros a puerta) para mercados de total y de equipo y
para «crear apuesta» (combinada del mismo partido). Mismo modelo que los goles: fuerzas local/visitante de cada equipo
con decaimiento y encogimiento (train_football.lambdas / analyst.strengths) y binomial negativa (totals_model).
`python football_stats.py` → backtest sin mirar el futuro (fuerzas recalculadas cada semana; dispersión ajustada en el
año anterior al test; test = 2 últimas temporadas) frente a la media de la liga, y covarianza entre mercados del mismo
partido (goles, córners, tarjetas…) → data/football_stats.json (lo lee card_lines)."""
import itertools, json
from pathlib import Path

import numpy as np
import pandas as pd

import analyst as A
import totals_model as T

PARAMS = Path(__file__).parent / "data" / "football_stats.json"  # sin A.DATA: analyst importa este módulo
STATS = {"corners": ("HC", "AC", "córners"), "cards": ("HY", "AY", "tarjetas"), "fouls": ("HF", "AF", "faltas"),
         "sot": ("HST", "AST", "tiros a puerta")}
COLS = ["HC", "AC", "HY", "AY", "HR", "AR", "HF", "AF", "HST", "AST"]
HLS, KS = [180, 365, 540, 1000], [2, 5, 10, 20, 40, 80]
BASE_K = 1e9  # k enorme = todos los equipos iguales a la media de la liga (la referencia a batir)
COMBO_MIN_P = 0.6
LONG_MIN_P = 0.55  # piernas de la combinada larga (hasta 5)
W = 0.5  # shortcut: peso del modelo frente al mercado sin margen sin ajustar (no hay cuotas históricas de córners); ajustarlo con senales.csv
_cache = {}


def prep(df):
    """Tarjetas = amarillas + rojas (lo que cuentan casi todas las casas); columnas a número."""
    df = df.copy()
    for c in COLS: df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["HY"], df["AY"] = df.HY + df.HR.fillna(0), df.AY + df.AR.fillna(0)
    return df


def median_line(x):
    """Línea a .5 independiente del modelo: mediana de los 200 valores anteriores de la liga."""
    return np.floor(x.rolling(200, min_periods=50).median().shift(1)) + 0.5


def backtest():
    import train_football as TF
    df = prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + COLS) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True))
    d = df.dropna(subset=["FTHG", "FTAG", *COLS[:4], "HF", "AF", "HST", "AST"]).sort_values("Date", ignore_index=True)
    end = d.Date.max(); since = end - pd.DateOffset(years=3)
    e = d[d.Date >= since].reset_index(drop=True)
    test = (e.Date >= end - pd.DateOffset(years=2)).to_numpy(); fit = ~test
    print(f"{len(d)} partidos con estadísticas; test {test.sum()} (2 últimas temporadas)")
    out, legs = {}, {}
    G = TF.lambdas(d, since, [A.HALF_LIFE], [A.SHRINK])[A.HALF_LIFE, A.SHRINK]  # goles: parámetros de producción
    mu_g = G[0] + G[1]
    legs["goals"] = (mu_g, 1e6, (e.FTHG + e.FTAG).to_numpy(), np.full(len(e), 2.5))
    for stat, (h, a, name) in STATS.items():
        s = d.assign(FTHG=d[h], FTAG=d[a])
        L = TF.lambdas(s, since, HLS, KS + [BASE_K])
        tot = (s.FTHG + s.FTAG)
        line, lh_, la_ = (x.groupby(s.Div).transform(median_line)[s.Date >= since].to_numpy() for x in (tot, s.FTHG, s.FTAG))
        y, yh, ya = (e[h] + e[a]).to_numpy(), e[h].to_numpy(), e[a].to_numpy()
        ok = np.isfinite(line) & np.isfinite(lh_) & np.isfinite(la_) & (L[365, BASE_K][0] > 0)
        F_, Te = fit & ok, test & ok

        def evaluate(hl, k):
            mu = L[hl, k][0] + L[hl, k][1]
            r = T.fit_nb(mu[F_], y[F_])
            return r, T.score("nbinom", mu[Te], r, y[Te], line[Te])

        base_r, base = evaluate(365, BASE_K)
        best = min(((hl, k, *evaluate(hl, k)) for hl, k in itertools.product(HLS, KS)), key=lambda b: b[3][0].mean())
        hl, k, r, sc = best
        mh, ma = L[hl, k]
        r_team = T.fit_nb(np.r_[mh[F_], ma[F_]], np.r_[yh[F_], ya[F_]])
        team = T.score("nbinom", np.r_[mh[Te], ma[Te]], r_team, np.r_[yh[Te], ya[Te]], np.r_[lh_[Te], la_[Te]])
        bh, ba = L[365, BASE_K]
        team0 = T.score("nbinom", np.r_[bh[Te], ba[Te]], T.fit_nb(np.r_[bh[F_], ba[F_]], np.r_[yh[F_], ya[F_]]),
                        np.r_[yh[Te], ya[Te]], np.r_[lh_[Te], la_[Te]])
        se = lambda x, b: float((x - b).std() / np.sqrt(len(x)))
        out[stat] = {"half_life": hl, "k": k, "r": round(r, 2), "r_team": round(r_team, 2),
                     "ll": round(float(sc[0].mean()), 4), "ll_base": round(float(base[0].mean()), 4),
                     "ou": round(float(sc[2].mean()), 4), "ou_base": round(float(base[2].mean()), 4), "ou_se": round(se(sc[2], base[2]), 4),
                     "team_ou": round(float(team[2].mean()), 4), "team_ou_base": round(float(team0[2].mean()), 4),
                     "team_ou_se": round(se(team[2], team0[2]), 4)}
        print(f"{name}: vida media {hl} d, k={k}, r={r:.1f} · log-loss más/menos total {sc[2].mean():.4f} vs media de liga "
              f"{base[2].mean():.4f} (±{out[stat]['ou_se']}) · de equipo {team[2].mean():.4f} vs {team0[2].mean():.4f} "
              f"(±{out[stat]['team_ou_se']})")
        legs[stat] = (mh + ma, r, y, line)
    # covarianza de «más de» entre mercados del mismo partido: lo que el producto de probabilidades no ve
    po, yo = {}, {}
    for st, (mu, r, y, line) in legs.items():
        ok = np.isfinite(line)
        po[st] = np.where(ok, 1 - T.cdf("nbinom", mu, r, np.floor(np.nan_to_num(line))), np.nan)
        yo[st] = np.where(ok, y > line, np.nan)
    cov = {}
    for a, b in itertools.combinations(legs, 2):
        m = test & np.isfinite(po[a]) & np.isfinite(po[b])
        c = (yo[a][m] * yo[b][m]).mean() - (po[a][m] * po[b][m]).mean() - ((yo[a][m] - po[a][m]).mean() * po[b][m].mean()
                                                                          + (yo[b][m] - po[b][m]).mean() * po[a][m].mean())
        cov[f"{a}|{b}"] = round(float(c), 4)
        print(f"covarianza {a} × {b}: {c:+.4f} (n={m.sum()})")
    out["cov"] = cov
    old = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
    out.update({k: old[k] for k in ("rho", "form", "market", "referee") if k in old})  # cópula (train_sgp.py) y forma (train_form.py)
    PARAMS.write_text(json.dumps(out, indent=1))


def _model(df, div, stat, today, P):
    ck = (div, stat, f"{today:%Y-%m-%d}")
    if ck not in _cache:
        h, a, _ = STATS[stat]
        d = df[(df.Div == div)].dropna(subset=[h, a])
        if len(d) < 200: _cache[ck] = None
        else:
            s = d.assign(FTHG=d[h], FTAG=d[a])
            st = A.strengths(s, div, today, P[stat]["k"], P[stat]["half_life"])
            tail = s.tail(200)
            line = lambda x: float(np.floor(x.median())) + 0.5
            _cache[ck] = (st, line(tail.FTHG + tail.FTAG), line(tail.FTHG), line(tail.FTAG))
    return _cache[ck]


def form_factor(df, stat, team, today, P):
    """exp(c · ln((media últimos 10 + 0.5) / (media últimos 76 + 0.5))) de la estadística a favor del equipo (train_form.py;
    c solo en las estadísticas donde mejoró fuera de muestra). 1 si no hay c o pocos partidos."""
    c = P.get("form", {}).get(stat, {}).get("c")
    if not c or stat not in STATS: return 1.0
    ck = ("form", stat, f"{today:%Y-%m-%d}")
    if ck not in _cache:
        h, a, _ = STATS[stat]
        d = df[df.Date < today].dropna(subset=[h, a])
        x = pd.concat([pd.DataFrame({"t": d.HomeTeam, "v": d[h], "D": d.Date}), pd.DataFrame({"t": d.AwayTeam, "v": d[a], "D": d.Date})])
        x = x.sort_values("D", kind="stable").groupby("t").tail(76).groupby("t").v
        n, short, long_ = x.count(), x.apply(lambda v: v.tail(10).mean()), x.mean()
        _cache[ck] = np.log((short + 0.5) / (long_ + 0.5)).where(n >= 20, 0.0).to_dict()
    return float(np.exp(c * _cache[ck].get(team, 0.0)))


MATCH_MK = {}  # (local, visitante) → (supremacía del local, logit P(más de 2.5)) sin margen; lo rellena analyst con set_market


def set_market(row):
    """Guarda las cuotas medias del partido (fila de fixtures.csv) para market_factor."""
    import train_market_stats as TM
    s, g = TM.market_features(pd.DataFrame([row]))
    if np.isfinite(s[0]) and np.isfinite(g[0]): MATCH_MK[row["HomeTeam"], row["AwayTeam"]] = (float(s[0]), float(g[0]))


def market_factor(stat, home, away, P):
    """(factor local, factor visitante) = exp(a·supremacía + b·(g − media)) de train_market_stats.py; (1, 1) sin cuotas."""
    m, mk = P.get("market", {}).get(stat), MATCH_MK.get((home, away))
    if not m or not mk: return 1.0, 1.0
    s, g = mk[0], mk[1] - m["g_mean"]
    return float(np.exp(m["a"] * s + m["b"] * g)), float(np.exp(-m["a"] * s + m["b"] * g))


def expected(df, div, stat, home, away, today, P):
    """(esperado local, esperado visitante, línea mediana total, local, visitante) o None si la liga no tiene datos.
    Incluye la forma reciente de cada equipo (form_factor) y las cuotas del partido (market_factor) donde mejoran fuera de muestra."""
    m = _model(df, div, stat, today, P)
    if not m: return None
    (mu_h, mu_a, s), lt, lth, lta = m
    g = lambda t, team: s[t].get(team, 1.0)
    fh, fa = market_factor(stat, home, away, P)  # cuotas del partido (favorito → más córners y tiros)
    return (mu_h * g("att_h", home) * g("def_a", away) * form_factor(df, stat, home, today, P) * fh,
            mu_a * g("att_a", away) * g("def_h", home) * form_factor(df, stat, away, today, P) * fa, lt, lth, lta)


# nombres de football-data → como los escribe ESPN (Champions / Europa League)
EURO_ALIASES = {"Man United": "Manchester United", "Man City": "Manchester City", "Inter": "Internazionale",
                "Sp Lisbon": "Sporting CP", "Paris SG": "Paris Saint-Germain", "Roma": "AS Roma", "Stuttgart": "VfB Stuttgart",
                "Feyenoord": "Feyenoord Rotterdam", "Ath Madrid": "Atletico Madrid", "Ath Bilbao": "Athletic Club",
                "Sp Braga": "Braga", "Ein Frankfurt": "Eintracht Frankfurt", "Leverkusen": "Bayer Leverkusen",
                "Dortmund": "Borussia Dortmund", "M'gladbach": "Borussia Monchengladbach", "Nott'm Forest": "Nottingham Forest",
                "Sociedad": "Real Sociedad", "Betis": "Real Betis", "Celta": "Celta Vigo", "PSV Eindhoven": "PSV",
                "Olympiakos": "Olympiacos", "Club Brugge": "Club Brugge", "St Gilloise": "Union St.-Gilloise"}


def team_div(df, name, today):
    """(liga, nombre en football-data) de un equipo con nombre de ESPN, o None (con exigencia alta: mejor nada que otro)."""
    import difflib
    ck = ("team", name, f"{today:%Y-%m-%d}")
    if ck not in _cache:
        r = df[df.Date >= today - pd.Timedelta(days=400)]
        last = pd.concat([r[["Date", "Div", "HomeTeam"]].rename(columns={"HomeTeam": "t"}),
                          r[["Date", "Div", "AwayTeam"]].rename(columns={"AwayTeam": "t"})]).sort_values("Date").groupby("t").Div.last()
        n, best = A.norm(name), (0.0, None)
        for t, div in last.items():
            m = A.norm(EURO_ALIASES.get(t, t))
            sc = 1.0 if m == n else 0.92 if len(min(m, n, key=len)) >= 5 and (f" {m} " in f" {n} " or f" {n} " in f" {m} ") \
                else difflib.SequenceMatcher(None, m, n).ratio()
            if sc > best[0]: best = (sc, (div, t))
        _cache[ck] = best[1] if best[0] >= 0.88 else None
    return _cache[ck]


def cross_expected(df, home, away, today, P):
    """Partido europeo entre equipos de ligas distintas (o de la misma): para cada estadística (y goles), esperado de cada
    equipo = media de local (visitante) de las dos ligas × ataque del equipo en su liga × defensa del rival en la suya;
    línea = mediana de las dos ligas. shortcut: no corrige el nivel de cada liga ni el estilo de la competición europea
    (sin datos de córners/tarjetas de Champions para medirlo). {stat: (eh, ea, lt, lth, lta)}; {} si falta algún equipo."""
    th, ta = team_div(df, home, today), team_div(df, away, today)
    if not th or not ta: return {}
    out = {}
    for stat in ["goals", *STATS]:
        h, a = ("FTHG", "FTAG") if stat == "goals" else STATS[stat][:2]
        Q = P.get(stat) or {"k": A.SHRINK, "half_life": A.HALF_LIFE}
        mh, ma = (_model(df, th[0], stat, today, P), _model(df, ta[0], stat, today, P)) if stat != "goals" else (None, None)
        if stat == "goals":
            mods = []
            for div in (th[0], ta[0]):
                d = df[df.Div == div]
                mods.append((A.strengths(d, div, today, Q["k"], Q["half_life"]), None, None, None))
            mh, ma = mods
        if not mh or not ma: continue
        (hh, ha, sh), (ah, aa, sa) = mh[0], ma[0]
        g = lambda s_, k, t: s_[k].get(t, 1.0)
        eh = np.sqrt(hh * ah) * g(sh, "att_h", th[1]) * g(sa, "def_a", ta[1]) * form_factor(df, stat, th[1], today, P)
        ea = np.sqrt(ha * aa) * g(sa, "att_a", ta[1]) * g(sh, "def_h", th[1]) * form_factor(df, stat, ta[1], today, P)
        ln = lambda x, y: float(np.floor((x + y) / 2)) + 0.5 if x is not None else 2.5
        out[stat] = (eh, ea, ln(mh[1], ma[1]), ln(mh[2], ma[2]), ln(mh[3], ma[3]))
    return out


def euro_lines(home, away, today):
    """Líneas de estadísticas para la tarjeta de un partido de Champions / Europa League (multisport); [] si falta un equipo."""
    if "euro_df" not in _cache:
        _cache["euro_df"] = prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + COLS) for f in sorted(A.HIST.glob("*.csv"))],
                                           ignore_index=True).dropna(subset=["FTHG", "FTAG"]).sort_values("Date", ignore_index=True))
    df, today = _cache["euro_df"], pd.Timestamp(today).tz_localize(None).normalize()
    if not PARAMS.exists(): return []
    cross = cross_expected(df, home, away, today, json.loads(PARAMS.read_text()))
    return card_lines(df, None, home, away, today, None, cross=cross) if cross else []


def over(mu, r, line): return float(1 - T.cdf("nbinom", mu, r, np.array(int(line))))  # P(X > line), línea a .5


def value_signals(df, div, home, away, today, offers):
    """Señales con cuota real (kambi.offers: córners), solo donde el modelo bate a la media de la liga.
    Mercado «<stat>_<t|h|a>_<o|u>» con su línea; por mercado, la línea de más ventaja."""
    if not PARAMS.exists() or not offers: return []
    P, best = json.loads(PARAMS.read_text()), {}
    for (stat, scope), lines in offers.items():
        e = expected(df, div, stat, home, away, today, P) if stat in P else None
        team = scope != "t"
        if not e or P[stat]["team_ou" if team else "ou"] >= P[stat]["team_ou_base" if team else "ou_base"]: continue
        mu, r = {"t": e[0] + e[1], "h": e[0], "a": e[1]}[scope], P[stat]["r_team" if team else "r"]
        for line, odds in lines.items():
            if line % 1 != 0.5: continue  # shortcut: solo líneas .5 (sin empate con la línea), las enteras casi no salen
            po, inv = over(mu, r, line), 1 / odds["o"] + 1 / odds["u"]
            for side, pm in (("o", po), ("u", 1 - po)):
                implied = 1 / odds[side] / inv
                p = W * pm + (1 - W) * implied
                v = {"market": f"{stat}_{scope}_{side}", "line": line, "odds": odds[side], "book": "Unibet", "p": round(p, 3),
                     "implied": round(implied, 3), "edge": round(p * odds[side] - 1, 3),
                     "pick": f"{'Más' if side == 'o' else 'Menos'} de {line:g} {STATS[stat][2]}" + (f" de {home if scope == 'h' else away}" if team else "")}
                if p >= A.MIN_P and v["edge"] > best.get(v["market"], {"edge": -1})["edge"]: best[v["market"]] = v
    return list(best.values())


STATS_ODDS = Path(__file__).parent / "data" / "kambi_stats_odds.csv"


def log_offers(df, div, home, away, today, start, offers):
    """Registra en data/kambi_stats_odds.csv cada línea de Unibet (todas las estadísticas, también las que no se apuestan)
    con la prob. del modelo de «más de», para medir modelo vs casa por estadística cuando acaben los partidos."""
    if not PARAMS.exists() or not offers: return 0
    P, rows = json.loads(PARAMS.read_text()), []
    for (stat, scope), lines in offers.items():
        e = expected(df, div, stat, home, away, today, P) if stat in P else None
        if not e: continue
        mu, r = {"t": e[0] + e[1], "h": e[0], "a": e[1]}[scope], P[stat]["r_team" if scope != "t" else "r"]
        for line, o in lines.items():
            rows.append({"seen": pd.Timestamp.now(tz="UTC").isoformat(), "start": pd.Timestamp(start).isoformat(), "div": div,
                         "home": home, "away": away, "stat": stat, "scope": scope, "line": line, "o_over": o["o"], "o_under": o["u"],
                         "p_over": round(over(mu, r, line) if line % 1 == 0.5 else float(1 - T.cdf("nbinom", mu, r, np.array(int(line)))), 4),
                         "mu": round(mu, 3)})
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(STATS_ODDS) if STATS_ODDS.exists() else new.iloc[:0]
        pd.concat([old, new]).drop_duplicates(["start", "home", "away", "stat", "scope", "line"], keep="last").to_csv(STATS_ODDS, index=False)
    return len(rows)


def stats_odds_report(df, edge=0.05):
    """Modelo frente a Unibet por estadística y alcance con los partidos ya jugados (df de football-data con prep):
    {(stat, scope): {n, ll_modelo, ll_mercado, apuestas, roi, roi_se}} con la última cuota previa de cada línea .5."""
    if not STATS_ODDS.exists(): return {}
    o = pd.read_csv(STATS_ODDS)
    o = o[o.line % 1 == 0.5].assign(Date=pd.to_datetime(o.start, utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize())
    m = o.merge(df.assign(Date=df.Date.dt.normalize()), left_on=["div", "Date", "home", "away"], right_on=["Div", "Date", "HomeTeam", "AwayTeam"])
    out = {}
    for (stat, scope), g in m.groupby(["stat", "scope"]):
        h, a = ("FTHG", "FTAG") if stat == "goals" else STATS[stat][:2]
        x = {"t": g[h] + g[a], "h": g[h], "a": g[a]}[scope]
        g = g[x.notna()]; y = (x[x.notna()] > g.line).to_numpy(float)
        if g.empty: continue
        po = g.p_over.clip(1e-6, 1 - 1e-6).to_numpy(); inv = 1 / g.o_over + 1 / g.o_under
        pm = (1 / g.o_over / inv).clip(1e-6, 1 - 1e-6).to_numpy()
        ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        ev_o, ev_u = po * g.o_over.to_numpy() - 1, (1 - po) * g.o_under.to_numpy() - 1
        side_o = ev_o >= ev_u; ev = np.where(side_o, ev_o, ev_u); bet = ev >= edge
        win = np.where(side_o, y == 1, y == 0)
        gain = np.where(win, np.where(side_o, g.o_over, g.o_under) - 1, -1.0)[bet]
        out[stat, scope] = {"n": len(g), "ll_modelo": round(ll(po), 4), "ll_mercado": round(ll(pm), 4), "apuestas": int(bet.sum()),
                            "roi": round(float(gain.mean()), 4) if bet.any() else None,
                            "roi_se": round(float(gain.std() / np.sqrt(bet.sum())), 4) if bet.sum() > 1 else None}
    return out


def result(market, line, r):
    """True/False con la fila de football-data.co.uk del partido; None si aún no trae la estadística."""
    stat, scope, side = market.split("_")
    h, a, _ = STATS[stat]
    x = {"t": r.get(h, np.nan) + r.get(a, np.nan), "h": r.get(h, np.nan), "a": r.get(a, np.nan)}[scope]
    return None if pd.isna(x) else bool(x > line) == (side == "o")


def joint(legs, cov, rho=None):
    """Prob. de acertar todas las piernas (cada una {stat, sign, p}; `var` = «<stat>_<t|h|a>» para la cópula).
    Con `rho` (train_sgp.py: cópula gaussiana, correlación latente de cada par, medida fuera de muestra) y todas las
    piernas en ella → P(Z_i ≤ Φ⁻¹(p_i)) con la correlación de signo corregido por «menos de». Si no, producto corregido
    por las covarianzas por pares (Bahadur de 1.er orden; shortcut: medida a la línea mediana)."""
    names = (rho or {}).get("names", [])
    if len(legs) >= 2 and all(l.get("var") in names for l in legs):
        from scipy import stats as S
        idx = [names.index(l["var"]) for l in legs]
        sg = np.array([l["sign"] for l in legs], float)
        R = np.array(rho["R"])[np.ix_(idx, idx)] * np.outer(sg, sg)
        z = S.norm.ppf(np.clip([l["p"] for l in legs], 1e-9, 1 - 1e-9))
        pj = float(S.multivariate_normal(np.zeros(len(legs)), R, allow_singular=True).cdf(z)) if len(legs) > 1 else float(legs[0]["p"])
        return min(max(pj, 0.0), min(l["p"] for l in legs))
    p = float(np.prod([l["p"] for l in legs]))
    corr = sum(cov.get(f"{x['stat']}|{y['stat']}", cov.get(f"{y['stat']}|{x['stat']}", 0.0)) * x["sign"] * y["sign"] / (x["p"] * y["p"])
               for x, y in itertools.combinations(legs, 2))
    return min(max(p * (1 + corr), 0.0), min(l["p"] for l in legs))


POOL = []  # piernas candidatas de la última card_lines (una por estadístico, p ≥ LONG_MIN_P): para la combinada con cuota
LAST = []  # combinadas de la última card_lines (las registra analyst para liquidarlas: historial real de acierto)
SGP_LOG = Path(__file__).parent / "data" / "sgp_log.csv"


def _combo(kind, legs, pj):
    return {"kind": kind, "p": round(float(pj), 4),
            "legs": [{"var": l["var"], "sign": l["sign"], "line": float(l["line"]), "p": round(float(l["p"]), 4), "text": l["text"]} for l in legs]}


def leg_hit(leg, r):
    """True/False de una pierna con la fila de football-data; None si la fila aún no trae la estadística."""
    stat, sc = leg["var"].rsplit("_", 1)
    h, a = ("FTHG", "FTAG") if stat == "goals" else STATS[stat][:2]
    x = {"t": r.get(h, np.nan) + r.get(a, np.nan), "h": r.get(h, np.nan), "a": r.get(a, np.nan)}[sc]
    return None if pd.isna(x) else bool(x > leg["line"]) == (leg["sign"] > 0)


SIM_EDGE = 0.05  # ventaja mínima (prob. conjunta del modelo × cuota aprox. − 1) para simular la combinada a 1 u


def price_combos(combos, offers, row=None):
    """Pone a cada pierna la cuota de Unibet de su línea exacta (offers de kambi; goles: cuota media de Más/Menos 2.5 de
    la fila de fixtures.csv) y a la combinada la cuota aprox. = producto de piernas y si se SIMULA (ventaja ≥ SIM_EDGE).
    shortcut: el «crear apuesta» real de la casa suele pagar menos que el producto (descuenta la correlación); por eso
    la ventaja se calcula con la prob. conjunta del modelo, que sí la incluye, y se mide con el resultado real."""
    for c in combos:
        for l in c["legs"]:
            stat, sc = l["var"].rsplit("_", 1)
            if stat == "goals":
                o = (row or {}).get("Avg>2.5" if l["sign"] > 0 else "Avg<2.5")
                l["odds"] = float(o) if o and pd.notna(o) and l["line"] == 2.5 else None
            else:
                o = (offers or {}).get((stat, sc), {}).get(l["line"])
                l["odds"] = o["o" if l["sign"] > 0 else "u"] if o else None
        if all(l.get("odds") for l in c["legs"]):
            c["odds"] = round(float(np.prod([l["odds"] for l in c["legs"]])), 2)
            c["edge"] = round(c["p"] * c["odds"] - 1, 3)
            c["sim"] = c["edge"] >= SIM_EDGE
    return combos


def priced_combo(pool, offers, row=None, rho=None, cov=None):
    """«Combinada con cuota»: entre las piernas candidatas que SÍ tienen cuota de Unibet (o goles de fixtures), la de 2-3
    piernas con más ventaja (prob. conjunta con la cópula × producto de cuotas − 1). None si no hay ≥ 2 con cuota."""
    import itertools
    legs = [dict(l) for l in pool]
    price_combos([{"legs": legs, "p": 0, "kind": "x"}], offers, row)
    legs = [l for l in legs if l.get("odds")]
    if len(legs) < 2: return None
    if rho is None or cov is None:
        P = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
        rho, cov = P.get("rho"), P.get("cov", {})
    best = None
    for n in (2, 3):
        for c in itertools.combinations(legs, n):
            pj = joint(list(c), cov, rho)
            odds = float(np.prod([l["odds"] for l in c]))
            if best is None or pj * odds - 1 > best["edge"]:
                best = {"kind": "valor", "p": round(pj, 4), "legs": list(c), "odds": round(odds, 2), "edge": round(pj * odds - 1, 3)}
    best["sim"] = best["edge"] >= SIM_EDGE
    return best


def sim_lines(combos):
    """Líneas de tarjeta: cuota aprox. de Unibet de cada combinada y si el bot la simula."""
    out = []
    for c in combos:
        if c.get("odds"):
            out.append(f"🎯 {'Simulada' if c.get('sim') else 'No simulada'} ({ {'corta': 'crear apuesta', 'larga': 'larga', 'valor': 'con cuota: ' + ' + '.join(l['text'] for l in c['legs'])}.get(c['kind'], c['kind'].upper())}): "
                       f"cuota {'DraftKings' if c['kind'] in ('nba', 'nfl', 'nhl') else 'Unibet'} aprox. {c['odds']:.2f} (producto de piernas) · prob. {c['p']:.0%} · ventaja {c['edge']:+.0%}"
                       + (" → 1 u en la banca de simulación" if c.get("sim") else ""))
    return out


def settle_sgp(state, recent, today):
    """Liquida las combinadas registradas (state["sgp"]) con los partidos ya jugados (recent: clave → fila) y las añade
    a data/sgp_log.csv; borra las de más de 10 días sin resultado."""
    rows, keep = [], {}
    for k, v in (state.get("sgp") or {}).items():
        r = recent.get(k)
        hits = [[leg_hit(l, r) for l in c["legs"]] for c in v["combos"]] if r is not None else None
        if hits and all(h is not None for hh in hits for h in hh):
            for c, hh in zip(v["combos"], hits):
                rows.append({"fecha": v["start"][:10], "partido": v["match"], "tipo": c["kind"], "piernas": len(c["legs"]),
                             "p": c["p"], "acierto": int(all(hh)), "piernas_acertadas": sum(hh),
                             "p_piernas": round(float(np.mean([l["p"] for l in c["legs"]])), 4),
                             "cuota": c.get("odds"), "ventaja": c.get("edge"), "simulada": int(bool(c.get("sim"))),
                             "ganancia": (c["odds"] - 1 if all(hh) else -1.0) if c.get("sim") else None,
                             "piernas_txt": " + ".join(l["text"] for l in c["legs"])})
        elif v["start"][:10] >= f"{today - pd.Timedelta(days=10):%Y-%m-%d}": keep[k] = v
    state["sgp"] = keep
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(SGP_LOG) if SGP_LOG.exists() else new.iloc[:0]
        pd.concat([old, new]).to_csv(SGP_LOG, index=False)
    return len(rows)


def sgp_history():
    """Texto con el historial real de las combinadas liquidadas (prob. media prevista frente a acierto), o ""."""
    if not SGP_LOG.exists(): return ""
    d = pd.read_csv(SGP_LOG)
    if d.empty: return ""
    out = []
    for kind, g in d.groupby("tipo"):
        out.append(f"{ {'corta': 'cortas', 'larga': 'largas', 'valor': 'con cuota'}.get(kind, kind.upper())} {len(g)}: previsto {g.p.mean():.0%}, acertadas {g.acierto.mean():.0%} "
                   f"(piernas {g.piernas_acertadas.sum() / g.piernas.sum():.0%} vs {g.p_piernas.mean():.0%})")
    txt = "📈 Historial real: " + " · ".join(out)
    if "simulada" in d and (sm := d[d.simulada == 1]).shape[0]:
        txt += (f"\n🎯 Simuladas a 1 u con cuota de Unibet: {len(sm)} · acertadas {sm.acierto.mean():.0%} · "
                f"<b>{sm.ganancia.sum():+.1f} u</b> (ROI {sm.ganancia.mean():+.0%})")
    return txt


def card_lines(df, div, home, away, today, P_goals, cross=None):
    """`cross`: {stat: esperados} de cross_expected (partidos europeos). Líneas para la tarjeta del partido (texto HTML), con la combinada «crear apuesta» y su cuota justa."""
    LAST.clear(); POOL.clear()
    if not PARAMS.exists(): return []
    P = json.loads(PARAMS.read_text())
    lines, legs = ["", "📐 <b>Estadísticas esperadas</b>"], []
    rho = P.get("rho")
    for stat, (_, _, name) in STATS.items():
        e = cross.get(stat) if cross is not None else expected(df, div, stat, home, away, today, P)
        if not e: continue
        eh, ea, lt, lth, lta = e
        pt, ph, pa = over(eh + ea, P[stat]["r"], lt), over(eh, P[stat]["r_team"], lth), over(ea, P[stat]["r_team"], lta)
        lines.append(f"{name.capitalize()} {eh:.1f} - {ea:.1f}: total +{lt} {pt:.0%} · {A.esc(home)} +{lth} {ph:.0%} · "
                     f"{A.esc(away)} +{lta} {pa:.0%}")
        # solo mercados donde el modelo bate a la media de la liga; por estadístico, la pierna más probable (total o de
        # equipo: la cópula de train_sgp no mide total × equipo del mismo estadístico, y la casa no suele combinarlas)
        opts = [(sc, p0, ln, f" de {A.esc(t)}" if t else "") for sc, p0, ln, t, ok in
                (("t", pt, lt, "", P[stat]["ou"] < P[stat]["ou_base"]),
                 ("h", ph, lth, home, rho and P[stat]["team_ou"] < P[stat]["team_ou_base"]),
                 ("a", pa, lta, away, rho and P[stat]["team_ou"] < P[stat]["team_ou_base"])) if ok]
        if opts:
            sc, p0, ln, who = max(opts, key=lambda o: max(o[1], 1 - o[1]))
            legs.append({"stat": stat, "var": f"{stat}_{sc}", "sign": 1 if p0 >= 0.5 else -1, "p": max(p0, 1 - p0), "line": ln,
                         "text": f"{'Más' if p0 >= 0.5 else 'Menos'} de {ln} {name}{who}"})
    if len(lines) == 2: return []
    if cross is not None:  # partido europeo: goles con el mismo cruce (Poisson), en vez del modelo de la liga
        from scipy import stats as S
        lines[1] += " · cruce de las dos ligas, sin backtest en Europa"
        P_goals = {"O25": float(1 - S.poisson.cdf(2, sum(cross["goals"][:2])))} if "goals" in cross else P_goals
        if not P_goals: return lines
    po = P_goals["O25"]
    legs.append({"stat": "goals", "var": "goals_t", "sign": 1 if po >= 0.5 else -1, "p": max(po, 1 - po), "line": 2.5,
                 "text": f"{'Más' if po >= 0.5 else 'Menos'} de 2.5 goles"})
    allr = sorted(legs, key=lambda l: -l["p"])
    POOL[:] = [l for l in allr if l["p"] >= LONG_MIN_P]
    legs = [l for l in allr if l["p"] >= COMBO_MIN_P][:3]
    if len(legs) >= 2:
        pj = joint(legs, P.get("cov", {}), rho)
        lines.append(f"🧩 Crear apuesta: {' + '.join(l['text'] for l in legs)} → prob. {pj:.0%}, cuota justa {1 / pj:.2f} "
                     f"(solo vale si la casa paga más)")
        LAST.append(_combo("corta", legs, pj))
    long_ = [l for l in allr if l["p"] >= LONG_MIN_P][:5]  # combinada larga: más cuota, una pierna por estadístico (cópula)
    if len(long_) > max(len(legs), 2) and rho:
        pj = joint(long_, P.get("cov", {}), rho)
        lines.append(f"🚀 Combinada larga: {' + '.join(l['text'] for l in long_)} → prob. {pj:.0%}, cuota justa {1 / pj:.2f}")
        LAST.append(_combo("larga", long_, pj))
    return lines


if __name__ == "__main__":
    cov = {"goals|corners": 0.02}
    l1 = [{"stat": "goals", "sign": 1, "p": 0.6}, {"stat": "corners", "sign": 1, "p": 0.6}]
    assert abs(joint(l1, cov) - 0.36 * (1 + 0.02 / 0.36)) < 1e-9
    assert abs(joint([l1[0], {**l1[1], "sign": -1}], cov) - (0.36 - 0.02)) < 1e-9  # «menos de» invierte la covarianza
    backtest()
