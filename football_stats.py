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
    out.update({k: old[k] for k in ("rho", "form") if k in old})  # cópula (train_sgp.py) y forma (train_form.py)
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


def expected(df, div, stat, home, away, today, P):
    """(esperado local, esperado visitante, línea mediana total, local, visitante) o None si la liga no tiene datos.
    Incluye la forma reciente de cada equipo (form_factor) donde mejora fuera de muestra."""
    m = _model(df, div, stat, today, P)
    if not m: return None
    (mu_h, mu_a, s), lt, lth, lta = m
    g = lambda t, team: s[t].get(team, 1.0)
    return (mu_h * g("att_h", home) * g("def_a", away) * form_factor(df, stat, home, today, P),
            mu_a * g("att_a", away) * g("def_h", home) * form_factor(df, stat, away, today, P), lt, lth, lta)


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


def card_lines(df, div, home, away, today, P_goals, cross=None):
    """`cross`: {stat: esperados} de cross_expected (partidos europeos). Líneas para la tarjeta del partido (texto HTML), con la combinada «crear apuesta» y su cuota justa."""
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
            legs.append({"stat": stat, "var": f"{stat}_{sc}", "sign": 1 if p0 >= 0.5 else -1, "p": max(p0, 1 - p0),
                         "text": f"{'Más' if p0 >= 0.5 else 'Menos'} de {ln} {name}{who}"})
    if len(lines) == 2: return []
    if cross is not None:  # partido europeo: goles con el mismo cruce (Poisson), en vez del modelo de la liga
        from scipy import stats as S
        lines[1] += " · cruce de las dos ligas, sin backtest en Europa"
        P_goals = {"O25": float(1 - S.poisson.cdf(2, sum(cross["goals"][:2])))} if "goals" in cross else P_goals
        if not P_goals: return lines
    po = P_goals["O25"]
    legs.append({"stat": "goals", "var": "goals_t", "sign": 1 if po >= 0.5 else -1, "p": max(po, 1 - po),
                 "text": f"{'Más' if po >= 0.5 else 'Menos'} de 2.5 goles"})
    allr = sorted(legs, key=lambda l: -l["p"])
    legs = [l for l in allr if l["p"] >= COMBO_MIN_P][:3]
    if len(legs) >= 2:
        pj = joint(legs, P.get("cov", {}), rho)
        lines.append(f"🧩 Crear apuesta: {' + '.join(l['text'] for l in legs)} → prob. {pj:.0%}, cuota justa {1 / pj:.2f} "
                     f"(solo vale si la casa paga más)")
    long_ = [l for l in allr if l["p"] >= LONG_MIN_P][:5]  # combinada larga: más cuota, una pierna por estadístico (cópula)
    if len(long_) > max(len(legs), 2) and rho:
        pj = joint(long_, P.get("cov", {}), rho)
        lines.append(f"🚀 Combinada larga: {' + '.join(l['text'] for l in long_)} → prob. {pj:.0%}, cuota justa {1 / pj:.2f}")
    return lines


if __name__ == "__main__":
    cov = {"goals|corners": 0.02}
    l1 = [{"stat": "goals", "sign": 1, "p": 0.6}, {"stat": "corners", "sign": 1, "p": 0.6}]
    assert abs(joint(l1, cov) - 0.36 * (1 + 0.02 / 0.36)) < 1e-9
    assert abs(joint([l1[0], {**l1[1], "sign": -1}], cov) - (0.36 - 0.02)) < 1e-9  # «menos de» invierte la covarianza
    backtest()
