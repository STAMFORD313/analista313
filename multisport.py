"""NBA, NFL, NHL, MLB y tenis ATP/WTA vía la API pública de ESPN: 10 años de historial, Elo, lesiones,
cuotas DraftKings, señales de valor, IA, grafo, PDF y Telegram. Reutiliza las utilidades de analyst.py."""
import json, math

import pandas as pd
import requests

import analyst as A
import bankroll
import football_stats
import tennis_model, totals_model, ufc_model

ESPN = "https://site.api.espn.com/apis/site/v2/sports"
SPORTS = {  # liga: (ruta ESPN, nombre, deporte, ventaja local en puntos Elo, K; K=None → K dinámico de tenis)
    "nba": ("basketball/nba", "🏀 NBA", "baloncesto", 100, 20),
    "nfl": ("football/nfl", "🏈 NFL", "fútbol americano", 55, 20),
    "nhl": ("hockey/nhl", "🏒 NHL", "hockey hielo", 50, 8),
    "mlb": ("baseball/mlb", "⚾ MLB", "béisbol", 24, 4),
    "atp": ("tennis/atp", "🎾 Tenis ATP", "tenis", 0, None),
    "wta": ("tennis/wta", "🎾 Tenis WTA", "tenis", 0, None),
    "ufc": ("mma/ufc", "🥊 UFC", "MMA", 0, None),
}
YEARS = 10
DIR = A.DATA / "espn"
_P = json.loads((A.DATA / "espn_params.json").read_text()) if (A.DATA / "espn_params.json").exists() else {}  # lo ajusta train_espn.py
for _l, _p in _P.items(): SPORTS[_l] = (*SPORTS[_l][:3], _p["hfa"], _p["K"])
_X = json.loads((A.DATA / "espn_extra_params.json").read_text()) if (A.DATA / "espn_extra_params.json").exists() else {}
SPORTS.update({l: (*v, _X[l]["hfa"], _X[l]["K"]) for l, v in {  # competiciones de espn_extra.py (agente-datos)
    "cfb": ("football/college-football", "🏈 NCAA Football", "fútbol americano"),
    "cbb": ("basketball/mens-college-basketball", "🏀 NCAA Basketball", "baloncesto"),
    "ucl": ("soccer/uefa.champions", "🏆 Champions League", "fútbol"),
    "uel": ("soccer/uefa.europa", "🏆 Europa League", "fútbol")}.items() if l in _X})  # Conference: sin señal en el backtest
_P = {**_X, **_P}
STARTERS = json.loads((A.DATA / "starters.json").read_text()) if (A.DATA / "starters.json").exists() else {}  # starters.py
SOCCER = ("ucl", "uel")  # 1X2 con empate: un Elo común con la Conference y probs. con espn_extra.soccer_probs
TOT = json.loads(totals_model.PARAMS.read_text()) if totals_model.PARAMS.exists() else {}  # lo ajusta totals_model.py
SPREAD = {**TOT.get("spread", {}), **{l: c["spread"] for l, c in _P.items() if "spread" in c}}  # margen desde el Elo (NBA, NFL)
TOT = {l: c for l, c in TOT.items() if l != "spread" and c["ou"] < c["ou_old"]}  # solo ligas donde mejora el más/menos
_MK = json.loads((A.DATA / "espn_markets.json").read_text()) if (A.DATA / "espn_markets.json").exists() else {}  # train_espn.py ncaa
TOT.update({l: c["totals"] for l, c in _MK.items() if "totals" in c})
SPREAD.update({l: c["spread"] for l, c in _MK.items() if "spread" in c})
COLS = ["id", "date", "event", "home", "away", "hs", "as", "res"]


def get(path, **params):
    r = requests.get(f"{ESPN}/{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def parse(data):
    """Partidos de un scoreboard. En tenis vienen dentro de 'groupings' (solo individuales)."""
    rows = []
    for ev in data.get("events", []):
        comps = ([c for g in ev["groupings"] if "Singles" in g["grouping"]["displayName"] for c in g["competitions"]]
                 if "groupings" in ev else ev.get("competitions", []))
        for c in comps:
            cs = c.get("competitors", [])
            if len(cs) != 2: continue
            h = next((x for x in cs if x.get("homeAway") == "home"), cs[0])
            a = cs[1] if h is cs[0] else cs[0]
            name = lambda x: (x.get("team") or x.get("athlete") or {}).get("displayName", "?")
            state = c["status"]["type"]["state"]
            res = (1.0 if h.get("winner") else 0.0 if a.get("winner") else 0.5) if state == "post" else None
            rows.append({"id": c["id"], "date": c.get("date") or ev["date"], "event": ev.get("name", ""),
                         "home": name(h), "away": name(a), "hs": h.get("score"), "as": a.get("score"), "res": res,
                         "state": state, "odds": c.get("odds"),  # titular probable (portero NHL / abridor MLB)
                         "prob": [str(((x.get("probables") or [{}])[0].get("athlete") or {}).get("id") or "") for x in (h, a)]})
    return rows


def update_history(league, today):
    d = DIR / league
    d.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(today - pd.DateOffset(years=YEARS), today, freq="M")
    for i, m in enumerate(months):
        f = d / f"{m.strftime('%Y%m')}.csv"
        if f.exists() and i < len(months) - 2: continue  # meses cerrados: una vez; los 2 últimos se refrescan
        try:
            rows = [r for r in parse(get(f"{SPORTS[league][0]}/scoreboard", dates=m.strftime("%Y%m"), limit=1000))
                    if r["res"] is not None]
        except (requests.RequestException, ValueError, KeyError) as e:
            print("ESPN", league, m, e); continue
        pd.DataFrame(rows, columns=COLS).to_csv(f, index=False)
    df = pd.concat([x for f in sorted(d.glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("id").sort_values("date", ignore_index=True)


def elo(df, hfa, k, pred=None, keep=0.75, kt=250, mov=False):
    """Ratings finales; si se pasa `pred`, guarda ahí la prob. local de cada partido ANTES de jugarse (backtest).
    mov=True: el K crece con el margen y se frena en las palizas del favorito (Elo de FiveThirtyEight)."""
    R, n, last = {}, {}, {}
    margin = (pd.to_numeric(df.hs, errors="coerce") - pd.to_numeric(df["as"], errors="coerce")).abs() if mov else [None] * len(df)
    for r, mg in zip(df.itertuples(), margin):
        for t in (r.home, r.away):
            if k and t in last and (r.date - last[t]).days > 90: R[t] = 1500 + keep * (R[t] - 1500)  # nueva temporada
            last[t] = r.date
        e = win_prob(R, r.home, r.away, hfa)
        if pred is not None: pred.append(e)
        dr = (R.get(r.home, 1500) + hfa - R.get(r.away, 1500)) * (1 if r.res > 0.5 else -1)  # ventaja Elo del ganador
        mult = 1 if mg is None or pd.isna(mg) else math.log(mg + 1) * 2.2 / (dr * 0.001 + 2.2)
        for t, s, sign in ((r.home, r.res - e, 1), (r.away, r.res - e, -1)):
            kk = (k or kt / (n.get(t, 0) + 5) ** 0.4) * mult  # K de tenis: baja con la experiencia del jugador
            R[t] = R.get(t, 1500) + sign * kk * s
            n[t] = n.get(t, 0) + 1
    return R


def win_prob(R, home, away, hfa): return 1 / (1 + 10 ** ((R.get(away, 1500) - R.get(home, 1500) - hfa) / 400))


def american(s):
    s = str(s).strip().upper()
    if s == "EVEN": return 2.0
    try: v = float(s.replace("+", ""))
    except ValueError: return None
    return round(1 + (v / 100 if v > 0 else 100 / -v), 3) if v else None


def moneyline(odds):
    try:
        ml = odds[0]["moneyline"]
        return american(ml["home"]["close"]["odds"]), american(ml["away"]["close"]["odds"])
    except (TypeError, KeyError, IndexError):
        return None, None


def signals(p_home, oh, oa, every=False):
    return choices((p_home, 1 - p_home), (oh, oa), ("home", "away"), every)


def choices(ps, odds, sides, every=False, w=0.5):
    if not (odds and all(odds)): return []
    inv, out = sum(1 / o for o in odds), []
    for side, p, o in zip(sides, ps, odds):
        q = w * p + (1 - w) / o / inv  # mezcla Elo + mercado sin margen
        if q >= A.MIN_P and (every or A.EDGE <= q * o - 1 <= A.MAX_EDGE):
            out.append({"market": side, "odds": o, "book": "DraftKings", "p": round(q, 3), "implied": round(1 / o / inv, 3),
                        "edge": round(q * o - 1, 3)})
    return out


def totals(odds):
    """(línea, cuota más, cuota menos) del total de puntos de DraftKings."""
    try:
        t = odds[0]["total"]
        return float(t["over"]["close"]["line"][1:]), american(t["over"]["close"]["odds"]), american(t["under"]["close"]["odds"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None, None, None


def spreads(odds):
    """(línea del local, cuota local, cuota visitante) del hándicap de DraftKings."""
    try:
        s = odds[0]["pointSpread"]
        return float(s["home"]["close"]["line"]), american(s["home"]["close"]["odds"]), american(s["away"]["close"]["odds"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None, None, None


def points_model(df, league, today):
    """Ataque/defensa por equipo en puntos (mismo modelo con decaimiento que el fútbol) + desviación típica del total."""
    d = df.dropna(subset=["hs", "as"]).rename(columns={"home": "HomeTeam", "away": "AwayTeam", "hs": "FTHG", "as": "FTAG", "date": "Date"})
    d = d.assign(Div=league, Date=d.Date.dt.tz_localize(None), FTHG=d.FTHG.astype(float), FTAG=d.FTAG.astype(float))
    recent = d[d.Date >= today - pd.Timedelta(days=400)]
    c = TOT.get(league, {})
    return A.strengths(d, league, today, c.get("k", 5.0), c.get("half_life", 365)), float((recent.FTHG + recent.FTAG).std())


def over_prob(model, home, away, line, league=None):
    """(prob. de superar la línea, total esperado) con el modelo de puntos."""
    (mu_h, mu_a, s), sd = model
    exp = mu_h * s["att_h"].get(home, 1) * s["def_a"].get(away, 1) + mu_a * s["att_a"].get(away, 1) * s["def_h"].get(home, 1)
    return (totals_model.p_over(league, exp, line, TOT) if league in TOT else
            0.5 * (1 - math.erf((line - exp) / (sd * math.sqrt(2))))), exp  # normal(total esperado, sd histórica)


def total_signals(model, home, away, line, oo, ou, league=None):
    p_over, exp = over_prob(model, home, away, line, league)
    inv, out = 1 / oo + 1 / ou, []
    for side, p, o in (("over", p_over, oo), ("under", 1 - p_over, ou)):
        q = 0.5 * p + 0.5 / o / inv
        if q >= A.MIN_P:
            out.append({"market": side, "line": line, "odds": o, "book": "DraftKings", "p": round(q, 3),
                        "implied": round(1 / o / inv, 3), "edge": round(q * o - 1, 3), "expected": round(exp, 1)})
    return out


def injuries(league):
    """{equipo: [jugadores lesionados/dudosos]} (solo deportes de equipo)."""
    if SPORTS[league][4] is None: return {}
    try: data = get(f"{SPORTS[league][0]}/injuries")
    except (requests.RequestException, ValueError) as e:
        print("lesiones no disponibles", league, e); return {}
    out = {}
    for t in data.get("injuries", []):
        out[t.get("displayName")] = [
            {"name": i["athlete"]["displayName"], "pos": (i["athlete"].get("position") or {}).get("abbreviation", ""),
             "games": 0, "goals": "-", "note": i.get("shortComment", ""),
             "status": "lesionado" if any(w in i.get("status", "").lower() for w in ("out", "reserve", "suspen")) else "duda",
             "ret": (i.get("details") or {}).get("returnDate", "-")}
            for i in t.get("injuries", []) if i.get("athlete")]
    return out


def form(df, t, n=5):
    d = df[(df.home == t) | (df.away == t)].tail(n)
    return "".join("➖" if r.res == 0.5 else "✅" if (r.home == t) == (r.res == 1) else "❌" for r in d.itertuples())


def key(league, g): return f"{league}|{str(g['date'])[:10]}|{g['home']}|{g['away']}"


def settle(df, league, state):
    done = {key(league, r): r for r in df.tail(5000).to_dict("records")}
    for s in state["signals"]:
        r = done.get(s["key"])
        if s.get("result") is None and r is not None:
            if s["market"] in ("sph", "spa"): s["result"] = bankroll.spread_result(s["market"], s["line"], r["hs"], r["as"])
            elif s["market"] in ("over", "under"):
                t = float(r["hs"]) + float(r["as"])
                s["result"] = "nula" if t == s["line"] else (t > s["line"]) == (s["market"] == "over")
            elif league in SOCCER: s["result"] = r["res"] == {"home": 1.0, "D": 0.5, "away": 0.0}[s["market"]]
            else: s["result"] = "nula" if r["res"] == 0.5 else (r["res"] == 1) == (s["market"] == "home")
            if pd.notna(r["hs"]) and pd.notna(r["as"]): s["score"] = f"{int(float(r['hs']))}-{int(float(r['as']))}"


def injury_elo(players):
    # shortcut: cada baja resta Elo fijo sin saber si es titular; usar minutos por jugador si hay fuente gratis
    return -min(60, sum(15 if p["status"] == "lesionado" else 5 for p in players))


def line_details(odds):
    try: return odds[0].get("details", ""), odds[0].get("overUnder")
    except (TypeError, IndexError, AttributeError): return "", None


def recent(path, today, q):
    """Resultados de ayer y hoy sin guardarlos (para liquidar en las horas sin descarga completa del historial)."""
    rows = []
    for d in pd.date_range(today - pd.Timedelta(days=1), today):
        try: rows += [r for r in parse(get(f"{path}/scoreboard", dates=f"{d:%Y%m%d}", limit=300, **q)) if r["res"] is not None]
        except (requests.RequestException, ValueError, KeyError) as e: print("ESPN", path, d, e)
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def save_odds(league, games, now, prob, pts, p3=None):
    """Guarda la última cuota PREVIA de DraftKings de cada partido (moneyline, total y hándicap) con la prob. del modelo
    en ese momento: train_espn.py informe mide con ellas log-loss frente al mercado, mezcla óptima y ROI."""
    rows = []
    for g in games:
        (oh, oa), (tl, to, tu), (sl, sh, sa) = moneyline(g["odds"]), totals(g["odds"]), spreads(g["odds"])
        if not (oh or to or sh): continue
        p = prob(g)
        if p3:  # fútbol: prob. del local y del empate, y cuota del empate
            import espn_extra
            (p, pd_, _), od = p3(p), (espn_extra.moneyline3(g["odds"]) or (None,) * 3)[1]
        rows.append({"id": g["id"], "seen": now.isoformat(), "start": g["date"], "home": g["home"], "away": g["away"],
                     "p": round(p, 4), "oh": oh, "oa": oa, **({"pd": round(pd_, 4), "od": od} if p3 else {}), "tl": tl, "to": to, "tu": tu,
                     "p_over": round(over_prob(pts, g["home"], g["away"], tl, league)[0], 4) if pts and tl and to else None,
                     "sl": sl, "sh": sh, "sa": sa,
                     "p_cover": round(totals_model.p_cover(league, p, sl, {"spread": SPREAD}), 4) if league in SPREAD and sl is not None else None})
    if not rows: return
    f = DIR / f"odds_{league}.csv"
    old = pd.read_csv(f, dtype={"id": str}) if f.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("id", keep="last").to_csv(f, index=False)


def run_league(league, state, today, kg):
    path, title, sport, hfa, k = SPORTS[league]
    soccer, extra = league in SOCCER, league in _X
    if extra:
        import espn_extra  # importación tardía: espn_extra importa este módulo
        if espn_extra.EXTRA[league][4] == "D" and pd.Timestamp.now(tz="UTC").hour != 5:  # historial día a día: completo a las 5 UTC
            df = pd.concat([espn_extra.load(league), recent(path, today, espn_extra.EXTRA[league][3])])
        else: df = pd.concat([espn_extra.history(l, today) for l in (espn_extra.SOCCER if soccer else [league])])
        df = df.drop_duplicates("id").sort_values("date", ignore_index=True)
    else: df = update_history(league, today)
    print(f"{league}: {len(df)} partidos de historial")
    settle(df, league, state)
    p_ = _P.get(league, {})
    R = elo(df, hfa, k, None, p_.get("keep", 0.75), p_.get("kt", 250), p_.get("mov", False)) if k else None
    pts = points_model(df, league, today) if k and (not extra or league in TOT) else None  # totales solo con backtest
    games = {}
    for d in pd.date_range(today, periods=A.HORIZON_DAYS + 1):
        q = {"limit": 300, **espn_extra.EXTRA[league][3]} if extra else {"limit": 1000}  # college: groups y limit 300
        try: games.update({g["id"]: g for g in parse(get(f"{path}/scoreboard", dates=f"{d:%Y%m%d}", **q)) if g["state"] == "pre"})
        except (requests.RequestException, ValueError, KeyError) as e: print("ESPN", league, d, e)
    now = pd.Timestamp.now(tz="UTC")
    pre = [g for g in games.values() if pd.Timestamp(g["date"]) > now and "TBD" not in (g["home"], g["away"])]
    new = [g for g in pre if key(league, g) not in state["analyzed"]]
    if not pre: return
    inj = injuries(league)
    if not k:  # tenis (Elo por superficie) y UFC: modelos de agente-datos; los partidos de hoy predicen sin actualizar
        up = pd.DataFrame([{"date": pd.Timestamp(g["date"]), "home": g["home"], "away": g["away"], "event": g["event"],
                            "res": math.nan} for g in pre], index=[g["id"] for g in pre])
        allg = pd.concat([df, up]).sort_values("date", kind="stable")
        pred = (tennis_model.predict(league, allg) if league in ("atp", "wta") else ufc_model.elo(allg, **{
            c: v for c, v in json.loads(ufc_model.PARAMS.read_text()).items() if c in ("kt", "pw", "hfa", "idle", "keep")})[0])
        pred = dict(zip(allg.index, pred))
    prob = lambda g: pred[g["id"]] if not k else win_prob(R, g["home"], g["away"], hfa + injury_elo(inj.get(g["home"], [])) - injury_elo(inj.get(g["away"], [])))
    if league in STARTERS:  # portero titular probable (NHL) frente al habitual del equipo
        import starters
        sd, elo_p = starters.live(league, df, pre, now), prob
        prob = lambda g: float(starters.adjust(elo_p(g), sd[g["id"]], STARTERS[league]["c"], starters.CONF[league][2]))
    p3 = (lambda p: espn_extra.soccer_probs([p], _P[league])[0]) if soccer else None
    save_odds(league, pre, now, prob, pts, p3)
    for g in sorted(new, key=lambda g: g["date"]):
        home, away = g["home"], g["away"]
        squads = [inj.get(t, []) for t in (home, away)]
        p = prob(g)
        hh = df[((df.home == home) & (df.away == away)) | ((df.home == away) & (df.away == home))]
        wins = lambda t: int((((hh.home == t) & (hh.res == 1)) | ((hh.away == t) & (hh.res == 0))).sum())
        h2h = f"{len(hh)} desde {hh.date.min():%Y}: {wins(home)}-{wins(away)}" if len(hh) else "sin precedentes"
        P3 = p3(p) if soccer else None
        kg.match(title, home, away, g["date"], dict(zip("1X2", P3)) if soccer else {"1": p, "2": 1 - p}, squads, h2h)
        state["analyzed"].append(key(league, g))
        line, oo, ou = totals(g["odds"])
        if pts and line and oo and ou:
            tc = total_signals(pts, home, away, line, oo, ou, league)
            for v in [v for v in tc if A.EDGE <= v["edge"] <= A.MAX_EDGE] or sorted(tc, key=lambda v: -v["edge"])[:1]:
                word = {"nba": "puntos", "nfl": "puntos", "nhl": "goles", "mlb": "carreras"}.get(league, "puntos")
                state["signals"].append({**v, "key": key(league, g), "match": f"{home} vs {away}", "start": g["date"],
                    "pick": f"{'Más' if v['market'] == 'over' else 'Menos'} de {line} {word}", "league": title, "sport": title.split()[0],
                    "info": [f"📊 <b>Escenarios</b>: total esperado <b>{v['expected']}</b> {word} · línea {line}",
                             f"Más {v['p'] if v['market'] == 'over' else 1 - v['p']:.0%} · Menos {v['p'] if v['market'] == 'under' else 1 - v['p']:.0%}",
                             "", "🔎 <b>Claves</b>", f"• Forma: {A.esc(home)} {form(df, home)} | {A.esc(away)} {form(df, away)}",
                             f"• H2H: {h2h}", f"• Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}"],
                    "context": f"{home} vs {away} ({title}). Modelo de puntos (ataque/defensa con {YEARS} años, decaimiento 1 año): "
                               f"total esperado {v['expected']} frente a línea {line}. Forma {home} {form(df, home)}, {away} {form(df, away)}. "
                               f"H2H {h2h}. Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}."})
        sl, sh, sa = spreads(g["odds"])
        if league in SPREAD and sl is not None and sh and sa:  # hándicap: prob. de cubrir desde el Elo, mezclada con DK
            sc = [{**v, "market": "sp" + v["market"][0], "line": sl if v["market"] == "home" else -sl}
                  for v in signals(totals_model.p_cover(league, p, sl, {"spread": SPREAD}), sh, sa, every=True)]
            for v in [v for v in sc if A.EDGE <= v["edge"] <= A.MAX_EDGE] or sorted(sc, key=lambda v: -v["edge"])[:1]:
                team = home if v["market"] == "sph" else away
                state["signals"].append({**v, "key": key(league, g), "match": f"{home} vs {away}", "start": g["date"],
                    "pick": f"{team} {v['line']:+g}", "league": title, "sport": title.split()[0],
                    "info": [f"📊 <b>Escenarios</b>: {A.esc(team)} cubre {v['line']:+g} con prob. {v['p']:.0%} (casas {v['implied']:.0%})",
                             "", "🔎 <b>Claves</b>", f"• Forma: {A.esc(home)} {form(df, home)} | {A.esc(away)} {form(df, away)}",
                             f"• H2H: {h2h}", f"• Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}"],
                    "context": f"{home} vs {away} ({title}). Hándicap {team} {v['line']:+g}: margen esperado desde el Elo "
                               f"(gana {home} {p:.0%}). Forma {home} {form(df, home)}, {away} {form(df, away)}. H2H {h2h}. "
                               f"Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}."})
        o3 = espn_extra.moneyline3(g["odds"]) if soccer else None
        # fútbol: mismo peso del modelo que el 1X2 de ligas (train_football: el mercado gana; con 0 solo hay candidatas)
        cands = choices(P3, o3, ("home", "D", "away"), True, A.W["1x2"]) if soccer else signals(p, *moneyline(g["odds"]), every=True)
        if not cands: continue
        oh, oa = (o3[0], o3[2]) if soccer else moneyline(g["odds"])
        odds = o3 if soccer else (oh, oa)
        mk = [1 / o / sum(1 / x for x in odds) for o in odds]  # casas sin margen
        vb = [v for v in cands if A.EDGE <= v["edge"] <= A.MAX_EDGE] or [max(cands, key=lambda v: v["edge"])]
        details, total = line_details(g["odds"])
        fav = home if oh < oa else away
        info = [f"📊 <b>Escenarios</b> (Elo con {len(df):,} partidos y bajas)".replace(",", "."),
                f"1 {P3[0]:.0%} · X {P3[1]:.0%} · 2 {P3[2]:.0%}" if soccer else f"Gana {A.esc(home)} {p:.0%} · Gana {A.esc(away)} {1 - p:.0%}",
                f"Casas: 1 {mk[0]:.0%} · X {mk[1]:.0%} · 2 {mk[2]:.0%}" if soccer else f"Casas: {A.esc(home)} {mk[0]:.0%} · {A.esc(away)} {mk[1]:.0%}"]
        if details or total: info.append(f"Hándicap {A.esc(details)} · Total puntos {total}")
        info += ["", "🔎 <b>Claves</b>",
                 f"• Forma: {A.esc(home)} {form(df, home)} | {A.esc(away)} {form(df, away)}",
                 f"• H2H: {h2h}",
                 f"• Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}",
                 f"• Favorito de las casas: {A.esc(fav)} @ {min(oh, oa):.2f}"]
        if soccer:  # Champions / Europa: córners, tarjetas, faltas y tiros a puerta con lo que hace cada equipo en su liga
            try: info += football_stats.euro_lines(home, away, today)
            except Exception as e: print("estadísticas europeas", repr(e))
        context = (f"{home} vs {away} ({title}, {sport}). Modelo Elo con {YEARS} años de historial ajustado por bajas: "
                   f"{home} {(P3[0] if soccer else p):.0%}, {'empate ' + format(P3[1], '.0%') + ', ' if soccer else ''}{away} {(P3[2] if soccer else 1 - p):.0%}. "
                   f"Cuotas DraftKings {'/'.join(map(str, odds))}, hándicap {details}, total {total}. "
                   f"Forma {home} {form(df, home)}, {away} {form(df, away)}. H2H {h2h}. Bajas {home}: "
                   f"{'; '.join(x['name'] + ' ' + x['status'] for x in squads[0]) or 'ninguna'}. Bajas {away}: "
                   f"{'; '.join(x['name'] + ' ' + x['status'] for x in squads[1]) or 'ninguna'}.")
        for v in vb:
            state["signals"].append({**v, "key": key(league, g), "match": f"{home} vs {away}", "pick": g.get(v["market"], "Empate"),
                                     "start": g["date"], "league": f"{title} · {g['event']}" if k is None else title,
                                     "sport": title.split()[0], "info": info, "context": context})


def euro_backfill(state, today):
    """Señales de Champions/Europa analizadas antes de tener estadísticas por equipo: se les añaden (una vez)."""
    now = pd.Timestamp.now(tz="UTC")
    done = {}
    for s in state["signals"]:
        if not s.get("key", "").startswith(tuple(f"{l}|" for l in SOCCER)) or s.get("result") is not None or not s.get("info") or \
                pd.Timestamp(s["start"]) <= now or any("Estadísticas esperadas" in l for l in s["info"]): continue
        if s["key"] not in done:
            home, away = s["match"].split(" vs ", 1)
            try: done[s["key"]] = football_stats.euro_lines(home, away, today)
            except Exception as e: print("estadísticas europeas", repr(e)); done[s["key"]] = []
        s["info"] = s["info"] + done[s["key"]]


def run(state, today, kg):
    try: euro_backfill(state, today)
    except Exception as e: print("fallo en euro_backfill", repr(e))
    for league in SPORTS:
        try: run_league(league, state, today, kg)
        except Exception as e:  # un deporte caído no debe parar los demás
            print("fallo en", league, repr(e))
