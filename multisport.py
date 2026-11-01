"""NBA, NFL, NHL, MLB y tenis ATP/WTA vía la API pública de ESPN: 10 años de historial, Elo, lesiones,
cuotas DraftKings, señales de valor, IA, grafo, PDF y Telegram. Reutiliza las utilidades de analyst.py."""
import json, math

import pandas as pd
import requests

import analyst as A

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
                         "state": state, "odds": c.get("odds")})
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
    if not (oh and oa): return []
    inv, out = 1 / oh + 1 / oa, []
    for side, p, o in (("home", p_home, oh), ("away", 1 - p_home, oa)):
        q = 0.5 * p + 0.5 / o / inv  # mezcla Elo + mercado sin margen, como en fútbol
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


def points_model(df, league, today):
    """Ataque/defensa por equipo en puntos (mismo modelo con decaimiento que el fútbol) + desviación típica del total."""
    d = df.dropna(subset=["hs", "as"]).rename(columns={"home": "HomeTeam", "away": "AwayTeam", "hs": "FTHG", "as": "FTAG", "date": "Date"})
    d = d.assign(Div=league, Date=d.Date.dt.tz_localize(None), FTHG=d.FTHG.astype(float), FTAG=d.FTAG.astype(float))
    recent = d[d.Date >= today - pd.Timedelta(days=400)]
    return A.strengths(d, league, today, 5.0, 365), float((recent.FTHG + recent.FTAG).std())


def total_signals(model, home, away, line, oo, ou):
    (mu_h, mu_a, s), sd = model
    exp = mu_h * s["att_h"].get(home, 1) * s["def_a"].get(away, 1) + mu_a * s["att_a"].get(away, 1) * s["def_h"].get(home, 1)
    p_over = 0.5 * (1 - math.erf((line - exp) / (sd * math.sqrt(2))))  # normal(total esperado, sd histórica)
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
            if s["market"] in ("over", "under"):
                t = float(r["hs"]) + float(r["as"])
                s["result"] = "nula" if t == s["line"] else (t > s["line"]) == (s["market"] == "over")
            else: s["result"] = "nula" if r["res"] == 0.5 else (r["res"] == 1) == (s["market"] == "home")
            if pd.notna(r["hs"]) and pd.notna(r["as"]): s["score"] = f"{int(float(r['hs']))}-{int(float(r['as']))}"


def injury_elo(players):
    # shortcut: cada baja resta Elo fijo sin saber si es titular; usar minutos por jugador si hay fuente gratis
    return -min(60, sum(15 if p["status"] == "lesionado" else 5 for p in players))


def line_details(odds):
    try: return odds[0].get("details", ""), odds[0].get("overUnder")
    except (TypeError, IndexError, AttributeError): return "", None


def run_league(league, state, today, kg):
    path, title, sport, hfa, k = SPORTS[league]
    df = update_history(league, today)
    print(f"{league}: {len(df)} partidos de historial")
    settle(df, league, state)
    p_ = _P.get(league, {})
    R = elo(df, hfa, k, None, p_.get("keep", 0.75), p_.get("kt", 250), p_.get("mov", False))
    pts = points_model(df, league, today) if k else None  # totales solo en deportes de equipo
    games = {}
    for d in pd.date_range(today, periods=A.HORIZON_DAYS + 1):
        try: games.update({g["id"]: g for g in parse(get(f"{path}/scoreboard", dates=f"{d:%Y%m%d}", limit=1000)) if g["state"] == "pre"})
        except (requests.RequestException, ValueError, KeyError) as e: print("ESPN", league, d, e)
    now = pd.Timestamp.now(tz="UTC")
    new = [g for g in games.values() if key(league, g) not in state["analyzed"]
           and pd.Timestamp(g["date"]) > now and "TBD" not in (g["home"], g["away"])]
    if not new: return
    inj = injuries(league)
    for g in sorted(new, key=lambda g: g["date"]):
        home, away = g["home"], g["away"]
        squads = [inj.get(t, []) for t in (home, away)]
        p = win_prob(R, home, away, hfa + injury_elo(squads[0]) - injury_elo(squads[1]))
        hh = df[((df.home == home) & (df.away == away)) | ((df.home == away) & (df.away == home))]
        wins = lambda t: int((((hh.home == t) & (hh.res == 1)) | ((hh.away == t) & (hh.res == 0))).sum())
        h2h = f"{len(hh)} desde {hh.date.min():%Y}: {wins(home)}-{wins(away)}" if len(hh) else "sin precedentes"
        kg.match(title, home, away, g["date"], {"1": p, "2": 1 - p}, squads, h2h)
        state["analyzed"].append(key(league, g))
        line, oo, ou = totals(g["odds"])
        if pts and line and oo and ou:
            tc = total_signals(pts, home, away, line, oo, ou)
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
        oh, oa = moneyline(g["odds"])
        cands = signals(p, oh, oa, every=True)
        if not cands: continue
        vb = [v for v in cands if A.EDGE <= v["edge"] <= A.MAX_EDGE] or [max(cands, key=lambda v: v["edge"])]
        details, total = line_details(g["odds"])
        fav = home if oh < oa else away
        info = [f"📊 <b>Escenarios</b> (Elo con {len(df):,} partidos y bajas)".replace(",", "."),
                f"Gana {A.esc(home)} {p:.0%} · Gana {A.esc(away)} {1 - p:.0%}",
                f"Casas: {A.esc(home)} {vb[0]['implied'] if vb[0]['market'] == 'home' else 1 - vb[0]['implied']:.0%} · "
                f"{A.esc(away)} {vb[0]['implied'] if vb[0]['market'] == 'away' else 1 - vb[0]['implied']:.0%}"]
        if details or total: info.append(f"Hándicap {A.esc(details)} · Total puntos {total}")
        info += ["", "🔎 <b>Claves</b>",
                 f"• Forma: {A.esc(home)} {form(df, home)} | {A.esc(away)} {form(df, away)}",
                 f"• H2H: {h2h}",
                 f"• Bajas: {A.absences_line(home, squads[0])}; {A.absences_line(away, squads[1])}",
                 f"• Favorito de las casas: {A.esc(fav)} @ {min(oh, oa):.2f}"]
        context = (f"{home} vs {away} ({title}, {sport}). Modelo Elo con {YEARS} años de historial ajustado por bajas: "
                   f"{home} {p:.0%}, {away} {1 - p:.0%}. Cuotas DraftKings {oh}/{oa}, hándicap {details}, total {total}. "
                   f"Forma {home} {form(df, home)}, {away} {form(df, away)}. H2H {h2h}. Bajas {home}: "
                   f"{'; '.join(x['name'] + ' ' + x['status'] for x in squads[0]) or 'ninguna'}. Bajas {away}: "
                   f"{'; '.join(x['name'] + ' ' + x['status'] for x in squads[1]) or 'ninguna'}.")
        for v in vb:
            state["signals"].append({**v, "key": key(league, g), "match": f"{home} vs {away}", "pick": g[v["market"]],
                                     "start": g["date"], "league": f"{title} · {g['event']}" if k is None else title,
                                     "sport": title.split()[0], "info": info, "context": context})


def run(state, today, kg):
    for league in SPORTS:
        try: run_league(league, state, today, kg)
        except Exception as e:  # un deporte caído no debe parar los demás
            print("fallo en", league, repr(e))
