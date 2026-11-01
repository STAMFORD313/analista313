"""NBA, NFL, NHL, MLB y tenis ATP/WTA vía la API pública de ESPN: 10 años de historial, Elo, lesiones,
cuotas DraftKings, señales de valor, IA, grafo, PDF y Telegram. Reutiliza las utilidades de analyst.py."""
import tempfile
from pathlib import Path

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
}
YEARS = 10
DIR = A.DATA / "espn"
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


def elo(df, hfa, k):
    R, n, last = {}, {}, {}
    for r in df.itertuples():
        for t in (r.home, r.away):
            if k and t in last and (r.date - last[t]).days > 90: R[t] = 1500 + 0.75 * (R[t] - 1500)  # nueva temporada
            last[t] = r.date
        e = win_prob(R, r.home, r.away, hfa)
        for t, s, sign in ((r.home, r.res - e, 1), (r.away, r.res - e, -1)):
            kk = k or 250 / (n.get(t, 0) + 5) ** 0.4  # K de tenis: baja con la experiencia del jugador
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


def signals(p_home, oh, oa):
    if not (oh and oa): return []
    inv, out = 1 / oh + 1 / oa, []
    for side, p, o in (("home", p_home, oh), ("away", 1 - p_home, oa)):
        q = 0.5 * p + 0.5 / o / inv  # mezcla Elo + mercado sin margen, como en fútbol
        if q >= A.MIN_P and q * o - 1 >= A.EDGE:
            out.append({"market": side, "odds": o, "p": round(q, 3), "edge": round(q * o - 1, 3)})
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


def form(df, t, n=6):
    d = df[(df.home == t) | (df.away == t)].tail(n)
    return "".join("E" if r.res == 0.5 else "G" if (r.home == t) == (r.res == 1) else "P" for r in d.itertuples())


def score(r): return f"{int(r.hs)}-{int(r['as'])}" if pd.notna(r.hs) and pd.notna(r["as"]) else ""


def key(league, g): return f"{league}|{str(g['date'])[:10]}|{g['home']}|{g['away']}"


def settle(df, league, state):
    done = {key(league, r): r for r in df.tail(3000).to_dict("records")}
    lines = []
    for s in state["signals"]:
        r = done.get(s["key"])
        if s.get("result") is None and r is not None:
            s["result"] = "nula" if r["res"] == 0.5 else (r["res"] == 1) == (s["market"] == "home")
            if s["result"] != "nula": lines.append(f"  {'✅' if s['result'] else '❌'} {s['match']} · {s['pick']} @ {s['odds']}")
    if lines: A.send_text("\n".join([f"🎯 {SPORTS[league][1]} · SEÑALES RESUELTAS"] + lines))


def run_league(league, state, today, tmp):
    path, title, sport, hfa, k = SPORTS[league]
    df = update_history(league, today)
    print(f"{league}: {len(df)} partidos de historial")
    settle(df, league, state)
    R = elo(df, hfa, k)
    games = {}
    for d in pd.date_range(today, periods=A.HORIZON_DAYS + 1):
        try: games.update({g["id"]: g for g in parse(get(f"{path}/scoreboard", dates=f"{d:%Y%m%d}", limit=1000)) if g["state"] == "pre"})
        except (requests.RequestException, ValueError, KeyError) as e: print("ESPN", league, d, e)
    now = pd.Timestamp.now(tz="UTC")
    new = [g for g in games.values() if key(league, g) not in state["analyzed"]
           and pd.Timestamp(g["date"]) > now and "TBD" not in (g["home"], g["away"])]
    if not new: return
    inj = injuries(league)
    lines, picks = [f"{title} · PRÓXIMOS PARTIDOS (Elo de {YEARS} años)"], []
    for g in sorted(new, key=lambda g: g["date"]):
        p = win_prob(R, g["home"], g["away"], hfa)
        oh, oa = moneyline(g["odds"])
        when = pd.Timestamp(g["date"]).tz_convert("Europe/Madrid")
        line = f"{when:%d/%m %H:%M} {g['home']} vs {g['away']} · {p:.0%}-{1 - p:.0%}" + (f" · cuotas {oh}/{oa}" if oh else "")
        lines.append(line)
        picks.append(line + f" · forma {form(df, g['home'])}/{form(df, g['away'])}")
        for v in signals(p, oh, oa):
            report(league, g, p, v, df, inj, tmp)
            state["signals"].append({**v, "key": key(league, g), "match": f"{g['home']} vs {g['away']}",
                                     "pick": g[v["market"]], "start": g["date"]})
        state["analyzed"].append(key(league, g))
    ai = A.ai_json(f"""Eres un analista experto de {sport}. Estos son los próximos partidos ({title}) con la probabilidad de
victoria de mi modelo Elo ({YEARS} años de historial), cuotas y forma reciente (G/E/P):
{chr(10).join(picks[:60])}
Lesionados/dudas por equipo: {'; '.join(f"{t}: {', '.join(p['name'] + ' (' + p['status'] + ')' for p in ps[:6])}" for t, ps in inj.items() if ps)[:6000] or 'sin datos'}
Responde SOLO con un JSON válido: {{"analysis": "en español: los 3-5 partidos más interesantes con tu pronóstico y por qué (máx. 12 frases)"}}""")
    if ai and ai.get("analysis"): lines += ["", "🤖 " + ai["analysis"]]
    A.send_text("\n".join(lines))


def report(league, g, p, v, df, inj, tmp):
    """Informe completo (mensaje + grafo de lesiones + PDF) para un partido con señal de valor."""
    title, sport = SPORTS[league][1], SPORTS[league][2]
    home, away, pick = g["home"], g["away"], g[v["market"]]
    hh = df[((df.home == home) & (df.away == away)) | ((df.home == away) & (df.away == home))]
    wins = lambda t: int((((hh.home == t) & (hh.res == 1)) | ((hh.away == t) & (hh.res == 0))).sum())
    stats = (f"Probabilidad Elo: {home} {p:.0%} - {1 - p:.0%} {away}\n"
             f"Forma (últimos 6): {home} {form(df, home)} | {away} {form(df, away)}\n"
             f"Cara a cara ({len(hh)} desde {hh.date.min():%Y}): {home} {wins(home)} - {wins(away)} {away}" if len(hh) else
             f"Probabilidad Elo: {home} {p:.0%} - {1 - p:.0%} {away}\nSin enfrentamientos previos")
    squads = [inj.get(t, []) for t in (home, away)]
    absences = "\n".join(f"  [{t}] {x['name']} ({x['pos']}) {x['status']}: {x['note']}" for t, s in zip((home, away), squads) for x in s)
    ai = A.ai_report(title, pd.Timestamp(g["date"]), home, away, stats, absences or "Sin datos de bajas", sport)
    msg = [f"{title} · {g['event']}", f"{home} vs {away}", "", stats]
    if absences: msg += ["", "🚑 Bajas y dudas:", absences]
    if ai and ai.get("pick"): msg += ["", f"🤖 IA: {ai['pick']} (confianza {ai.get('confidence', '?')}/5)"]
    msg += ["", f"💰 SEÑAL: gana {pick} @ {v['odds']} · prob {v['p']:.0%} · ventaja {v['edge']:+.0%}",
            "", "Pronóstico estadístico, no garantía. Juega con responsabilidad."]
    A.send_text("\n".join(msg))
    name = f"{league}_{g['id']}"
    img = None
    if any(squads):
        img = tmp / f"{name}.png"; A.player_graph(home, away, squads, img)
        A.send_file("sendPhoto", "photo", img, f"Lesiones · {home} vs {away}")
    h2h_lines = "\n".join(f"{r.date:%d/%m/%Y}  {r.home} {score(r) or ('gana' if r.res == 1 else 'pierde')} {r.away}"
                          for _, r in hh.iloc[::-1].head(25).iterrows()) or "—"
    pdf = tmp / f"{name}.pdf"
    A.pdf_report(pdf, f"{home} vs {away} · {title}", [
        ("Modelo y datos", stats), ("Señal de valor", msg[-3]),
        ("Análisis IA", (ai or {}).get("analysis", "IA no configurada")),
        ("Bajas y dudas", absences or "Sin datos"), ("Últimos enfrentamientos", h2h_lines)], img)
    A.send_file("sendDocument", "document", pdf, f"Informe {home} vs {away}")


def run(state, today):
    tmp = Path(tempfile.mkdtemp())
    for league in SPORTS:
        try: run_league(league, state, today, tmp)
        except Exception as e:  # un deporte caído no debe parar los demás
            print("fallo en", league, repr(e))
