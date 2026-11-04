"""Cuotas de córners de fútbol (total y por equipo) y del 1X2 de esoccer (FC 27) de la API pública de Kambi (la que usa la web de Unibet), sin sesión.
La lee football_stats.value_signals. Los tiros a puerta no: Kambi los liquida con Opta, que cuenta más que football-data
(Championship 25/26: P(>7.5) histórica 59 % frente a ~67 % de la casa), y la ventaja sería falsa."""
import re

import pandas as pd
import requests

import analyst as A

URL = "https://eu-offering-api.kambicdn.com/offering/v2018/ub"
Q = {"lang": "en_GB", "market": "GB"}
LABEL = re.compile(r"^Total Corners(?: by (.+?))?$", re.I)
# otras estadísticas que se REGISTRAN (no se apuestan) para medir el modelo frente a Unibet: Kambi puede contarlas distinto
STAT_LABELS = {"cards": re.compile(r"^Total (?:Cards|Bookings)(?:(?: by| -) (.+?))?$", re.I),  # «Total Cards - Derby County»
               "sot": re.compile(r"^Total Shots on Target(?:(?: by| -) (.+?))?$", re.I),
               "fouls": re.compile(r"^Total Fouls(?:(?: by| -) (.+?))?$", re.I)}
OPTA = re.compile(r"\s*\(Settled using Opta data\)\s*$", re.I)  # sufijo de los mercados que Kambi liquida con Opta
LABELS_SEEN = {}  # nombre de mercado → veces visto en este run (data/kambi_labels.json: para descubrir mercados)
_events = None
_raw = None
NICK = re.compile(r"\(([^()]+)\)")  # «Arsenal (Kray) Esports» → Kray


def get(path):
    try:
        r = requests.get(f"{URL}/{path}.json", params=Q, headers=A.UA, timeout=30)
        return r.json() if r.ok else {}
    except (requests.RequestException, ValueError) as e:
        print("Kambi", path, e); return {}


def events():
    """Partidos de fútbol (sin los de esports) de los próximos ~2 días: [{id, home, away, start}]. Una descarga por ejecución."""
    global _events
    if _events is None:
        _events = [{"id": e["id"], "home": e.get("homeName", ""), "away": e.get("awayName", ""), "start": pd.Timestamp(e["start"])}
                   for x in raw_football() if (e := x["event"]) and "esports" not in str(e.get("path")).lower()]
    return _events


def raw_football():
    """Lista de Kambi de fútbol (incluye el fútbol de esports): una descarga por ejecución."""
    global _raw
    if _raw is None: _raw = get("listView/football/all/all/all/matches").get("events", [])
    return _raw


def nick(name):
    """Nickname del jugador de esoccer en un nombre de Kambi («Arsenal (Kray) Esports» → «kray»), o None."""
    m = NICK.search(name or "")
    return m.group(1).strip().lower() if m else None


def esoccer_events():
    """Partidos de fútbol de esports (EA FC / Esoccer) con cuota 1X2 de Unibet: [{id, p1, p2, start, liga, odds{1,X,2}}],
    p1/p2 = nickname en minúsculas. Busca en la lista de fútbol (path con «esports») y en la de esports."""
    out, seen = [], set()
    for x in raw_football() + get("listView/esports/all/all/all/matches").get("events", []):
        e = x.get("event") or {}
        path = " ".join(str(t.get("englishName", t.get("name", ""))) for t in e.get("path", [])).lower()
        n1, n2 = nick(e.get("homeName")), nick(e.get("awayName"))
        if e.get("id") in seen or not (n1 and n2) or not any(w in path for w in ("esport", "esoccer", "fifa", "fc 2", "volta")): continue
        odds = {}
        for b in x.get("betOffers", []):
            for o in b.get("outcomes", []):
                k = {"OT_ONE": "1", "OT_CROSS": "X", "OT_TWO": "2"}.get(o.get("type"))
                if k and o.get("odds") and o.get("status", "OPEN") == "OPEN": odds[k] = o["odds"] / 1000
        if len(odds) == 3:
            seen.add(e["id"])
            out.append({"id": e["id"], "p1": n1, "p2": n2, "start": pd.Timestamp(e["start"]), "liga": path, "odds": odds})
    return out


def find_esoccer(p1, p2, start, evs=None):
    """Evento de Unibet de un partido de Esoccer Battle (mismos nicknames, ±20 min) con las cuotas en el orden p1/p2, o None."""
    a, b = p1.lower(), p2.lower()
    for e in (esoccer_events() if evs is None else evs):
        if abs((e["start"] - start).total_seconds()) > 1200: continue
        if (e["p1"], e["p2"]) == (a, b): return e
        if (e["p1"], e["p2"]) == (b, a): return {**e, "odds": {"1": e["odds"]["2"], "X": e["odds"]["X"], "2": e["odds"]["1"]}}
    return None


YOUTH = re.compile(r"\b(U\d{2}|II|B|Women|W|Reserves|Youth)\b", re.I)  # «Real Madrid U19» no es el Real Madrid
ALIAS = {"Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves", "Brighton & Hove Albion": "Brighton",
         "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Newcastle United": "Newcastle",
         "Nottingham Forest": "Nottm Forest", "Internazionale": "Inter", "Paris Saint-Germain": "PSG"}


def find(home, away, start):
    """Evento de Kambi del partido (±1 día, el de hora más cercana) con los dos equipos emparejados por nombre, sin
    confundirlo con su equipo juvenil/filial/femenino (la Youth League juega el mismo día que la Champions) o None."""
    young = bool(YOUTH.search(f"{home} {away}"))
    evs = sorted((e for e in events() if abs((e["start"] - start).total_seconds()) < 86400
                  and bool(YOUTH.search(f"{e['home']} {e['away']}")) == young),
                 key=lambda e: abs((e["start"] - start).total_seconds()))
    for e in evs:
        if all(A.find_team(n, [k]) or A.find_team(ALIAS.get(n, n), [k]) for n, k in ((home, e["home"]), (away, e["away"]))): return e
    return None


def parse(d, home, away, stats=("corners",)):
    """{(stat, 't'|'h'|'a'): {línea: {'o': cuota, 'u': cuota}}} de un betoffer/event de Kambi (solo partido completo)."""
    out = {}
    labels = {"corners": LABEL, **{k: v for k, v in STAT_LABELS.items() if k in stats}}
    for b in d.get("betOffers", []):
        lab = b["criterion"].get("englishLabel", "")
        LABELS_SEEN[lab] = LABELS_SEEN.get(lab, 0) + 1
        lab = OPTA.sub("", lab)
        m = next(((st, m) for st, rx in labels.items() if st in stats and (m := rx.match(lab))), None)
        if not m or b["criterion"].get("lifetime") != "FULL_TIME": continue
        stat, m = m
        scope = "t" if not m.group(1) else "h" if m.group(1) == home else "a" if m.group(1) == away else None
        for o in b.get("outcomes", []):
            if scope and o.get("status") == "OPEN" and o.get("type") in ("OT_OVER", "OT_UNDER") and o.get("odds") and o.get("line"):
                out.setdefault((stat, scope), {}).setdefault(o["line"] / 1000, {})[o["type"][3].lower()] = o["odds"] / 1000
    return {k: {l: v for l, v in lines.items() if len(v) == 2} for k, lines in out.items()}


def offers(home, away, start, stats=("corners",)):
    """Cuotas de córners (y de `stats`) del partido de football-data.co.uk; {} si Kambi no lo tiene. Una descarga por partido."""
    e = find(home, away, start)
    if not e: return {}
    ck = ("offer", e["id"])
    if ck not in _offer_cache: _offer_cache[ck] = get(f"betoffer/event/{e['id']}")
    return parse(_offer_cache[ck], e["home"], e["away"], stats)


_offer_cache = {}
PLAYER_LABELS = re.compile(r"^(To Score|To Get a Card|To give an assist|To score or give an assist|Player's shots on target|"
                           r"To score at least \d goals|First Goal Scorer)", re.I)


_nba = None


def _debug(k, v):
    """Diagnóstico en data/kambi_debug.json (se sube con los datos): para ver desde fuera qué devuelve Kambi."""
    import json
    f = A.DATA / "kambi_debug.json"
    try: d = json.loads(f.read_text()) if f.exists() else {}
    except ValueError: d = {}
    d[k] = {**v, "hora": str(pd.Timestamp.now(tz="UTC"))[:16]}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
_esp = None
ESPORT_WORDS = {"cs2": ("counter", "cs2", "cs:go", "csgo"), "dota2": ("dota",), "lol": ("league of legends", "lol"),
                "r6": ("rainbow", "r6"), "mlbb": ("mobile legends", "mlbb")}


def esports_events():
    """Partidos de esports de Unibet con su cuota de ganador: [{disc, home, away, start, o1, o2}] (disciplina por el path)."""
    global _esp
    if _esp is None:
        _esp = []
        for x in get("listView/esports/all/all/all/matches").get("events", []):
            e = x.get("event") or {}
            path = " ".join(str(t.get("englishName", t.get("name", ""))) for t in e.get("path", [])).lower()
            disc = next((d for d, ws in ESPORT_WORDS.items() if any(w in path for w in ws)), None)
            o = {o.get("type"): o["odds"] / 1000 for b in x.get("betOffers", []) for o in b.get("outcomes", []) if o.get("odds")}
            if disc and o.get("OT_ONE") and o.get("OT_TWO"):
                _esp.append({"disc": disc, "home": e.get("homeName", ""), "away": e.get("awayName", ""), "start": pd.Timestamp(e["start"]),
                             "o1": o["OT_ONE"], "o2": o["OT_TWO"]})
    return _esp


def find_esport(disc, t1, t2, start):
    """Cuotas de Unibet (o1 para t1, o2 para t2) de un partido de bo3.gg, emparejando nombres y hora (±3 h), o None."""
    for e in esports_events():
        if e["disc"] != disc or abs((e["start"] - pd.Timestamp(start)).total_seconds()) > 3 * 3600: continue
        if A.find_team(t1, [e["home"]]) and A.find_team(t2, [e["away"]]): return e["o1"], e["o2"]
        if A.find_team(t1, [e["away"]]) and A.find_team(t2, [e["home"]]): return e["o2"], e["o1"]
    return None


def nba_props(home, away, start):
    """Props de jugador NBA de Unibet (puntos, rebotes, asistencias, triples… según los publique): todas las salidas con
    jugador (`participant`) del partido, [{mercado, jugador, tipo, línea, cuota}]. También anota los nombres de mercado."""
    global _nba
    if _nba is None:
        _nba = [{"id": e["id"], "home": e.get("homeName", ""), "away": e.get("awayName", ""), "start": pd.Timestamp(e["start"])}
                for x in get("listView/basketball/nba/all/all/matches").get("events", []) if (e := x.get("event"))]
    ev = next((e for e in _nba if abs((e["start"] - pd.Timestamp(start)).total_seconds()) < 86400
               and A.find_team(home, [e["home"]]) and A.find_team(away, [e["away"]])), None)
    if not ev:
        print(f"Unibet NBA: {len(_nba)} partidos, sin emparejar {home} vs {away}", [f"{e['home']} vs {e['away']}" for e in _nba[:3]])
        _debug("nba", {"kambi": len(_nba), "ejemplo_kambi": [f"{e['home']} vs {e['away']} {e['start']}" for e in _nba[:3]],
                       "buscado": f"{home} vs {away} {start}"})
        return []
    out = []
    for b in get(f"betoffer/event/{ev['id']}").get("betOffers", []):
        lab = OPTA.sub("", b["criterion"].get("englishLabel", ""))
        LABELS_SEEN["NBA · " + lab] = LABELS_SEEN.get("NBA · " + lab, 0) + 1
        for o in b.get("outcomes", []):
            if o.get("participant") and o.get("odds") and o.get("status", "OPEN") == "OPEN":
                out.append({"mercado": lab, "jugador": o["participant"], "tipo": o.get("type"),
                            "linea": o["line"] / 1000 if o.get("line") else None, "cuota": o["odds"] / 1000})
    return out


def player_props(home, away, start):
    """Cuotas de props de jugador de fútbol (marcar, asistir, tarjeta, tiros a puerta del jugador…) del partido ya
    descargado por offers(): [{mercado, jugador, tipo, línea, cuota}]. Solo registro (aún no hay estadísticas por
    jugador para liquidarlas ni medir el modelo)."""
    e = find(home, away, start)
    d = _offer_cache.get(("offer", e["id"])) if e else None
    out = []
    for b in (d or {}).get("betOffers", []):
        lab = b["criterion"].get("englishLabel", "")
        if not PLAYER_LABELS.match(lab): continue
        for o in b.get("outcomes", []):
            if o.get("status", "OPEN") != "OPEN" or not o.get("odds"): continue
            out.append({"mercado": re.sub(r" \(Settled using Opta data\)", "", lab), "jugador": o.get("participant") or o.get("englishLabel") or o.get("label"),
                        "tipo": o.get("type"), "linea": o.get("line", 0) / 1000 if o.get("line") else None, "cuota": o["odds"] / 1000})
    return out


def save_labels(path):
    """Acumula en `path` (JSON) cuántas veces se ha visto cada mercado de Kambi."""
    import json
    old = json.loads(path.read_text()) if path.exists() else {}
    for k, v in LABELS_SEEN.items(): old[k] = old.get(k, 0) + v
    path.write_text(json.dumps(dict(sorted(old.items(), key=lambda x: -x[1])), ensure_ascii=False, indent=0))
