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


def find(home, away, start):
    """Evento de Kambi del partido (mismo día ±1 y los dos equipos emparejados por nombre) o None."""
    evs = [e for e in events() if abs((e["start"] - start).total_seconds()) < 86400]
    for e in evs:
        if A.find_team(home, [e["home"]]) and A.find_team(away, [e["away"]]): return e
    return None


def parse(d, home, away):
    """{(stat, 't'|'h'|'a'): {línea: {'o': cuota, 'u': cuota}}} de un betoffer/event de Kambi (solo partido completo)."""
    out = {}
    for b in d.get("betOffers", []):
        m = LABEL.match(b["criterion"].get("englishLabel", ""))
        if not m or b["criterion"].get("lifetime") != "FULL_TIME": continue
        scope = "t" if not m.group(1) else "h" if m.group(1) == home else "a" if m.group(1) == away else None
        for o in b.get("outcomes", []):
            if scope and o.get("status") == "OPEN" and o.get("type") in ("OT_OVER", "OT_UNDER") and o.get("odds") and o.get("line"):
                out.setdefault(("corners", scope), {}).setdefault(o["line"] / 1000, {})[o["type"][3].lower()] = o["odds"] / 1000
    return {k: {l: v for l, v in lines.items() if len(v) == 2} for k, lines in out.items()}


def offers(home, away, start):
    """Cuotas de córners del partido de football-data.co.uk; {} si Kambi no lo tiene."""
    e = find(home, away, start)
    if not e: return {}
    return parse(get(f"betoffer/event/{e['id']}"), e["home"], e["away"])
