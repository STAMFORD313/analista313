"""Cuotas de córners de fútbol (total y por equipo) de la API pública de Kambi (la que usa la web de Unibet), sin sesión.
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
                   for x in get("listView/football/all/all/all/matches").get("events", [])
                   if (e := x["event"]) and "esports" not in str(e.get("path"))]
    return _events


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
