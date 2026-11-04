"""Smarkets (bolsa de apuestas, API pública sin clave): 1X2 de fútbol de las próximas horas para comparar con las casas.

Precio de Smarkets = probabilidad en puntos básicos; para apostar A FAVOR se toma la mejor «offer»: cuota = 10000 / precio.
Comisión del 2 % sobre la ganancia → cuota efectiva = 1 + (cuota − 1) × 0.98. Solo ofertas con algo de liquidez."""
import pandas as pd
import requests

import analyst as A

API = "https://api.smarkets.com/v3"
COMMISSION, MIN_QTY = 0.02, 200_000  # cantidad en unidades de Smarkets (≈ 20 £ disponibles)


def get(path, **params):
    r = requests.get(f"{API}/{path}", params=params, headers={**A.UA, "Accept": "application/json"}, timeout=30)
    r.raise_for_status(); return r.json()


def events(hours=48, max_pages=8):
    now = pd.Timestamp.now(tz="UTC"); out, params = [], {"state": "upcoming", "type": "football_match", "limit": 500, "sort": "start_datetime,id"}
    for _ in range(max_pages):
        d = get("events/", **params)
        for e in d.get("events", []):
            t = pd.Timestamp(e["start_datetime"])
            if now < t < now + pd.Timedelta(hours=hours) and " vs " in e["name"]: out.append((e["id"], t, *e["name"].split(" vs ", 1)))
        nxt = (d.get("pagination") or {}).get("next_page")
        if not nxt or (d.get("events") and pd.Timestamp(d["events"][-1]["start_datetime"]) > now + pd.Timedelta(hours=hours)): break
        params = dict(p.split("=", 1) for p in nxt.lstrip("?").split("&"))
    return out


def chunks(x, n=40): return [x[i:i + n] for i in range(0, len(x), n)]


def football_1x2(hours=48):
    """[{home, away, start, odds{H,D,A}, book: "Smarkets"}] con la cuota efectiva tras comisión."""
    ev = {e[0]: e for e in events(hours)}
    mk = {}
    for ids in chunks(list(ev)):
        for m in get(f"events/{','.join(ids)}/markets/").get("markets", []):
            if (m.get("market_type") or {}).get("name") == "WINNER_3_WAY": mk[m["id"]] = m["event_id"]
    out = []
    for ids in chunks(list(mk)):
        cts = get(f"markets/{','.join(ids)}/contracts/").get("contracts", [])
        quotes = get(f"markets/{','.join(ids)}/quotes/")
        by = {}
        for c in cts:
            side = {"HOME": "H", "DRAW": "D", "AWAY": "A"}.get((c.get("contract_type") or {}).get("name"))
            offers = [o for o in (quotes.get(c["id"]) or {}).get("offers", []) if o.get("quantity", 0) >= MIN_QTY and o.get("price")]
            if side and offers:
                o = 10000 / min(x["price"] for x in offers)
                by.setdefault(c["market_id"], {})[side] = round(1 + (o - 1) * (1 - COMMISSION), 3)
        for mid, odds in by.items():
            if len(odds) == 3:
                _, t, h, a = ev[mk[mid]]
                out.append({"home": h, "away": a, "start": t, "odds": odds, "book": "Smarkets"})
    return out
