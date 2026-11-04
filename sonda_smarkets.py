import requests, json
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
B = "https://api.smarkets.com/v3"
ev = requests.get(f"{B}/events/", params={"state": "upcoming", "type": "football_match", "limit": 3}, headers=UA, timeout=20).json()
e = ev["events"][0]; print("EVENTO", {k: e.get(k) for k in ("id", "name", "start_datetime", "type", "parent_id")}, "pag", ev.get("pagination"))
mk = requests.get(f"{B}/events/{e['id']}/markets/", headers=UA, timeout=20).json()
print("MERCADOS", [(m["id"], m["name"], m.get("market_type")) for m in mk["markets"]][:6])
m = next(m for m in mk["markets"] if "result" in m["name"].lower() or m.get("market_type", {}).get("name") == "WINNER_3_WAY")
ct = requests.get(f"{B}/markets/{m['id']}/contracts/", headers=UA, timeout=20).json()
print("CONTRATOS", [(c["id"], c["name"], c.get("contract_type")) for c in ct["contracts"]])
q = requests.get(f"{B}/markets/{m['id']}/quotes/", headers=UA, timeout=20).json()
print("QUOTES", json.dumps(q)[:600])
