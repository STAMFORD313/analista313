import requests, json
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json"}
r = requests.get("https://www.fotmob.com/api/data/matches?date=20261102", headers=UA, timeout=30).json()
mid = next(m["id"] for lg in r["leagues"] for m in lg["matches"] if m.get("status", {}).get("finished"))
print("partido", mid)
for u in (f"https://www.fotmob.com/api/data/matchDetails?matchId={mid}", f"https://www.fotmob.com/api/matchDetails?matchId={mid}"):
    x = requests.get(u, headers=UA, timeout=30)
    t = x.text
    print(u, x.status_code, len(t), "xG" in t or "expected_goals" in t, t[:150].replace("\n", " "))
