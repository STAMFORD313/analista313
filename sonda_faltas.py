import json, espn_soccer as ES
g = json.loads(open(ES.path("SP1")).readlines()[-1])
d = ES.get("esp.1/summary", event=g["id"])
names = set()
for ro in d.get("rosters") or []:
    for p in ro.get("roster") or []:
        for s in p.get("stats") or []: names.add((s.get("name"), s.get("abbreviation")))
print("STATS JUGADOR:", sorted(names))
