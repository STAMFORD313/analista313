"""Sonda diaria para ampliar deportes (corre en GitHub Actions, donde la red funciona): qué disciplinas tiene bo3.gg
(id → nombre y partidos próximos) y qué esports/ligas con cuota publica Unibet (Kambi listView/esports y fútbol de
esports) → data/sondeo.json. Lo revisa el agente para añadir disciplinas con apuestas."""
import collections, json

import pandas as pd
import requests

import analyst as A
import kambi

OUT = A.DATA / "sondeo.json"


def bo3():
    out = {}
    for d in range(1, 16):
        try:
            r = requests.get("https://api.bo3.gg/api/v1/matches", params={"page[limit]": 20, "filter[matches.discipline_id][eq]": d,
                             "filter[matches.status][in]": "upcoming", "with": "tournament"}, headers=A.UA, timeout=30).json()
            res = r.get("results") or []
            out[d] = {"próximos": len(res), "ejemplo": [(m.get("slug"), (m.get("tournament") or {}).get("name")) for m in res[:3]]}
        except Exception as e: out[d] = {"error": repr(e)[:120]}
    return out


def kambi_esports():
    c = collections.Counter()
    for x in kambi.get("listView/esports/all/all/all/matches").get("events", []):
        e = x.get("event") or {}
        c[" / ".join(str(t.get("englishName", t.get("name", ""))) for t in e.get("path", []))] += 1
    return dict(c.most_common(60))


if __name__ == "__main__":
    OUT.write_text(json.dumps({"fecha": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}", "bo3": bo3(), "kambi_esports": kambi_esports()},
                              ensure_ascii=False, indent=1))
    print(OUT.read_text()[:3000])
