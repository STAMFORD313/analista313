"""Test sin red de cs2_players.py: datos sintéticos en un directorio temporal."""
import json, tempfile
from pathlib import Path

import pandas as pd

import cs2_players as C

tmp = Path(tempfile.mkdtemp())
for n in ("GAMES", "PMATCH", "PMAPS", "NOSTATS", "TRANSF", "GRAPH", "PARAMS"):
    setattr(C, n, tmp / getattr(C, n).name)

# equipo 1 (jugadores 1-5, buenos) gana siempre a equipo 2 (jugadores 6-10) en de_nuke y pierde en de_inferno
games, pmaps, pmatch = [], [], []
for i in range(15):
    mp, w = ("de_nuke", 1) if i % 3 else ("de_inferno", 2)
    gid, mid = 100 + i, 10 + i
    games.append({"game_id": gid, "match_id": mid, "slug": f"m{i}", "date": f"2025-01-{i + 1:02d}T10:00:00+00:00", "tier": "a", "bo": 1,
                  "tournament_id": 7, "tournament": "Copa", "t1_id": 1, "t2_id": 2, "team1": "Uno", "team2": "Dos",
                  "number": 1, "map": mp, "winner_id": w, "ws": 13, "ls": 5})
    for p in range(1, 11):
        t = 1 if p <= 5 else 2
        pmaps.append({"game_id": gid, "match_id": mid, "player_id": p, "team_id": t, "map": mp, "rating": 7.0 if t == 1 else 5.0})
        pmatch.append({"match_id": mid, "player_id": p, "nick": f"p{p}", "slug": f"p{p}", "team_id": t, "kills": 20, "deaths": 15,
                       "assists": 3, "adr": 80, "kast": 0.7, "rating": 7.0 if t == 1 else 5.0, "fk": 2, "fd": 1, "clutches": 0})
pd.DataFrame(games).to_csv(C.GAMES, index=False)
pd.DataFrame(pmaps).to_csv(C.PMAPS, index=False)
pd.DataFrame(pmatch).to_csv(C.PMATCH, index=False)
pd.DataFrame([{"player_id": 1, "date": "2024-12-01", "from_id": 3, "from_team": "Tres", "to_id": 1, "to_team": "Uno", "action": 1,
               "fetched": "2025-01-01"}]).to_csv(C.TRANSF, index=False)

f = C.team_features(1, 2, "2025-02-01")
assert f["fuerza"] > 0 and f["h2h"] > 0 and f["quimica"] == 0, f   # misma química en los dos
assert C.team_features(2, 1, "2025-02-01")["fuerza"] < 0
st = C.Estado(pd.read_csv(C.PMAPS))
assert st.features(1, 2, pd.Timestamp("2025-02-01", tz="UTC"))["fuerza"] == 0  # sin partidos jugados aún: neutro

G = C.build_graph()
assert G.nodes["jugador:1"]["equipo_actual"] == "Uno" and G.nodes["jugador:1"]["mapas_fuertes"]
assert G.nodes["equipo:1"]["plantilla"] == ["p1", "p2", "p3", "p4", "p5"]
assert G.nodes["equipo:1"]["mapa_fuerte"][0] == "nuke" and G.nodes["equipo:1"]["mapa_debil"][0] == "inferno"
assert G.has_edge("equipo:1", "equipo:2", key="gana_a") and G.has_edge("equipo:2", "equipo:1", key="pierde_con")
assert G.edges["jugador:1", "jugador:2", "compañero_de"]["mapas"] == 15
assert G.has_edge("jugador:1", "equipo:1", key="fichado_por 2024-12-01")
assert json.loads(C.GRAPH.read_text(encoding="utf-8"))["links"]
assert C.ajustar(0.6, 1, 2) == 0.6  # sin coeficientes entrenados no toca la prob.
print("ok cs2_players")
