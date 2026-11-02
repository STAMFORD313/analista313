"""Test sin red de fc_players.py: datos sintéticos en un directorio temporal."""
import tempfile
from pathlib import Path

import pandas as pd

import fc_players as F

tmp = Path(tempfile.mkdtemp())
F.HIST, F.TEAMS, F.GRAPH = tmp / "matches.csv", tmp / "players_teams.csv", tmp / "grafo_fc.json.gz"
rows, teams = [], []
for i in range(6):  # Ana (con Spain) gana siempre 4-1 a Bea (con Italy)
    rows.append({"id": i, "date": f"2026-09-18T0{i}:00:00Z", "tournament_id": 1, "league": "Volta", "p1": "Ana", "p2": "Bea", "s1": 4, "s2": 1})
    teams.append({"id": i, "tournament_id": 1, "club1": "Spain", "club2": "Italy", "ht1": 2, "ht2": 0})
pd.DataFrame(rows).to_csv(F.HIST, index=False)
pd.DataFrame(teams).to_csv(F.TEAMS, index=False)
G = F.build_graph()
a = G.nodes["jugador_fc:Ana"]
assert a["v"] == 6 and a["gf_pp"] == 4 and a["gc_pp"] == 1 and a["mejor_club"] == "Spain" and a["forma"] == "VVVVVV", a
assert G.nodes["jugador_fc:Bea"]["rivales_desfavorables"] == ["Ana"]
assert G.has_edge("jugador_fc:Ana", "jugador_fc:Bea", key="gana_a") and G.has_edge("jugador_fc:Ana", "club:Spain", key="usa_club")
assert F.load_graph().number_of_edges() == G.number_of_edges()
print("ok fc_players")
