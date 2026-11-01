"""Grafo de conocimiento persistente (data/grafo.json, formato node-link de networkx):
ligas → equipos → jugadores (estado, posición, partidos), partidos (probabilidades), H2H entre equipos y apuestas del bot."""
import json

import networkx as nx
import pandas as pd

import analyst as A

PATH = A.DATA / "grafo.json"
KEEP_DAYS = 30


class Graph:
    def __init__(self):
        self.g = nx.node_link_graph(json.loads(PATH.read_text(encoding="utf-8")), edges="links") if PATH.exists() else nx.Graph()

    def match(self, league, home, away, start, probs, squads, h2h):
        g = self.g
        g.add_node(league, type="liga")
        for team, players in zip((home, away), squads):
            g.add_node(team, type="equipo")
            g.add_edge(team, league, rel="juega_en")
            for old in [n for n in g.neighbors(team) if g.nodes[n].get("type") == "jugador"]: g.remove_node(old)  # bajas al día
            for p in players:
                pid = f"{p['name']} ({team})"
                g.add_node(pid, type="jugador", **{k: p.get(k) for k in ("pos", "status", "games", "note", "ret")})
                g.add_edge(pid, team, rel=p.get("status", "baja"))
        mid = f"{home} vs {away} {pd.Timestamp(start):%Y-%m-%d}"
        g.add_node(mid, type="partido", start=pd.Timestamp(start).isoformat(), **{f"p_{k}": round(v, 3) for k, v in probs.items()})
        g.add_edge(mid, home, rel="local"); g.add_edge(mid, away, rel="visitante")
        g.add_edge(home, away, rel="h2h", resumen=h2h)

    def bet(self, b, signals):
        bid = f"apuesta {b['date']} {b['desc']}"
        self.g.add_node(bid, type="apuesta", stake=b["stake"], odds=b["odds"], status=b["status"])
        for k, m in b["legs"]:
            s = next((s for s in signals if s["key"] == k and s["market"] == m), None)
            if s:
                mid = f"{s['match']} {pd.Timestamp(s['start']):%Y-%m-%d}"
                if mid in self.g: self.g.add_edge(bid, mid, rel="apuesta_a", pick=s["pick"])

    def save(self):
        cut = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)).isoformat()
        self.g.remove_nodes_from([n for n, d in self.g.nodes(data=True) if d.get("type") == "partido" and d["start"] < cut])
        self.g.remove_nodes_from([n for n, d in self.g.nodes(data=True) if d.get("type") == "apuesta" and not self.g.degree(n)])
        PATH.write_text(json.dumps(nx.node_link_data(self.g, edges="links"), ensure_ascii=False), encoding="utf-8")
