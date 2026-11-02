"""Une los grafos del bot (CS2, NBA, NHL y el de partidos/apuestas) en un extracto para graphify y dibuja el HTML.
Uso: python grafo_visual.py  → grafo_visual/graphify-out/graph.html"""
import gzip, json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "grafo_visual"
MIN_SERIES, MAX_DAYS = 25, 365  # CS2: solo equipos activos con historial (si no, >5.000 nodos y graphify agrega)


def node(id_, label, src, **attrs): return {"id": id_, "label": label, "file_type": "document", "source_file": src, **attrs}
def edge(s, t, rel, src): return {"source": s, "target": t, "relation": rel, "confidence": "EXTRACTED", "source_file": src}


def cs2(nodes, edges):
    g = json.load(gzip.open(DATA / "cs2" / "grafo_cs2.json.gz", "rt", encoding="utf-8"))
    cut = (pd.Timestamp.now() - pd.Timedelta(days=MAX_DAYS)).strftime("%Y-%m-%d")
    teams = {n["id"] for n in g["nodes"] if n["type"] == "equipo" and n.get("series", 0) >= MIN_SERIES and n.get("ultimo", "") >= cut}
    players = {l["source"] for l in g["links"] if l["key"] == "juega_en" and l["target"] in teams and l.get("hasta", "") >= cut}
    keep = teams | players | {n["id"] for n in g["nodes"] if n["type"] == "mapa"}
    src = "data/cs2/grafo_cs2.json.gz"
    for n in g["nodes"]:
        if n["id"] not in keep: continue
        name = n.get("nombre") or n.get("nick") or n.get("name") or n["id"]
        tipo = {"equipo": "Equipo CS2", "jugador": "Jugador CS2", "mapa": "Mapa CS2"}[n["type"]]
        extra = {k: n[k] for k in ("equipo_actual", "rating", "adr", "kast", "kd", "forma", "wr_series", "wr_mapas", "mapa_fuerte", "mapa_debil", "mapas_fuertes", "mapas_debiles", "rivales_favorables", "rivales_desfavorables", "mejores_compañeros", "plantilla") if k in n}
        nodes.append(node(f"cs2:{n['id']}", f"{name} ({tipo})", src, **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in extra.items()}))
    for l in g["links"]:
        if l["source"] in keep and l["target"] in keep and l["key"] in ("juega_en", "compañero_de", "domina_mapa", "gana_a"):
            edges.append(edge(f"cs2:{l['source']}", f"cs2:{l['target']}", l["key"], src))


def espn(nodes, edges, league):
    path = DATA / league / f"grafo_{league}.json"
    if not path.exists(): return
    g = json.loads(path.read_text(encoding="utf-8"))
    src, tag = f"data/{league}/grafo_{league}.json", league.upper()
    for n in g["nodes"]:
        tipo = f"Equipo {tag}" if n["type"] == "equipo" else f"Jugador {tag}"
        extra = {k: n[k] for k in ("team", "games", "min", "pts", "pm") if k in n}
        nodes.append(node(f"{league}:{n['id']}", f"{n.get('name', n['id'])} ({tipo})", src, **extra))
    for e in g.get("edges", g.get("links", [])):
        edges.append(edge(f"{league}:{e['source']}", f"{league}:{e['target']}", e.get("type") or e.get("rel") or e.get("key") or "relación", src))


def main():
    nodes, edges = [], []
    cs2(nodes, edges)
    for lg in ("nba", "nhl"): espn(nodes, edges, lg)
    ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in ids and e["target"] in ids]
    out = OUT / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / ".graphify_extract.json").write_text(json.dumps({"nodes": nodes, "edges": edges, "hyperedges": [],
                                                            "input_tokens": 0, "output_tokens": 0}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(nodes)} nodos, {len(edges)} aristas")


if __name__ == "__main__":
    main()
