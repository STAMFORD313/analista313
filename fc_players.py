"""EA FC 27 (Esoccer Battle) a nivel de JUGADOR: grafo de conocimiento data/fc/grafo_fc.json.gz con un nodo por jugador
(ficha: clubes que usa y con cuál rinde mejor, rivales favorables/desfavorables, forma, goles a favor/en contra por partido,
tendencia a over/under), por club y por liga. Usa data/fc/matches.csv de esoccer.py ([agente-cs2]) y añade el club de cada
jugador en cada partido (data/fc/players_teams.csv, API JSON pública de football.esportsbattle.com, 1 llamada por torneo).
Uso: python fc_players.py [update|graph|backtest]   (sin argumento: update + graph)."""
import gzip, json, math, sys, time

import networkx as nx
import pandas as pd
import requests

API = "https://football.esportsbattle.com/api"
UA = {"User-Agent": "Mozilla/5.0 (analista313; github.com/STAMFORD313/analista313)"}
from pathlib import Path
DIR = Path(__file__).parent / "data" / "fc"
HIST, TEAMS, GRAPH = DIR / "matches.csv", DIR / "players_teams.csv", DIR / "grafo_fc.json.gz"
T_COLS = ["id", "tournament_id", "club1", "club2", "ht1", "ht2"]


def update(pause=0.3, limit=None):
    """Club usado por cada jugador (y marcador al descanso) en los torneos de matches.csv que aún no están en players_teams.csv."""
    if not HIST.exists(): return
    h = pd.read_csv(HIST)
    old = pd.read_csv(TEAMS) if TEAMS.exists() else pd.DataFrame(columns=T_COLS)
    todo = sorted(set(h.tournament_id) - set(old.tournament_id))[:limit]
    print(f"fc torneos pendientes: {len(todo)}", flush=True)
    rows = []
    for tid in todo:
        try:
            r = requests.get(f"{API}/tournaments/{tid}/matches", headers=UA, timeout=30)
            r.raise_for_status()
            for m in r.json():
                a, b = m.get("participant1") or {}, m.get("participant2") or {}
                ht = lambda p: (p.get("prevPeriodsScores") or [None])[0]
                rows.append({"id": m["id"], "tournament_id": tid, "club1": (a.get("team") or {}).get("token_international"),
                             "club2": (b.get("team") or {}).get("token_international"), "ht1": ht(a), "ht2": ht(b)})
        except (requests.RequestException, ValueError) as e:
            print("esportsbattle", tid, e)
        time.sleep(pause)
    pd.concat([old, pd.DataFrame(rows, columns=T_COLS)], ignore_index=True).drop_duplicates("id").to_csv(TEAMS, index=False)


def load():
    """Partidos con club de cada lado, en orden (una fila por partido)."""
    h = pd.read_csv(HIST)
    if TEAMS.exists(): h = h.merge(pd.read_csv(TEAMS).drop(columns="tournament_id"), on="id", how="left")
    else: h = h.assign(club1=None, club2=None, ht1=None, ht2=None)
    h["date"] = pd.to_datetime(h.date, utc=True)
    return h.dropna(subset=["s1", "s2"]).sort_values(["date", "id"], ignore_index=True)


def _side(h):
    """Una fila por jugador y partido: gf, gc, puntos, club, rival, liga y total del partido."""
    a = h.rename(columns={"p1": "p", "p2": "r", "s1": "gf", "s2": "gc", "club1": "club"})
    b = h.rename(columns={"p2": "p", "p1": "r", "s2": "gf", "s1": "gc", "club2": "club"})
    x = pd.concat([a[["id", "date", "league", "p", "r", "gf", "gc", "club"]], b[["id", "date", "league", "p", "r", "gf", "gc", "club"]]])
    x["pts"] = (x.gf > x.gc) * 3 + (x.gf == x.gc)
    x["tot"] = x.gf + x.gc
    return x.sort_values(["date", "id"])


def _top(d, n=3, rev=True, min_n=3, key="ppg"):
    ok = [(k, v) for k, v in d.items() if v["n"] >= min_n and (v[key] > 1.5 if rev else v[key] < 1.5)]
    return [k for k, _ in sorted(ok, key=lambda kv: kv[1][key], reverse=rev)[:n]]


def build_graph():
    """Grafo node-link (MultiDiGraph) comprimido: jugadores, clubes y ligas de FC con sus relaciones."""
    h = load()
    x = _side(h)
    lg_tot = x.groupby("league").tot.mean()
    x["over"] = x.tot - x.league.map(lg_tot)  # goles por encima de la media de su liga
    G = nx.MultiDiGraph()
    for lg, y in x.groupby("league"):
        G.add_node(f"liga:{lg}", type="liga", nombre=lg, partidos=len(y) // 2, goles_por_partido=round(y.tot.mean(), 2))
    for c, y in x.dropna(subset=["club"]).groupby("club"):
        G.add_node(f"club:{c}", type="club", nombre=c, partidos=len(y), ppg=round(y.pts.mean(), 2), gf=round(y.gf.mean(), 2),
                   gc=round(y.gc.mean(), 2))
    for p, y in x.groupby("p"):
        clubs = {c: {"n": len(z), "ppg": round(z.pts.mean(), 2), "gf": round(z.gf.mean(), 2), "gc": round(z.gc.mean(), 2)}
                 for c, z in y.dropna(subset=["club"]).groupby("club")}
        riv = {r: {"n": len(z), "ppg": round(z.pts.mean(), 2)} for r, z in y.groupby("r")}
        best = _top(clubs, 1)
        G.add_node(f"jugador_fc:{p}", type="jugador_fc", nick=p, partidos=len(y), v=int((y.pts == 3).sum()), e=int((y.pts == 1).sum()),
                   d=int((y.pts == 0).sum()), ppg=round(y.pts.mean(), 2), gf_pp=round(y.gf.mean(), 2), gc_pp=round(y.gc.mean(), 2),
                   forma="".join("VED"[{3: 0, 1: 1, 0: 2}[v]] for v in y.pts.tail(10)), forma_ppg=round(y.pts.tail(10).mean(), 2),
                   tendencia_goles=round(y.over.mean(), 2), tendencia="over" if y.over.mean() > 0.3 else "under" if y.over.mean() < -0.3 else "neutra",
                   clubes=sorted(clubs, key=lambda c: -clubs[c]["n"])[:5], mejor_club=best[0] if best else None,
                   rivales_favorables=_top(riv), rivales_desfavorables=_top(riv, rev=False), ultimo=str(y.date.max().date()))
        for c, v in clubs.items():
            G.add_edge(f"jugador_fc:{p}", f"club:{c}", key="usa_club", partidos=v["n"], ppg=v["ppg"], gf=v["gf"], gc=v["gc"])
        for lg, z in y.groupby("league"):
            G.add_edge(f"jugador_fc:{p}", f"liga:{lg}", key="juega_en", partidos=len(z), ppg=round(z.pts.mean(), 2))
        for r, v in riv.items():
            if v["ppg"] != 1.5 and v["n"] >= 2:
                k = "gana_a" if v["ppg"] > 1.5 else "pierde_con"
                G.add_edge(f"jugador_fc:{p}", f"jugador_fc:{r}", key=k, partidos=v["n"], ppg=v["ppg"])
    G.graph["actualizado"] = pd.Timestamp.now(tz="UTC").isoformat()
    js = json.dumps(nx.node_link_data(G, edges="links"), ensure_ascii=False, separators=(",", ":"),
                    default=lambda o: o.item() if hasattr(o, "item") else str(o))
    GRAPH.write_bytes(gzip.compress(js.encode("utf-8"), 9, mtime=0))
    c = pd.Series([d["type"] for _, d in G.nodes(data=True)]).value_counts().to_dict()
    print(f"grafo FC: {c}, {G.number_of_edges()} aristas")
    return G


def load_graph():
    return nx.node_link_graph(json.loads(gzip.decompress(GRAPH.read_bytes())), edges="links")


def refresh():
    """Para el run horario: clubes de los torneos nuevos y grafo si tiene más de 1 día (pesa poco). Nunca rompe el run."""
    try:
        update()
        done = load_graph().graph.get("actualizado", "") if GRAPH.exists() else ""
        if done < (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).isoformat(): build_graph()
    except Exception as e:
        print("fc_players refresh:", repr(e))


def backtest(test_frac=0.5, lr_club=(0.0, 0.005, 0.01, 0.02, 0.04)):
    """¿El club usado mejora el modelo de esoccer.py? Goles ~ Poisson(λ_esoccer · exp(ataque_club_A − defensa_club_B)), club
    aprendido online. Log-loss 1X2 en la 2.ª mitad del historial (predicción previa a cada partido)."""
    import esoccer as F
    h = load()
    since = h.date.iloc[int(len(h) * test_frac)]
    res = {}
    for lr in lr_club:
        m, ca, cd, loss, cnt = F.Model(), {}, {}, 0.0, 0
        for r in h.itertuples():
            c1, c2 = (r.club1 if isinstance(r.club1, str) else None), (r.club2 if isinstance(r.club2, str) else None)
            k1 = math.exp(ca.get(c1, 0) - cd.get(c2, 0)); k2 = math.exp(ca.get(c2, 0) - cd.get(c1, 0))
            l1, l2 = m.lam(r.league, r.p1, r.p2) * k1, m.lam(r.league, r.p2, r.p1) * k2
            if r.date >= since:
                p = F.markets(l1, l2)[0]
                o = "1" if r.s1 > r.s2 else "2" if r.s2 > r.s1 else "X"
                loss -= math.log(max(p[o], 1e-9)); cnt += 1
            m.update(r.league, r.p1, r.p2, r.s1, r.s2)
            for c, d, g, l in ((c1, c2, r.s1, l1), (c2, c1, r.s2, l2)):
                if c and d:
                    ca[c] = ca.get(c, 0) + lr * (g - l); cd[d] = cd.get(d, 0) - lr * (g - l)
        res[lr] = (loss / max(cnt, 1), cnt)
        print(f"club lr {lr}: log-loss 1X2 {res[lr][0]:.4f} ({cnt} partidos)", flush=True)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("update", "all"): update()
    if cmd in ("graph", "all"): build_graph()
    if cmd == "backtest": backtest()
