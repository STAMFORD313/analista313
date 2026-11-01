"""CS2 a nivel de JUGADOR con la API pública de bo3.gg (sin clave): partidos de nivel S/A/B con sus mapas, estadísticas
por jugador y partido (rating, ADR, KAST, K/D, primeras bajas) y rating por mapa, y fichajes. Construye el grafo de
conocimiento data/cs2/grafo_cs2.json (jugadores, equipos, mapas, torneos) y da features para el modelo (team_features).
Uso: python cs2_players.py [update|graph|backtest]   (sin argumento: update + graph). Incremental: solo baja lo nuevo."""
import json, math, sys, time, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import pandas as pd
import requests

API = "https://api.bo3.gg/api/v1"
UA = {"User-Agent": "Mozilla/5.0 (analista313; github.com/STAMFORD313/analista313)"}
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)  # concat con tablas vacías la primera vez
DIR = Path(__file__).parent / "data" / "cs2"
GAMES, PMATCH, PMAPS, NOSTATS, TRANSF, GRAPH = (DIR / f for f in (
    "players_games.csv", "players_matches.csv", "players_maps.csv", "players_nostats.csv", "players_transfers.csv", "grafo_cs2.json"))
SINCE = "2024-05-01"  # 2 años y medio: medio año de calentamiento para las features
TIERS = "s,a,b"
G_COLS = ["game_id", "match_id", "slug", "date", "tier", "bo", "tournament_id", "tournament", "t1_id", "t2_id", "team1", "team2",
          "number", "map", "winner_id", "ws", "ls"]
PM_COLS = ["match_id", "player_id", "nick", "slug", "team_id", "kills", "deaths", "assists", "adr", "kast", "rating", "fk", "fd", "clutches"]
PP_COLS = ["game_id", "match_id", "player_id", "team_id", "map", "rating"]


def get(path, pause=0.15, **params):
    """GET con reintentos cortos; respeta la API (pausa y User-Agent identificable)."""
    for i in range(4):
        try:
            r = requests.get(f"{API}/{path}", params=params, headers=UA, timeout=30)
            if r.status_code == 429 or r.status_code >= 500: raise requests.RequestException(r.status_code)
            r.raise_for_status(); time.sleep(pause)
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if i == 3: raise
            time.sleep(2 * (i + 1))


def _read(path, cols):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def update_games(max_pages=400):
    """Partidos S/A/B terminados con sus mapas, del más nuevo hacia atrás hasta enlazar con lo guardado (o hasta SINCE)."""
    old = _read(GAMES, G_COLS)
    known, rows = set(old.match_id), []
    for page in range(max_pages):
        res = get("matches", **{"page[limit]": 100, "page[offset]": page * 100, "sort": "-start_date", "with": "games,teams,tournament",
                                "filter[matches.discipline_id][eq]": 1, "filter[matches.status][in]": "finished",
                                "filter[matches.tier][in]": TIERS, "filter[matches.start_date][gt]": SINCE})["results"]
        new = [m for m in res if m["id"] not in known]
        for m in new:
            t1, t2, t = m.get("team1") or {}, m.get("team2") or {}, m.get("tournament") or {}
            for g in m.get("games") or []:
                w = (g.get("winner_team_clan") or {}).get("team_id")
                if not w or g.get("status") != "finished": continue
                rows.append({"game_id": g["id"], "match_id": m["id"], "slug": m["slug"], "date": m["start_date"], "tier": m.get("tier"),
                             "bo": m.get("bo_type"), "tournament_id": m.get("tournament_id"), "tournament": t.get("name", ""),
                             "t1_id": m["team1_id"], "t2_id": m["team2_id"], "team1": t1.get("name"), "team2": t2.get("name"),
                             "number": g.get("number"), "map": g.get("map_name"), "winner_id": w,
                             "ws": g.get("winner_clan_score"), "ls": g.get("loser_clan_score")})
        if page % 20 == 0: print("páginas", page, res[-1]["start_date"] if res else "", len(rows), flush=True)
        if len(new) < len(res) or not res: break
    df = pd.concat([old, pd.DataFrame(rows, columns=G_COLS)], ignore_index=True).drop_duplicates("game_id")
    df.sort_values(["date", "match_id", "number"]).to_csv(GAMES, index=False)
    return df


def _stats(slug, mid):
    """Estadísticas del partido por jugador (1 llamada por partido): agregado + rating de cada mapa."""
    pm, pp = [], []
    for s in get(f"matches/{slug}/players_stats"):
        sp = s.get("steam_profile") or {}
        pl = sp.get("player") or {}
        pid = pl.get("id") or -int(s["steam_profile_id"])  # sin ficha de jugador: id negativo del perfil de Steam
        tid = (s.get("team_clan") or {}).get("team_id")
        pm.append({"match_id": mid, "player_id": pid, "nick": pl.get("nickname") or sp.get("nickname"), "slug": pl.get("slug"),
                   "team_id": tid, "kills": s.get("kills"), "deaths": s.get("death"), "assists": s.get("assists"),
                   "adr": round(s.get("adr") or 0, 2), "kast": round(s.get("kast") or 0, 3), "rating": round(s.get("player_rating") or 0, 3),
                   "fk": s.get("first_kills"), "fd": s.get("first_death"), "clutches": s.get("clutches")})
        for g in sp.get("game_steam_profiles") or []:
            if g.get("player_rating") is not None:
                pp.append({"game_id": g["game_id"], "match_id": mid, "player_id": pid, "team_id": tid, "map": g.get("map_name"),
                           "rating": round(g["player_rating"], 3)})
    return mid, pm, pp


def update_stats(games, workers=4, limit=None, chunk=400):
    """Baja las estadísticas de jugadores de los partidos que aún no las tienen (en paralelo moderado, guardando cada `chunk`)."""
    pm_old, pp_old, nos = _read(PMATCH, PM_COLS), _read(PMAPS, PP_COLS), _read(NOSTATS, ["match_id"])
    m = games[games.date >= SINCE].drop_duplicates("match_id").sort_values("date", ascending=False)
    recent = pd.to_datetime(m.date, utc=True) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)  # stats aún pueden llegar
    todo = m[~m.match_id.isin(set(pm_old.match_id) | (set(nos.match_id) - set(m[recent].match_id)))]
    todo = list(zip(todo.slug, todo.match_id))[:limit]
    print(f"stats pendientes: {len(todo)}", flush=True)
    for i in range(0, len(todo), chunk):
        pm, pp, empty = [], [], []
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(lambda a: _safe_stats(*a), todo[i:i + chunk]):
                if res is None: continue
                mid, a, b = res
                pm += a; pp += b
                if not a: empty.append(mid)
        pm_old = pd.concat([pm_old, pd.DataFrame(pm, columns=PM_COLS)], ignore_index=True).drop_duplicates(["match_id", "player_id"])
        pp_old = pd.concat([pp_old, pd.DataFrame(pp, columns=PP_COLS)], ignore_index=True).drop_duplicates(["game_id", "player_id"])
        nos = pd.concat([nos, pd.DataFrame({"match_id": empty})]).drop_duplicates()
        pm_old.to_csv(PMATCH, index=False); pp_old.to_csv(PMAPS, index=False); nos.to_csv(NOSTATS, index=False)
        print(f"stats {i + len(todo[i:i + chunk])}/{len(todo)}", flush=True)
    return pm_old, pp_old


def _safe_stats(slug, mid):
    try: return _stats(slug, mid)
    except Exception as e:
        print("stats", slug, e, flush=True); return None


def update_transfers(pm, min_matches=10, refresh_days=7, workers=4):
    """Historial de equipos (fichajes/bajas con fecha) de los jugadores con al menos `min_matches` partidos.
    Se refresca cada `refresh_days` días para quien haya jugado desde el último refresco."""
    old = _read(TRANSF, ["player_id", "date", "from_id", "from_team", "to_id", "to_team", "action", "fetched"])
    cnt = pm[pm.player_id > 0].groupby("player_id").size()
    want = set(cnt[cnt >= min_matches].index)
    fetched = old.groupby("player_id").fetched.max() if len(old) else pd.Series(dtype=str)
    cut = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=refresh_days)).isoformat()
    todo = [p for p in want if p not in fetched.index or fetched[p] < cut]
    print(f"fichajes pendientes: {len(todo)}", flush=True)
    now = pd.Timestamp.now(tz="UTC").isoformat()

    def one(pid):
        try: res = get("player_transfers", **{"page[limit]": 50, "sort": "-action_date", "filter[player_id][eq]": int(pid),
                                              "filter[is_coach][eq]": "false"})["results"]
        except Exception as e:
            print("fichajes", pid, e, flush=True); return []
        rows = [{"player_id": pid, "date": t.get("action_date"), "from_id": t.get("team_from_id"), "from_team": t.get("team_from_name"),
                 "to_id": t.get("team_to_id"), "to_team": t.get("team_to_name"), "action": t.get("action_type"), "fetched": now} for t in res]
        return rows or [{"player_id": pid, "fetched": now}]  # marca de "consultado" aunque no tenga fichajes

    with ThreadPoolExecutor(workers) as ex: rows = [r for rs in ex.map(one, todo) for r in rs]
    df = pd.concat([old[~old.player_id.isin(todo)], pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(TRANSF, index=False)
    return df


def update(limit=None):
    DIR.mkdir(parents=True, exist_ok=True)
    games = update_games()
    pm, _ = update_stats(games, limit=limit)
    update_transfers(pm)


def _top(d, key, n=3, rev=True, min_n=5, wr=False):
    """Los n mejores (o peores) de un dict {nombre: {"n":…, key:…}} con muestra mínima; wr=True exige ganar (o perder) más de la mitad."""
    ok = [(k, v) for k, v in d.items() if v["n"] >= min_n and (not wr or (v[key] > 0.5 if rev else v[key] < 0.5))]
    return [k for k, _ in sorted(ok, key=lambda kv: kv[1][key], reverse=rev)[:n]]


def build_graph():
    """Grafo node-link (MultiDiGraph, como data/grafo.json) con jugadores, equipos, mapas y torneos y sus relaciones."""
    g, pp = load()
    pm = _read(PMATCH, PM_COLS)
    tr = _read(TRANSF, ["player_id", "date", "from_id", "from_team", "to_id", "to_team", "action", "fetched"])
    G = nx.MultiDiGraph()
    names = dict(zip(g.t1_id, g.team1)) | dict(zip(g.t2_id, g.team2))
    date = dict(zip(g.game_id, g.date))
    now = g.date.max()
    T = lambda t: f"equipo:{int(t)}"
    P = lambda p: f"jugador:{int(p)}"
    # --- mapas por equipo, series, H2H, nivel de torneo
    gm = pd.concat([g.assign(team=g.t1_id, rival=g.t2_id), g.assign(team=g.t2_id, rival=g.t1_id)])
    gm["win"] = (gm.winner_id == gm.team).astype(int)
    ser = gm.groupby(["match_id", "team", "rival", "date", "tier", "tournament_id"], as_index=False).win.agg(["sum", "count"])
    ser["won"] = ser["sum"] > ser["count"] - ser["sum"]
    ser = ser.sort_values("date")
    for t in set(g.t1_id) | set(g.t2_id):
        G.add_node(T(t), type="equipo", nombre=names.get(t))
    for m in g["map"].dropna().unique(): G.add_node(f"mapa:{m}", type="mapa", nombre=m.replace("de_", ""))
    for tid, tg in g.groupby("tournament_id"):
        G.add_node(f"torneo:{int(tid)}", type="torneo", nombre=tg.tournament.iloc[0], nivel=tg.tier.iloc[0], fecha=str(tg.date.min().date()))
    for (t, m), x in gm.groupby(["team", "map"]):
        G.add_edge(T(t), f"mapa:{m}", key="domina_mapa", rel="domina_mapa", mapas=len(x), wr=round(x.win.mean(), 3),
                   wr_ult_6m=round(x[x.date > now - pd.Timedelta(days=182)].win.mean(), 3) if (x.date > now - pd.Timedelta(days=182)).any() else None)
    for (t, r), x in ser.groupby(["team", "rival"]):
        w, l = int(x.won.sum()), int((~x.won).sum())
        mw, ml = int(x["sum"].sum()), int((x["count"] - x["sum"]).sum())
        if w != l or mw != ml:
            rel = "gana_a" if (w, mw) > (l, ml) else "pierde_con"
            G.add_edge(T(t), T(r), key=rel, rel=rel, series=f"{w}-{l}", mapas=f"{mw}-{ml}", ultima=str(x.date.max().date()))
    for (t, tid), x in ser.groupby(["team", "tournament_id"]):
        G.add_edge(T(t), f"torneo:{int(tid)}", key="participa_en", rel="participa_en", series=f"{int(x.won.sum())}-{int((~x.won).sum())}")
    # --- jugadores: fichas, mapas, compañeros, rivales
    pp = pp.assign(date=pp.game_id.map(date)).merge(g[["game_id", "winner_id", "t1_id", "t2_id"]], on="game_id")
    pp["win"] = (pp.winner_id == pp.team_id).astype(int)
    pp["rival"] = pp.t2_id.where(pp.team_id == pp.t1_id, pp.t1_id)
    pp = pp.sort_values("date")
    mdate = g.groupby("match_id").date.min()
    pm = pm.assign(date=pm.match_id.map(mdate)).sort_values("date")
    info = {p: x for p, x in pm.groupby("player_id")}
    for p, x in pp.groupby("player_id"):
        a = info.get(p)
        if a is None or len(x) < 3: continue
        last = a.iloc[-1]
        k, dth = a.kills.sum(), a.deaths.sum()
        maps = {m: {"n": len(y), "wr": round(y.win.mean(), 3), "rating": round(y.rating.mean(), 2)} for m, y in x.groupby("map")}
        riv = {names.get(r, str(r)): {"n": len(y), "wr": round(y.win.mean(), 3)} for r, y in x.groupby("rival")}
        G.add_node(P(p), type="jugador", nick=last.nick, slug=last.slug if isinstance(last.slug, str) else None,
                   equipo_actual=names.get(last.team_id), partidos=len(a), mapas=len(x), rating=round(x.rating.mean(), 2),
                   adr=round(a.adr.mean(), 1), kast=round(a.kast.mean(), 3), kd=round(k / max(dth, 1), 2),
                   primeras_bajas=f"{int(a.fk.sum())}-{int(a.fd.sum())}", wr_mapas=round(x.win.mean(), 3),
                   forma=round(x.rating.tail(10).mean(), 2), forma_wr=round(x.win.tail(10).mean(), 2), ultimo=str(x.date.max().date()),
                   mapas_fuertes=[m.replace("de_", "") for m in _top(maps, "rating")],
                   mapas_debiles=[m.replace("de_", "") for m in _top(maps, "rating", rev=False)],
                   rivales_favorables=_top(riv, "wr", min_n=4, wr=True), rivales_desfavorables=_top(riv, "wr", rev=False, min_n=4, wr=True),
                   por_mapa={m.replace("de_", ""): v for m, v in maps.items()})
        for m, v in maps.items():
            G.add_edge(P(p), f"mapa:{m}", key="domina_mapa", rel="domina_mapa", mapas=v["n"], wr=v["wr"], rating=v["rating"])
        for t, y in x.groupby("team_id"):
            if pd.isna(t): continue
            if T(t) not in G: G.add_node(T(t), type="equipo", nombre=None)
            G.add_edge(P(p), T(t), key="juega_en", rel="juega_en", desde=str(y.date.min().date()), hasta=str(y.date.max().date()),
                       mapas=len(y), wr=round(y.win.mean(), 3), rating=round(y.rating.mean(), 2))
    # compañeros: mapas jugados juntos, win rate y rating juntos
    pair = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for (_, t), x in pp.groupby(["game_id", "team_id"]):
        lu = sorted(zip(x.player_id, x.rating))
        w = int(x.win.iloc[0])
        for i, (a, ra) in enumerate(lu):
            for b, rb in lu[i + 1:]:
                s = pair[a, b]; s[0] += 1; s[1] += w; s[2] += ra; s[3] += rb
    best = defaultdict(list)
    for (a, b), (n, w, ra, rb) in pair.items():
        if n < 5 or P(a) not in G or P(b) not in G: continue
        G.add_edge(P(a), P(b), key="compañero_de", rel="compañero_de", mapas=n, wr=round(w / n, 3),
                   rating_juntos=[round(ra / n, 2), round(rb / n, 2)])
        best[a].append((w / n, n, b)); best[b].append((w / n, n, a))
    for p, l in best.items():
        G.nodes[P(p)]["mejores_compañeros"] = [G.nodes[P(b)]["nick"] for _, n, b in sorted(l, reverse=True) if n >= 10][:3]
    # fichajes (historial de equipos con fecha)
    for t in tr.dropna(subset=["to_id"]).itertuples():
        if P(t.player_id) in G:
            if T(t.to_id) not in G: G.add_node(T(t.to_id), type="equipo", nombre=t.to_team)
            G.add_edge(P(t.player_id), T(t.to_id), key=f"fichado_por {t.date}", rel="fichado_por", fecha=t.date,
                       desde=t.from_team if isinstance(t.from_team, str) else None)
    # --- fichas de equipo
    for t, x in ser.groupby("team"):
        mp = {G.nodes[v]["nombre"]: {"n": d["mapas"], "wr": d["wr"]} for _, v, d in G.out_edges(T(t), data=True) if d["rel"] == "domina_mapa"}
        rv = {names.get(r, str(r)): {"n": len(y), "wr": round(y.won.mean(), 3)} for r, y in x.groupby("rival")}
        last = pp[pp.team_id == t]
        lu = last[last.game_id == last.game_id.iloc[-1]].player_id if len(last) else []
        G.nodes[T(t)].update(
            series=len(x), wr_series=round(x.won.mean(), 3), forma="".join("V" if w else "D" for w in x.won.tail(5)),
            ultimo=str(x.date.max().date()), plantilla=[G.nodes[P(p)]["nick"] for p in lu if P(p) in G],
            mapa_fuerte=_top(mp, "wr", 2, wr=True), mapa_debil=_top(mp, "wr", 2, rev=False, wr=True),
            rivales_favorables=_top(rv, "wr", min_n=3, wr=True), rivales_desfavorables=_top(rv, "wr", rev=False, min_n=3, wr=True),
            por_nivel={k: {"series": len(y), "wr": round(y.won.mean(), 3)} for k, y in x.groupby("tier")})
    GRAPH.write_text(json.dumps(nx.node_link_data(G, edges="links"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    c = pd.Series([d["type"] for _, d in G.nodes(data=True)]).value_counts().to_dict()
    print(f"grafo: {c}, {G.number_of_edges()} aristas", pd.Series([d["rel"] for *_, d in G.edges(data=True)]).value_counts().to_dict())
    return G


PARAMS = DIR / "players_params.json"  # coeficientes del ajuste sobre el Elo (los escribe backtest())
FEATS = ["fuerza", "h2h", "quimica"]  # "mapas" (ventaja de mapa pool) no aporta en el backtest: se queda solo en el grafo


def load():
    g = pd.read_csv(GAMES)
    g["date"] = pd.to_datetime(g.date, utc=True)
    return g.sort_values(["date", "match_id", "number"], ignore_index=True), pd.read_csv(PMAPS)


class Estado:
    """Estado acumulado partido a partido (sin mirar el futuro): rating de cada jugador (media con olvido por mapa),
    última alineación de cada equipo, mapas jugados juntos por cada pareja, rendimiento por mapa de cada equipo
    (olvido temporal) y H2H por mapas."""
    DECAY, K, HL = 0.995, 2.0, 180  # (backtest: 0.97→0.995 baja el log-loss 0.6160→0.6139) olvido por mapa del jugador, encogimiento a la media, vida media (días) del mapa del equipo

    def __init__(self, pp):
        self.by_game = defaultdict(list)
        for r in pp.itertuples(index=False): self.by_game[r.game_id].append((r.player_id, r.team_id, r.rating))
        self.mu = float(pp.rating.mean()) if len(pp) else 6.0
        self.ps = defaultdict(lambda: [0.0, 0.0])
        self.lineup, self.pair, self.h2h = {}, defaultdict(float), defaultdict(float)
        self.tmap = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, None]))  # equipo → mapa → victorias, mapas, última fecha

    def _dec(self, s, d):
        f = 0.5 ** ((d - s[2]).days / self.HL) if s[2] is not None else 0
        return s[0] * f, s[1] * f

    def update(self, r):
        """Añade un mapa jugado (fila de players_games.csv)."""
        rows = self.by_game.get(r.game_id, [])
        for t in (r.t1_id, r.t2_id):
            lu = sorted({p for p, tt, _ in rows if tt == t})
            if len(lu) >= 3:
                self.lineup[t] = tuple(lu)
                for i, a in enumerate(lu):
                    for b in lu[i + 1:]: self.pair[a, b] += 1
            w, n = self._dec(self.tmap[t][r.map], r.date)
            self.tmap[t][r.map] = [w + (r.winner_id == t), n + 1, r.date]
        for p, _, rt in rows:
            s = self.ps[p]; s[0] = s[0] * self.DECAY + rt; s[1] = s[1] * self.DECAY + 1
        lo = r.t2_id if r.winner_id == r.t1_id else r.t1_id
        self.h2h[r.winner_id, lo] += 1

    def strength(self, t):
        lu = self.lineup.get(t)
        if not lu: return None
        return sum((self.ps[p][0] + self.K * self.mu) / (self.ps[p][1] + self.K) for p in lu) / len(lu)

    def chem(self, t):
        lu = self.lineup.get(t) or ()
        pairs = [math.log1p(self.pair[a, b]) for i, a in enumerate(lu) for b in lu[i + 1:]]
        return sum(pairs) / len(pairs) if pairs else 0.0

    def map_dev(self, t, d):
        """Por mapa: (victorias encogidas a su media global) − media global; y mapas jugados (con olvido)."""
        st = {m: self._dec(s, d) for m, s in self.tmap[t].items()}
        W, N = sum(w for w, _ in st.values()), sum(n for _, n in st.values())
        wr = W / N if N else 0.5
        return {m: ((w + 3 * wr) / (n + 3) - wr, n) for m, (w, n) in st.items()}

    def features(self, t, r, d):
        s1, s2 = self.strength(t), self.strength(r)
        d1, d2 = self.map_dev(t, d), self.map_dev(r, d)
        num = den = 0.0
        for m in set(d1) | set(d2):
            (a, n1), (b, n2) = d1.get(m, (0, 0)), d2.get(m, (0, 0))
            num += (n1 + n2) * (a - b); den += n1 + n2
        w, l = self.h2h[t, r], self.h2h[r, t]
        return {"fuerza": s1 - s2 if s1 is not None and s2 is not None else 0.0, "mapas": num / den if den else 0.0,
                "h2h": (w - l) / (w + l + 4), "quimica": self.chem(t) - self.chem(r)}


def run_states(g, pp, on_match=None):
    """Recorre los partidos en orden: on_match(match_id, t1, t2, fecha, estado) ANTES de añadir sus mapas."""
    st = Estado(pp)
    for mid, mg in g.groupby("match_id", sort=False):
        f = mg.iloc[0]
        if on_match: on_match(mid, f.t1_id, f.t2_id, f.date, st)
        for r in mg.itertuples(index=False): st.update(r)
    return st


_STATE = None


def team_features(team_id, rival_id, fecha=None):
    """Features de team_id frente a rival_id con lo jugado hasta ahora: fuerza de la plantilla actual (rating medio de su
    última alineación), ventaja de mapa pool, H2H por mapas y química (mapas jugados juntos por las parejas)."""
    global _STATE
    if _STATE is None: _STATE = run_states(*load())
    d = pd.Timestamp(fecha or pd.Timestamp.now(tz="UTC"))
    return _STATE.features(team_id, rival_id, d if d.tzinfo else d.tz_localize("UTC"))


def ajustar(p_elo, team1_id, team2_id, fecha=None):
    """Prob. de la serie del Elo de esports.py corregida con las features (coeficientes de players_params.json)."""
    if not PARAMS.exists() or not GAMES.exists(): return p_elo
    c = json.loads(PARAMS.read_text())
    f = team_features(team1_id, team2_id, fecha)
    p = min(max(p_elo, 1e-6), 1 - 1e-6)
    z = c["a"] * math.log(p / (1 - p)) + sum(c[k] * f[k] for k in FEATS)
    return 1 / (1 + math.exp(-z))


def logit_fit(X, y, l2=1e-3, iters=50):
    """Regresión logística sin intercepto (Newton); la simetría t1/t2 la hace innecesaria."""
    import numpy as np
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - y) + l2 * w)
    return w


def backtest(train_from="2024-11-01", test_from="2025-11-01"):
    """Ajusta con un año y mide en el siguiente (features calculadas ANTES de cada partido); compara con el Elo de esports.py."""
    import numpy as np
    import esports as E
    df = pd.read_csv(E.HIST)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date", ignore_index=True)
    df = df[df.date >= df.date.max() - pd.DateOffset(years=4)]
    since = pd.Timestamp(train_from, tz="UTC")
    rec, orig = [], E.series_prob
    E.series_prob = lambda p, bo: rec.append(orig(p, bo)) or rec[-1]  # captura la prob. previa de cada serie
    try: E.map_elo(df, since)
    finally: E.series_prob = orig
    elo = df[df.date >= since].assign(p_elo=rec)[["id", "t1_id", "winner_id", "p_elo"]]
    g, pp = load()
    rows = []
    run_states(g, pp, lambda mid, t1, t2, d, st: rows.append({"id": mid, "t1": t1, "date": d, **st.features(t1, t2, d)}))
    x = pd.DataFrame(rows).merge(elo, on="id")
    x = x[x.t1 == x.t1_id]
    x["y"] = (x.winner_id == x.t1_id).astype(float)
    p = x.p_elo.clip(1e-6, 1 - 1e-6)
    x["le"] = np.log(p / (1 - p))
    tr, te = x[(x.date >= since) & (x.date < pd.Timestamp(test_from, tz="UTC"))], x[x.date >= pd.Timestamp(test_from, tz="UTC")]

    def ll(cols, w=None):
        w = logit_fit(tr[cols].to_numpy(), tr.y.to_numpy()) if w is None else w
        q = 1 / (1 + np.exp(-te[cols].to_numpy() @ w))
        l = -(te.y * np.log(q) + (1 - te.y) * np.log(1 - q))
        return l.mean(), l, w

    base, lb, _ = ll(["le"], np.array([1.0]))
    cal, lc, _ = ll(["le"])
    full, lf, w = ll(["le", *FEATS])
    print(f"series S/A/B: ajuste {len(tr)} · prueba {len(te)}")
    print(f"log-loss Elo esports.py {base:.4f} · Elo recalibrado {cal:.4f} · Elo + features jugadores {full:.4f} "
          f"(Δ {full - base:+.4f} ± {(lf - lb).std() / len(te) ** 0.5:.4f})")
    for f in FEATS:
        print(f"  sin {f}: {ll(['le', *[k for k in FEATS if k != f]])[0]:.4f}")
    print("coeficientes", dict(zip(["a", *FEATS], np.round(w, 3))))
    res = {"a": float(w[0]), **{k: float(v) for k, v in zip(FEATS, w[1:])}, "logloss_elo": round(base, 4), "logloss": round(full, 4),
           "n_test": len(te)}
    if full < base: PARAMS.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("update", "all"): update(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if cmd in ("graph", "all") and "build_graph" in globals(): build_graph()
    if cmd == "backtest" and "backtest" in globals(): backtest()
