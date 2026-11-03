"""Esports con la API pública de bo3.gg (CS2, Dota 2, LoL, R6 Siege, MLBB): historial, cuotas reales de Thunderpick
y ganador. Modelo: Elo por MAPA/partida (en CS2 con margen de rondas) con K extra para equipos nuevos, vuelta al rating
inicial por inactividad y conversión mapa → serie (BO1/BO3/BO5); en BO3 también total de mapas y hándicap ±1.5."""
import json, math, time

import pandas as pd
import requests

import analyst as A
import cs2_props

API = "https://api.bo3.gg/api/v1/matches"
COLS = ["id", "date", "team1", "team2", "t1_id", "t2_id", "s1", "s2", "winner_id", "bo", "tier", "tournament"]
TIER_K_BASE = {"s": 1.25, "a": 1.1, "b": 1.0, "c": 0.85, "d": 0.7}       # los torneos grandes enseñan más
TIER_START_BASE = {"s": 1650, "a": 1600, "b": 1520, "c": 1450, "d": 1400}  # rating de un equipo nuevo según dónde juega
# disciplinas de bo3.gg con cuotas de Thunderpick: carpeta/clave → (discipline_id, icono, nombre). Valorant (2) no trae partidos.
DISCIPLINES = {"cs2": (1, "🎮", "CS2"), "dota2": (4, "🐉", "Dota 2"), "lol": (3, "🧙", "LoL"), "r6": (7, "🔫", "R6 Siege"),
               "mlbb": (8, "📱", "MLBB"), "fc": (None, "🎮⚽", "FC 27")}  # FC 27 no está en bo3.gg: lo lleva esoccer.py


def configure(slug):
    """Apunta el módulo a una disciplina: sus archivos en data/<slug>/ y sus parámetros entrenados (train_cs2.py <slug>)."""
    global SLUG, DISC, ICON, NAME, TITLE, DIR, HIST, GAMES, ODDS, K, TIER_K, TIER_START, NEW_K, DECAY, K_BO1, SERIES_SCALE, MAPS3, MOV, REGION_K, BLEND
    SLUG, (DISC, ICON, NAME) = slug, DISCIPLINES[slug]
    TITLE = f"{ICON} {NAME}"
    DIR = A.DATA / slug
    HIST = DIR / "matches.csv"
    GAMES = DIR / "games.csv"  # un mapa por fila (rondas y ganador; solo CS2), desde 2023 (train_cs2.py fetch) y luego cada hora
    ODDS = DIR / "odds.csv"    # cuotas previas de Thunderpick (la última vista antes del inicio)
    P = json.loads((DIR / "params.json").read_text()) if (DIR / "params.json").exists() else {}  # lo ajusta train_cs2.py
    K = P.get("K", 24)
    TIER_K = {t: 1 + (v - 1) * P.get("tier_scale", 1) for t, v in TIER_K_BASE.items()}
    TIER_START = {t: 1500 + (v - 1500) * P.get("start_spread", 1) for t, v in TIER_START_BASE.items()}
    NEW_K = P.get("new_k", 0)                # equipos nuevos: K más alto las primeras series
    DECAY = P.get("decay", 0)                # vuelta al rating inicial por inactividad (por año)
    K_BO1 = P.get("k_bo1", 1)                # peso extra de un BO1 (un único mapa decide la serie)
    SERIES_SCALE = P.get("series_scale", 1)  # escala de la diferencia de Elo en BO3/BO5
    MAPS3 = P.get("maps3", [0, 1])           # recalibración de la prob. de ir al 3er mapa en BO3 (ver bo3_scores)
    MOV = P.get("mov", 0)
    REGION_K = P.get("region_k", 0)
    BLEND = P.get("blend", 0.5)              # peso del modelo frente a la prob. sin margen de Thunderpick (ver market_report)          # K del nivel de región (cruces interregionales); 0 = sin ajuste                    # margen de rondas: K × (ln(dif+1)/ln 7)^MOV (13-7 → ×1)


configure("cs2")


def get(**params):
    p = {"page[limit]": 100, "filter[matches.discipline_id][eq]": DISC, "with": "teams,tournament", **params}
    r = requests.get(API, params=p, headers=A.UA, timeout=60)
    r.raise_for_status()
    return r.json()


def row(m):
    t1, t2 = m.get("team1") or {}, m.get("team2") or {}
    if not (t1.get("name") and t2.get("name")): return None
    return {"id": m["id"], "date": m["start_date"], "team1": t1["name"], "team2": t2["name"], "t1_id": m["team1_id"],
            "t2_id": m["team2_id"], "s1": m.get("team1_score"), "s2": m.get("team2_score"), "winner_id": m.get("winner_team_id"),
            "bo": m.get("bo_type") or 1, "tier": m.get("tier") or "c", "tournament": (m.get("tournament") or {}).get("name", "")}


def game_rows(m):
    """Mapas jugados de un partido (with=games): ganador y rondas para el Elo con margen."""
    return [{"match_id": m["id"], "n": g.get("number"), "map": g.get("map_name"), "winner_id": g["winner_team_clan"]["team_id"],
             "ws": g.get("winner_clan_score"), "ls": g.get("loser_clan_score")}
            for g in m.get("games") or [] if g.get("status") == "finished" and (g.get("winner_team_clan") or {}).get("team_id")]


def load_games():
    if not GAMES.exists(): return {}
    g = pd.read_csv(GAMES).sort_values(["match_id", "n"])
    return {k: list(zip(v.winner_id, v.ws, v.ls)) for k, v in g.groupby("match_id")}


def update_history(max_pages=800):
    """Primera vez baja todo (~770 páginas); después solo lo nuevo hasta enlazar con lo guardado."""
    DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=COLS)
    known, rows, games = set(old.id), [], []
    for page in range(max_pages):
        try: res = get(**{"page[offset]": page * 100, "sort": "-start_date", "filter[matches.status][in]": "finished",
                          "with": "teams,tournament,games"})["results"]
        except (requests.RequestException, ValueError) as e:
            print("bo3.gg", e); break
        batch = [r for r in map(row, res) if r]
        rows += [r for r in batch if r["id"] not in known]
        games += [g for m in res if m["id"] not in known for g in game_rows(m)]
        if not res or any(r["id"] in known for r in batch): break
        time.sleep(0.3)  # pausa corta entre páginas (la primera descarga son cientos)
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True).drop_duplicates("id")
    df = df[df.winner_id.notna()]
    df.to_csv(HIST, index=False)
    if games:
        pd.concat([pd.read_csv(GAMES) if GAMES.exists() else None, pd.DataFrame(games)]).drop_duplicates(["match_id", "n"]).to_csv(GAMES, index=False)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date", ignore_index=True)


def decayed(r, gap_days, start):
    """Tras >30 días sin jugar el rating vuelve poco a poco al inicial de su nivel (plantillas que se deshacen)."""
    if gap_days <= 30: return r
    return start + (r - start) * math.exp(-(gap_days - 30) / 365 * DECAY)


G = {}  # nivel aprendido de cada región (lo rellena map_elo)


def load_regions():
    """Equipo → región de bo3.gg (data/<slug>/teams.csv, train_cs2.py <slug> regiones).
    shortcut: equipos creados después de bajar teams.csv no tienen región (sin ajuste); rehacerlo cada pocos meses."""
    f = DIR / "teams.csv"
    if not f.exists(): return {}
    t = pd.read_csv(f).dropna()
    return dict(zip(t.id, t.region.astype(int)))


def map_elo(df, since=None, games=None, preds=None):
    """Elo por mapa: una serie 2-1 son 3 actualizaciones; los equipos con pocas series aprenden más rápido (K×(1+NEW_K/(1+n))).
    Devuelve (ratings, series jugadas, último nivel, última fecha, log-loss de series desde `since` predichas antes de jugarse)."""
    R, n, tier, last, loss, cnt = {}, {}, {}, {}, 0.0, 0
    G.clear()  # nivel de cada región: solo lo mueven los cruces entre regiones (REGION_K)
    reg = load_regions() if REGION_K else {}
    off = lambda t: G.get(reg.get(t), 0.0)
    for r in df.itertuples():
        for t in (r.t1_id, r.t2_id):
            s0 = TIER_START.get(r.tier, 1450)
            R[t] = decayed(R[t], (r.date - last[t]).days, s0) if t in R else s0
            n.setdefault(t, 0); tier[t] = r.tier; last[t] = r.date
        if since is not None and r.date >= since:
            pm = map_prob(R[r.t1_id] + off(r.t1_id), R[r.t2_id] + off(r.t2_id), r.bo)
            p = series_prob(pm, int(r.bo))
            loss -= math.log(min(max(p if r.winner_id == r.t1_id else 1 - p, 1e-6), 1)); cnt += 1
            if preds is not None: preds[r.id] = (p, pm)
        maps = [(w == r.t1_id, (math.log(ws - ls + 1) / math.log(7)) ** MOV if ws > ls else 1)  # margen de rondas
                for w, ws, ls in (games or {}).get(r.id, [])]
        if not maps:
            s1, s2 = (int(r.s1), int(r.s2)) if pd.notna(r.s1) and pd.notna(r.s2) and r.s1 + r.s2 > 0 else                 ((1, 0) if r.winner_id == r.t1_id else (0, 1))
            if r.bo == 1 and s1 + s2 > 1: s1, s2 = int(s1 > s2), int(s2 > s1)  # en BO1 el marcador son rondas, no mapas
            maps = [(1, 1)] * s1 + [(0, 1)] * s2
        k = K * TIER_K.get(r.tier, 1.0) * (K_BO1 if r.bo == 1 else 1)  # un BO1 es todo el partido en un mapa
        k1, k2 = k * (1 + NEW_K / (1 + n[r.t1_id])), k * (1 + NEW_K / (1 + n[r.t2_id]))
        g1, g2 = reg.get(r.t1_id), reg.get(r.t2_id)
        cross = g1 is not None and g2 is not None and g1 != g2
        for won1, mov in maps:
            e = 1 / (1 + 10 ** ((R[r.t2_id] + off(r.t2_id) - R[r.t1_id] - off(r.t1_id)) / 400))
            R[r.t1_id] += k1 * mov * (won1 - e); R[r.t2_id] -= k2 * mov * (won1 - e)
            if cross: G[g1] = G.get(g1, 0.0) + REGION_K * mov * (won1 - e); G[g2] = G.get(g2, 0.0) - REGION_K * mov * (won1 - e)
        n[r.t1_id] += 1; n[r.t2_id] += 1
    return R, n, tier, last, loss / max(cnt, 1), cnt


def map_prob(r1, r2, bo):
    """Prob. de ganar un mapa; en BO3/BO5 la diferencia se encoge (backtest: los mapas de una serie no son independientes)."""
    return 1 / (1 + 10 ** ((r2 - r1) * (SERIES_SCALE if bo > 1 else 1) / 400))


def series_prob(p, bo):
    """De probabilidad por mapa a probabilidad de ganar la serie."""
    need = bo // 2 + 1
    return sum(math.comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def form(df, tid, n=5):
    d = df[(df.t1_id == tid) | (df.t2_id == tid)].tail(n)
    return "".join("✅" if w == tid else "❌" for w in d.winner_id) or "—"


def settle(df, state):
    done = {f"{SLUG}|{r['id']}": r for r in df.tail(5000).to_dict("records")}
    for s in state["signals"]:
        r = done.get(s["key"])
        if s.get("result") is None and r is not None:
            s["score"] = f"{int(r['s1'])}-{int(r['s2'])}" if pd.notna(r["s1"]) else ""
            if s["market"] in ("t1", "t2"): s["result"] = int(r["winner_id"]) == s["team_id"]
            elif pd.isna(r["s1"]) or pd.isna(r["s2"]) or max(r["s1"], r["s2"]) != 2 or r["s1"] + r["s2"] not in (2, 3): s["result"] = "nula"
            elif s["market"] in ("over", "under"): s["result"] = (r["s1"] + r["s2"] == 3) == (s["market"] == "over")
            else: s["result"] = (r["s1"] - r["s2"]) * (1 if s["market"] == "AHH" else -1) + s["line"] > 0  # hándicap de mapas ±1.5


def bo3_scores(p):
    """Marcador de un BO3 desde la prob. por mapa: la prob. de llegar al 3er mapa se recalibra (MAPS3 = a, b sobre el logit de
    2p(1-p); con mapas independientes sale demasiado alta: 43 % frente al 40 % real)."""
    a, b = MAPS3
    x = 2 * p * (1 - p)
    q3 = 1 / (1 + math.exp(-(a + b * math.log(x / (1 - x)))))
    two = 1 - q3
    return {(2, 0): two * p * p / (p * p + (1 - p) ** 2), (0, 2): two * (1 - p) ** 2 / (p * p + (1 - p) ** 2), (2, 1): q3 * p, (1, 2): q3 * (1 - p)}


# mercados extra de Thunderpick en BO3 → (código de mercado, línea, quién, función del marcador que gana la apuesta)
EXTRA = {"total_maps_over_2_5": ("over", None, 0, lambda a, b: a + b == 3), "total_maps_under_2_5": ("under", None, 0, lambda a, b: a + b == 2),
         "team_1_handicap_under_1_5": ("AHH", -1.5, 1, lambda a, b: a - b >= 2), "team_1_handicap_over_1_5": ("AHH", 1.5, 1, lambda a, b: b - a < 2),
         "team_2_handicap_under_1_5": ("AHA", -1.5, 2, lambda a, b: b - a >= 2), "team_2_handicap_over_1_5": ("AHA", 1.5, 2, lambda a, b: a - b < 2)}
PAIRS = [("total_maps_over_2_5", "total_maps_under_2_5"), ("team_1_handicap_under_1_5", "team_2_handicap_over_1_5"),
         ("team_2_handicap_under_1_5", "team_1_handicap_over_1_5")]


def extra_markets(bet, p_map, g):
    """Total de mapas y hándicap ±1.5 en BO3: modelo (bo3_scores) mezclado 50/50 con la prob. sin margen de Thunderpick."""
    odds = {x["bet_type"]: x.get("coeff") for x in bet.get("additional_markets") or [] if x.get("active", True)}
    sc, out = bo3_scores(p_map), []
    for pair in PAIRS:
        o = [odds.get(k) for k in pair]
        if not all(x and x > 1 for x in o): continue
        inv = sum(1 / x for x in o)
        for k, ok in zip(pair, o):
            market, line, side, wins = EXTRA[k]
            q = BLEND * sum(v for (a, b), v in sc.items() if wins(a, b)) + (1 - BLEND) / ok / inv
            team = g["team1"] if side == 1 else g["team2"]
            pick = f"{team} {line:+g} mapas" if line else ("Más" if market == "over" else "Menos") + " de 2.5 mapas"
            if q >= A.MIN_P and A.EDGE <= q * ok - 1 <= A.MAX_EDGE:
                out.append({"market": market, "line": line, "pick": pick, "odds": ok, "book": "Thunderpick",
                            "p": round(q, 3), "implied": round(1 / ok / inv, 3), "edge": round(q * ok - 1, 3)})
    best = {}
    for v in out:  # una apuesta por código de mercado (AHH/AHA/over/under): la de más ventaja
        if v["market"] not in best or v["edge"] > best[v["market"]]["edge"]: best[v["market"]] = v
    return list(best.values())


def sgp_lines(bet, p_map, g):
    """«Crear apuesta» del mismo partido en BO3: ganador + más de 2.5 mapas = marcador 2-1 (train_sgp_esports.py: el
    marcador de bo3_scores está calibrado, el producto de ganador × más de 2.5 lo sobrestima ~8 %). Solo candidata:
    cuota justa del modelo y, si Thunderpick da ganador y hándicap, su prob. sin margen implícita (ganar − ganar 2-0)."""
    sc = bo3_scores(p_map)
    odds = {x["bet_type"]: x.get("coeff") for x in bet.get("additional_markets") or [] if x.get("active", True)}
    o1, o2 = ((bet.get(t) or {}).get("coeff") for t in ("team_1", "team_2"))
    out = []
    for side, team, cell, win, ah in ((1, g["team1"], (2, 1), sc[2, 0] + sc[2, 1], ("team_1_handicap_under_1_5", "team_2_handicap_over_1_5")),
                                      (2, g["team2"], (1, 2), sc[0, 2] + sc[1, 2], ("team_2_handicap_under_1_5", "team_1_handicap_over_1_5"))):
        pj, naive = sc[cell], win * (sc[2, 1] + sc[1, 2])
        txt = f"🧩 {A.esc(team)} gana + Más de 2.5 mapas (2-1): {pj:.0%}, cuota justa {1 / pj:.2f} (producto ingenuo {1 / naive:.2f})"
        a, b = (odds.get(k) for k in ah)
        if o1 and o2 and o1 > 1 and o2 > 1 and a and b and a > 1 and b > 1:
            pw = (1 / (o1 if side == 1 else o2)) / (1 / o1 + 1 / o2)
            implied = pw - (1 / a) / (1 / a + 1 / b)
            if implied > 0.01: txt += f" · casa {implied:.0%}"
        out.append(txt)
    return ["", "🧩 <b>Crear apuesta</b> (marcador exacto, sin stake)", *out]


def save_odds(up, now, prob=lambda m: None):
    """Guarda la última cuota PREVIA de cada partido (la de bo3.gg en partidos terminados es la del final, en directo):
    sirve para medir calibración y ROI del modelo frente a Thunderpick sin mirar el futuro."""
    rows = [{"id": m["id"], "seen": now.isoformat(), "start": m["start_date"], "p": prob(m),
             "o1": (m.get("bet_updates") or {}).get("team_1", {}).get("coeff"), "o2": (m.get("bet_updates") or {}).get("team_2", {}).get("coeff"),
             **{k: x.get("coeff") for x in (m.get("bet_updates") or {}).get("additional_markets") or [] for k in [x.get("bet_type")] if k in EXTRA}}
            for m in up if pd.Timestamp(m["start_date"]) > now]
    rows = [r for r in rows if r["o1"] and r["o2"]]
    if not rows: return
    old = pd.read_csv(ODDS) if ODDS.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("id", keep="last").to_csv(ODDS, index=False)


def market_report():
    """Modelo vs Thunderpick con la prob. del modelo y la cuota PREVIA guardadas antes de cada partido (odds.csv):
    partidos, log-loss de modelo / mezcla del bot / mercado sin margen / mezclas 0.25-0.75, y ROI plano (± error típico)
    de lo que la mezcla habría apostado (ventaja entre EDGE y MAX_EDGE)."""
    if not (ODDS.exists() and HIST.exists()): return {"n": 0}
    o = pd.read_csv(ODDS)
    if "p" not in o: return {"n": 0}
    o = o.dropna(subset=["p", "o1", "o2"]).merge(pd.read_csv(HIST, usecols=["id", "t1_id", "winner_id"]), on="id")
    o = o[(o.o1 > 1) & (o.o2 > 1)]
    if o.empty: return {"n": 0}
    y = (o.winner_id == o.t1_id).astype(float)
    mk = (1 / o.o1) / (1 / o.o1 + 1 / o.o2)
    ll = lambda q: float(-(y * q.clip(1e-6, 1).map(math.log) + (1 - y) * (1 - q).clip(1e-6, 1).map(math.log)).mean())
    q = BLEND * o.p + (1 - BLEND) * mk
    ret = [(o1 - 1 if yy else -1) if A.EDGE <= q1 * o1 - 1 <= A.MAX_EDGE else (o2 - 1 if not yy else -1)
           for q1, o1, o2, yy in zip(q, o.o1, o.o2, y) if A.EDGE <= q1 * o1 - 1 <= A.MAX_EDGE or A.EDGE <= (1 - q1) * o2 - 1 <= A.MAX_EDGE]
    sd = pd.Series(ret, dtype=float)
    return {"n": len(o), "modelo": ll(o.p), "mezcla": ll(q), "mercado": ll(mk), "blend": BLEND,
            **{f"mezcla_{w}": ll(w * o.p + (1 - w) * mk) for w in (0.25, 0.5, 0.75)},
            "apuestas": len(ret), "roi": float(sd.mean()) if ret else None,
            "roi_err": float(sd.std() / len(sd) ** 0.5) if len(ret) > 1 else None}


def market_report_md():
    """Tabla Markdown del informe frente a Thunderpick por esport (para ESTADISTICAS.md)."""
    out = ["| Esport | Partidos | Log-loss modelo | Mezcla del bot | Mercado | Apuestas | ROI (± error) |", "|---|---|---|---|---|---|---|"]
    for slug in [s for s, d in DISCIPLINES.items() if d[0] is not None]:  # FC 27 no tiene cuotas
        configure(slug)
        r = market_report()
        if not r["n"]: out.append(f"| {TITLE} | 0 | — | — | — | — | — |"); continue
        roi = f"{r['roi']:+.1%} ± {r['roi_err']:.1%}" if r["roi_err"] is not None else (f"{r['roi']:+.1%}" if r["roi"] is not None else "—")
        out.append(f"| {TITLE} | {r['n']} | {r['modelo']:.4f} | {r['mezcla']:.4f} (peso {r['blend']}) | {r['mercado']:.4f} | {r['apuestas']} | {roi} |")
    configure("cs2")
    return "\n".join(out)


PLAYER_SLUGS = ("cs2", "dota2", "lol")  # juegos con estadísticas por jugador en bo3.gg (R6 y MLBB no las publican)


def players_adjust(p, t1, t2):
    """S/A/B de CS2, Dota 2 y LoL: corrige la prob. de serie con la plantilla actual, H2H y química de cs2_players.py
    ([agente-grafo]). Walk-forward fuera de muestra frente a este Elo: CS2 0.6150 → 0.6089, Dota 2 0.6078 → 0.5906,
    LoL 0.6070 → 0.5866. Sin sus datos devuelve p tal cual."""
    try:
        import cs2_players
        return cs2_players.ajustar(p, t1, t2, slug=SLUG)
    except ModuleNotFoundError: return p  # cs2_players.py aún no está en el repo
    except Exception as e:
        print("cs2_players:", repr(e)); return p


def players_lines(g):
    """Líneas del grafo de jugadores para la tarjeta (forma de los jugadores, mapas y química); [] si no hay datos."""
    if SLUG not in PLAYER_SLUGS or g["tier"] not in ("s", "a", "b"): return []
    try:
        import cs2_players
        return [A.esc(l) for l in cs2_players.card_lines(g["t1_id"], g["t2_id"], g["team1"], g["team2"], slug=SLUG)]
    except Exception as e:
        print("cs2_players:", repr(e)); return []


def run(state, today, kg):
    for slug in DISCIPLINES:
        configure(slug)
        try:
            if DISC is None: __import__("esoccer").run(state, today, kg)
            else: run_one(state, today, kg)
        except Exception as e: print(slug, "error:", repr(e))  # un esport roto no tumba los demás
    configure("cs2")


def run_one(state, today, kg):
    df = update_history()
    print(f"{SLUG}: {len(df)} partidos de historial")
    settle(df, state)
    reg = load_regions() if REGION_K else {}
    R, n, tier, last, *_ = map_elo(games=load_games(), df=df[df.date >= pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=4)])  # el CS cambia rápido: 4 años
    now = pd.Timestamp.now(tz="UTC")
    try: up = get(**{"sort": "start_date", "filter[matches.status][in]": "upcoming"})["results"]
    except (requests.RequestException, ValueError) as e:
        print("bo3.gg", e); return

    def model(m):
        """(fila, Elo 1, Elo 2, prob. mapa, prob. serie) con lo sabido ahora; None si no hay equipos o es BO par."""
        g, start = row(m), pd.Timestamp(m["start_date"])
        if not g or g["bo"] % 2 == 0: return None  # BO2 admite empate: el Elo de serie no lo cubre
        r1, r2 = (decayed(R[t], (start - last[t]).days, TIER_START.get(tier[t], 1450)) if t in R else TIER_START.get(g["tier"], 1450)
                  for t in (g["t1_id"], g["t2_id"]))
        r1, r2 = r1 + G.get(reg.get(g["t1_id"]), 0.0), r2 + G.get(reg.get(g["t2_id"]), 0.0)  # nivel de su región
        p_map = map_prob(r1, r2, g["bo"])
        p = series_prob(p_map, g["bo"])
        if SLUG in PLAYER_SLUGS and g["tier"] in ("s", "a", "b"): p = players_adjust(p, g["t1_id"], g["t2_id"])
        return g, r1, r2, p_map, p

    save_odds(up, now, lambda m: round(x[4], 4) if (x := model(m)) else None)  # la prob. previa queda guardada para el informe
    for m in up:
        start, bet, key = pd.Timestamp(m["start_date"]), m.get("bet_updates") or {}, f"{SLUG}|{m['id']}"
        if not (now < start < now + pd.Timedelta(hours=36)) or key in state["analyzed"]: continue
        x = model(m)
        if not x: continue
        g, r1, r2, p_map, p = x
        hh = df[((df.t1_id == g["t1_id"]) & (df.t2_id == g["t2_id"])) | ((df.t1_id == g["t2_id"]) & (df.t2_id == g["t1_id"]))]
        h2h = f"{len(hh)}: {int((hh.winner_id == g['t1_id']).sum())}-{int((hh.winner_id == g['t2_id']).sum())}" if len(hh) else "sin precedentes"
        kg.match(TITLE, g["team1"], g["team2"], start, {"1": p, "2": 1 - p}, ([], []), h2h)
        state["analyzed"].append(key)
        o1, o2 = (bet.get(t, {}).get("coeff") for t in ("team_1", "team_2"))
        if not (o1 and o2 and o1 > 1 and o2 > 1): continue
        inv, cands = 1 / o1 + 1 / o2, []
        for side, q_model, o, tid, name in ((1, p, o1, g["t1_id"], g["team1"]), (2, 1 - p, o2, g["t2_id"], g["team2"])):
            q = BLEND * q_model + (1 - BLEND) / o / inv
            if q >= A.MIN_P:
                cands.append({"market": f"t{side}", "team_id": tid, "pick": name, "odds": o, "book": "Thunderpick",
                              "p": round(q, 3), "implied": round(1 / o / inv, 3), "edge": round(q * o - 1, 3)})
        vb = [v for v in cands if A.EDGE <= v["edge"] <= A.MAX_EDGE] or sorted(cands, key=lambda v: -v["edge"])[:1]
        if g["bo"] == 3: vb += extra_markets(bet, p_map, g)
        roster = [t for t, f in ((g["team1"], m.get("team1_new_participant")), (g["team2"], m.get("team2_new_participant"))) if f]
        info = [f"📊 <b>Escenarios</b> (Elo por mapa, {len(df):_} partidos)".replace("_", "."),
                f"Mapa: {A.esc(g['team1'])} {p_map:.0%} · Serie BO{g['bo']}: {A.esc(g['team1'])} {p:.0%} - {1 - p:.0%} {A.esc(g['team2'])}",
                f"Casas: {A.esc(g['team1'])} {1 / o1 / inv:.0%} · {A.esc(g['team2'])} {1 / o2 / inv:.0%}",
                "", "🔎 <b>Claves</b>",
                f"• Elo: {A.esc(g['team1'])} {r1:.0f} ({n.get(g['t1_id'], 0)} series) | {A.esc(g['team2'])} {r2:.0f} ({n.get(g['t2_id'], 0)} series)",
                f"• Forma: {A.esc(g['team1'])} {form(df, g['t1_id'])} | {A.esc(g['team2'])} {form(df, g['t2_id'])}",
                f"• H2H: {h2h}",
                f"• Torneo: {A.esc(g['tournament'])} (nivel {g['tier'].upper()})",
                f"• Cambios de plantilla: {A.esc(', '.join(roster)) if roster else 'ninguno'}", *players_lines(g),
                *(sgp_lines(bet, p_map, g) if g["bo"] == 3 else []),
                *(cs2_props.card_lines(g, p_map, bo3_scores(p_map)) if SLUG == "cs2" and g["tier"] in ("s", "a", "b") else [])]
        context = (f"{NAME} {g['team1']} vs {g['team2']}, BO{g['bo']}, torneo {g['tournament']} nivel {g['tier']}. Elo por mapa "
                   f"{r1:.0f} vs {r2:.0f} (prob. mapa {p_map:.0%}, serie {p:.0%}). Forma {form(df, g['t1_id'])} / {form(df, g['t2_id'])} "
                   f"(✅ victoria ❌ derrota). H2H {h2h}. Cambios de plantilla: {', '.join(roster) or 'ninguno'}. Cuotas Thunderpick {o1}/{o2}.")
        for v in vb:
            state["signals"].append({**v, "key": key, "match": f"{g['team1']} vs {g['team2']}", "start": start.isoformat(),
                                     "league": f"{TITLE} · {g['tournament']}", "sport": ICON, "info": info, "context": context})
