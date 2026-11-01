"""Counter-Strike 2 con la API pública de bo3.gg: ~77.000 partidos de historial, cuotas reales de Thunderpick
(casa de esports) y ganador del partido. Modelo: Elo por MAPA (cada mapa ganado cuenta) con peso por nivel del
torneo, rating inicial según el nivel, castigo por cambio de plantilla y conversión mapa → serie (BO1/BO3/BO5)."""
import json, math

import pandas as pd
import requests

import analyst as A

API = "https://api.bo3.gg/api/v1/matches"
DIR = A.DATA / "cs2"
HIST = DIR / "matches.csv"
COLS = ["id", "date", "team1", "team2", "t1_id", "t2_id", "s1", "s2", "winner_id", "bo", "tier", "tournament"]
TIER_K_BASE = {"s": 1.25, "a": 1.1, "b": 1.0, "c": 0.85, "d": 0.7}       # los torneos grandes enseñan más
TIER_START_BASE = {"s": 1650, "a": 1600, "b": 1520, "c": 1450, "d": 1400}  # rating de un equipo nuevo según dónde juega
ROSTER_SHRINK = 0.3  # cambio de plantilla: se acerca un 30 % a la media
_P = json.loads((DIR / "params.json").read_text()) if (DIR / "params.json").exists() else {}  # lo ajusta train_cs2.py
K = _P.get("K", 24)
TIER_K = {t: 1 + (v - 1) * _P.get("tier_scale", 1) for t, v in TIER_K_BASE.items()}
TIER_START = {t: 1500 + (v - 1500) * _P.get("start_spread", 1) for t, v in TIER_START_BASE.items()}
NEW_K = _P.get("new_k", 0)    # equipos nuevos: K más alto las primeras series
DECAY = _P.get("decay", 0)    # vuelta al rating inicial por inactividad (por año)
TITLE = "🎮 CS2"


def get(**params):
    p = {"page[limit]": 100, "filter[matches.discipline_id][eq]": 1, "with": "teams,tournament", **params}
    r = requests.get(API, params=p, headers=A.UA, timeout=60)
    r.raise_for_status()
    return r.json()


def row(m):
    t1, t2 = m.get("team1") or {}, m.get("team2") or {}
    if not (t1.get("name") and t2.get("name")): return None
    return {"id": m["id"], "date": m["start_date"], "team1": t1["name"], "team2": t2["name"], "t1_id": m["team1_id"],
            "t2_id": m["team2_id"], "s1": m.get("team1_score"), "s2": m.get("team2_score"), "winner_id": m.get("winner_team_id"),
            "bo": m.get("bo_type") or 1, "tier": m.get("tier") or "c", "tournament": (m.get("tournament") or {}).get("name", "")}


def update_history(max_pages=800):
    """Primera vez baja todo (~770 páginas); después solo lo nuevo hasta enlazar con lo guardado."""
    DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=COLS)
    known, rows = set(old.id), []
    for page in range(max_pages):
        try: res = get(**{"page[offset]": page * 100, "sort": "-start_date", "filter[matches.status][in]": "finished"})["results"]
        except (requests.RequestException, ValueError) as e:
            print("bo3.gg", e); break
        batch = [r for r in map(row, res) if r]
        rows += [r for r in batch if r["id"] not in known]
        if not res or any(r["id"] in known for r in batch): break
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True).drop_duplicates("id")
    df = df[df.winner_id.notna()]
    df.to_csv(HIST, index=False)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date", ignore_index=True)


def decayed(r, gap_days, start):
    """Tras >30 días sin jugar el rating vuelve poco a poco al inicial de su nivel (plantillas que se deshacen)."""
    if gap_days <= 30: return r
    return start + (r - start) * math.exp(-(gap_days - 30) / 365 * DECAY)


def map_elo(df, since=None):
    """Elo por mapa: una serie 2-1 son 3 actualizaciones; los equipos con pocas series aprenden más rápido (K×(1+NEW_K/(1+n))).
    Devuelve (ratings, series jugadas, último nivel, última fecha, log-loss de series desde `since` predichas antes de jugarse)."""
    R, n, tier, last, loss, cnt = {}, {}, {}, {}, 0.0, 0
    for r in df.itertuples():
        for t in (r.t1_id, r.t2_id):
            s0 = TIER_START.get(r.tier, 1450)
            R[t] = decayed(R[t], (r.date - last[t]).days, s0) if t in R else s0
            n.setdefault(t, 0); tier[t] = r.tier; last[t] = r.date
        if since is not None and r.date >= since:
            p = series_prob(1 / (1 + 10 ** ((R[r.t2_id] - R[r.t1_id]) / 400)), int(r.bo))
            loss -= math.log(min(max(p if r.winner_id == r.t1_id else 1 - p, 1e-6), 1)); cnt += 1
        s1, s2 = (int(r.s1), int(r.s2)) if pd.notna(r.s1) and pd.notna(r.s2) and r.s1 + r.s2 > 0 else             ((1, 0) if r.winner_id == r.t1_id else (0, 1))
        if r.bo == 1 and s1 + s2 > 1: s1, s2 = int(s1 > s2), int(s2 > s1)  # en BO1 el marcador son rondas, no mapas
        k = K * TIER_K.get(r.tier, 1.0)
        k1, k2 = k * (1 + NEW_K / (1 + n[r.t1_id])), k * (1 + NEW_K / (1 + n[r.t2_id]))
        for won1 in [1] * s1 + [0] * s2:
            e = 1 / (1 + 10 ** ((R[r.t2_id] - R[r.t1_id]) / 400))
            R[r.t1_id] += k1 * (won1 - e); R[r.t2_id] -= k2 * (won1 - e)
        n[r.t1_id] += 1; n[r.t2_id] += 1
    return R, n, tier, last, loss / max(cnt, 1), cnt


def series_prob(p, bo):
    """De probabilidad por mapa a probabilidad de ganar la serie."""
    need = bo // 2 + 1
    return sum(math.comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def form(df, tid, n=5):
    d = df[(df.t1_id == tid) | (df.t2_id == tid)].tail(n)
    return "".join("✅" if w == tid else "❌" for w in d.winner_id) or "—"


def settle(df, state):
    done = {f"cs2|{r['id']}": r for r in df.tail(5000).to_dict("records")}
    for s in state["signals"]:
        r = done.get(s["key"])
        if s.get("result") is None and r is not None:
            s["result"] = int(r["winner_id"]) == s["team_id"]
            s["score"] = f"{int(r['s1'])}-{int(r['s2'])}" if pd.notna(r["s1"]) else ""


def run(state, today, kg):
    df = update_history()
    print(f"cs2: {len(df)} partidos de historial")
    settle(df, state)
    R, n, tier, last, *_ = map_elo(df[df.date >= pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=4)])  # el CS cambia rápido: 4 años
    now = pd.Timestamp.now(tz="UTC")
    try: up = get(**{"sort": "start_date", "filter[matches.status][in]": "upcoming"})["results"]
    except (requests.RequestException, ValueError) as e:
        print("bo3.gg", e); return
    for m in up:
        start, bet, key = pd.Timestamp(m["start_date"]), m.get("bet_updates") or {}, f"cs2|{m['id']}"
        if not (now < start < now + pd.Timedelta(hours=36)) or key in state["analyzed"]: continue
        g = row(m)
        if not g: continue
        r1, r2 = (decayed(R[t], (start - last[t]).days, TIER_START.get(tier[t], 1450)) if t in R else TIER_START.get(g["tier"], 1450)
                  for t in (g["t1_id"], g["t2_id"]))
        if m.get("team1_new_participant"): r1 = 1500 + (1 - ROSTER_SHRINK) * (r1 - 1500)
        if m.get("team2_new_participant"): r2 = 1500 + (1 - ROSTER_SHRINK) * (r2 - 1500)
        p_map = 1 / (1 + 10 ** ((r2 - r1) / 400))
        p = series_prob(p_map, g["bo"])
        hh = df[((df.t1_id == g["t1_id"]) & (df.t2_id == g["t2_id"])) | ((df.t1_id == g["t2_id"]) & (df.t2_id == g["t1_id"]))]
        h2h = f"{len(hh)}: {int((hh.winner_id == g['t1_id']).sum())}-{int((hh.winner_id == g['t2_id']).sum())}" if len(hh) else "sin precedentes"
        kg.match(TITLE, g["team1"], g["team2"], start, {"1": p, "2": 1 - p}, ([], []), h2h)
        state["analyzed"].append(key)
        o1, o2 = (bet.get(t, {}).get("coeff") for t in ("team_1", "team_2"))
        if not (o1 and o2 and o1 > 1 and o2 > 1): continue
        inv, cands = 1 / o1 + 1 / o2, []
        for side, q_model, o, tid, name in ((1, p, o1, g["t1_id"], g["team1"]), (2, 1 - p, o2, g["t2_id"], g["team2"])):
            q = 0.5 * q_model + 0.5 / o / inv
            if q >= A.MIN_P:
                cands.append({"market": f"t{side}", "team_id": tid, "pick": name, "odds": o, "book": "Thunderpick",
                              "p": round(q, 3), "implied": round(1 / o / inv, 3), "edge": round(q * o - 1, 3)})
        vb = [v for v in cands if A.EDGE <= v["edge"] <= A.MAX_EDGE] or sorted(cands, key=lambda v: -v["edge"])[:1]
        roster = [t for t, f in ((g["team1"], m.get("team1_new_participant")), (g["team2"], m.get("team2_new_participant"))) if f]
        info = [f"📊 <b>Escenarios</b> (Elo por mapa, {len(df):,} partidos)".replace(",", "."),
                f"Mapa: {A.esc(g['team1'])} {p_map:.0%} · Serie BO{g['bo']}: {A.esc(g['team1'])} {p:.0%} - {1 - p:.0%} {A.esc(g['team2'])}",
                f"Casas: {A.esc(g['team1'])} {1 / o1 / inv:.0%} · {A.esc(g['team2'])} {1 / o2 / inv:.0%}",
                "", "🔎 <b>Claves</b>",
                f"• Elo: {A.esc(g['team1'])} {r1:.0f} ({n.get(g['t1_id'], 0)} series) | {A.esc(g['team2'])} {r2:.0f} ({n.get(g['t2_id'], 0)} series)",
                f"• Forma: {A.esc(g['team1'])} {form(df, g['t1_id'])} | {A.esc(g['team2'])} {form(df, g['t2_id'])}",
                f"• H2H: {h2h}",
                f"• Torneo: {A.esc(g['tournament'])} (nivel {g['tier'].upper()})",
                f"• Cambios de plantilla: {A.esc(', '.join(roster)) if roster else 'ninguno'}"]
        context = (f"CS2 {g['team1']} vs {g['team2']}, BO{g['bo']}, torneo {g['tournament']} nivel {g['tier']}. Elo por mapa "
                   f"{r1:.0f} vs {r2:.0f} (prob. mapa {p_map:.0%}, serie {p:.0%}). Forma {form(df, g['t1_id'])} / {form(df, g['t2_id'])} "
                   f"(✅ victoria ❌ derrota). H2H {h2h}. Cambios de plantilla: {', '.join(roster) or 'ninguno'}. Cuotas Thunderpick {o1}/{o2}.")
        for v in vb:
            state["signals"].append({**v, "key": key, "match": f"{g['team1']} vs {g['team2']}", "start": start.isoformat(),
                                     "league": f"{TITLE} · {g['tournament']}", "sport": "🎮", "info": info, "context": context})
