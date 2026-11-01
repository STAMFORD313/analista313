"""Esports (CS2, LoL, Dota 2, Valorant) con PandaScore (plan gratis, token en PANDASCORE_TOKEN):
historial de partidos, Elo por equipo, próximos partidos e IA. El plan gratis no trae cuotas: no hay señales de valor."""
import os

import pandas as pd
import requests

import analyst as A
import multisport as M

TOKEN = os.getenv("PANDASCORE_TOKEN")
GAMES = {"csgo": "🎮 CS2", "lol": "🎮 League of Legends", "dota2": "🎮 Dota 2", "valorant": "🎮 Valorant"}
MAX_PAGES = 60  # primera carga: ~6000 partidos por juego; después solo lo nuevo
DIR = A.DATA / "esports"
COLS = ["id", "date", "event", "home", "away", "hs", "as", "res"]


def get(path, **params):
    r = requests.get(f"https://api.pandascore.co/{path}", params=params,
                     headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def row(m):
    ops = [o["opponent"] for o in m.get("opponents", []) if o.get("opponent")]
    if len(ops) != 2: return None
    sc = {r["team_id"]: r["score"] for r in m.get("results", [])}
    wid = m.get("winner_id")
    return {"id": str(m["id"]), "date": m.get("begin_at") or m.get("scheduled_at"),
            "event": f"{(m.get('league') or {}).get('name', '')} {(m.get('serie') or {}).get('full_name', '')}".strip(),
            "home": ops[0]["name"], "away": ops[1]["name"], "hs": sc.get(ops[0]["id"]), "as": sc.get(ops[1]["id"]),
            "res": None if wid is None else 1.0 if wid == ops[0]["id"] else 0.0}


def update_history(game):
    DIR.mkdir(parents=True, exist_ok=True)
    f = DIR / f"{game}.csv"
    old = pd.read_csv(f, dtype={"id": str}) if f.exists() else pd.DataFrame(columns=COLS)
    known, rows = set(old.id), []
    for page in range(1, MAX_PAGES + 1):
        batch = [r for r in map(row, get(f"{game}/matches/past", sort="-begin_at", per_page=100, page=page)) if r]
        rows += [r for r in batch if r["id"] not in known and r["res"] is not None and r["date"]]
        if not batch or any(r["id"] in known for r in batch): break  # ya enlazamos con lo guardado
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True).drop_duplicates("id")
    df.to_csv(f, index=False)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date", ignore_index=True)


def run_game(game, state, today):
    title = GAMES[game]
    df = update_history(game)
    print(f"{game}: {len(df)} partidos de historial")
    R = M.elo(df, 0, 32)
    end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=A.HORIZON_DAYS)
    new = [g for g in map(row, get(f"{game}/matches/upcoming", sort="begin_at", per_page=100)) if g and g["date"]
           and pd.Timestamp.now(tz="UTC") < pd.Timestamp(g["date"]) <= end and "TBD" not in (g["home"], g["away"]) and M.key(game, g) not in state["analyzed"]]
    if not new: return
    lines = [f"{title} · PRÓXIMOS PARTIDOS (Elo con {len(df)} partidos)"]
    for g in new:
        p = M.win_prob(R, g["home"], g["away"], 0)
        when = pd.Timestamp(g["date"]).tz_convert("Europe/Madrid")
        lines.append(f"{when:%d/%m %H:%M} {g['home']} vs {g['away']} · {p:.0%}-{1 - p:.0%} · "
                     f"forma {M.form(df, g['home'])}/{M.form(df, g['away'])} · {g['event']}")
        state["analyzed"].append(M.key(game, g))
    ai = A.ai_json(f"""Eres un analista experto de esports ({title}). Próximos partidos con probabilidad de mi modelo Elo,
forma reciente (G/P) y torneo:
{chr(10).join(lines[1:60])}
Responde SOLO con un JSON válido: {{"analysis": "en español: los 3-5 partidos más interesantes con tu pronóstico y por qué (máx. 12 frases)"}}""")
    if ai and ai.get("analysis"): lines += ["", "🤖 " + ai["analysis"]]
    A.send_text("\n".join(lines))


def run(state, today):
    if not TOKEN: return print("esports: falta PANDASCORE_TOKEN")
    for game in GAMES:
        try: run_game(game, state, today)
        except Exception as e:  # un juego caído no debe parar los demás
            print("fallo en", game, repr(e))
