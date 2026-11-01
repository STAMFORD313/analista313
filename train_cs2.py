"""Entrena el modelo de CS2: prueba combinaciones de K, peso por nivel y rating inicial, y mide el log-loss
sobre el último año (predicción hecha ANTES de cada partido). Guarda los mejores en data/cs2/params.json."""
import itertools, json, math, sys, time

import pandas as pd

import esports as E

GAMES = E.DIR / "games.csv"   # un mapa por fila: rondas y ganador
META = E.DIR / "meta.csv"     # por partido: cambios de plantilla, cuotas guardadas por bo3.gg, ámbito/premio del torneo


def fetch_games(years=3, pause=0.3):
    """Baja los mapas (with=games) de los últimos `years` años, 100 partidos por página con una pausa corta."""
    since, games, meta = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years), [], []
    for page in range(2000):
        try: res = E.get(**{"page[offset]": page * 100, "sort": "-start_date", "filter[matches.status][in]": "finished",
                            "with": "games,tournament"})["results"]
        except Exception as e:
            print("bo3.gg", e); time.sleep(5); continue
        for m in res:
            b, t = m.get("bet_updates") or {}, m.get("tournament") or {}
            meta.append({"id": m["id"], "new1": m.get("team1_new_participant"), "new2": m.get("team2_new_participant"),
                         "o1": (b.get("team_1") or {}).get("coeff"), "o2": (b.get("team_2") or {}).get("coeff"),
                         "scope": t.get("event_scope"), "prize": t.get("prize")})
            for g in m.get("games") or []:
                w = (g.get("winner_team_clan") or {}).get("team_id")
                if w and g.get("status") == "finished":
                    games.append({"match_id": m["id"], "n": g.get("number"), "map": g.get("map_name"), "winner_id": w,
                                  "ws": g.get("winner_clan_score"), "ls": g.get("loser_clan_score")})
        if not res or pd.Timestamp(res[-1]["start_date"]) < since: break
        if page % 20 == 0: print(page, res[-1]["start_date"], len(games), flush=True)
        time.sleep(pause)
    pd.DataFrame(games).drop_duplicates(["match_id", "n"]).to_csv(GAMES, index=False)
    pd.DataFrame(meta).drop_duplicates("id").to_csv(META, index=False)


def main():
    df = pd.read_csv(E.HIST)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")
    df = df[df.date >= df.date.max() - pd.DateOffset(years=4)]
    since = df.date.max() - pd.DateOffset(years=1)
    best = None
    for k, nk, dc in itertools.product([12, 14, 16, 20], [0, 4, 6, 8], [0, 0.5, 1, 2]):
        E.K, E.NEW_K, E.DECAY = k, nk, dc
        ll, n = E.map_elo(df, since)[4:]
        print(f"K={k} nuevos×{nk} inactividad×{dc}: log-loss {ll:.4f} ({n} partidos)", flush=True)
        if best is None or ll < best[0]: best = (ll, {"K": k, "tier_scale": 0, "start_spread": 1, "new_k": nk, "decay": dc})
    base = -(math.log(0.5))
    print(f"MEJOR {best[1]} log-loss {best[0]:.4f} (azar = {base:.4f})")
    (E.DIR / "params.json").write_text(json.dumps({**best[1], "logloss": round(best[0], 4)}))


if __name__ == "__main__":
    fetch_games() if "fetch" in sys.argv else main()
