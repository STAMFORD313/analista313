"""Entrena el modelo de CS2: prueba combinaciones de K, peso por nivel y rating inicial, y mide el log-loss
sobre el último año (predicción hecha ANTES de cada partido). Guarda los mejores en data/cs2/params.json."""
import itertools, json, math, sys, time

import pandas as pd

import esports as E


def fetch_games(years=3, pause=0.3):
    """Baja los mapas (with=games) de los últimos `years` años, 100 partidos por página con una pausa corta (~1 h)."""
    since, games = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years), []
    for page in range(2000):
        try: res = E.get(**{"page[offset]": page * 100, "sort": "-start_date", "filter[matches.status][in]": "finished",
                            "with": "games"})["results"]
        except Exception as e:
            print("bo3.gg", e); time.sleep(5); continue
        games += [g for m in res for g in E.game_rows(m)]
        if not res or pd.Timestamp(res[-1]["start_date"]) < since: break
        if page % 20 == 0: print(page, res[-1]["start_date"], len(games), flush=True)
        time.sleep(pause)
    pd.DataFrame(games).drop_duplicates(["match_id", "n"]).to_csv(E.GAMES, index=False)


def main():
    df = pd.read_csv(E.HIST)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")
    df = df[df.date >= df.date.max() - pd.DateOffset(years=4)]
    since = df.date.max() - pd.DateOffset(years=1)
    best, games = None, E.load_games()
    for k, nk, kb, sc, mv in itertools.product([16, 20, 24], [6, 8], [1.5, 2], [0.65, 0.7, 0.8], [1, 1.5, 2]):
        E.K, E.NEW_K, E.DECAY, E.K_BO1, E.SERIES_SCALE, E.MOV = k, nk, 0.5, kb, sc, mv
        ll, n = E.map_elo(df, since, games)[4:]
        print(f"K={k} nuevos×{nk} BO1×{kb} escala serie {sc} margen^{mv}: log-loss {ll:.4f} ({n} partidos)", flush=True)
        if best is None or ll < best[0]:
            best = (ll, {"K": k, "tier_scale": 0, "start_spread": 1, "new_k": nk, "decay": 0.5, "k_bo1": kb, "series_scale": sc, "mov": mv})
    base = -(math.log(0.5))
    print(f"MEJOR {best[1]} log-loss {best[0]:.4f} (azar = {base:.4f})")
    (E.DIR / "params.json").write_text(json.dumps({**best[1], "logloss": round(best[0], 4)}))


if __name__ == "__main__":
    fetch_games() if "fetch" in sys.argv else main()
