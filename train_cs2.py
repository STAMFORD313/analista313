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


def vs_market(df, since, games):
    """Modelo vs cuotas PREVIAS de Thunderpick (data/cs2/odds.csv): log-loss y ROI plano con ventaja 5-30 %."""
    if not E.ODDS.exists(): return
    preds = {}
    E.map_elo(df, since, games, preds)
    o = pd.read_csv(E.ODDS).merge(df[["id", "t1_id", "winner_id"]], on="id")
    o = o[o.id.isin(preds) & (o.o1 > 1) & (o.o2 > 1)]
    if o.empty: return print("vs Thunderpick: aún no hay partidos terminados con cuota previa")
    o["p"], o["y"] = o.id.map(preds), (o.winner_id == o.t1_id).astype(int)
    o["m"] = (1 / o.o1) / (1 / o.o1 + 1 / o.o2)
    ll = lambda p: -(o.y * p.clip(1e-6, 1).map(math.log) + (1 - o.y) * (1 - p).clip(1e-6, 1).map(math.log)).mean()
    for name, q in (("modelo", o.p), ("mezcla 50/50 (la del bot)", 0.5 * o.p + 0.5 * o.m)):
        bets = [(e1, o.o1[i] - 1 if o.y[i] else -1) if 0.05 <= e1 <= 0.30 else (e2, o.o2[i] - 1 if not o.y[i] else -1)
                for i, e1, e2 in zip(o.index, q * o.o1 - 1, (1 - q) * o.o2 - 1) if 0.05 <= e1 <= 0.30 or 0.05 <= e2 <= 0.30]
        roi = sum(b[1] for b in bets) / len(bets) if bets else float("nan")
        print(f"vs Thunderpick ({len(o)} partidos) {name}: log-loss {ll(q):.4f} vs mercado {ll(o.m):.4f}; "
              f"ROI con ventaja 5-30 %: {roi:+.1%} en {len(bets)} apuestas")


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
    for k, v in best[1].items(): setattr(E, {"K": "K", "new_k": "NEW_K", "decay": "DECAY", "k_bo1": "K_BO1", "series_scale": "SERIES_SCALE", "mov": "MOV"}.get(k, "_"), v)
    vs_market(df, since, games)


if __name__ == "__main__":
    if "fetch" in sys.argv: fetch_games()
    elif "mercado" in sys.argv:  # solo la comparación con Thunderpick, con los parámetros guardados
        d = pd.read_csv(E.HIST); d["date"] = pd.to_datetime(d["date"], utc=True); d = d.sort_values("date")
        vs_market(d[d.date >= d.date.max() - pd.DateOffset(years=4)], d.date.max() - pd.DateOffset(years=1), E.load_games())
    else: main()
