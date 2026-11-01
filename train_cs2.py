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
    o["p"], o["y"] = o.id.map(lambda i: preds[i][0]), (o.winner_id == o.t1_id).astype(int)
    o["m"] = (1 / o.o1) / (1 / o.o1 + 1 / o.o2)
    ll = lambda p: -(o.y * p.clip(1e-6, 1).map(math.log) + (1 - o.y) * (1 - p).clip(1e-6, 1).map(math.log)).mean()
    for name, q in (("modelo", o.p), ("mezcla 50/50 (la del bot)", 0.5 * o.p + 0.5 * o.m)):
        bets = [(e1, o.o1[i] - 1 if o.y[i] else -1) if 0.05 <= e1 <= 0.30 else (e2, o.o2[i] - 1 if not o.y[i] else -1)
                for i, e1, e2 in zip(o.index, q * o.o1 - 1, (1 - q) * o.o2 - 1) if 0.05 <= e1 <= 0.30 or 0.05 <= e2 <= 0.30]
        roi = sum(b[1] for b in bets) / len(bets) if bets else float("nan")
        print(f"vs Thunderpick ({len(o)} partidos) {name}: log-loss {ll(q):.4f} vs mercado {ll(o.m):.4f}; "
              f"ROI con ventaja 5-30 %: {roi:+.1%} en {len(bets)} apuestas")


def fit_maps3(df, s1, s2, games):
    """Prob. de ir al 3er mapa en BO3: logística sobre logit(2p(1-p)) del Elo. Ajusta en el año -2, mide en el -1 y guarda
    el ajuste con los dos años. Devuelve (a, b) y lo imprime frente a mapas independientes y a la tasa base."""
    import numpy as np
    preds = {}
    E.map_elo(df, s1, games, preds)
    d = df[df.id.isin(preds) & (df.bo == 3) & (df[["s1", "s2"]].max(axis=1) == 2) & (df.s1 + df.s2).isin([2, 3])]
    pm = d.id.map(lambda i: preds[i][1]).to_numpy()
    x = np.log(2 * pm * (1 - pm) / (1 - 2 * pm * (1 - pm)))
    X, y, b = np.column_stack([np.ones_like(x), x]), (d.s1 + d.s2 == 3).to_numpy(float), (d.date >= s2).to_numpy()

    def fit(m):
        w = np.array([0.0, 1.0])
        for _ in range(30):
            p = 1 / (1 + np.exp(-X[m] @ w))
            w -= np.linalg.solve(X[m].T @ (X[m] * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2), X[m].T @ (p - y[m]))
        return w
    ll = lambda w, m: float(-np.mean(y[m] * np.log(1 / (1 + np.exp(-X[m] @ w))) + (1 - y[m]) * np.log(1 - 1 / (1 + np.exp(-X[m] @ w)))))
    base = np.log(y[~b].mean() / (1 - y[~b].mean()))
    print(f"BO3 ¿3er mapa? año-1 ({b.sum()}): independientes {ll(np.array([0, 1.0]), b):.4f} · tasa base {ll(np.array([base, 0]), b):.4f}"
          f" · recalibrado en año-2 {ll(fit(~b), b):.4f}")
    w = fit(np.ones_like(b))
    return [round(float(w[0]), 4), round(float(w[1]), 4)]


NAMES = {"K": "K", "new_k": "NEW_K", "decay": "DECAY", "k_bo1": "K_BO1", "series_scale": "SERIES_SCALE", "mov": "MOV"}


def main():
    """Walk-forward: cada combinación se evalúa en el año -2 (A) y en el -1 (B) en una sola pasada causal.
    Fuera de muestra = la mejor de A medida en B y viceversa; se guarda la mejor en la media de A y B (la más robusta)."""
    df = pd.read_csv(E.HIST)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")
    df = df[df.date >= df.date.max() - pd.DateOffset(years=4)]
    s1, s2 = df.date.max() - pd.DateOffset(years=2), df.date.max() - pd.DateOffset(years=1)
    y = dict(zip(df.id, df.winner_id == df.t1_id))
    in_b = set(df.id[df.date >= s2])
    games, res = E.load_games(), []
    for k, nk, dc, kb, sc, mv in itertools.product([16, 20, 24, 28], [4, 8], [0.5, 1], [1, 1.5, 2], [0.65, 0.75, 0.85], [1, 1.5, 2]):
        P = {"K": k, "new_k": nk, "decay": dc, "k_bo1": kb, "series_scale": sc, "mov": mv}
        for n, v in P.items(): setattr(E, NAMES[n], v)
        preds = {}
        E.map_elo(df, s1, games, preds)
        ll = {"A": [], "B": []}
        for i, (p, _) in preds.items(): ll["B" if i in in_b else "A"].append(-math.log(min(max(p if y[i] else 1 - p, 1e-6), 1)))
        a, b = (sum(v) / len(v) for v in (ll["A"], ll["B"]))
        res.append((a, b, P))
        print(f"{P}: año-2 {a:.4f} año-1 {b:.4f}", flush=True)
    bestA, bestB, best = min(res, key=lambda r: r[0]), min(res, key=lambda r: r[1]), min(res, key=lambda r: r[0] + r[1])
    print(f"Fuera de muestra: ajustado en año-2 → año-1 {bestA[1]:.4f} (en muestra {bestB[1]:.4f}); "
          f"ajustado en año-1 → año-2 {bestB[0]:.4f} (en muestra {bestA[0]:.4f})")
    print(f"MEJOR robusto {best[2]}: año-2 {best[0]:.4f} año-1 {best[1]:.4f} (azar = {math.log(2):.4f})")
    (E.DIR / "params.json").write_text(json.dumps({**best[2], "tier_scale": 0, "start_spread": 1, "logloss": round(best[1], 4),
                                                   "logloss_prev_year": round(best[0], 4), "oos_next_year": round(bestA[1], 4)}))
    for n, v in best[2].items(): setattr(E, NAMES[n], v)
    maps3 = fit_maps3(df, s1, s2, games)
    (E.DIR / "params.json").write_text(json.dumps({**json.loads((E.DIR / "params.json").read_text()), "maps3": maps3}))
    vs_market(df, s2, games)


if __name__ == "__main__":
    if "fetch" in sys.argv: fetch_games()
    elif "mapas" in sys.argv:  # solo recalibra el 3er mapa con los parámetros guardados
        d = pd.read_csv(E.HIST); d["date"] = pd.to_datetime(d["date"], utc=True); d = d.sort_values("date")
        d = d[d.date >= d.date.max() - pd.DateOffset(years=4)]
        m3 = fit_maps3(d, d.date.max() - pd.DateOffset(years=2), d.date.max() - pd.DateOffset(years=1), E.load_games())
        (E.DIR / "params.json").write_text(json.dumps({**json.loads((E.DIR / "params.json").read_text()), "maps3": m3})); print(m3)
    elif "mercado" in sys.argv:  # solo la comparación con Thunderpick, con los parámetros guardados
        d = pd.read_csv(E.HIST); d["date"] = pd.to_datetime(d["date"], utc=True); d = d.sort_values("date")
        vs_market(d[d.date >= d.date.max() - pd.DateOffset(years=4)], d.date.max() - pd.DateOffset(years=1), E.load_games())
    else: main()
