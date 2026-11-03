"""Backtest decisivo frente al mercado: con las cuotas de CIERRE de DraftKings guardadas por nba_players.download
(data/<liga>/box.jsonl, ESPN las tiene desde ~nov 2025) y el modelo de producción de multisport (Elo, modelo de
puntos para totales, p_cover para hándicap, soccer_probs en fútbol), mide por liga y mercado: log-loss modelo vs
mercado sin margen, peso óptimo del modelo elegido en la 1.ª mitad y medido en la 2.ª, y ROI de la regla del bot
(mezcla 50/50, ventaja 5-30 %, prob >= 25 %) con su error. `python market_check.py [ligas]` → data/market_check.json."""
import json, sys

import numpy as np
import pandas as pd

import multisport as M
import nba_players as N
import totals_model

OUT = M.A.DATA / "market_check.json"
W = np.round(np.arange(0, 1.01, 0.1), 1)


def dec(a, default=None):
    return M.american(a) if a not in (None, "") and not pd.isna(a) else default


def history(league):
    if league in M._X:
        import espn_extra as E
        df = pd.concat([E.load(l) for l in (E.SOCCER if league in M.SOCCER else [league])])
    else:
        df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not f.name.startswith("odds")
                        and not (x := pd.read_csv(f, dtype={"id": str})).empty])
        df["date"] = pd.to_datetime(df.date, utc=True)
    return df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)


def rows(league):
    """Una fila por (partido, mercado) con probs. del modelo por lado, cuotas decimales y resultado por lado (1/0/0.5 push)."""
    df = history(league)
    _, _, _, hfa, k = M.SPORTS[league]
    c = M._P.get(league, {})
    pred = []
    M.elo(df, hfa, k, pred, c.get("keep", 0.75), c.get("kt", 250), c.get("mov", False))
    pe = dict(zip(df.id, pred))
    box = [g for g in N.load_box(league) if g.get("odds") and g["odds"].get("ml_h") and g["id"] in pe]
    pts_cache, out = {}, []
    for g in box:
        o, p, hs, as_ = g["odds"], pe[g["id"]], float(g["hs"]), float(g["as"])
        date = pd.Timestamp(g["date"])
        if league in M.SOCCER:
            import espn_extra as E
            P3 = E.soccer_probs(np.array([p]), M._P[league])[0]
            res = [hs > as_, hs == as_, hs < as_]
            out.append({"date": date, "mk": "1X2", "q": list(P3), "odds": [dec(o["ml_h"]), dec(o.get("ml_d")), dec(o["ml_a"])], "res": res})
        else:
            out.append({"date": date, "mk": "ganador", "q": [p, 1 - p], "odds": [dec(o["ml_h"]), dec(o["ml_a"])], "res": [hs > as_, hs < as_]})
        if o.get("total") and league not in M.SOCCER and (league not in M._X or league in M.TOT):  # como run_league
            wk = date.normalize() - pd.Timedelta(days=date.dayofweek)
            if wk not in pts_cache:
                pts_cache[wk] = M.points_model(df[df.date < wk], league, wk.tz_localize(None))
            po, _ = M.over_prob(pts_cache[wk], g["home"], g["away"], float(o["total"]), league)
            t = hs + as_
            push = t == float(o["total"])
            out.append({"date": date, "mk": "totales", "q": [po, 1 - po], "odds": [dec(o.get("over"), 1.91), dec(o.get("under"), 1.91)],
                        "res": [0.5 if push else t > float(o["total"]), 0.5 if push else t < float(o["total"])]})
        if o.get("spread") is not None and league in M.SPREAD:
            sl = float(o["spread"])
            pc = totals_model.p_cover(league, p, sl, {"spread": M.SPREAD})
            m = hs - as_ + sl
            out.append({"date": date, "mk": "hándicap", "q": [pc, 1 - pc], "odds": [dec(o.get("sp_h"), 1.91), dec(o.get("sp_a"), 1.91)],
                        "res": [0.5 if m == 0 else m > 0, 0.5 if m == 0 else m < 0]})
    return pd.DataFrame(out)


def evaluate(R):
    """Métricas de un mercado (R = filas de rows() de ese mercado)."""
    R = R[R.odds.map(lambda x: all(v and v > 1 for v in x))].sort_values("date", ignore_index=True)
    q, od = np.array(R.q.tolist(), float), np.array(R.odds.tolist(), float)
    res = np.array(R.res.tolist(), float)
    mk = 1 / od; mk /= mk.sum(1, keepdims=True)
    nopush = ~(res == 0.5).any(1)
    win = res.argmax(1)
    ll = lambda P, m: -np.log(np.clip(P[np.arange(len(P)), win], 1e-9, 1))[m]
    half = np.arange(len(R)) < len(R) // 2
    lw = {w: ll(w * q + (1 - w) * mk, half & nopush).mean() for w in W}
    w_opt = min(lw, key=lw.get)
    def roi(w, m):
        p = w * q + (1 - w) * mk
        ev = p * od - 1
        bet = (ev >= M.A.EDGE) & (ev <= M.A.MAX_EDGE) & (p >= M.A.MIN_P) & m[:, None]
        g = np.where(res == 1, od - 1, np.where(res == 0.5, 0.0, -1.0))[bet]
        return (float(g.mean()) if len(g) else 0.0), (float(g.std() / np.sqrt(len(g))) if len(g) > 1 else 1.0), int(len(g))
    d = ll(q, ~half & nopush) - ll(mk, ~half & nopush)
    return {"n": int(len(R)), "ll_modelo": round(float(ll(q, nopush).mean()), 4), "ll_mercado": round(float(ll(mk, nopush).mean()), 4),
            "w_opt": float(w_opt), "ll_2a_mitad_w_opt": round(float(ll(w_opt * q + (1 - w_opt) * mk, ~half & nopush).mean()), 4),
            "ll_2a_mitad_mercado": round(float(ll(mk, ~half & nopush).mean()), 4),
            "modelo_menos_mercado_2a": [round(float(d.mean()), 4), round(float(d.std() / np.sqrt(max(len(d), 1))), 4)],
            "roi_bot_w05": [round(x, 4) for x in roi(0.5, np.ones(len(R), bool))],
            "roi_2a_mitad_w_opt": [round(x, 4) for x in roi(w_opt, ~half)]}


def main(leagues):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for league in leagues:
        if not N.box_path(league).exists(): print(league, "sin data/box.jsonl"); continue
        R = rows(league)
        if R.empty: print(league, "sin cuotas"); continue
        out[league] = {}
        for mk, g in R.groupby("mk"):
            e = out[league][mk] = evaluate(g)
            r, se, n = e["roi_bot_w05"]
            print(f"{league:4s} {mk:9s} n={e['n']:5d} log-loss modelo {e['ll_modelo']:.4f} mercado {e['ll_mercado']:.4f} | "
                  f"peso óptimo {e['w_opt']} (2.ª mitad {e['ll_2a_mitad_w_opt']:.4f} vs {e['ll_2a_mitad_mercado']:.4f}) | "
                  f"ROI bot {r:+.1%} ± {se:.1%} en {n}; ROI 2.ª mitad con peso óptimo {e['roi_2a_mitad_w_opt'][0]:+.1%} ± {e['roi_2a_mitad_w_opt'][1]:.1%} en {e['roi_2a_mitad_w_opt'][2]}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nba", "nhl", "nfl", "mlb", "cfb", "cbb", "ucl", "uel"])
