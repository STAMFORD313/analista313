"""NFL con nflverse (games.csv: resultados, cuotas de cierre y QB titular desde 1999; CC-BY 4.0): Elo con margen y
ajuste por cambio de QB frente a la cuota de cierre, walk-forward. Mezcla y ventaja elegidas en 2012-2019, medidas en
2020-2025 (log-loss y ROI a la cuota de cierre). Uso: python train_nfl.py [games.csv] → data/nfl/nflverse.json"""
import json, math, sys

import numpy as np
import pandas as pd

import analyst as A

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT = A.DATA / "nfl" / "nflverse.json"


def dec(ml):
    return np.where(ml > 0, 1 + ml / 100, 1 + 100 / np.abs(ml))


def elo(d, k=20, hfa=48, qb_pen=0.0, keep=0.67):
    R, last_season, qbs, p = {}, {}, {}, np.zeros(len(d))
    for i, r in enumerate(d.itertuples()):
        for t in (r.home_team, r.away_team):
            if last_season.get(t) not in (None, r.season): R[t] = 1500 + (R[t] - 1500) * keep
            last_season[t] = r.season; R.setdefault(t, 1500)
        adj = 0.0
        for t, q, s in ((r.home_team, r.home_qb_id, 1), (r.away_team, r.away_qb_id, -1)):
            usual = pd.Series(qbs.get(t, [])[-6:]).mode()
            if len(usual) and q != usual.iloc[0]: adj -= s * qb_pen  # QB distinto del habitual
        diff = R[r.home_team] - R[r.away_team] + hfa + adj
        p[i] = 1 / (1 + 10 ** (-diff / 400))
        m = r.home_score - r.away_score
        if m != 0:
            mov = math.log(abs(m) + 1) * 2.2 / ((abs(diff) * 0.001) + 2.2)
            dlt = k * mov * ((m > 0) - p[i]); R[r.home_team] += dlt; R[r.away_team] -= dlt
        for t, q in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)): qbs.setdefault(t, []).append(q)
    return p


def main(path=None):
    d = pd.read_csv(path or URL)
    d = d[d.home_score.notna()].sort_values(["gameday", "game_id"], ignore_index=True)
    ok = d.home_moneyline.notna() & d.away_moneyline.notna() & (d.home_score != d.away_score)
    oh, oa = dec(d.home_moneyline.fillna(-110).to_numpy()), dec(d.away_moneyline.fillna(-110).to_numpy())
    mk = (1 / oh) / (1 / oh + 1 / oa); y = (d.home_score > d.away_score).to_numpy(float)
    fit, test = (ok & d.season.between(2012, 2019)).to_numpy(), (ok & (d.season >= 2020)).to_numpy()
    ll = lambda p, m: -(y[m] * np.log(np.clip(p[m], 1e-6, 1)) + (1 - y[m]) * np.log(np.clip(1 - p[m], 1e-6, 1)))
    best = None
    for qb in (0, 30, 60, 90):
        p = elo(d, qb_pen=qb)
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            q = w * p + (1 - w) * mk
            s = ll(q, fit).mean()
            if best is None or s < best[0]: best = (s, qb, w, p)
    _, qb, w, p = best
    p0 = elo(d, qb_pen=0)
    q = w * p + (1 - w) * mk
    se = lambda u, v: float((u - v).std() / math.sqrt(len(u)))
    l_mk, l_q, l_elo, l_elo0 = ll(mk, test), ll(q, test), ll(p, test), ll(p0, test)
    print(f"NFL nflverse: ajuste 2012-2019 ({fit.sum()}), prueba 2020+ ({test.sum()}); elegido QB −{qb} Elo, peso modelo {w}")
    print(f"  Elo sin QB {l_elo0.mean():.4f} · con QB {l_elo.mean():.4f} ({l_elo.mean() - l_elo0.mean():+.4f} ± {se(l_elo, l_elo0):.4f}) · "
          f"mercado {l_mk.mean():.4f} · mezcla {l_q.mean():.4f} ({l_q.mean() - l_mk.mean():+.4f} ± {se(l_q, l_mk):.4f})")
    res = {"qb_pen": qb, "w": w, "n_test": int(test.sum()), "ll_mercado": round(float(l_mk.mean()), 4), "ll_mezcla": round(float(l_q.mean()), 4),
           "se": round(se(l_q, l_mk), 4), "ll_elo_qb": round(float(l_elo.mean()), 4), "ll_elo": round(float(l_elo0.mean()), 4)}
    for edge in (0.03, 0.05, 0.08):
        eh, ea = q * oh - 1, (1 - q) * oa - 1
        side = eh >= ea; ev = np.where(side, eh, ea); bet = test & (ev >= edge) & (ev <= 0.3)
        g = np.where(side, np.where(y == 1, oh - 1, -1), np.where(y == 0, oa - 1, -1))[bet]
        print(f"  ROI 2020+ con ventaja ≥ {edge:.0%}: {g.mean():+.1%} ± {g.std() / math.sqrt(max(len(g), 1)):.1%} en {len(g)}")
        res[f"roi_{edge}"] = [round(float(g.mean()), 4), round(float(g.std() / math.sqrt(max(len(g), 1))), 4), int(len(g))]
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
