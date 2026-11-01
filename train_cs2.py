"""Entrena el modelo de CS2: prueba combinaciones de K, peso por nivel y rating inicial, y mide el log-loss
sobre el último año (predicción hecha ANTES de cada partido). Guarda los mejores en data/cs2/params.json."""
import itertools, json, math

import pandas as pd

import esports as E


def logloss(df, k, tier_scale, start_spread, since):
    tier_k = {t: 1 + (v - 1) * tier_scale for t, v in E.TIER_K_BASE.items()}
    start = {t: 1500 + (v - 1500) * start_spread for t, v in E.TIER_START_BASE.items()}
    R, loss, n = {}, 0.0, 0
    for r in df.itertuples():
        for t in (r.t1_id, r.t2_id): R.setdefault(t, start.get(r.tier, 1450))
        if r.date >= since:  # se evalúa con lo que el modelo sabía antes del partido
            p = E.series_prob(1 / (1 + 10 ** ((R[r.t2_id] - R[r.t1_id]) / 400)), int(r.bo))
            y = r.winner_id == r.t1_id
            loss -= math.log(min(max(p if y else 1 - p, 1e-6), 1)); n += 1
        s1, s2 = (int(r.s1), int(r.s2)) if pd.notna(r.s1) and pd.notna(r.s2) and r.s1 + r.s2 > 0 else \
            ((1, 0) if r.winner_id == r.t1_id else (0, 1))
        if r.bo == 1 and s1 + s2 > 1: s1, s2 = int(s1 > s2), int(s2 > s1)
        kk = k * tier_k.get(r.tier, 1.0)
        for won1 in [1] * s1 + [0] * s2:
            e = 1 / (1 + 10 ** ((R[r.t2_id] - R[r.t1_id]) / 400))
            R[r.t1_id] += kk * (won1 - e); R[r.t2_id] -= kk * (won1 - e)
    return loss / max(n, 1), n


def main():
    df = pd.read_csv(E.HIST)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")
    df = df[df.date >= df.date.max() - pd.DateOffset(years=4)]
    since = df.date.max() - pd.DateOffset(years=1)
    best = None
    for k, ts, ss in itertools.product([12, 16, 20, 24, 32, 40], [0, 1, 2], [0, 1, 1.5]):
        ll, n = logloss(df, k, ts, ss, since)
        print(f"K={k} nivel×{ts} inicio×{ss}: log-loss {ll:.4f} ({n} partidos)")
        if best is None or ll < best[0]: best = (ll, {"K": k, "tier_scale": ts, "start_spread": ss})
    base = -(math.log(0.5))
    print(f"MEJOR {best[1]} log-loss {best[0]:.4f} (azar = {base:.4f})")
    (E.DIR / "params.json").write_text(json.dumps({**best[1], "logloss": round(best[0], 4)}))


if __name__ == "__main__":
    main()
