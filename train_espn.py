"""Entrena el Elo de ESPN (NBA, NFL, NHL, MLB): prueba K y ventaja local por liga y mide el log-loss de las 2 últimas
temporadas con la probabilidad calculada ANTES de cada partido. Guarda los mejores en data/espn_params.json."""
import itertools, json

import numpy as np
import pandas as pd

import multisport as M

GRID = {"K": [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48], "hfa": [0, 15, 25, 35, 50, 65, 80, 100, 120]}


def logloss(df, hfa, k, since):
    pred = []
    M.elo(df, hfa, k, pred)
    p, y, ev = np.clip(np.array(pred), 1e-6, 1 - 1e-6), df.res.to_numpy(float), (df.date >= since).to_numpy()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p))[ev].mean()), int(ev.sum())  # empates (0.5) cuentan medio


def main():
    out = {}
    for league in [l for l, v in M.SPORTS.items() if v[4]]:  # tenis y UFC usan K dinámico sin ventaja local
        d = M.DIR / league
        df = pd.concat([x for f in sorted(d.glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)
        since = df.date.max() - pd.DateOffset(years=2)
        old = logloss(df, M.SPORTS[league][3], M.SPORTS[league][4], since)
        res = {(k, h): logloss(df, h, k, since)[0] for k, h in itertools.product(GRID["K"], GRID["hfa"])}
        k, h = min(res, key=res.get)
        print(f"{league}: ANTES K={M.SPORTS[league][4]} ventaja {M.SPORTS[league][3]} log-loss {old[0]:.4f} → "
              f"MEJOR K={k} ventaja {h} log-loss {res[k, h]:.4f} ({old[1]} partidos)")
        out[league] = {"K": k, "hfa": h, "logloss": round(res[k, h], 4), "logloss_old": round(old[0], 4)}
    (M.A.DATA / "espn_params.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
