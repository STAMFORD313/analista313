"""Entrena el Elo de ESPN: en NBA, NFL, NHL y MLB prueba K, ventaja local y cuánto rating se conserva entre temporadas;
en tenis y UFC la escala del K dinámico. Mide el log-loss de las 2 últimas temporadas con la probabilidad calculada
ANTES de cada partido y guarda los mejores en data/espn_params.json (lo lee multisport.py)."""
import itertools, json

import numpy as np
import pandas as pd

import multisport as M

BASE = {"nba": (100, 20), "nfl": (55, 20), "nhl": (50, 8), "mlb": (24, 4)}  # (ventaja, K) de partida, para comparar
GRID = {"K": [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48], "hfa": [0, 15, 25, 35, 50, 65, 80, 100, 120],
        "keep": [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0], "kt": [100, 150, 200, 250, 300, 400]}


def logloss(df, hfa, k, since, keep=0.75, kt=250):
    pred = []
    M.elo(df, hfa, k, pred, keep, kt)
    p, y, ev = np.clip(np.array(pred), 1e-6, 1 - 1e-6), df.res.to_numpy(float), (df.date >= since).to_numpy()
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p))[ev].mean()), int(ev.sum())  # empates (0.5) cuentan medio


def main():
    out = {}
    for league, v in M.SPORTS.items():
        d = M.DIR / league
        df = pd.concat([x for f in sorted(d.glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)
        since = df.date.max() - pd.DateOffset(years=2)
        if v[4]:  # deportes de equipo
            old = logloss(df, *BASE[league], since)
            res = {(k, h, kp): logloss(df, h, k, since, kp)[0] for k, h, kp in itertools.product(GRID["K"], GRID["hfa"], GRID["keep"])}
            k, h, kp = min(res, key=res.get)
            out[league] = {"K": k, "hfa": h, "keep": kp, "logloss": round(res[k, h, kp], 4), "logloss_old": round(old[0], 4)}
        else:  # tenis y UFC: K dinámico sin ventaja local
            old = logloss(df, 0, None, since)
            res = {kt: logloss(df, 0, None, since, kt=kt)[0] for kt in GRID["kt"]}
            kt = min(res, key=res.get)
            out[league] = {"K": None, "hfa": 0, "kt": kt, "logloss": round(res[kt], 4), "logloss_old": round(old[0], 4)}
        print(league, out[league], f"({old[1]} partidos)")
    (M.A.DATA / "espn_params.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
