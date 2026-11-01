"""Elo de tenis ATP/WTA por superficie (deducida del nombre del torneo) mezclado con el general, y K mayor en
Grand Slams, como en la literatura de tennis Elo (FiveThirtyEight / Tennis Abstract). Usa ufc_model.elo.
`python tennis_model.py`: elige parámetros con los 2 años anteriores al test y mide en las 2 últimas temporadas;
guarda data/tennis_params.json. predict(league, df) → prob. local de cada fila con esos parámetros."""
import itertools, json
from pathlib import Path

import numpy as np
import pandas as pd

import ufc_model as U

PARAMS = Path(__file__).parent / "data" / "tennis_params.json"
CLAY = ["roland garros", "french open", "madrid", "italia", "rome", "monte-carlo", "monte carlo", "barcelona", "estoril",
        "hassan", "chile", "clay court", "umag", "croatia", "cordoba", "geneva", "bmw open", "munich", "gstaad", "hamburg",
        "tiriac", "bucharest", "argentina open", "rio open", "generali open", "kitzb", "nordea", "bastad", "auvergne", "lyon",
        "charleston", "porsche", "stuttgart open", "istanbul", "bogota", "colsanitas", "rabat", "palermo", "strasbourg",
        "parma", "budapest", "lausanne", "warsaw", "iasi", "portoroz", "bol open", "marrakech", "houston", "cagliari",
        "sardegna", "belgrade", "serbia open", "parma", "copa colsanitas", "ladies open"]
GRASS = ["wimbledon", "eastbourne", "queen", "cinch", "hsbc championships", "terra wortmann", "noventi",
         "hertogenbosch", "libema", "lib�ma", "libéma", "mallorca", "boss open", "stuttgart", "nottingham", "rothesay",
         "birmingham", "berlin", "bett1", "ecotrans", "bad homburg", "aegon", "nature valley", "newport", "hall of fame",
         "infosys", "surbiton", "ilkley", "grass"]
SLAMS = ["australian open", "roland garros", "french open", "wimbledon", "us open"]


def surface(event, date):
    e = str(event).lower()
    if "olympic" in e: return "clay" if date.year == 2024 else "hard"  # París 2024 en tierra; Tokio 2021 en dura
    if any(k in e for k in GRASS) and "stuttgart open" not in e: return "grass"
    if any(k in e for k in CLAY): return "clay"
    return "hard"  # shortcut: dura por defecto (incluye pista cubierta); añadir torneos si se ven fallos


def predict(league, df, params=None):
    c = (params or json.loads(PARAMS.read_text()))[league]
    surf = pd.Series([surface(e, d) for e, d in zip(df.event, df.date)], index=df.index)
    slam = df.event.str.lower().str.contains("|".join(SLAMS), regex=True).map({True: c["kslam"], False: 1.0})
    return U.elo(df, c["kt"], c["pw"], 0, c["idle"], c["keep"], surf, c["mix"], slam)[0]


def main():
    import multisport as M
    out = {}
    for league in ("atp", "wta"):
        df = U.load(league)
        end = df.date.max()
        y, ev = df.res.to_numpy(float), (df.date >= end - pd.DateOffset(years=2)).to_numpy()
        val = ~ev & (df.date >= end - pd.DateOffset(years=4)).to_numpy()
        old = []
        M.elo(df, 0, None, old, kt=M._P.get(league, {}).get("kt", 250))
        base = U.logloss(np.array(old), y, ev)
        surf = pd.Series([surface(e, d) for e, d in zip(df.event, df.date)], index=df.index)
        print(f"{league}: superficies {surf.value_counts().to_dict()}")
        res = {}
        for kt, pw, mix, kslam, idle in itertools.product([150, 250, 350], [0.3, 0.4, 0.5], [0, 0.3, 0.5, 0.7], [1.0, 1.2], [180, 10 ** 6]):
            c = {"kt": kt, "pw": pw, "mix": mix, "kslam": kslam, "idle": idle, "keep": 0.8}
            res[tuple(c.items())] = U.logloss(predict(league, df, {league: c}), y, val).mean()
        c = dict(min(res, key=res.get))
        d = U.logloss(predict(league, df, {league: c}), y, ev) - base
        nosurf = dict(min((k for k in res if dict(k)["mix"] == 0), key=res.get))
        d0 = U.logloss(predict(league, df, {league: nosurf}), y, ev) - base
        print(f"{league} base log-loss {base.mean():.4f} ({ev.sum()} partidos) → mejor {c}: {base.mean() + d.mean():.4f} "
              f"({d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}); sin superficie: {d0.mean():+.4f}")
        out[league] = {**c, "logloss": round(float(base.mean() + d.mean()), 4), "logloss_old": round(float(base.mean()), 4),
                       "se": round(float(d.std() / np.sqrt(len(d))), 4)}
    PARAMS.write_text(json.dumps(out))


if __name__ == "__main__":
    assert surface("Roland Garros", pd.Timestamp("2025-06-01")) == "clay" and surface("Wimbledon", pd.Timestamp("2025-07-01")) == "grass"
    assert surface("US Open", pd.Timestamp("2025-09-01")) == "hard" and surface("Olympics", pd.Timestamp("2024-08-01")) == "clay"
    main()
