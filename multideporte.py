"""Piernas «gana X» de todos los deportes del bot para el chat (combinadas y simuladas), con las probabilidades que ya
calculan sus modelos entrenados (no se recalcula nada aquí) y su liquidación con los resultados que guarda el bot.

Fuentes (las escribe el run de analista cada 30 min):
  NBA/NHL/NFL/MLB  data/espn/odds_<liga>.csv (prob. local del Elo + cuotas DraftKings) · resultados data/espn/<liga>/AAAAMM.csv
  esports          data/grafo.json (partidos con prob. de serie) + data/<juego>/odds.csv (Thunderpick) · resultados data/<juego>/matches.csv
  FC 27            data/fc/predicciones.csv (1X2 recalibrado + cuota Unibet) · resultados data/fc/matches.csv

Calibración de los favoritos medida en el historial (walk-forward, últimos años; «dice → pasa»): NBA 87→87 %, CS2 87→87 %,
Dota 2 82→86 %, LoL 92→87 %, R6 87→79 %, MLBB 92→85 %, FC 27 83→76 % (n 17). OFFSET resta lo que se pasa de optimista."""
import json

import numpy as np
import pandas as pd

import analyst as A

US = {"nba": "🏀 NBA", "nhl": "🏒 NHL", "nfl": "🏈 NFL", "mlb": "⚾ MLB"}
ESPORTS = {"cs2": "🎮 CS2", "lol": "🧙 LoL", "dota2": "🐉 Dota 2", "r6": "🔫 R6", "mlbb": "📱 MLBB"}
OFFSET = {"lol": 0.04, "r6": 0.08, "mlbb": 0.05, "fc": 0.06, "nhl": 0.0, "nfl": 0.0, "mlb": 0.02}
_TEAMS = {}


def _latest(path):
    try: d = pd.read_csv(path, dtype={"id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError): return pd.DataFrame()
    return d.sort_values("seen").groupby("id").last().reset_index() if "seen" in d else d


def esports_teams(slug):
    if slug not in _TEAMS:
        try: m = pd.read_csv(A.DATA / slug / "matches.csv", usecols=["team1", "team2"]); _TEAMS[slug] = set(m.team1) | set(m.team2)
        except (FileNotFoundError, ValueError): _TEAMS[slug] = set()
    return _TEAMS[slug]


def legs(hours=48, sports=None):
    """[{src, sport, home, away, start, side, txt, p, odds}] de los partidos de las próximas `hours` horas: el favorito de
    cada partido (p ya corregida por OFFSET) y su cuota si la hay. sports: lista de claves (nba, cs2, fc…) o None = todas."""
    now = pd.Timestamp.now(tz="UTC"); end = now + pd.Timedelta(hours=hours); out = []
    want = lambda s: sports is None or s in sports
    for lg, label in US.items():
        if not want(lg): continue
        d = _latest(A.DATA / "espn" / f"odds_{lg}.csv")
        for r in d.itertuples():
            t = pd.Timestamp(r.start)
            if not (now < t < end) or pd.isna(r.p): continue
            home_fav = r.p >= 0.5
            p = (r.p if home_fav else 1 - r.p) - OFFSET.get(lg, 0)
            odds = (r.oh if home_fav else r.oa) if hasattr(r, "oh") else None
            out.append(dict(src=f"espn:{lg}", sport=label, home=r.home, away=r.away, start=str(t), side="H" if home_fav else "A",
                            txt=f"Gana {r.home if home_fav else r.away}", p=round(float(p), 4),
                            odds=float(odds) if odds and not pd.isna(odds) else None))
    try: g = json.loads((A.DATA / "grafo.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError): g = {"nodes": []}
    odds_es = {s: _latest(A.DATA / s / "odds.csv") for s in ESPORTS if want(s)}
    for n in g["nodes"]:
        if n.get("type") != "partido" or "p_X" in n: continue
        t = pd.Timestamp(n["start"])
        if not (now < t < end): continue
        h, a = n["id"].rsplit(" ", 1)[0].split(" vs ", 1)
        slug = next((s for s in odds_es if h in esports_teams(s) and a in esports_teams(s)), None)
        if not slug: continue
        p1 = n["p_1"]; fav1 = p1 >= 0.5
        odds = None
        o = odds_es[slug]
        if len(o):
            m = o[(pd.to_datetime(o.start, utc=True) == t) & ((o.p - p1).abs() < 0.002)]
            if len(m): odds = float(m.iloc[0].o1 if fav1 else m.iloc[0].o2)
        out.append(dict(src=f"esports:{slug}", sport=ESPORTS[slug], home=h, away=a, start=str(t), side="H" if fav1 else "A",
                        txt=f"Gana {h if fav1 else a}", p=round((p1 if fav1 else 1 - p1) - OFFSET.get(slug, 0), 4),
                        odds=odds if odds and odds > 1 else None))
    if want("fc"):
        try: f = pd.read_csv(A.DATA / "fc" / "predicciones.csv", dtype={"id": str})
        except FileNotFoundError: f = pd.DataFrame()
        for r in f.drop_duplicates("id", keep="last").itertuples():
            t = pd.Timestamp(r.start)
            if not (now < t < end) or pd.isna(r.p_pick): continue
            pick = str(r.pick)
            txt = "Empate" if pick == "X" else f"Gana {r.p1 if pick == '1' else r.p2}"
            out.append(dict(src="fc", sport="🎮⚽ FC 27", home=r.p1, away=r.p2, start=str(t), side={"1": "H", "2": "A"}.get(pick, "X"),
                            txt=txt, p=round(float(r.p_pick) - OFFSET["fc"], 4),
                            odds=float(r.odds) if getattr(r, "book", "") == "Unibet" and not pd.isna(r.odds) else None))
    return out


# ── liquidación ─────────────────────────────────────────────────────────────────────────────────────────────────
def _result(leg):
    """(goles/puntos local, visitante) del partido terminado, o None."""
    src, after = leg["src"], pd.Timestamp(leg["start"]) - pd.Timedelta(hours=6)
    if src.startswith("espn:"):
        lg = src.split(":")[1]
        fs = sorted((A.DATA / "espn" / lg).glob("*.csv"))[-2:]
        if not fs: return None
        d = pd.concat([pd.read_csv(f) for f in fs])
        d = d[(d.home == leg["home"]) & (d.away == leg["away"]) & (pd.to_datetime(d.date, utc=True) >= after)].dropna(subset=["hs", "as"])
        return (float(d.iloc[0].hs), float(d.iloc[0]["as"])) if len(d) else None
    if src.startswith("esports:"):
        d = pd.read_csv(A.DATA / src.split(":")[1] / "matches.csv")
        d = d[(pd.to_datetime(d.date, utc=True) >= after) & (((d.team1 == leg["home"]) & (d.team2 == leg["away"])) |
                                                            ((d.team1 == leg["away"]) & (d.team2 == leg["home"])))].dropna(subset=["winner_id"])
        if not len(d): return None
        r = d.iloc[0]; w1 = r.winner_id == r.t1_id
        s1, s2 = (r.s1, r.s2) if not (pd.isna(r.s1) or pd.isna(r.s2)) else (float(w1), float(not w1))
        if (s1 > s2) != w1: s1, s2 = float(w1), float(not w1)
        return (s1, s2) if r.team1 == leg["home"] else (s2, s1)
    if src == "fc":
        d = pd.read_csv(A.DATA / "fc" / "matches.csv")
        d = d[(d.p1 == leg["home"]) & (d.p2 == leg["away"]) & (pd.to_datetime(d.date, utc=True) >= after)].dropna(subset=["s1", "s2"])
        return (float(d.iloc[0].s1), float(d.iloc[0].s2)) if len(d) else None
    return None


def judge(leg):
    """(acierto True/False o None si aún no hay resultado, porqué)."""
    r = _result(leg)
    if r is None: return None, "sin resultado todavía"
    hs, as_ = r
    ok = hs > as_ if leg["side"] == "H" else as_ > hs if leg["side"] == "A" else hs == as_
    return ok, f"acabó {hs:g}-{as_:g}"
