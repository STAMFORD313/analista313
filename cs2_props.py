"""Props de jugador de CS2 para las tarjetas: kills esperados en la serie de cada jugador de la última alineación de cada
equipo, con rango del 80 % (binomial negativa). Parámetros de train_cs2_props.py (data/cs2/props_params.json). Solo
candidata: no hay cuotas de props guardadas para medir ROI frente a una casa."""
import json

import numpy as np
import pandas as pd
from scipy import stats

import analyst as A

DIR = A.DATA / "cs2"
_state = {}


def params():
    f = DIR / "props_params.json"
    return json.loads(f.read_text()) if f.exists() else None


def load():
    """{player_id: (kills, rondas) con olvido}, KPR global, {team_id: [(player_id, nick)] de su última serie}."""
    if "kpr" in _state: return _state["kpr"]
    P = params()
    pm, g = pd.read_csv(DIR / "players_matches.csv"), pd.read_csv(DIR / "players_games.csv")
    rounds = (g.ws + g.ls).groupby(g.match_id).sum()
    date = pd.to_datetime(g.drop_duplicates("match_id").set_index("match_id").date, utc=True)
    pm = pm[(pm.kills >= 0) & pm.match_id.isin(rounds.index)].assign(rounds=lambda x: x.match_id.map(rounds), date=lambda x: x.match_id.map(date))
    pm = pm.sort_values(["date", "match_id"])
    a, S = 0.5 ** (1 / P["half_life"]), {}
    for pid, kl, rd in zip(pm.player_id, pm.kills, pm.rounds):
        K, R = S.get(pid, (0.0, 0.0)); S[pid] = (K * a + kl, R * a + rd)
    glob = pm.kills.sum() / pm.rounds.sum()
    last = pm.groupby("team_id").match_id.max()
    cur = pm[pm.match_id == pm.team_id.map(last)]
    lineup = {t: list(zip(x.player_id, x.nick)) for t, x in cur.groupby("team_id")}
    _state["kpr"] = (S, glob, lineup, P)
    return _state["kpr"]


def expected(t1, t2, p_map, bo, scores=None):
    """[(equipo 1|2, nick, kills esperados, p10, p90)] o [] sin parámetros o sin alineación conocida."""
    if not params() or bo not in (1, 3): return []
    S, glob, lineup, P = load()
    e_maps = 1.0 if bo == 1 else 2 + scores[2, 1] + scores[1, 2]
    out = []
    for side, team, p in ((1, t1, p_map), (2, t2, 1 - p_map)):
        p = min(max(p, 1e-3), 1 - 1e-3)
        for pid, nick in lineup.get(team, [])[:5]:
            K, R = S.get(pid, (0.0, 0.0))
            mu = (K + P["k"] * glob) / (R + P["k"]) * e_maps * P["rounds_per_map"] * np.exp(P["c"] * np.log(p / (1 - p)))
            r = P["r"]; d = stats.nbinom(r, r / (r + mu))
            out.append((side, nick, float(mu), int(d.ppf(0.1)), int(d.ppf(0.9))))
    return out


def card_lines(g, p_map, scores=None):
    """Líneas de tarjeta (texto plano) con los kills esperados de cada jugador."""
    try: ex = expected(g["t1_id"], g["t2_id"], p_map, g["bo"], scores)
    except Exception as e: print("props CS2", repr(e)); return []
    if not ex: return []
    P = params()
    n = f"{P['n_test']:,}".replace(",", ".")
    lines = ["", f"🔫 <b>Kills esperados en la serie</b> (rango 80 %; mejor que la media de sus 10 últimas en {n} casos: "
                 f"log-loss {P['ou_ref']}→{P['ou']}; sin cuotas de props, solo referencia)"]
    for side, name in ((1, g["team1"]), (2, g["team2"])):
        pl = sorted((x for x in ex if x[0] == side), key=lambda x: -x[2])
        if pl: lines.append(f"{A.esc(name)}: " + " · ".join(f"{A.esc(n)} {m:.0f} ({lo}-{hi})" for _, n, m, lo, hi in pl))
    return lines if len(lines) > 2 else []
