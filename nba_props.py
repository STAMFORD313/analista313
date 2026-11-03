"""Props de jugador NBA para las tarjetas: puntos, rebotes, asistencias y triples esperados de los jugadores con más
minutos de cada equipo (sin los lesionados de ESPN), con rango del 80 %. Parámetros de train_nba_props.py
(data/nba/props_params.json). Solo referencia: no hay cuotas de props guardadas para medir ROI frente a una casa."""
import json

import numpy as np
from scipy import stats

import analyst as A
import nba_players as N
import train_nba_props as T

_state = {}


def params():
    return json.loads(T.OUT.read_text()) if T.OUT.exists() else None


def load():
    """{equipo: [(nombre, minutos esperados, {stat: ritmo por minuto})]} con el estado actual de cada jugador."""
    if "teams" in _state: return _state["teams"]
    P = params(); d = T.rows()
    f = T.features(d, P["a_min"], P["a_rate"], P["k"])  # estado ANTES de cada partido: se actualiza con el último
    d = d.assign(**f)
    last = d.groupby("pid").tail(1)
    out = {}
    for r in last.itertuples(index=False):
        n = r.n + 1
        if n < 10: continue
        a = P["a_min"]; e_min = r.e_min + a * (r.min - r.e_min)
        rates = {s: getattr(r, f"r_{s}") for s in T.STATS}  # shortcut: ritmo previo al último partido (casi igual)
        out.setdefault(r.team, []).append((r.name, e_min, rates, r.date))
    recent = d.date.max() - np.timedelta64(30, "D")
    _state["teams"] = {t: sorted([x for x in v if x[3] >= recent], key=lambda x: -x[1]) for t, v in out.items()}
    return _state["teams"]


def card_lines(home, away, injured=(), top=4):
    P = params()
    if not P: return []
    try: teams = load()
    except Exception as e: print("props NBA", repr(e)); return []
    out = ["", "🏀 <b>Props esperados</b> (pts · reb · ast · triples; rango 80 %; mejor que la media de sus 10 últimos: "
               f"log-loss puntos {P['stats']['puntos']['ou_ref']}→{P['stats']['puntos']['ou']}; solo referencia)"]
    bad = {n.lower() for n in injured}
    for team in (home, away):
        pl = [x for x in teams.get(team, []) if x[0].lower() not in bad][:top]
        for name, m, rates, _ in pl:
            parts = []
            for s, short in (("puntos", "pts"), ("rebotes", "reb"), ("asistencias", "ast"), ("triples", "3p")):
                mu, r = rates[s] * m, P["stats"][s]["r"]
                dd = stats.nbinom(r, r / (r + max(mu, 1e-6)))
                parts.append(f"{mu:.1f} ({int(dd.ppf(0.1))}-{int(dd.ppf(0.9))}) {short}")
            out.append(f"{A.esc(name)} ({A.esc(team.split()[-1])}, {m:.0f} min): " + " · ".join(parts))
    return out if len(out) > 2 else []
