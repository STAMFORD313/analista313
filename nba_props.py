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


# ── liquidación de las props de Unibet y comparación modelo vs casa (experto «props_nba» de nutricion.py) ─────────
import math, re

import pandas as pd

LOG = A.DATA / "nba" / "props_log.csv"
STAT_RX = [("triples", re.compile(r"3-?point|three", re.I)), ("rebotes", re.compile(r"rebound", re.I)),
           ("asistencias", re.compile(r"assist", re.I)), ("puntos", re.compile(r"^points|points scored", re.I))]


def stat_of(label):
    l = label.lower()
    if any(x in l for x in ("+", " and ", "&", ",", "double", "spread", "moneyline", "total")) or l[:1].isdigit():
        return None  # combinados (pts & reb…), dobles-dobles, «6+ rebotes» y mercados de equipo: sin modelo aún
    return next((s for s, rx in STAT_RX if rx.search(label)), None)


def norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    if "," in s: s = " ".join(reversed([x.strip() for x in s.split(",", 1)]))  # «James, LeBron» → «lebron james»
    return re.sub(r"[^a-z ]", "", s).strip()


def settle():
    """Props de Unibet (data/nba/kambi_props.csv, más/menos) × lo que hizo el jugador (box.jsonl) × prob. previa del
    modelo (ritmo y minutos ANTES del partido) → data/nba/props_log.csv."""
    f = A.DATA / "nba" / "kambi_props.csv"
    P = params()
    if not f.exists() or not P: return pd.DataFrame()
    k = pd.read_csv(f)
    k = k[k.tipo.isin(["OT_OVER", "OT_UNDER"]) & k.linea.notna()].assign(stat=lambda x: x.mercado.map(stat_of)).dropna(subset=["stat"])
    if k.empty: return pd.DataFrame()
    k = k.sort_values("seen").drop_duplicates(["start", "jugador", "stat", "linea", "tipo"], keep="last")
    d = T.rows()
    fw = T.features(d, P["a_min"], P["a_rate"], P["k"])
    d = d.assign(**fw, key=d.name.map(norm), day=d.date.dt.strftime("%Y-%m-%d"))
    idx = {}
    for r in d.itertuples(index=False): idx[(r.day, r.key)] = r
    out = []
    for (start, jug, stat, line), g in k.groupby(["start", "jugador", "stat", "linea"]):
        day = pd.Timestamp(start).tz_convert("UTC")
        x = next((idx.get((dd.strftime("%Y-%m-%d"), norm(jug))) for dd in (day, day - pd.Timedelta(days=1)) if idx.get((dd.strftime("%Y-%m-%d"), norm(jug)))), None)
        if x is None: continue  # aún sin jugar o no jugó (la casa anula)
        mu, r = getattr(x, f"r_{stat}") * x.e_min, P["stats"][stat]["r"]
        p_over = float(1 - stats.nbinom.cdf(math.floor(line), r, r / (r + max(mu, 1e-6))))
        o = dict(zip(g.tipo, g.cuota))
        oo, ou = o.get("OT_OVER"), o.get("OT_UNDER")
        p_casa = (1 / oo) / (1 / oo + 1 / ou) if oo and ou else (1 / oo if oo else None)
        val = getattr(x, stat)
        out.append({"fecha": str(start)[:10], "jugador": jug, "stat": stat, "linea": line, "cuota_mas": oo, "cuota_menos": ou,
                    "p_casa_mas": round(p_casa, 4) if p_casa else None, "p_modelo_mas": round(p_over, 4), "esperado": round(mu, 2),
                    "valor": val, "mas": int(val > line)})
    res = pd.DataFrame(out)
    if not res.empty: res.to_csv(LOG, index=False)
    return res


def report(min_edge=0.05):
    """Por estadística: n, log-loss modelo vs casa, y ROI apostando el lado con ventaja ≥ min_edge (1 u)."""
    if not LOG.exists(): return []
    d = pd.read_csv(LOG).dropna(subset=["p_casa_mas"])
    lines = []
    for s, g in d.groupby("stat"):
        y = g.mas.to_numpy(); pm = g.p_modelo_mas.clip(1e-4, 1 - 1e-4).to_numpy(); pc = g.p_casa_mas.clip(1e-4, 1 - 1e-4).to_numpy()
        ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        eo, eu = pm * g.cuota_mas - 1, (1 - pm) * g.cuota_menos - 1
        side_o = eo >= eu; ev = np.where(side_o, eo, eu); bet = ev >= min_edge
        gain = np.where(side_o, np.where(y == 1, g.cuota_mas - 1, -1), np.where(y == 0, g.cuota_menos - 1, -1))[bet]
        lines.append(f"{s} ({len(g)}): log-loss modelo {ll(pm):.3f} vs casa {ll(pc):.3f}"
                     + (f" · con ventaja {bet.sum()}: ROI {gain.mean():+.0%}" if bet.sum() else ""))
    return lines


if __name__ == "__main__":
    print(f"props NBA liquidadas: {len(settle())}")
    for l in report(): print(" ", l)
