"""Props de jugador de fútbol con los datos por jugador de ESPN (data/espn_soccer, espn_soccer.py) y liquidación de las
cuotas de Unibet guardadas (data/kambi_player_props.csv).

Modelo por jugador, solo con partidos anteriores: por aparición (titular = 1, suplente = SUB_W de un partido) ritmo con
olvido de goles, tiros a puerta, asistencias y tarjetas, encogido hacia la media de su posición (portero, defensa,
medio, delantero) con K apariciones. P(marca) = 1 − e^(−λ), P(tiros a puerta > línea) Poisson, P(tarjeta) = 1 − e^(−λ).
`python football_players.py` → backtest walk-forward (2.ª mitad de los datos) frente a la frecuencia del propio
jugador, y liquidación de las props de Unibet → data/player_props_log.csv (prob. del modelo, prob. de la casa, acierto)."""
import json, math, re, unicodedata

import numpy as np
import pandas as pd

import analyst as A
import espn_soccer as ES

LOG = A.DATA / "player_props_log.csv"
SUB_W, ALPHA, K = 0.35, 0.05, 6.0
STAT_IX = {"goles": 5, "asist": 6, "tiros": 7, "tiros_puerta": 8, "faltas": 11, "amarillas": 12, "rojas": 13}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def group(pos):
    p = str(pos).upper()
    return "G" if p.startswith("G") else "D" if p.startswith(("CD", "LB", "RB", "D", "WB", "SW")) else \
        "F" if p.startswith(("F", "CF", "ST", "LW", "RW", "S")) else "M"


def rows():
    out = []
    for code in ES.LEAGUES:
        for g in ES.load(code):
            for p in g["players"]:
                out.append({"date": g["date"], "game": g["id"], "league": code, "pid": p[0], "name": p[1],
                            "team": g["home"] if p[2] == "h" else g["away"], "starter": bool(p[3]), "pos": group(p[4]),
                            **{k: float(p[i]) for k, i in STAT_IX.items()}})
    d = pd.DataFrame(out)
    if d.empty: return d
    d["date"] = pd.to_datetime(d.date, utc=True); d["cards"] = d.amarillas + d.rojas
    return d.sort_values(["date", "game"], ignore_index=True)


STATS = ["goles", "asist", "tiros_puerta", "cards"]


def walk(d):
    """Por fila, ritmo previo (por partido como titular) de cada estadística, con olvido y encogido a su posición."""
    S, P = {}, {}
    out = {s: np.zeros(len(d)) for s in STATS}; out["n"] = np.zeros(len(d))
    for i, r in enumerate(d.itertuples(index=False)):
        w = 1.0 if r.starter else SUB_W
        st, ps = S.get(r.pid), P.setdefault(r.pos, {s: [0.0, 0.0] for s in STATS})
        out["n"][i] = st["n"] if st else 0
        for s in STATS:
            prior = ps[s][0] / ps[s][1] if ps[s][1] else 0.05
            x, e = st[s] if st else (0.0, 0.0)
            out[s][i] = (x + K * prior) / (e + K)  # por partido completo como titular
        if st is None: st = S[r.pid] = {"n": 0, **{s: (0.0, 0.0) for s in STATS}}
        st["n"] += 1
        for s in STATS:
            v = getattr(r, s); x, e = st[s]
            st[s] = (x * (1 - ALPHA) + v, e * (1 - ALPHA) + w)
            ps[s][0] += v; ps[s][1] += w
    return pd.DataFrame(out, index=d.index), S, P


def backtest():
    d = rows()
    f, _, _ = walk(d)
    te = (d.date >= d.date.quantile(0.5)).to_numpy() & (f.n >= 3).to_numpy() & d.starter.to_numpy()
    print(f"ESPN fútbol: {len(d)} jugador-partido, {d.game.nunique()} partidos; test (2.ª mitad, titulares con ≥3 previos): {te.sum()}")
    res = {}
    for s, name in (("goles", "marca"), ("tiros_puerta", "≥1 tiro a puerta"), ("cards", "tarjeta"), ("asist", "asistencia")):
        y = (d[s].to_numpy() >= 1)[te]
        p = np.clip(1 - np.exp(-f[s].to_numpy()[te]), 1e-4, 1 - 1e-4)
        base = np.clip(y.mean(), 1e-4, 1 - 1e-4)  # referencia: frecuencia media de los titulares
        ll = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q))
        l1, l0 = ll(p), ll(np.full(len(y), base))
        se = float((l1 - l0).std() / math.sqrt(len(y)))
        res[s] = (round(float(l0.mean()), 4), round(float(l1.mean()), 4), round(se, 4))
        print(f"  {name}: log-loss frecuencia media {l0.mean():.4f} → modelo por jugador {l1.mean():.4f} ({l1.mean() - l0.mean():+.4f} ± {se:.4f}); "
              f"prob. media {p.mean():.3f} vs real {y.mean():.3f}")
    return res


MARKET = {"To Score": ("goles", 1), "To score at least 2 goals": ("goles", 2), "To score at least 3 goals": ("goles", 3),
          "To give an assist": ("asist", 1), "To Get a Card": ("cards", 1), "Player's shots on target": ("tiros_puerta", None)}


def prob(rate, k):
    """P(X ≥ k) con X ~ Poisson(rate)."""
    return float(1 - sum(math.exp(-rate) * rate ** j / math.factorial(j) for j in range(int(k))))


def settle():
    """Cruza las cuotas de Unibet (última vista antes del partido) con lo que hizo el jugador según ESPN y con la prob.
    del modelo calculada solo con partidos anteriores. Reescribe data/player_props_log.csv. Devuelve el DataFrame."""
    f = A.DATA / "kambi_player_props.csv"
    if not f.exists(): return pd.DataFrame()
    k = pd.read_csv(f)
    k = k[k.mercado.isin(MARKET)].sort_values("seen").drop_duplicates(["start", "home", "jugador", "mercado", "linea"], keep="last")
    d = rows()
    if d.empty or k.empty: return pd.DataFrame()
    fw, _, _ = walk(d)
    d = d.assign(**{f"r_{s}": fw[s] for s in STATS}, key=d.name.map(norm), day=d.date.dt.strftime("%Y-%m-%d"))
    idx = {(r.day, r.key): r for r in d.itertuples(index=False)}
    out = []
    for r in k.itertuples(index=False):
        day = pd.Timestamp(r.start).tz_convert("UTC").strftime("%Y-%m-%d")
        x = idx.get((day, norm(r.jugador)))
        if x is None: continue  # no jugó (la casa anula) o ESPN aún no tiene el partido
        stat, n = MARKET[r.mercado]
        line = r.linea if n is None else n - 0.5
        rate = getattr(x, f"r_{stat}") * (1 if x.starter else SUB_W)  # shortcut: titular/suplente real, no el previsto
        p = prob(rate, math.floor(line) + 1)
        val = getattr(x, stat)
        out.append({"fecha": day, "partido": f"{r.home} vs {r.away}", "jugador": r.jugador, "mercado": r.mercado, "linea": line,
                    "cuota": r.cuota, "p_casa": round(1 / r.cuota, 4), "p_modelo": round(p, 4), "titular": int(x.starter),
                    "valor": val, "acierto": int(val > line)})
    res = pd.DataFrame(out)
    if not res.empty: res.to_csv(LOG, index=False)
    return res


def report(min_edge=0.05):
    """Por mercado: n, acierto real, prob. de la casa (con margen), prob. del modelo y ROI de apostar donde el modelo ve
    ventaja ≥ min_edge (1 u)."""
    d = settle() if not LOG.exists() else pd.read_csv(LOG)
    if d.empty: return []
    lines = []
    for m, g in d.groupby("mercado"):
        bet = g[g.p_modelo * g.cuota - 1 >= min_edge]
        gain = np.where(bet.acierto == 1, bet.cuota - 1, -1.0)
        lines.append(f"{m} ({len(g)}): real {g.acierto.mean():.0%} · casa {g.p_casa.mean():.0%} · modelo {g.p_modelo.mean():.0%}"
                     + (f" · con ventaja {len(bet)}: ROI {gain.mean():+.0%}" if len(bet) else ""))
    return lines


if __name__ == "__main__":
    backtest()
    s = settle()
    print(f"props de Unibet liquidadas: {len(s)}")
    for l in report(): print(" ", l)
