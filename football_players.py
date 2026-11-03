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
_P = json.loads((A.DATA / "football_players.json").read_text()) if (A.DATA / "football_players.json").exists() else {}
SUB_W, ALPHA, K = _P.get("sub_w", 0.35), _P.get("alpha", 0.05), _P.get("k", 6.0)  # backtest() los elige fuera de muestra
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


def walk(d, alpha=None, k=None, sub_w=None):
    """Por fila, ritmo previo (por partido como titular) de cada estadística, con olvido y encogido a su posición."""
    ALPHA_, K_, SUB_ = (ALPHA if alpha is None else alpha), (K if k is None else k), (SUB_W if sub_w is None else sub_w)
    S, P = {}, {}
    out = {s: np.zeros(len(d)) for s in STATS}; out["n"] = np.zeros(len(d))
    for i, r in enumerate(d.itertuples(index=False)):
        w = 1.0 if r.starter else SUB_
        st, ps = S.get(r.pid), P.setdefault(r.pos, {s: [0.0, 0.0] for s in STATS})
        out["n"][i] = st["n"] if st else 0
        for s in STATS:
            prior = ps[s][0] / ps[s][1] if ps[s][1] else 0.05
            x, e = st[s] if st else (0.0, 0.0)
            out[s][i] = (x + K_ * prior) / (e + K_)  # por partido completo como titular
        if st is None: st = S[r.pid] = {"n": 0, **{s: (0.0, 0.0) for s in STATS}}
        st["n"] += 1
        for s in STATS:
            v = getattr(r, s); x, e = st[s]
            st[s] = (x * (1 - ALPHA_) + v, e * (1 - ALPHA_) + w)
            ps[s][0] += v; ps[s][1] += w
    return pd.DataFrame(out, index=d.index), S, P


def backtest(save=True):
    """Olvido, K y peso del suplente elegidos con los titulares del tramo 40-70 % de los datos; medido en el 30 % final
    frente a (a) la frecuencia media de los titulares y (b) la frecuencia del PROPIO jugador (sus partidos previos,
    contando suplencias con el mismo peso), la referencia dura. Guarda data/football_players.json."""
    import itertools
    d = rows()
    q40, q70 = d.date.quantile(0.4), d.date.quantile(0.7)
    y_all = {s: (d[s].to_numpy() >= 1) for s in STATS}
    ll = lambda y, q: -(y * np.log(q) + (1 - y) * np.log(1 - q))
    best = None
    for a, k, sw in itertools.product([0.02, 0.05, 0.1], [3.0, 6.0, 12.0], [0.25, 0.35, 0.5]):
        f, _, _ = walk(d, a, k, sw)
        m = ((d.date >= q40) & (d.date < q70)).to_numpy() & (f.n >= 3).to_numpy() & d.starter.to_numpy()
        sc = sum(ll(y_all[s][m], np.clip(1 - np.exp(-f[s].to_numpy()[m]), 1e-4, 1 - 1e-4)).mean() for s in ("goles", "tiros_puerta"))
        if best is None or sc < best[0]: best = (sc, a, k, sw, f)
    _, a, k, sw, f = best
    own = {}
    for s in STATS:  # frecuencia propia: nº de partidos con ≥1 / partidos jugados, previos
        hit = pd.Series(y_all[s].astype(float), index=d.index)
        g = hit.groupby(d.pid)
        own[s] = (g.cumsum() - hit) / g.cumcount().replace(0, np.nan)
    te = (d.date >= q70).to_numpy() & (f.n >= 3).to_numpy() & d.starter.to_numpy()
    print(f"ESPN fútbol: {len(d)} jugador-partido, {d.game.nunique()} partidos desde {d.date.min():%Y-%m-%d}; elegido olvido {a}, K {k}, "
          f"suplente {sw}; test (30 % final, titulares con ≥3 previos): {te.sum()}")
    res = {"alpha": a, "k": k, "sub_w": sw, "n_test": int(te.sum()), "stats": {}}
    for s, name in (("goles", "marca"), ("tiros_puerta", "≥1 tiro a puerta"), ("cards", "tarjeta"), ("asist", "asistencia")):
        y = y_all[s][te]
        p = np.clip(1 - np.exp(-f[s].to_numpy()[te]), 1e-4, 1 - 1e-4)
        o = np.clip(np.nan_to_num(own[s].to_numpy()[te], nan=y.mean()), 0.02, 0.98)  # propia, recortada (con 3 partidos puede ser 0 o 1)
        l1, l0, lo = ll(y, p), ll(y, np.full(len(y), np.clip(y.mean(), 1e-4, 1 - 1e-4))), ll(y, o)
        se = lambda u, v: float((u - v).std() / math.sqrt(len(y)))
        res["stats"][s] = {"ll": round(float(l1.mean()), 4), "ll_media": round(float(l0.mean()), 4), "ll_propia": round(float(lo.mean()), 4),
                           "se_propia": round(se(l1, lo), 4)}
        print(f"  {name}: media de titulares {l0.mean():.4f} · frecuencia propia {lo.mean():.4f} · modelo {l1.mean():.4f} "
              f"(vs propia {l1.mean() - lo.mean():+.4f} ± {se(l1, lo):.4f}); prob. media {p.mean():.3f} vs real {y.mean():.3f}")
    if save: (A.DATA / "football_players.json").write_text(json.dumps(res, indent=1))
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
