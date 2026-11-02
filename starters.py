"""Titulares que el Elo no ve: porteros de la NHL y lanzadores abridores de la MLB, desde el resumen público de ESPN
(summary?event=<id>). Guarda por partido y jugador "toi" (minutos; en MLB 1 si abrió), "sa" (volumen: tiros recibidos o
entradas) y "ga" (daño: goles encajados o numerador del FIP 13·HR+3·BB−2·K). Rating = daño evitado por unidad de volumen
frente a la media de la liga (encogido y con decaimiento); la prob. del Elo se ajusta por el titular frente al habitual del
equipo. `python starters.py <nhl|mlb> descargar` baja 3 temporadas; `python starters.py <liga>` hace el backtest."""
import itertools, json, sys, time

import numpy as np
import pandas as pd

import multisport as M

CONF = {"nhl": ("hockey/nhl", "goalies.csv", 30),        # (ruta ESPN, archivo, volumen típico del titular por partido)
        "mlb": ("baseball/mlb", "pitchers.csv", 5.5 / 9)}  # FIP en carreras por 9 entradas; un abridor lanza ~5.5
FILE = lambda league: M.A.DATA / league / CONF[league][1]
PARAMS = M.A.DATA / "starters.json"
COLS = ["id", "date", "side", "goalie", "name", "toi", "sa", "ga"]  # goalie = id del jugador (portero o lanzador)


def rows(d, g):
    """Porteros o lanzadores del resumen ESPN `d` del partido `g` (fila del historial)."""
    out = []
    num = lambda x: float(str(x).replace("--", "0") or 0)
    for t in d.get("boxscore", {}).get("players", []):
        side = "home" if t["team"]["displayName"] == g["home"] else "away" if t["team"]["displayName"] == g["away"] else None
        for s in t.get("statistics", []):
            lab = s.get("labels") or []
            if s.get("name") != "goalies" and "IP" not in lab: continue
            ix = {l: i for i, l in enumerate(lab)}
            for a in s.get("athletes", []):
                st = a.get("stats") or []
                v = lambda k: st[ix[k]] if k in ix and ix[k] < len(st) else "0"
                if "IP" in lab:  # béisbol: 5.2 entradas = 5 y 2/3
                    ip = int(num(v("IP"))) + round(num(v("IP")) % 1 * 10) / 3
                    vals = [int(bool(a.get("starter"))), round(ip, 3), 13 * num(v("HR")) + 3 * num(v("BB")) - 2 * num(v("K"))]
                else:
                    toi = str(v("TOI"))
                    vals = [round(sum(float(x) / 60 ** i for i, x in enumerate(toi.split(":"))), 2) if ":" in toi else 0.0, num(v("SA")), num(v("GA"))]
                out.append([g["id"], g["date"], side, a["athlete"]["id"], a["athlete"]["displayName"], *vals])
    return out


def history(league):
    d = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty])
    d["date"] = pd.to_datetime(d.date, utc=True)
    return d.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)


def download(league, years=3, pause=0.35, since_days=None):
    """Incremental: salta partidos ya guardados. since_days limita a los partidos recientes (uso en cada run)."""
    d = history(league)
    d = d[d.date >= d.date.max() - (pd.Timedelta(days=since_days) if since_days else pd.DateOffset(years=years))]
    f = FILE(league)
    done = set(pd.read_csv(f, dtype={"id": str}).id) if f.exists() else set()
    f.parent.mkdir(parents=True, exist_ok=True)
    new = []
    for g in d[~d.id.isin(done)].to_dict("records"):
        try: new += rows(M.get(f"{CONF[league][0]}/summary", event=g["id"]), g)
        except (M.requests.RequestException, ValueError, KeyError) as e: print("ESPN", g["id"], e); time.sleep(5); continue
        if len(new) > 400:  # guarda a trozos: la descarga larga puede cortarse
            pd.DataFrame(new, columns=COLS).to_csv(f, mode="a", header=not f.exists(), index=False); new = []
        time.sleep(pause)
    if new: pd.DataFrame(new, columns=COLS).to_csv(f, mode="a", header=not f.exists(), index=False)


def ratings(df, gl, hl=180, k=600, alpha=0.15):
    """Recorre los partidos en orden. Devuelve ({id: diferencia local − visitante de (titular − habitual) ANTES del
    partido}, estado final para predecir: ratings por jugador y habitual por equipo)."""
    teams = df.set_index("id")[["home", "away"]].to_dict("index")
    gl = gl[gl.id.isin(teams)].assign(t=lambda x: pd.to_datetime(x.date, utc=True)).sort_values(["t", "id"])
    G, team, lg, out = {}, {}, [0.0, 0.0, None], {}
    def rate(g, t):
        e, s, t0 = G.get(g, (0.0, 0.0, t))
        f = 0.5 ** ((t - t0).days / hl)
        return e * f / (s * f + k)
    for gid, x in gl.groupby("id", sort=False):
        t = x.t.iloc[0]
        st = x.sort_values("toi").groupby("side").tail(1).set_index("side")  # titular: más minutos / el que abrió
        diff = 0.0
        for side, sign in (("home", 1), ("away", -1)):
            if side not in st.index: continue
            name, r = teams[gid][side], rate(st.goalie[side], t)
            diff += sign * (r - team.get(name, r))
            team[name] = team.get(name, r) + alpha * (r - team.get(name, r))
        out[gid] = diff
        if lg[2] is not None:  # media de la liga, también con decaimiento
            f = 0.5 ** ((t - lg[2]).days / hl); lg[0] *= f; lg[1] *= f
        lg[0] += x.ga.sum(); lg[1] += x.sa.sum(); lg[2] = t
        rate_lg = lg[0] / max(lg[1], 1)
        for g, sa, ga in zip(x.goalie, x.sa, x.ga):
            e, s, t0 = G.get(g, (0.0, 0.0, t))
            f = 0.5 ** ((t - t0).days / hl)
            G[g] = (e * f + sa * rate_lg - ga, s * f + sa, t)
    return out, {"G": G, "team": team}


def adjust(p, diff, c, n):
    """Prob. local del Elo + `c` puntos Elo por gol/carrera evitada (`n` unidades de volumen del titular por partido)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + 10 ** (-(400 * np.log10(p / (1 - p)) + c * n * np.asarray(diff, float)) / 400))


def live(league, df, games, now):
    """{id: diferencia de titulares probables} para los partidos de hoy (0 si ESPN no publica el probable)."""
    c = json.loads(PARAMS.read_text())[league]
    download(league, since_days=4, pause=0.2)  # incremental: solo los partidos terminados que falten
    _, st = ratings(df, pd.read_csv(FILE(league), dtype={"id": str, "goalie": str}), c["hl"], c["k"], c["alpha"])
    def rate(g):
        e, s, t0 = st["G"].get(g, (0.0, 0.0, now))
        f = 0.5 ** ((now - t0).days / c["hl"])
        return e * f / (s * f + c["k"])
    return {g["id"]: sum(sign * (rate(pid) - st["team"].get(team, rate(pid)))
                         for pid, team, sign in zip(g.get("prob") or ("", ""), (g["home"], g["away"]), (1, -1)) if pid)
            for g in games}


def main(league):
    df, gl, n = history(league), pd.read_csv(FILE(league), dtype={"id": str, "goalie": str}), CONF[league][2]
    P = M._P[league]
    pred = []
    M.elo(df, P["hfa"], P["K"], pred, P["keep"], mov=P["mov"])
    df["p"] = pred
    first = pd.to_datetime(gl.date, utc=True).min()
    ev = df[df.id.isin(set(gl.id)) & (df.res != 0.5)]
    ev = ev[ev.date >= first + pd.DateOffset(years=1)]  # 1.er año: solo calienta los ratings
    val = (ev.date < first + pd.DateOffset(years=2)).to_numpy()  # 2.º año elige; el resto mide
    ll = lambda q, yy: -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
    y, res = ev.res.to_numpy(), {}
    for hl, k, a in itertools.product((480, 960, 1920, 10 ** 5), (30, 100, 300, 600, 1200, 2500), (0.05, 0.15, 0.3)):
        d = ev.id.map(ratings(df, gl, hl, k, a)[0]).fillna(0).to_numpy()
        for c in (0, 25, 50, 75, 100, 150, 200):
            res[hl, k, a, c] = (ll(adjust(ev.p, d, c, n), y)[val].mean(), d)
    hl, k, a, c = min(res, key=lambda x: res[x][0])
    base, new = ll(adjust(ev.p, 0, 0, n), y)[~val], ll(adjust(ev.p, res[hl, k, a, c][1], c, n), y)[~val]
    dd = new - base
    print(f"{league} titulares: vida media {hl} d, k {k}, habitual {a}, {c} Elo por gol/carrera → log-loss {base.mean():.4f} → "
          f"{new.mean():.4f} ({dd.mean():+.4f} ± {dd.std() / np.sqrt(len(dd)):.4f}, {len(dd)} partidos fuera de muestra)")
    if dd.mean() < -2 * dd.std() / np.sqrt(len(dd)):
        old = json.loads(PARAMS.read_text()) if PARAMS.exists() else {}
        PARAMS.write_text(json.dumps({**old, league: {"hl": hl, "k": k, "alpha": a, "c": c, "logloss": round(float(new.mean()), 4),
                                                      "logloss_old": round(float(base.mean()), 4)}}))
    else: print("sin mejora fuera del ruido: no se guarda")


if __name__ == "__main__":
    download(sys.argv[1]) if sys.argv[2:] == ["descargar"] else main(sys.argv[1])
