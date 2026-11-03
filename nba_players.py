"""Jugadores y bajas en NBA (y NHL/NFL) desde el resumen público de partido de ESPN (summary?event=<id>).
download(league): guarda en data/<league>/box.jsonl el boxscore compacto (minutos, puntos, +/- por jugador) y la
cuota de cierre de DraftKings (pickcenter) de cada partido del historial de data/espn/<league>.
Uso: python nba_players.py descargar [liga] | python nba_players.py (grafo + backtest de bajas)."""
import json, sys, time

import numpy as np
import pandas as pd

import multisport as M

SUMMARY = {"nba": "basketball/nba", "nhl": "hockey/nhl", "nfl": "football/nfl", "mlb": "baseball/mlb",
           "cfb": "football/college-football", "cbb": "basketball/mens-college-basketball",
           "ucl": "soccer/uefa.champions", "uel": "soccer/uefa.europa"}
PLAYERS = ("nba", "nhl")  # en el resto solo se guardan las cuotas


def box_path(league): return M.A.DATA / league / "box.jsonl"


def _min(v):
    try: return float(str(v).split(":")[0])  # NHL: "18:34" → 18
    except ValueError: return 0.0


def compact(d):
    """Resumen ESPN → dict compacto: equipos con [id, nombre, min, pts, +/-, titular, no jugó] y cuotas DK."""
    teams = []
    for t in d.get("boxscore", {}).get("players", []):
        rows = []
        for s in t.get("statistics", []):
            lab = s.get("labels") or s.get("names") or []
            ix = {l: i for i, l in enumerate(lab)}
            for a in s.get("athletes", []):
                if "id" not in a.get("athlete", {}): continue  # algunos All-Star/eventos traen atletas sin id
                st = a.get("stats") or []
                g = lambda k: st[ix[k]] if k in ix and ix[k] < len(st) else None
                mins = _min(g("MIN") or g("TOI") or 0)
                pts = g("PTS") or g("G") or 0
                pm = g("+/-") or 0
                rows.append([a["athlete"]["id"], a["athlete"]["displayName"], mins, float(str(pts).replace("--", "0") or 0),
                             float(str(pm).replace("+", "").replace("--", "0") or 0), bool(a.get("starter")), bool(a.get("didNotPlay"))])
        teams.append({"team": t["team"]["displayName"], "players": rows})
    pc = next((p for p in d.get("pickcenter") or [] if (p.get("provider") or {}).get("name") == "DraftKings"), None)
    h, a = (pc or {}).get("homeTeamOdds") or {}, (pc or {}).get("awayTeamOdds") or {}
    odds = None if not pc else {"ml_h": h.get("moneyLine"), "ml_a": a.get("moneyLine"), "ml_d": (pc.get("drawOdds") or {}).get("moneyLine"),
                                "spread": pc.get("spread"), "sp_h": h.get("spreadOdds"), "sp_a": a.get("spreadOdds"),
                                "total": pc.get("overUnder"), "over": pc.get("overOdds"), "under": pc.get("underOdds")}
    return {"teams": teams, "odds": odds}


def download(league, years=3, pause=0.35):
    """(ESPN solo guarda cuotas de cierre en el summary desde ~nov 2025: para cuotas basta years=1.)"""
    """Descarga incremental (salta ids ya guardados) de los partidos de los últimos `years` años."""
    df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty])
    df["date"] = pd.to_datetime(df.date, utc=True)
    df = df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date")
    df = df[df.date >= df.date.max() - pd.DateOffset(years=years)]
    p = box_path(league); p.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in p.open(encoding="utf-8")} if p.exists() else set()
    with p.open("a", encoding="utf-8") as f:
        for r in df.rename(columns={"as": "as_"}).itertuples():
            if r.id in done: continue
            try: d = M.get(f"{SUMMARY[league]}/summary", event=r.id)
            except (M.requests.RequestException, ValueError) as e: print("ESPN", r.id, e); time.sleep(5); continue
            c = compact(d)
            if league not in PLAYERS: c.pop("teams")
            f.write(json.dumps({"id": r.id, "date": str(r.date), "home": r.home, "away": r.away, "hs": r.hs, "as": r.as_,
                                "res": r.res, **c}, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(pause)


def load_box(league):
    return [json.loads(l) for l in box_path(league).open(encoding="utf-8")]


FEATS = ["d_min", "d_pts", "d_pm"]
WEIGHT = {"lesionado": 1.0, "duda": 0.5}


def coef_path(league): return M.A.DATA / league / "absences.json"


def missing(team, injured, st):
    """Bajas del informe actual (multisport.injuries()[equipo]) → (min/240, pts, pm) ausentes según el estado
    de player_table. Solo cuentan los habituales (≥15 min) que este equipo tenía; dudas a medias."""
    by_name = {(v.get("team"), v["name"]): v for v in st.values()}
    tot = np.zeros(3)
    for p in injured:
        v = by_name.get((team, p["name"]))
        if v and v["min"] >= REG_MIN:
            tot += WEIGHT.get(p.get("status"), 0.5) * np.array([v["min"] / 240, v["pts"], v["pm"]])
    return tot


def adjust(league, p_elo, miss_home, miss_away, coef=None):
    """Prob. local del Elo corregida por bajas (miss_* = missing(...) de cada equipo). Sin coeficientes: p_elo."""
    c = coef or (json.loads(coef_path(league).read_text()) if coef_path(league).exists() else None)
    if not c: return p_elo
    p = min(max(p_elo, 1e-6), 1 - 1e-6)
    d = np.asarray(miss_home) - np.asarray(miss_away)
    z = c["coef"][0] * np.log(p / (1 - p)) + sum(c["coef"][1 + i] * d[i] / c["std"][i] for i in range(3)) + c["intercept"]
    return float(1 / (1 + np.exp(-z)))


def backtest(league):
    from sklearn.linear_model import LogisticRegression
    games = sorted(load_box(league), key=lambda g: g["date"])
    T, st = player_table(games)
    P = M._P.get(league, {})
    df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty])
    df["date"] = pd.to_datetime(df.date, utc=True)
    df = df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)
    pred = []
    M.elo(df, P.get("hfa", 0), P.get("K", 20), pred, P.get("keep", 0.75), P.get("kt", 250), P.get("mov", False))
    pe = dict(zip(df.id, pred))
    m = T.set_index(["id", "team"])
    rows = []
    for g in games:
        if g["id"] not in pe or g["res"] == 0.5 or (g["id"], g["home"]) not in m.index or (g["id"], g["away"]) not in m.index: continue
        h, a = m.loc[(g["id"], g["home"])], m.loc[(g["id"], g["away"])]
        o = g.get("odds") or {}
        rows.append({"date": g["date"], "y": g["res"], "p": pe[g["id"]], "d_min": h.miss_min - a.miss_min, "d_pts": h.miss_pts - a.miss_pts,
                     "d_pm": h.miss_pm - a.miss_pm, "oh": M.american(o["ml_h"]) if o.get("ml_h") else np.nan,
                     "oa": M.american(o["ml_a"]) if o.get("ml_a") else np.nan})
    R = pd.DataFrame(rows)
    R["date"] = pd.to_datetime(R.date, utc=True)
    start = R.date.min() + pd.DateOffset(days=60)  # primeras semanas: aún no se conocen los habituales
    te = (R.date >= R.date.max() - pd.DateOffset(years=1)).to_numpy()
    tr = ~te & (R.date >= start).to_numpy()
    y, p = R.y.to_numpy(), np.clip(R.p.to_numpy(), 1e-6, 1 - 1e-6)
    std = [float(R[c].std()) or 1.0 for c in FEATS]
    X = np.column_stack([np.log(p / (1 - p)), *(R[c] / s for c, s in zip(FEATS, std))])
    ll = lambda q: -(y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q))
    base = ll(p[te])
    print(f"{league}: {len(R)} partidos ({tr.sum()} ajuste, {te.sum()} test); Elo {base.mean():.4f}")
    fits = {}
    for name, cols in {"Elo recalibrado": [0], "+min ausentes": [0, 1], "+pts ausentes": [0, 2], "+(+/-) ausente": [0, 3], "+todo": [0, 1, 2, 3]}.items():
        lr = LogisticRegression(C=10, max_iter=1000).fit(X[tr][:, cols], y[tr])
        q = lr.predict_proba(X[te][:, cols])[:, 1]; d = ll(q) - base
        fits[name] = (d.mean(), d.std() / np.sqrt(len(d)), q)
        print(f"  {name:16s} {ll(q).mean():.4f} ({d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f})")
    mk = np.column_stack([1 / R.oh, 1 / R.oa]); mk = mk[:, 0] / mk.sum(1)
    okm = te & np.isfinite(mk)
    if okm.sum() > 100:
        lm = lambda q, s: -(y[s] * np.log(q) + (1 - y[s]) * np.log(1 - q))
        q = fits["+todo"][2][okm[te]]
        print(f"  mercado DK sin margen {lm(mk[okm], okm).mean():.4f} vs Elo+bajas {lm(q, okm).mean():.4f} ({okm.sum()} partidos con cuota)")
        for edge in (0.02, 0.05):
            o = np.column_stack([R.oh, R.oa])[okm]; pq = 0.5 * q + 0.5 * mk[okm]
            for nm, pp in (("Elo", 0.5 * p[okm] + 0.5 * mk[okm]), ("Elo+bajas", pq)):
                ev = np.column_stack([pp * o[:, 0] - 1, (1 - pp) * o[:, 1] - 1]); win = np.column_stack([y[okm] == 1, y[okm] == 0])
                bet = (ev >= edge) & (ev <= 0.3)
                gain = np.where(win, o - 1, -1.0)[bet]
                print(f"  ROI mezcla 50/50 {nm:9s} EDGE {edge}: {gain.mean():+.2%} ± {gain.std() / np.sqrt(max(len(gain), 1)):.2%} en {len(gain)}")
    best = min(("+min ausentes", "+pts ausentes", "+(+/-) ausente", "+todo"), key=lambda k: fits[k][0])
    gain, se = fits["+todo"][:2]
    if gain < -2 * se:
        lr = LogisticRegression(C=10, max_iter=1000).fit(X[tr | te], y[tr | te])
        coef_path(league).write_text(json.dumps({"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "std": std,
                                                  "gain": round(float(gain), 4), "se": round(float(se), 4)}))
        print(f"  guardado {coef_path(league)}")
    elif coef_path(league).exists(): coef_path(league).unlink()
    (M.A.DATA / league / f"grafo_{league}.json").write_text(json.dumps(graph(games, st), ensure_ascii=False), encoding="utf-8")
    return fits, best


ALPHA, REG_MIN, WINDOW = 0.15, 15.0, 10  # media exponencial por partido; "habitual" = ≥15 min de media; ventana de equipo


def player_table(games):
    """Recorre los partidos en orden y devuelve, para cada (partido, equipo), los valores PREVIOS al partido de los
    habituales que no jugaron: [(id, equipo, min_ausentes/48·5, pts_ausentes, pm_ausentes)], más el estado final
    {jugador: {"team", "min", "pts", "pm", "n"}} (sirve para valorar las bajas del informe de lesiones actual)."""
    st, recent, last_team, out = {}, {}, {}, []
    future = {}  # (equipo, jugador) → nº de partidos futuros con ese equipo: ¿sigue en la plantilla?
    for g in games:
        for t in g["teams"]:
            for p in t["players"]:
                if p[2] > 0: future[t["team"], p[0]] = future.get((t["team"], p[0]), 0) + 1
    for g in games:
        for t in g["teams"]:
            team, played = t["team"], {p[0] for p in t["players"] if p[2] > 0}
            for p in t["players"]:
                if p[2] > 0: future[team, p[0]] -= 1
            cand = {pid for gm in recent.get(team, []) for pid in gm}
            miss = [pid for pid in cand - played if st.get(pid, {}).get("min", 0) >= REG_MIN
                    and last_team.get(pid) == team and future.get((team, pid), 0) > 0]
            out.append((g["id"], team, sum(st[p]["min"] for p in miss) / 240, sum(st[p]["pts"] for p in miss),
                        sum(st[p]["pm"] for p in miss)))
            for p in t["players"]:
                if p[2] <= 0: continue
                s = st.setdefault(p[0], {"name": p[1], "min": p[2], "pts": p[3], "pm": p[4], "n": 0})
                for k, v in (("min", p[2]), ("pts", p[3]), ("pm", p[4])): s[k] += ALPHA * (v - s[k])
                s["n"] += 1; s["team"] = team; last_team[p[0]] = team
            recent[team] = (recent.get(team, []) + [played])[-WINDOW:]
    return pd.DataFrame(out, columns=["id", "team", "miss_min", "miss_pts", "miss_pm"]), st


def graph(games, st):
    """Grafo JSON: nodos jugador (equipo, partidos, medias, mejores compañeros por +/- juntos) y equipo."""
    pair, opp, tm = {}, {}, {}
    for g in games:
        res = {g["home"]: g["res"], g["away"]: 1 - g["res"]}
        names = [t["team"] for t in g["teams"]]
        for t in g["teams"]:
            on = [p for p in t["players"] if p[2] >= 10]
            for i, a in enumerate(on):
                for b in on[i + 1:]:
                    k = tuple(sorted((a[0], b[0])))
                    s = pair.setdefault(k, [0, 0.0]); s[0] += 1; s[1] += (a[4] + b[4]) / 2
            o = next((n for n in names if n != t["team"]), None)
            if o and t["team"] in res:
                r = opp.setdefault(t["team"], {}).setdefault(o, [0, 0]); r[0] += 1; r[1] += res[t["team"]] == 1
            x = tm.setdefault(t["team"], {"games": 0, "wins": 0}); x["games"] += 1; x["wins"] += res.get(t["team"]) == 1
    mates = {}
    for (a, b), (n, pm) in pair.items():
        if n >= 20:
            mates.setdefault(a, []).append((pm / n, b, n)); mates.setdefault(b, []).append((pm / n, a, n))
    nodes = [{"id": f"p:{pid}", "type": "jugador", "name": s["name"], "team": s.get("team"), "games": s["n"],
              "min": round(s["min"], 1), "pts": round(s["pts"], 1), "pm": round(s["pm"], 1),
              "best_mates": [{"id": f"p:{b}", "name": st[b]["name"], "pm_juntos": round(v, 1), "games": n}
                             for v, b, n in sorted(mates.get(pid, []), reverse=True)[:3]]}
             for pid, s in st.items() if s["n"] >= 10]
    nodes += [{"id": f"t:{t}", "type": "equipo", "name": t, **v,
               "rivals": sorted(({"team": o, "games": r[0], "wins": r[1]} for o, r in opp.get(t, {}).items()),
                                key=lambda r: -r["games"])[:5]} for t, v in tm.items()]
    edges = [{"source": n["id"], "target": f"t:{n['team']}", "type": "juega_en"} for n in nodes if n["type"] == "jugador"]
    edges += [{"source": f"p:{a}", "target": f"p:{b}", "type": "compañeros", "games": n, "pm_juntos": round(pm / n, 1)}
              for (a, b), (n, pm) in pair.items() if n >= 40 and a in st and b in st and st[a]["n"] >= 10 and st[b]["n"] >= 10]
    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    if sys.argv[1:2] == ["descargar"]:
        download(sys.argv[2] if len(sys.argv) > 2 else "nba")
    else:
        backtest(sys.argv[1] if len(sys.argv) > 1 else "nba")
