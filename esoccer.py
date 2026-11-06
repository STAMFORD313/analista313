"""EA SPORTS FC (Esoccer Battle, partidos de 8 min) con la API JSON pública de football.esportsbattle.com (sin clave;
robots.txt: Allow /). Modelo: goles Poisson por partido con ataque/defensa por JUGADOR aprendidos online y nivel de goles
por liga (Volta, 2x4, 2x6...) → 1X2, total de goles y hándicap. No hay cuotas gratuitas legítimas de esoccer: las señales
salen como candidatas a cuota justa (ventaja 0, el bot no las apuesta) para medir calibración hasta tener cuotas."""
import json, math, time

import numpy as np
import pandas as pd
import requests

import analyst as A
import kambi

API = "https://football.esportsbattle.com/api"
DIR = A.DATA / "fc"
HIST = DIR / "matches.csv"
COLS = ["id", "date", "tournament_id", "league", "p1", "p2", "s1", "s2"]
SLUG, ICON, NAME = "fc", "🎮⚽", "FC 27"
TITLE = f"{ICON} {NAME}"
_P = json.loads((DIR / "params.json").read_text()) if (DIR / "params.json").exists() else {}  # lo ajusta train_fc.py
LR = _P.get("lr", 0.03)        # paso del ataque/defensa de cada jugador por partido
LR_LEAGUE = _P.get("lr_league", 0.01)
DRAW_LIGA = _P.get("draw_liga") or {}  # factor del empate por liga (train_fc_calib.py: CSL/Volta menos, World Cup B más)
CALIB = _P.get("calib")          # [a, b] de train_fc_calib.py: p_i ∝ p_i^a · e^(b·empate); None = sin recalibrar
H2H = _P.get("h2h", 0.0)         # train_fc_calib.py: peso del cara a cara (balance encogido) sobre local/visitante
SHRINK = _P.get("shrink", 0.0)  # encogimiento de las fuerzas hacia 0 por partido (olvido)
MAX_GOALS = 15
PER_RUN = 3                     # candidatas por run (hay ~1.500 partidos al día): las de pick más seguro en las próximas 2 h


def get(path, **params):
    r = requests.get(f"{API}/{path}", params=params, headers=A.UA, timeout=30)
    r.raise_for_status()
    return r.json()


def tournaments(day_from, day_to):
    out, page = [], 1
    while True:
        d = get("tournaments", page=page, dateFrom=f"{day_from:%Y-%m-%d}", dateTo=f"{day_to + pd.Timedelta(days=1):%Y-%m-%d}")  # dateTo excluye ese día
        out += d.get("tournaments", [])
        if page >= d.get("totalPages", 1): return out
        page += 1; time.sleep(0.3)


def match_rows(t, finished=True):
    """Partidos de un torneo: terminados (status 3) o programados (status 1)."""
    league = (t.get("league") or {}).get("token_international", "")
    return [{"id": m["id"], "date": m["date"], "tournament_id": t["id"], "league": league,
             "p1": m["participant1"]["nickname"], "p2": m["participant2"]["nickname"],
             "s1": m["participant1"].get("score"), "s2": m["participant2"].get("score")}
            for m in get(f"tournaments/{t['id']}/matches") if m.get("status_id") == (3 if finished else 1)]


def update_history(days=2, pause=0.3):
    """Guarda los partidos TERMINADOS de los últimos `days` días: de los torneos acabados (status 4) aún no completos y,
    además, los ya jugados de los torneos en curso (antes esperaba al final del torneo: horas de retraso en los resultados)."""
    DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=COLS)
    done_f = DIR / "torneos_hechos.json"
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set(int(x) for x in old.tournament_id.dropna())
    rows, now, finished = [], pd.Timestamp.now(tz="UTC"), []
    try: ts = [t for t in tournaments(now - pd.Timedelta(days=days), now) if t["id"] not in done and t.get("status_id") != 1]
    except (requests.RequestException, ValueError) as e: print("esportsbattle", e); ts = []
    for t in ts:
        try:
            rows += match_rows(t)
            if t.get("status_id") == 4: finished.append(t["id"])
        except (requests.RequestException, ValueError) as e: print("esportsbattle", t["id"], e)  # se reintenta en el siguiente run
        time.sleep(pause)
    df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True).drop_duplicates("id").dropna(subset=["s1", "s2"])
    df.to_csv(HIST, index=False)
    done_f.write_text(json.dumps(sorted(int(x) for x in done | set(finished))))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values(["date", "id"], ignore_index=True)


def poisson(k, lam): return math.exp(-lam) * lam ** k / math.factorial(k)


def calibrate(p, calib=None):
    """1X2 recalibrado (train_fc_calib.py: el Poisson independiente se queda corto en empates). Sin calib, igual."""
    c = CALIB if calib is None else calib
    if not c: return p
    w = {k: math.exp(c[0] * math.log(max(v, 1e-9)) + (c[1] if k == "X" else 0)) for k, v in p.items()}
    t = sum(w.values()); return {k: v / t for k, v in w.items()}


def draw_league(p, league, table=None):
    """Empate × factor de su liga (aprendido fuera de muestra) y renormaliza; sin factor, igual."""
    f = (DRAW_LIGA if table is None else table).get(league)
    if not f: return p
    w = {**p, "X": p["X"] * f}; t = sum(w.values()); return {k: v / t for k, v in w.items()}


def h2h_table(df):
    """{(a, b) ordenado: [victorias del 1.º − del 2.º, partidos]} de todo el historial."""
    H = {}
    for a, b, s1, s2 in zip(df.p1, df.p2, df.s1, df.s2):
        k = tuple(sorted((a, b))); s = H.setdefault(k, [0, 0])
        s[0] += (1 if s1 > s2 else -1 if s2 > s1 else 0) * (1 if k[0] == a else -1); s[1] += 1
    return H


def h2h_value(H, a, b):
    """Balance de A frente a B encogido: (victorias − derrotas) / (partidos + 5)."""
    k = tuple(sorted((a, b))); s = H.get(k, [0, 0])
    return (s[0] if k[0] == a else -s[0]) / (s[1] + 5)


def h2h_adjust(p, x, w=None):
    """1X2 con el cara a cara: sube el logaritmo de «1» en w·x y baja el de «2» lo mismo (w=0: igual)."""
    w = H2H if w is None else w
    if not w or not x: return p
    q = {"1": p["1"] * math.exp(w * x), "X": p["X"], "2": p["2"] * math.exp(-w * x)}
    t = sum(q.values()); return {k: v / t for k, v in q.items()}


def markets(l1, l2):
    """De goles esperados a 1X2, total de goles y diferencia (matriz Poisson independiente)."""
    P = [[poisson(i, l1) * poisson(j, l2) for j in range(MAX_GOALS)] for i in range(MAX_GOALS)]
    p1 = sum(P[i][j] for i in range(MAX_GOALS) for j in range(i))
    px = sum(P[i][i] for i in range(MAX_GOALS))
    tot = {g: sum(P[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j == g) for g in range(2 * MAX_GOALS)}
    diff = {d: sum(P[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i - j == d) for d in range(-MAX_GOALS, MAX_GOALS)}
    return {"1": p1, "X": px, "2": 1 - p1 - px}, tot, diff


class Model:
    """Goles de A contra B ~ Poisson(exp(base_liga + ataque_A − defensa_B)); actualización online (gradiente del log-lik)."""
    def __init__(self, lr=None, lr_league=None, shrink=None):
        self.lr, self.lrl, self.sh = LR if lr is None else lr, LR_LEAGUE if lr_league is None else lr_league, SHRINK if shrink is None else shrink
        self.att, self.dfn, self.base, self.n = {}, {}, {}, {}

    def lam(self, league, a, b):
        return math.exp(self.base.get(league, 1.0) + self.att.get(a, 0) - self.dfn.get(b, 0))

    def update(self, league, a, b, ga, gb):
        for x, y, g in ((a, b, ga), (b, a, gb)):
            r = g - self.lam(league, x, y)
            self.att[x] = self.att.get(x, 0) * (1 - self.sh) + self.lr * r
            self.dfn[y] = self.dfn.get(y, 0) * (1 - self.sh) - self.lr * r
            self.base[league] = self.base.get(league, 1.0) + self.lrl * r
        self.n[a] = self.n.get(a, 0) + 1; self.n[b] = self.n.get(b, 0) + 1


def fit(df, since=None, model=None):
    """Recorre el historial en orden; devuelve (modelo, log-loss 1X2 desde `since` con la predicción previa, nº)."""
    m, loss, cnt = model or Model(), 0.0, 0
    for r in df.itertuples():
        if since is not None and r.date >= since:
            p, _, _ = markets(m.lam(r.league, r.p1, r.p2), m.lam(r.league, r.p2, r.p1))
            o = "1" if r.s1 > r.s2 else "2" if r.s2 > r.s1 else "X"
            loss -= math.log(max(p[o], 1e-9)); cnt += 1
        m.update(r.league, r.p1, r.p2, r.s1, r.s2)
    return m, loss / max(cnt, 1), cnt


def form(df, nick, n=5):
    d = df[(df.p1 == nick) | (df.p2 == nick)].tail(n)
    return "".join("✅" if (r.s1 > r.s2) == (r.p1 == nick) and r.s1 != r.s2 else "➖" if r.s1 == r.s2 else "❌" for r in d.itertuples()) or "—"


def settle(df, state):
    done = {f"{SLUG}|{r['id']}": r for r in df.tail(20000).to_dict("records")}
    for s in state["signals"]:
        r = done.get(s["key"])
        if s.get("result") is None and r is not None:
            s1, s2 = int(r["s1"]), int(r["s2"])
            s["score"] = f"{s1}-{s2}"
            s["result"] = {"H": s1 > s2, "D": s1 == s2, "A": s2 > s1}[s["market"]]


ODDS = A.DATA / "fc" / "odds.csv"  # última cuota 1X2 de Unibet vista ANTES de cada partido, con la prob. del modelo


def save_odds(rows):
    if not rows: return
    new = pd.DataFrame(rows)
    old = pd.read_csv(ODDS) if ODDS.exists() else new.iloc[:0]
    pd.concat([old, new]).drop_duplicates("id", keep="last").to_csv(ODDS, index=False)


def market_report(edge=0.05):
    """Modelo frente a Unibet con los partidos ya jugados: log-loss modelo / mercado sin margen y ROI de apostar el
    resultado de más ventaja si es >= edge (a la cuota guardada antes del partido). {} sin datos."""
    if not ODDS.exists() or not HIST.exists(): return {}
    o = pd.read_csv(ODDS).merge(pd.read_csv(HIST)[["id", "s1", "s2"]], on="id").dropna(subset=["s1", "s2"])
    if o.empty: return {}
    y = np.select([o.s1 > o.s2, o.s1 == o.s2], [0, 1], 2)
    M = o[["m1", "mX", "m2"]].to_numpy(float); O = o[["o1", "oX", "o2"]].to_numpy(float)
    K = 1 / O; K /= K.sum(1, keepdims=True)
    ll = lambda P: float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())
    ev = M * O - 1; j = ev.argmax(1); bet = ev[np.arange(len(y)), j] >= edge
    gain = np.where(j == y, O[np.arange(len(y)), j] - 1, -1.0)[bet]
    return {"partidos": len(o), "ll_modelo": round(ll(M), 4), "ll_mercado": round(ll(K), 4), "apuestas": int(bet.sum()),
            "roi": round(float(gain.mean()), 4) if bet.any() else None,
            "roi_se": round(float(gain.std() / np.sqrt(bet.sum())), 4) if bet.sum() > 1 else None}


PRED = A.DATA / "fc" / "predicciones.csv"  # cada pronóstico enviado (ganador, total, hándicap) para la simulación
PRED_COLS = ["id", "start", "p1", "p2", "liga", "l1", "l2", "pick", "p_pick", "line", "p_over", "fav", "p_h15", "odds", "book", "lote"]


def save_predictions(rows):
    if not rows: return
    new = pd.DataFrame(rows, columns=PRED_COLS)
    old = pd.read_csv(PRED) if PRED.exists() else new.iloc[:0]
    pd.concat([old, new]).drop_duplicates("id", keep="first").reindex(columns=PRED_COLS).to_csv(PRED, index=False)


def backfill_predictions(state):
    """Pronósticos de FC ya enviados antes de existir predicciones.csv: se leen de su texto en state.json."""
    import re
    done = set(pd.read_csv(PRED).id) if PRED.exists() else set()
    rows = []
    for x in state.get("signals", []):
        if x.get("sport") != ICON: continue
        mid = int(x["key"].split("|")[1])
        if mid in done: continue
        t = "\n".join(x.get("info") or [])  # sin info (ya empezado): solo el ganador
        g = re.search(r"Goles esperados: (.+?) ([\d.]+) · (.+?) ([\d.]+)", t)
        tot = re.search(r"más de ([\d.]+) goles (\d+)%", t)
        h = re.search(r"Hándicap (.+?): −1.5 (\d+)%", t)
        p1, p2 = x["match"].split(" vs ", 1)
        rows.append({"id": mid, "start": x["start"], "p1": p1, "p2": p2, "liga": x.get("league", "").split(" · ")[-1],
                     "l1": float(g.group(2)) if g else None, "l2": float(g.group(4)) if g else None,
                     "pick": {"H": "1", "D": "X", "A": "2"}[x["market"]], "p_pick": x["p"],
                     "line": float(tot.group(1)) if tot else None, "p_over": int(tot.group(2)) / 100 if tot else None,
                     "fav": ("1" if A.esc(p1) == h.group(1) else "2") if h else None, "p_h15": int(h.group(2)) / 100 if h else None,
                     "odds": x["odds"] if x.get("book") == "Unibet" else None, "book": x.get("book")})
    save_predictions(rows)
    return len(rows)


def simulation():
    """Pronósticos ya jugados con su resultado: acierto del ganador, del total (lado más probable) y del hándicap −1.5 del
    favorito, y ganancia simulada a 1 u por pronóstico a la cuota justa (1/p) y, si la hubo, a la cuota de Unibet."""
    if not PRED.exists() or not HIST.exists(): return pd.DataFrame()
    d = pd.read_csv(PRED).merge(pd.read_csv(HIST)[["id", "s1", "s2"]], on="id").dropna(subset=["s1", "s2"])
    if d.empty: return d
    res = np.select([d.s1 > d.s2, d.s1 == d.s2], ["1", "X"], "2")
    d["ok_1x2"] = res == d.pick.astype(str)
    over, has_t, has_h = d.p_over >= 0.5, d.line.notna(), d.fav.notna()  # sin total/hándicap guardado: NaN (no cuenta)
    d["pick_tot"] = np.where(over, "Más", "Menos"); d["p_tot"] = np.where(has_t, np.where(over, d.p_over, 1 - d.p_over), np.nan)
    d["ok_tot"] = np.where(has_t, (((d.s1 + d.s2) > d.line) == over).astype(float), np.nan)
    fav1 = d.fav.astype(str).str.replace(".0", "", regex=False) == "1"
    d["ok_h15"] = np.where(has_h, (np.where(fav1, d.s1 - d.s2, d.s2 - d.s1) >= 2).astype(float), np.nan)
    d["g_justa"] = np.where(d.ok_1x2, 1 / d.p_pick - 1, -1.0)
    d["p_wt"] = [joint_win_total(r.l1, r.l2, str(r.pick), r.line, r.p_over >= 0.5) if pd.notna(r.l1) and pd.notna(r.line) else np.nan
                 for r in d.itertuples()]  # «crear apuesta» ganador + total del mismo partido (prob. conjunta exacta del Poisson)
    d["ok_wt"] = np.where(d.p_wt.notna(), (d.ok_1x2 & (d.ok_tot == 1)).astype(float), np.nan)
    d["g_unibet"] = np.where(d.odds.notna(), np.where(d.ok_1x2, d.odds - 1, -1.0), np.nan)
    return d.sort_values("start")


RULE_SINCE = "2026-11-06T08:30:00+00:00"  # regla actual (1/X/2 del modelo, ventaja ≥ 5 %): su resultado desde aquí es el honesto
RULE_LABEL = "6/11 9:30, 1/X/2 según estadísticas"
SEND_PICKS = {"1", "X", "2"}  # 6-nov: el pick sale de las estadísticas del modelo (1, X o 2), ventaja ≥ 5 %; sin stake. Antes: 6-nov: los empates tampoco aguantan fuera de muestra (35 desde la regla, 2 aciertos, −100/−12/−100 % por tramo) → no se manda nada; se sigue midiendo
# antes: {"X"}  # 1.ª/2.ª mitad (ventaja ≥ 5 %): 1 −10 %/−17 %, 2 −20 %/−4 %, X +82 % (34)/+59 % (18) → solo empates
MIN_EDGE_SEND = 0.05  # 999 partidos con cuota de Unibet: ventaja > 0 → ROI −8.4 % ± 6.1 % (535); > 5 % → +1.7 % ± 8.5 % (311)


def joint_win_total(l1, l2, pick, line, over):
    """P(gana el pick y el total cae del lado elegido) con goles Poisson independientes (como markets)."""
    return sum(poisson(i, l1) * poisson(j, l2) for i in range(MAX_GOALS) for j in range(MAX_GOALS)
               if {"1": i > j, "X": i == j, "2": j > i}[pick] and ((i + j > line) == over))


def lot_combos(d):
    """Combinada de los pronósticos de una misma tarjeta (mismo run, ≥2, todos jugados): prob. = producto (partidos
    distintos), acierto si aciertan todos; ganancia a 1 u a cuota justa y, si todos tenían cuota de Unibet, a esa."""
    if "lote" not in d or d.lote.isna().all(): return pd.DataFrame()
    rows = []
    for lote, g in d.dropna(subset=["lote"]).groupby("lote"):
        if len(g) < 2: continue
        p, ok = float(g.p_pick.prod()), bool(g.ok_1x2.all())
        odds = float(g.odds.prod()) if g.odds.notna().all() else np.nan
        rows.append({"lote": lote, "n": len(g), "p": p, "ok": ok, "g_justa": 1 / p - 1 if ok else -1.0,
                     "g_unibet": (odds - 1 if ok else -1.0) if pd.notna(odds) else np.nan})
    return pd.DataFrame(rows)


FC_EDGE = 0.05  # solo se «apuesta» (simulación y tarjeta) si prob. del bot × cuota de Unibet − 1 ≥ 5 %: elegir por valor,
                # no por favorito (4-nov, 142 con cuota: sin ventaja ROI −27 %, con ventaja 5-30 % +23 %)


def value_bets(d):
    return d[d.odds.notna() & (d.p_pick * d.odds - 1 >= FC_EDGE) & (d.p_pick * d.odds - 1 <= 0.3)]


def sim_lines(last=8):
    """Resultado de las apuestas simuladas de FC 27: solo las de valor (cuota de Unibet por encima de la prob. del bot)."""
    d = simulation()
    v = value_bets(d) if not d.empty else d
    if v.empty: return ["Aún no hay apuestas de valor de FC 27 jugadas."]
    out = [f"💰 Total ({len(v)} apuestas de 1 € con valor): <b>{v.g_unibet.sum():+.2f} €</b> · acertadas {v.ok_1x2.mean():.0%}"]
    import boleto
    for r in v.tail(last).itertuples():
        win = {"1": f"Gana {r.p1}", "X": "Empate", "2": f"Gana {r.p2}"}[str(r.pick)]
        out += ["", *boleto.boleto(f"{r.p1} vs {r.p2} ({int(r.s1)}-{int(r.s2)})", [(win, bool(r.ok_1x2))], r.odds, 1.0)]
    return out


def run(state, today, kg):
    df = update_history()
    print(f"{SLUG}: {len(df)} partidos de historial")
    if df.empty: return
    settle(df, state)
    try: backfill_predictions(state)
    except Exception as e: print("simulación FC", repr(e))
    m, *_ = fit(df[df.date >= df.date.max() - pd.Timedelta(days=120)])
    H = h2h_table(df)
    now = pd.Timestamp.now(tz="UTC")
    cands = []
    try:
        for t in tournaments(now, now + pd.Timedelta(hours=3)):
            if t.get("status_id") not in (2, 3) or not (now - pd.Timedelta(hours=3) < pd.Timestamp(t["start_date"]) < now + pd.Timedelta(hours=2)):
                continue  # solo torneos en juego o que empiezan pronto (hay ~60 al día)
            for g in match_rows(t, finished=False):
                start, key = pd.Timestamp(g["date"]), f"{SLUG}|{g['id']}"
                if not (now < start < now + pd.Timedelta(hours=2)) or key in state["analyzed"]: continue
                l1, l2 = m.lam(g["league"], g["p1"], g["p2"]), m.lam(g["league"], g["p2"], g["p1"])
                p, tot, diff = markets(l1, l2)
                p = h2h_adjust(draw_league(calibrate(p), g["league"]), h2h_value(H, g["p1"], g["p2"]))
                cands.append((max(p.values()), g, start, key, l1, l2, p, tot, diff))
            time.sleep(0.3)
    except (requests.RequestException, ValueError) as e:
        print("esportsbattle", e); return
    try: evs = kambi.esoccer_events()
    except Exception as e: print("Kambi esoccer", repr(e)); evs = []
    es_evs = {}  # casas con licencia en España (mismos partidos, su propio precio): para que se pueda apostar desde allí
    for op in kambi.ES_BOOKS:
        try: es_evs[op] = kambi.esoccer_events(op)
        except Exception as e: print("Kambi esoccer", op, repr(e)); es_evs[op] = []
    book, rows = {}, []
    for c in cands:  # cuotas reales de Unibet: se guardan todas (para medir el modelo), aunque no salgan como señal
        g, start, p = c[1], c[2], c[6]
        if (e := kambi.find_esoccer(g["p1"], g["p2"], start, evs)):
            book[c[3]] = e["odds"]
            rows.append({"id": g["id"], "seen": now.isoformat(), "start": start.isoformat(), "p1": g["p1"], "p2": g["p2"],
                         "o1": e["odds"]["1"], "oX": e["odds"]["X"], "o2": e["odds"]["2"],
                         "m1": round(p["1"], 4), "mX": round(p["X"], 4), "m2": round(p["2"], 4)})
    save_odds(rows)
    print(f"{SLUG}: {len(evs)} partidos de esoccer en Unibet, {len(rows)} emparejados con Esoccer Battle")
    # primero los que tienen cuota de Unibet (por ventaja), luego el resto (por seguridad del pick)
    edge_of = lambda c: max(c[6][k] * book[c[3]][k] - 1 for k in "1X2") if c[3] in book else -9
    def sendable(c):  # su mejor pick es de los que se mandan (empate) con ventaja suficiente: va primero
        if c[3] not in book: return False
        e = {k: c[6][k] * book[c[3]][k] - 1 for k in "1X2"}; k = max(e, key=e.get)
        return k in SEND_PICKS and e[k] >= MIN_EDGE_SEND
    preds = []
    for _, g, start, key, l1, l2, p, tot, diff in sorted(cands, key=lambda c: (sendable(c), c[3] in book, edge_of(c), c[0]), reverse=True)[:PER_RUN]:
        state["analyzed"].append(key)
        kg.match(TITLE, g["p1"], g["p2"], start, p, ([], []), "")
        mk = max(p, key=p.get)
        line = math.floor(l1 + l2) + 0.5
        over = sum(v for k, v in tot.items() if k > line)
        fav, sign = (g["p1"], 1) if l1 >= l2 else (g["p2"], -1)
        hcp = {h: sum(v for d, v in diff.items() if d * sign > h) for h in (0.5, 1.5, 2.5)}  # favorito gana por más de h
        info = [f"📊 <b>Escenarios</b> (Poisson por jugador, {len(df):_} partidos)".replace("_", "."),
                f"Goles esperados: {A.esc(g['p1'])} {l1:.2f} · {A.esc(g['p2'])} {l2:.2f}",
                f"Ganador: {A.esc(g['p1'])} {p['1']:.0%} · empate {p['X']:.0%} · {A.esc(g['p2'])} {p['2']:.0%}",
                f"Total: más de {line:g} goles {over:.0%} · menos {1 - over:.0%}",
                f"Hándicap {A.esc(fav)}: −1.5 {hcp[1.5]:.0%} · −2.5 {hcp[2.5]:.0%}",
                "", "🔎 <b>Claves</b>",
                f"• Forma: {A.esc(g['p1'])} {form(df, g['p1'])} | {A.esc(g['p2'])} {form(df, g['p2'])}",
                f"• Partidos: {A.esc(g['p1'])} {m.n.get(g['p1'], 0)} | {A.esc(g['p2'])} {m.n.get(g['p2'], 0)}",
                f"• Liga: {A.esc(g['league'])}",
                "• Sin cuota de Unibet para este partido: cuota justa del modelo, solo candidata (no se apuesta)."]
        sig = {"odds": round(1 / p[mk], 2), "book": "cuota justa (sin casa)", "implied": round(p[mk], 3), "edge": 0.0}
        es = [(kambi.ES_BOOKS[op], e["odds"]) for op, ev in es_evs.items() if (e := kambi.find_esoccer(g["p1"], g["p2"], start, ev))]
        if (o := book.get(key)):  # cuota real: el resultado de más ventaja; sin stake hasta que market_report demuestre ROI
            mk = max("1X2", key=lambda k: p[k] * o[k] - 1)
            inv = sum(1 / v for v in o.values())
            sig = {"odds": o[mk], "book": "Unibet", "implied": round(1 / o[mk] / inv, 3), "edge": round(p[mk] * o[mk] - 1, 3), "sin_stake": True}
            info[-1] = (f"• Unibet: 1 {o['1']:.2f} · X {o['X']:.2f} · 2 {o['2']:.2f} (sin margen {1 / o['1'] / inv:.0%}/"
                        f"{1 / o['X'] / inv:.0%}/{1 / o['2'] / inv:.0%}) · ventaja {sig['edge']:+.0%}. Sin stake hasta demostrar ROI con estas cuotas.")
        if es:  # dónde apostarlo en España: el mismo pick en LeoVegas/Paf (con su ventaja según el modelo)
            info.append("• En España: " + " · ".join(f"{b} {eo[mk]:.2f} ({p[mk] * eo[mk] - 1:+.0%})" for b, eo in es))
        preds.append({"id": g["id"], "start": start.isoformat(), "p1": g["p1"], "p2": g["p2"], "liga": g["league"], "l1": round(l1, 3),
                      "l2": round(l2, 3), "pick": mk, "p_pick": round(p[mk], 4), "line": line, "p_over": round(over, 4),
                      "fav": "1" if sign == 1 else "2", "p_h15": round(hcp[1.5], 4), "odds": sig["odds"] if o else None, "book": sig["book"],
                      "lote": now.isoformat()[:16]})  # mismo run = misma tarjeta (para simular su combinada)
        if o and sig["edge"] < MIN_EDGE_SEND: continue  # medido (999 partidos): con < 5 % de ventaja se pierde; no se manda
        if o and mk not in SEND_PICKS: continue  # 1/2 con ventaja pierden en las dos mitades del historial; se siguen midiendo
        state["signals"].append({"market": {"1": "H", "X": "D", "2": "A"}[mk], "pick": {"1": g["p1"], "X": "Empate", "2": g["p2"]}[mk],
                                 **sig, "p": round(p[mk], 3), "key": key, "match": f"{g['p1']} vs {g['p2']}",
                                 "start": start.isoformat(), "league": f"{TITLE} · {g['league']}", "sport": ICON, "info": info,
                                 "context": f"EA FC Esoccer Battle {g['p1']} vs {g['p2']} ({g['league']}): goles esperados {l1:.2f}-{l2:.2f}, "
                                            f"1X2 {p['1']:.0%}/{p['X']:.0%}/{p['2']:.0%}. Sin cuotas de casa."})
    save_predictions(preds)
