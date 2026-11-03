import analyst
analyst.TG = analyst.CHAT = analyst.GEMINI = None  # los tests nunca envían a Telegram ni llaman a la IA
analyst.W, analyst.MARKET_EDGE = dict.fromkeys(analyst.W, 0.5), dict.fromkeys(analyst.W, 0.05)  # lógica con la mezcla clásica

from analyst import probs, value_bets, won

P = probs(1.5, 1.1)
assert abs(P["H"] + P["D"] + P["A"] - 1) < 1e-3 and abs(P["O25"] + P["U25"] - 1) < 1e-3 and P["H"] > P["A"]
vb = value_bets({"H": 0.6, "D": 0.2, "A": 0.2, "O25": 0.5, "U25": 0.5}, {"AvgH": 2.0, "AvgD": 3.0, "AvgA": 5.0})
assert [v["market"] for v in vb] == ["H"], vb  # A tiene ventaja pero prob < MIN_P
assert abs(sum(p for _, p in P["scores"])) < 1 and P["O15"] > P["O25"] > P["O35"]
assert won("O25", 2, 1) and won("U25", 1, 1) and not won("H", 1, 1)
print("ok")

import pandas as pd
from multisport import american, elo, signals, win_prob
assert american("+240") == 3.4 and american("-300") == 1.333 and american("EVEN") == 2.0
df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 10, utc=True),
                   "home": ["A", "B"] * 10, "away": ["B", "A"] * 10, "res": [1.0, 0.0] * 10})
R = elo(df, 0, 20)
assert R["A"] > 1600 > 1400 > R["B"] and win_prob(R, "A", "B", 0) > 0.8
assert [s["market"] for s in signals(0.7, 2.0, 1.9)] == ["home"]
print("ok multisport")

import bankroll, tempfile
from pathlib import Path
TMP = Path(tempfile.mkdtemp())
bankroll.STATS_MD, bankroll.LOG = TMP / "ESTADISTICAS.md", TMP / "senales.csv"  # los tests no tocan los archivos reales
st = {"signals": [{"key": "k1", "market": "H", "odds": 2.2, "p": 0.55, "edge": 0.21, "match": "A vs B", "start": "2099-01-01T00:00Z"},
                  {"key": "k2", "market": "home", "odds": 1.9, "p": 0.6, "edge": 0.14, "match": "C vs D", "pick": "C", "start": "2099-01-01T00:00Z"}]}
bk = st.setdefault("bank", {"balance": 100.0, "bets": []})
class KG:
    def bet(self, *a): pass
    def result(self, *a): pass
bankroll.A.GEMINI = None  # sin IA en el test
bankroll.place(bk, st["signals"], pd.Timestamp("2026-11-01"), KG())
assert len(bk["bets"]) == 3 and all(b["stake"] <= 5 for b in bk["bets"]) and bk["balance"] < 100
st["signals"][0]["result"], st["signals"][1]["result"] = True, False
bankroll.settle(bk, st["signals"], KG())
assert sorted(b["status"] for b in bk["bets"]) == ["ganada", "perdida", "perdida"]
w = next(b for b in bk["bets"] if b["status"] == "ganada")
assert abs(bk["balance"] - (100 - sum(b["stake"] for b in bk["bets"]) + w["stake"] * 2.2)) < 0.02
print("ok banca")

from analyst import absence_impact, best_odds
assert best_odds({"B365H": 2.1, "PSH": 2.25, "MaxH": 2.3}, "H") == (2.25, "Pinnacle")
imp = absence_impact([{"pos": "F", "status": "lesionado", "games": 10}, {"pos": "D", "status": "duda", "games": 10}], 10)
assert imp == {"att": 0.10, "def": 0.03}, imp
print("ok bajas y casas")
txt = bankroll.scoreboard({"bank": bk, "signals": st["signals"]})
assert "Ganadas: <b>1</b> (33 %)" in txt and sorted(txt.split("Últimas 3:")[1].split()[0]) == sorted("🟥🟩🟥"), txt  # el orden depende de la prioridad
print("ok marcador")

# nula: tarjeta y marcador no la cuentan como perdida; combinada con una pierna perdida se liquida sin esperar al resto
old = "2026-10-01T00:00Z"
sig = [{"key": "n1", "market": "home", "odds": 2.0, "p": 0.6, "edge": 0.2, "match": "G vs H", "pick": "G", "start": "2099-01-01T00:00Z", "result": "nula"},
       {"key": "c1", "market": "H", "odds": 2.0, "p": 0.6, "edge": 0.2, "match": "I vs J", "start": "2099-01-01T00:00Z", "result": False},
       {"key": "c2", "market": "H", "odds": 2.0, "p": 0.6, "edge": 0.2, "match": "K vs L", "start": "2099-01-01T00:00Z"},
       {"key": "v1", "market": "t1", "odds": 2.0, "p": 0.6, "edge": 0.2, "match": "M vs N", "pick": "M", "start": old},
       {"key": "a1", "market": "AHH", "odds": 2.0, "p": 0.6, "edge": 0.2, "match": "O vs P", "start": "2099-01-01T00:00Z", "result": False, "factor": 0.0}]
bk2 = {"balance": 50.0, "bets": [{"date": "2026-11-01", "legs": [["n1", "home"]], "stake": 5.0, "odds": 2.0, "desc": "n", "status": "abierta"},
                                 {"date": "2026-11-01", "legs": [["c1", "H"], ["c2", "H"]], "stake": 2.0, "odds": 4.0, "desc": "c", "status": "abierta"},
                                 {"date": "2026-11-01", "legs": [["v1", "t1"]], "stake": 3.0, "odds": 2.0, "desc": "v", "status": "abierta"},
                                 {"date": "2026-11-01", "legs": [["a1", "AHH"], ["c2", "H"]], "stake": 1.0, "odds": 4.0, "desc": "a", "status": "abierta"}]}
cards = []
bankroll.A.send_card = cards.append
bankroll.settle(bk2, sig, KG())
assert [b["status"] for b in bk2["bets"]] == ["nula", "perdida", "nula", "perdida"], bk2["bets"]
assert bk2["balance"] == 58.0 and "NULA" in cards[0] and "PERDIDA" not in cards[0]
txt = bankroll.scoreboard({"bank": bk2, "signals": sig})
assert "Perdidas: <b>2</b>" in txt and "Nulas: 2" in txt, txt
assert "| ⬜ +0.00 $ |" in bankroll.STATS_MD.read_text(encoding="utf-8")
print("ok nulas y combinadas")

# registro de calibración: una vez por señal, Brier y acierto por tramo
cal = {"bank": {"bets": [{"legs": [["k1", "H"]]}]}, "signals": [
    {"key": "k1", "market": "H", "odds": 2.0, "p": 0.6, "implied": 0.5, "edge": 0.2, "start": "2026-11-01", "sport": "⚽", "result": True, "clv": 0.05},
    {"key": "k2", "market": "A", "odds": 3.0, "p": 0.3, "implied": 0.3, "edge": -0.1, "start": "2026-11-01", "sport": "⚽", "result": False}]}
bankroll.log_signals(cal); bankroll.log_signals(cal)
d = pd.read_csv(bankroll.LOG)
assert len(d) == 2 and d.gano.tolist() == [1, 0]
md = "\n".join(bankroll.calibration_md())
assert "| 55%-70% | 1 | 60% | 100% | 0.160 | 0.250 | +100.0 % |" in md and "⚽ +5.0 % (1)" in md, md
print("ok calibración")

import menu
assert menu.target_message({"message": {"message_id": 7}}, {"scoreboard_id": 9}) == 7
assert menu.target_message({"message": {"message_id": 9}}, {"scoreboard_id": 9}) is None and menu.target_message(None, {}) is None
print("ok menú")

# hándicap de puntos NBA/NFL (push = nula) y cabecera sin icono repetido
assert bankroll.spread_result("sph", -5.5, 110, 100) is True and bankroll.spread_result("sph", -5.5, 104, 100) is False
assert bankroll.spread_result("spa", 5.0, 105, 100) == "nula" and bankroll.spread_result("spa", 7.0, 105, 100) is True
assert bankroll.pick_text({"market": "sph", "pick": "Lakers -5.5", "match": "Lakers vs Suns"}) == "Lakers -5.5"
assert bankroll.header({"sport": "🏀", "league": "🏀 NBA"}) == "<b>🏀 NBA</b>" and bankroll.header({"sport": "⚽", "league": "🇪🇸 LaLiga"}).startswith("⚽ ")
print("ok hándicap NBA/NFL")

# claves de CS2 ("cs2|<id>", sin fecha): el fútbol no debe anularlas ni sacarlas de `analyzed`; una nula imposible vuelve a pendiente
import analyst as A_
st3 = {"analyzed": ["cs2|132573", "E0|2026-10-01|A|B", "E0|2099-01-01|C|D"],
       "signals": [{"key": "cs2|132573", "market": "t1", "start": "2099-01-01T00:00Z"}, {"key": "E0|2020-01-01|X|Y", "market": "H"}]}
empty = pd.DataFrame({"Date": pd.to_datetime([]), "Div": [], "HomeTeam": [], "AwayTeam": [], "FTHG": [], "FTAG": []})
A_.settle_signals(empty, st3, pd.Timestamp("2026-11-01"))
assert st3["signals"][0].get("result") is None and st3["signals"][1]["result"] == "nula"
fx_orig, A_.fixtures = A_.fixtures, lambda: pd.DataFrame(columns=["Div", "Date", "Time", "HomeTeam", "AwayTeam"]).astype({"Date": "datetime64[ns]"})
A_.analyze_fixtures(empty, st3, pd.Timestamp("2026-11-01"), None)
A_.fixtures = fx_orig
assert sorted(st3["analyzed"]) == ["E0|2099-01-01|C|D", "cs2|132573"], st3["analyzed"]
bad = [{"key": "cs2|1", "market": "t1", "start": "2099-01-01T00:00Z", "result": "nula"},
       {"key": "nba|x", "market": "home", "start": "2099-01-01T00:00Z", "result": "nula", "score": "100-100"},
       {"key": "E0|2020-01-01|X|Y", "market": "H", "start": "2020-01-01T00:00Z", "result": "nula"}]
bankroll.sanity(bad)
assert [x["result"] for x in bad] == [None, "nula", "nula"]
print("ok claves cs2")
A_.GEMINI, A_._T0 = "x", 0  # fuera de presupuesto: no llama a la IA aunque haya clave
assert A_.ai_json("hola") is None
A_.GEMINI = None
print("ok presupuesto IA")

# IA: parseo tolerante, reintento y modelo de respaldo con 429 (sin red: cliente falso)
assert A_.parse_json('```json\n{"apostar": false}\n```') == {"apostar": False} and A_.parse_json("{malo") is None and A_.parse_json(None) is None
import types, google.genai as gg
class E429(Exception): code = 429
calls = []
def fake_gen(model, contents, config):
    calls.append(model)
    if model == "m1": raise E429("cuota")
    return types.SimpleNamespace(text='Claro: {"apostar": true, "confianza": 4, "motivo": "x"}')
real_client, real_sleep = gg.Client, A_.time.sleep
gg.Client = lambda **k: types.SimpleNamespace(models=types.SimpleNamespace(generate_content=fake_gen))
A_.time.sleep, A_.GEMINI, A_._T0, A_.AI_MODELS = (lambda s: None), "x", A_.time.time(), ["m1", "m2"]
assert A_.ai_json("p") == {"apostar": True, "confianza": 4, "motivo": "x", "_modelo": "m2"} and calls == ["m1", "m2"]
assert A_.ai_json("p")["_modelo"] == "m2" and calls == ["m1", "m2", "m2"]  # m1 queda descartado en el run
assert bankroll.ai_review({"match": "A vs B", "market": "H", "odds": 2.0, "p": 0.6, "edge": 0.2}) == (True, "x", 4, "m2")
gg.Client, A_.time.sleep, A_.GEMINI = real_client, real_sleep, None
cal["signals"].append({"key": "k3", "market": "H", "odds": 2.0, "p": 0.6, "implied": 0.5, "edge": 0.2, "start": "2026-11-02",
                       "sport": "⚽", "result": False, "ai": {"apostar": True, "modelo": None}})
bankroll.log_signals(cal)
d = pd.read_csv(bankroll.LOG, keep_default_na=False)
assert len(d) == 3 and d.ia.tolist() == ["", "", "sin IA"], d
print("ok IA con respaldo")

# FC 27: tarjeta de pronósticos sin inundar y fuera de la calibración
fc = {"bank": {"bets": []}, "signals": [{"key": f"fc|{i}", "market": "H", "pick": f"P{i}", "match": f"P{i} vs Q{i}", "odds": 2.0, "p": 0.5,
      "implied": 0.5, "edge": 0.0, "book": "cuota justa (sin casa)", "sport": "🎮⚽", "start": "2099-01-01T00:00Z",
      "info": ["x", "Goles esperados 1-1", "Ganador P 50%", "Total", "Hándicap", "Forma"]} for i in range(7)]}
import radar
cards.clear(); radar.A.send_card = cards.append
radar.fc_card(fc); radar.fc_card(fc)
assert len(cards) == 1 and cards[0].count("🆚") == 5 and "Gana P0" in cards[0] and "Forma" not in cards[0], cards
fc["fc_sent"] = "2000-01-01T00:00Z"; radar.fc_card(fc)
assert len(cards) == 2 and cards[1].count("🆚") == 2
fc["signals"][0]["result"] = True; fc["signals"][0]["start"] = "2026-11-02"
n0 = len(pd.read_csv(bankroll.LOG)); bankroll.log_signals(fc)
assert len(pd.read_csv(bankroll.LOG)) == n0 + 1 and "| Total | 3 |" in "\n".join(bankroll.calibration_md())
print("ok FC 27")

# revisión experta: decisión por valor con incertidumbre, autoajuste con IC, tope por partido, combinada diaria y congelado
v = {"p": 0.6, "implied": 0.5, "odds": 2.0, "edge": 0.2}
assert abs(bankroll.margin(v) - 0.10) < 1e-9 and bankroll.value_ok(v) and not bankroll.value_ok({**v, "edge": 0.08})
assert abs(bankroll.kelly_frac(v) - 0.15) < 1e-9 and bankroll.kelly_frac({**v, "p": 0.5}) == 0
seg = {"a": {"n": 60, "r": -0.5 * 60, "r2": 0.5 * 60}, "b": {"n": 60, "r": -0.05 * 60, "r2": 1.0 * 60}, "c": {"n": 10, "r": -9.0, "r2": 9.0}}
assert bankroll.bad_segments(seg) == {"a"}  # b: pérdida dentro del ruido; c: pocas apuestas
ev = [{"key": "e1", "market": m, "odds": 2.0, "p": 0.62, "implied": 0.6, "edge": 0.24, "match": "X vs Y", "pick": "X", "start": "2099-01-01T00:00Z"}
      for m in ("H", "O25", "AHH")]
bk3 = {"balance": 100.0, "bets": []}
bankroll.place(bk3, ev, pd.Timestamp("2026-11-03"), KG())
assert len(bk3["bets"]) == 2 and sum(b["stake"] for b in bk3["bets"]) <= 5.0 + 1e-9 and all(len(b["legs"]) == 1 for b in bk3["bets"]), bk3
fz = [{"key": "z", "market": "H", "info": ["x", "• Bajas: <b>A</b>: sin bajas"]}]
bankroll.freeze(fz); bankroll.freeze([{**fz[0], "captured": "otra"}])
assert fz[0]["bajas"] == "A: sin bajas" and fz[0]["version"] == bankroll.os.getenv("GITHUB_SHA", "local")[:7] and len(fz[0]["captured"]) == 16
print("ok revisión experta")

import radar
s0 = {"key": "k9", "market": "U25", "odds": 2.0, "p": 0.5, "edge": 0.06, "match": "E vs F", "sport": "⚽"}
assert [ok for ok, _ in radar.checklist(s0)] == [True, True]
assert len(sum(radar.menu(), [])) == len(radar.SPORTS)
print("ok radar")

from multisport import total_signals
st_ = (100.0, 100.0, {"att_h": {}, "def_h": {}, "att_a": {}, "def_a": {}})
tot = {v["market"]: v for v in total_signals((st_, 15.0), "X", "Y", 190.5, 1.91, 1.91)}
assert tot["over"]["p"] > 0.6 and tot["over"]["edge"] > 0.1 and tot["under"]["p"] < 0.4
print("ok totales")

from analyst import ah_factor, ah_bets
assert ah_factor(1, -0.5, 2.0) == 2.0 and ah_factor(0, -0.5, 2.0) == 0 and ah_factor(1, -1, 2.0) == 1.0
assert ah_factor(1, -0.75, 2.0) == 1.5 and ah_factor(0, -0.25, 2.0) == 0.5 and ah_factor(0, 0.25, 2.0) == 1.5
P2 = probs(2.0, 0.6)
ab = {v["market"]: v for v in ah_bets(P2, {"AHh": -0.5, "B365AHH": 1.95, "B365AHA": 1.95, "AvgAHH": 1.9, "AvgAHA": 1.9}, every=True)}
assert 0.59 < ab["AHH"]["p"] < 0.61 and ab["AHH"]["line"] == -0.5 and ab["AHH"]["edge"] > 0
print("ok hándicap asiático")

from esports import series_prob
assert series_prob(0.6, 1) == 0.6 and abs(series_prob(0.6, 3) - 0.648) < 1e-9 and series_prob(0.6, 5) > series_prob(0.6, 3)
print("ok cs2")

import json, multisport
fp = json.loads((analyst.DATA / "football_params.json").read_text())
assert (analyst.HALF_LIFE, analyst.SHRINK, analyst.RHO) == (fp["half_life"], fp["k"], fp["rho"])
import importlib; A2 = importlib.reload(analyst)  # recarga sin la sobrescritura de arriba
assert A2.W == {m: fp[m]["w"] for m in A2.W} and all(A2.MARKET_EDGE[m] == max(A2.EDGE, fp[m]["edge"]) for m in A2.W)
row = {"AvgH": 1.5, "AvgD": 4.0, "AvgA": 9.0, "B365H": 2.5, "B365D": 4.0, "B365A": 9.0}  # cuota local muy por encima
P9 = {"H": 0.9, "D": 0.05, "A": 0.05}
if A2.MARKET_EDGE["1x2"] > A2.MAX_EDGE:  # 1X2 apagado: lista candidatas pero no da señales
    assert A2.value_bets(P9, row) == [] and A2.value_bets(P9, row, every=True)
ep = json.loads((analyst.DATA / "espn_params.json").read_text())
assert all(multisport.SPORTS[l][3:] == (p["hfa"], p["K"]) for l, p in ep.items())
assert set(multisport.TOT) <= {"nba", "mlb", "nhl", "nfl", "cfb", "cbb"} and 0 < multisport.totals_model.p_over("nba", 200, 199.5, multisport.TOT) < 1
st2 = {"signals": [{"key": "nba|2026-11-02|A|B", "market": "sph", "line": -2.5}]}
multisport.settle(pd.DataFrame([{"date": "2026-11-02", "home": "A", "away": "B", "hs": 100, "as": 97, "res": 1.0}]), "nba", st2)
assert st2["signals"][0]["result"] is True and multisport.spreads(None) == (None, None, None)
print("ok parámetros entrenados")

import contextlib, io as io_, pathlib, tempfile, train_espn  # informe de cuotas previas de DraftKings con datos sintéticos
d = pathlib.Path(tempfile.mkdtemp()); (d / "nba").mkdir(); old_dir, multisport.DIR = multisport.DIR, d
pd.DataFrame({"id": [str(i) for i in range(20)], "date": "2026-01-01", "event": "", "home": "A", "away": "B",
              "hs": [100, 90] * 10, "as": [95, 99] * 10, "res": [1.0, 0.0] * 10}).to_csv(d / "nba" / "202601.csv", index=False)
pd.DataFrame({"id": [str(i) for i in range(20)], "p": 0.6, "oh": 1.8, "oa": 2.1, "tl": 190.5, "to": 1.9, "tu": 1.9,
              "p_over": 0.5, "sl": -2.5, "sh": 1.9, "sa": 1.9, "p_cover": 0.55}).to_csv(d / "odds_nba.csv", index=False)
with contextlib.redirect_stdout(io_.StringIO()) as out: train_espn.market_report()
multisport.DIR = old_dir
assert "nba ganador: 20 partidos" in out.getvalue() and "nba hándicap: 20 partidos" in out.getvalue(), out.getvalue()
print("ok informe de cuotas ESPN")
st3 = {"signals": [{"key": "ucl|2026-11-02|A|B", "market": "D"}, {"key": "ucl|2026-11-02|A|B", "market": "home"}]}
multisport.settle(pd.DataFrame([{"date": "2026-11-02", "home": "A", "away": "B", "hs": 1, "as": 1, "res": 0.5}]), "ucl", st3)
assert [s["result"] for s in st3["signals"]] == [True, False]  # Champions/Europa: el empate cuenta
assert [v["market"] for v in multisport.choices((0.5, 0.3, 0.2), (2.2, 3.3, 5.0), ("home", "D", "away"))] == ["home"]
assert {"cfb", "cbb", "ucl", "uel"} <= set(multisport.SPORTS) and "uecl" not in multisport.SPORTS
print("ok competiciones ESPN nuevas")

import starters  # porteros titulares: un portero que encaja menos que la media sube la prob. de su equipo
g1 = pd.DataFrame({"id": ["1", "2"], "home": ["A", "A"], "away": ["B", "B"]})
gl1 = pd.DataFrame({"id": ["1", "1", "2", "2"], "date": ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-03"],
                    "side": ["home", "away"] * 2, "goalie": ["g1", "g2", "g3", "g2"], "toi": 60.0, "sa": 30.0, "ga": [0.0, 6.0, 3.0, 3.0]})
d1, st1 = starters.ratings(g1, gl1, 180, 30)
assert d1["1"] == 0 and st1["G"]["g1"][0] > 0 > st1["G"]["g2"][0] and d1["2"] > 0  # g3 nuevo (media) vs g2 malo
assert starters.adjust(0.5, 0, 75, 30) == 0.5 and starters.adjust(0.5, 0.01, 75, 30) > 0.5
print("ok porteros")

import esports
sc = esports.bo3_scores(0.6)
assert abs(sum(sc.values()) - 1) < 1e-9 and sc[(2, 0)] > sc[(0, 2)]
xm = esports.extra_markets({"additional_markets": [{"bet_type": "total_maps_over_2_5", "coeff": 2.6}, {"bet_type": "total_maps_under_2_5", "coeff": 1.45}]},
                           0.5, {"team1": "A", "team2": "B"})
assert all(v["market"] in ("over", "under") and analyst.EDGE <= v["edge"] <= analyst.MAX_EDGE for v in xm)
st = {"signals": [{"key": "cs2|1", "market": "AHH", "line": -1.5}, {"key": "cs2|1", "market": "over"},
                  {"key": "cs2|1", "market": "AHA", "line": 1.5}, {"key": "cs2|1", "market": "t1", "team_id": 5},
                  {"key": "cs2|2", "market": "under"}]}
esports.settle(pd.DataFrame([{"id": 1, "s1": 2, "s2": 0, "winner_id": 5}, {"id": 2, "s1": 1, "s2": 0, "winner_id": 5}]), st)
assert [s["result"] for s in st["signals"]] == [True, False, False, True, "nula"]
print("ok cs2 mercados de mapas")
esports.configure("dota2")
assert esports.DISC == 4 and esports.HIST.parent.name == "dota2" and esports.TITLE.endswith("Dota 2")
esports.configure("cs2")
assert esports.DISC == 1 and esports.SLUG == "cs2"
print("ok más esports")
_d = Path(tempfile.mkdtemp())
pd.DataFrame({"id": [1, 2], "p": [0.8, 0.4], "o1": [1.6, 2.6], "o2": [2.4, 1.5]}).to_csv(_d / "odds.csv", index=False)
pd.DataFrame({"id": [1, 2], "t1_id": [5, 6], "winner_id": [5, 7]}).to_csv(_d / "matches.csv", index=False)
esports.ODDS, esports.HIST = _d / "odds.csv", _d / "matches.csv"
_r = esports.market_report()
assert _r["n"] == 2 and _r["modelo"] < _r["mercado"] and _r["apuestas"] >= 1
esports.configure("cs2")
assert "| 🎮 CS2 |" in esports.market_report_md()
print("ok informe esports vs Thunderpick")

import football_stats, kambi
_o = lambda t, odds, line: {"type": t, "odds": odds, "line": line, "status": "OPEN"}
_d = {"betOffers": [{"criterion": {"englishLabel": "Total Corners", "lifetime": "FULL_TIME"}, "outcomes": [_o("OT_OVER", 1850, 9500), _o("OT_UNDER", 1900, 9500)]},
                    {"criterion": {"englishLabel": "Total Corners by Leeds", "lifetime": "FULL_TIME"}, "outcomes": [_o("OT_OVER", 2000, 5500), _o("OT_UNDER", 1750, 5500)]},
                    {"criterion": {"englishLabel": "Total Corners - 1st Half", "lifetime": "FIRST_HALF"}, "outcomes": [_o("OT_OVER", 2000, 4500), _o("OT_UNDER", 1750, 4500)]}]}
_k = kambi.parse(_d, "Leeds", "Hull")
assert _k == {("corners", "t"): {9.5: {"o": 1.85, "u": 1.9}}, ("corners", "h"): {5.5: {"o": 2.0, "u": 1.75}}}, _k
_e, football_stats.expected = football_stats.expected, lambda *a: (7.0, 4.0, 9.5, 5.5, 4.5)  # local muy de córners
_sv = {v["market"]: v for v in football_stats.value_signals(None, "E1", "Leeds", "Hull", None, _k)}
football_stats.expected = _e
assert _sv["corners_t_o"]["p"] > _sv["corners_t_o"]["implied"] and _sv["corners_h_o"]["pick"] == "Más de 5.5 córners de Leeds", _sv
assert football_stats.result("corners_t_o", 9.5, {"HC": 6, "AC": 4}) and not football_stats.result("corners_h_o", 5.5, {"HC": 5, "AC": 9})
assert football_stats.result("corners_a_u", 4.5, {"HC": 6, "AC": float("nan")}) is None  # football-data aún sin córners: espera
assert bankroll.pick_text({**_sv["corners_h_o"], "match": "Leeds vs Hull"}) == "Más de 5.5 córners de Leeds" and bankroll.segment({**_sv["corners_h_o"], "sport": "⚽"}) == "⚽ corners"
print("ok córners con cuota real")

# combinadas del mismo partido con cópula (train_sgp.py → football_stats.json["rho"])
import train_sgp
import numpy as np
_rho = {"names": ["goals_t", "sot_t", "cards_h"], "R": [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]}
_l = [{"stat": "goals", "var": "goals_t", "sign": 1, "p": 0.6}, {"stat": "sot", "var": "sot_t", "sign": 1, "p": 0.6}]
_pj = football_stats.joint(_l, {}, _rho)
assert 0.36 < _pj < 0.6 and abs(_pj - train_sgp.cells_copula(_l[0]["p"], _l[1]["p"], 0.5)[0, 0]) < 1e-3  # positiva → > producto
assert football_stats.joint([_l[0], {**_l[1], "sign": -1}], {}, _rho) < 0.31  # «menos de» invierte la correlación
_l3 = _l + [{"stat": "cards", "var": "cards_h", "sign": 1, "p": 0.7}]
assert abs(football_stats.joint(_l3, {}, _rho) - _pj * 0.7) < 2e-3  # pierna independiente multiplica
assert abs(football_stats.joint(_l, {"goals|sot": 0.02}) - (0.36 + 0.02)) < 1e-9  # sin cópula: covarianzas de antes
_c = train_sgp.cells_copula(np.array([0.3, 0.7]), np.array([0.6, 0.2]), -0.3)
assert np.allclose(_c.sum(1), 1) and np.allclose(_c[:, 0] + _c[:, 1], [0.3, 0.7], atol=1e-6)
print("ok combinadas del mismo partido (cópula)")

# perfiles de equipo de 10 años en el grafo (team_profiles.py)
import team_profiles, knowledge
_n = 260
_d = pd.DataFrame({"Div": "E0", "Date": pd.date_range("2024-01-01", periods=_n, freq="3D"),
                   "HomeTeam": np.where(np.arange(_n) % 2, "Leeds", "Hull"), "AwayTeam": np.where(np.arange(_n) % 2, "Hull", "Leeds"),
                   "FTHG": 1, "FTAG": 0, "HS": 10, "AS": 8, "HST": 4, "AST": 3, "HF": 11, "AF": 12,
                   "HC": np.where(np.arange(_n) % 2, 7, 3), "AC": 4, "HY": 2, "AY": 1, "HR": 0, "AR": 0})
_p = team_profiles.profiles(_d)
assert _p["Leeds"]["local"]["corners"][:2] == [7.0, 4.0] and _p["Hull"]["local"]["corners"][:2] == [3.0, 4.0]
assert _p["Leeds"]["local"]["corners"][2] == 1.0 and _p["Hull"]["local"]["corners"][2] == 0.0  # línea de liga 4.5
assert _p["Leeds"]["partidos"] == _n and abs(_p["Leeds"]["forma10"]["corners"]) < 0.3
_kg = knowledge.Graph.__new__(knowledge.Graph); _kg.g = knowledge.nx.Graph(); _kg.g.add_node("Leeds", type="equipo")
_kg.team_stats(_p)
assert _kg.g.nodes["Leeds"]["stats"]["visitante"]["n"] == _n // 2
print("ok perfiles de equipo en el grafo")

# combinada del mismo partido en BO3 (esports.sgp_lines: ganador + más de 2.5 = 2-1)
import esports as _E, train_sgp_esports as _TS
_sc = _E.bo3_scores(0.6)
assert abs(sum(_sc.values()) - 1) < 1e-9 and np.allclose(_TS.cells(np.array([0.6]), np.array([_sc[2, 1] / 0.6]))[0], [_sc[2, 0], _sc[2, 1], _sc[1, 2], _sc[0, 2]])
_bet = {"team_1": {"coeff": 1.5}, "team_2": {"coeff": 2.5}, "additional_markets": [
    {"bet_type": "team_1_handicap_under_1_5", "coeff": 2.6}, {"bet_type": "team_2_handicap_over_1_5", "coeff": 1.45}]}
_sl = _E.sgp_lines(_bet, 0.6, {"team1": "FaZe", "team2": "NaVi"})
assert len(_sl) == 4 and "FaZe gana + Más de 2.5 mapas (2-1)" in _sl[2] and "casa" in _sl[2] and "casa" not in _sl[3]
_j = 1 / float(_sl[2].split("cuota justa ")[1].split()[0]); _n = 1 / float(_sl[2].split("ingenuo ")[1].split(")")[0])
assert _j < _n  # el producto ingenuo sobrestima la combinada (más de 2.5 y ganador están correlacionados)
print("ok crear apuesta esports BO3")

# Champions / Europa: estadísticas cruzando las ligas de cada equipo (football_stats.cross_expected / euro_lines)
_td = pd.DataFrame({"Date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]), "Div": ["E0", "SP1", "I1"],
                    "HomeTeam": ["Man United", "Barcelona", "Inter"], "AwayTeam": ["Arsenal", "Sevilla", "Roma"]})
_t0 = pd.Timestamp("2026-11-03")
assert football_stats.team_div(_td, "Manchester United", _t0) == ("E0", "Man United")
assert football_stats.team_div(_td, "AS Roma", _t0) == ("I1", "Roma") and football_stats.team_div(_td, "Internazionale", _t0) == ("I1", "Inter")
assert football_stats.team_div(_td, "Shakhtar Donetsk", _t0) is None and football_stats.team_div(_td, "Manchester City", _t0) is None
_cr = {"goals": (1.9, 0.9, 2.5, None, None), "corners": (6.4, 4.1, 10.5, 5.5, 4.5), "cards": (1.9, 2.1, 4.5, 2.5, 2.5),
       "fouls": (11.3, 10.1, 24.5, 12.5, 12.5), "sot": (6.3, 3.1, 9.5, 4.5, 4.5)}
_cl = football_stats.card_lines(None, None, "Barcelona", "Aston Villa", _t0, None, cross=_cr)
assert "sin backtest en Europa" in _cl[1] and any(l.startswith("🧩 Crear apuesta") for l in _cl)
_lg = [l for l in _cl if l.startswith("🚀 Combinada larga")]
assert _lg and float(_lg[0].split("cuota justa ")[1]) > float([l for l in _cl if l.startswith("🧩")][0].split("cuota justa ")[1].split()[0])
print("ok estadísticas Champions/Europa y combinada larga")

# forma reciente en las estadísticas de fútbol (train_form.py → football_stats.form_factor)
_fd = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=80, freq="7D"), "HomeTeam": "Leeds", "AwayTeam": "Hull",
                    "HC": [4] * 70 + [8] * 10, "AC": 4.0})
_P = {"form": {"corners": {"c": 0.2}}}
football_stats._cache.clear()
_ff = football_stats.form_factor(_fd, "corners", "Leeds", pd.Timestamp("2027-01-01"), _P)
assert abs(_ff - np.exp(0.2 * np.log((8 + 0.5) / ((66 * 4 + 80) / 76 + 0.5)))) < 1e-9 and _ff > 1
assert football_stats.form_factor(_fd, "corners", "Hull", pd.Timestamp("2027-01-01"), _P) == 1.0  # forma plana
assert football_stats.form_factor(_fd, "cards", "Leeds", pd.Timestamp("2027-01-01"), _P) == 1.0   # sin c: sin ajuste
football_stats._cache.clear()
print("ok forma reciente en estadísticas")

# tarjeta «🧩 COMBINADAS» (radar.combos_card) y relleno de estadísticas en Champions (multisport.euro_backfill)
_fut = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat()
_cs = {"bank": {"bets": []}, "signals": [
    {"key": "ucl|x|Barcelona|Aston Villa", "market": "home", "match": "Barcelona vs Aston Villa", "start": _fut, "p": 0.7, "odds": 1.3,
     "edge": -0.04, "league": "🏆 Champions League", "info": ["📊", "Córners 6.4 - 3.9: total +10.5 45%", "🧩 Crear apuesta: A + B → prob. 45%, cuota justa 2.22"]},
    {"key": "ucl|x|Barcelona|Aston Villa", "market": "D", "match": "Barcelona vs Aston Villa", "start": _fut, "p": 0.2, "odds": 6,
     "edge": -0.05, "league": "🏆 Champions League", "info": ["📊", "🧩 Crear apuesta: A + B → prob. 45%, cuota justa 2.22"]},
    {"key": "nba|x|A|B", "market": "home", "match": "A vs B", "start": _fut, "p": 0.6, "odds": 1.8, "edge": 0.0, "info": ["📊"]}]}
cards.clear(); radar.A.send_card = cards.append
radar.combos_card(_cs); radar.combos_card(_cs)
assert len(cards) == 1 and cards[0].count("Barcelona 🆚 Aston Villa") == 1 and "Córners 6.4" in cards[0] and "A 🆚 B" not in cards[0]
assert all(s.get("combo_anunciada") for s in _cs["signals"][:2])
import multisport as _MS
_eb = {"signals": [{"key": "ucl|2026-11-03|Barcelona|Aston Villa", "match": "Barcelona vs Aston Villa", "start": _fut, "info": ["📊"]}]}
_el, _MS.football_stats.euro_lines = _MS.football_stats.euro_lines, lambda h, a, t: ["", "📐 <b>Estadísticas esperadas</b>", "🧩 x"]
_MS.euro_backfill(_eb, None); _MS.euro_backfill(_eb, None)
_MS.football_stats.euro_lines = _el
assert _eb["signals"][0]["info"] == ["📊", "", "📐 <b>Estadísticas esperadas</b>", "🧩 x"]  # una sola vez
print("ok tarjeta de combinadas")

# historial real de las combinadas de estadísticas (football_stats.settle_sgp / sgp_history)
football_stats.SGP_LOG = Path(tempfile.mkdtemp()) / "sgp_log.csv"
_leg = lambda v, s, l, p: {"var": v, "sign": s, "line": l, "p": p, "text": v}
_sg = {"sgp": {"E0|2026-11-01|Leeds|Hull": {"match": "Leeds vs Hull", "start": "2026-11-01T15:00:00+00:00", "combos": [
          {"kind": "corta", "p": 0.4, "legs": [_leg("corners_h", 1, 5.5, 0.6), _leg("goals_t", -1, 2.5, 0.65)]},
          {"kind": "larga", "p": 0.2, "legs": [_leg("corners_h", 1, 5.5, 0.6), _leg("cards_a", 1, 1.5, 0.6)]}]},
       "E0|2026-11-02|Spurs|Wolves": {"match": "Spurs vs Wolves", "start": "2026-11-02T15:00:00+00:00", "combos": [
          {"kind": "corta", "p": 0.4, "legs": [_leg("corners_t", 1, 9.5, 0.6), _leg("goals_t", 1, 2.5, 0.6)]}]}}}
_rec = {"E0|2026-11-01|Leeds|Hull": {"FTHG": 1, "FTAG": 0, "HC": 7, "AC": 2, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
        "E0|2026-11-02|Spurs|Wolves": {"FTHG": 3, "FTAG": 1, "HC": float("nan"), "AC": float("nan")}}  # sin córners aún
assert football_stats.settle_sgp(_sg, _rec, pd.Timestamp("2026-11-03")) == 2
assert list(_sg["sgp"]) == ["E0|2026-11-02|Spurs|Wolves"]  # la de Spurs espera a que football-data traiga los córners
_h = pd.read_csv(football_stats.SGP_LOG)
assert _h.acierto.tolist() == [1, 0] and _h.piernas_acertadas.tolist() == [2, 1]
assert "cortas 1: previsto 40%, acertadas 100%" in football_stats.sgp_history()
assert football_stats.settle_sgp(_sg, {}, pd.Timestamp("2026-11-20")) == 0 and not _sg["sgp"]  # >10 días: fuera
print("ok historial de combinadas")

# reloj propio (el cron de GitHub casi no salta): espera, lanza analista y se relanza; un solo reloj a la vez
_wf = (Path(__file__).parent / ".github/workflows/reloj.yml").read_text()
assert "gh workflow run analista.yml" in _wf and "gh workflow run reloj.yml" in _wf and "cancel-in-progress: true" in _wf
assert "actions: write" in _wf and "workflow_dispatch" in (Path(__file__).parent / ".github/workflows/analista.yml").read_text()
print("ok reloj")

# FC 27 con cuotas reales de Unibet (kambi.esoccer_events / esoccer.market_report); sin stake hasta demostrar ROI
import esoccer
_ev = lambda i, h, a, path: {"event": {"id": i, "homeName": h, "awayName": a, "start": "2026-11-03T12:00:00Z",
                                        "path": [{"englishName": x} for x in path]},
                              "betOffers": [{"outcomes": [{"type": "OT_ONE", "odds": 2100}, {"type": "OT_CROSS", "odds": 4000},
                                                          {"type": "OT_TWO", "odds": 2900}]}]}
_kg, _kr = kambi.get, kambi._raw
kambi._raw = [_ev(1, "Arsenal (Kray) Esports", "Chelsea (Boulevard) Esports", ["Football", "Esports Battle"]),
              _ev(2, "Arsenal", "Chelsea", ["Football", "England", "Premier League"])]
kambi.get = lambda path: {"events": []}
_es = kambi.esoccer_events()
assert len(_es) == 1 and (_es[0]["p1"], _es[0]["p2"]) == ("kray", "boulevard") and _es[0]["odds"] == {"1": 2.1, "X": 4.0, "2": 2.9}
assert [e["home"] for e in kambi.events()] == ["Arsenal"] or kambi._events is not None  # el fútbol normal sigue sin esports
_f = kambi.find_esoccer("Boulevard", "Kray", pd.Timestamp("2026-11-03T12:10:00Z"), _es)
assert _f["odds"]["1"] == 2.9 and _f["odds"]["2"] == 2.1  # orden de Esoccer Battle
assert kambi.find_esoccer("Kray", "Boulevard", pd.Timestamp("2026-11-03T13:00:00Z"), _es) is None  # otra hora
kambi.get, kambi._raw = _kg, _kr
_tmp = Path(tempfile.mkdtemp())
esoccer.ODDS, _eh = _tmp / "odds.csv", esoccer.HIST
esoccer.HIST = _tmp / "matches.csv"
pd.DataFrame({"id": [1, 2, 3], "s1": [3, 1, 0], "s2": [1, 1, 2]}).to_csv(esoccer.HIST, index=False)
esoccer.save_odds([{"id": i, "o1": 2.0, "oX": 4.0, "o2": 3.0, "m1": m1, "mX": 0.2, "m2": 0.8 - m1} for i, m1 in ((1, 0.6), (2, 0.6), (3, 0.3))])
_mr = esoccer.market_report()
assert _mr["partidos"] == 3 and _mr["apuestas"] == 3 and abs(_mr["roi"] - (1 - 1 + 2) / 3) < 1e-3  # gana 1 (+1), empate en 2 (−1), gana 3 a 3.0 (+2)
esoccer.HIST = _eh
assert "s.get(\"sin_stake\")" in Path(bankroll.__file__).read_text()  # la banca no apuesta las señales sin_stake
print("ok FC 27 con cuotas de Unibet")

# registro de todas las estadísticas de Unibet (no solo córners) e informe modelo vs casa
_bo = {"betOffers": [
    {"criterion": {"englishLabel": "Total Cards", "lifetime": "FULL_TIME"},
     "outcomes": [{"type": "OT_OVER", "odds": 1900, "line": 4500, "status": "OPEN"}, {"type": "OT_UNDER", "odds": 1850, "line": 4500, "status": "OPEN"}]},
    {"criterion": {"englishLabel": "Total Shots on Target by Leeds", "lifetime": "FULL_TIME"},
     "outcomes": [{"type": "OT_OVER", "odds": 2000, "line": 4500, "status": "OPEN"}, {"type": "OT_UNDER", "odds": 1750, "line": 4500, "status": "OPEN"}]},
    {"criterion": {"englishLabel": "Total Corners", "lifetime": "FULL_TIME"},
     "outcomes": [{"type": "OT_OVER", "odds": 1800, "line": 9500, "status": "OPEN"}, {"type": "OT_UNDER", "odds": 1950, "line": 9500, "status": "OPEN"}]}]}
assert set(kambi.parse(_bo, "Leeds", "Hull")) == {("corners", "t")}  # por defecto, solo córners (lo que se apuesta)
_pa = kambi.parse(_bo, "Leeds", "Hull", ("corners", *kambi.STAT_LABELS))
assert _pa[("cards", "t")] == {4.5: {"o": 1.9, "u": 1.85}} and _pa[("sot", "h")][4.5]["o"] == 2.0 and kambi.LABELS_SEEN["Total Cards"] >= 1
football_stats.STATS_ODDS = Path(tempfile.mkdtemp()) / "k.csv"
_e2, football_stats.expected = football_stats.expected, lambda *a: (2.5, 2.0, 4.5, 2.5, 2.5)
_pp = football_stats.PARAMS
football_stats.PARAMS = Path(tempfile.mkdtemp()) / "p.json"
football_stats.PARAMS.write_text(json.dumps({s: {"r": 50, "r_team": 50} for s in football_stats.STATS}))
assert football_stats.log_offers(None, "E0", "Leeds", "Hull", None, "2026-11-01T15:00:00+00:00", _pa) == 3
football_stats.expected, football_stats.PARAMS = _e2, _pp
_hd = pd.DataFrame({"Div": ["E0"], "Date": [pd.Timestamp("2026-11-01")], "HomeTeam": ["Leeds"], "AwayTeam": ["Hull"],
                    "FTHG": [1], "FTAG": [0], "HC": [6], "AC": [5], "HY": [3], "AY": [3], "HST": [6], "AST": [2], "HF": [9], "AF": [9]})
_rp = football_stats.stats_odds_report(_hd)
assert _rp["cards", "t"]["n"] == 1 and _rp["corners", "t"]["n"] == 1 and _rp["sot", "h"]["n"] == 1
print("ok registro de estadísticas de Unibet")

# simulación de FC 27: pronósticos enviados → resultado real (esoccer.simulation / sim_lines / radar.fc_sim_card)
_t2 = Path(tempfile.mkdtemp())
esoccer.PRED, _eh2 = _t2 / "pred.csv", esoccer.HIST
esoccer.HIST = _t2 / "m.csv"
pd.DataFrame({"id": [10, 11, 12], "s1": [6, 2, 1], "s2": [2, 2, 3]}).to_csv(esoccer.HIST, index=False)
_sim_state = {"signals": [
    {"sport": "🎮⚽", "key": "fc|10", "match": "Kril vs Fireline", "start": "2026-11-03T12:00:00+00:00", "market": "H", "p": 0.8,
     "odds": 1.25, "book": "cuota justa (sin casa)", "league": "🎮⚽ FC 27 · Volta",
     "info": ["x", "Goles esperados: Kril 5.44 · Fireline 2.56", "Total: más de 8.5 goles 41% · menos 59%", "Hándicap Kril: −1.5 68% · −2.5 54%"]},
    {"sport": "🎮⚽", "key": "fc|11", "match": "Kril vs Haiko", "start": "2026-11-03T12:10:00+00:00", "market": "H", "p": 0.7,
     "odds": 1.9, "book": "Unibet"},  # ya empezado: sin info, solo el ganador
    {"sport": "🎮⚽", "key": "fc|12", "match": "A vs B", "start": "2026-11-03T12:20:00+00:00", "market": "A", "p": 0.6, "odds": 2.1, "book": "Unibet"}]}
assert esoccer.backfill_predictions(_sim_state) == 3 and esoccer.backfill_predictions(_sim_state) == 0
_d = esoccer.simulation()
assert _d.ok_1x2.tolist() == [True, False, True] and _d.ok_tot.iloc[0] == 1.0 and _d.ok_h15.iloc[0] == 1.0  # 6-2: menos de 8.5 ✅, −1.5 ✅
assert _d.ok_tot.iloc[1:].isna().all() and abs(_d.g_unibet.sum() - (-1 + 1.1)) < 1e-9
_sl = "\n".join(esoccer.sim_lines())
assert "3 pronósticos jugados" in _sl and "Kril <b>6-2</b> Fireline" in _sl and "con cuota de Unibet (2)" in _sl
cards.clear(); _ss = {}
radar.fc_sim_card(_ss); radar.fc_sim_card(_ss)
assert len(cards) == 1 and "FC 27 · SIMULACIÓN" in cards[0] and _ss["fc_sim_n"] == 3
esoccer.HIST = _eh2
print("ok simulación FC 27")

# simulación de combinadas de FC 27: ganador + total del mismo partido y combinada de cada tarjeta
_jw = esoccer.joint_win_total(2.0, 1.0, "1", 2.5, True)
_p1 = esoccer.markets(2.0, 1.0)[0]["1"]
assert 0 < _jw < _p1 and abs(esoccer.joint_win_total(2.0, 1.0, "1", 2.5, False) + _jw - _p1) < 1e-9
_t3 = Path(tempfile.mkdtemp()); esoccer.PRED, _eh3 = _t3 / "p.csv", esoccer.HIST
esoccer.HIST = _t3 / "m.csv"
pd.DataFrame({"id": [1, 2, 3], "s1": [3, 2, 0], "s2": [1, 0, 1]}).to_csv(esoccer.HIST, index=False)
esoccer.save_predictions([
    {"id": 1, "start": "a", "p1": "A", "p2": "B", "l1": 2.0, "l2": 1.0, "pick": "1", "p_pick": 0.6, "line": 2.5, "p_over": 0.6, "fav": "1", "p_h15": 0.4, "odds": 1.8, "lote": "L1"},
    {"id": 2, "start": "b", "p1": "C", "p2": "D", "l1": 2.0, "l2": 1.0, "pick": "1", "p_pick": 0.6, "line": 2.5, "p_over": 0.6, "fav": "1", "p_h15": 0.4, "odds": 1.7, "lote": "L1"},
    {"id": 3, "start": "c", "p1": "E", "p2": "F", "l1": 2.0, "l2": 1.0, "pick": "1", "p_pick": 0.6, "line": 2.5, "p_over": 0.6, "fav": "1", "p_h15": 0.4, "odds": None, "lote": "L2"}])
_d3 = esoccer.simulation()
assert _d3.ok_wt.tolist() == [1.0, 0.0, 0.0]  # 3-1 gana y más de 2.5 ✅; 2-0 gana pero menos ❌; 0-1 pierde ❌
_lc = esoccer.lot_combos(_d3)
assert len(_lc) == 1 and _lc.ok.iloc[0] and abs(_lc.p.iloc[0] - 0.36) < 1e-9 and abs(_lc.g_unibet.iloc[0] - (1.8 * 1.7 - 1)) < 1e-9
assert "Combinada de cada tarjeta" in "\n".join(esoccer.sim_lines())
esoccer.HIST = _eh3
print("ok simulación de combinadas FC 27")

# FC 27: recalibración del 1X2 (train_fc_calib.py → esoccer.calibrate)
_pc = {"1": 0.7, "X": 0.12, "2": 0.18}
_c1 = esoccer.calibrate(_pc, [1.0, 0.0]); assert all(abs(_c1[k] - _pc[k]) < 1e-9 for k in _pc)  # identidad
_c2 = esoccer.calibrate(_pc, [1.16, 0.32]); assert abs(sum(_c2.values()) - 1) < 1e-9 and _c2["X"] > _pc["X"]
import train_fc_calib
_Q = train_fc_calib.calibrate_arr(np.array([[0.7, 0.12, 0.18]]), 1.16, 0.32)[0]
assert np.allclose(_Q, [_c2["1"], _c2["X"], _c2["2"]])
print("ok recalibración FC 27")

# props de jugador CS2: kills esperados en la serie (train_cs2_props.py → cs2_props)
import cs2_props
_P0 = {"half_life": 50, "k": 300, "c": 0.06, "r": 10.0, "rounds_per_map": 21.0, "n_test": 52293, "ou": 0.6561, "ou_ref": 0.6821}
_ps, _pl = cs2_props.params, cs2_props._state.get("kpr")
cs2_props.params = lambda: _P0
cs2_props._state["kpr"] = ({1: (70.0, 100.0), 2: (60.0, 100.0)}, 0.65, {10: [(1, "ZywOo")], 20: [(2, "s1mple")]}, _P0)
_ex = cs2_props.expected(10, 20, 0.6, 3, esoccer.np and __import__("esports").bo3_scores(0.6))
assert len(_ex) == 2 and _ex[0][2] > _ex[1][2] and _ex[0][3] < _ex[0][2] < _ex[0][4]  # más KPR y equipo favorito → más kills
_e1 = cs2_props.expected(10, 20, 0.6, 1)
assert _e1[0][2] < _ex[0][2]  # BO1 (1 mapa) < BO3 (2-3 mapas)
_cl2 = cs2_props.card_lines({"t1_id": 10, "t2_id": 20, "team1": "Vitality", "team2": "NaVi", "bo": 3}, 0.6, __import__("esports").bo3_scores(0.6))
assert "Kills esperados" in _cl2[1] and "52.293 casos" in _cl2[1] and _cl2[2].startswith("Vitality: ZywOo")
cs2_props.params = _ps; cs2_props._state.clear()
print("ok props CS2")
