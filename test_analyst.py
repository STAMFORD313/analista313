import os as _os0
for _k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GH_MODELS_TOKEN"): _os0.environ.pop(_k, None)  # el chat de los tests nunca llama a una IA (en Actions hay clave)
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
fc = {"bank": {"bets": []}, "signals": [{"key": f"fc|{i}", "market": "H", "pick": f"P{i}", "match": f"P{i} vs Q{i}", "odds": 2.0, "p": 0.55,
      "implied": 0.5, "edge": 0.1, "book": "Unibet", "sport": "🎮⚽", "start": "2099-01-01T00:00Z",
      "info": ["x", "Goles esperados 1-1", "Ganador P 50%", "Total", "Hándicap", "Forma"]} for i in range(7)]}
import radar
cards.clear(); radar.A.send_card = cards.append
radar.fc_card(fc); radar.fc_card(fc)
assert len(cards) == 1 and cards[0].count("🆚") == 5 and "Gana P0" in cards[0] and "Forma" not in cards[0], cards
fc["fc_sent"] = "2000-01-01T00:00Z"; radar.fc_card(fc)
assert len(cards) == 2 and cards[1].count("🆚") == 2
fc["signals"][0]["result"] = True; fc["signals"][0]["start"] = "2026-11-02"; fc["signals"][0]["book"] = "cuota justa (sin casa)"
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
assert len(_sl) == 4 and "FaZe gana 2-1" in _sl[2] and "casa" in _sl[2] and "casa" not in _sl[3] and "como mucho UNA" in _sl[1]
_j = 1 / float(_sl[2].split("cuota justa ")[1].split()[0])
assert abs(_j - _sc[2, 1]) < 0.01  # prob. del marcador 2-1 de bo3_scores
# Thunderpick incoherente (MongolZ-FURIA 4-nov: MongolZ 2-0 al 46 % siendo ganador al 28 %): se ignoran los hándicaps
_bad = {"team_1": {"coeff": 3.25}, "team_2": {"coeff": 1.29}}
_bo = {"team_1_handicap_over_1_5": 1.09, "team_2_handicap_under_1_5": 6.0, "team_1_handicap_under_1_5": 2.02, "team_2_handicap_over_1_5": 1.7}
assert not _E.consistent(_bad, _bo) and _E.consistent(_bet, {x["bet_type"]: x["coeff"] for x in _bet["additional_markets"]})
assert _E.extra_markets({**_bad, "additional_markets": [{"bet_type": k, "coeff": v} for k, v in _bo.items()]}, 0.3, {"team1": "MongolZ", "team2": "FURIA"}) == []
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
_cs["sgp"] = {"ucl|x|Barcelona|Aston Villa": {"match": "Barcelona vs Aston Villa", "start": _fut, "combos": [
                  {"kind": "corta", "p": 0.45, "legs": [{"text": "Menos de 4.5 córners de Aston Villa"}]},  # sin cuota: no se apuesta
                  {"kind": "valor", "p": 0.40, "odds": 3.2, "edge": 0.28, "sim": True,
                   "legs": [{"text": "Más de 2.5 goles"}, {"text": "Menos de 2.5 tarjetas de Barcelona"}]}]},
              "nba|x|A|B": {"match": "A vs B", "start": _fut, "combos": [{"kind": "nba", "p": 0.3, "odds": 3.0, "edge": -0.1, "sim": False, "legs": [{"text": "Gana A"}]}]}}
cards.clear(); radar.A.send_card = cards.append
radar.combos_card(_cs); radar.combos_card(_cs)
assert len(cards) == 1 and cards[0].count("Barcelona 🆚 Aston Villa") == 1 and "  ↳ Más de 2.5 goles" in cards[0] and "Apuesta: 1.00 €" in cards[0]
assert "A 🆚 B" not in cards[0] and "córners" not in cards[0] and _cs["sgp"]["ucl|x|Barcelona|Aston Villa"]["anunciada"]  # solo lo que se apuesta
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
assert "Total (1 apuestas de 1 € con valor)" in _sl and "A 🆚 B (1-3)" in _sl and "✅ Ganado" in _sl and "Kril" not in _sl  # solo las de valor
cards.clear(); _ss = {}
radar.fc_sim_card(_ss); radar.fc_sim_card(_ss)
assert len(cards) == 1 and "FC 27 · RESULTADOS" in cards[0] and _ss["fc_sim_n"] == 3
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
assert "Total" in "\n".join(esoccer.sim_lines())
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

# partidos ya analizados: se vuelve a mirar Unibet hasta encontrarlos (analyst.kambi_recheck)
import analyst as _AN
_ko, _lo, _vs = _AN.kambi.offers, _AN.football_stats.log_offers, _AN.football_stats.value_signals
_AN.kambi.offers = lambda *a, **k: {}
_ks = {"signals": [{"key": "E1|2026-11-04|Cardiff|West Ham", "market": "H", "info": ["📊 x"], "context": "c"}], "kambi_ok": []}
_row = {"Div": "E1", "HomeTeam": "Cardiff", "AwayTeam": "West Ham"}
_st = pd.Timestamp("2026-11-04 20:00", tz="Europe/London")
_AN.kambi_recheck(_ks, None, _row, "E1|2026-11-04|Cardiff|West Ham", _st, None)
assert _ks["kambi_ok"] == [] and len(_ks["signals"]) == 1  # Unibet aún no lo tiene: se reintenta en el próximo run
_logged = []
_AN.kambi.offers = lambda *a, **k: {("corners", "t"): {9.5: {"o": 1.9, "u": 1.9}}, ("cards", "t"): {4.5: {"o": 1.8, "u": 2.0}}}
_AN.football_stats.log_offers = lambda *a: _logged.append(a[-1])
_AN.football_stats.value_signals = lambda df, div, h, a, t, off: [{"market": "corners_t_o", "line": 9.5, "odds": 1.9, "p": 0.6, "edge": 0.14, "pick": "Más de 9.5 córners"}] if list(off) == [("corners", "t")] else []
_AN.kambi_recheck(_ks, None, _row, "E1|2026-11-04|Cardiff|West Ham", _st, None)
assert _ks["kambi_ok"] == ["E1|2026-11-04|Cardiff|West Ham"] and len(_logged[0]) == 2  # registra todas las estadísticas
assert _ks["signals"][-1]["market"] == "corners_t_o" and _ks["signals"][-1]["info"] == ["📊 x"]  # señal solo de córners, con su info
_AN.kambi.offers, _AN.football_stats.log_offers, _AN.football_stats.value_signals = _ko, _lo, _vs
print("ok Unibet se vuelve a mirar")

# banco de pruebas: todas las candidatas liquidadas a 1 u por deporte y mercado (radar.paper_lines / paper_card)
_lg = B_LOG = bankroll.LOG
bankroll.LOG = Path(tempfile.mkdtemp()) / "s.csv"
pd.DataFrame({"deporte": ["🏀"] * 25 + ["⚽"] * 3 + ["🎮⚽"], "mercado": ["home"] * 25 + ["corners_t_o"] * 3 + ["H"],
              "p": 0.6, "odds": 2.0, "gano": [1] * 20 + [0] * 5 + [0, 0, 0] + [1], "retorno": [2.0] * 20 + [0] * 5 + [0, 0, 0] + [2],
              "casa": ["DraftKings"] * 25 + ["Unibet"] * 3 + ["cuota justa (sin casa)"]}).to_csv(bankroll.LOG, index=False)
_pl = radar.paper_lines()
assert "28 candidatas" in _pl[0] and any(l.startswith("🟢 NBA / baloncesto · ganador (25)") for l in _pl)
assert any(l.startswith("⚪ Fútbol · córners (3)") for l in _pl)  # 3 partidos: ruido aunque pierdan todos
cards.clear(); _ps = {}; radar.paper_card(_ps); radar.paper_card(_ps); assert len(cards) == 1 and "BANCO DE PRUEBAS" in cards[0]
bankroll.LOG = _lg
print("ok banco de pruebas")

# props de jugador de Unibet (kambi.player_props → data/kambi_player_props.csv), solo registro
_fe, _oc = kambi.find, dict(kambi._offer_cache)
kambi.find = lambda h, a, s: {"id": 99, "home": h, "away": a}
kambi._offer_cache[("offer", 99)] = {"betOffers": [
    {"criterion": {"englishLabel": "Player's shots on target (Settled using Opta data)"},
     "outcomes": [{"participant": "Mbappé, Kylian", "type": "OT_OVER", "line": 1500, "odds": 2100}]},
    {"criterion": {"englishLabel": "To Score"}, "outcomes": [{"participant": "Vinícius Júnior", "type": "OT_YES", "odds": 2600}]},
    {"criterion": {"englishLabel": "Total Goals"}, "outcomes": [{"type": "OT_OVER", "line": 2500, "odds": 1900}]}]}
_pp = kambi.player_props("Real Madrid", "Barcelona", None)
assert len(_pp) == 2 and _pp[0] == {"mercado": "Player's shots on target", "jugador": "Mbappé, Kylian", "tipo": "OT_OVER", "linea": 1.5, "cuota": 2.1}
_AN.PROPS_LOG = Path(tempfile.mkdtemp()) / "pp.csv"
assert _AN.log_player_props("SP1", "Real Madrid", "Barcelona", "2026-11-05T20:00:00+00:00") == 2 and len(pd.read_csv(_AN.PROPS_LOG)) == 2
kambi.find = _fe; kambi._offer_cache.clear(); kambi._offer_cache.update(_oc)
print("ok props de jugador de Unibet")

# cuotas del partido como feature de las estadísticas (train_market_stats.py → football_stats.market_factor)
import train_market_stats
_s, _g = train_market_stats.market_features(pd.DataFrame([{"AvgH": 1.5, "AvgD": 4.0, "AvgA": 6.0, "Avg>2.5": 1.8, "Avg<2.5": 2.0}]))
_t = 1 / 1.5 + 1 / 4 + 1 / 6
assert abs(_s[0] - (1 / 1.5 - 1 / 6) / _t) < 1e-9 and abs(_g[0] - np.log((1 / 1.8) / (1 / 2.0))) < 1e-9
_s2, _ = train_market_stats.market_features(pd.DataFrame([{"BbAvH": 2.0, "BbAvD": 3.4, "BbAvA": 3.8, "BbAv>2.5": 1.9, "BbAv<2.5": 1.9}]))
assert _s2[0] > 0  # temporadas antiguas: columnas BbAv
football_stats.MATCH_MK.clear()
_Pm = {"market": {"corners": {"a": 0.35, "b": 0.0, "g_mean": 0.1}}}
assert football_stats.market_factor("corners", "City", "Burnley", _Pm) == (1.0, 1.0)  # sin cuotas: sin ajuste
football_stats.set_market({"HomeTeam": "City", "AwayTeam": "Burnley", "AvgH": 1.2, "AvgD": 7, "AvgA": 13, "Avg>2.5": 1.4, "Avg<2.5": 3.0})
_fh, _fa = football_stats.market_factor("corners", "City", "Burnley", _Pm)
assert _fh > 1 > _fa and abs(_fh * _fa - 1) < 1e-9  # el favorito saca más córners y el rival menos
assert football_stats.market_factor("cards", "City", "Burnley", _Pm) == (1.0, 1.0)
football_stats.MATCH_MK.clear()
print("ok cuotas como feature de estadísticas")

# ESPN: fútbol por jugador y árbitro (espn_soccer.compact) y NBA con rebotes/asistencias/triples (nba_players.compact)
import espn_soccer, nba_players
_sd = {"header": {"competitions": [{"competitors": [{"homeAway": "home", "team": {"id": "1"}}, {"homeAway": "away", "team": {"id": "2"}}]}]},
       "gameInfo": {"officials": [{"displayName": "Michael Oliver", "position": {"name": "Referee"}},
                                  {"displayName": "X", "position": {"name": "Assistant Referee"}}]},
       "boxscore": {"teams": [{"team": {"id": "1"}, "statistics": [{"name": "wonCorners", "displayValue": "7"}, {"name": "shotsOnTarget", "displayValue": "5"}]},
                              {"team": {"id": "2"}, "statistics": [{"name": "wonCorners", "displayValue": "3"}]}]},
       "rosters": [{"team": {"id": "1"}, "homeAway": "home", "roster": [
           {"starter": True, "athlete": {"id": "9", "displayName": "Kylian Mbappé"}, "position": {"abbreviation": "F"},
            "stats": [{"name": "totalShots", "value": 5.0}, {"name": "shotsOnTarget", "value": 3.0}, {"name": "totalGoals", "value": 2.0}]},
           {"starter": False, "athlete": {"id": "8", "displayName": "Suplente"}, "stats": []}]},
                   {"team": {"id": "2"}, "homeAway": "away", "roster": [
           {"starter": True, "athlete": {"id": "1", "displayName": "Portero"}, "position": {"abbreviation": "G"},
            "stats": [{"name": "saves", "value": 4.0}, {"name": "shotsFaced", "value": 6.0}]}]}]}
_sc = espn_soccer.compact(_sd)
assert _sc["referee"] == "Michael Oliver" and _sc["team_stats"]["h"]["wonCorners"] == 7.0 and _sc["team_stats"]["a"]["wonCorners"] == 3.0
assert len(_sc["players"]) == 2  # el suplente que no jugó no se guarda
_mb = dict(zip(["id", "nombre", "lado", "titular", "pos", *espn_soccer.PSTATS], _sc["players"][0]))
assert _mb["lado"] == "h" and _mb["totalGoals"] == 2 and _mb["shotsOnTarget"] == 3 and _mb["totalShots"] == 5
assert _sc["players"][1][2] == "a" and _sc["players"][1][5 + espn_soccer.PSTATS.index("saves")] == 4
_nd = {"boxscore": {"players": [{"team": {"displayName": "Lakers"}, "statistics": [{
    "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"],
    "athletes": [{"athlete": {"id": "3", "displayName": "LeBron James"}, "starter": True,
                  "stats": ["35", "10-20", "3-7", "4-5", "1", "7", "8", "9", "2", "1", "3", "2", "+5", "27"]}]}]}]}}
_nb = nba_players.compact(_nd)["teams"][0]["players"][0]
assert _nb[:7] == ["3", "LeBron James", 35.0, 27.0, 5.0, True, False] and _nb[7:13] == [8.0, 9.0, 3.0, 2.0, 1.0, 3.0] and len(_nb) == nba_players.N_COLS
print("ok ESPN jugadores fútbol y NBA")

# combinadas simuladas con la cuota de Unibet de cada pierna (football_stats.price_combos / sim_lines / settle_sgp)
_cb = [{"kind": "corta", "p": 0.40, "legs": [_leg("corners_h", 1, 5.5, 0.6), _leg("goals_t", -1, 2.5, 0.65)]},
       {"kind": "larga", "p": 0.20, "legs": [_leg("corners_h", 1, 5.5, 0.6), _leg("cards_a", 1, 1.5, 0.6)]}]
football_stats.price_combos(_cb, {("corners", "h"): {5.5: {"o": 1.8, "u": 2.0}}}, {"Avg>2.5": 1.7, "Avg<2.5": 2.1})
assert _cb[0]["odds"] == round(1.8 * 2.1, 2) and _cb[0]["sim"] and abs(_cb[0]["edge"] - (0.4 * 3.78 - 1)) < 1e-3
assert "odds" not in _cb[1]  # la tarjeta del visitante no tiene cuota: no se simula
assert football_stats.sim_lines(_cb)[0].startswith("🎯 Simulada (crear apuesta): cuota Unibet aprox. 3.78")
football_stats.SGP_LOG = Path(tempfile.mkdtemp()) / "sgp2.csv"
_sg2 = {"sgp": {"E0|2026-11-01|Leeds|Hull": {"match": "Leeds vs Hull", "start": "2026-11-01T15:00:00+00:00", "combos": _cb}}}
football_stats.settle_sgp(_sg2, {"E0|2026-11-01|Leeds|Hull": {"FTHG": 1, "FTAG": 0, "HC": 7, "AC": 2, "HY": 1, "AY": 3, "HR": 0, "AR": 0}}, pd.Timestamp("2026-11-03"))
_h2 = pd.read_csv(football_stats.SGP_LOG)
assert _h2.simulada.tolist() == [1, 0] and abs(_h2.ganancia.iloc[0] - 2.78) < 1e-9 and "Simuladas a 1 u" in football_stats.sgp_history()
print("ok combinadas simuladas con cuota de Unibet")

# NBA/NFL/NHL: combinada «ganador + total» simulada con DraftKings y liquidada con el marcador (multisport.us_combo)
import multisport as _MS2
_ml, _MS2.moneyline = _MS2.moneyline, lambda odds: (1.5, 2.8)
_usg = {"date": "2026-11-04T00:00Z", "home": "Lakers", "away": "Celtics", "odds": None}
_tc = [{"market": "over", "p": 0.55, "odds": 1.91}, {"market": "under", "p": 0.45, "odds": 1.91}]
_us = {"signals": [{"key": "nba|2026-11-04|Lakers|Celtics", "market": "home", "info": ["📊"]}]}
_mcs, _MS2._MC = _MS2._MC, {}  # sin el juez del mercado (peso 0.5) para probar la mecánica
_MS2.us_combo(_us, "nba", _usg, "Lakers", "Celtics", 0.70, _tc, 224.5)
_MS2._MC = _mcs
_MS2.moneyline = _ml
_cbu = _us["sgp"]["nba|2026-11-04|Lakers|Celtics"]["combos"][0]
assert [l["text"] for l in _cbu["legs"]] == ["Gana Lakers", "Más de 224.5 puntos"] and _cbu["kind"] == "nba"
assert _cbu["odds"] == round(1.5 * 1.91, 2) and abs(_cbu["edge"] - (_cbu["p"] * _cbu["odds"] - 1)) < 2e-3
_MS2.attach_combo_lines(_us); assert any(l.startswith("🧩 Combinada: Gana Lakers") for l in _us["signals"][0]["info"])
football_stats.SGP_LOG = Path(tempfile.mkdtemp()) / "sgp3.csv"
_dfu = pd.DataFrame([{"date": "2026-11-04T00:00Z", "home": "Lakers", "away": "Celtics", "hs": 120, "as": 110, "res": 1.0}])
assert _MS2.settle_us_combos(_dfu, "nba", _us) == 1 and not _us["sgp"]
_lgu = pd.read_csv(football_stats.SGP_LOG); assert _lgu.acierto.iloc[0] == 1 and _lgu.tipo.iloc[0] == "nba"
print("ok combinadas NBA simuladas")

# Champions/Europa: liquidación de combinadas con las estadísticas de ESPN (espn_soccer.recent_rows)
_el2 = espn_soccer.load
_fut2 = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%dT20:00Z")
espn_soccer.load = lambda code: [{"date": _fut2, "home": "Barcelona", "away": "Aston Villa", "hs": 2, "as": 1,
                                  "team_stats": {"h": {"wonCorners": 7, "yellowCards": 1, "redCards": 0, "foulsCommitted": 10, "shotsOnTarget": 6},
                                                 "a": {"wonCorners": 3, "yellowCards": 2, "redCards": 1, "foulsCommitted": 12, "shotsOnTarget": 2}}}] if code == "ucl" else []
_rr = espn_soccer.recent_rows()
espn_soccer.load = _el2
_kk = f"ucl|{_fut2[:10]}|Barcelona|Aston Villa"
assert _rr[_kk]["HC"] == 7 and _rr[_kk]["AY"] == 3 and _rr[_kk]["FTHG"] == 2
assert football_stats.leg_hit({"var": "sot_a", "sign": -1, "line": 4.5}, _rr[_kk]) is True  # menos de 4.5 tiros a puerta del Villa ✅
print("ok combinadas de Champions con ESPN")

# Kambi: nombres reales «… (Settled using Opta data)» y «Total Cards - Equipo»; combinada solo con piernas con cuota
_bo2 = {"betOffers": [
    {"criterion": {"englishLabel": "Total Shots on Target by Leeds (Settled using Opta data)", "lifetime": "FULL_TIME"},
     "outcomes": [{"type": "OT_OVER", "odds": 2000, "line": 4500, "status": "OPEN"}, {"type": "OT_UNDER", "odds": 1750, "line": 4500, "status": "OPEN"}]},
    {"criterion": {"englishLabel": "Total Cards - Hull", "lifetime": "FULL_TIME"},
     "outcomes": [{"type": "OT_OVER", "odds": 1900, "line": 1500, "status": "OPEN"}, {"type": "OT_UNDER", "odds": 1850, "line": 1500, "status": "OPEN"}]}]}
_pa2 = kambi.parse(_bo2, "Leeds", "Hull", ("corners", *kambi.STAT_LABELS))
assert _pa2[("sot", "h")][4.5]["o"] == 2.0 and _pa2[("cards", "a")][1.5]["u"] == 1.85
_pool = [_leg("fouls_h", -1, 11.5, 0.7), _leg("sot_h", -1, 4.5, 0.62), _leg("cards_a", -1, 1.5, 0.6), _leg("goals_t", 1, 2.5, 0.58)]
for _l, _st in zip(_pool, ["fouls", "sot", "cards", "goals"]): _l["stat"] = _st
_pc = football_stats.priced_combo(_pool, _pa2, {"Avg>2.5": 1.9, "Avg<2.5": 1.9}, rho=None, cov={})
assert _pc and _pc["kind"] == "valor" and all(l["var"] != "fouls_h" for l in _pc["legs"])  # faltas sin mercado: fuera
assert abs(_pc["odds"] - np.prod([l["odds"] for l in _pc["legs"]])) < 0.01 and abs(_pc["edge"] - (_pc["p"] * _pc["odds"] - 1)) < 0.02
assert football_stats.priced_combo(_pool[:1], _pa2) is None
print("ok combinada con cuota de Unibet")

# props de jugador NBA (train_nba_props.py → nba_props.card_lines)
import nba_props
_np0, _ns0 = nba_props.params, dict(nba_props._state)
nba_props.params = lambda: {"a_min": 0.2, "a_rate": 0.05, "k": 50, "stats": {s: {"r": 20.0, "ou": 0.66, "ou_ref": 0.68} for s in
                                                                           ("puntos", "rebotes", "asistencias", "triples")}}
nba_props._state["teams"] = {"Miami Heat": [("Giannis Antetokounmpo", 34.0, {"puntos": 0.9, "rebotes": 0.35, "asistencias": 0.2, "triples": 0.01}, None),
                                            ("Bam Adebayo", 32.0, {"puntos": 0.5, "rebotes": 0.3, "asistencias": 0.12, "triples": 0.02}, None)],
                             "Boston Celtics": [("Derrick White", 30.0, {"puntos": 0.45, "rebotes": 0.12, "asistencias": 0.14, "triples": 0.07}, None)]}
_nl = nba_props.card_lines("Miami Heat", "Boston Celtics", ["Bam Adebayo"])
assert any(l.startswith("Giannis Antetokounmpo (Heat, 34 min): 30.6") for l in _nl) and not any("Bam" in l for l in _nl)
assert any(l.startswith("Derrick White (Celtics") for l in _nl)
nba_props.params = _np0; nba_props._state.clear(); nba_props._state.update(_ns0)
print("ok props NBA")

# props de jugador de fútbol con ESPN y liquidación de las cuotas de Unibet (football_players)
import football_players as FP, math
assert FP.group("CD-L") == "D" and FP.group("G") == "G" and FP.group("F") == "F" and FP.group("AM") == "M"
assert FP.norm("Kylian Mbappé") == "kylian mbappe" and abs(FP.prob(0.5, 1) - (1 - math.exp(-0.5))) < 1e-12
_fr = pd.DataFrame([{"date": pd.Timestamp("2026-10-0%d" % i, tz="UTC"), "game": str(i), "pid": "9", "name": "Kylian Mbappé", "team": "RM",
                     "starter": True, "pos": "F", "goles": 1.0, "asist": 0.0, "tiros_puerta": 2.0, "cards": 0.0} for i in range(1, 6)])
_fw, _, _ = FP.walk(_fr)
assert abs(_fw.goles.iloc[0] - 0.05) < 1e-9 and _fw.goles.iloc[-1] > 0.9  # sin historial: valor por defecto; luego aprende que marca siempre
_rows0, _log0 = FP.rows, FP.LOG
FP.rows = lambda: _fr.assign(date=_fr.date + pd.Timedelta(hours=20))
FP.LOG = Path(tempfile.mkdtemp()) / "pp.csv"
_kp = A_.DATA / "kambi_player_props.csv"
_kpb = _kp.read_bytes() if _kp.exists() else None
pd.DataFrame([{"mercado": "To Score", "jugador": "Kylian Mbappe", "tipo": "OT_YES", "linea": 1.0, "cuota": 1.9, "seen": "2026-10-05T10:00:00+00:00",
               "start": "2026-10-05T20:00:00+00:00", "div": "SP1", "home": "Real Madrid", "away": "Getafe"}]).to_csv(_kp, index=False)
try:
    _st = FP.settle()
    assert len(_st) == 1 and _st.acierto.iloc[0] == 1 and _st.p_casa.iloc[0] == round(1 / 1.9, 4) and 0.3 < _st.p_modelo.iloc[0] < 1
finally:
    if _kpb is None: _kp.unlink()
    else: _kp.write_bytes(_kpb)
    FP.rows, FP.LOG = _rows0, _log0
print("ok props de jugador de fútbol")

# formato único de boleto (boleto.py) y tarjeta de resultados de combinadas (radar.sgp_results_card)
import boleto
_b = boleto.boleto("Madrid vs Sevilla", [("Gana Madrid", True), ("Marca Mbappé", True), ("Más de 2.5 córners", True)], 5.6, 1.9)
assert _b[0] == "<b>Madrid 🆚 Sevilla</b>" and _b[1] == "  ↳ Gana Madrid ✅" and "✅ Ganado: <b>10.64 €</b>" in _b[-1]
_b2 = boleto.boleto("Madrid vs Sevilla", [("Gana Madrid", True), ("Marca Mbappé", False)], 3.0, 2.0)
assert "  ↳ Marca Mbappé ❌" in _b2 and "❌ Perdido: <b>2.00 €</b>" in _b2[-1]
_b3 = boleto.boleto("Madrid vs Sevilla", [("Gana Madrid", None)], 1.8, 2.0, when="05/11 · 21:00")
assert "✅" not in "".join(_b3) and "❌" not in "".join(_b3) and "Si acierta: 3.60 €" in _b3[-1]  # predicción: sin marcas
football_stats.SGP_LOG = Path(tempfile.mkdtemp()) / "sgp4.csv"
pd.DataFrame([{"fecha": "2026-11-03", "partido": "QPR vs Burnley", "tipo": "valor", "piernas": 2, "p": 0.4, "acierto": 0, "piernas_acertadas": 1,
               "p_piernas": 0.6, "cuota": 3.2, "ventaja": 0.28, "simulada": 1, "ganancia": -1.0,
               "piernas_txt": "Menos de 2.5 tarjetas de QPR + Menos de 5.5 tiros a puerta de QPR", "piernas_ok": "10"}]).to_csv(football_stats.SGP_LOG, index=False)
cards.clear(); _rs = {}
radar.sgp_results_card(_rs); radar.sgp_results_card(_rs)
assert len(cards) == 1 and "  ↳ Menos de 2.5 tarjetas de QPR ✅" in cards[0] and "  ↳ Menos de 5.5 tiros a puerta de QPR ❌" in cards[0]
assert "❌ Perdido: <b>1.00 €</b>" in cards[0] and "Total simuladas: 1 · -1.00 €" in cards[0]
print("ok formato boleto")

# quitar el margen con Shin (devig.py): suma 1 y carga más margen en la sorpresa que el reparto proporcional
import devig
_o = [1.5, 4.2, 7.0]
_ps, _pp = devig.shin(_o), devig.proportional(_o)
assert abs(_ps.sum() - 1) < 1e-9 and _ps[0] > _pp[0] and _ps[2] < _pp[2]
assert np.allclose(devig.shin([2.0, 2.0]), [0.5, 0.5]) and devig.shin(np.array([_o, _o])).shape == (2, 3)
print("ok Shin")

# la banca no apuesta cuotas > MAX_ODDS (sesgo favorito-sorpresa: el bot sobrevalora las sorpresas)
assert bankroll.MAX_ODDS == 2.5 and 's["odds"] > MAX_ODDS' in Path(bankroll.__file__).read_text()
print("ok cuota máxima")

# bot «nutrición»: buscador de estrategias walk-forward (estrategias.py) y su efecto en el stake
import estrategias
_rng = np.random.default_rng(1)
_dd = pd.DataFrame({"fecha": [f"2026-10-{1 + i % 28:02d}" for i in range(120)], "deporte": ["🏀"] * 60 + ["⚽"] * 60,
                    "mercado": ["home"] * 60 + ["H"] * 60, "odds": 1.9, "edge": 0.08, "casa": "DraftKings",
                    "gano": [1] * 50 + [0] * 10 + [0] * 55 + [1] * 5, "retorno": [1.9] * 50 + [0] * 10 + [0] * 55 + [1.9] * 5})
_es = estrategias.search(_dd)
assert any(r["segmento"] == "deporte=🏀" for r in _es["rentables"]) and any(r["segmento"] == "deporte=⚽" for r in _es["bloqueados"])
_eo = estrategias.OUT
estrategias.OUT = Path(tempfile.mkdtemp()) / "e.json"; estrategias.OUT.write_text(json.dumps(_es))
assert estrategias.factor({"sport": "⚽", "market": "H", "odds": 1.9, "edge": 0.08}) == 0.0
assert estrategias.factor({"sport": "🏀", "market": "home", "odds": 1.9, "edge": 0.08}) == 1.5
assert estrategias.factor({"sport": "🎮", "market": "over", "odds": 3.0, "edge": 0.2}) == 1.0  # ningún segmento con datos
estrategias.OUT = _eo
print("ok estrategias")

# equipo de bots expertos (nutricion.py): ejecuta los que tocan, aislados, y deja el parte
import nutricion as NU, sys
_ex, _stf = dict(NU.EXPERTS), NU.STATE
NU.STATE = Path(tempfile.mkdtemp()) / "x.json"
NU.EXPERTS.clear(); NU.EXPERTS.update({"bien": (0, 1, [sys.executable, "-c", "print('hecho')"], "a"),
                                       "mal": (0, 1, [sys.executable, "-c", "raise SystemExit(1)"], "b"),
                                       "diario": (24, 1, [sys.executable, "-c", "print(1)"], "c")})
_st1 = NU.run()
assert _st1["bien"]["ok"] and _st1["bien"]["salida"] == "hecho" and not _st1["mal"]["ok"] and "diario" in _st1  # uno falla, los demás siguen
assert not NU.due("diario", _st1, pd.Timestamp.now(tz="UTC")) and NU.due("bien", _st1, pd.Timestamp.now(tz="UTC"))
NU.EXPERTS.clear(); NU.EXPERTS.update(_ex); NU.STATE = _stf
print("ok equipo de expertos")

# peso del modelo por liga y mercado validado contra las cuotas de cierre (multisport.weight ← market_check.json)
import multisport as _MW
_mc0 = _MW._MC
_MW._MC = {"nba": {"ganador": {"w_opt": 0.1, "ll_2a_mitad_w_opt": 0.5322, "ll_2a_mitad_mercado": 0.5297}},
           "nfl": {"hándicap": {"w_opt": 0.7, "ll_2a_mitad_w_opt": 0.6880, "ll_2a_mitad_mercado": 0.6960}}}
_nfx, _MW._NFLV = _MW._NFLV, {}
assert _MW.weight("nba", "ganador") == 0.0 and _MW.weight("nfl", "hándicap") == 0.7 and _MW.weight("mlb", "ganador") == 0.5
_MW._NFLV = _nfx
assert _MW.signals(0.7, 1.9, 2.0, every=True, w=0.0)[0]["edge"] < 0.01  # peso 0 → prob. del mercado: sin ventaja
_MW._MC = _mc0
print("ok peso por liga y mercado")

# props de jugador NBA de Unibet (kambi.nba_props): todas las salidas con jugador
_kg2, _nb0 = kambi.get, kambi._nba
kambi._nba = [{"id": 7, "home": "Los Angeles Lakers", "away": "Boston Celtics", "start": pd.Timestamp("2026-11-05T00:30:00Z")}]
kambi.get = lambda path: {"betOffers": [
    {"criterion": {"englishLabel": "Points scored by the player - Including Overtime"},
     "outcomes": [{"participant": "LeBron James", "type": "OT_OVER", "line": 25500, "odds": 1870}, {"participant": "LeBron James", "type": "OT_UNDER", "line": 25500, "odds": 1950}]},
    {"criterion": {"englishLabel": "Moneyline"}, "outcomes": [{"type": "OT_ONE", "odds": 1500}]}]}
_np2 = kambi.nba_props("Los Angeles Lakers", "Boston Celtics", "2026-11-05T00:30:00Z")
assert len(_np2) == 2 and _np2[0]["jugador"] == "LeBron James" and _np2[0]["linea"] == 25.5 and kambi.LABELS_SEEN.get("NBA · Moneyline")
kambi.get, kambi._nba = _kg2, _nb0
print("ok props NBA de Unibet")

# props NBA: liquidación de Unibet y modelo vs casa (nba_props.settle / report)
assert nba_props.stat_of("Points scored by the player - Including Overtime") == "puntos" and nba_props.stat_of("3-point field goals made") == "triples"
assert nba_props.stat_of("Points + Rebounds by the player") == "puntos+rebotes" and nba_props.norm("James, LeBron") == "lebron james"
_tr, _pa, _lg = nba_props.T.rows, nba_props.params, nba_props.LOG
_rr = pd.DataFrame([{"date": pd.Timestamp(f"2026-10-{i:02d}T00:30:00Z"), "game": str(i), "pid": "3", "name": "LeBron James", "team": "LAL",
                     "min": 35.0, "puntos": 25.0 + i, "rebotes": 8.0, "asistencias": 8.0, "triples": 2.0} for i in range(1, 15)])
nba_props.T.rows = lambda: _rr
nba_props.params = lambda: {"a_min": 0.2, "a_rate": 0.05, "k": 50, "stats": {s: {"r": 30.0} for s in ("puntos", "rebotes", "asistencias", "triples")}}
nba_props.LOG = Path(tempfile.mkdtemp()) / "pl.csv"
_kf = A_.DATA / "nba" / "kambi_props.csv"; _kb = _kf.read_bytes() if _kf.exists() else None
pd.DataFrame([{"mercado": "Points scored by the player", "jugador": "LeBron James", "tipo": t, "linea": 35.5, "cuota": c, "seen": "2026-10-14T10:00:00+00:00",
               "start": "2026-10-14T00:30:00+00:00", "home": "Lakers", "away": "Celtics"} for t, c in (("OT_OVER", 1.9), ("OT_UNDER", 1.9))]).to_csv(_kf, index=False)
try:
    _sn = nba_props.settle()
    assert len(_sn) == 1 and _sn.mas.iloc[0] == 1 and _sn.p_casa_mas.iloc[0] == 0.5 and _sn.esperado.iloc[0] > 25  # 39 puntos > 35.5
    assert nba_props.report()[0].startswith("puntos (1)")
finally:
    if _kb is None: _kf.unlink()
    else: _kf.write_bytes(_kb)
    nba_props.T.rows, nba_props.params, nba_props.LOG = _tr, _pa, _lg
print("ok liquidación props NBA")

# esports: Unibet como 2.ª casa (kambi.esports_events / find_esport)
_kg3, _es0 = kambi.get, kambi._esp
kambi._esp = None
kambi.get = lambda path: {"events": [
    {"event": {"homeName": "Natus Vincere", "awayName": "FaZe Clan", "start": "2026-11-05T15:00:00Z", "path": [{"englishName": "Esports"}, {"englishName": "Counter-Strike 2"}]},
     "betOffers": [{"outcomes": [{"type": "OT_ONE", "odds": 1750}, {"type": "OT_TWO", "odds": 2050}]}]},
    {"event": {"homeName": "T1", "awayName": "Gen.G", "start": "2026-11-05T10:00:00Z", "path": [{"englishName": "League of Legends"}]},
     "betOffers": [{"outcomes": [{"type": "OT_ONE", "odds": 1600}, {"type": "OT_TWO", "odds": 2300}]}]}]}
assert len(kambi.esports_events()) == 2 and kambi.esports_events()[1]["disc"] == "lol"
assert kambi.find_esport("cs2", "FaZe Clan", "Natus Vincere", "2026-11-05T16:00:00Z") == (2.05, 1.75)  # orden de bo3.gg
assert kambi.find_esport("cs2", "FaZe Clan", "Natus Vincere", "2026-11-06T16:00:00Z") is None
kambi.get, kambi._esp = _kg3, _es0
print("ok esports con Unibet")

# NFL: si nflverse dice que la casa gana, peso 0 aunque el juez de 263 partidos diga otra cosa
_nf0, _mc1 = _MW._NFLV, _MW._MC
_MW._NFLV = {"ll_mezcla": 0.6114, "ll_mercado": 0.6098}
_MW._MC = {"nfl": {"ganador": {"w_opt": 0.4, "ll_2a_mitad_w_opt": 0.64, "ll_2a_mitad_mercado": 0.65}}}
assert _MW.weight("nfl", "ganador") == 0.0
_MW._NFLV, _MW._MC = _nf0, _mc1
print("ok NFL nflverse")

# liquidar con ESPN cuando football-data aún no publica (espn_soccer.rows_for_keys)
_el3 = espn_soccer.load
espn_soccer.load = lambda code: [{"date": "2026-11-03T19:45Z", "home": "Blackburn Rovers", "away": "Stoke City", "hs": 1, "as": 2,
    "team_stats": {"h": {"wonCorners": 6, "yellowCards": 2, "foulsCommitted": 11, "shotsOnTarget": 3},
                   "a": {"wonCorners": 4, "yellowCards": 1, "redCards": 1, "foulsCommitted": 9, "shotsOnTarget": 5}}}] if code == "E1" else []
_rk = espn_soccer.rows_for_keys(["E1|2026-11-03|Blackburn|Stoke", "E1|2026-11-03|Leeds|Hull"])
espn_soccer.load = _el3
assert list(_rk) == ["E1|2026-11-03|Blackburn|Stoke"] and _rk["E1|2026-11-03|Blackburn|Stoke"]["HC"] == 6 and _rk["E1|2026-11-03|Blackburn|Stoke"]["AY"] == 2
assert football_stats.result("corners_t_o", 9.5, _rk["E1|2026-11-03|Blackburn|Stoke"]) is True
print("ok liquidar con ESPN")

# props NBA de Unibet en todos los partidos próximos, analizados o no (multisport.nba_props_pass)
_np3 = kambi.nba_props
kambi.nba_props = lambda h, a, s: [{"mercado": "Points", "jugador": "X", "tipo": "OT_OVER", "linea": 20.5, "cuota": 1.9}]
_fp = A_.DATA / "nba" / "kambi_props.csv"; _fpb = _fp.read_bytes() if _fp.exists() else None
_pst = {"nba_props_ok": []}
_g1 = {"date": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=5)).isoformat(), "home": "A", "away": "B"}
try:
    _MS.nba_props_pass(_pst, [_g1]); _MS.nba_props_pass(_pst, [_g1])
    assert len(_pst["nba_props_ok"]) == 1 and len(pd.read_csv(_fp)) >= 1
finally:
    if _fpb is None: _fp.unlink(missing_ok=True)
    else: _fp.write_bytes(_fpb)
    kambi.nba_props = _np3
print("ok pasada de props NBA")
assert nba_props.stat_of("Points & Rebounds by the player - Including Overtime") == "puntos+rebotes" and nba_props.stat_of("6+ Rebounds By The Player") is None
assert nba_props.stat_of("Points, rebounds & assists by the player - Including Overtime") == "puntos+rebotes+asistencias"
assert nba_props.stat_of("Point Spread - Including Overtime") is None and nba_props.stat_of("Rebounds by the player - Including Overtime") == "rebotes"
print("ok mercados NBA reales")
# FC 27: cara a cara entre jugadores (train_fc_calib.py → esoccer.h2h_adjust)
_hdf = pd.DataFrame({"p1": ["a", "b", "a"], "p2": ["b", "a", "c"], "s1": [3, 0, 1], "s2": [1, 2, 1]})
_H = esoccer.h2h_table(_hdf)
assert abs(esoccer.h2h_value(_H, "a", "b") - 2 / 7) < 1e-9 and abs(esoccer.h2h_value(_H, "b", "a") + 2 / 7) < 1e-9 and esoccer.h2h_value(_H, "a", "z") == 0
_ph = {"1": 0.4, "X": 0.2, "2": 0.4}; _qh = esoccer.h2h_adjust(_ph, 0.3, 0.15)
assert _qh["1"] > _qh["2"] and abs(sum(_qh.values()) - 1) < 1e-9 and esoccer.h2h_adjust(_ph, 0.3, 0.0) == _ph
print("ok cara a cara FC 27")
# chat de Telegram sin IA (chat.py): lectura de la frase, búsqueda de nombres y opinión
import chat as _CH
_CH.LAST_FILE = A_.DATA / "_test_ultima.json"; _CH.LAST_FILE.unlink(missing_ok=True)  # los tests no tocan la del bot
assert _CH.parse_leg("Mbappé más de 1,5 tiros a puerta") == {"stat": "tiros_puerta", "line": 1.5, "over": True}
assert _CH.parse_leg("Curry 4+ triples") == {"stat": "triples", "line": 3.5, "over": True}
assert _CH.parse_leg("menos de 9.5 córners")["over"] is False and _CH.parse_leg("Yamal marca")["line"] == 0.5
assert _CH.odds_in("Jokic más de 11.5 rebotes cuota 1,80") == [1.8] and _CH.odds_in("x @3.1") == [3.1] and _CH.odds_in("más de 25.5 puntos") == []
_g = lambda v, a=None: [("2026-10-%02d" % (i + 1), {**v, "_titular": True}, a or {}) for i in range(12)]
_D = {"futbol|p|1": {"name": "Kylian Mbappé", "sport": "futbol", "kind": "jugador", "team": "Real Madrid", "games": _g({"tiros_puerta": 3.0})},
      "futbol|p|2": {"name": "Gustavo Puerta", "sport": "futbol", "kind": "jugador", "team": "X", "games": _g({"tiros_puerta": 0.0})},
      "futbol|t|RM": {"name": "Real Madrid", "sport": "futbol", "kind": "equipo", "team": None, "games": _g({"corners": 6.0}, {"corners": 4.0})},
      "futbol|t|AM": {"name": "Atlético Madrid", "sport": "futbol", "kind": "equipo", "team": None, "games": _g({"corners": 5.0}, {"corners": 5.0})},
      "futbol|t|B": {"name": "Barcelona", "sport": "futbol", "kind": "equipo", "team": None, "games": _g({"corners": 7.0}, {"corners": 3.0})}}
for _e in _D.values(): _e["norm"] = _CH.norm(_e["name"])
assert _CH.find("Mbappe más de 1.5 tiros a puerta", _D) == ["futbol|p|1"]  # «puerta» no es Gustavo Puerta
assert _CH.find("Madrid Barcelona más de 9.5 córners", _D, kind="equipo") == ["futbol|t|RM", "futbol|t|B"]
assert _CH.find("Real Madrid más de 5.5 córners", _D, kind="equipo") == ["futbol|t|RM"]
_CH._DB.update(db=_D, t=1e18)
_r = _CH.answer("Mbappé más de 1.5 tiros a puerta a 1.9"); assert "🟢" in _r and "Mbappé" in _r, _r
assert "Combinada" in _CH.answer("Mbappé más de 1.5 tiros a puerta y Madrid Barcelona más de 9.5 córners @2.0")
assert "Real Madrid" in _CH.answer("estadísticas del Real Madrid") and "No sé si te he entendido" in _CH.answer("qwerty zzz")
for _e in _D.values():
    _e["leagues"] = {"SP1"}
    if _e["kind"] == "equipo":
        for _x in _e["games"]: _x[1].setdefault("goles", 1.5); _x[2].setdefault("goles", 1.0)
_D["futbol|t|RM"]["games"][-1][1].update(_rival="Barcelona", _casa=True, goles=2.0); _D["futbol|t|RM"]["games"][-1][2]["goles"] = 1.0
_sb = _CH.scoreboard
_CH.scoreboard = lambda sp, lg, days=21, team=None: [(pd.Timestamp("2026-11-08 19:00", tz="UTC"), "Barcelona", "Real Madrid", "LaLiga")]
try:
    _r = _CH.answer("¿Cuál es el próximo rival del Real Madrid?"); assert "Barcelona – Real Madrid" in _r and "20:00" in _r, _r
    _r = _CH.answer("próximo partido de Mbappé"); assert "Barcelona – Real Madrid" in _r, _r  # su equipo
    _r = _CH.answer("último resultado del Real Madrid"); assert "ganó 2-1 contra Barcelona en casa" in _r, _r
    assert _CH.plays("Almería", (None, "Cádiz", "UD Almería")) and not _CH.plays("Almería", (None, "Cádiz", "")) and not _CH.plays("Real Madrid", (None, "Real Betis", "Madrid CFF"))
    _r = _CH.answer("¿Cuál ganaría?"); assert "Barcelona – Real Madrid" in _r and "Favorito" in _r or "igualado" in _r, _r  # recuerda el partido
    _r = _CH.answer("gana Real Madrid a 9.0"); assert "🟢" in _r, _r
finally: _CH.scoreboard = _sb; _CH._CTX.clear()
for _x in _D["futbol|t|RM"]["games"]: _x[1]["_liga"] = "SP1"
_CH.LEAGUE["SP1"] = [({"corners": 5.0}, {"corners": 4.0})] * 40
_mm = _CH.match_model("futbol|t|RM", "futbol|t|B", _D, "corners")
assert _mm and _mm[1] > 4.0 and 8.5 < _mm[0] + _mm[1] < 10.5, _mm  # el Barça (7 a favor) saca más que la media de visitante (4)
_r = _CH.answer("Madrid Barcelona más de 9.5 córners"); assert "esperado" in _r, _r
assert "moneda" in _CH.coin(0.5) and _CH.coin(0.7) == ""
assert "seguro" in "\n".join(_CH.preview("Real Madrid", "Barcelona", {**_D, "futbol|t|Real Madrid": _D["futbol|t|RM"], "futbol|t|Barcelona": _D["futbol|t|B"]}, "futbol"))
assert "curarte en salud" in _CH.answer("Madrid Barcelona más de 9.5 córners")
assert _CH.safe_line(lambda l: 1.0 if l < 4 else 0.0) == 3.5 and _CH.safer(lambda l: 0.9 if l < 6 else 0.5, 9.5, "corners").count("más de 5.5") == 1
_CH._CTX["partido"] = ("futbol|t|RM", "futbol|t|B")
_r = _CH.answer("combinada segura"); assert "Combinada segura" in _r and "cuota justa" in _r, _r
_sb2, _up2 = _CH.scoreboard, _CH.upcoming
_CH.scoreboard = lambda sp, lg, days=21, team=None: [(pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=5), "Real Madrid", "Barcelona", "")] if lg == "SP1" else []
_CH.upcoming = lambda codes, hours=48: {"SP1": _CH.scoreboard("futbol", "SP1")}
_ol2 = _CH.other_legs; _CH.other_legs = lambda hours, sports: []
try:
    _r = _CH.answer("Añádeles más para subir la cuota, lo máximo que pueda pero de forma segura"); assert "del día" in _r and "cuota justa" in _r, _r
    _lg, _p = _CH.pick_legs([("a", "x", 0.9), ("a", "y", 0.9), ("a", "z", 0.9), ("b", "w", 0.9), ("b", "v", 0.9)], per_match=2)
    assert len(_lg) == 4 and _p >= _CH.COMBO_FLOOR and sum(m == "a" for m, _, _ in _lg) == 2
finally: _CH.scoreboard, _CH.upcoming, _CH.other_legs = _sb2, _up2, _ol2
_CH.LEAGUE.clear(); _CH._DB.clear(); _CH._DB["t"] = 0; _CH._BASE.clear(); _CH._CTX.clear()
print("ok chat de Telegram")
# ESPN: partidos sin estadísticas (todo a 0) no sirven para liquidar
import espn_soccer as _ESZ
assert not _ESZ.has_stats({"wonCorners": 0.0, "totalShots": 0.0, "foulsCommitted": 0.0}) and _ESZ.has_stats({"wonCorners": 0.0, "totalShots": 7.0})
_ld = _ESZ.load
_z = {k: 0.0 for k in _ESZ.TSTATS}
_ESZ.load = lambda code: [{"id": "1", "date": "2026-11-01T15:00Z", "home": "Arsenal", "away": "Chelsea", "hs": 1, "as": 0, "team_stats": {"h": _z, "a": _z}}] if code == "E0" else []
try: assert _ESZ.rows_for_keys(["E0|2026-11-01|Arsenal|Chelsea"]) == {}
finally: _ESZ.load = _ld
print("ok ESPN sin estadísticas no liquida")
# notas de voz: el audio se transcribe y el texto va al chat (Whisper simulado)
import voz as _VZ
_dl, _tr = _VZ.download, _VZ.transcribe
_VZ.download = lambda fid: (open(_p := "/tmp/_voz_test.oga", "wb").close(), _p)[1]
_VZ.transcribe = lambda p: "estadísticas de Mbappé"
try:
    assert _VZ.from_message({"voice": {"file_id": "x"}}) == "estadísticas de Mbappé" and _VZ.from_message({"text": "hola"}) is None
finally: _VZ.download, _VZ.transcribe = _dl, _tr
print("ok notas de voz")
# conversación: «sí» responde a la pregunta del bot, «no» la descarta; sin entender, propone algo (intérprete simulado)
import interpreta as _IP
_CH._DB.update(db=_D, t=1e18); _CH._CTX.clear()
_sb3 = _CH.scoreboard
_CH.scoreboard = lambda sp, lg, days=21, team=None: [(pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=5), "Real Madrid", "Barcelona", "")] if lg == "SP1" else []
_rt = _IP.route
try:
    _r = _CH.answer("¿quién gana Real Madrid Barcelona?"); assert "❓" in _r and _CH._CTX.get("oferta"), _r
    _r = _CH.answer("sí"); assert "Real Madrid – Barcelona" in _r and "piernas" in _r, _r  # ejecuta lo ofrecido (la combinada)
    _CH.answer("quién gana Real Madrid Barcelona"); assert "Vale" in _CH.answer("no") and not _CH._CTX.get("oferta")
    _IP.route = lambda text, hist=(): {"orden": "estadísticas de Real Madrid", "respuesta": ""}
    _r = _CH.answer("oye y qué tal le va últimamente al equipo blanco de la capital"); assert "Real Madrid" in _r and "Últimos" in _r, _r
    _IP.route = lambda text, hist=(): None
    _r = _CH.answer("zzz qqq"); assert "No te he entendido" not in _r and "❓" in _r, _r
    assert len(_CH._CTX["hist"]) <= 12
finally: _CH.scoreboard = _sb3; _IP.route = _rt; _CH._DB.clear(); _CH._DB["t"] = 0; _CH._CTX.clear(); _CH._BASE.clear()
print("ok conversación del chat")
# intérprete con GitHub Models (respuesta simulada; sin token no se llama)
import interpreta as _IP2, requests as _rq
assert _IP2.route_github("hola") is None or __import__("os").getenv("GH_MODELS_TOKEN")
_pst2, _env = _rq.post, __import__("os").environ.get("GH_MODELS_TOKEN")
class _R:
    status_code, ok = 200, True
    def json(self): return {"choices": [{"message": {"content": '{"orden": "combinada Girona Almería", "respuesta": ""}'}}]}
_rq.post = lambda *a, **k: _R(); __import__("os").environ["GH_MODELS_TOKEN"] = "x"; _gk = __import__("os").environ.pop("GEMINI_API_KEY", None)
try: assert _IP2.route("sí", [("bot", "¿Quieres la combinada de Girona – Almería?")]) == {"orden": "combinada Girona Almería", "respuesta": ""}
finally:
    _rq.post = _pst2
    if _gk: __import__("os").environ["GEMINI_API_KEY"] = _gk
    if _env is None: __import__("os").environ.pop("GH_MODELS_TOKEN")
    else: __import__("os").environ["GH_MODELS_TOKEN"] = _env
assert _IP2._parse('```json\n{"orden": "ayuda", "respuesta": ""}\n```')["orden"] == "ayuda"
print("ok intérprete con GitHub Models")
# intérprete con Gemini (cliente simulado)
import types as _ty, google.genai as _gg, os as _os
_cl = _gg.Client
class _GC:
    def __init__(self, **k): self.models = self
    def generate_content(self, **k): return _ty.SimpleNamespace(text='{"orden": "combinada del día", "respuesta": ""}')
_gg.Client = _GC; _os.environ["GEMINI_API_KEY"] = "x"
try: assert _IP2.route("súbele la cuota") == {"orden": "combinada del día", "respuesta": ""}
finally: _gg.Client = _cl; _os.environ.pop("GEMINI_API_KEY")
assert _IP2.GEMINI_MODELS[0] == "gemini-flash-lite-latest" and isinstance(_IP2.GEMINI_MODELS, list)
print("ok intérprete con Gemini")
# redactor: la IA escribe con los datos del bot; si mete un número que el bot no dio, se descarta
_cg = _IP2._complete_gemini; _os.environ["GEMINI_API_KEY"] = "x"
try:
    _IP2._complete_gemini = lambda s, p: "Te diría que sí: más de 4.5 córners tiene un 91% 👍"
    assert _IP2.redactar("¿y los córners?", "Más de 4.5 córners (91%) · cuota justa 1.10") == "Te diría que sí: más de 4.5 córners tiene un 91% 👍"
    _IP2._complete_gemini = lambda s, p: "Yo creo que hay un 95% de que salgan más de 4.5 córners"
    assert _IP2.redactar("¿y los córners?", "Más de 4.5 córners (91%)") is None  # 95 no lo dio el bot
    _IP2._complete_gemini = lambda s, p: "Te cuento <b>todo</b> & más"
    assert _CH.natural("hola", "<b>informe</b> sin números") == "Te cuento &lt;b&gt;todo&lt;/b&gt; &amp; más"  # texto escapado para Telegram
    assert _CH.natural("ayuda", _CH.HELP) == _CH.HELP
finally: _IP2._complete_gemini = _cg; _os.environ.pop("GEMINI_API_KEY")
print("ok redactor con datos del bot")
# combinada del día con cuota objetivo («popurrí a cuota 2.5»): una pierna segura por partido hasta llegar
_lg2, _p2 = _CH.pick_legs([(f"m{i}", "x", 0.9) for i in range(12)], per_match=1, target=2.5)
assert 1 / _p2 >= 2.5 and len(_lg2) == 9 and len({m for m, _, _ in _lg2}) == 9
_up = _CH.upcoming
_CH.upcoming = lambda codes, hours=48: {}
_ol = _CH.other_legs; _CH.other_legs = lambda hours, sports: []
try: assert "ESPN no me devuelve partidos" in _CH.day_combo({}, target=2.5)
finally: _CH.upcoming, _CH.other_legs = _up, _ol
_lg3, _p3 = _CH.pick_legs([("a", "x", 0.97), ("a", "y", 0.86), ("b", "z", 0.96), ("b", "w", 0.87)], per_match=1, target=1.3)
assert [x[1] for x in _lg3] == ["y", "w"]  # la segura que más paga de cada partido
print("ok combinada a cuota objetivo")
# combinada de cuota alta con cuotas de Unibet (simuladas): piernas con ventaja tras mezclar modelo y casa, hasta la cuota
import kambi as _KB
_off, _up4 = _KB.offers, _CH.upcoming
_now = pd.Timestamp.now(tz="UTC")
_CH._DB.update(db=_D, t=1e18); _CH.LEAGUE["SP1"] = [({"corners": 5.0}, {"corners": 4.0})] * 40
for _e in _D.values():
    if _e["kind"] == "equipo":
        for _x in _e["games"]: _x[1]["_liga"] = "SP1"
_CH.upcoming = lambda codes, hours=48: {"SP1": [(_now + pd.Timedelta(hours=5 + i), h, a, "") for i, (h, a) in
                                                 enumerate([("Real Madrid", "Barcelona"), ("Barcelona", "Atlético Madrid"), ("Atlético Madrid", "Real Madrid")])]}
_KB.offers = lambda h, a, t, stats=(): {("corners", "t"): {6.5: {"o": 1.30, "u": 3.40}, 12.5: {"o": 4.5, "u": 1.18}}}
_ol3 = _CH.other_legs; _CH.other_legs = lambda hours, sports: []
try:
    _r = _CH.answer("combinada del día a cuota 80"); assert "objetivo 80" in _r or "no tengo cuotas" in _r.lower(), _r
    _KB.offers = lambda h, a, t, stats=(): {}
    assert "Hoy no tengo cuotas de Unibet" in _CH.long_combo(_D, 80)
finally: _KB.offers, _CH.upcoming, _CH.other_legs = _off, _up4, _ol3; _CH.LEAGUE.clear(); _CH._DB.clear(); _CH._DB["t"] = 0; _CH._BASE.clear()
print("ok combinada de cuota alta")
# Kambi: no confundir el partido con el de juveniles (Youth League el mismo día) y alias de nombres
_kev = _KB._events
_KB._events = [{"id": 1, "home": "AEK Athens FC U19", "away": "Real Madrid U19", "start": pd.Timestamp("2026-11-04 10:00", tz="UTC")},
               {"id": 2, "home": "AEK Athens", "away": "Real Madrid", "start": pd.Timestamp("2026-11-04 17:45", tz="UTC")},
               {"id": 3, "home": "Bristol City", "away": "Wolves", "start": pd.Timestamp("2026-11-04 19:45", tz="UTC")}]
try:
    assert _KB.find("AEK Athens", "Real Madrid", pd.Timestamp("2026-11-04 17:45", tz="UTC"))["id"] == 2
    assert _KB.find("Bristol City", "Wolverhampton Wanderers", pd.Timestamp("2026-11-04 19:45", tz="UTC"))["id"] == 3
finally: _KB._events = _kev
print("ok Kambi sin juveniles")
# apuestas simuladas desde el chat: «simúlala» la apunta; al llegar las estadísticas de ESPN se liquida con el porqué
import simulaciones as _SM
_smf, _sml, _smp = _SM.FILE, _SM.games, _SM.push
_SM.FILE = A_.DATA / "_test_simuladas.json"; _SM.FILE.unlink(missing_ok=True); _SM._MEM.clear()
_SM.push = lambda msg: None
_CH._CTX.clear()
try:
    _CH.LAST_FILE.unlink(missing_ok=True); assert "Primero pídeme" in _CH.answer("simula la apuesta")
    _CH.remember("Combinada cuota 9.0", [dict(code="ucl", start="2026-11-04T20:00:00+00:00", home="Slavia Prague", away="Arsenal",
                                               txt="Más de 4.5 tarjetas", odds=3.0, p=0.4),
                                          dict(code="ucl", start="2026-11-04T20:00:00+00:00", home="FC Porto", away="Napoli",
                                               txt="Napoli o empate (X2)", odds=3.0, p=0.5)], 9.0, 0.2)
    _r = _CH.answer("Simula la apuesta y cuando se terminen los partidos me dices si hemos ganado"); assert "Apuntada" in _r, _r
    assert _SM.settle() == []  # aún sin partidos
    _z = {"wonCorners": 5.0, "totalShots": 10.0, "foulsCommitted": 10.0, "yellowCards": 2.0, "redCards": 0.0}
    _SM.games = lambda code: [{"home": "Slavia Prague", "away": "Arsenal", "date": "2026-11-04T20:00Z", "hs": 0, "as": 2, "team_stats": {"h": _z, "a": _z}},
                              {"home": "FC Porto", "away": "Napoli", "date": "2026-11-04T20:00Z", "hs": 1, "as": 1, "team_stats": {"h": _z, "a": _z}}]
    _msgs = _SM.settle(); assert len(_msgs) == 1 and "perdida" in _msgs[0] and "hubo 4 tarjetas" in _msgs[0] and "✅" in _msgs[0], _msgs
    assert _SM.settle() == [] and "1 terminadas" in _CH.answer("¿cómo van mis simuladas?")
finally: _SM.FILE.unlink(missing_ok=True); _SM.FILE, _SM.games, _SM.push = _smf, _sml, _smp; _SM._MEM.clear(); _CH._CTX.clear()
print("ok apuestas simuladas del chat")
# todos los deportes en el chat: piernas «gana X» de NBA/NHL/esports/FC 27 (multideporte) y su liquidación en simuladas
import multideporte as _MD
_L = _MD.legs(24 * 365)
assert all({"src", "home", "away", "start", "side", "txt", "p"} <= set(l) for l in _L)
assert _CH.sports_in("combinada de esports y nba") >= {"nba", "cs2", "fc"} and _CH.sports_in("combinada del día") is None
assert _CH._leg({"src": "esports:cs2", "side": "H", "start": "2026-11-04T20:00Z", "home": "NAVI", "away": "G2", "odds": 1.5}, "Gana NAVI", 0.9)["src"] == "esports:cs2"
_jd = _MD._result
_MD._result = lambda leg: (2.0, 1.0)
try: assert _MD.judge({"src": "esports:cs2", "side": "H"}) == (True, "acabó 2-1") and _MD.judge({"src": "fc", "side": "X"})[0] is False
finally: _MD._result = _jd
_smf2, _smp2 = _SM.FILE, _SM.push
_SM.FILE = A_.DATA / "_test_simuladas2.json"; _SM.FILE.unlink(missing_ok=True); _SM._MEM.clear(); _SM.push = lambda m: None
_mj = _MD.judge
try:
    _SM.record({"titulo": "Combinada del día", "legs": [dict(src="espn:nba", side="H", start="2026-11-04T23:00Z", home="Boston Celtics",
                                                             away="Milwaukee Bucks", txt="Gana Boston Celtics", odds=None, p=0.85)], "odds": 1.18})
    _MD.judge = lambda leg: (None, "sin resultado todavía"); assert _SM.settle() == []
    _MD.judge = lambda leg: (True, "acabó 112-104"); _m = _SM.settle(); assert len(_m) == 1 and "GANADA" in _m[0] and "112-104" in _m[0], _m
finally: _MD.judge = _mj; _SM.FILE.unlink(missing_ok=True); _SM.FILE, _SM.push = _smf2, _smp2; _SM._MEM.clear()
print("ok todos los deportes en el chat")
_mdl = _MD.legs
_MD.legs = lambda hours=48, sports=None: [dict(src="esports:lol", sport="🧙 LoL", home="Gen.G Esports", away="LYON", start="2026-11-04T22:00Z",
                                               side="H", txt="Gana Gen.G Esports", p=0.88, odds=1.20)]
try: _r = _CH.answer("¿quién gana Gen.G Esports contra LYON?"); assert "Favorito" in _r and "Gen.G" in _r and "🟢" in _r, _r
finally: _MD.legs = _mdl
print("ok quién gana en esports")
# cuota alta de un solo partido («crear apuesta») y liquidación de piernas de un equipo
_up5, _off5 = _CH.upcoming, _KB.offers
_CH._DB.update(db=_D, t=1e18); _CH.LEAGUE["SP1"] = [({"corners": 5.0, "tarjetas": 2.0}, {"corners": 4.0, "tarjetas": 2.0})] * 40
for _e in _D.values():
    if _e["kind"] == "equipo":
        for _x in _e["games"]: _x[1]["_liga"] = "SP1"; _x[1].setdefault("tarjetas", 2.0); _x[2].setdefault("tarjetas", 2.0)
_CH.upcoming = lambda codes, hours=48: {"SP1": [(pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=6), "Real Madrid", "Barcelona", "")]}
_KB.offers = lambda h, a, t, stats=(): {("corners", "h"): {5.5: {"o": 1.9, "u": 1.9}}, ("corners", "t"): {9.5: {"o": 1.9, "u": 1.9}},
                                        ("cards", "t"): {3.5: {"o": 1.9, "u": 1.9}}}
try:
    _r = _CH.answer("combinada solo del Real Madrid contra el Barcelona a cuota 80"); assert "Crear apuesta" in _r, _r
    assert not ("córners de Real Madrid" in _r and "Más de 9.5 córners @" in _r)  # total o de equipo, no ambos
finally: _CH.upcoming, _KB.offers = _up5, _off5; _CH.LEAGUE.clear(); _CH._DB.clear(); _CH._DB["t"] = 0; _CH._BASE.clear(); _CH._CTX.clear()
_zz = {"wonCorners": 7.0, "totalShots": 9.0, "foulsCommitted": 9.0, "yellowCards": 1.0, "redCards": 0.0}
_g = {"home": "AEK Athens", "away": "Real Madrid", "hs": 0, "as": 3, "team_stats": {"h": dict(_zz, wonCorners=2.0), "a": _zz}}
assert _SM.judge({"txt": "Menos de 6.5 córners de Real Madrid"}, _g) == (False, "hubo 7 córners de Real Madrid")
assert _SM.judge({"txt": "Más de 2.5 córners de AEK Athens"}, _g)[0] is False and _SM.judge({"txt": "Más de 8.5 córners"}, _g)[0] is True
print("ok crear apuesta de un partido")
# Pinnacle: cuotas americanas → decimales
import pinnacle as _PN
assert _PN.dec(150) == 2.5 and _PN.dec(-200) == 1.5 and _PN.dec(None) is None
print("ok Pinnacle")
# Pinnacle sin margen como referencia del mercado en las piernas «gana X»
_MD._PIN["baloncesto"] = pd.DataFrame([{"start": "2026-11-05T00:30:00Z", "home": "Boston Celtics", "away": "Milwaukee Bucks",
                                        "market": "moneyline", "o_home": 1.25, "o_away": 4.2}])
_pf = _MD.pinnacle_fair({"src": "espn:nba", "side": "A", "start": "2026-11-05T00:30:00Z", "home": "Boston Celtics", "away": "Milwaukee Bucks"})
assert abs(_pf - (1 / 4.2) / (1 / 1.25 + 1 / 4.2)) < 1e-3
assert _MD.pinnacle_fair({"src": "espn:nba", "side": "H", "start": "2026-11-09T00:30:00Z", "home": "Boston Celtics", "away": "Milwaukee Bucks"}) is None
_MD._PIN.clear()
print("ok Pinnacle como referencia")
# boleto en imagen: las piernas leídas se analizan con los modelos del bot y se pueden simular
_CH._DB.update(db=_D, t=1e18); _CH._CTX.clear()
_slip = {"piernas": [{"partido": "Real Madrid - Barcelona", "tipo": "ganador", "equipo": "Real Madrid", "cuota": 2.1, "texto": "Real Madrid"},
                     {"partido": "Real Madrid - Barcelona", "tipo": "otro", "texto": "Mbappé marca primero", "cuota": 5.0}], "cuota_total": 10.5}
_mdl2 = _MD.legs; _MD.legs = lambda hours=48, sports=None: []
try:
    _r = _CH.analyze_slip(_slip); assert "Tu boleto" in _r and "Real Madrid" in _r and "no la puedo evaluar" in _r and "1 de 2" in _r, _r
    _slip["piernas"] = _slip["piernas"][:1]; _r = _CH.analyze_slip(_slip); assert "Todas a la vez" in _r and "❓" in _r, _r
    assert _CH._CTX["ultima"]["legs"][0]["txt"] == "Gana Real Madrid"
finally: _MD.legs = _mdl2; _CH._DB.clear(); _CH._DB["t"] = 0; _CH._CTX.clear()
assert _SM.judge({"txt": "Gana Real Madrid"}, {"home": "Real Madrid", "away": "Barcelona", "hs": 2, "as": 1})[0] is True
print("ok boleto en imagen")
# córners de un equipo contra un rival concreto, con la supremacía de las cuotas (el débil saca menos)
_CH._DB.update(db=_D, t=1e18); _CH.LEAGUE["SP1"] = [({"corners": 5.0}, {"corners": 4.0})] * 40
for _e in _D.values():
    if _e["kind"] == "equipo":
        for _x in _e["games"]: _x[1]["_liga"] = "SP1"
_mm0 = _CH.match_market
try:
    _CH.match_market = lambda h, a: None; _base = _CH.match_model("futbol|t|RM", "futbol|t|B", _D, "corners")
    _CH.match_market = lambda h, a: 0.6; _fav = _CH.match_model("futbol|t|RM", "futbol|t|B", _D, "corners")
    assert _fav[0] > _base[0] and _fav[1] < _base[1]  # el favorito local saca más córners, el visitante menos
    _r = _CH.answer("¿hará el Barcelona tantos córners contra el Real Madrid?"); assert "Barcelona</b> contra <b>Real Madrid" in _r and "esperados" in _r, _r
finally: _CH.match_market = _mm0; _CH.LEAGUE.clear(); _CH._DB.clear(); _CH._DB["t"] = 0; _CH._BASE.clear(); _CH._CTX.clear()
print("ok córners contra un rival")
# boletos: respaldo de clasificación («anotará», «3+ remates») y nombres de ciudades en español
assert _CH.find("AEK Atenas Real Madrid", {"x|t|1": {"name": "AEK Athens", "norm": "aek athens", "kind": "equipo", "sport": "futbol", "games": []}}, n=1) == ["x|t|1"]
print("ok boletos: ciudades y piernas de jugador")
# faltas recibidas por jugador (ESPN foulsSuffered)
assert _CH.parse_leg("Bellingham 2+ faltas recibidas") == {"stat": "faltas_recibidas", "line": 1.5, "over": True}
assert _CH.parse_leg("Bellingham más de 1.5 faltas")["stat"] == "faltas" and "foulsSuffered" in _ESZ.PSTATS
print("ok faltas recibidas")
# simuladas: piernas de jugador (goles, tiros a puerta, faltas recibidas) con las estadísticas de ESPN del partido
_gp = {"home": "AEK Athens", "away": "Real Madrid", "hs": 0, "as": 2,
       "players": [["1", "Kylian Mbappé", "a", True, "F", 1.0, 0.0, 5.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0]]}
assert _SM.judge({"jugador": "Kylian Mbappé", "txt": "Más de 0.5 goles"}, _gp) == (True, "Kylian Mbappé: 1 goles")
assert _SM.judge({"jugador": "Kylian Mbappe", "txt": "Más de 2.5 tiros a puerta"}, _gp)[0] is True
assert _SM.judge({"jugador": "Kylian Mbappé", "txt": "Más de 1.5 faltas recibidas"}, _gp)[0] is True
assert _SM.judge({"jugador": "Jude Bellingham", "txt": "Más de 0.5 goles"}, _gp)[0] == "nula"
print("ok simuladas de jugador")
_CH.LAST_FILE.unlink(missing_ok=True)
# boleto con una pierna sin datos: se guarda igual para simular (sin la que no se puede liquidar) y sobrevive a un reinicio
_CH._DB.update(db=_D, t=1e18); _CH._CTX.clear(); _MDl3 = _MD.legs; _MD.legs = lambda hours=48, sports=None: []
try:
    _r = _CH.analyze_slip({"piernas": [{"partido": "Real Madrid - Barcelona", "tipo": "ganador", "equipo": "Real Madrid", "cuota": 2.1, "texto": "Real Madrid"},
                                       {"partido": "Real Madrid - Barcelona", "tipo": "otro", "texto": "Bellingham marca de chilena", "cuota": 9.0}]})
    assert "❓" in _r and "sin la pierna" in _r, _r
    _CH._CTX.clear(); assert _CH.last_combo()["titulo"] == "Tu boleto"  # leída de disco tras «reinicio»
finally: _MD.legs = _MDl3; _CH._DB.clear(); _CH._DB["t"] = 0; _CH._CTX.clear(); _CH.LAST_FILE.unlink(missing_ok=True)
print("ok boleto simulable con piernas sin datos")
# respuestas con listas (boletos, combinadas) no pasan por la IA; la confirmación de simulada lleva las piernas
_cg2 = _IP2._complete_gemini; _os.environ["GEMINI_API_KEY"] = "x"; _IP2._complete_gemini = lambda s, p: "texto de la IA"
try: assert _CH.natural("hola", "🧩 Combinada\n• A (90%)\n• B (85%)") == "🧩 Combinada\n• A (90%)\n• B (85%)"
finally: _IP2._complete_gemini = _cg2; _os.environ.pop("GEMINI_API_KEY")
print("ok listas sin IA")
# boleto «crear apuesta» de un partido: no multiplica piernas ligadas
_CH._DB.update(db=_D, t=1e18); _MDl4 = _MD.legs; _MD.legs = lambda hours=48, sports=None: []
try:
    _lg = {"partido": "Real Madrid - Barcelona", "tipo": "ganador", "equipo": "Real Madrid", "cuota": 2.1, "texto": "Real Madrid"}
    _r = _CH.analyze_slip({"piernas": [_lg, _lg, _lg], "cuota_total": 14.0}); assert "muy ligadas" in _r and "Todas a la vez" not in _r, _r
finally: _MD.legs = _MDl4; _CH._DB.clear(); _CH._DB["t"] = 0; _CH._CTX.clear(); _CH.LAST_FILE.unlink(missing_ok=True)
print("ok crear apuesta sin multiplicar")
# simuladas: la misma combinada no se apunta dos veces
_smf3, _smp3 = _SM.FILE, _SM.push
_SM.FILE = A_.DATA / "_test_simuladas3.json"; _SM.FILE.unlink(missing_ok=True); _SM._MEM.clear(); _SM.push = lambda m: None
try:
    _cb = {"titulo": "X", "legs": [dict(home="A", away="B", txt="Más de 4.5 córners", start=None, code=None)], "odds": 1.2}
    assert not _SM.record(_cb).get("duplicada") and _SM.record(_cb).get("duplicada") and len(_SM.load()) == 1
finally: _SM.FILE.unlink(missing_ok=True); _SM.FILE, _SM.push = _smf3, _smp3; _SM._MEM.clear()
print("ok simuladas sin duplicados")
