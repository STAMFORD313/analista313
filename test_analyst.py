import analyst
analyst.TG = analyst.CHAT = analyst.GEMINI = None  # los tests nunca envían a Telegram ni llaman a la IA

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
assert "Ganadas: <b>1</b> (33 %)" in txt and "🟥🟩🟥" in txt, txt
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

import radar
s0 = {"key": "k9", "market": "U25", "odds": 2.0, "p": 0.5, "edge": 0.06, "match": "E vs F", "sport": "⚽"}
assert [ok for ok, _ in radar.checklist(s0)] == [False, True, True, False]
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
assert (analyst.HALF_LIFE, analyst.SHRINK, analyst.RHO, analyst.BLEND, analyst.EDGE) == (fp["half_life"], fp["k"], fp["rho"], fp["blend"], fp["edge"])
ep = json.loads((analyst.DATA / "espn_params.json").read_text())
assert all(multisport.SPORTS[l][3:] == (p["hfa"], p["K"]) for l, p in ep.items())
assert set(multisport.TOT) <= {"nba", "mlb", "nhl", "nfl"} and 0 < multisport.totals_model.p_over("nba", 200, 199.5, multisport.TOT) < 1
st2 = {"signals": [{"key": "nba|2026-11-02|A|B", "market": "sph", "line": -2.5}]}
multisport.settle(pd.DataFrame([{"date": "2026-11-02", "home": "A", "away": "B", "hs": 100, "as": 97, "res": 1.0}]), "nba", st2)
assert st2["signals"][0]["result"] is True and multisport.spreads(None) == (None, None, None)
print("ok parámetros entrenados")

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
