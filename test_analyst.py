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

import bankroll
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

import radar
s0 = {"key": "k9", "market": "U25", "odds": 2.0, "p": 0.5, "edge": 0.06, "match": "E vs F", "sport": "⚽"}
assert [ok for ok, _ in radar.checklist(s0)] == [False, True, True, False]
assert len(sum(radar.menu(), [])) == len(radar.SPORTS)
print("ok radar")
