from analyst import probs, value_bets, won

P = probs(1.5, 1.1)
assert abs(P["H"] + P["D"] + P["A"] - 1) < 1e-3 and abs(P["O25"] + P["U25"] - 1) < 1e-3 and P["H"] > P["A"]
vb = value_bets({"H": 0.6, "D": 0.2, "A": 0.2, "O25": 0.5, "U25": 0.5}, {"AvgH": 2.0, "AvgD": 3.0, "AvgA": 5.0})
assert [v["market"] for v in vb] == ["H"], vb  # A tiene ventaja pero prob < MIN_P
assert won("O25", 2, 1) and won("U25", 1, 1) and not won("H", 1, 1)
print("ok")
