"""Banca virtual: el bot empieza con 100 $ y apuesta (simulado) a las señales de valor con cuotas reales:
la mejor cuota entre casas (Max de football-data.co.uk) en fútbol y DraftKings (ESPN) en deportes USA.
Simples con 1/4 de Kelly + una combinada por ejecución. Se liquida con los resultados reales."""
import math

import pandas as pd

import analyst as A

START = 100.0
KELLY, MAX_STAKE, MAX_EXPOSURE = 0.25, 0.05, 0.30  # 1/4 Kelly, máx. 5 % por apuesta y 30 % de la banca en juego
MAX_SINGLES, COMBO_LEGS, COMBO_MIN_P, COMBO_STAKE = 8, 3, 0.45, 0.02


def pick(s):
    home, away = s["match"].split(" vs ", 1)
    return s.get("pick") or {"H": home, "A": away, "D": "Empate", "O25": "Más de 2.5 goles", "U25": "Menos de 2.5 goles"}[s["market"]]


def label(s): return f"{s['match']} → {pick(s)} @ {s['odds']:.2f}"


def settle(bank, signals):
    by = {(s["key"], s["market"]): s for s in signals}
    lines = []
    for b in bank["bets"]:
        if b["status"] != "abierta": continue
        res = [by[tuple(l)].get("result") for l in b["legs"]]
        if any(r is False for r in res): b["status"], b["return"] = "perdida", 0.0
        elif all(r is True or r == "nula" for r in res):
            odds = math.prod(by[tuple(l)]["odds"] for l, r in zip(b["legs"], res) if r is True)
            b["status"], b["return"] = "ganada", round(b["stake"] * odds, 2)
        else: continue
        bank["balance"] = round(bank["balance"] + b["return"], 2)
        lines.append(f"{'✅' if b['status'] == 'ganada' else '❌'} {b['desc']}: {b['return'] - b['stake']:+.2f} $")
    return lines


def place(bank, signals, today):
    now = pd.Timestamp.now(tz="UTC")
    new = [s for s in signals if not s.get("considered")]
    for s in new: s["considered"] = True
    new = [s for s in new if s.get("start") and pd.Timestamp(s["start"]) > now]  # nunca apostar a algo ya empezado
    open_stake = sum(b["stake"] for b in bank["bets"] if b["status"] == "abierta")
    room = lambda: max(0.0, MAX_EXPOSURE * (bank["balance"] + open_stake) - open_stake)
    lines = []

    def bet(legs, stake, desc, odds):
        nonlocal open_stake
        stake = round(min(stake, room(), bank["balance"]), 2)
        if stake < 1: return
        bank["balance"] = round(bank["balance"] - stake, 2); open_stake += stake
        bank["bets"].append({"date": f"{today:%Y-%m-%d}", "legs": legs, "stake": stake, "odds": round(odds, 2),
                             "desc": desc, "status": "abierta"})
        lines.append(f"{desc} · {stake:.2f} $")

    for s in sorted(new, key=lambda s: s["edge"], reverse=True)[:MAX_SINGLES]:
        kelly = (s["p"] * s["odds"] - 1) / (s["odds"] - 1)
        bet([[s["key"], s["market"]]], bank["balance"] * min(KELLY * kelly, MAX_STAKE), label(s), s["odds"])

    legs, seen = [], set()
    for s in sorted(new, key=lambda s: s["p"], reverse=True):
        if s["p"] >= COMBO_MIN_P and s["key"] not in seen and len(legs) < COMBO_LEGS:
            legs.append(s); seen.add(s["key"])
    if len(legs) >= 2:
        odds = math.prod(s["odds"] for s in legs)
        bet([[s["key"], s["market"]] for s in legs], bank["balance"] * COMBO_STAKE,
            "🔗 Combinada: " + " + ".join(pick(s) + (f" ({s['match']})" if s["market"] in ("D", "O25", "U25") else "") for s in legs) + f" @ {odds:.2f}", odds)
    return lines


def run(state, today):
    bank = state.setdefault("bank", {"balance": START, "bets": []})
    settled = settle(bank, state["signals"])
    placed = place(bank, state["signals"], today)
    if not (settled or placed): return
    open_bets = [b for b in bank["bets"] if b["status"] == "abierta"]
    total = bank["balance"] + sum(b["stake"] for b in open_bets)
    msg = (["🎯 APUESTAS"] + placed + [""] if placed else []) + (["📋 RESULTADOS"] + settled + [""] if settled else [])
    msg.append(f"💰 Banca: {total:.2f} $ ({total - START:+.2f} $ desde 100 $)")
    A.send_text("\n".join(msg), force=True)
