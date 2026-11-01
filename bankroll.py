"""Banca virtual: el bot empieza con 100 $ y apuesta (simulado) a las señales de valor con cuotas reales
(la mejor entre casas en fútbol; DraftKings en NBA/NFL/NHL/MLB/UFC). Antes de cada apuesta la IA revisa
todos los datos y puede vetarla. Simples con 1/4 de Kelly + una combinada por ejecución.
Cada apuesta y cada resultado se envía como una tarjeta de Telegram."""
import math

import pandas as pd

import analyst as A

START = 100.0
KELLY, MAX_STAKE, MAX_EXPOSURE = 0.25, 0.05, 0.30  # 1/4 Kelly, máx. 5 % por apuesta y 30 % de la banca en juego
MAX_SINGLES, COMBO_LEGS, COMBO_MIN_P, COMBO_STAKE = 8, 3, 0.45, 0.02
LINE = "━━━━━━━━━━━━━━━"
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]


def pick(s):
    home, away = s["match"].split(" vs ", 1)
    return s.get("pick") or {"H": home, "A": away}.get(s["market"]) or A.NAMES[s["market"]]


def pick_text(s):
    p = pick(s)
    return p if s["market"] in ("D", "O25", "U25") else f"Gana {p}"


def when(s): return pd.Timestamp(s["start"]).tz_convert("Europe/Madrid").strftime("%d/%m · %H:%M")


def stars(n): return "⭐" * n + "☆" * (5 - n)


def total(bank): return bank["balance"] + sum(b["stake"] for b in bank["bets"] if b["status"] == "abierta")


def bank_line(bank):
    t = total(bank)
    return f"💰 Banca: <b>{t:.2f} $</b> ({t - START:+.2f} $ desde 100 $) · libre {bank['balance']:.2f} $"


def ai_review(s):
    """La IA cruza todos los datos y decide; sin IA (o si falla) se apuesta con el criterio estadístico."""
    r = A.ai_json(f"""Eres un analista profesional de apuestas deportivas. Datos del partido:
{s.get('context', s['match'])}
Mi modelo propone apostar a "{pick_text(s)}" a cuota {s['odds']} ({s.get('book', '')}); probabilidad estimada {s['p']:.0%},
probabilidad implícita de las casas {s.get('implied', 1 / s['odds']):.0%}, ventaja {s['edge']:+.0%}.
Valora forma, bajas (titulares o no), historial y contexto, y decide si la apuesta tiene valor real.
Responde SOLO con JSON: {{"apostar": true, "confianza": 3, "motivo": "1-2 frases en español"}} (confianza de 1 a 5).""")
    if not r: return True, None, None
    return bool(r.get("apostar", True)), r.get("motivo"), r.get("confianza")


def single_card(s, stake, bank, reason, conf):
    conf = int(conf) if str(conf).isdigit() else 1 + min(4, round(s["edge"] * 20))
    return "\n".join([
        f"{s.get('sport', '🎯')} <b>{A.esc(s.get('league', ''))}</b>",
        f"🕒 {when(s)} (hora España)",
        f"<b>{A.esc(s['match']).replace(' vs ', ' 🆚 ')}</b>",
        LINE,
        f"🎯 <b>APUESTA: {A.esc(pick_text(s))}</b>",
        f"💸 Cuota <b>{s['odds']:.2f}</b> · {A.esc(s.get('book', ''))}",
        f"💵 Apuesta <b>{stake:.2f} $</b> → cobra <b>{stake * s['odds']:.2f} $</b> si acierta",
        f"📈 Prob. bot <b>{s['p']:.0%}</b> vs casas {s.get('implied', 1 / s['odds']):.0%} · ventaja <b>{s['edge']:+.0%}</b>",
        f"{stars(max(1, min(5, conf)))} confianza",
        LINE, *s.get("info", []), LINE,
        *([f"🤖 <i>{A.esc(reason)}</i>"] if reason else []),
        bank_line(bank)])


def combo_card(legs, stake, odds, bank):
    p = math.prod(s["p"] for s in legs)
    return "\n".join([
        f"🔗 <b>COMBINADA</b> · {len(legs)} selecciones", LINE,
        *[f"{NUMS[i]} {s.get('sport', '')} {A.esc(s['match'])}\n     → <b>{A.esc(pick_text(s))}</b> @ {s['odds']:.2f} · 🕒 {when(s)}"
          for i, s in enumerate(legs)],
        LINE,
        f"💸 Cuota total <b>{odds:.2f}</b>",
        f"💵 Apuesta <b>{stake:.2f} $</b> → cobra <b>{stake * odds:.2f} $</b> si aciertan todas",
        f"📈 Prob. bot conjunta {p:.0%} · ventaja {p * odds - 1:+.0%}",
        bank_line(bank)])


def result_card(b, legs, bank):
    profit = b["return"] - b["stake"]
    head = "✅ <b>GANADA</b>" if b["status"] == "ganada" else "❌ <b>PERDIDA</b>"
    rows = [f"{'✅' if s.get('result') is True else '❌' if s.get('result') is False else '➖'} {s.get('sport', '')} "
            f"{A.esc(s['match'])} {s.get('score', '')}\n     → {A.esc(pick_text(s))} @ {s['odds']:.2f}" for s in legs]
    return "\n".join([f"{head} · {profit:+.2f} $" + (" · COMBINADA" if len(legs) > 1 else ""), LINE, *rows, LINE,
                      f"💵 Apostado {b['stake']:.2f} $ → cobrado {b['return']:.2f} $", bank_line(bank)])


def settle(bank, signals):
    by = {(s["key"], s["market"]): s for s in signals}
    for b in bank["bets"]:
        if b["status"] != "abierta": continue
        legs = [by[tuple(l)] for l in b["legs"]]
        res = [s.get("result") for s in legs]
        if any(r is False for r in res): b["status"], b["return"] = "perdida", 0.0
        elif all(r is True or r == "nula" for r in res):
            b["status"], b["return"] = "ganada", round(b["stake"] * math.prod(s["odds"] for s in legs if s["result"] is True), 2)
        else: continue
        bank["balance"] = round(bank["balance"] + b["return"], 2)
        A.send_card(result_card(b, legs, bank))


def place(bank, signals, today, kg):
    now = pd.Timestamp.now(tz="UTC")
    new = [s for s in signals if not s.get("considered")]
    for s in new: s["considered"] = True
    new = [s for s in new if s.get("start") and pd.Timestamp(s["start"]) > now]  # nunca apostar a algo ya empezado
    open_stake = sum(b["stake"] for b in bank["bets"] if b["status"] == "abierta")

    def bet(legs, stake, odds, desc):
        nonlocal open_stake
        room = max(0.0, MAX_EXPOSURE * (bank["balance"] + open_stake) - open_stake)
        stake = round(min(stake, room, bank["balance"]), 2)
        if stake < 1: return None
        bank["balance"] = round(bank["balance"] - stake, 2); open_stake += stake
        b = {"date": f"{today:%Y-%m-%d}", "legs": [[s["key"], s["market"]] for s in legs], "stake": stake,
             "odds": round(odds, 2), "desc": desc, "status": "abierta"}
        bank["bets"].append(b); kg.bet(b, signals)
        return stake

    approved = []
    for s in sorted(new, key=lambda s: s["edge"], reverse=True)[:MAX_SINGLES * 2]:
        if len(approved) >= MAX_SINGLES: break
        ok, reason, conf = ai_review(s)
        s["ai"] = {"apostar": ok, "motivo": reason, "confianza": conf}
        if not ok: continue
        kelly = (s["p"] * s["odds"] - 1) / (s["odds"] - 1)
        stake = bet([s], bank["balance"] * min(KELLY * kelly, MAX_STAKE), s["odds"], f"{s['match']} → {pick_text(s)}")
        if stake:
            approved.append(s)
            A.send_card(single_card(s, stake, bank, reason, conf))

    legs, seen = [], set()
    for s in sorted(approved, key=lambda s: s["p"], reverse=True):
        if s["p"] >= COMBO_MIN_P and s["key"] not in seen and len(legs) < COMBO_LEGS:
            legs.append(s); seen.add(s["key"])
    if len(legs) >= 2:
        odds = math.prod(s["odds"] for s in legs)
        stake = bet(legs, bank["balance"] * COMBO_STAKE, odds, "Combinada: " + " + ".join(pick_text(s) for s in legs))
        if stake: A.send_card(combo_card(legs, stake, odds, bank))


def run(state, today, kg):
    bank = state.setdefault("bank", {"balance": START, "bets": []})
    settle(bank, state["signals"])
    place(bank, state["signals"], today, kg)
