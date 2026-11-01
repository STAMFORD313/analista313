"""Banca virtual: el bot empieza con 100 $ y apuesta (simulado) a las señales de valor con cuotas reales
(la mejor entre casas en fútbol; DraftKings en NBA/NFL/NHL/MLB/UFC). Antes de cada apuesta la IA revisa
todos los datos y puede vetarla. Simples con 1/4 de Kelly + una combinada por ejecución.
Cada apuesta y cada resultado se envía como una tarjeta de Telegram."""
import json, math, os

import pandas as pd

import analyst as A

START = 100.0
KELLY, MAX_STAKE, MAX_EXPOSURE = 0.25, 0.05, 0.30  # 1/4 Kelly, máx. 5 % por apuesta y 30 % de la banca en juego
MAX_SINGLES, COMBO_LEGS, COMBO_MIN_P, COMBO_STAKE = 8, 3, 0.55, 0.02
SURE_P, SURE_CONF = 0.55, 4  # solo apuesta (y manda) lo que de verdad cree que gana: prob >= 55 % y la IA con confianza >= 4/5
SEG_MIN, SEG_ROI = 15, -0.10  # autoajuste: deja de apostar en segmentos con >= 15 apuestas y ROI < -10 %
REPO = os.getenv("GITHUB_REPOSITORY", "STAMFORD313/analista313")
STATS_MD = A.DATA.parent / "ESTADISTICAS.md"
LINE = "━━━━━━━━━━━━━━━"
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]


def pick(s):
    home, away = s["match"].split(" vs ", 1)
    return s.get("pick") or {"H": home, "A": away}.get(s["market"]) or A.NAMES[s["market"]]


def pick_text(s):
    p = pick(s)
    return p if s["market"] in ("D", "O25", "U25", "over", "under", "AHH", "AHA") else f"Gana {p}"


def when(s): return pd.Timestamp(s["start"]).tz_convert("Europe/Madrid").strftime("%d/%m · %H:%M")


def stars(n): return "⭐" * n + "☆" * (5 - n)


def total(bank): return bank["balance"] + sum(b["stake"] for b in bank["bets"] if b["status"] == "abierta")


def bank_line(bank):
    t = total(bank)
    return f"💰 Banca: <b>{t:.2f} $</b> ({t - START:+.2f} $ desde 100 $) · libre {bank['balance']:.2f} $"


def segment(s):
    kind = "goles" if s["market"] in ("O25", "U25") else "totales" if s["market"] in ("over", "under") else "hándicap" if s["market"] in ("AHH", "AHA") else "empate" if s["market"] == "D" else "favorito" if s["odds"] < 2 else "no favorito"
    return f"{s.get('sport', '?')} {kind}"


def seg_stats(bank, signals):
    by, out = {(s["key"], s["market"]): s for s in signals}, {}
    for b in bank["bets"]:
        if b["status"] == "abierta" or len(b["legs"]) != 1: continue
        o = out.setdefault(segment(by[tuple(b["legs"][0])]), {"n": 0, "won": 0, "profit": 0.0, "staked": 0.0})
        o["n"] += 1; o["won"] += b["status"] == "ganada"; o["profit"] += b["return"] - b["stake"]; o["staked"] += b["stake"]
    return out


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


def autopsy(b, legs):
    """La IA explica por qué falló la apuesta; queda en el grafo y en ESTADISTICAS.md para mejorar."""
    detail = "\n".join(f"- {x.get('context', x['match'])} Apuesta: {pick_text(x)} @ {x['odds']}, prob {x['p']:.0%}. "
                       f"Resultado final {x.get('score', '?')} ({'acertada' if x.get('result') is True else 'fallada'})." for x in legs)
    r = A.ai_json(f"""Eres un analista profesional revisando una apuesta perdida para mejorar el modelo.
{detail}
Explica en 1-2 frases en español la causa más probable del fallo (varianza normal, bajas mal valoradas, forma, cuota sin valor real...)
y si el modelo debe corregir algo. Responde SOLO con JSON: {{"causa": "...", "corregir": "..."}}""")
    return r or {"causa": "sin análisis (IA no disponible)", "corregir": ""}


def settle(bank, signals, kg):
    by = {(s["key"], s["market"]): s for s in signals}
    for b in bank["bets"]:
        if b["status"] != "abierta": continue
        legs = [by[tuple(l)] for l in b["legs"]]
        res = [s.get("result") for s in legs]
        if any(r is None for r in res) and not any(r is False and "factor" not in s for r, s in zip(res, legs)): continue
        # cobro por pierna: cuota si gana, 1 si nula, 0 si pierde; el hándicap asiático trae su propio factor (medias apuestas)
        f = [s.get("factor", s["odds"] if r is True else 1.0 if r == "nula" else 0.0) for r, s in zip(res, legs) if r is not None]
        ret = 0.0 if 0.0 in f else round(b["stake"] * math.prod(f), 2)
        b["status"], b["return"] = ("ganada" if ret > b["stake"] else "perdida" if ret < b["stake"] else "nula"), ret
        bank["balance"] = round(bank["balance"] + b["return"], 2)
        if b["status"] == "perdida": b["autopsia"] = autopsy(b, legs)
        kg.result(b)
        A.send_card(result_card(b, legs, bank))


def place(bank, signals, today, kg):
    now = pd.Timestamp.now(tz="UTC")
    new = [s for s in signals if not s.get("considered")]
    for s in new: s["considered"] = True
    new = [s for s in new if s.get("start") and pd.Timestamp(s["start"]) > now]  # nunca apostar a algo ya empezado
    already = {tuple(l) for x in bank["bets"] for l in x["legs"]}
    new = [s for s in new if (s["key"], s["market"]) not in already]  # nunca repetir una apuesta
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

    segs = seg_stats(bank, signals)
    bad = {k for k, v in segs.items() if v["n"] >= SEG_MIN and v["profit"] / v["staked"] < SEG_ROI}
    approved = []
    for s in sorted(new, key=lambda s: s["p"] * (1 + s["edge"]), reverse=True):
        if len(approved) >= MAX_SINGLES: break
        if s["p"] < SURE_P or not A.EDGE <= s["edge"] <= A.MAX_EDGE or segment(s) in bad: continue
        ok, reason, conf = ai_review(s)
        s["ai"] = {"apostar": ok, "motivo": reason, "confianza": conf}
        if not ok or (str(conf).isdigit() and int(conf) < SURE_CONF): continue
        kelly = (s["p"] * s["odds"] - 1) / (s["odds"] - 1)
        stake = bet([s], bank["balance"] * min(KELLY * kelly, MAX_STAKE), s["odds"], f"{s['match']} → {pick_text(s)}")
        if stake:
            approved.append(s); s["placed"] = True
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
    settle(bank, state["signals"], kg)
    place(bank, state["signals"], today, kg)
    import radar
    radar.run(state, today)
    update_scoreboard(state)


def squares(bets): return "".join({"ganada": "🟩", "nula": "⬜"}.get(b["status"], "🟥") for b in bets) or "—"


def scoreboard(state):
    bank, signals = state["bank"], state["signals"]
    closed = [b for b in bank["bets"] if b["status"] != "abierta"]
    won = [b for b in closed if b["status"] == "ganada"]
    staked = sum(b["stake"] for b in closed) or 1
    profit = sum(b["return"] - b["stake"] for b in closed)
    clv = [s["clv"] for s in signals if "clv" in s and s.get("considered")]
    pct = 100 * len(won) / len(closed) if closed else 0
    text = "\n".join([
        "📊 <b>MARCADOR DEL BOT</b>", LINE,
        f"🟩 Ganadas: <b>{len(won)}</b> ({pct:.0f} %)",
        f"🟥 Perdidas: <b>{len(closed) - len(won)}</b> ({100 - pct if closed else 0:.0f} %)",
        f"⏳ En juego: {sum(b['status'] == 'abierta' for b in bank['bets'])}",
        "", f"Últimas {min(30, len(closed))}:", squares(closed[-30:]), LINE,
        f"💵 Beneficio {profit:+.2f} $ · ROI {100 * profit / staked:+.1f} %",
        *([f"📐 CLV medio {100 * sum(clv) / len(clv):+.1f} % (cuota cogida vs cierre)"] if clv else []),
        bank_line(bank)])
    seg = seg_stats(bank, signals)
    md = ["# Estadísticas del bot", "", f"Actualizado: {pd.Timestamp.now(tz='Europe/Madrid'):%d/%m/%Y %H:%M}", "",
          f"**Ganadas {len(won)} / perdidas {len(closed) - len(won)}** ({pct:.0f} % acierto) · beneficio {profit:+.2f} $ · "
          f"ROI {100 * profit / staked:+.1f} % · banca {total(bank):.2f} $", "", squares(closed), "",
          "## Por segmento (deporte y tipo de apuesta)", "",
          "| Segmento | Apuestas | Ganadas | % | Beneficio | ROI |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['won']} | {100 * v['won'] / v['n']:.0f} % | {v['profit']:+.2f} $ | {100 * v['profit'] / v['staked']:+.1f} % |"
           for k, v in sorted(seg.items())]
    md += ["", "## Historial (lo más reciente primero)", "",
           "| Fecha | Apuesta | Cuota | Stake | Resultado | Por qué falló |", "|---|---|---|---|---|---|"]
    md += [f"| {b['date']} | {b['desc']} | {b['odds']} | {b['stake']} $ | "
           f"{'🟩' if b['status'] == 'ganada' else '🟥' if b['status'] == 'perdida' else '⏳'} "
           f"{(b.get('return', 0) - b['stake']) if b['status'] != 'abierta' else 0:+.2f} $ | {(b.get('autopsia') or {}).get('causa', '')} |"
           for b in reversed(bank["bets"])]
    STATS_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return text


def update_scoreboard(state):
    """Mensaje fijado arriba del chat, editado en cada ejecución, con botón al historial completo."""
    text = scoreboard(state)
    import radar
    markup = json.dumps({"inline_keyboard": [[{"text": "📈 Ver historial completo y fallos",
                                                "url": f"https://github.com/{REPO}/blob/main/ESTADISTICAS.md"}]] + radar.menu()})
    if state.get("scoreboard_id") and state.get("scoreboard_text") == text + markup: return  # sin cambios
    mid = state.get("scoreboard_id")
    if not (mid and A.tg("editMessageText", message_id=mid, text=text, parse_mode="HTML", reply_markup=markup)):
        msg = A.tg("sendMessage", text=text, parse_mode="HTML", reply_markup=markup)
        if msg:
            state["scoreboard_id"] = msg["message_id"]
            A.tg("pinChatMessage", message_id=msg["message_id"], disable_notification="true")
    state["scoreboard_text"] = text + markup
