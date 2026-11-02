"""Banca virtual: el bot empieza con 100 $ y apuesta (simulado) a las señales de valor con cuotas reales
(la mejor entre casas en fútbol; DraftKings en NBA/NFL/NHL/MLB/UFC). Antes de cada apuesta la IA revisa
todos los datos y puede vetarla. Simples con 1/10 de Kelly + una combinada por ejecución.
Cada apuesta y cada resultado se envía como una tarjeta de Telegram."""
import json, math, os

import pandas as pd

import analyst as A

START = 100.0
KELLY, MAX_STAKE, MAX_EXPOSURE = 0.10, 0.05, 0.30  # 1/10 Kelly (backtest: ver MEJORAS.md), máx. 5 % por apuesta y 30 % en juego
MAX_SINGLES, COMBO_LEGS, COMBO_MIN_P, COMBO_STAKE = 8, 3, 0.55, 0.02
SURE_P, SURE_CONF = 0.55, 4  # solo apuesta (y manda) lo que de verdad cree que gana: prob >= 55 % y la IA con confianza >= 4/5
SEG_MIN, SEG_ROI = 15, -0.10  # autoajuste: deja de apostar en segmentos con >= 15 apuestas y ROI < -10 %
VOID_DAYS = 14  # apuesta sin resultado 14 días después del inicio → nula (devuelve el stake)
REPO = os.getenv("GITHUB_REPOSITORY", "STAMFORD313/analista313")
STATS_MD = A.DATA.parent / "ESTADISTICAS.md"
LOG = A.DATA / "senales.csv"  # todas las señales liquidadas (apostadas o no): calibración, Brier y CLV a largo plazo
BANDS = [0, 0.4, 0.55, 0.7, 1.01]
LINE = "━━━━━━━━━━━━━━━"
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]


def pick(s):
    home, away = s["match"].split(" vs ", 1)
    return s.get("pick") or {"H": home, "A": away}.get(s["market"]) or A.NAMES[s["market"]]


def pick_text(s):
    p = pick(s)
    return p if s["market"] in ("D", "O25", "U25", "over", "under", "AHH", "AHA", "sph", "spa") else f"Gana {p}"


def when(s): return pd.Timestamp(s["start"]).tz_convert("Europe/Madrid").strftime("%d/%m · %H:%M")


def stars(n): return "⭐" * n + "☆" * (5 - n)


def total(bank): return bank["balance"] + sum(b["stake"] for b in bank["bets"] if b["status"] == "abierta")


def bank_line(bank):
    t = total(bank)
    return f"💰 Banca: <b>{t:.2f} $</b> ({t - START:+.2f} $ desde 100 $) · libre {bank['balance']:.2f} $"


def segment(s):
    kind = "goles" if s["market"] in ("O25", "U25") else "totales" if s["market"] in ("over", "under") else "hándicap" if s["market"] in ("AHH", "AHA", "sph", "spa") else "empate" if s["market"] == "D" else "favorito" if s["odds"] < 2 else "no favorito"
    return f"{s.get('sport', '?')} {kind}"


def seg_stats(bank, signals):
    by, out = {(s["key"], s["market"]): s for s in signals}, {}
    for b in bank["bets"]:
        if b["status"] == "abierta" or len(b["legs"]) != 1: continue
        o = out.setdefault(segment(by[tuple(b["legs"][0])]), {"n": 0, "won": 0, "profit": 0.0, "staked": 0.0})
        o["n"] += 1; o["won"] += b["status"] == "ganada"; o["profit"] += b["return"] - b["stake"]; o["staked"] += b["stake"]
    return out


def spread_result(market, line, hs, as_):
    """Hándicap de puntos (NBA/NFL): market 'sph' (local) o 'spa' (visitante), `line` desde ese lado (-5.5 = da 5.5).
    True si cubre, False si no, 'nula' si empata con la línea (push: se devuelve el stake)."""
    m = (float(hs) - float(as_)) * (1 if market == "sph" else -1) + line
    return "nula" if m == 0 else m > 0


def header(s):
    """Cabecera deporte + liga sin repetir el icono (las ligas de ESPN y CS2 ya lo llevan)."""
    sport, league = s.get("sport", "🎯"), s.get("league", "")
    return f"{'' if league.startswith(sport) else sport + ' '}<b>{A.esc(league)}</b>"


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
        header(s),
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
    head = {"ganada": "✅ <b>GANADA</b>", "nula": "➖ <b>NULA</b> (se devuelve)"}.get(b["status"], "❌ <b>PERDIDA</b>")
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
    now = pd.Timestamp.now(tz="UTC")
    by = {(s["key"], s["market"]): s for s in signals}
    for b in bank["bets"]:
        if b["status"] != "abierta": continue
        legs = [by[tuple(l)] for l in b["legs"]]
        for s in legs:  # suspendido/cancelado sin resultado (ESPN y CS2 no lo marcan): nula a los VOID_DAYS
            if s.get("result") is None and s.get("start") and pd.Timestamp(s["start"]) < now - pd.Timedelta(days=VOID_DAYS): s["result"] = "nula"
        res = [s.get("result") for s in legs]
        # cobro por pierna: cuota si gana, 1 si nula, 0 si pierde; el hándicap asiático trae su propio factor (medias apuestas)
        f = [s.get("factor", s["odds"] if r is True else 1.0 if r == "nula" else 0.0) for r, s in zip(res, legs) if r is not None]
        if None in res and 0.0 not in f: continue  # pendiente (una pierna perdida ya liquida la combinada)
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


def log_signals(state):
    """Añade a LOG cada señal ya decidida (ganada/perdida) una sola vez; state.json las poda a los 3 días."""
    used = {tuple(l) for b in state["bank"]["bets"] for l in b["legs"]}
    rows = [{"fecha": str(s.get("start", ""))[:10], "deporte": s.get("sport") or ("⚽" if s["key"].split("|")[0] in A.ALL_LEAGUES else s["key"].split("|")[0]), "mercado": s["market"], "p": s["p"],
             "implied": s.get("implied", round(1 / s["odds"], 3)), "odds": s["odds"], "edge": s["edge"],
             "gano": int(s["result"] is True), "retorno": s.get("factor", s["odds"] if s["result"] is True else 0.0),
             "clv": s.get("clv"), "apostada": int((s["key"], s["market"]) in used)}
            for s in state["signals"] if s.get("result") in (True, False) and not s.get("logged")]
    for s in state["signals"]:
        if s.get("result") in (True, False): s["logged"] = True
    if rows: pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def calibration_md(path=None):
    """Calibración por tramos de probabilidad (prob. del bot vs acierto real), Brier del bot frente al de las casas,
    ROI plano y CLV por deporte: lo que miran los analistas profesionales para saber si hay ventaja real."""
    path = path or LOG
    if not path.exists(): return []
    d = pd.read_csv(path)
    if d.empty: return []
    d["tramo"] = pd.cut(d.p, BANDS, right=False)
    out = ["## Calibración del modelo (todas las señales liquidadas)", "",
           "Si el bot está bien calibrado, el acierto real se parece a la prob. media. Brier: menor es mejor; "
           "si el del bot no baja del de las casas, no hay ventaja real.", "",
           "| Prob. bot | Señales | Prob. media | Acierto real | Brier bot | Brier casas | ROI a 1 u. |", "|---|---|---|---|---|---|---|"]
    for name, g in [*((f"{t.left:.0%}-{min(t.right, 1):.0%}", g) for t, g in d.groupby("tramo", observed=True)), ("Total", d)]:
        out.append(f"| {name} | {len(g)} | {g.p.mean():.0%} | {g.gano.mean():.0%} | {((g.p - g.gano) ** 2).mean():.3f} | "
                   f"{((g.implied - g.gano) ** 2).mean():.3f} | {100 * (g.retorno.mean() - 1):+.1f} % |")
    c = d[d.apostada.eq(1) & d.clv.notna()]
    if len(c):
        out += ["", "**CLV de lo apostado** (cuota cogida vs cierre; positivo = se gana al mercado): " +
                " · ".join(f"{k} {100 * g.clv.mean():+.1f} % ({len(g)})" for k, g in c.groupby("deporte"))]
    return out + [""]


def sanity(signals):
    """Una nula sin marcador antes de los VOID_DAYS es imposible (bug de claves CS2 en analyst, nov-2026): vuelve a pendiente."""
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=VOID_DAYS)
    for s in signals:
        if s.get("result") == "nula" and not s.get("score") and s.get("start") and pd.Timestamp(s["start"]) > cut: s["result"] = None


def run(state, today, kg):
    bank = state.setdefault("bank", {"balance": START, "bets": []})
    sanity(state["signals"])
    settle(bank, state["signals"], kg)
    log_signals(state)
    place(bank, state["signals"], today, kg)
    import radar
    radar.run(state, today)
    update_scoreboard(state)


def squares(bets): return "".join({"ganada": "🟩", "nula": "⬜"}.get(b["status"], "🟥") for b in bets) or "—"


def scoreboard(state):
    bank, signals = state["bank"], state["signals"]
    closed = [b for b in bank["bets"] if b["status"] != "abierta"]
    won = [b for b in closed if b["status"] == "ganada"]
    lost = sum(b["status"] == "perdida" for b in closed)
    staked = sum(b["stake"] for b in closed) or 1
    profit = sum(b["return"] - b["stake"] for b in closed)
    bet_legs = {tuple(l) for b in bank["bets"] for l in b["legs"]}
    clv = [s["clv"] for s in signals if "clv" in s and (s["key"], s["market"]) in bet_legs]  # solo lo apostado
    decided = len(won) + lost
    pct = 100 * len(won) / decided if decided else 0  # % de acierto sin contar nulas
    text = "\n".join([
        "📊 <b>MARCADOR DEL BOT</b>", LINE,
        f"🟩 Ganadas: <b>{len(won)}</b> ({pct:.0f} %)",
        f"🟥 Perdidas: <b>{lost}</b> ({100 - pct if decided else 0:.0f} %)",
        *([f"⬜ Nulas: {len(closed) - decided}"] if len(closed) > decided else []),
        f"⏳ En juego: {sum(b['status'] == 'abierta' for b in bank['bets'])}",
        "", f"Últimas {min(30, len(closed))}:", squares(closed[-30:]), LINE,
        f"💵 Beneficio {profit:+.2f} $ · ROI {100 * profit / staked:+.1f} %",
        *([f"📐 CLV medio {100 * sum(clv) / len(clv):+.1f} % (cuota cogida vs cierre)"] if clv else []),
        bank_line(bank)])
    seg = seg_stats(bank, signals)
    md = ["# Estadísticas del bot", "", f"Actualizado: {pd.Timestamp.now(tz='Europe/Madrid'):%d/%m/%Y %H:%M}", "",
          f"**Ganadas {len(won)} / perdidas {lost}** ({pct:.0f} % acierto) · beneficio {profit:+.2f} $ · "
          f"ROI {100 * profit / staked:+.1f} % · banca {total(bank):.2f} $", "", squares(closed), "",
          "## Por segmento (deporte y tipo de apuesta)", "",
          "| Segmento | Apuestas | Ganadas | % | Beneficio | ROI |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['won']} | {100 * v['won'] / v['n']:.0f} % | {v['profit']:+.2f} $ | {100 * v['profit'] / v['staked']:+.1f} % |"
           for k, v in sorted(seg.items())]
    md += ["", *calibration_md(), "## Historial (lo más reciente primero)", "",
           "| Fecha | Apuesta | Cuota | Stake | Resultado | Por qué falló |", "|---|---|---|---|---|---|"]
    md += [f"| {b['date']} | {b['desc']} | {b['odds']} | {b['stake']} $ | "
           f"{ {'ganada': '🟩', 'perdida': '🟥', 'nula': '⬜'}.get(b['status'], '⏳')} "
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
