"""Checklist de cada apuesta candidata, radar diario de "casi apuestas" cuando el bot no apuesta nada,
y una página por deporte (apuestas/<deporte>.md) que se abre desde los botones del mensaje fijado."""
import pandas as pd

import football_stats

import analyst as A
import bankroll as B
import esports

SPORTS = {"⚽": ("futbol", "Fútbol"), "🏀": ("nba", "NBA·NCAA"), "🏈": ("nfl", "NFL·NCAA"), "🏆": ("europa", "Champions·Europa"), "🏒": ("nhl", "NHL"),
          "⚾": ("mlb", "MLB"), "🥊": ("ufc", "UFC"), "🎾": ("tenis", "Tenis")}
SPORTS.update({icon: (slug, name) for slug, (_, icon, name) in esports.DISCIPLINES.items()})  # cada esport nuevo (p. ej. FC 27) sale solo
PAGES = A.DATA.parent / "apuestas"
RADAR_HOUR, RADAR_TOP = 10, 5  # a partir de las 10:00 (España), una vez al día, las 5 más cercanas


def checklist(s, bad=frozenset()):
    """[(ok, texto)] con cada condición que debe cumplir una apuesta para que el bot la coja."""
    return [
        (B.value_ok(s), f"Valor: ventaja {s['edge']:+.0%} (mín. {B.margin(s):.0%} con la incertidumbre, máx. {A.MAX_EDGE:.0%})"),
        (B.segment(s) not in bad, f"Segmento «{B.segment(s)}» sin pérdidas demostradas"),
    ]


def upcoming(state):
    """Señales con datos completos de partidos aún no empezados; marca las que ya son apuesta del bot."""
    now = pd.Timestamp.now(tz="UTC")
    placed = {tuple(l) for b in state["bank"]["bets"] for l in b["legs"]}
    out = [s for s in state["signals"] if s.get("info") and pd.Timestamp(s["start"]) > now and s.get("result") is None]
    for s in out: s["placed"] = (s["key"], s["market"]) in placed
    return out


def radar(state, today):
    """Si hoy no ha salido ninguna apuesta, manda una vez las candidatas más cercanas a cumplir el checklist."""
    bank, now = state["bank"], pd.Timestamp.now(tz="Europe/Madrid")
    if state.get("radar_date") == f"{today:%Y-%m-%d}" or now.hour < RADAR_HOUR: return
    if any(b["date"] == f"{today:%Y-%m-%d}" for b in bank["bets"]): return
    cands = [s for s in upcoming(state) if not s["placed"]
             and pd.Timestamp(s["start"]) < pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=36)]
    if not cands: return
    bad = B.bad_segments(B.seg_stats(bank, state["signals"]))
    cands.sort(key=lambda s: (sum(ok for ok, _ in checklist(s, bad)), s["p"]), reverse=True)
    lines = ["🔭 <b>RADAR DEL DÍA</b>", "Hoy no hay apuestas seguras. Las más cercanas y por qué aún no las cojo:", B.LINE]
    for s in cands[:RADAR_TOP]:
        lines += [f"{s.get('sport', '')} <b>{A.esc(s['match'])}</b> · 🕒 {B.when(s)}",
                  f"→ {A.esc(B.pick_text(s))} @ {s['odds']:.2f} ({A.esc(s.get('book', ''))})",
                  *[f"   {'✅' if ok else '❌'} {A.esc(t)}" for ok, t in checklist(s, bad)], ""]
    A.send_card("\n".join(lines))
    state["radar_date"] = f"{today:%Y-%m-%d}"


def pages(state):
    """Una página por deporte con todas sus apuestas de valor próximas (cogidas o no) y su checklist."""
    PAGES.mkdir(exist_ok=True)
    bank = state["bank"]
    bad = B.bad_segments(B.seg_stats(bank, state["signals"]))
    for icon, (slug, name) in SPORTS.items():
        sig = sorted((s for s in upcoming(state) if s.get("sport") == icon), key=lambda s: s["start"])
        md = [f"# {icon} {name} · apuestas de valor", "",
              f"Actualizado {pd.Timestamp.now(tz='Europe/Madrid'):%d/%m/%Y %H:%M} (hora España). "
              "✅ = el bot apuesta · ❌ = condición que falla.", ""]
        if not sig: md.append("_No hay apuestas de valor próximas en este deporte ahora mismo._")
        for s in sig:
            head = "💵 **APOSTADA**" if s["placed"] else "👀 candidata"
            md += [f"## {s['match']} — {B.when(s)}", "",
                   f"{head} · {s.get('league', '')} · **{B.pick_text(s)}** @ **{s['odds']:.2f}** ({s.get('book', '')}) · "
                   f"prob {s['p']:.0%} vs casas {s.get('implied', 1 / s['odds']):.0%} · ventaja {s['edge']:+.0%}", "",
                   *[f"- {'✅' if ok else '❌'} {t}" for ok, t in checklist(s, bad)], ""]
            md += [l.replace("<b>", "**").replace("</b>", "**") + "  " for l in s["info"] if l] + [""]  # "  " = salto de línea en markdown
        (PAGES / f"{slug}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def menu(): return sport_buttons()  # botones de filtro (los atiende menu.py dentro de Telegram)


FC_ICON, FC_EVERY = "🎮⚽", pd.Timedelta(hours=4)


def fc_card(state):
    """FC 27 no se apuesta (sin cuotas de casa): como mucho cada 4 h, una tarjeta con sus pronósticos aún no enviados."""
    now = pd.Timestamp.now(tz="UTC")
    if state.get("fc_sent") and now - pd.Timestamp(state["fc_sent"]) < FC_EVERY: return
    new = sorted((s for s in upcoming(state) if s.get("sport") == FC_ICON and not s.get("anunciada")), key=lambda s: -s["p"])[:5]
    if not new: return
    lines = [f"{FC_ICON} <b>FC 27 · PRONÓSTICOS</b>", "Con cuota de Unibet cuando la hay; el bot aún no apuesta (sin ROI demostrado)."]
    try:
        import esoccer
        if (m := esoccer.market_report()):
            lines.append(f"📈 Frente a Unibet: {m['partidos']} partidos · log-loss modelo {m['ll_modelo']} vs casa {m['ll_mercado']}"
                         + (f" · {m['apuestas']} con ventaja ≥5 %: ROI {m['roi']:+.0%} ± {m['roi_se'] or 0:.0%}" if m["apuestas"] else ""))
    except Exception as e: print("informe FC", repr(e))
    lines.append(B.LINE)
    for s in new:
        lines += [f"<b>{A.esc(s['match']).replace(' vs ', ' 🆚 ')}</b> · 🕒 {B.when(s)}", f"→ <b>{A.esc(B.pick_text(s))}</b> ({s['p']:.0%})",
                  *[f"   {l}" for l in s["info"][1:5]], *[f"   {l}" for l in s["info"] if l.startswith("• Unibet")], ""]
        s["anunciada"] = True
    A.send_card("\n".join(lines))
    state["fc_sent"] = now.isoformat()


COMBO_EVERY, COMBO_MAX = pd.Timedelta(hours=3), 5


def combos_card(state):
    """Tarjeta «🧩 COMBINADAS» con las combinadas de estadísticas (córners, faltas, tarjetas, tiros a puerta, goles) de los
    próximos partidos de fútbol, de football_stats.card_lines / euro_lines: una vez por partido, como mucho cada 3 h y
    5 partidos por tarjeta (los de antes primero). Son candidatas: el bot no las apuesta."""
    now = pd.Timestamp.now(tz="UTC")
    if state.get("combo_sent") and now - pd.Timestamp(state["combo_sent"]) < COMBO_EVERY: return
    by = {}
    for s in sorted(upcoming(state), key=lambda s: s["start"]):
        if s["key"] in by or s.get("combo_anunciada") or not any(l.startswith(("🧩", "🚀")) for l in s["info"]): continue
        by[s["key"]] = s
    new = list(by.values())[:COMBO_MAX]
    if not new: return
    lines = ["🧩 <b>COMBINADAS DE ESTADÍSTICAS</b>",
             "Córners, faltas, tarjetas y tiros a puerta de cada equipo (su media, forma y rival). Cuota justa = la mínima "
             "que tiene que pagar la casa para que valga la pena. Candidatas: el bot no las apuesta.",
             *([h] if (h := football_stats.sgp_history()) else ["📈 Historial real: aún sin combinadas liquidadas (se registran desde hoy)."]),
             B.LINE]
    for s in new:
        st = [l for l in s["info"] if l.startswith(("Córners", "Faltas", "Tarjetas", "Tiros a puerta"))]
        lines += [f"<b>{A.esc(s['match']).replace(' vs ', ' 🆚 ')}</b> · 🕒 {B.when(s)} · {A.esc(s.get('league', ''))}",
                  *[f"   {l}" for l in st], *[l for l in s["info"] if l.startswith(("🧩", "🚀"))], ""]
        for x in state["signals"]:
            if x["key"] == s["key"]: x["combo_anunciada"] = True
    A.send_card("\n".join(lines))
    state["combo_sent"] = now.isoformat()


SIM_EVERY = pd.Timedelta(hours=6)


def fc_sim_card(state):
    """Tarjeta «FC 27 · SIMULACIÓN» como mucho cada 6 h y solo si hay pronósticos jugados nuevos desde la anterior."""
    import esoccer
    now = pd.Timestamp.now(tz="UTC")
    if state.get("fc_sim_sent") and now - pd.Timestamp(state["fc_sim_sent"]) < SIM_EVERY: return
    d = esoccer.simulation()
    if d.empty or len(d) <= state.get("fc_sim_n", 0): return
    A.send_card("\n".join([f"{FC_ICON} <b>FC 27 · SIMULACIÓN</b>", "Cómo habrían salido los pronósticos que te mandó el bot.", B.LINE,
                            *esoccer.sim_lines(10)]))
    state["fc_sim_sent"], state["fc_sim_n"] = now.isoformat(), len(d)


def run(state, today):
    radar(state, today)
    fc_card(state)
    try: fc_sim_card(state)
    except Exception as e: print("simulación FC", repr(e))
    combos_card(state)
    pages(state)


def sport_view(state, slug, limit=12):
    """Texto (HTML de Telegram) con las próximas apuestas y candidatas de un deporte."""
    icon, name = next((i, n) for i, (s, n) in SPORTS.items() if s == slug)
    bad = B.bad_segments(B.seg_stats(state["bank"], state["signals"]))
    sig = sorted((s for s in upcoming(state) if s.get("sport") == icon),
                 key=lambda s: (not s["placed"], -sum(ok for ok, _ in checklist(s, bad)), s["start"]))[:limit]
    lines = [f"{icon} <b>{name.upper()}</b> · próximas apuestas", "💵 apostada · 👀 candidata · ✅/❌ checklist", B.LINE]
    if not sig: lines.append("Ahora mismo no hay partidos con cuotas analizados en este deporte.")
    for s in sig:
        marks = "".join("✅" if ok else "❌" for ok, _ in checklist(s, bad))
        lines += [f"{'💵' if s['placed'] else '👀'} <b>{A.esc(s['match']).replace(' vs ', ' 🆚 ')}</b> · 🕒 {B.when(s)}",
                  f"   → {A.esc(B.pick_text(s))} @ <b>{s['odds']:.2f}</b> ({A.esc(s.get('book', ''))}) · {s['p']:.0%} · {s['edge']:+.0%} {marks}"]
    lines += [B.LINE, "Checklist: valor (con incertidumbre) · segmento"]
    if slug == "fc":  # FC 27: cómo habrían salido los pronósticos ya enviados
        try:
            import esoccer
            lines += [B.LINE, *esoccer.sim_lines()]
        except Exception as e: print("simulación FC", repr(e))
    return "\n".join(lines)


def sport_buttons():
    btns = [{"text": f"{icon} {name}", "callback_data": f"dep:{slug}"} for icon, (slug, name) in SPORTS.items()]
    return [btns[i:i + 3] for i in range(0, len(btns), 3)]
