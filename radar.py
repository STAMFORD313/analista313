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
    new = sorted((s for s in upcoming(state) if s.get("sport") == FC_ICON and not s.get("anunciada") and s.get("book") == "Unibet"
                  and 0.05 <= s["edge"] <= 0.3), key=lambda s: -s["edge"])[:5]  # solo con valor frente a Unibet (no el favorito)
    if not new: return
    import boleto
    lines = [f"{FC_ICON} <b>FC 27 · APUESTAS</b> (simuladas, 1 €)"]
    for s in new:
        lines += ["", *boleto.boleto(s["match"], [(B.pick_text(s), None)], s["odds"], 1.0, when=B.when(s))]
        s["anunciada"] = True
    A.send_card("\n".join(lines))
    state["fc_sent"] = now.isoformat()


COMBO_EVERY, COMBO_MAX = pd.Timedelta(hours=3), 5


def combo_slips(state, s):
    """Predicción en formato boleto: cada combinada del partido con sus selecciones (sin marcas), cuota y apuesta simulada."""
    import boleto
    v = (state.get("sgp") or {}).get(s["key"])
    head = [f"<b>{A.esc(s['match']).replace(' vs ', ' 🆚 ')}</b> · 🕒 {B.when(s)} · {A.esc(s.get('league', ''))}"]
    if not v:  # sin combinadas registradas: las líneas de texto de siempre
        i = next((j for j, l in enumerate(s["info"]) if l.startswith("🧩 <b>Marcador exacto")), None)  # esports: el bloque entero
        if i is not None: return head + s["info"][i:i + 3] + [""]
        return head + [l for l in s["info"] if l.startswith(("🧩", "🚀", "🎯"))] + [""]
    out = head
    name = {"corta": "🧩 Crear apuesta", "larga": "🚀 Combinada larga", "valor": "🎯 Con cuota real"}
    for c in v["combos"]:
        odds, lab = (c["odds"], "Cuota aprox.") if c.get("odds") else (1 / c["p"], "Cuota justa")
        out += [f"{name.get(c['kind'], '🧩 Combinada')} · prob. {c['p']:.0%}" + (" · <b>SIMULADA</b>" if c.get("sim") else ""),
                *boleto.boleto(None, [(l["text"], None) for l in c["legs"]], odds, 1.0 if c.get("sim") else None, odds_label=lab)]
    return out + [""]


SGP_EVERY = pd.Timedelta(hours=6)


def sgp_results_card(state):
    """Resultados de las combinadas ya jugadas (data/sgp_log.csv), en formato boleto con ✅/❌ en cada selección:
    las nuevas desde la última tarjeta, como mucho cada 6 h y 8 por tarjeta."""
    import boleto, football_stats
    now = pd.Timestamp.now(tz="UTC")
    if state.get("sgp_res_sent") and now - pd.Timestamp(state["sgp_res_sent"]) < SGP_EVERY: return
    if not football_stats.SGP_LOG.exists(): return
    d = pd.read_csv(football_stats.SGP_LOG)
    new = d.iloc[state.get("sgp_res_n", 0):]
    if new.empty or "piernas_ok" not in d: return
    new = new[new.piernas_ok.notna()]
    sim = new[new.simulada == 1] if "simulada" in new else new.iloc[:0]
    show = sim.head(8)  # solo las que el bot «apostó»
    if show.empty: state["sgp_res_n"] = len(d); return
    lines = ["📋 <b>RESULTADOS DE COMBINADAS</b>"]
    for r in show.itertuples():
        oks = str(r.piernas_ok).split(".")[0].zfill(int(r.piernas))
        legs = [(t, o == "1") for t, o in zip(str(r.piernas_txt).split(" + "), oks)]
        odds = r.cuota if pd.notna(getattr(r, "cuota", None)) else 1 / r.p
        lines += ["", *boleto.boleto(r.partido, legs, odds, 1.0)]
    sm = d[d.simulada == 1] if "simulada" in d else d.iloc[:0]
    if len(sm): lines += ["", f"💰 Total simuladas: {len(sm)} · {sm.ganancia.sum():+.2f} €"]
    A.send_card("\n".join(lines))
    state["sgp_res_sent"], state["sgp_res_n"] = now.isoformat(), len(d)


def combos_card(state):
    """Tarjeta «🧩 COMBINADAS»: solo las combinadas que el bot apuesta (simuladas a 1 €), una por partido (la de más
    ventaja), como boleto: partido, selecciones, cuota y apuesta. Como mucho cada 3 h y 6 partidos."""
    import boleto
    now = pd.Timestamp.now(tz="UTC")
    if state.get("combo_sent") and now - pd.Timestamp(state["combo_sent"]) < COMBO_EVERY: return
    picks = []
    for k, v in (state.get("sgp") or {}).items():
        if v.get("anunciada") or pd.Timestamp(v["start"]) <= now: continue
        sims = [c for c in v["combos"] if c.get("sim") and c.get("odds")]
        if sims: picks.append((v["start"], k, v, max(sims, key=lambda c: c["edge"])))
    picks = sorted(picks, key=lambda x: x[0])[:6]
    if not picks: return
    lines = ["🧩 <b>COMBINADAS</b> (simuladas, 1 €)"]
    for start, k, v, c in picks:
        lines += ["", *boleto.boleto(v["match"], [(l["text"], None) for l in c["legs"]], c["odds"], 1.0,
                                     when=pd.Timestamp(start).tz_convert("Europe/Madrid").strftime("%d/%m · %H:%M"))]
        v["anunciada"] = True
    A.send_card("\n".join(lines))
    state["combo_sent"] = now.isoformat()


SIM_EVERY = pd.Timedelta(hours=6)
NAMES_SPORT = {"⚽": "Fútbol", "🏀": "NBA / baloncesto", "🏈": "NFL / americano", "🏒": "NHL", "⚾": "MLB", "🎾": "Tenis", "🥊": "UFC",
               "🎮": "CS2", "🐉": "Dota 2", "🧙": "LoL", "🔫": "R6", "📱": "MLBB", "🎮⚽": "FC 27", "🏆": "Champions/Europa",
               "nba": "NBA / baloncesto", "nfl": "NFL / americano", "nhl": "NHL", "mlb": "MLB"}


def paper_lines(min_n=3):
    """Banco de pruebas: TODAS las candidatas liquidadas con cuota real (data/senales.csv), apostadas o no, a 1 u cada
    una: por deporte y mercado, prob. del bot → acierto real y ROI ± error típico. Para ver dónde falla con muestra."""
    import numpy as np
    if not B.LOG.exists(): return []
    d = pd.read_csv(B.LOG)
    d = d[~d.casa.astype(str).str.startswith("cuota justa")]
    if d.empty: return []
    d["g"] = np.where(d.gano.astype(float) == 1, d.retorno.astype(float) - 1, -1.0)
    d["mk"] = d.mercado.astype(str).str.replace(r"^(corners|cards|fouls|sot)_.*", r"\1", regex=True).map(
        lambda m: {"H": "1X2", "D": "1X2", "A": "1X2", "home": "ganador", "away": "ganador", "t1": "ganador", "t2": "ganador",
                   "over": "total", "under": "total", "O25": "total", "U25": "total", "AHH": "hándicap", "AHA": "hándicap",
                   "sph": "hándicap", "spa": "hándicap", "corners": "córners"}.get(m, m))
    out = [f"🧪 <b>BANCO DE PRUEBAS</b> · {len(d)} candidatas liquidadas a 1 u (apostadas o no)",
           f"Total: prob. bot {d.p.mean():.0%} → acierto {d.gano.mean():.0%} · ROI <b>{d.g.mean():+.1%}</b> ± {d.g.std() / np.sqrt(len(d)):.1%}"]
    for (sp, mk), g in sorted(d.groupby([d.deporte.astype(str), "mk"]), key=lambda x: -len(x[1])):
        if len(g) < min_n: continue
        se = g.g.std() / np.sqrt(len(g))
        flag = "⚪" if len(g) < 20 else "🔴" if g.g.mean() + 2 * se < 0 else "🟢" if g.g.mean() - 2 * se > 0 else "⚪"
        out.append(f"{flag} {NAMES_SPORT.get(sp, sp)} · {mk} ({len(g)}): {g.p.mean():.0%} → {g.gano.mean():.0%} · ROI {g.g.mean():+.0%} ± {se:.0%}")
    try:  # props de jugador de fútbol con cuota real de Unibet, liquidadas con ESPN (football_players.py)
        import football_players
        if football_players.LOG.exists() and (pp := football_players.report()):
            out += ["⚽👤 <b>Props de jugador (Unibet)</b>: real · prob. casa · prob. modelo", *pp]
        import nba_props
        if (pn := nba_props.report()): out += ["🏀👤 <b>Props NBA (Unibet)</b>: modelo vs casa", *pn]
    except Exception as e: print("props de jugador", repr(e))
    out.append("🟢/🔴 = gana/pierde con margen (2 errores típicos, mín. 20) · ⚪ = aún es ruido: hacen falta más partidos")
    return out


PAPER_EVERY = pd.Timedelta(hours=24)


def paper_card(state):
    """Una vez al día: tarjeta del banco de pruebas."""
    now = pd.Timestamp.now(tz="UTC")
    if state.get("paper_sent") and now - pd.Timestamp(state["paper_sent"]) < PAPER_EVERY: return
    lines = paper_lines()
    if not lines: return
    A.send_card("\n".join(lines))
    state["paper_sent"] = now.isoformat()


def fc_sim_card(state):
    """Tarjeta «FC 27 · SIMULACIÓN» como mucho cada 6 h y solo si hay pronósticos jugados nuevos desde la anterior."""
    import esoccer
    now = pd.Timestamp.now(tz="UTC")
    if state.get("fc_sim_sent") and now - pd.Timestamp(state["fc_sim_sent"]) < SIM_EVERY: return
    d = esoccer.simulation()
    if d.empty or len(d) <= state.get("fc_sim_n", 0): return
    A.send_card("\n".join([f"{FC_ICON} <b>FC 27 · RESULTADOS</b>", *esoccer.sim_lines(10)]))
    state["fc_sim_sent"], state["fc_sim_n"] = now.isoformat(), len(d)


def run(state, today):
    radar(state, today)
    fc_card(state)
    try: fc_sim_card(state)
    except Exception as e: print("simulación FC", repr(e))
    try: paper_card(state)
    except Exception as e: print("banco de pruebas", repr(e))
    try: sgp_results_card(state)
    except Exception as e: print("resultados de combinadas", repr(e))
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
