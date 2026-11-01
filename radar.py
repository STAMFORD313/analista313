"""Checklist de cada apuesta candidata, radar diario de "casi apuestas" cuando el bot no apuesta nada,
y una página por deporte (apuestas/<deporte>.md) que se abre desde los botones del mensaje fijado."""
import pandas as pd

import analyst as A
import bankroll as B

SPORTS = {"⚽": ("futbol", "Fútbol"), "🏀": ("nba", "NBA"), "🏈": ("nfl", "NFL"), "🏒": ("nhl", "NHL"),
          "⚾": ("mlb", "MLB"), "🥊": ("ufc", "UFC")}
PAGES = A.DATA.parent / "apuestas"
RADAR_HOUR, RADAR_TOP = 10, 5  # a partir de las 10:00 (España), una vez al día, las 5 más cercanas


def checklist(s, bad=frozenset()):
    """[(ok, texto)] con cada condición que debe cumplir una apuesta para que el bot la coja."""
    ai = s.get("ai") or {}
    conf = ai.get("confianza")
    return [
        (s["p"] >= B.SURE_P, f"Probabilidad {s['p']:.0%} (mín. {B.SURE_P:.0%})"),
        (A.EDGE <= s["edge"] <= A.MAX_EDGE, f"Ventaja {s['edge']:+.0%} (entre {A.EDGE:.0%} y {A.MAX_EDGE:.0%})"),
        (B.segment(s) not in bad, f"Segmento «{B.segment(s)}» sin pérdidas"),
        (bool(ai) and ai.get("apostar") is not False and (not str(conf).isdigit() or int(conf) >= B.SURE_CONF),
         "IA: " + (f"confianza {conf}/5 (mín. {B.SURE_CONF})" + (f" · {ai['motivo']}" if ai.get("motivo") else "")
                   if ai else "sin revisar (no pasó lo anterior)")),
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
    bad = {k for k, v in B.seg_stats(bank, state["signals"]).items() if v["n"] >= B.SEG_MIN and v["profit"] / v["staked"] < B.SEG_ROI}
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
    bad = {k for k, v in B.seg_stats(bank, state["signals"]).items() if v["n"] >= B.SEG_MIN and v["profit"] / v["staked"] < B.SEG_ROI}
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


def menu():
    url = lambda slug: f"https://github.com/{B.REPO}/blob/main/apuestas/{slug}.md"
    btns = [{"text": f"{icon} {name}", "url": url(slug)} for icon, (slug, name) in SPORTS.items()]
    return [btns[i:i + 3] for i in range(0, len(btns), 3)]


def run(state, today):
    radar(state, today)
    pages(state)
