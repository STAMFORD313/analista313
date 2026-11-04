"""Apuestas simuladas desde el chat de Telegram («simula la apuesta y avísame»): se apuntan en data/chat_simuladas.json
(se sube al repo para que no se pierdan cuando menu.py se releva) y se liquidan solas con las estadísticas de ESPN
(data/espn_soccer, que actualiza el experto datos_futbol_jugadores). Al acabar, menu.py manda el boleto con ✅/❌ en cada
pierna y el porqué (lo que pasó de verdad frente a la línea).

Piernas que entiende (las que escribe chat.py): «Más/Menos de X córners|tarjetas|goles» (totales del partido) y
«<equipo> o empate (1X|X2)»."""
import json, re, subprocess, time

import pandas as pd

import analyst as A
import boleto as B
import espn_soccer as ES

FILE = A.DATA / "chat_simuladas.json"
_MEM = {}  # id → apuesta: copia en memoria por si un git reset de menu.py borra un cambio aún sin subir
LEG_RX = re.compile(r"^(Más|Menos) de ([\d.]+) (córners|tarjetas|goles)$")
DC_RX = re.compile(r"^(.+) o empate \((1X|X2)\)$")
GIVE_UP = pd.Timedelta(days=4)  # sin datos de ESPN 4 días después del partido: la pierna se da por nula


def load():
    bets = {b["id"]: b for b in (json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else [])}
    for k, b in _MEM.items():  # lo de memoria manda (puede ir por delante de lo subido)
        if k not in bets or b.get("rev", 0) >= bets[k].get("rev", 0): bets[k] = b
    return bets


def save(bets, msg="chat: apuestas simuladas"):
    for k, b in bets.items(): _MEM[k] = b
    FILE.write_text(json.dumps(sorted(bets.values(), key=lambda b: b["id"]), ensure_ascii=False, indent=1), encoding="utf-8")
    push(msg)


def push(msg):
    """Sube data/chat_simuladas.json (menu.yml da permiso de escritura). Sin token (pruebas, local) no hace nada."""
    import os
    if not os.getenv("GITHUB_ACTIONS"): return
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
    sh('git config user.name "analista-bot" && git config user.email "bot@users.noreply.github.com"')
    sh(f"git add {FILE} && git commit -q -m '{msg}'")
    for _ in range(4):
        if sh("git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main").returncode == 0: return
        sh("git rebase --abort"); time.sleep(5)
    print("simuladas: no se pudo subir (queda en memoria y se reintenta)")


def record(combo, stake=1.0):
    """Apunta una combinada de chat.py ({"titulo", "legs": [{code, home, away, start, txt, odds, p}], "odds", "p"})."""
    bets = load()
    bid = f"{pd.Timestamp.now(tz='UTC'):%Y%m%d%H%M%S}"
    bets[bid] = {"id": bid, "creada": str(pd.Timestamp.now(tz="UTC")), "titulo": combo.get("titulo", "Combinada"),
                 "legs": combo["legs"], "odds": combo.get("odds"), "p": combo.get("p"), "stake": stake,
                 "estado": "pendiente", "avisada": False, "rev": 1}
    save(bets, f"chat: simulada {bid}")
    return bets[bid]


# ── liquidación ─────────────────────────────────────────────────────────────────────────────────────────────────
_GAMES = {}


def games(code):
    """Partidos guardados de ESPN de una liga (en caché por fecha de modificación del archivo)."""
    p = ES.path(code)
    if not p.exists(): return []
    mt = p.stat().st_mtime
    if _GAMES.get(code, (None,))[0] != mt: _GAMES[code] = (mt, ES.load(code))
    return _GAMES[code][1]


def find_game(leg):
    after = pd.Timestamp(leg.get("start") or leg.get("creada")) - pd.Timedelta(hours=6)
    codes = [leg["code"]] if leg.get("code") in ES.LEAGUES else list(ES.LEAGUES)
    for code in codes:
        for g in games(code):
            if g["home"] == leg["home"] and g["away"] == leg["away"] and pd.Timestamp(g["date"]) >= after: return g
    return None


def judge(leg, g):
    """(acierto True/False/'nula', porqué) de una pierna con el partido de ESPN."""
    h, a = (g.get("team_stats") or {}).get("h") or {}, (g.get("team_stats") or {}).get("a") or {}
    if m := DC_RX.match(leg["txt"]):
        ok = g["hs"] >= g["as"] if m.group(2) == "1X" else g["as"] >= g["hs"]
        return ok, f"acabó {g['hs']:g}-{g['as']:g}"
    m = LEG_RX.match(leg["txt"])
    if not m: return "nula", "pierna no reconocida"
    side, line, stat = m.group(1), float(m.group(2)), m.group(3)
    if stat == "goles": v = g["hs"] + g["as"]
    else:
        if not (ES.has_stats(h) and ES.has_stats(a)): return None, "ESPN aún no tiene las estadísticas"
        k = "wonCorners" if stat == "córners" else None
        v = (h[k] + a[k]) if k else sum(d.get("yellowCards", 0) + d.get("redCards", 0) for d in (h, a))
    ok = v > line if side == "Más" else v < line
    return ok, f"hubo {v:g} {stat}"


def settle():
    """Liquida las simuladas pendientes; devuelve los mensajes (HTML) de las que acaban de terminar."""
    bets, now, out, changed = load(), pd.Timestamp.now(tz="UTC"), [], False
    for b in bets.values():
        if b["estado"] != "pendiente": continue
        for leg in b["legs"]:
            if leg.get("ok") is not None: continue
            g = find_game({**leg, "creada": b["creada"]})
            if g:
                ok, why = judge(leg, g)
                if ok is not None: leg["ok"], leg["por"] = ok, why; changed = True
            elif now - pd.Timestamp(leg.get("start") or b["creada"]) > GIVE_UP:
                leg["ok"], leg["por"] = "nula", "sin datos del partido"; changed = True
        if all(l.get("ok") is not None for l in b["legs"]):
            res = [l["ok"] for l in b["legs"]]
            b["estado"] = "perdida" if False in res else "nula" if all(r == "nula" for r in res) else "ganada"
            b["rev"] = b.get("rev", 1) + 1; changed = True
            out.append(card(b)); b["avisada"] = True
    if changed: save(bets, "chat: simuladas liquidadas")
    return out


def card(b):
    legs = [(f"{l['home']} – {l['away']}", f"{l['txt']}" + (f" @ {l['odds']:.2f}" if l.get("odds") else "")
             + (f" ({l['por']})" if l.get("por") else ""), l.get("ok")) for l in b["legs"]]
    res = {"ganada": True, "perdida": False, "nula": "nula"}.get(b["estado"])
    head = {"ganada": "🎉 ¡Simulada GANADA!", "perdida": "Simulada perdida", "nula": "Simulada nula"}.get(b["estado"], "Simulada en juego")
    fail = [l for l in b["legs"] if l.get("ok") is False]
    why = (f"\nFalló{'ron' if len(fail) > 1 else ''}: " + "; ".join(f"{l['txt']} en {l['home']} – {l['away']} ({l['por']})" for l in fail)) if fail else ""
    return "\n".join([f"🧾 <b>{head}</b> · {A.esc(b['titulo'])}", *B.boleto(None, legs, b.get("odds"), b.get("stake"), res)]) + A.esc(why)


def summary():
    """Estado de las simuladas del chat (para «¿cómo van mis apuestas?»)."""
    bets = sorted(load().values(), key=lambda b: b["id"])[-10:]
    if not bets: return "Aún no has simulado ninguna apuesta desde el chat."
    done = [b for b in bets if b["estado"] in ("ganada", "perdida")]
    bal = sum((b["stake"] * (b["odds"] or 0) - b["stake"]) if b["estado"] == "ganada" else -b["stake"] for b in done)
    lines = [f"🧾 <b>Tus simuladas</b> · {len(done)} terminadas, {sum(b['estado'] == 'ganada' for b in done)} ganadas · balance {bal:+.2f} €"]
    for b in bets:
        n_ok = sum(l.get("ok") is True for l in b["legs"])
        icon = {"ganada": "✅", "perdida": "❌", "nula": "➖"}.get(b["estado"], "⏳")
        lines.append(f"{icon} {A.esc(b['titulo'])} · {n_ok}/{len(b['legs'])} piernas" + (f" · cuota {b['odds']:.2f}" if b.get("odds") else ""))
    return "\n".join(lines)
