"""Apuestas simuladas desde el chat de Telegram («simula la apuesta y avísame»): se apuntan en data/chat_simuladas.json
(se sube al repo para que no se pierdan cuando menu.py se releva) y se liquidan solas con las estadísticas de ESPN
(data/espn_soccer, que actualiza el experto datos_futbol_jugadores). Al acabar, menu.py manda el boleto con ✅/❌ en cada
pierna y el porqué (lo que pasó de verdad frente a la línea).

Piernas que entiende (las que escribe chat.py): «Más/Menos de X córners|tarjetas|goles» (totales del partido) y
«<equipo> o empate (1X|X2)» de fútbol; y «Gana X» / «Empate» de NBA, NHL, NFL, MLB, esports y FC 27
(multideporte.py)."""
import json, re, subprocess, time

import pandas as pd

import analyst as A
import boleto as B
import espn_soccer as ES

FILE = A.DATA / "chat_simuladas.json"
_MEM = {}  # id → apuesta: copia en memoria por si un git reset de menu.py borra un cambio aún sin subir
LEG_RX = re.compile(r"^(Más|Menos) de ([\d.]+) (córners|tarjetas|goles)(?: de (.+))?$")  # «… de <equipo>»: solo ese equipo
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
    sig = lambda legs: sorted((l.get("home"), l.get("away"), l.get("txt"), l.get("jugador")) for l in legs)
    dup = next((b for b in bets.values() if b["estado"] == "pendiente" and sig(b["legs"]) == sig(combo["legs"])), None)
    if dup: return {**dup, "duplicada": True}  # la misma combinada ya apuntada (p. ej. un «sí» repetido)
    bid = f"{pd.Timestamp.now(tz='UTC'):%Y%m%d%H%M%S%f}"  # con microsegundos: dos en el mismo segundo no se pisan
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


PIX = {"goles": 5, "asistencias": 6, "tiros": 7, "tiros a puerta": 8, "paradas": 9, "faltas": 11, "faltas recibidas": 15}
PLAYER_RX = re.compile(r"^(Más|Menos) de ([\d.]+) (goles|asistencias|tiros a puerta|tiros|paradas|faltas recibidas|faltas|tarjetas)$")


def judge_player(leg, g):
    """Pierna de jugador («Más de 0.5 goles» de Mbappé) con sus estadísticas en el partido de ESPN."""
    import unicodedata
    n = lambda s: unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    m = PLAYER_RX.match(leg["txt"])
    if not m: return "nula", "pierna no reconocida"
    side, line, stat = m.group(1), float(m.group(2)), m.group(3)
    pl = next((p for p in g.get("players") or [] if n(p[1]) == n(leg["jugador"])), None)
    if pl is None: return "nula", f"{leg['jugador']} no jugó"
    if stat == "tarjetas": v = pl[12] + pl[13]
    elif PIX[stat] < len(pl): v = pl[PIX[stat]]
    else: return None, "ESPN aún sin ese dato"
    ok = v > line if side == "Más" else v < line
    return ok, f"{leg['jugador']}: {v:g} {stat}"


def judge(leg, g):
    """(acierto True/False/'nula', porqué) de una pierna con el partido de ESPN."""
    if leg.get("jugador"): return judge_player(leg, g)
    h, a = (g.get("team_stats") or {}).get("h") or {}, (g.get("team_stats") or {}).get("a") or {}
    if leg["txt"] == "Empate" or leg["txt"].startswith("Gana "):  # 1X2 de fútbol leído de un boleto
        w = leg["txt"][5:]
        ok = g["hs"] == g["as"] if leg["txt"] == "Empate" else (g["hs"] > g["as"]) if w == g["home"] else (g["as"] > g["hs"])
        return ok, f"acabó {g['hs']:g}-{g['as']:g}"
    if m := DC_RX.match(leg["txt"]):
        ok = g["hs"] >= g["as"] if m.group(2) == "1X" else g["as"] >= g["hs"]
        return ok, f"acabó {g['hs']:g}-{g['as']:g}"
    m = LEG_RX.match(leg["txt"])
    if not m: return "nula", "pierna no reconocida"
    side, line, stat, team = m.group(1), float(m.group(2)), m.group(3), m.group(4)
    sides = (h, a) if not team else (h,) if team == g["home"] else (a,) if team == g["away"] else None
    if sides is None: return "nula", f"equipo {team} no está en el partido"
    if stat == "goles": v = (g["hs"] + g["as"]) if not team else g["hs"] if team == g["home"] else g["as"]
    else:
        if not (ES.has_stats(h) and ES.has_stats(a)): return None, "ESPN aún no tiene las estadísticas"
        k = "wonCorners" if stat == "córners" else None
        v = sum(d[k] for d in sides) if k else sum(d.get("yellowCards", 0) + d.get("redCards", 0) for d in sides)
    ok = v > line if side == "Más" else v < line
    return ok, f"hubo {v:g} {stat}" + (f" de {team}" if team else "")


def final_game(leg, b, now, boards):
    """Partido ya terminado según el marcador de ESPN en directo (los datos guardados tardan horas en llegar):
    así la apuesta se liquida en cuanto acaba. Solo mira partidos de los últimos 2 días que ya deberían haber acabado."""
    t = pd.Timestamp(leg.get("start") or b["creada"])
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    if not (pd.Timedelta(minutes=100) <= now - t <= pd.Timedelta(days=2)) and leg.get("start"): return None
    if not leg.get("start") and now - t > pd.Timedelta(days=2): return None
    try: g = live_game({**leg, "ahora": t}, boards)
    except Exception as e: print("final", repr(e)[:80]); return None
    return g if g and g["estado"] == "post" else None


def settle(now=None):
    """Liquida las simuladas pendientes; devuelve los mensajes (HTML) de las que acaban de terminar."""
    bets, now, out, changed, boards = load(), now or pd.Timestamp.now(tz="UTC"), [], False, {}
    for b in bets.values():
        if b["estado"] != "pendiente": continue
        for leg in b["legs"]:
            if leg.get("ok") is not None: continue
            if leg.get("src"):  # NBA/NHL/esports/FC 27: resultados que guarda el bot (multideporte.py)
                import multideporte
                ok, why = multideporte.judge(leg)
                if ok is not None: leg["ok"], leg["por"] = ok, why; changed = True
                elif now - pd.Timestamp(leg["start"]) > GIVE_UP: leg["ok"], leg["por"] = "nula", "sin resultado"; changed = True
                continue
            g = find_game({**leg, "creada": b["creada"]}) or final_game(leg, b, now, boards)
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


# ── en directo ──────────────────────────────────────────────────────────────────────────────────────────────────
def _scoreboard(slug, day, sport="soccer"):
    """[(evento ESPN, estado 'pre'|'in'|'post', reloj, local, visitante, goles local, goles visitante)] de un día."""
    out = []
    data = ES.get(f"{slug}/scoreboard", dates=f"{day:%Y%m%d}") if sport == "soccer" else \
        __import__("requests").get(f"https://site.api.espn.com/apis/site/v2/sports/{slug}/scoreboard",
                                   params={"dates": f"{day:%Y%m%d}"}, headers=A.UA, timeout=30).json()
    for e in data.get("events", []):
        c = (e.get("competitions") or [{}])[0]
        st = (c.get("status") or e.get("status") or {}).get("type") or {}
        t = {x.get("homeAway"): x for x in c.get("competitors", [])}
        if "home" in t and "away" in t:
            out.append((e["id"], st.get("state"), st.get("shortDetail") or "", t["home"]["team"]["displayName"],
                        t["away"]["team"]["displayName"], ES._f(t["home"].get("score")), ES._f(t["away"].get("score"))))
    return out


def live_game(leg, boards):
    """Partido de ESPN de una pierna empezada: {home, away, hs, as, estado, reloj, team_stats, players} o None."""
    day = pd.Timestamp(leg.get("start") or leg.get("ahora") or pd.Timestamp.now(tz="UTC"))  # boletos de foto: sin hora
    codes = [leg["code"]] if leg.get("code") in ES.LEAGUES else list(ES.LEAGUES)
    for code in codes:
        key = (code, f"{day:%Y%m%d}")
        if key not in boards:
            try: boards[key] = _scoreboard(ES.LEAGUES[code], day)
            except Exception as e: print("directo", key, repr(e)[:80]); boards[key] = []
        for ev, state, clock, h, a, hs, as_ in boards[key]:
            if h == leg["home"] and a == leg["away"] and state in ("in", "post"):
                try: det = ES.compact(ES.get(f"{ES.LEAGUES[code]}/summary", event=ev))
                except Exception: det = {}
                return {**det, "home": h, "away": a, "hs": hs, "as": as_, "estado": state, "reloj": clock}
    return None


def live_other(leg, boards):
    """NBA/NHL/NFL/MLB (piernas «src: espn:<liga>»): (icono, texto) con el marcador de ESPN en juego, o None."""
    import multisport
    lg = leg["src"].split(":")[1]
    if lg not in multisport.SPORTS: return None
    slug, t = multisport.SPORTS[lg][0], pd.Timestamp(leg["start"])
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    for day in dict.fromkeys(x.tz_localize(None).normalize() for x in (t.tz_convert("America/New_York"), t.tz_convert("UTC"))):  # ESPN fecha en hora de EE. UU.
        key = (lg, f"{day:%Y%m%d}")
        if key not in boards:
            try: boards[key] = _scoreboard(slug, day, sport=lg)
            except Exception as e: print("directo", key, repr(e)[:80]); boards[key] = []
        for ev, state, clock, h, a, hs, as_ in boards[key]:
            if h == leg["home"] and a == leg["away"] and state in ("in", "post"):
                ok = hs > as_ if leg["side"] == "H" else as_ > hs if leg["side"] == "A" else hs == as_
                if state == "post": return ("✅" if ok else "❌"), f"terminado: acabó {hs:g}-{as_:g}"
                return ("🟢" if ok else "🔴"), f"va {hs:g}-{as_:g}" + (f" ({clock})" if clock else "")
    return None


def live_leg(leg, g):
    """(icono, texto) de una pierna con el partido en juego: ✅/❌ si ya está decidida, ⏳ si aún no."""
    if g["estado"] == "post":
        ok, why = judge(leg, g)
        return {True: "✅", False: "❌"}.get(ok, "⏳"), f"terminado: {why}"
    if leg.get("jugador"):
        ok, why = judge(leg, g)
        if ok is True and leg["txt"].startswith("Más") or ok is False and leg["txt"].startswith("Menos"):
            return ("✅" if ok else "❌"), f"ya {why}"
        return "⏳", f"va: {why}"
    score = f"{g['hs']:g}-{g['as']:g}"
    if leg["txt"] == "Empate" or leg["txt"].startswith("Gana ") or DC_RX.match(leg["txt"]):
        ok, _ = judge(leg, g)
        return ("🟢" if ok else "🔴"), f"va {score}" + (" (de momento la ganas)" if ok else " (de momento la pierdes)")
    m = LEG_RX.match(leg["txt"])
    if not m: return "⏳", f"va {score}"
    ok, why = judge(leg, g)
    if ok is None: return "⏳", f"va {score}, ESPN aún sin estadísticas"
    side = m.group(1)
    if side == "Más" and ok: return "✅", why.replace("hubo", "ya lleva")
    if side == "Menos" and not ok: return "❌", why.replace("hubo", "ya lleva")
    return "⏳", why.replace("hubo", "lleva")


def _norm(x):
    import unicodedata
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()


def matches(b, filtro):
    """¿La apuesta es de la que habla el texto (nombra alguno de sus equipos)?"""
    f = _norm(filtro)
    return any(n and _norm(n) in f for l in b["legs"] for n in (l.get("home"), l.get("away")))


def live(now=None, boards=None, filtro=None):
    """«¿Cómo van mis apuestas?» con partidos en juego: marcador y estadísticas de ESPN ahora mismo, pierna a pierna.
    `filtro`: texto que nombra los equipos de una apuesta («cómo va esta» citando el boleto) → solo esa, empezada o no.
    None si no hay ninguna que enseñar. Las repetidas (mismo boleto apuntado dos veces) salen una sola vez."""
    now, boards = now or pd.Timestamp.now(tz="UTC"), {} if boards is None else boards
    started = lambda l: l.get("ok") is None and (not l.get("start") or pd.Timestamp(l["start"]) <= now)
    pend = [b for b in load().values() if b["estado"] == "pendiente"]
    sel = [b for b in pend if matches(b, filtro)] if filtro else []
    pend = sel or [b for b in pend if any(started(l) for l in b["legs"])
                   and (not all(not l.get("start") for l in b["legs"]) or now - pd.Timestamp(b["creada"]) < pd.Timedelta(days=2))]
    seen, uniq = set(), []
    for b in sorted(pend, key=lambda b: b["id"], reverse=True):
        k = tuple(sorted((l.get("home"), l.get("away"), l.get("txt"), l.get("jugador")) for l in b["legs"]))
        if k not in seen: seen.add(k); uniq.append(b)
    pend = uniq[::-1]
    if not pend: return None
    out = ["📡 <b>Tus apuestas en juego</b>"]
    for b in sorted(pend, key=lambda b: b["id"]):
        out.append(f"\n🧾 <b>{A.esc(b['titulo'])}</b>" + (f" · cuota {b['odds']:.2f}" if b.get("odds") else ""))
        for l in b["legs"]:
            name = f"{A.esc(l['home'])} – {A.esc(l['away'])}: {A.esc(l['txt'])}" + (f" ({A.esc(l['jugador'])})" if l.get("jugador") else "")
            if l.get("ok") is not None:
                icon = {True: "✅", False: "❌"}.get(l["ok"], "➖")
                out.append(f"{icon} {name} · {A.esc(l.get('por', ''))}"); continue
            if l.get("start") and pd.Timestamp(l["start"]) > now:
                out.append(f"🕒 {name} · aún no ha empezado"); continue
            if l.get("src"):
                r = live_other(l, boards) if l["src"].startswith("espn:") else None
                out.append(f"{r[0]} {name} · {A.esc(r[1])}" if r else f"⏳ {name} · sin directo"); continue
            g = live_game({**l, "ahora": now}, boards)
            if not g: out.append(f"{'⏳' if l.get('start') else '🕒'} {name} · " + ("sin directo" if l.get("start") else "ESPN no lo da en juego ahora")); continue
            icon, why = live_leg(l, g)
            out.append(f"{icon} {name} · {A.esc(why)}" + (f" ({A.esc(g['reloj'])})" if g["estado"] == "in" and g["reloj"] else ""))
    return "\n".join(out)


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


def compare(filtro=None):
    """Simuladas pendientes ordenadas por posibilidades: producto de las probabilidades del bot (con piernas del mismo
    partido es solo orientativo: van ligadas) y la pierna más floja, que es la que más la pone en riesgo."""
    import math
    pend = [b for b in load().values() if b["estado"] == "pendiente"]
    if filtro:
        f = filtro.lower()
        pend = [b for b in pend if any(f in f"{l.get('home', '')} {l.get('away', '')}".lower() for l in b["legs"])] or pend
    if not pend: return "No tienes apuestas simuladas pendientes."
    rows = []
    for b in pend:
        ps = [l["p"] for l in b["legs"] if l.get("p") is not None]
        prod = math.prod(ps) if ps else None
        weak = min((l for l in b["legs"] if l.get("p") is not None), key=lambda l: l["p"], default=None)
        rows.append((prod or 0, b, weak, len(ps)))
    rows.sort(key=lambda r: -r[0])
    out = ["📊 <b>Tus simuladas pendientes</b> (de más a menos posibilidades)"]
    for i, (prod, b, weak, n) in enumerate(rows, 1):
        out.append(f"{i}. {A.esc(b['titulo'])} · {len(b['legs'])} piernas" + (f" · cuota {b['odds']:.2f}" if b.get("odds") else ""))
        if weak: out.append(f"   La más floja: {A.esc((weak.get('jugador') + ': ') if weak.get('jugador') else '')}{A.esc(weak['txt'])} ({weak['p']:.0%})")
    out.append("ℹ️ Con piernas del mismo partido el orden es orientativo: van ligadas. Menos piernas y más seguras = más posibilidades.")
    return "\n".join(out)
