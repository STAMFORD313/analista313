"""Conversación con el bot por Telegram, sin IA (0 tokens): entiende frases sueltas en español y contesta con los datos.

  «estadísticas de Mbappé» / «stats Real Madrid»          → medias de sus últimos partidos
  «Mbappé más de 1.5 tiros a puerta a 1.9»                 → probabilidad, cuota justa y opinión (🟢/🟡/🔴)
  «LeBron más de 25.5 puntos y Curry 4+ triples @3.1»      → combinada: cada pierna y el total
  «Real Madrid más de 5.5 córners» · «Madrid Barça más de 9.5 córners» (total del partido)

Datos: data/espn_soccer (jugadores y equipos de 29 competiciones), data/nba y data/nhl/box.jsonl (jugadores ESPN).
Probabilidad: binomial negativa con la media y la varianza de sus últimos partidos (más peso a los recientes); con dos
equipos, media del partido = (a favor de A + en contra de B)/2 + (a favor de B + en contra de A)/2. Lo usa menu.py."""
import json, math, re, time, unicodedata

import numpy as np
from scipy import stats as SS

import analyst as A
import espn_soccer as ES

N_LAST, DECAY = 20, 0.93
# estadística: (palabras que la nombran, etiqueta) · el orden importa: «tiros a puerta» antes que «tiros»
STATS = [("tiros_puerta", r"tiros? a puerta|a puerta|shots? on target|\bsot\b", "tiros a puerta"),
         ("corners", r"c[oó]rners?|saques? de esquina", "córners"),
         ("tarjetas", r"tarjetas?|amarillas?|cards?", "tarjetas"),
         ("faltas", r"faltas?|fouls?", "faltas"),
         ("fueras", r"fueras? de juego|offsides?", "fueras de juego"),
         ("paradas", r"paradas?|saves?", "paradas"),
         ("asistencias", r"asistencias?|assists?|\bast\b", "asistencias"),
         ("rebotes", r"rebotes?|rebounds?|\breb\b", "rebotes"),
         ("triples", r"triples?|3 ?pt|threes?", "triples"),
         ("robos", r"robos?|steals?", "robos"),
         ("tapones", r"tapones?|blocks?", "tapones"),
         ("puntos", r"puntos?|points?|\bpts\b", "puntos"),
         ("tiros", r"tiros?|disparos?|remates?|shots?", "tiros"),
         ("goles", r"goles|gol\b|marca|anota|goals?|scorer", "goles")]
LABEL = {k: l for k, _, l in STATS}
PLAYER_STATS = {"futbol": ["goles", "asistencias", "tiros", "tiros_puerta", "faltas", "tarjetas", "fueras", "paradas"],
                "nba": ["puntos", "rebotes", "asistencias", "triples", "robos", "tapones"], "nhl": ["goles", "tiros"]}
TEAM_STATS = {"futbol": ["goles", "corners", "tiros", "tiros_puerta", "faltas", "tarjetas", "fueras"], "nba": ["puntos"], "nhl": ["goles"]}
HELP = ("💬 Escríbeme, por ejemplo:\n• <i>estadísticas de Mbappé</i>\n• <i>Mbappé más de 1.5 tiros a puerta a 1.9</i>\n"
        "• <i>LeBron más de 25.5 puntos y Curry 4+ triples @3.1</i>\n• <i>Real Madrid más de 5.5 córners</i>\n"
        "• <i>Madrid Barcelona más de 9.5 córners</i> (total del partido)")
_DB = {"t": 0}
# apodos de uso común → nombre de ESPN (solo si el nombre completo no está ya en la frase)
ALIAS = {"madrid": "real madrid", "real": "real madrid", "barca": "barcelona", "atleti": "atletico madrid", "atletico": "atletico madrid",
         "inter": "internazionale", "milan": "ac milan", "psg": "paris saint germain", "united": "manchester united",
         "city": "manchester city", "bayern": "bayern munich", "spurs": "tottenham hotspur", "juve": "juventus", "roma": "as roma",
         "sporting": "sporting cp", "porto": "fc porto", "dortmund": "borussia dortmund", "leverkusen": "bayer leverkusen",
         "betis": "real betis", "sociedad": "real sociedad", "lebron": "lebron james", "giannis": "giannis antetokounmpo"}


def low(s): return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def names_only(text):
    """El texto sin estadísticas, líneas, cuotas ni palabras de relleno (para que «puerta» o «santos» no nombren a nadie)."""
    t = low(text)
    for _, rx, _ in STATS: t = re.sub(rx, " ", t)
    t = re.sub(r"\d+(?:[.,]\d+)?\+?|@|\b(?:estadisticas?|stats|datos|mas de|menos de|al menos|over|under|cuota|de|del|la|el|los|las|partido|total|en|contra|a|y)\b", " ", t)
    return t


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


# ── datos: {clave: {"name", "sport", "kind", "team", "games": [(fecha, {stat: valor}, {stat: valor en contra})]}} ──
def _add(db, key, name, sport, kind, date, vals, against=None, team=None):
    e = db.setdefault(key, {"name": name, "sport": sport, "kind": kind, "team": team, "games": []})
    e["games"].append((str(date)[:10], vals, against or {})); e["team"] = team or e["team"]


def build():
    db = {}
    for code in ES.LEAGUES:
        for g in ES.load(code):
            ts = g.get("team_stats") or {}
            for sd, team, gf, ga in (("h", g["home"], g["hs"], g["as"]), ("a", g["away"], g["as"], g["hs"])):
                o = "a" if sd == "h" else "h"
                tv = lambda d: {"goles": None, "corners": d.get("wonCorners"), "tiros": d.get("totalShots"), "tiros_puerta": d.get("shotsOnTarget"),
                                "faltas": d.get("foulsCommitted"), "tarjetas": (d.get("yellowCards") or 0) + (d.get("redCards") or 0) if d else None,
                                "fueras": d.get("offsides")}
                f, a = tv(ts.get(sd) or {}), tv(ts.get(o) or {})
                f["goles"], a["goles"] = gf, ga
                _add(db, f"futbol|t|{team}", team, "futbol", "equipo", g["date"], f, a)
            for p in g["players"]:
                team = g["home"] if p[2] == "h" else g["away"]
                v = dict(zip(["goles", "asistencias", "tiros", "tiros_puerta", "paradas", "_rec", "faltas", "_am", "_ro", "fueras"], p[5:15]))
                v["tarjetas"] = v.pop("_am") + v.pop("_ro"); v.pop("_rec"); v["_titular"] = bool(p[3])
                _add(db, f"futbol|p|{p[0]}", p[1], "futbol", "jugador", g["date"], v, team=team)
    for lg in ("nba", "nhl"):
        f = A.DATA / lg / "box.jsonl"
        if not f.exists(): continue
        for line in f.open(encoding="utf-8"):
            g = json.loads(line)
            for t in g.get("teams") or []:
                me = t["team"]; home = me == g["home"]
                gf, ga = (g["hs"], g["as"]) if home else (g["as"], g["hs"])
                key = "puntos" if lg == "nba" else "goles"
                _add(db, f"{lg}|t|{me}", me, lg, "equipo", g["date"], {key: gf}, {key: ga})
                for p in t["players"]:
                    if p[6] or p[2] <= 0: continue
                    x = p + [0] * 14
                    v = ({"puntos": x[3], "rebotes": x[7], "asistencias": x[8], "triples": x[9], "robos": x[10], "tapones": x[11]}
                         if lg == "nba" else {"goles": x[3], "tiros": x[13]})
                    v["_min"] = p[2]
                    _add(db, f"{lg}|p|{p[0]}", p[1], lg, "jugador", g["date"], v, team=me)
    for e in db.values(): e["games"].sort(key=lambda x: x[0]); e["norm"] = norm(e["name"])
    return db


def db():
    if time.time() - _DB["t"] > 3600 or "db" not in _DB: _DB["db"], _DB["t"] = build(), time.time()
    return _DB["db"]


def find(text, D=None, n=2, kind=None):
    """Entidades nombradas en el texto (nombre completo, o nombre/apellido de ≥ 4 letras), por orden de aparición.
    kind="equipo" para estadísticas que solo tienen los equipos (córners)."""
    D = D or db(); t = f" {norm(names_only(text))} "
    for a, full in ALIAS.items():
        if f" {a} " in t and f" {full} " not in t and not any(f" {x} {a} " in t or f" {a} {x} " in t for x in full.split() if x != a):
            t = t.replace(f" {a} ", f" {full} ", 1)
    hits = []
    for key, e in D.items():
        nm = e["norm"]
        if not nm or (kind and e["kind"] != kind): continue
        if f" {nm} " in t: score, pos, ln = (3 + len(nm) / 100 if " " in nm else 2), t.find(f" {nm} "), len(nm)
        else:
            words = [w for w in nm.split() if len(w) >= 4 and w not in ("real", "club", "city", "united", "sporting", "atletico", "athletic", "deportivo")]
            w = next((w for w in reversed(words) if f" {w} " in t), None)
            if not w: continue
            score, pos, ln = (2 if e["kind"] == "jugador" and w in (nm.split()[0], nm.split()[-1]) else 1), t.find(f" {w} "), len(w)
        recent = e["games"][-1][0] if e["games"] else ""
        hits.append((score, recent, len(e["games"]), pos, key, ln))
    hits.sort(key=lambda h: (-h[0], h[1] < "2025", -h[2]))  # mejor coincidencia, con datos recientes, con más partidos
    out, used = [], []
    for h in hits:
        if any(h[3] < b and a < h[3] + h[5] for a, b in used): continue  # las mismas palabras no nombran a dos
        out.append(h[4]); used.append((h[3], h[3] + h[5]))
        if len(out) == n: break
    return sorted(out, key=lambda k: next(h[3] for h in hits if h[4] == k))


# ── modelo ──────────────────────────────────────────────────────────────────────────────────────────────────
def series(e, stat, against=False):
    vals = [g[2 if against else 1].get(stat) for g in e["games"]]
    if e["kind"] == "jugador" and e["sport"] == "futbol":  # si suele ser titular, solo sus partidos de titular
        st = [g[1].get("_titular") for g in e["games"][-N_LAST:]]
        if sum(st) >= 0.6 * len(st): vals = [v for v, g in zip(vals, e["games"]) if g[1].get("_titular")]
    return [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))][-N_LAST:]


def moments(x):
    w = DECAY ** np.arange(len(x))[::-1]; x = np.asarray(x, float)
    mu = float((w * x).sum() / w.sum()); var = float((w * (x - mu) ** 2).sum() / w.sum()) * len(x) / max(len(x) - 1, 1)
    return mu, var


def p_over(mu, var, line):
    """P(X > línea) con binomial negativa (Poisson si no hay sobredispersión)."""
    k = math.floor(line)
    if mu <= 0: return 0.0
    if var <= mu * 1.05: return float(SS.poisson(mu).sf(k))
    r = mu * mu / (var - mu); return float(SS.nbinom(r, r / (r + mu)).sf(k))


# ── lectura de la frase ─────────────────────────────────────────────────────────────────────────────────────
NUM = r"(\d+(?:[.,]\d+)?)"


def _f(s): return float(s.replace(",", "."))


def parse_leg(seg):
    """{'stat', 'line', 'over'} de un trozo: «más de 1.5 tiros», «4+ triples», «menos de 9.5 córners», «marca»."""
    s = re.sub(r"(\d),(\d)", r"\1.\2", low(seg))
    stat = next((k for k, rx, _ in STATS if re.search(rx, s)), None)
    over, line = True, None
    if m := re.search(rf"(?:menos de|under|bajo)\s*{NUM}", s): over, line = False, _f(m.group(1))
    elif m := re.search(rf"(?:mas de|over|sobre|\+)\s*{NUM}", s): line = _f(m.group(1))
    elif m := re.search(rf"(?:al menos|minimo)\s*{NUM}", s): line = _f(m.group(1)) - 0.5
    elif m := re.search(rf"{NUM}\s*\+", s): line = _f(m.group(1)) - 0.5
    if stat == "goles" and line is None and re.search(r"marca|anota|gol\b|scorer", s): line = 0.5
    return {"stat": stat, "line": line, "over": over}


def odds_in(text):
    t = text.replace(",", ".").lower()
    return [float(x) for x in re.findall(r"(?:@|cuota(?: de)?|a cuota|\ba)\s*(\d+\.\d+)", t) if 1.01 <= float(x) <= 100]


def split_legs(text):
    t = re.sub(r"(?:@|cuota(?: de)?)\s*\d+[.,]\d+", " ", text, flags=re.I)
    t = re.sub(r"\ba\s+\d+[.,]\d+\s*$", " ", t)
    return [p for p in re.split(r"\n|;|\s+y\s+|\s+e\s+|,\s+(?=[A-ZÁÉÍÓÚ])", t) if p.strip()]


# ── respuestas ──────────────────────────────────────────────────────────────────────────────────────────────
def _who(e):
    sp = {"futbol": "⚽", "nba": "🏀", "nhl": "🏒"}[e["sport"]]
    return f"{sp} <b>{A.esc(e['name'])}</b>" + (f" ({A.esc(e['team'])})" if e.get("team") and e["kind"] == "jugador" else "")


def stats_card(key, D=None):
    D = D or db(); e = D[key]
    names = (PLAYER_STATS if e["kind"] == "jugador" else TEAM_STATS)[e["sport"]]
    lines = [_who(e), f"Últimos {min(N_LAST, len(e['games']))} partidos (hasta {e['games'][-1][0]}):"]
    for s in names:
        x = series(e, s)
        if not x or (e["kind"] == "jugador" and s == "paradas" and sum(x) == 0): continue
        l10 = x[-10:]
        txt = f"• {LABEL[s]}: {np.mean(x):.1f} de media · últimos 5: {' '.join(f'{v:g}' for v in x[-5:])}"
        if e["kind"] == "equipo":
            y = series(e, s, against=True)
            if y: txt += f" · en contra {np.mean(y):.1f}"
        lines.append(txt if len(l10) else "")
    return "\n".join(l for l in lines if l)


def leg_prob(keys, leg, D):
    """(probabilidad, texto) de una pierna con 1 entidad (jugador/equipo) o 2 equipos (total del partido)."""
    es = [D[k] for k in keys]
    stat, line, over = leg["stat"], leg["line"], leg["over"]
    if len(es) == 2 and all(e["kind"] == "equipo" for e in es):
        (a, b) = es; parts = [(series(a, stat), series(b, stat, True)), (series(b, stat), series(a, stat, True))]
        if any(len(u) < 5 or len(v) < 5 for u, v in parts): return None, f"pocos datos de {LABEL[stat]} de {A.esc(a['name'])} o {A.esc(b['name'])}"
        mus = [(moments(u)[0] + moments(v)[0]) / 2 for u, v in parts]
        vr = [(moments(u)[1] + moments(v)[1]) / 2 for u, v in parts]
        mu, var = sum(mus), sum(vr); who = f"{A.esc(a['name'])} – {A.esc(b['name'])}"
        hist = None
    else:
        e = es[0]; x = series(e, stat)
        if len(x) < 5: return None, f"pocos datos de {LABEL[stat]} de {A.esc(e['name'])}"
        mu, var = moments(x); who = A.esc(e["name"])
        hist = sum(v > line for v in x[-10:]) if over else sum(v < line for v in x[-10:]), len(x[-10:])
    p = p_over(mu, var, line); p = p if over else 1 - p
    txt = f"{who} {'más' if over else 'menos'} de {line:g} {LABEL[stat]}: media {mu:.1f} → <b>{p:.0%}</b>"
    if hist: txt += f" (lo cumplió en {hist[0]} de sus {hist[1]} últimos)"
    return p, txt


def verdict(p, odds):
    if not odds: return f"Cuota justa {1 / max(p, 1e-6):.2f}: apuéstala solo si te la pagan más alta."
    ev = p * odds - 1
    icon = "🟢 Me gusta" if ev >= 0.05 else "🟡 Justa" if ev >= -0.02 else "🔴 No la cogería"
    return f"{icon}: a {odds:.2f} ganas {ev:+.0%} por € a la larga (cuota justa {1 / max(p, 1e-6):.2f})."


def answer(text):
    """Respuesta HTML a un mensaje libre del dueño."""
    t = text.strip()
    if not t or re.match(r"^/?(ayuda|help|hola|que puedes)", norm(t)): return HELP
    D = db()
    legs, odds = split_legs(t), odds_in(t)
    parsed = []
    for seg in legs:
        leg = parse_leg(seg)
        keys = find(seg, D, kind="equipo" if leg["stat"] == "corners" else None)
        if keys and leg["stat"] and leg["line"] is not None: parsed.append((keys, leg))
    if not parsed:
        keys = find(t, D, n=1)
        if keys: return stats_card(keys[0], D)
        return "No te he entendido 🤔\n\n" + HELP
    out, ps = [], []
    for keys, leg in parsed:
        if len(keys) == 2 and not all(D[k]["kind"] == "equipo" for k in keys): keys = keys[:1]
        p, txt = leg_prob(keys, leg, D)
        out.append(("• " if len(parsed) > 1 else "") + txt); ps.append(p)
    if any(p is None for p in ps): return "\n".join(out)
    if len(parsed) == 1:
        return out[0] + "\n" + verdict(ps[0], odds[0] if odds else None)
    p = float(np.prod(ps))
    o = (odds[0] if len(odds) == 1 else float(np.prod(odds)) if len(odds) == len(ps) else None)
    return ("🎯 <b>Combinada</b>\n" + "\n".join(out) + f"\nTodas a la vez: <b>{p:.0%}</b> (si son del mismo partido la correlación lo mueve un poco)\n"
            + verdict(p, o))
