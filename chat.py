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
        "• <i>Madrid Barcelona más de 9.5 córners</i> (total del partido)\n• <i>¿Cuál es el próximo rival del Almería?</i> · <i>último resultado del Betis</i>\n"
        "• <i>Combinada segura</i> (del último partido) · <i>combinada Girona Almería</i>\n"
        "• <i>¿Quién gana?</i> (del último partido que te mostré) · <i>quién gana Girona Almería</i> · <i>gana Girona a 2.10</i>")
_DB = {"t": 0}
LEAGUE = {}  # liga → [(estadísticas del local, del visitante)] en orden (sin partidos sin estadísticas: ESPN los da a 0)
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
def _add(db, key, name, sport, kind, date, vals, against=None, team=None, league=None):
    e = db.setdefault(key, {"name": name, "sport": sport, "kind": kind, "team": team, "games": [], "leagues": set()})
    e["games"].append((str(date)[:10], vals, against or {})); e["team"] = team or e["team"]
    if league: e["leagues"].add(league)


def build():
    db = {}; LEAGUE.clear()
    for code in ES.LEAGUES:
        for g in ES.load(code):
            ts = g.get("team_stats") or {}
            for sd, team, gf, ga in (("h", g["home"], g["hs"], g["as"]), ("a", g["away"], g["as"], g["hs"])):
                o = "a" if sd == "h" else "h"
                tv = lambda d: {} if not (d.get("totalShots") or d.get("wonCorners") or d.get("foulsCommitted")) else {"goles": None, "corners": d.get("wonCorners"), "tiros": d.get("totalShots"), "tiros_puerta": d.get("shotsOnTarget"),
                                "faltas": d.get("foulsCommitted"), "tarjetas": (d.get("yellowCards") or 0) + (d.get("redCards") or 0) if d else None,
                                "fueras": d.get("offsides")}
                f, a = tv(ts.get(sd) or {}), tv(ts.get(o) or {})
                f["goles"], a["goles"] = gf, ga
                if sd == "h": LEAGUE.setdefault(code, []).append((f, a))  # (local, visitante) para las medias de la liga
                f["_rival"], f["_casa"], f["_liga"] = g["away"] if sd == "h" else g["home"], sd == "h", code
                _add(db, f"futbol|t|{team}", team, "futbol", "equipo", g["date"], f, a, league=code)
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
                _add(db, f"{lg}|t|{me}", me, lg, "equipo", g["date"], {key: gf, "_rival": g["away"] if home else g["home"], "_casa": home},
                     {key: ga}, league=lg)
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


# ── totales de partido (fútbol): media de la liga × ataque y defensa de cada equipo (olvido 0.07, encogido con K=10 partidos)
# y binomial negativa con r común. Medido (2.ª mitad, ~7.500 partidos ESPN limpios): córners ≈ media de la liga (8.5: −0.0004
# ± 0.0014; el historial de los equipos casi no predice el total de córners), tarjetas −0.0058 ± 0.0018 en 4.5; calibrado
# (córners ≥ 65 % → 69 % previsto, 70 % real; tarjetas 73 % → 73 %). Goles: el 1X2 de winner().
ALPHA_T, K_T = 0.07, 10.0
R_NB = {"corners": 20.0, "tarjetas": 40.0, "goles": 40.0}


def _rate(e, stat, base, against=False):
    x = w = 0.0
    for g in e["games"]:
        v = g[2 if against else 1].get(stat)
        if v is None or v != v: continue
        x, w = x * (1 - ALPHA_T) + v, w * (1 - ALPHA_T) + 1
    return (x + K_T * base) / (w + K_T) / base if base > 0 else 1.0


def match_model(kh, ka, D, stat):
    """(media del local, media del visitante, r) del modelo anclado a la liga, o None (sin liga o sin datos)."""
    h, a = D[kh], D[ka]
    code = next((g[1].get("_liga") for g in reversed(h["games"]) if g[1].get("_liga")), None)
    games = [x for x in LEAGUE.get(code, [])[-380:] if x[0].get(stat) is not None and x[1].get(stat) is not None]
    if len(games) < 30: return None
    Lh = sum(x[0][stat] for x in games) / len(games); La = sum(x[1][stat] for x in games) / len(games)
    L = (Lh + La) / 2  # shortcut: ritmos de cada equipo relativos a la media de la liga sin separar casa/fuera (casa/fuera medido peor)
    mh = Lh * _rate(h, stat, L) * _rate(a, stat, L, True)
    ma = La * _rate(a, stat, L) * _rate(h, stat, L, True)
    return mh, ma, R_NB.get(stat, 20.0)


def nb_over(mu, r, line): return float(SS.nbinom.sf(math.floor(line), r, r / (r + mu)))


def leg_prob(keys, leg, D):
    """(probabilidad, texto) de una pierna con 1 entidad (jugador/equipo) o 2 equipos (total del partido)."""
    es = [D[k] for k in keys]
    stat, line, over = leg["stat"], leg["line"], leg["over"]
    mm = match_model(*keys, D, stat) if len(es) == 2 and all(e["kind"] == "equipo" and e["sport"] == "futbol" for e in es) else None
    if mm:
        p = nb_over(mm[0] + mm[1], mm[2], line); p = p if over else 1 - p
        txt = f"{A.esc(es[0]['name'])} – {A.esc(es[1]['name'])} {'más' if over else 'menos'} de {line:g} {LABEL[stat]}: esperado {mm[0] + mm[1]:.1f} → <b>{p:.0%}</b>"
        txt += coin(p)
        if over and p < 0.75: txt += safer(lambda l: nb_over(mm[0] + mm[1], mm[2], l), line, stat)
        return p, txt
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
    p = p_over(mu, var, line)
    p = shrink(p, es[0]["sport"], es[0]["kind"], stat, line, D, total=len(es) == 2)
    p = p if over else 1 - p
    txt = f"{who} {'más' if over else 'menos'} de {line:g} {LABEL[stat]}: media {mu:.1f} → <b>{p:.0%}</b>"
    if hist: txt += f" (lo cumplió en {hist[0]} de sus {hist[1]} últimos)"
    if over and p < 0.75:
        txt += safer(lambda l: shrink(p_over(mu, var, l), es[0]["sport"], es[0]["kind"], stat, l, D, total=len(es) == 2), line, stat)
    return p, txt


# confianza: las probabilidades extremas de sus 20 últimos partidos se pasaban de optimistas (dice 76 %, ocurre 66 %);
# temperatura sobre el logit alrededor de la frecuencia de toda la liga, elegida con ~1,4 M de partidos previos
TEMP = {("futbol", "jugador"): 0.8, ("futbol", "equipo"): 0.65, ("nba", "jugador"): 0.85, ("nba", "equipo"): 0.85,
        ("nhl", "jugador"): 0.85, ("nhl", "equipo"): 0.85}
_BASE = {}


def base_rate(D, sport, kind, stat, line):
    k = (sport, kind, stat, line)
    if k not in _BASE:
        n = h = 0
        for e in D.values():
            if e["sport"] != sport or e["kind"] != kind: continue
            for g in e["games"][-N_LAST:]:
                v = g[1].get(stat)
                if v is None or v != v: continue
                n += 1; h += v > line
        _BASE[k] = (h + 1) / (n + 2)
    return _BASE[k]


def shrink(p, sport, kind, stat, line, D, total=False):
    c = TEMP.get((sport, kind), 0.8)
    # jugador/equipo: hacia la media de la liga (mejor log-loss). Totales de partido: hacia el 50 % con c 0.65 (medido en 7.524
    # partidos: córners >9.5 0.6943→0.6896, tarjetas >4.5 0.6612→0.6602, goles >2.5 0.6879→0.6852; apenas mejor que una moneda)
    b = 0.5 if total else base_rate(D, sport, kind, stat, line)
    lg = lambda q: math.log(min(max(q, 1e-4), 1 - 1e-4) / (1 - min(max(q, 1e-4), 1 - 1e-4)))
    return 1 / (1 + math.exp(-(lg(b) + c * (lg(p) - lg(b)))))


SAFE_P = 0.85  # «curarse en salud»: líneas del 85 % (medidas: córners 88→91 % real, tarjetas 89→89 %, goles 90→92 %)


def safe_line(f, target=SAFE_P, top=40):
    """La línea más alta (x.5) con P(más de la línea) ≥ target, o None."""
    return max((l + 0.5 for l in range(top) if f(l + 0.5) >= target), default=None)


def safer(f, line, stat):
    l = safe_line(f)
    if l is None or l >= line: return ""
    return f"\n👉 Para curarte en salud: <b>más de {l:g} {LABEL[stat]}</b> ({f(l):.0%}); buena pierna para una combinada"


def coin(p):
    return " · 📉 casi una moneda: aquí no hay ventaja" if 0.42 < p < 0.58 else ""


def verdict(p, odds):
    if not odds: return f"Cuota justa {1 / max(p, 1e-6):.2f}: apuéstala solo si te la pagan más alta."
    ev = p * odds - 1
    icon = "🟢 Me gusta" if ev >= 0.05 else "🟡 Justa" if ev >= -0.02 else "🔴 No la cogería"
    return f"{icon}: a {odds:.2f} ganas {ev:+.0%} por € a la larga (cuota justa {1 / max(p, 1e-6):.2f})."


# ── calendario: próximo partido y último resultado ─────────────────────────────────────────────────────────────
NEXT_RX = r"proxim|siguiente|cuando juega|contra quien|\brival|calendario|juega (?:hoy|manana|el)"
LAST_RX = r"ultimo (?:partido|resultado)|como (?:quedo|acabo|salio)|resultado"
SPORT_PATH = {"nba": "basketball/nba", "nhl": "hockey/nhl"}
TZ = "Europe/Madrid"
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _events(path, dates):
    import pandas as pd, requests
    r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard", params={"dates": dates, "limit": 500},
                     headers=A.UA, timeout=20)
    r.raise_for_status(); out = []
    for e in r.json().get("events", []):
        c = (e.get("competitions") or [{}])[0]
        if ((c.get("status") or e.get("status") or {}).get("type") or {}).get("completed"): continue
        t = {x.get("homeAway"): x["team"]["displayName"] for x in c.get("competitors", []) if x.get("team")}
        if "home" in t and "away" in t:
            out.append((pd.Timestamp(e["date"]), t["home"], t["away"], (e.get("league") or {}).get("name") or ""))
    return out


def scoreboard(sport, league, days=21, team=None):
    """[(fecha UTC, local, visitante, competición)] de los próximos días en ESPN: primero todo el rango de una vez y, si
    ahí no sale `team`, día a día (como espn_soccer.events, que sí funciona) hasta encontrarlo."""
    import pandas as pd
    a = pd.Timestamp.now(tz="UTC"); b = a + pd.Timedelta(days=days)
    path = SPORT_PATH.get(sport) or f"soccer/{ES.LEAGUES.get(league, league)}"
    try: out = _events(path, f"{a:%Y%m%d}-{b:%Y%m%d}")
    except Exception as ex: print("calendario rango", path, repr(ex)); out = []
    if team is None or any(plays(team, m) for m in out): return out
    for d in pd.date_range(a.normalize(), b, freq="D"):
        day = _events(path, f"{d:%Y%m%d}")
        if any(plays(team, m) for m in day): return day
    return out


def plays(name, m):
    """¿Juega el equipo `name` en el partido m? (mismo nombre de ESPN, o uno contenido en el otro: «UD Almería»)."""
    n = norm(name)
    return any(x and (n == x or (min(len(n), len(x)) >= 5 and (n in x or x in n))) for x in (norm(m[1] or ""), norm(m[2] or "")))


def tkey(sport, name, D):
    """Clave del equipo `name` (nombre de ESPN) en la base, o None."""
    k = f"{sport}|t|{name}"
    if k in D: return k
    return next((k for k, x in D.items() if x["kind"] == "equipo" and x["sport"] == sport and plays(x["name"], (None, name, ""))), None)


def team_of(key, D):
    e = D[key]
    if e["kind"] == "equipo": return key
    k = f"{e['sport']}|t|{e['team']}"
    if k in D: return k
    return next((k for k, x in D.items() if x["kind"] == "equipo" and x["sport"] == e["sport"] and x["name"] == e["team"]), None)


def exp_pair(kh, ka, D, stat):
    """(media del local, media del visitante) de una estadística: (a favor de uno + en contra del otro) / 2."""
    u = [(series(D[kh], stat), series(D[ka], stat, True)), (series(D[ka], stat), series(D[kh], stat, True))]
    if any(len(x) < 5 or len(y) < 5 for x, y in u): return None
    return [(moments(x)[0] + moments(y)[0]) / 2 for x, y in u]


# ── ¿quién gana? ───────────────────────────────────────────────────────────────────────────────────────────────
# fútbol: goles esperados (a favor/en contra de sus 20 últimos) × ventaja de local, Poisson con Dixon-Coles (analyst.probs).
# Medido en 8.339 partidos ESPN (2.ª mitad): log-loss 1X2 1.0332 con local ×1.12 vs 1.0725 de las frecuencias base
# (ventaja 1.0 → 1.0472; 1.2 → 1.0368). La casa suele afinar más: sirve para opinar, no para apostar a ciegas.
HFA = {"futbol": 1.12, "nhl": 1.05}
NBA_HOME, NBA_SD = 2.5, 12.5  # puntos de ventaja de local y desviación del margen final
WIN_RX = r"\bgana|ganador|ganaria|quien gana|favorito|1x2|pronostico|como (?:va a )?quedar|quien se lleva"
_CTX = {}


COMBO_RX = r"combinad|combi\b|parlay|apuesta segura|segura"


def safe_combo(kh, ka, D):
    """Combinada «segura» del partido: piernas del ≥ 85 % (córners, tarjetas, goles, doble oportunidad del favorito)."""
    h, a = D[kh], D[ka]
    legs = []
    for stat in ("corners", "tarjetas"):
        mm = match_model(kh, ka, D, stat)
        if not mm: continue
        f = lambda l, t=mm[0] + mm[1], r=mm[2]: nb_over(t, r, l)
        if (l := safe_line(f)) is not None: legs.append((f"Más de {l:g} {LABEL[stat]}", f(l)))
    m = exp_pair(kh, ka, D, "goles")
    if m:
        lh, la = max(m[0] * HFA["futbol"], 0.05), max(m[1] / HFA["futbol"], 0.05)
        P = A.probs(lh, la); t = P["H"] + P["D"] + P["A"]
        p00 = math.exp(-lh - la) * (1 - lh * la * A.RHO) / t
        if P["O15"] / t >= SAFE_P: legs.append(("Más de 1.5 goles", P["O15"] / t))
        elif 1 - p00 >= SAFE_P: legs.append(("Más de 0.5 goles", 1 - p00))
        dc = max((("1X", h["name"], (P["H"] + P["D"]) / t), ("X2", a["name"], (P["A"] + P["D"]) / t)), key=lambda x: x[2])
        if dc[2] >= 0.8: legs.append((f"{A.esc(dc[1])} o empate ({dc[0]})", dc[2]))
    if len(legs) < 2: return f"No veo piernas suficientemente seguras en {A.esc(h['name'])} – {A.esc(a['name'])}."
    p = float(np.prod([x[1] for x in legs]))
    return "\n".join([f"🧩 <b>Combinada segura · {A.esc(h['name'])} – {A.esc(a['name'])}</b>"]
                      + [f"• {txt} ({q:.0%})" for txt, q in legs]
                      + [f"Todas: <b>{p:.0%}</b> → cuota justa {1 / p:.2f}. Si la casa te la paga a más de {1.05 / p:.2f}, 🟢 adelante."])


def winner(kh, ka, D, odds=None, pick=None):
    h, a, sp = D[kh], D[ka], D[kh]["sport"]
    stat = "puntos" if sp == "nba" else "goles"
    m = exp_pair(kh, ka, D, stat)
    if not m: return f"Pocos datos de {A.esc(h['name'])} o {A.esc(a['name'])} para opinar."
    if sp == "nba":
        ph = float(SS.norm.sf(0, loc=m[0] - m[1] + NBA_HOME, scale=NBA_SD)); P = {"H": ph, "D": 0.0, "A": 1 - ph}
    else:
        P = A.probs(max(m[0] * HFA[sp], 0.05), max(m[1] / HFA[sp], 0.05)); t = P["H"] + P["D"] + P["A"]
        P = {k: P[k] / t for k in ("H", "D", "A")} | {"scores": P["scores"]}
        if sp == "nhl": P = {"H": P["H"] + P["D"] / 2, "D": 0.0, "A": P["A"] + P["D"] / 2}  # la prórroga/penaltis reparte el empate
    names = {"H": h["name"], "D": "Empate", "A": a["name"]}
    lines = [f"⚔️ <b>{A.esc(h['name'])} – {A.esc(a['name'])}</b>"]
    lines.append(" · ".join(f"{A.esc(names[k])} {P[k]:.0%} (cuota justa {1 / P[k]:.2f})" for k in ("H", "D", "A") if P[k] > 0))
    fav = max(("H", "A"), key=lambda k: P[k]); gap = abs(P["H"] - P["A"])
    lines.append(f"👉 Favorito: <b>{A.esc(names[fav])}</b>" if gap >= 0.12 else "👉 Muy igualado: no me mojaría con el ganador")
    if P.get("scores"): lines.append("Resultado más probable: " + " · ".join(f"{sc} ({p:.0%})" for sc, p in P["scores"][:3]))
    if odds and pick: lines.append(verdict(P[pick], odds[0]))
    return "\n".join(lines)


def preview(home, away, D, sport):
    """Totales esperados del partido (córners, tarjetas, goles) con lo de cada equipo a favor y en contra."""
    kh, ka = f"{sport}|t|{home}", f"{sport}|t|{away}"
    if kh not in D or ka not in D: return []
    out = []
    for stat in (("goles", "corners", "tarjetas") if sport == "futbol" else ("puntos",) if sport == "nba" else ("goles",)):
        mm = match_model(kh, ka, D, stat) if sport == "futbol" and stat != "goles" else None
        if mm:
            tot = mm[0] + mm[1]
            safe = safe_line(lambda l: nb_over(tot, mm[2], l))
            out.append(f"• {LABEL[stat]}: {mm[0]:.1f} – {mm[1]:.1f} (total {tot:.1f})"
                       + (f" · seguro ({SAFE_P:.0%}): más de {safe:g}" if safe is not None else ""))
            continue
        u = [(series(D[kh], stat), series(D[ka], stat, True)), (series(D[ka], stat), series(D[kh], stat, True))]
        if any(len(x) < 5 or len(y) < 5 for x, y in u): continue
        m = [(moments(x)[0] + moments(y)[0]) / 2 for x, y in u]
        out.append(f"• {LABEL[stat]}: {m[0]:.1f} – {m[1]:.1f} (total {sum(m):.1f})")
    return (["Lo esperado:"] + out) if out else []


def next_match(key, D):
    import pandas as pd
    tk = team_of(key, D)
    if not tk: return f"No sé en qué equipo juega {A.esc(D[key]['name'])}."
    e = D[tk]; name = norm(e["name"])
    found, errs = [], []
    recent = sorted({g[1].get("_liga") for g in e["games"][-8:] if g[1].get("_liga")})
    for lg in recent + sorted(e["leagues"] - set(recent)) or [e["sport"]]:
        try: found += [m for m in scoreboard(e["sport"], lg, team=e["name"]) if plays(e["name"], m)]
        except Exception as ex: print("calendario", lg, repr(ex)); errs.append(type(ex).__name__)
        if found and lg in recent: break
    if not found:
        return f"{_who(e)}\nNo encuentro su próximo partido en las próximas 3 semanas" + (f" (ESPN no responde: {errs[0]})." if errs else ".")
    when, home, away, comp = min(found, key=lambda m: m[0])
    comp = comp or ""
    loc = when.tz_convert(TZ)
    rival = away if plays(e["name"], (None, home, "")) else home
    _CTX["partido"] = (tkey(e["sport"], home, D), tkey(e["sport"], away, D))
    lines = [_who(e), f"Próximo partido: <b>{A.esc(home)} – {A.esc(away)}</b>",
             f"📅 {DIAS[loc.dayofweek]} {loc:%d/%m} a las {loc:%H:%M} (hora España)" + (f" · {A.esc(comp)}" if comp else "")]
    kr = f"{e['sport']}|t|{rival}"
    if kr in D:
        last = [g for g in D[tk]["games"] if g[1].get("_rival") == rival][-3:]
        if last: lines.append("Últimos entre ellos: " + " · ".join(f"{g[1].get('goles', g[1].get('puntos')):g}-{g[2].get('goles', g[2].get('puntos')):g}" for g in last)
                              + f" (goles de {A.esc(e['name'])} primero)")
    return "\n".join(lines + preview(home, away, D, e["sport"]))


def last_match(key, D):
    tk = team_of(key, D)
    if not tk: return f"No sé en qué equipo juega {A.esc(D[key]['name'])}."
    e = D[tk]; d, f, a = e["games"][-1]
    k = "puntos" if e["sport"] == "nba" else "goles"
    gf, ga = f.get(k), a.get(k)
    res = "✅ ganó" if gf > ga else "❌ perdió" if gf < ga else "➖ empató"
    return f"{_who(e)}\nÚltimo partido ({d}): {res} {gf:g}-{ga:g} contra {A.esc(f.get('_rival', '?'))}" + (" en casa" if f.get("_casa") else " fuera")


def answer(text):
    """Respuesta HTML a un mensaje libre del dueño."""
    t = text.strip()
    if not t or re.match(r"^/?(ayuda|help|hola|que puedes)", norm(t)): return HELP
    D = db()
    q = low(t)
    if re.search(COMBO_RX, q) and not re.search(r"mas de|menos de|over|under|\d\+", q):
        ks = find(t, D, n=2, kind="equipo")
        if len(ks) == 2: kh, ka = ks
        elif ks:
            _CTX.pop("partido", None); next_match(ks[0], D); kh, ka = _CTX.get("partido") or (None, None)
        else: kh, ka = _CTX.get("partido") or (None, None)
        if not (kh and ka and kh in D and ka in D): return "¿De qué partido? Por ejemplo: <i>combinada Girona Almería</i>"
        _CTX["partido"] = (kh, ka)
        if D[kh]["sport"] != "futbol": return "La combinada segura de momento es solo de fútbol."
        return safe_combo(kh, ka, D)
    if re.search(WIN_RX, q) and not re.search(r"mas de|menos de|over|under|\d\+", q):
        ks = find(t, D, n=2, kind="equipo")
        if len(ks) == 2: kh, ka = ks
        elif ks and _CTX.get("partido") and ks[0] in _CTX["partido"]: kh, ka = _CTX["partido"]
        elif ks:  # un equipo: su próximo partido
            _CTX.pop("partido", None); next_match(ks[0], D); kh, ka = _CTX.get("partido") or (None, None)
            if not kh: return f"No encuentro el próximo partido de {A.esc(D[ks[0]]['name'])}."
        elif _CTX.get("partido"): kh, ka = _CTX["partido"]
        else: return "¿De qué partido? Escríbeme los dos equipos, por ejemplo: <i>quién gana Girona Almería</i>"
        if kh not in D or ka not in D: return "No tengo datos de uno de los dos equipos."  # (None si ESPN trae un recién ascendido)
        _CTX["partido"] = (kh, ka)
        pick = next((k for k, x in (("H", kh), ("A", ka)) if x in ks), None) if odds_in(t) else None
        if pick is None and odds_in(t) and re.search(r"empate|\bx\b", q): pick = "D"
        return winner(kh, ka, D, odds_in(t), pick)
    if re.search(NEXT_RX, q) or re.search(LAST_RX, q):
        keys = find(t, D, n=1)
        if keys: return next_match(keys[0], D) if re.search(NEXT_RX, q) else last_match(keys[0], D)
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
