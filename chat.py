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
         ("faltas_recibidas", r"faltas? recibidas?|recibir\w* (?:\d+\+? )?faltas?|le hac\w+ faltas?|fouls? (?:won|suffered|drawn)", "faltas recibidas"),
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
PLAYER_STATS = {"futbol": ["goles", "asistencias", "tiros", "tiros_puerta", "faltas", "faltas_recibidas", "tarjetas", "fueras", "paradas"],
                "nba": ["puntos", "rebotes", "asistencias", "triples", "robos", "tapones"], "nhl": ["goles", "tiros"]}
TEAM_STATS = {"futbol": ["goles", "corners", "tiros", "tiros_puerta", "faltas", "tarjetas", "fueras"], "nba": ["puntos"], "nhl": ["goles"]}
HELP = ("💬 Escríbeme, por ejemplo:\n• <i>estadísticas de Mbappé</i>\n• <i>Mbappé más de 1.5 tiros a puerta a 1.9</i>\n"
        "• <i>LeBron más de 25.5 puntos y Curry 4+ triples @3.1</i>\n• <i>Real Madrid más de 5.5 córners</i>\n"
        "• <i>Madrid Barcelona más de 9.5 córners</i> (total del partido)\n• <i>¿Cuál es el próximo rival del Almería?</i> · <i>último resultado del Betis</i>\n"
        "• <i>Combinada segura</i> (del último partido) · <i>combinada Girona Almería</i> · <i>combinada del día</i>\n"
        "• <i>¿Quién gana?</i> (del último partido que te mostré) · <i>quién gana Girona Almería</i> · <i>gana Girona a 2.10</i>")
_DB = {"t": 0}
CITY = {"atenas": "athens", "napoles": "napoli", "oporto": "porto", "praga": "prague", "munich": "munich", "lisboa": "lisbon",
        "estambul": "istanbul", "copenhague": "copenhagen", "belgrado": "belgrade", "zagreb": "zagreb", "bucarest": "bucharest",
        "varsovia": "warsaw", "amsterdam": "amsterdam", "londres": "london", "brujas": "brugge", "marsella": "marseille",
        "lyon": "lyon", "roma": "roma", "glasgow": "glasgow", "moscu": "moscow", "salzburgo": "salzburg", "eindhoven": "eindhoven",
        "bilbao": "bilbao", "sevilla": "sevilla", "milan": "milan", "turin": "torino", "genova": "genoa", "florencia": "fiorentina",
        "leverkusen": "leverkusen", "dortmund": "dortmund", "colonia": "koln", "friburgo": "freiburg", "maguncia": "mainz",
        "monchengladbach": "monchengladbach", "estrasburgo": "strasbourg", "niza": "nice", "lille": "lille", "mallorca": "mallorca"}
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
                v["faltas_recibidas"] = p[15] if len(p) > 15 else None  # campo nuevo: None en partidos aún sin rellenar
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
    for es, en in CITY.items(): t = t.replace(f" {es} ", f" {en} ")  # «AEK Atenas» (casas españolas) → «AEK Athens» (ESPN)
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


_MKP = {}


def match_market(home, away):
    """Supremacía del local sin margen (P local − P visitante) con las cuotas del partido: DraftKings de Champions/Europa
    (data/espn/odds_ucl|uel.csv) o el 1X2 de Pinnacle (data/pinnacle/futbol.csv). None si no hay cuotas."""
    import pandas as pd, multideporte
    now = pd.Timestamp.now(tz="UTC")
    for f, cols in ((A.DATA / "espn" / "odds_ucl.csv", ("oh", "od", "oa")), (A.DATA / "espn" / "odds_uel.csv", ("oh", "od", "oa")),
                    (A.DATA / "pinnacle" / "futbol.csv", ("o_home", "o_draw", "o_away"))):
        import pinnacle
        d = multideporte._latest(f) if "espn" in str(f) else pinnacle.read("futbol")
        if not len(d): continue
        if "market" in d: d = d[d.market == "moneyline"]
        for r in d.itertuples():
            o = [getattr(r, c, None) for c in cols]
            if any(x is None or x != x or x <= 1 for x in o) or abs((pd.Timestamp(r.start) - now).days) > 5: continue
            if A.find_team(home, [r.home]) and A.find_team(away, [r.away]):
                ih, idr, ia = (1 / x for x in o); t = ih + idr + ia
                return (ih - ia) / t
    return None


def market_params():
    if "p" not in _MKP:
        try: _MKP["p"] = json.loads((A.DATA / "football_stats.json").read_text()).get("market", {})
        except (FileNotFoundError, ValueError): _MKP["p"] = {}
    return _MKP["p"]


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
    # cuotas del partido (train_market_stats.py, medido: córners por equipo −0.012 de log-loss): el favorito saca más córners
    # y ve menos tarjetas; corrige además los cruces entre ligas (el AEK domina en Grecia, no ante el Madrid)
    mp = market_params().get({"corners": "corners", "tarjetas": "cards"}.get(stat, ""), {})
    s = match_market(h["name"], a["name"]) if mp else None
    if s is not None:
        mh, ma = mh * math.exp(mp["a"] * s), ma * math.exp(-mp["a"] * s)
    return mh, ma, R_NB.get(stat, 20.0)


def fixture_order(k1, k2, D):
    """(local, visitante) del partido entre dos equipos: el de la memoria del chat o el de las cuotas guardadas; si no, tal cual."""
    import pandas as pd, multideporte
    if set(_CTX.get("partido") or ()) == {k1, k2}: return _CTX["partido"]
    n1, n2 = D[k1]["name"], D[k2]["name"]
    for f in ("odds_ucl", "odds_uel"):
        d = multideporte._latest(A.DATA / "espn" / f"{f}.csv")
        for r in d.itertuples() if len(d) else []:
            if A.find_team(n2, [r.home]) and A.find_team(n1, [r.away]): return k2, k1
            if A.find_team(n1, [r.home]) and A.find_team(n2, [r.away]): return k1, k2
    return k1, k2


def team_vs(focus, rival, stat, D, line=None, over=True):
    """Cuántos córners/tarjetas sacará un equipo CONTRA un rival concreto (a favor del equipo × en contra del rival, ajustado
    por las cuotas del partido) y la probabilidad de la línea."""
    kh, ka = fixture_order(focus, rival, D)
    mm = match_model(kh, ka, D, stat)
    if not mm: return f"Pocos datos de {LABEL[stat]} de {A.esc(D[focus]['name'])} o {A.esc(D[rival]['name'])}."
    mu = mm[0] if focus == kh else mm[1]
    f = lambda l: nb_over(mu, mm[2], l)
    own = series(D[focus], stat)[-5:]
    lines = [f"⚽ <b>{A.esc(D[focus]['name'])}</b> contra <b>{A.esc(D[rival]['name'])}</b>"
             + (" (en casa)" if focus == kh else " (fuera)"),
             f"• {LABEL[stat]} esperados: <b>{mu:.1f}</b> (su media normal {moments(series(D[focus], stat))[0]:.1f}; últimos 5: {' '.join(f'{v:g}' for v in own)})"]
    if line is not None:
        p = f(line) if over else 1 - f(line)
        lines.append(f"• {'Más' if over else 'Menos'} de {line:g}: <b>{p:.0%}</b>{coin(p)}")
        if over and p < 0.75: lines.append(safer(f, line, stat).strip())
    else:
        ls = [l + 0.5 for l in range(max(0, int(mu) - 3), int(mu) + 3)]
        lines.append("• " + " · ".join(f"más de {l:g}: {f(l):.0%}" for l in ls))
    return "\n".join(lines)


VS_RX = r"\b(contra|vs|frente a|ante|enfrente)\b"


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


COMBO_RX = r"combinad|combi\b|parlay|apuesta segura|segura|popurri|lo mas seguro"


COMBO_FLOOR, COMBO_MAX = 0.60, 8  # «lo máximo de forma segura»: se añaden piernas mientras todas juntas sigan ≥ 60 %
DAY_LEAGUES = ["SP1", "E0", "I1", "D1", "F1", "ucl", "uel", "SP2", "E1", "P1", "N1", "T1", "B1",
               # con datos de ESPN y medias de liga también (córners/tarjetas): más partidos donde elegir piernas
               "I2", "D2", "F2", "G1", "SC0", "ARG", "BRA", "MEX", "USA", "JPN", "AUT", "DNK", "NOR", "SWE", "SWZ"]


def safe_legs(kh, ka, D):
    """Piernas seguras del partido: córners y tarjetas totales (≥ 85 %), goles (más de 1.5 o 0.5) y doble oportunidad
    del favorito (≥ 80 %). Todas medidas en partidos previos (ver SAFE_P, match_model y HFA)."""
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
        # medido (2.ª mitad ESPN): más de 1.5 con 87 % previsto acierta 84 % → se exige 88 %; más de 0.5 cumple (88→91 %)
        if P["O15"] / t >= 0.88: legs.append(("Más de 1.5 goles", P["O15"] / t))
        elif 1 - p00 >= SAFE_P: legs.append(("Más de 0.5 goles", 1 - p00))
        dc = max((("1X", h["name"], (P["H"] + P["D"]) / t), ("X2", a["name"], (P["A"] + P["D"]) / t)), key=lambda x: x[2])
        # 1X cumple de sobra (82 % previsto → 86 % real); X2 se pasa de optimista (82 % → 75 %): al visitante se le exige 88 %
        if dc[2] >= (0.8 if dc[0] == "1X" else 0.88): legs.append((f"{A.esc(dc[1])} o empate ({dc[0]})", dc[2]))
    return legs


def pick_legs(cands, per_match=None, target=None):
    """De [(partido, texto, p)], las más seguras primero mientras el total siga ≥ COMBO_FLOOR (máx. COMBO_MAX piernas).
    Con `target` (cuota buscada, p. ej. 2.5): se añaden piernas seguras hasta que la cuota justa llega a esa cifra."""
    out, p, n = [], 1.0, {}
    # con cuota objetivo, primero las piernas seguras que más pagan (las de 85-90 %): llega con menos piernas a la misma
    # probabilidad total, y cada pierna de más suma el margen de la casa
    for m, txt, q in sorted(cands, key=lambda x: x[2] if target else -x[2]):
        if target:
            if 1 / p >= target or len(out) >= 15: break
        elif len(out) >= COMBO_MAX or p * q < COMBO_FLOOR: continue
        if per_match and n.get(m, 0) >= per_match: continue
        out.append((m, txt, q)); p *= q; n[m] = n.get(m, 0) + 1
    return out, p


def _combo_text(title, legs, p, grouped=False):
    lines = [title]
    if grouped:
        last = None
        for m, txt, q in legs:
            if m != last: lines.append(f"<b>{m}</b>"); last = m
            lines.append(f"  • {txt} ({q:.0%})")
    else: lines += [f"• {txt} ({q:.0%})" for _, txt, q in legs]
    lines.append(f"{len(legs)} piernas · todas: <b>{p:.0%}</b> → cuota justa <b>{1 / p:.2f}</b>. "
                 f"Si la casa te la paga a más de {1.05 / p:.2f}, 🟢 adelante.")
    return "\n".join(lines)


LAST_FILE = A.DATA / "chat_ultima.json"  # local (sin subir): la última combinada sobrevive a un reinicio del lector


def remember(title, legs, odds, p):
    """Última combinada mostrada, para «simúlala» (simulaciones.record)."""
    _CTX["ultima"] = {"titulo": title, "legs": legs, "odds": round(odds, 2) if odds else (round(1 / p, 2) if p else None), "p": p}
    try: LAST_FILE.write_text(json.dumps(_CTX["ultima"], ensure_ascii=False, default=str))
    except OSError: pass


def last_combo():
    if not _CTX.get("ultima") and LAST_FILE.exists():
        try: _CTX["ultima"] = json.loads(LAST_FILE.read_text())
        except ValueError: pass
    return _CTX.get("ultima")


SIM_RX = r"\bsimul|\bapunta|\bregistra|\bguardala|\bjuegala|\bjugamos|\bmetele|\bvamos con (esa|ella)"
SIMS_RX = r"como van|como vamos|como (va|lleva|esta) (esta|esa|mi|la)\b|mis (apuestas|simuladas|combinadas)|balance|que tal (van|mis)"


def safe_combo(kh, ka, D):
    h, a = D[kh], D[ka]
    legs, p = pick_legs([("", txt, q) for txt, q in safe_legs(kh, ka, D)])
    if len(legs) < 2: return f"No veo piernas suficientemente seguras en {A.esc(h['name'])} – {A.esc(a['name'])}."
    code = next((g[1].get("_liga") for g in reversed(h["games"]) if g[1].get("_liga")), None)
    remember(f"Combinada segura {h['name']} – {a['name']}", [dict(code=code, home=h["name"], away=a["name"], start=None, txt=txt,
                                                                   odds=None, p=q) for _, txt, q in legs], None, p)
    extra = "\n💡 Para subir más la cuota sin arriesgar: <i>combinada del día</i> (junta piernas seguras de varios partidos)."
    return _combo_text(f"🧩 <b>Combinada segura · {A.esc(h['name'])} – {A.esc(a['name'])}</b>", legs, p) + extra


def upcoming(codes, hours=48):
    """{liga: [(fecha, local, visitante, competición)]} de las próximas `hours` horas, pidiendo a ESPN día a día y en
    paralelo (la consulta de un rango de fechas no devolvía partidos)."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    now = pd.Timestamp.now(tz="UTC")
    days = [f"{d:%Y%m%d}" for d in pd.date_range(now.normalize(), now + pd.Timedelta(hours=hours), freq="D")]
    jobs = [(c, d) for c in codes for d in days]
    def get(job):
        c, d = job
        try: return c, _events(f"soccer/{ES.LEAGUES.get(c, c)}", d)
        except Exception as ex: print("calendario", c, d, repr(ex)); return c, []
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for c, evs in ex.map(get, jobs):
            out.setdefault(c, []).extend(m for m in evs if now < m[0] < now + pd.Timedelta(hours=hours))
    return out


SPORT_WORDS = {"nba": r"\bnba\b|baloncesto|basket", "nhl": r"\bnhl\b|hockey", "nfl": r"\bnfl\b|americano", "mlb": r"\bmlb\b|beisbol",
               "cs2": r"\bcs2?\b|counter", "lol": r"\blol\b|league of legends", "dota2": r"\bdota", "r6": r"\br6\b|rainbow",
               "mlbb": r"mlbb|mobile legends", "fc": r"fc ?2[67]|fifa|esoccer", "futbol": r"futbol|liga|champions|europa league"}
ESPORTS_WORDS = r"e-?sports?|videojuegos"


def sports_in(q):
    """Deportes nombrados en la frase (None = todos)."""
    got = {k for k, rx in SPORT_WORDS.items() if re.search(rx, q)}
    if re.search(ESPORTS_WORDS, q): got |= {"cs2", "lol", "dota2", "r6", "mlbb", "fc"}
    return got or None


def other_legs(hours, sports):
    """Piernas «gana X» de NBA/NHL/NFL/MLB, esports y FC 27 (multideporte.py: probabilidades de los modelos del bot)."""
    import multideporte
    try: L = multideporte.legs(hours, None if sports is None else [s for s in sports if s != "futbol"])
    except Exception as ex: print("multideporte", repr(ex)); return []
    out = []
    for l in L:
        loc = pd_ts(l["start"]).tz_convert(TZ)
        l["m"] = f"{l['sport']} {A.esc(l['home'])} – {A.esc(l['away'])} · {DIAS[loc.dayofweek]} {loc:%H:%M}"
        out.append(l)
    return out


def pd_ts(x):
    import pandas as pd
    return pd.Timestamp(x)


def day_combo(D, hours=48, target=None, sports=None):
    """Combinada segura «popurrí» con los partidos de las próximas 48 h de las ligas principales: la pierna más segura de
    cada partido (las de partidos distintos son independientes), hasta COMBO_FLOOR o hasta la cuota `target`."""
    import pandas as pd
    cands, when, seen, meta = [], {}, 0, {}
    for code, evs in (upcoming(DAY_LEAGUES, hours) if sports is None or "futbol" in sports else {}).items():
        seen += len(evs)
        for t, home, away, _ in evs:
            kh, ka = tkey("futbol", home, D), tkey("futbol", away, D)
            if not (kh and ka): continue
            loc = t.tz_convert(TZ); m = f"{A.esc(home)} – {A.esc(away)} · {DIAS[loc.dayofweek]} {loc:%H:%M}"
            if m in when: continue  # el mismo partido listado en dos competiciones
            when[m] = t; meta[m] = (code, str(t), home, away)
            cands += [(m, txt, q) for txt, q in safe_legs(kh, ka, D)]
    for l in other_legs(hours, sports):  # NBA, NHL, esports, FC 27: favorito claro según el modelo del bot
        if l["p"] < SAFE_P: continue
        when[l["m"]] = pd_ts(l["start"]); meta[l["m"]] = l; seen += 1
        cands.append((l["m"], l["txt"], l["p"]))
    print(f"combinada del día: {seen} partidos, {len(when)} con datos, {len(cands)} piernas seguras")
    legs, p = pick_legs(cands, per_match=1, target=target)
    if (target and 1 / p < target) or len(legs) < 2:  # no llega con una por partido: hasta 2 (siguen siendo ≥ 85 %)
        legs, p = pick_legs(cands, per_match=2, target=target)
    if len(legs) < 2:
        return (f"No encuentro partidos con piernas seguras en las próximas {hours} h"
                + (f" (ESPN me da {seen} partidos, pero de {len(when)} tengo datos)." if seen else " (ESPN no me devuelve partidos)."))
    legs.sort(key=lambda x: (when[x[0]], -x[2]))
    remember("Combinada segura del día" + (f" (cuota {target:g})" if target else ""), [_leg(meta[m], txt, q) for m, txt, q in legs], None, p)
    title = "🧩 <b>Combinada segura del día</b>" + (f" · buscando cuota {target:g}" if target else "")
    note = (f"\n⚠️ Con piernas seguras solo llego a cuota justa {1 / p:.2f}." if target and 1 / p < target else "")
    return _combo_text(title, legs, p, grouped=True) + note


COMBO_MIN_LEG = 0.40  # cada pierna ≥ 40 %: el laboratorio muestra que las cuotas altas sueltas pierden (> 6: −15 % sin valor)
LONG_P = (0.30, 0.92)  # piernas de la combinada larga: probabilidad del modelo en la zona calibrada
# prudencia: en totales de córners el modelo apenas supera a la media de la liga (medido) y a Unibet aún no se le ha
# medido aquí → probabilidad usada = 30 % modelo + 70 % la de la casa sin margen; candidata hasta medirla con resultados
W_LONG, LONG_EDGE = 0.30, 0.02


def _leg(meta, txt, q):
    """Pierna para simular: fútbol (code, start, home, away) o de otro deporte (dict de multideporte con src y side)."""
    if isinstance(meta, dict):
        return dict(src=meta["src"], side=meta["side"], code=None, start=meta["start"], home=meta["home"], away=meta["away"],
                    txt=txt, odds=meta.get("odds"), p=q)
    return dict(code=meta[0], start=meta[1], home=meta[2], away=meta[3], txt=txt, odds=None, p=q)


def shooter_legs(D, team_key, home, away, t, n=2, min_p=0.35):
    """Tiros a puerta de los rematadores de un equipo con cuotas de Unibet: [(ventaja, p, cuota, txt, 'sot', jugador)].
    Probabilidad del modelo (medido en 166 props liquidadas: predice 21.5 % y acierta 21.7 %; la de Unibet sin margen
    no sirve aquí: daba 44.7 %). Los n con más probabilidad (los delanteros/rematadores salen solos: más tiros)."""
    import kambi
    try: rows = [r for r in kambi.player_props(home, away, t) if r["mercado"].startswith("Player's shots on target") and r["linea"]]
    except Exception as ex: print("Kambi props", repr(ex)); rows = []
    team = D[team_key]["name"]
    out = []
    for r in rows:
        ks = [k for k in find(r["jugador"], D, n=1, kind="jugador") if D[k].get("team") == team]
        if not ks: continue
        p, _ = leg_prob(ks, {"stat": "tiros_puerta", "line": r["linea"], "over": True}, D)
        if p is None or p < min_p: continue
        out.append((p * r["cuota"] - 1, p, r["cuota"], f"Más de {r['linea']:g} tiros a puerta", "sot", D[ks[0]]["name"]))
    best = {}
    for c in sorted(out, key=lambda c: -c[1]): best.setdefault(c[5], c)  # una línea por jugador: la más probable
    return sorted(best.values(), key=lambda c: -c[1])[:n]


def match_combo(D, kh, ka, target, hours=72, by_team=False, shooters=None):
    """Combinada de cuota alta de UN partido («crear apuesta»): todas las líneas de Unibet del partido (córners y tarjetas,
    total y de cada equipo), la de más ventaja de cada mercado (prob. = 30 % modelo + 70 % casa sin margen) y, por
    estadística, o el total o los de equipo (no los dos: el total es la suma). Ojo: en un mismo partido las piernas van
    algo ligadas y la casa suele pagar menos que el producto: es una aproximación."""
    import kambi
    h, a = D[kh]["name"], D[ka]["name"]
    fx = [(t, home, away) for evs in upcoming(DAY_LEAGUES, hours).values() for t, home, away, _ in evs
          if {tkey("futbol", home, D), tkey("futbol", away, D)} == {kh, ka}]
    if not fx: return f"No encuentro {A.esc(h)} – {A.esc(a)} en las próximas {hours} h."
    t, home, away = fx[0]
    kh, ka = tkey("futbol", home, D), tkey("futbol", away, D)
    try: off = kambi.offers(home, away, t, stats=("corners", "cards"))
    except Exception as ex: print("Kambi", repr(ex)); off = {}
    if not off: return f"Unibet aún no tiene mercados de córners y tarjetas de {A.esc(home)} – {A.esc(away)}."
    _CTX["crear"] = (kh, ka, target)
    cands = []
    for (stat, scope), lines in off.items():
        mm = match_model(kh, ka, D, "corners" if stat == "corners" else "tarjetas")
        if not mm: continue
        mu = mm[0] + mm[1] if scope == "t" else mm[0] if scope == "h" else mm[1]
        who = "" if scope == "t" else f" de {home if scope == 'h' else away}"
        best = None
        if by_team and scope == "t": continue  # «de cada equipo»: solo las de equipo
        for line, o in lines.items():
            po = nb_over(mu, mm[2], line); mo = (1 / o["o"]) / (1 / o["o"] + 1 / o["u"])
            for side, pmod, pmk, odd in (("Más", po, mo, o["o"]), ("Menos", 1 - po, 1 - mo, o["u"])):
                q = W_LONG * pmod + (1 - W_LONG) * pmk
                if not (LONG_P[0] <= q <= LONG_P[1]): continue
                leg = (q * odd - 1, q, odd, f"{side} de {line:g} {LABEL['corners' if stat == 'corners' else 'tarjetas']}{who}", stat, scope)
                if best is None or leg[0] > best[0]: best = leg
        if best: cands.append(best)
    forced = []
    if shooters: forced = shooter_legs(D, shooters, home, away, t)
    legs, odds, p, used = list(forced), 1.0, 1.0, {}
    for c in forced: odds *= c[2]; p *= c[1]
    for c in sorted(cands, key=lambda c: -c[0]):
        if odds >= target and not by_team: break  # lo pedido (cada equipo) va entero aunque pase de la cuota
        st, sc = c[4], c[5]
        if (sc == "t" and used.get(st)) or (sc != "t" and "t" in used.get(st, set())): continue  # total o equipos, no ambos
        legs.append(c); odds *= c[2]; p *= c[1]; used.setdefault(st, set()).add(sc)
    if not legs: return f"No veo líneas útiles de Unibet en {A.esc(home)} – {A.esc(away)}."
    ev = p * odds - 1
    remember(f"Crear apuesta {home} – {away}", [dict(code=None, start=str(t), home=home, away=away, txt=c[3], odds=c[2], p=c[1],
                                                     **({"jugador": c[5]} if c[4] == "sot" else {})) for c in legs], odds, p)
    loc = t.tz_convert(TZ)
    out = [f"🎰 <b>Crear apuesta · {A.esc(home)} – {A.esc(away)}</b> · {DIAS[loc.dayofweek]} {loc:%H:%M} (cuotas de Unibet)"]
    out += [f"  • {A.esc(c[5] + ': ') if c[4] == 'sot' else ''}{A.esc(c[3])} @ {c[2]:.2f} ({c[1]:.0%}, ventaja {c[0]:+.0%})" for c in legs]
    if shooters and not forced: out.append(f"⚠️ Unibet aún no da tiros a puerta de jugadores de {A.esc(D[shooters]['name'])} (o no tengo sus datos).")
    if by_team and not any(c[5] in ("h", "a") for c in legs if c[4] != "sot"): out.append("⚠️ Unibet no tiene líneas de córners/tarjetas por equipo en este partido.")
    out.append(f"{len(legs)} piernas · cuota aprox. <b>{odds:.1f}</b> (producto) · acierta el <b>{p:.1%}</b>.")
    if odds < target and not by_team and not shooters:
        out.append(f"⚠️ Con los mercados de córners y tarjetas de este partido llego a {odds:.1f}, no a {target:g}. "
                   "Para cuota más alta hay que juntar varios partidos («combinada a cuota 80»).")
    out.append("ℹ️ En «crear apuesta» Unibet calcula su propia cuota (suele pagar algo menos que el producto): compárala.")
    out.append(f"{'🟢' if ev > 0 else '🔴'} Según el modelo, a la larga {'ganas' if ev > 0 else 'pierdes'} {abs(ev):.0%}.")
    return "\n".join(out)


def long_combo(D, target, hours=48, sports=None, only_stats=False, value_only=False, exclude=()):
    """Combinada de cuota alta (p. ej. 80) con cuotas REALES de Unibet (Kambi): de cada partido de las próximas 48 h, la
    línea de córners o tarjetas totales con más ventaja (prob. del modelo × cuota − 1 ≥ 0); se añaden, de más a menos
    ventaja, hasta que el producto de cuotas llega a `target`. Entre partidos distintos la casa paga el producto exacto."""
    import kambi
    from concurrent.futures import ThreadPoolExecutor
    matches = []
    for code, evs in (upcoming(DAY_LEAGUES, hours) if sports is None or "futbol" in sports else {}).items():
        for t, home, away, _ in evs:
            kh, ka = tkey("futbol", home, D), tkey("futbol", away, D)
            if kh and ka: matches.append((t, home, away, kh, ka, code))
    def get(m):
        try: return m, kambi.offers(m[1], m[2], m[0], stats=("corners", "cards"))
        except Exception as ex: print("Kambi", m[1], m[2], repr(ex)); return m, {}
    with ThreadPoolExecutor(6) as ex: got = list(ex.map(get, matches))
    cands, priced = [], 0
    meta = {}
    for (t, home, away, kh, ka, code), off in got:
        if not off: continue
        priced += 1; best = None
        for stat in ("corners", "cards"):
            lines = off.get((stat, "t")) or {}
            mm = match_model(kh, ka, D, "corners" if stat == "corners" else "tarjetas")
            if not mm or not lines: continue
            for line, o in lines.items():
                po = nb_over(mm[0] + mm[1], mm[2], line)
                mo = (1 / o["o"]) / (1 / o["o"] + 1 / o["u"])  # probabilidad de la casa sin margen
                for side, pmod, pmk, odd in (("Más", po, mo, o["o"]), ("Menos", 1 - po, 1 - mo, o["u"])):
                    p = W_LONG * pmod + (1 - W_LONG) * pmk
                    if not (LONG_P[0] <= p <= LONG_P[1]): continue
                    leg = (p * odd - 1, p, odd, f"{side} de {line:g} {LABEL['corners' if stat == 'corners' else 'tarjetas']}")
                    if best is None or leg[0] > best[0]: best = leg
        if best:
            loc = t.tz_convert(TZ); m = f"{A.esc(home)} – {A.esc(away)} · {DIAS[loc.dayofweek]} {loc:%H:%M}"
            meta[m] = (code, str(t), home, away)
            cands.append((m, t, *best))
    if not only_stats and (sports is None or "futbol" in sports):  # 1X2 de Unibet frente al precio justo de Pinnacle
        try:
            import valor_pinnacle
            for v in valor_pinnacle.candidates(min_edge=-0.02, max_odds=4.0):
                tt = pd_ts(v["start"]); loc = tt.tz_convert(TZ); hh, aa = v["match"].split(" vs ", 1)
                m = f"⚽ {A.esc(hh)} – {A.esc(aa)} · {DIAS[loc.dayofweek]} {loc:%H:%M}"
                if any(x[0].split(" · ")[0].endswith(f"{A.esc(hh)} – {A.esc(aa)}") for x in cands): continue
                meta[m] = (None, str(tt), hh, aa)
                txt = "Empate" if v["market"] == "D" else f"Gana {v['pick']}"
                cands.append((m, tt, v["edge"], v["p"], v["odds"], txt)); priced += 1
        except Exception as ex: print("combinada: 1X2 Pinnacle", repr(ex)[:120])
    for l in ([] if only_stats else other_legs(hours, sports)):  # ganador en NBA/NHL/esports/FC 27 con la cuota guardada por el bot
        if not l["odds"]: continue
        import multideporte
        pm = multideporte.pinnacle_fair(l) or min(0.99, 1 / l["odds"] / 1.05)  # Pinnacle sin margen; si no, la casa − ~5 %
        q = W_LONG * l["p"] + (1 - W_LONG) * pm
        if not (LONG_P[0] <= q <= LONG_P[1]): continue
        meta[l["m"]] = l; priced += 1
        cands.append((l["m"], pd_ts(l["start"]), q * l["odds"] - 1, q, l["odds"], l["txt"]))
    print(f"combinada larga: {len(matches)} partidos, {priced} con cuotas, {len(cands)} piernas candidatas")
    value = [c for c in cands if c[2] >= LONG_EDGE]
    pool = value if len(value) >= 2 else cands  # sin ventaja suficiente: la menos mala, avisando de lo que se pierde
    legs, odds, p, kinds, games = [], 1.0, 1.0, {}, set()
    kind = lambda txt: "córners" if "córner" in txt else "tarjetas" if "tarjeta" in txt else "ganador"
    if value_only: pool = value  # barrido: solo piernas con ventaja, aunque no se llegue a la cuota objetivo
    rest = [] if value_only else sorted([c for c in cands if c not in pool], key=lambda c: -c[2])
    for c in sorted(pool, key=lambda c: -c[2]) + rest:
        if odds >= target: break  # primero las de ventaja; si no llegan, las menos malas (la ganancia esperada lo refleja)
        mt = meta.get(c[0])
        if isinstance(mt, dict): hh, aa = mt.get("home", ""), mt.get("away", "")
        elif mt: hh, aa = mt[2], mt[3]
        else: hh, _, aa = re.sub(r"^\W+\s*", "", c[0].split(" · ")[0]).partition(" – ")
        g = (norm(hh)[:6], norm(aa)[:6])  # mismo partido aunque Pinnacle y ESPN escriban distinto los nombres
        if g in exclude: continue  # partido ya usado en otra combinada pendiente (barridos: no repetir piernas)
        if c[3] < COMBO_MIN_LEG or kinds.get(kind(c[5]), 0) >= 3 or g in games: continue  # con cabeza: nada de longshots
        legs.append(c); odds *= c[4]; p *= c[3]; kinds[kind(c[5])] = kinds.get(kind(c[5]), 0) + 1; games.add(g)
    if len(legs) < 2:
        return (f"Hoy no tengo cuotas de Unibet suficientes para una combinada grande ({priced} partidos con cuotas "
                f"en las próximas {hours} h; los de mañana suelen salir por la noche).")
    legs.sort(key=lambda c: c[1]); ev = p * odds - 1
    remember(f"Combinada cuota {odds:.1f}", [{**_leg(meta[m], txt, q), "odds": o} for m, _, e, q, o, txt in legs], odds, p)
    lines = [f"🎰 <b>Combinada con cabeza · objetivo {target:g}</b> (cuotas reales; piernas ≥ 40 %, una por partido, mercados variados)"]
    n_bad = sum(x[2] < LONG_EDGE for x in legs)
    if n_bad: lines.append(f"⚠️ {n_bad} de las {len(legs)} piernas no tienen ventaja clara frente a la casa: son las menos malas para llegar a la cuota.")
    for m, _, e, q, o, txt in legs: lines.append(f"<b>{m}</b>\n  • {txt} @ {o:.2f} ({q:.0%}, ventaja {e:+.0%})")
    lines.append(f"{len(legs)} piernas · cuota <b>{odds:.1f}</b>: 1 € → {odds:.0f} € · acierta el <b>{p:.1%}</b> de las veces.")
    if odds < target: lines.append(f"⚠️ Con piernas con ventaja solo llego a cuota {odds:.1f}.")
    lines.append(f"{'🟢' if ev > 0 else '🔴'} Según el modelo, a la larga {'ganas' if ev > 0 else 'pierdes'} {abs(ev):.0%} de lo apostado. "
                 "Es lotería: apuesta solo lo que puedas perder (1 €).")
    return "\n".join(lines)


def other_winner(q, hours=72):
    """«¿Quién gana NAVI – G2?» en esports, FC 27, NBA, NHL…: la probabilidad del modelo del bot para ese partido."""
    import multideporte
    try: L = multideporte.legs(hours)
    except Exception as ex: print("multideporte", repr(ex)); return None
    t = f" {norm(q)} "
    hit = [l for l in L if (f" {norm(l['home'])} " in t or f" {norm(l['away'])} " in t) and l["src"] != "espn:nba" or
           (f" {norm(l['home'])} " in t and f" {norm(l['away'])} " in t)]
    if not hit: return None
    l = min(hit, key=lambda l: l["start"])
    loc = pd_ts(l["start"]).tz_convert(TZ)
    fav = l["txt"].replace("Gana ", "")
    out = [f"⚔️ {l['sport']} <b>{A.esc(l['home'])} – {A.esc(l['away'])}</b> · {DIAS[loc.dayofweek]} {loc:%d/%m %H:%M}",
           f"👉 Favorito: <b>{A.esc(fav)}</b> ({l['p']:.0%}, cuota justa {1 / l['p']:.2f})" if l["txt"].startswith("Gana") else
           f"👉 Lo más probable: empate ({l['p']:.0%})"]
    if (pin := multideporte.pinnacle_fair(l)): out.append(f"📈 Pinnacle (la casa más afinada) le da {pin:.0%}")
    if l.get("odds"): out.append(verdict(l["p"], l["odds"]))
    return "\n".join(out)


def slip_leg(leg, D):
    """(probabilidad según el bot, texto de la pierna, dict para simular o None) de una pierna leída de un boleto."""
    tipo, txt, eq = leg.get("tipo"), (leg.get("texto") or "").split("\n")[0], leg.get("equipo") or ""
    teams = [x.strip() for x in re.split(r"\s+(?:-|–|vs\.?|v)\s+", leg.get("partido") or "") if x.strip()]
    ks = find(" ".join(teams), D, n=2, kind="equipo") if teams else []
    lt = low(txt)
    if tipo in ("otro", None):  # respaldo si la IA no lo clasificó: «anotará», «3+ remates a puerta»
        if re.search(r"anotar|marcar|goleador", lt): leg = {**leg, "tipo": "mas_menos", "estadistica": "goles", "lado": "mas", "linea": 0.5}
        elif m := re.search(r"(\d+)\s*\+\s*(faltas recibidas|remates a puerta|tiros a puerta|remates|tiros|corners|tarjetas)", lt):
            leg = {**leg, "tipo": "mas_menos", "estadistica": m.group(2).replace("remates", "tiros"), "lado": "mas", "linea": int(m.group(1)) - 0.5}
        tipo = leg.get("tipo")
    if tipo == "comparacion" and len(ks) == 2:  # «Mayor número de córners: Real Madrid»
        kh, ka = fixture_order(ks[0], ks[1], D)
        stat = "tarjetas" if "tarjeta" in low(leg.get("estadistica") or txt) else "corners"
        mm = match_model(kh, ka, D, stat)
        if mm:
            pmf = lambda mu: SS.nbinom.pmf(np.arange(40), mm[2], mm[2] / (mm[2] + mu))
            ph, pa = pmf(mm[0]), pmf(mm[1])
            p_home = float(sum(ph[i] * pa[:i].sum() for i in range(40)))
            home_pick = find_team_name(eq, (D[kh]["name"],)) if eq else True
            p = p_home if home_pick else float(sum(pa[i] * ph[:i].sum() for i in range(40)))
            return p, f"{A.esc(txt)}: <b>{p:.0%}</b> según el bot (esperados {mm[0]:.1f} – {mm[1]:.1f}; empate = perdida)", None
    if tipo == "mas_menos" and leg.get("linea") is not None:
        side = "más de" if (leg.get("lado") or "mas") == "mas" else "menos de"
        frase = f"{eq or ' '.join(teams)} {side} {leg['linea']} {(leg.get('estadistica') or '').replace('remates', 'tiros')}"
        pl = parse_leg(frase); keys = find(frase, D, kind="equipo" if pl["stat"] == "corners" else None)
        if not eq and len(ks) == 2: keys = ks  # total del partido
        if pl["stat"] and pl["line"] is not None and keys:
            p, t = leg_prob(keys, pl, D)
            word = f"{'Más' if pl['over'] else 'Menos'} de {pl['line']:g} {LABEL[pl['stat']]}"
            player = D[keys[0]]["kind"] == "jugador" and len(keys) == 1
            sim = (dict(code=None, start=None, home=D[ks[0]]["name"], away=D[ks[1]]["name"], p=p, odds=leg.get("cuota"),
                        txt=word + ("" if len(keys) == 2 or player else f" de {D[keys[0]]['name']}"),
                        **({"jugador": D[keys[0]]["name"]} if player else {}))
                   if len(ks) == 2 and (player or pl["stat"] in ("corners", "tarjetas", "goles")) else None)  # simulable aunque falte p
            return p, t, sim
    if tipo in ("ganador", "empate", "doble_oportunidad"):
        import multideporte
        try: L = multideporte.legs(96)
        except Exception: L = []
        for l in L:  # NBA, NHL, esports, FC 27: modelos del bot
            if teams and all(find_team_name(x, (l["home"], l["away"])) for x in teams[:2]):
                pfav = l["p"]; fav = l["txt"].replace("Gana ", "")
                p = pfav if (eq and find_team_name(eq, (fav,))) else 1 - pfav if tipo == "ganador" else None
                if p is None: break
                return p, f"{A.esc(txt)}: <b>{p:.0%}</b> según el modelo del bot", dict(src=l["src"], side=l["side"] if p == pfav else
                         ("A" if l["side"] == "H" else "H"), start=l["start"], home=l["home"], away=l["away"], txt=f"Gana {eq or fav}",
                         odds=leg.get("cuota"), p=p)
        if len(ks) == 2 and D[ks[0]]["sport"] == "futbol":  # fútbol: Elo europeo del bot o el 1X2 del chat (goles + Dixon-Coles)
            P = multideporte.euro_probs(D[ks[0]]["name"], D[ks[1]]["name"])
            m = None if P else exp_pair(ks[0], ks[1], D, "goles")
            if m:
                P = A.probs(max(m[0] * HFA["futbol"], 0.05), max(m[1] / HFA["futbol"], 0.05)); t = P["H"] + P["D"] + P["A"]
                P = {k: P[k] / t for k in ("H", "D", "A")}
            if P:
                h, a = D[ks[0]]["name"], D[ks[1]]["name"]
                if tipo == "empate": p, tx = P["D"], "Empate"
                elif tipo == "doble_oportunidad":
                    d = (leg.get("doble") or "1X").upper(); p = sum(P[{"1": "H", "X": "D", "2": "A"}[c]] for c in d if c in "1X2")
                    tx = f"{h if d == '1X' else a} o empate ({d})" if d in ("1X", "X2") else None
                else:
                    home_side = find_team_name(eq, (h,)) if eq else True
                    p, tx = (P["H"], f"Gana {h}") if home_side else (P["A"], f"Gana {a}")
                sim = dict(code=None, start=None, home=h, away=a, txt=tx, odds=leg.get("cuota"), p=p) if tx else None
                return p, f"{A.esc(txt)}: <b>{p:.0%}</b> según el bot", sim
    return None, f"{A.esc(txt)}: no la puedo evaluar con mis datos", None


def find_team_name(x, names):
    nx = norm(x)
    return any(nx and (nx == norm(n) or nx in norm(n) or norm(n) in nx) for n in names)


def analyze_slip(slip, D=None):
    """Análisis del boleto leído de una imagen: probabilidad de cada pierna con los modelos del bot, total y opinión."""
    D = D or db()
    rows, ps, sims = [], [], []
    for leg in slip.get("piernas") or []:
        p, t, sim = slip_leg(leg, D)
        o = leg.get("cuota")
        mark = "" if p is None or not o else (" 🟢" if p * o - 1 >= 0.03 else " 🔴" if p * o - 1 < -0.05 else " 🟡")
        rows.append(f"• {t}" + (f" · cuota {o:g}" if o else "") + mark); ps.append(p); sims.append(sim)
    total = slip.get("cuota_total") or (float(np.prod([l.get("cuota") for l in slip["piernas"]])) if all(l.get("cuota") for l in slip["piernas"]) else None)
    out = ["🧾 <b>Tu boleto</b> (leído de la imagen)", *rows]
    known = [p for p in ps if p is not None]
    games = [l.get("partido") for l in slip["piernas"]]
    same = max((games.count(g) for g in set(games)), default=0)
    if same >= 3:  # «crear apuesta»: las piernas de un partido van muy ligadas (si Mbappé marca, gana el Madrid más a menudo)
        out.append(f"ℹ️ {same} piernas del mismo partido van muy ligadas: multiplicarlas no sirve y la cuota de la casa"
                   + (f" ({total:g})" if total else "") + " ya lo tiene en cuenta. Mira pierna a pierna: las 🟢 son las que la casa paga por encima de lo que valen.")
    elif known and len(known) == len(ps):
        p = float(np.prod(known)); out.append(f"Todas a la vez: <b>{p:.1%}</b> (cuota justa {1 / p:.2f})")
        if same == 2: out.append("ℹ️ Hay piernas del mismo partido: van algo ligadas y el cálculo conjunto es aproximado.")
        if total: out.append(verdict(p, total))
    elif known: out.append(f"Puedo evaluar {len(known)} de {len(ps)} piernas; el resto no lo cubren mis datos.")
    ok_sims = [s for s in sims if s is not None]
    if ok_sims:
        remember("Tu boleto", ok_sims, total, float(np.prod(known)) if known else None)
        nx = len(sims) - len(ok_sims)
        extra = (f" (sin {'la pierna' if nx == 1 else f'las {nx} piernas'} que no puedo liquidar)") if nx else ""
        out.append(f"\n❓ ¿Te la simulo con 1 € y te aviso cuando acaben{extra}?"); _CTX["oferta"] = "simula la última combinada"
    return "\n".join(out)


SWEEP_TARGET, SWEEP_MIN_ODDS, SWEEP_MIN_EV = 16.0, 5.0, 0.05  # solo piernas con ventaja: cuota ≥ 5 (no se rellena con malas)


def fc_table(days=None):
    """«Tabla de ganadas y perdidas de FC 27»: pronósticos ya jugados con cuota real (1 € cada uno), total, por día y
    con la regla actual (ventaja ≥ 5 %): ganadas, perdidas, beneficio y ROI."""
    import esoccer as E, pandas as pd
    try: d = E.simulation()
    except Exception as ex: return f"No puedo leer el historial de FC 27 ahora ({A.esc(type(ex).__name__)})."
    if d is None or d.empty or "g_unibet" not in d: return "Aún no hay pronósticos de FC 27 jugados."
    d = d[d.g_unibet.notna()].copy()
    if d.empty: return "Aún no hay pronósticos de FC 27 con cuota real jugados."
    d["edge"] = d.p_pick * d.odds - 1
    d["dia"] = pd.to_datetime(d.start, utc=True).dt.tz_convert(TZ).dt.strftime("%d/%m")
    row = lambda x: (len(x), int(x.ok_1x2.sum()), int((~x.ok_1x2.astype(bool)).sum()), float(x.g_unibet.sum()))
    eur = lambda v: f"{'+' if v >= 0 else '−'}{abs(v):.2f} €".replace(".", ",")
    icon = lambda v: "🟢" if v > 0 else "🔴" if v < 0 else "⚪"
    per10 = lambda r: f"de cada 10 € apostados {'ganas' if r[3] >= 0 else 'pierdes'} {abs(r[3] / r[0] * 10):.2f} €".replace(".", ",")
    out = ["🎮⚽ <b>FC 27 · cómo van los pronósticos</b>", "<i>(como si hubieras apostado 1 € a cada uno, con la cuota real de la casa)</i>", ""]
    for dia, x in d.groupby("dia", sort=False):
        r = row(x)
        out.append(f"📅 <b>{dia}</b> · {r[0]} apuestas · ✅ {r[1]} ganadas · ❌ {r[2]} perdidas → {icon(r[3])} <b>{eur(r[3])}</b>")
    tot, rule = row(d), row(d[d.edge >= E.MIN_EDGE_SEND])
    xs = d[(d.edge >= E.MIN_EDGE_SEND) & (d.pick.astype(str) == "X")]
    out += ["", f"💰 <b>En total: {eur(tot[3])}</b> {icon(tot[3])}",
            f"{tot[0]} apuestas: ✅ {tot[1]} ganadas y ❌ {tot[2]} perdidas. Es decir, {per10(tot)}.",
            "", f"🎯 Solo las que el bot ve claras (ventaja de 5 % o más): <b>{eur(rule[3])}</b> {icon(rule[3])} en {rule[0]} apuestas ({per10(rule)})."]
    if len(xs):
        r = row(xs)
        out.append(f"🤝 Empates con ventaja: <b>{eur(r[3])}</b> {icon(r[3])} en {r[0]} apuestas "
                   f"(✅ {r[1]} · ❌ {r[2]}; {per10(r)}).")
    new = d[(pd.to_datetime(d.start, utc=True) >= pd.Timestamp(E.RULE_SINCE)) & (d.edge >= E.MIN_EDGE_SEND)
            & d.pick.astype(str).isin(E.SEND_PICKS or {"X"})]
    r = row(new) if len(new) else (0, 0, 0, 0.0)
    out.append(f"🆕 <b>Desde la regla actual</b> ({E.RULE_LABEL}, partidos nuevos): {r[0]} apuestas · ✅ {r[1]} · ❌ {r[2]} → "
               f"{icon(r[3])} <b>{eur(r[3])}</b>" + (" (aún sin partidos jugados)" if not r[0] else "") + ".")
    import json as _j
    try:
        prm = _j.loads((E.DIR / "params.json").read_text())
        base, now_ = (prm.get("calib_test") or {}).get("ll"), min(v for v in ((prm.get("club_test") or {}).get("ll_club") if prm.get("club_w") else None,
            (prm.get("h2h_test") or {}).get("ll_h2h") if prm.get("h2h") else None, (prm.get("draw_test") or {}).get("ll_draw"),
            (prm.get("calib_test") or {}).get("ll_cal"), 9) if v)
        if base and now_:
            out.append(f"🧠 <b>Entrenamiento</b>: error del modelo en partidos que no ha visto {now_:.4f} (sin las mejoras {base:.4f}). "
                       f"Más bajo = acierta más; cada 30 min intenta bajarlo y solo guarda si mejora.")
    except (OSError, ValueError): pass
    if not E.SEND_PICKS:
        out.append("⛔ Ahora mismo el bot <b>no manda</b> FC 27: ninguna regla ha ganado con partidos nuevos. "
                   "Sigue entrenando y midiendo; volverá cuando una gane de verdad.")
    out += [
            "", "⚠️ Es simulado, sin dinero real. Aún son pocos días: puede ser suerte, para bien o para mal."]
    return "\n".join(out)


def sweep():
    """Barrido automático (menu.py cada pocas horas): combinada con cabeza a cuota ~16 con TODAS las piernas con ventaja;
    si la hay y no es repetida, se apunta como simulada y se devuelve el mensaje para Telegram; si no, None."""
    keep = _CTX.get("ultima")
    try:
        import simulaciones
        used = {(norm(l["home"])[:6], norm(l["away"])[:6]) for b in simulaciones.load().values()
                if b["estado"] == "pendiente" and b["titulo"].startswith("Barrido") for l in b["legs"]}
        text = long_combo(db(), SWEEP_TARGET, value_only=True, exclude=used)
        u = _CTX.get("ultima")
        if not u or u is keep or not u.get("odds") or u["odds"] < SWEEP_MIN_ODDS: return None
        if any(l.get("p") is None or l["p"] * (l.get("odds") or 0) - 1 < LONG_EDGE for l in u["legs"]): return None
        if (u["p"] or 0) * u["odds"] - 1 < SWEEP_MIN_EV: return None
        import simulaciones
        b = simulaciones.record({**u, "titulo": f"Barrido · {u['titulo']}"})
        if b.get("duplicada"): return None
        return "🔎 <b>Barrido del bot</b>: he encontrado una combinada buena\n" + text + "\n🧾 La simulo con 1 € y te digo cómo acaba."
    finally:
        if keep is not None: _CTX["ultima"] = keep
        else: _CTX.pop("ultima", None)


def winner(kh, ka, D, odds=None, pick=None):
    h, a, sp = D[kh], D[ka], D[kh]["sport"]
    stat = "puntos" if sp == "nba" else "goles"
    import multideporte
    eu = multideporte.euro_probs(h["name"], a["name"]) if sp == "futbol" else None
    m = None if eu else exp_pair(kh, ka, D, stat)
    if not (m or eu): return f"Pocos datos de {A.esc(h['name'])} o {A.esc(a['name'])} para opinar."
    if eu: P = {k: eu[k] for k in ("H", "D", "A")}  # partido europeo: Elo de Champions/Europa League del bot
    elif sp == "nba":
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


# ── conversación: memoria corta, respuestas a «sí/no» y el intérprete cuando las reglas no llegan ───────────────────
YES_RX = r"^(si|sip|vale|venga|ok|okey|dale|claro|por favor|eso|ese|esa|adelante|hazlo|perfecto|de acuerdo|va)\b"
NO_RX = r"^(no|nop|nada|deja|dejalo|olvidalo|otro|otra)\b"


def offer(question, command):
    """Pregunta que el bot deja hecha: si contestas «sí», ejecuta `command`."""
    _CTX["oferta"] = command
    return f"\n\n❓ {question}"


def _plain(html_text): return re.sub(r"<[^>]+>", "", html_text).replace("\n", " ")[:300]


QUOTE_RX = r"simulad|boleto|combinada|cuota|@"


def answer(text, depth=0, quoted=None):
    """Respuesta HTML a un mensaje libre del dueño (escrito o de voz). `quoted`: texto del mensaje al que responde
    en Telegram («me refiero a esta», «¿cómo va esta?» citando un boleto) → se tiene en cuenta como contexto."""
    t = text.strip()
    if quoted and depth == 0:
        ql = low(quoted)
        if re.search(QUOTE_RX, ql) and not re.search(SIM_RX, low(t)) and (len(t.split()) <= 8 or re.search(SIMS_RX, low(t))):
            import simulaciones
            r = simulaciones.live(filtro=quoted)
            if r:
                _CTX.setdefault("hist", []).extend([("usuario", t), ("bot", _plain(r))])
                return r
        t = f"{t} (sobre: {quoted.strip()[:300]})"
    hist = _CTX.setdefault("hist", [])
    q = low(t)
    pend = _CTX.pop("oferta", None)
    if pend and re.match(YES_RX, q) and len(q.split()) <= 4: reply = answer(pend, depth + 1)
    elif pend and re.match(NO_RX, q) and len(q.split()) <= 3: reply = "Vale 👍 ¿Qué quieres mirar entonces?"
    else:
        try: reply = _answer(t, short=len(t.split()) <= 5 or depth > 0)
        except Exception as e: print("chat reglas", repr(e)); reply = None
        if reply is None and depth == 0:
            import interpreta
            r = interpreta.route(t, hist + ([("bot", f"(ofrecía: {pend})")] if pend else []))
            if r and r["orden"] and low(r["orden"]) != q: reply = answer(r["orden"], depth + 1)
            elif r and r["respuesta"]: reply = A.esc(r["respuesta"])
        if reply is None:
            reply = _fallback(t)
    if depth == 0:
        reply = natural(t, reply, hist)
        hist += [("usuario", t), ("bot", _plain(reply))]; del hist[:-12]
    return reply


def natural(user_text, reply, hist=()):
    """La IA (interpreta.redactar) cuenta con palabras naturales lo que ha calculado el bot; si no hay IA, falla o mete un
    número que el bot no dio, se manda la respuesta del bot tal cual."""
    if reply == HELP or reply.startswith(("📡", "🧾", "🎮⚽")): return reply  # directo de apuestas: con sus iconos tal cual
    if sum(l.lstrip().startswith(("•", "↳")) for l in reply.split("\n")) >= 2: return reply  # listas: tal cual (la IA las estropeaba)
    import html, interpreta
    report = html.unescape(re.sub(r"<[^>]+>", "", reply))
    try: out = interpreta.redactar(user_text, report, hist)
    except Exception as e: print("redactor", repr(e)); out = None
    return A.esc(out) if out else reply


def _fallback(t):
    """Sin entender la frase: mejor proponer algo concreto que un «no te entiendo»."""
    D = db()
    if _CTX.get("partido") and all(k in D for k in _CTX["partido"]):
        h, a = (D[k]["name"] for k in _CTX["partido"])
        return (f"No sé si te he entendido bien 🤔 ¿Hablas de <b>{A.esc(h)} – {A.esc(a)}</b>? Te puedo decir quién gana, "
                f"las estadísticas o armarte la combinada segura." + offer("¿Te armo la combinada segura?", f"combinada {h} {a}"))
    keys = find(t, D, n=1)
    if keys: return stats_card(keys[0], D)
    return ("No sé si te he entendido bien 🤔 Si me nombras un equipo o un jugador te digo sus números, quién gana o "
            "una combinada segura." + offer("¿Te saco la combinada segura del día?", "combinada del día"))


def _answer(t, short=True):
    """Las reglas (sin IA). None si no entienden la frase."""
    if not t or re.match(r"^/?(ayuda|help|hola|que puedes)", norm(t)): return HELP
    D = db()
    q = low(t)
    if re.search(r"(mas|mayor) posibilidades|cual (tiene|es) (mas|mejor)|compara", q) and re.search(r"apuesta|simulad|combinad|boleto", q):
        import simulaciones
        ks = find(t, D, n=1, kind="equipo")
        return simulaciones.compare(D[ks[0]]["name"] if ks else None)
    if re.search(r"fc ?2[67]|esoccer|e-?soccer|fifa", q) and re.search(r"tabla|ganad|perdid|balance|cuanto|resultado|como va|historial|lleva", q):
        return fc_table()
    if re.search(r"(ha|han|ya) (finalizado|terminado|acabado)|(finalizo|termino|acabo) (el|los) partido|ya (acabo|termino|finalizo)", q):
        import simulaciones
        done = simulaciones.settle()
        if done: return "\n\n".join(done)
        return (simulaciones.live(filtro=t) or "") + "\n\nESPN aún no lo da por terminado; en cuanto lo haga te mando el resultado 👍"
    if re.search(SIMS_RX, q) or re.search(r"directo|en juego|jugando|ahora|marcador", q) and re.search(r"apuesta|combinad|simulad|boleto", q):
        import simulaciones
        return simulaciones.live(filtro=t) or simulaciones.summary()
    if re.search(SIM_RX, q):
        if not last_combo(): return "Primero pídeme una combinada (o mándame la foto del boleto) y luego te la simulo."
        import simulaciones
        b = simulaciones.record(_CTX.pop("ultima")); LAST_FILE.unlink(missing_ok=True)
        if b.get("duplicada"): return "👍 Esa ya la tenías apuntada; te aviso cuando acaben los partidos."
        import boleto as BO
        legs = [(f"{l['home']} – {l['away']}", (f"{l['jugador']}: " if l.get("jugador") else "") + l["txt"]
                 + (f" @ {l['odds']:.2f}" if l.get("odds") else ""), None) for l in b["legs"]]
        return "\n".join(["✅ <b>Apuntada</b> (simulada): " + A.esc(b["titulo"]), *BO.boleto(None, legs, b.get("odds"), b.get("stake")),
                          "Cuando acaben los partidos te digo si hemos ganado o perdido, pierna a pierna y por qué."])
    split = re.search(r"cada equipo|por equipos?|separ", q)
    shoot = re.search(r"tiros? a puerta de (?:los )?(?:delanteros|jugadores|rematadores|atacantes)", q)
    if (split or shoot) and (_CTX.get("crear") or len(find(t, D, n=2, kind="equipo")) == 2):  # orden sobre «crear apuesta»
        ks = find(t, D, n=2, kind="equipo")
        kh, ka, tg = _CTX["crear"] if _CTX.get("crear") and len(ks) < 2 else (*ks, 20.0)
        sh = None
        if shoot:
            named = [k for k in ks if k in (kh, ka)]
            sh = named[-1] if named else None
        m_t = re.search(r"cuota (?:de )?(\d+(?:[.,]\d+)?)", q)
        return match_combo(D, kh, ka, _f(m_t.group(1)) if m_t else tg, by_team=bool(split), shooters=sh)
    if re.search(COMBO_RX, q) and not re.search(r"mas de|menos de|over|under|\d\+", q):
        ks = find(t, D, n=2, kind="equipo")
        if len(ks) == 2: kh, ka = ks
        elif ks:
            _CTX.pop("partido", None); next_match(ks[0], D); kh, ka = _CTX.get("partido") or (None, None)
        else: kh, ka = _CTX.get("partido") or (None, None)
        m_t = re.search(r"(?:cuota|a)\s*(?:de\s*)?(\d+(?:[.,]\d+)?)", q) or re.search(r"@\s*(\d+(?:[.,]\d+)?)", q)
        target = _f(m_t.group(1)) if m_t and 1.2 <= _f(m_t.group(1)) <= 1000 else None
        sp = sports_in(q)
        if target and target >= 10 and len(ks) == 2: return match_combo(D, kh, ka, target)  # «solo de este partido»
        stats_only = bool(re.search(r"corner|tarjeta", q)) and not re.search(r"gana|ganador|nba|esports|cs2|dota|lol|todo", q)
        if not target and re.search(r"intelig|con cabeza|estrategia|de todo|profesional", q): target = 16.0
        if target and target >= 10: return long_combo(D, target, sports=sp, only_stats=stats_only)  # «de córners y tarjetas a cuota 16»
        if target or re.search(r"del dia|de hoy|de manana|varios partidos|popurri|sub\w* la cuota|anad|mas piernas|mas cuota|lo maximo", q) \
                or sp is not None or not (kh and ka and kh in D and ka in D): return day_combo(D, target=target, sports=sp)
        _CTX["partido"] = (kh, ka)
        if D[kh]["sport"] != "futbol": return "La combinada segura de momento es solo de fútbol."
        h, a = D[kh]["name"], D[ka]["name"]
        return safe_combo(kh, ka, D) + offer("¿La junto con otros partidos para subir más la cuota?", "combinada del día")
    if re.search(WIN_RX, q) and not re.search(r"mas de|menos de|over|under|\d\+", q) and (r := other_winner(q)): return r
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
        return winner(kh, ka, D, odds_in(t), pick) + offer("¿Te armo la combinada segura de este partido?",
                                                           f"combinada {D[kh]['name']} {D[ka]['name']}")
    if re.search(NEXT_RX, q) or re.search(LAST_RX, q):
        keys = find(t, D, n=1)
        if keys and re.search(NEXT_RX, q):
            out = next_match(keys[0], D); m = _CTX.get("partido")
            if m and all(m) and "Próximo partido" in out:
                out += offer("¿Quieres la combinada segura de este partido?", f"combinada {D[m[0]]['name']} {D[m[1]]['name']}")
            return out
        if keys: return last_match(keys[0], D)
    if re.search(VS_RX, q):  # «córners del AEK contra el Real Madrid», «¿hará el Athens más de 4.5 córners ante el Madrid?»
        pl = parse_leg(t); ks = find(t, D, n=2, kind="equipo")
        if pl["stat"] in ("corners", "tarjetas") and len(ks) == 2:
            return team_vs(ks[0], ks[1], pl["stat"], D, pl["line"], pl["over"])
    legs, odds = split_legs(t), odds_in(t)
    parsed = []
    for seg in legs:
        leg = parse_leg(seg)
        keys = find(seg, D, kind="equipo" if leg["stat"] == "corners" else None)
        if keys and leg["stat"] and leg["line"] is not None: parsed.append((keys, leg))
    if not parsed:
        keys = find(t, D, n=1)
        if keys and (short or not __import__("interpreta").enabled()): return stats_card(keys[0], D)
        return None
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
