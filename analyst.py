"""Analista deportivo (fútbol): historial desde 2005 de 16 ligas europeas + 11 del resto del mundo (football-data.co.uk),
modelo Poisson con decaimiento temporal ajustado por bajas reales (sportsgambler.com), mejor cuota entre casas
y escenarios completos. Genera señales que bankroll.py convierte en apuestas y tarjetas de Telegram.
Pensado para ejecutarse cada hora en GitHub Actions (ver .github/workflows/analista.yml)."""
import difflib, html, io, json, math, os, re, time, unicodedata
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
HIST, NEW, STATE = DATA / "hist", DATA / "new", DATA / "state.json"
BASE = "https://www.football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0"}
LEAGUES = {"E0": "🏴 Premier League", "E1": "🏴 Championship", "SP1": "🇪🇸 LaLiga", "SP2": "🇪🇸 LaLiga 2",
           "I1": "🇮🇹 Serie A", "I2": "🇮🇹 Serie B", "D1": "🇩🇪 Bundesliga", "D2": "🇩🇪 2. Bundesliga",
           "F1": "🇫🇷 Ligue 1", "F2": "🇫🇷 Ligue 2", "N1": "🇳🇱 Eredivisie", "P1": "🇵🇹 Primeira Liga",
           "B1": "🇧🇪 Pro League", "T1": "🇹🇷 Süper Lig", "G1": "🇬🇷 Super League", "SC0": "🏴 Premiership escocesa"}
NEW_LEAGUES = {"ARG": ("Argentina", "🇦🇷 Liga Profesional"), "BRA": ("Brazil", "🇧🇷 Brasileirão"),
               "MEX": ("Mexico", "🇲🇽 Liga MX"), "USA": ("USA", "🇺🇸 MLS"), "JPN": ("Japan", "🇯🇵 J1 League"),
               "AUT": ("Austria", "🇦🇹 Bundesliga"), "DNK": ("Denmark", "🇩🇰 Superliga"), "NOR": ("Norway", "🇳🇴 Eliteserien"),
               "SWE": ("Sweden", "🇸🇪 Allsvenskan"), "SWZ": ("Switzerland", "🇨🇭 Super League"), "POL": ("Poland", "🇵🇱 Ekstraklasa")}
ALL_LEAGUES = {**LEAGUES, **{c: n for c, (_, n) in NEW_LEAGUES.items()}}
FIRST_SEASON = 2005
_P = json.loads((DATA / "football_params.json").read_text()) if (DATA / "football_params.json").exists() else {}  # lo ajusta train_football.py
EDGE, MAX_EDGE, MIN_P = _P.get("edge", 0.05), 0.30, 0.25  # señal si ventaja entre EDGE y 30% (más suele ser cuota errónea) y prob >= 25%
BLEND = _P.get("blend", 0.5)  # peso del modelo frente al mercado sin margen
HALF_LIFE, SHRINK = _P.get("half_life", 365), _P.get("k", 5.0)  # decaimiento (días) y encogimiento hacia la media de la liga
HORIZON_DAYS = 2                  # partidos de los próximos 2 días
TG, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
GEMINI, MODEL = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-flash-latest")
CLOSE = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA", "O25": "AvgC>2.5", "U25": "AvgC<2.5"}  # cuotas de cierre → CLV
PINNACLE = {"H": "PSH", "D": "PSD", "A": "PSA", "O25": "P>2.5", "U25": "P<2.5"}  # la casa más afinada: ancla del mercado
COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *CLOSE.values()]
RHO = _P.get("rho", -0.10)  # Dixon-Coles: corrige la frecuencia de 0-0, 1-0, 0-1 y 1-1
BOOKIES = {"B365": "Bet365", "PS": "Pinnacle", "BW": "Bwin", "BV": "BetVictor", "BFD": "Betfred", "PP": "Paddy Power",
           "SKB": "Sky Bet", "WH": "William Hill"}  # sin Betfair Exchange: comisión y poca liquidez
SUFFIX = {"H": "H", "D": "D", "A": "A", "O25": ">2.5", "U25": "<2.5"}
NAMES = {"H": "Local", "D": "Empate", "A": "Visitante", "O25": "Más de 2.5 goles", "U25": "Menos de 2.5 goles",
         "AHH": "Hándicap local", "AHA": "Hándicap visitante"}
AH_BOOKS = {"B365": "Bet365", "P": "Pinnacle", "Max": "mejor del mercado"}  # columnas <casa>AHH / <casa>AHA
POS_WEIGHT = {"F": ("att", 0.10), "M": ("att", 0.06), "D": ("def", 0.06), "G": ("def", 0.10)}
STATUS_WEIGHT = {"lesionado": 1.0, "sancionado": 1.0, "duda": 0.5}


def season_code(y): return f"{y % 100:02d}{(y + 1) % 100:02d}"
def key(r): return f"{r['Div']}|{r['Date']:%Y-%m-%d}|{r['HomeTeam']}|{r['AwayTeam']}"
def esc(s): return html.escape(str(s), quote=False)


def load_csv(raw, cols=COLS):
    try: txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError: txt = raw.decode("latin-1")  # temporadas antiguas
    df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip", usecols=(lambda c: c in cols) if cols else None)
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def fetch(url):
    try:
        r = requests.get(url, headers=UA, timeout=60)
        return r.content if r.ok and r.content else None
    except requests.RequestException as e:
        print("descarga fallida", url, e)


def update_history(today):
    HIST.mkdir(parents=True, exist_ok=True); NEW.mkdir(parents=True, exist_ok=True)
    cur = today.year if today.month >= 7 else today.year - 1
    for div in LEAGUES:
        for y in range(FIRST_SEASON, cur + 1):
            f = HIST / f"{div}_{season_code(y)}.csv"
            if f.exists() and y < cur: continue  # temporadas cerradas: se descargan una vez
            if raw := fetch(f"{BASE}/mmz4281/{season_code(y)}/{div}.csv"): f.write_bytes(raw)
    frames = [load_csv(f.read_bytes()) for f in sorted(HIST.glob("*.csv"))]
    for code in NEW_LEAGUES:  # ligas "nuevas": un archivo con todo su historial, se baja en cada ejecución (no va a git)
        f = NEW / f"{code}.csv"
        if raw := fetch(f"{BASE}/new/{code}.csv"): f.write_bytes(raw)
        if f.exists(): frames.append(load_csv(f.read_bytes(), ["Date", "Home", "Away", "HG", "AG", "AvgCH", "AvgCD", "AvgCA"]).assign(Div=code))
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["FTHG", "FTAG"]).sort_values("Date", ignore_index=True)


def fixtures():
    fx = load_csv(fetch(f"{BASE}/fixtures.csv") or b"Div,Date,HomeTeam,AwayTeam\n", None)
    new = fetch(f"{BASE}/new_league_fixtures.csv")
    if new:
        n = load_csv(new, None)
        n["Div"] = n["Country"].map({c: code for code, (c, _) in NEW_LEAGUES.items()})
        fx = pd.concat([fx, n.dropna(subset=["Div"])], ignore_index=True)
    return fx[fx.Div.isin(list(ALL_LEAGUES))]


# ---------- modelo ----------
def strengths(df, div, today, k=SHRINK, half_life=HALF_LIFE):
    d = df[(df.Div == div) & (df.Date >= today - pd.DateOffset(years=12))]
    w = 0.5 ** ((today - d.Date).dt.days / half_life)  # la última década pesa, lo reciente más
    mu_h, mu_a = (w * d.FTHG).sum() / w.sum(), (w * d.FTAG).sum() / w.sum()
    def rate(team, goals, mu):  # media ponderada con encogimiento hacia la media de la liga
        return ((w * d[goals]).groupby(d[team]).sum() + k * mu) / (w.groupby(d[team]).sum() + k) / mu
    return mu_h, mu_a, {"att_h": rate("HomeTeam", "FTHG", mu_h), "def_h": rate("HomeTeam", "FTAG", mu_a),
                        "att_a": rate("AwayTeam", "FTAG", mu_a), "def_a": rate("AwayTeam", "FTHG", mu_h)}


def probs(lh, la, n=11):
    pois = lambda l: [math.exp(-l) * l ** i / math.factorial(i) for i in range(n)]
    ph, pa = pois(lh), pois(la)
    P = dict.fromkeys(["H", "D", "A", "O15", "O25", "U25", "O35", "BTTS"], 0.0)
    scores = {}
    for i in range(n):
        for j in range(n):
            tau = {(0, 0): 1 - lh * la * RHO, (1, 0): 1 + la * RHO, (0, 1): 1 + lh * RHO, (1, 1): 1 - RHO}.get((i, j), 1)
            p = scores[f"{i}-{j}"] = ph[i] * pa[j] * tau
            P["H" if i > j else "D" if i == j else "A"] += p
            P["O25" if i + j > 2 else "U25"] += p
            P["O15"] += p * (i + j > 1); P["O35"] += p * (i + j > 3)
            if i and j: P["BTTS"] += p
    P["scores"] = sorted(scores.items(), key=lambda x: -x[1])[:4]
    P["diff"] = {}
    for sc, p in scores.items():
        i, j = map(int, sc.split("-")); P["diff"][i - j] = P["diff"].get(i - j, 0) + p
    return P


def absence_impact(players, played):
    """Pérdida de ataque y de defensa por bajas, según posición y peso del jugador (partidos jugados)."""
    imp = {"att": 0.0, "def": 0.0}
    for p in players:
        side, w = POS_WEIGHT.get(p["pos"][:1].upper(), ("att", 0.04))
        imp[side] += w * STATUS_WEIGHT.get(p["status"], 0.5) * min(1.0, p["games"] / max(played, 1))
    return {k: min(v, 0.25) for k, v in imp.items()}  # shortcut: pesos fijos por posición; usar minutos/xG si hay fuente gratis


def predict(st, home, away, squads=([], []), played=(1, 1)):
    mu_h, mu_a, s = st
    g = lambda t, team: s[t].get(team, 1.0)  # shortcut: recién ascendidos = media de la liga
    ih, ia = absence_impact(squads[0], played[0]), absence_impact(squads[1], played[1])
    lh = mu_h * g("att_h", home) * g("def_a", away) * (1 - ih["att"]) * (1 + ia["def"])
    la = mu_a * g("att_a", away) * g("def_h", home) * (1 - ia["att"]) * (1 + ih["def"])
    return lh, la, probs(lh, la)


def best_odds(row, m):
    """(cuota, casa) más alta entre las casas del CSV; si no hay, Max/Avg del mercado."""
    offers = [(float(row[c]), name) for b, name in BOOKIES.items()
              if pd.notna(row.get(c := f"{b}{SUFFIX[m]}")) and float(row[c]) > 1]
    if offers: return max(offers)
    for c, name in ((f"Max{SUFFIX[m]}", "mejor del mercado"), (f"Avg{SUFFIX[m]}", "media del mercado")):
        if pd.notna(row.get(c)) and float(row[c]) > 1: return float(row[c]), name
    return None, None


def value_bets(P, row, every=False):
    """Mercados con ventaja (o todos los de prob >= MIN_P si every=True, para listar candidatas)."""
    out = []
    for group in (("H", "D", "A"), ("O25", "U25")):
        best = {m: best_odds(row, m) for m in group}
        if not all(o for o, _ in best.values()): continue
        pin = {m: row.get(PINNACLE[m]) for m in group}
        avg = ({m: float(pin[m]) for m in group} if all(pd.notna(v) and float(v) > 1 for v in pin.values()) else
               {m: float(row[c]) if pd.notna(row.get(c := f"Avg{SUFFIX[m]}")) else best[m][0] for m in group})
        inv = sum(1 / o for o in avg.values())
        for m in group:
            implied = 1 / avg[m] / inv  # probabilidad del mercado sin margen
            p, (o, book) = BLEND * P[m] + (1 - BLEND) * implied, best[m]  # mezcla modelo + mercado: el Poisson solo sobreestima ventajas
            if p >= MIN_P and (every or EDGE <= p * o - 1 <= MAX_EDGE):
                out.append({"market": m, "odds": o, "book": book, "p": round(p, 3), "implied": round(implied, 3),
                            "edge": round(p * o - 1, 3)})
    return out


def ah_factor(diff, line, odds):
    """Cobro por unidad en hándicap asiático; las líneas de cuarto (±0.25, ±0.75…) se parten en dos medias apuestas."""
    halves = [line - 0.25, line + 0.25] if (line * 4) % 2 == 1 else [line]
    return sum(odds if diff + h > 0 else 1.0 if diff + h == 0 else 0.0 for h in halves) / len(halves)


def ah_bets(P, row, every=False):
    h = row.get("AHh")
    if pd.isna(h): return []
    h, out = float(h), []
    best = {}
    for side in ("H", "A"):
        offers = [(float(row[c]), name) for b, name in AH_BOOKS.items() if pd.notna(row.get(c := f"{b}AH{side}")) and float(row[c]) > 1]
        if not offers: return []
        best[side] = max(offers)
    avg = {s: float(row[c]) if pd.notna(row.get(c := f"AvgAH{s}")) else best[s][0] for s in ("H", "A")}
    inv = sum(1 / o for o in avg.values())
    for side, sign, line in (("H", 1, h), ("A", -1, -h)):
        o, book = best[side]
        ev = sum(p * ah_factor(sign * d, line, o) for d, p in P["diff"].items()) - 1
        implied = 1 / avg[side] / inv
        q = BLEND * (ev + 1) / o + (1 - BLEND) * implied  # prob. equivalente (cobro esperado / cuota) mezclada con el mercado
        if q >= MIN_P and (every or EDGE <= q * o - 1 <= MAX_EDGE):
            out.append({"market": f"AH{side}", "line": line, "odds": o, "book": book, "p": round(q, 3),
                        "implied": round(implied, 3), "edge": round(q * o - 1, 3)})
    return out


def won(m, hg, ag): return {"H": hg > ag, "D": hg == ag, "A": hg < ag, "O25": hg + ag > 2, "U25": hg + ag < 3}[m]


def h2h(df, a, b):
    return df[((df.HomeTeam == a) & (df.AwayTeam == b)) | ((df.HomeTeam == b) & (df.AwayTeam == a))]


def form(df, t, n=5):
    d = df[(df.HomeTeam == t) | (df.AwayTeam == t)].tail(n)
    return "".join("➖" if r.FTHG == r.FTAG else "✅" if (r.HomeTeam == t) == (r.FTHG > r.FTAG) else "❌"
                   for r in d.itertuples())


def h2h_summary(hh, home, away):
    if hh.empty: return "sin precedentes"
    wins = lambda t: int((((hh.HomeTeam == t) & (hh.FTHG > hh.FTAG)) | ((hh.AwayTeam == t) & (hh.FTAG > hh.FTHG))).sum())
    goals = hh.FTHG + hh.FTAG
    return (f"{len(hh)} desde {hh.Date.min():%Y}: {wins(home)}V-{int((hh.FTHG == hh.FTAG).sum())}E-{wins(away)}D "
            f"· {goals.mean():.1f} goles/partido · +2.5 en {100 * (goals > 2).mean():.0f}%")


# ---------- bajas (sportsgambler.com) ----------
INJ_SLUGS = {"E0": "england-premier-league", "E1": "england-championship", "SP1": "spain-la-liga", "SP2": "spain-la-liga-2",
             "I1": "italy-serie-a", "I2": "italy-serie-b", "D1": "germany-bundesliga", "D2": "germany-2-bundesliga",
             "F1": "france-ligue-1", "F2": "france-ligue-2", "N1": "netherlands-eredivisie", "P1": "portugal-primeira-liga",
             "B1": "belgium-pro-league", "T1": "turkey-super-lig", "SC0": "scotland-premiership", "USA": "usa-mls",
             "BRA": "brazil-serie-a", "ARG": "argentina-liga-profesional", "MEX": "mexico-liga-mx"}
INJ_TYPES = {"injury-plus": "lesionado", "injury-questionmark": "duda", "redcard": "sancionado"}
ALIASES = {"paris sg": "psg", "ath madrid": "atletico madrid", "ath bilbao": "athletic club", "la coruna": "deportivo",
           "santander": "racing club", "espanol": "espanyol", "ein frankfurt": "frankfurt"}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def injuries(div):
    """{equipo: [jugadores de baja/duda]} de la liga; {} si no hay fuente o la web falla."""
    if div not in INJ_SLUGS: return {}
    raw = fetch(f"https://www.sportsgambler.com/injuries/football/{INJ_SLUGS[div]}/")
    out = {}
    for block in (raw or b"").decode("utf-8", "ignore").split('class="injuries-title"')[1:]:
        team = re.search(r'<a [^>]*>([^<]+)</a>', block)
        if not team: continue
        out[team.group(1)] = [{"name": p, "pos": pos, "games": int(g) if g.isdigit() else 0, "goals": gl,
                               "status": INJ_TYPES.get(t, "duda"), "note": info, "ret": ret}
                              for t, p, pos, g, gl, info, ret in re.findall(
                                  r'inj-type ([a-z-]+)".*?inj-player">([^<]+)<.*?inj-position[^>]*>([^<]*)<.*?inj-game[^>]*>([^<]*)<'
                                  r'.*?inj-goals[^>]*>([^<]*)<.*?inj-info">([^<]*)<.*?inj-return[^>]*>([^<]*)<', block, re.S)]
    return out


def find_team(name, teams):
    """Empareja el nombre de football-data.co.uk con el de sportsgambler."""
    n = ALIASES.get(norm(name), norm(name))
    by = {norm(t): t for t in teams}
    cands = [t for t in by if n == t or n in t.split() or n in t or t in n]
    if cands: return by[max(cands, key=lambda t: difflib.SequenceMatcher(None, n, t).ratio())]
    m = difflib.get_close_matches(n, by, n=1, cutoff=0.75)
    return by[m[0]] if m else None


def absences_line(team, players):
    out = [p for p in players if p["status"] != "duda"]
    if not players: return f"{esc(team)}: sin bajas"
    key_out = sorted(out, key=lambda p: -p["games"])[:3]
    return (f"{esc(team)}: {len(out)} fuera, {len(players) - len(out)} duda"
            + (f" ({', '.join(esc(p['name']) for p in key_out)})" if key_out else ""))


# ---------- IA ----------
def ai_json(prompt):
    if not GEMINI: return None
    from google import genai
    client = genai.Client(api_key=GEMINI)  # variable aparte: si no, se cierra antes de la petición
    try:
        txt = client.models.generate_content(model=MODEL, contents=prompt).text or ""
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else None
    except Exception as e:  # una respuesta mala de la IA no debe tumbar la ejecución
        print("IA fallida", e)
        return None
    finally:
        time.sleep(7)  # límite de peticiones por minuto del plan gratis


# ---------- Telegram ----------
def tg(method, **data):
    """Llama a la API de Telegram; devuelve el 'result' o None."""
    if not (TG and CHAT):  # sin token: modo prueba, imprime en consola
        print(f"\n[telegram {method}]\n{data.get('text', '')}")
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG}/{method}", data={"chat_id": CHAT, **data}, timeout=60)
        if not r.ok and "not modified" not in r.text: print("Telegram error", r.status_code, r.text[:200])
        return r.json().get("result") if r.ok else None
    except requests.RequestException as e: print("Telegram error", e)
    finally: time.sleep(1)


def send_card(text): tg("sendMessage", text=text[:4000], parse_mode="HTML", disable_web_page_preview="true")


# ---------- tareas ----------
def settle_signals(df, state, today):
    recent = {key(r): r for r in df[df.Date >= today - pd.Timedelta(days=30)].to_dict("records")}
    for s in state["signals"]:
        if s.get("result") is None and s["key"] in recent:
            r = recent[s["key"]]
            if s["market"] in ("AHH", "AHA"):
                d = (r["FTHG"] - r["FTAG"]) * (1 if s["market"] == "AHH" else -1)
                s["factor"] = ah_factor(d, s["line"], s["odds"])
                s["result"] = True if s["factor"] > 1 else False if s["factor"] < 1 else "nula"
            else: s["result"] = bool(won(s["market"], r["FTHG"], r["FTAG"]))
            s["score"] = f"{int(r['FTHG'])}-{int(r['FTAG'])}"
            if s["market"] in CLOSE and pd.notna(r.get(CLOSE[s["market"]])): s["clv"] = round(s["odds"] / float(r[CLOSE[s["market"]]]) - 1, 3)
        elif s.get("result") is None and s["key"].split("|")[0] in ALL_LEAGUES and s["key"].split("|")[1] < f"{today - pd.Timedelta(days=30):%Y-%m-%d}":
            s["result"] = "nula"  # aplazado o sin resultado


def analyze_fixtures(df, state, today, kg):
    fx = fixtures()
    fx = fx[(fx.Date >= today) & (fx.Date <= today + pd.Timedelta(days=HORIZON_DAYS))]
    dated = lambda k: k.split("|")[1][4:5] == "-"  # las claves de CS2 son "cs2|<id>", sin fecha: se guardan las 1000 últimas
    state["analyzed"] = ([k for k in state["analyzed"] if dated(k) and k.split("|")[1] >= f"{today - pd.Timedelta(days=7):%Y-%m-%d}"]
                         + [k for k in state["analyzed"] if not dated(k)][-1000:])
    models, inj, now = {}, {}, pd.Timestamp.now(tz="UTC")
    season = df[df.Date >= today - pd.Timedelta(days=300)]
    for row in fx.sort_values(["Date", "Time"]).to_dict("records"):
        k = key(row)
        # shortcut: hora de fixtures.csv tomada como hora de Londres (las ligas "nuevas" también la dan así)
        start = pd.Timestamp(f"{row['Date']:%Y-%m-%d} {row.get('Time') or '00:00'}", tz="Europe/London")
        if k in state["analyzed"] or start <= now: continue  # ya empezado
        div, home, away = row["Div"], row["HomeTeam"], row["AwayTeam"]
        if div not in models: models[div], inj[div] = strengths(df, div, today), injuries(div)
        squads = [inj[div].get(find_team(t, inj[div]), []) for t in (home, away)]
        played = [int(((season.HomeTeam == t) | (season.AwayTeam == t)).sum()) for t in (home, away)]
        lh, la, P = predict(models[div], home, away, squads, played)
        hh = h2h(df, home, away)
        kg.match(ALL_LEAGUES[div], home, away, start, {"1": P["H"], "X": P["D"], "2": P["A"]}, squads, h2h_summary(hh, home, away))
        state["analyzed"].append(k)
        cands = value_bets(P, row, every=True) + ah_bets(P, row, every=True)
        if not cands: continue
        vb = [v for v in cands if EDGE <= v["edge"] <= MAX_EDGE] or [max(cands, key=lambda v: v["edge"])]  # sin valor: la mejor como candidata
        fav = min(("H", "D", "A"), key=lambda m: best_odds(row, m)[0] or 99)
        pick = {"H": home, "A": away}
        info = [f"📊 <b>Escenarios</b> (goles esperados {lh:.2f} - {la:.2f})",
                f"1 {P['H']:.0%} · X {P['D']:.0%} · 2 {P['A']:.0%}",
                f"1X {P['H'] + P['D']:.0%} · X2 {P['D'] + P['A']:.0%} · 12 {P['H'] + P['A']:.0%}",
                f"+1.5 {P['O15']:.0%} · +2.5 {P['O25']:.0%} · +3.5 {P['O35']:.0%} · Ambos marcan {P['BTTS']:.0%}",
                "Marcadores: " + " · ".join(f"{s} {p:.0%}" for s, p in P["scores"]),
                "", "🔎 <b>Claves</b>",
                f"• Forma: {esc(home)} {form(df, home)} | {esc(away)} {form(df, away)}",
                f"• H2H: {h2h_summary(hh, home, away)}",
                f"• Bajas: {absences_line(home, squads[0])}; {absences_line(away, squads[1])}",
                f"• Favorito de las casas: {esc(pick.get(fav, 'Empate'))} @ {best_odds(row, fav)[0]:.2f}"]
        context = (f"{home} vs {away} ({ALL_LEAGUES[div]}). Modelo Poisson con historial desde 2005 y bajas: "
                   f"1 {P['H']:.0%} X {P['D']:.0%} 2 {P['A']:.0%}, +2.5 {P['O25']:.0%}, ambos marcan {P['BTTS']:.0%}. "
                   f"Forma {home} {form(df, home)}, {away} {form(df, away)} (✅ victoria ➖ empate ❌ derrota). "
                   f"H2H {h2h_summary(hh, home, away)}. Bajas {home}: "
                   f"{'; '.join(p['name'] + ' ' + p['pos'] + ' ' + p['status'] + ' ' + str(p['games']) + ' PJ' for p in squads[0]) or 'ninguna'}. "
                   f"Bajas {away}: {'; '.join(p['name'] + ' ' + p['pos'] + ' ' + p['status'] + ' ' + str(p['games']) + ' PJ' for p in squads[1]) or 'ninguna'}.")
        for v in vb:
            name = (f"{home if v['market'] == 'AHH' else away} {v['line']:+g} (hándicap asiático)" if v["market"] in ("AHH", "AHA")
                    else pick.get(v["market"], NAMES[v["market"]]))
            state["signals"].append({**v, "key": k, "match": f"{home} vs {away}", "pick": name,
                                     "start": start.isoformat(), "league": ALL_LEAGUES[div], "sport": "⚽",
                                     "info": info, "context": context})


def prune(state):
    """Mantiene state.json pequeño: fuera candidatas viejas no apostadas y textos de partidos ya empezados."""
    now = pd.Timestamp.now(tz="UTC")
    used = {tuple(l) for b in state.get("bank", {}).get("bets", []) for l in b["legs"]}
    keep, seen = [], set()
    for s in state["signals"]:
        if (s["key"], s["market"]) in seen: continue  # sin duplicados
        seen.add((s["key"], s["market"]))
        started = not s.get("start") or pd.Timestamp(s["start"]) < now
        if (s["key"], s["market"]) not in used and (not s.get("start") or pd.Timestamp(s["start"]) < now - pd.Timedelta(days=3)): continue
        if started: s.pop("info", None)
        keep.append(s)
    state["signals"] = keep


def main():
    import bankroll, esports, knowledge, multisport
    today = pd.Timestamp.now().normalize()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"analyzed": [], "signals": []}
    kg = knowledge.Graph()
    df = update_history(today)
    print(f"Historial fútbol: {len(df)} partidos ({df.Date.min():%Y}-{df.Date.max():%Y})")
    settle_signals(df, state, today)
    analyze_fixtures(df, state, today, kg)
    multisport.run(state, today, kg)
    try: esports.run(state, today, kg)
    except Exception as e: print("fallo en cs2", repr(e))  # los esports no deben parar el resto
    bankroll.run(state, today, kg)
    kg.save()
    prune(state)
    state.pop("seen", None)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
