"""Analista deportivo: historial desde 2005 (football-data.co.uk), modelo Poisson con decaimiento temporal,
IA (Gemini + Google Search) para bajas/noticias, grafo de jugadores, PDF y envío por Telegram.
Pensado para ejecutarse cada hora en GitHub Actions (ver .github/workflows/analista.yml)."""
import difflib, io, json, math, os, re, tempfile, time, unicodedata
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
HIST, STATE = DATA / "hist", DATA / "state.json"
BASE = "https://www.football-data.co.uk"
LEAGUES = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
FIRST_SEASON = 2005
EDGE, MIN_P = 0.05, 0.25          # señal si prob*cuota-1 >= 5% y prob >= 25%
HORIZON_DAYS, MAX_AI = 2, 12      # partidos de los próximos 2 días; máx. llamadas IA por ejecución (cuota gratis)
TG, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
VERBOSE = os.getenv("VERBOSE") == "1"  # 1 = también análisis, grafos y PDF; si no, solo apuestas y resultados
GEMINI, MODEL = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", "gemini-flash-latest")
COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "MaxH", "MaxD", "MaxA", "Max>2.5", "Max<2.5", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
        "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
# mejor cuota entre casas (Max) primero: es la que el bot "apuesta"
MARKETS = {"H": ("MaxH", "AvgH", "B365H"), "D": ("MaxD", "AvgD", "B365D"), "A": ("MaxA", "AvgA", "B365A"),
           "O25": ("Max>2.5", "Avg>2.5", "B365>2.5"), "U25": ("Max<2.5", "Avg<2.5", "B365<2.5")}
NAMES = {"H": "Local", "D": "Empate", "A": "Visitante", "O25": "Más de 2.5", "U25": "Menos de 2.5"}
STATUS_COLORS = {"duda": "#e0a100", "lesionado": "#d64545", "sancionado": "#7b4fd0"}
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]


def season_code(y): return f"{y % 100:02d}{(y + 1) % 100:02d}"
def key(r): return f"{r['Div']}|{r['Date']:%Y-%m-%d}|{r['HomeTeam']}|{r['AwayTeam']}"


def load_csv(raw):
    try: txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError: txt = raw.decode("latin-1")  # temporadas antiguas
    df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip", usecols=lambda c: c in COLS)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def update_history(today):
    HIST.mkdir(parents=True, exist_ok=True)
    cur = today.year if today.month >= 7 else today.year - 1
    for div in LEAGUES:
        for y in range(FIRST_SEASON, cur + 1):
            f = HIST / f"{div}_{season_code(y)}.csv"
            if f.exists() and y < cur: continue  # temporadas cerradas: se descargan una vez
            try:
                r = requests.get(f"{BASE}/mmz4281/{season_code(y)}/{div}.csv", timeout=30)
                if r.ok and r.content: f.write_bytes(r.content)
            except requests.RequestException as e: print("descarga fallida", f.name, e)
    df = pd.concat([load_csv(f.read_bytes()) for f in sorted(HIST.glob("*.csv"))], ignore_index=True)
    return df.dropna(subset=["FTHG", "FTAG"]).sort_values("Date", ignore_index=True)


# ---------- modelo ----------
def strengths(df, div, today, k=5.0):
    d = df[df.Div == div]
    w = 0.5 ** ((today - d.Date).dt.days / 365)  # vida media de 1 año: la última década pesa, lo reciente más
    mu_h, mu_a = (w * d.FTHG).sum() / w.sum(), (w * d.FTAG).sum() / w.sum()
    def rate(team, goals, mu):  # media ponderada con encogimiento hacia la media de la liga
        return ((w * d[goals]).groupby(d[team]).sum() + k * mu) / (w.groupby(d[team]).sum() + k) / mu
    return mu_h, mu_a, {"att_h": rate("HomeTeam", "FTHG", mu_h), "def_h": rate("HomeTeam", "FTAG", mu_a),
                        "att_a": rate("AwayTeam", "FTAG", mu_a), "def_a": rate("AwayTeam", "FTHG", mu_h)}


def probs(lh, la, n=11):
    pois = lambda l: [math.exp(-l) * l ** i / math.factorial(i) for i in range(n)]
    ph, pa = pois(lh), pois(la)
    P = dict.fromkeys(["H", "D", "A", "O25", "U25", "BTTS"], 0.0)
    for i in range(n):
        for j in range(n):
            p = ph[i] * pa[j]
            P["H" if i > j else "D" if i == j else "A"] += p
            P["O25" if i + j > 2 else "U25"] += p
            if i and j: P["BTTS"] += p
    return P


def predict(st, home, away):
    mu_h, mu_a, s = st
    g = lambda t, team: s[t].get(team, 1.0)  # shortcut: recién ascendidos = media de la liga; cargar 2ª división si falla mucho
    lh = mu_h * g("att_h", home) * g("def_a", away)
    la = mu_a * g("att_a", away) * g("def_h", home)
    return lh, la, probs(lh, la)


def value_bets(P, row):
    odds = {m: next((float(row[c]) for c in cols if pd.notna(row.get(c))), None) for m, cols in MARKETS.items()}
    out = []
    for group in (("H", "D", "A"), ("O25", "U25")):
        if not all(odds[m] for m in group): continue
        inv = sum(1 / odds[m] for m in group)
        for m in group:
            p = 0.5 * P[m] + 0.5 / odds[m] / inv  # mezcla modelo + mercado (sin margen): el Poisson solo sobreestima ventajas
            if p >= MIN_P and p * odds[m] - 1 >= EDGE:
                out.append({"market": m, "odds": odds[m], "p": round(p, 3), "edge": round(p * odds[m] - 1, 3)})
    return out


def won(m, hg, ag): return {"H": hg > ag, "D": hg == ag, "A": hg < ag, "O25": hg + ag > 2, "U25": hg + ag < 3}[m]


def h2h(df, a, b):
    return df[((df.HomeTeam == a) & (df.AwayTeam == b)) | ((df.HomeTeam == b) & (df.AwayTeam == a))]


def form(df, t, n=6):
    d = df[(df.HomeTeam == t) | (df.AwayTeam == t)].tail(n)
    return "".join("E" if r.FTHG == r.FTAG else "G" if (r.HomeTeam == t) == (r.FTHG > r.FTAG) else "P"
                   for r in d.itertuples())


def h2h_summary(hh, home, away):
    if hh.empty: return "Sin enfrentamientos previos desde 2005."
    wins = lambda t: int((((hh.HomeTeam == t) & (hh.FTHG > hh.FTAG)) | ((hh.AwayTeam == t) & (hh.FTAG > hh.FTHG))).sum())
    goals = hh.FTHG + hh.FTAG
    return (f"{len(hh)} partidos desde {hh.Date.min():%Y}: {home} {wins(home)} - {int((hh.FTHG == hh.FTAG).sum())} empates - "
            f"{wins(away)} {away}. Media {goals.mean():.2f} goles, +2.5 en {100 * (goals > 2).mean():.0f}%.")


# ---------- bajas (sportsgambler.com) ----------
INJ_SLUGS = {"E0": "england-premier-league", "SP1": "spain-la-liga", "I1": "italy-serie-a",
             "D1": "germany-bundesliga", "F1": "france-ligue-1"}
INJ_TYPES = {"injury-plus": "lesionado", "injury-questionmark": "duda", "redcard": "sancionado"}
ALIASES = {"paris sg": "psg", "ath madrid": "atletico madrid", "ath bilbao": "athletic club", "la coruna": "deportivo",
           "santander": "racing club", "espanol": "espanyol", "ein frankfurt": "frankfurt"}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def injuries(div):
    """{equipo: [jugadores de baja/duda]} de la liga; {} si la web falla."""
    try:
        html = requests.get(f"https://www.sportsgambler.com/injuries/football/{INJ_SLUGS[div]}/",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    except requests.RequestException as e:
        print("bajas no disponibles", div, e); return {}
    out = {}
    for block in html.split('class="injuries-title"')[1:]:
        team = re.search(r'<a [^>]*>([^<]+)</a>', block).group(1)
        out[team] = [{"name": p, "pos": pos, "games": int(g) if g.isdigit() else 0, "goals": gl,
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


# ---------- IA ----------
def ai_report(league, date, home, away, stats, absences, sport="fútbol"):
    return ai_json(f"""Eres un analista de {sport} experto. Partido: {home} vs {away} ({league}), {date:%d/%m/%Y}.
Datos de mi modelo estadístico e historial:
{stats}

Bajas y dudas actuales (partidos jugados esta temporada, motivo, regreso previsto):
{absences}

Valora el peso real de cada baja (titular o suplente, posición) y cruza todo con el historial y la forma.
Responde SOLO con un JSON válido, sin texto extra:
{{"analysis": "análisis experto en español (8-12 frases)", "pick": "pronóstico principal", "confidence": 3}}
confidence va de 1 a 5.""")


def ai_json(prompt):
    if not (GEMINI and VERBOSE): return None
    from google import genai
    client = genai.Client(api_key=GEMINI)  # variable aparte: si no, se cierra antes de la petición
    try:
        txt = client.models.generate_content(model=MODEL, contents=prompt).text or ""
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {"analysis": txt}
    except Exception as e:  # una respuesta mala de la IA no debe tumbar la ejecución
        print("IA fallida", e)
        return None
    finally:
        time.sleep(7)  # límite de peticiones por minuto del plan gratis


# ---------- salida ----------
def player_graph(home, away, squads, path):
    if not VERBOSE: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, networkx as nx
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, team, players in zip(axes, (home, away), squads):
        G = nx.star_graph([team] + [p["name"] for p in players])
        pos = nx.shell_layout(G, [[team], [p["name"] for p in players]]) if players else {team: (0, 0)}
        info = {p["name"]: p for p in players}
        colors = ["#1f3b73" if n == team else STATUS_COLORS[info[n]["status"]] for n in G]
        labels = {n: n if n == team else f"{n}\n{info[n]['pos']}" + (f" · {info[n]['games']} PJ" if info[n]["games"] else "")
                  + f"\nvuelve {info[n]['ret']}" for n in G}
        sizes = [2600 if n == team else 500 + 120 * info[n]["games"] for n in G]  # tamaño = importancia (partidos)
        nx.draw_networkx(G, pos, ax=ax, labels=labels, node_color=colors, edge_color="#ccc", node_size=sizes, font_size=7)
        ax.set_title(f"{team} · {len(players)} bajas/dudas", fontsize=14); ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=s) for s, c in STATUS_COLORS.items()], loc="lower center", ncol=3)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def pdf_report(path, title, sections, img=None):
    if not VERBOSE: return
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page()
    font = next((f for f in FONTS if Path(f).exists()), None)
    if font: pdf.add_font("U", fname=font); pdf.set_font("U")
    else: pdf.set_font("Helvetica")
    t = (lambda s: s) if font else (lambda s: s.encode("latin-1", "replace").decode("latin-1"))
    pdf.set_font_size(16); pdf.multi_cell(0, 9, t(title), new_x="LMARGIN", new_y="NEXT")
    for head, body in sections:
        pdf.ln(2); pdf.set_font_size(12); pdf.multi_cell(0, 7, t(head), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font_size(9); pdf.multi_cell(0, 5, t(body), new_x="LMARGIN", new_y="NEXT")
    if img: pdf.add_page(); pdf.image(str(img), w=190)
    pdf.output(str(path))


def tg(method, files=None, **data):
    if not (TG and CHAT):  # sin token: modo prueba, imprime en consola
        print(f"\n[telegram {method}]", data.get("text") or data.get("caption") or "", *(files or {}))
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG}/{method}", data={"chat_id": CHAT, **data}, files=files, timeout=60)
        if not r.ok: print("Telegram error", r.status_code, r.text[:200])
    except requests.RequestException as e: print("Telegram error", e)
    time.sleep(1)


def send_text(text, force=False):
    if not (VERBOSE or force): return
    for i in range(0, len(text), 4000): tg("sendMessage", text=text[i:i + 4000])


def send_file(method, field, path, caption=""):
    if not VERBOSE: return
    with open(path, "rb") as f: tg(method, files={field: f}, caption=caption[:1000])


# ---------- tareas ----------
def report_results(df, state, today):
    recent = {key(r): r for r in df[df.Date >= today - pd.Timedelta(days=30)].to_dict("records")}
    new = [r for k, r in recent.items() if k not in set(state["seen"])]
    first_run = not state["seen"]
    state["seen"] = list(recent)
    settled = []
    for s in state["signals"]:
        if s.get("result") is None and s["key"] in recent:
            r = recent[s["key"]]; s["result"] = bool(won(s["market"], r["FTHG"], r["FTAG"])); settled.append(s)
        elif s.get("result") is None and s["key"].split("|")[1] < f"{today - pd.Timedelta(days=30):%Y-%m-%d}":
            s["result"] = "nula"  # aplazado o sin resultado
    if first_run or not (new or settled): return
    lines = ["📊 NUEVOS RESULTADOS"]
    for div, name in LEAGUES.items():
        rs = [r for r in new if r["Div"] == div]
        if rs: lines += [f"\n{name}"] + [f"  {r['HomeTeam']} {int(r['FTHG'])}-{int(r['FTAG'])} {r['AwayTeam']}" for r in rs]
    if settled:
        lines.append("\n🎯 SEÑALES RESUELTAS")
        lines += [f"  {'✅' if s['result'] else '❌'} {s['match']} · {NAMES[s['market']]} @ {s['odds']}" for s in settled]
    done = [s for s in state["signals"] if isinstance(s.get("result"), bool)]
    if done:
        profit = sum(s["odds"] - 1 if s["result"] else -1 for s in done)
        lines.append(f"\nHistórico señales: {sum(s['result'] for s in done)}/{len(done)} acertadas, "
                     f"beneficio {profit:+.2f} u (ROI {100 * profit / len(done):+.1f}%)")
    send_text("\n".join(lines))


def analyze_fixtures(df, state, today):
    fx = load_csv(requests.get(f"{BASE}/fixtures.csv", timeout=30).content)
    fx = fx[fx.Div.isin(list(LEAGUES)) & (fx.Date >= today) & (fx.Date <= today + pd.Timedelta(days=HORIZON_DAYS))]
    state["analyzed"] = [k for k in state["analyzed"] if k.split("|")[1] >= f"{today - pd.Timedelta(days=7):%Y-%m-%d}"]
    models, inj, n_ai, tmp = {}, {}, 0, Path(tempfile.mkdtemp())
    for row in fx.sort_values(["Date", "Time"]).to_dict("records"):
        k = key(row)
        start = pd.Timestamp(f"{row['Date']:%Y-%m-%d} {row.get('Time') or '00:00'}", tz="Europe/London")
        if k in state["analyzed"] or start <= pd.Timestamp.now(tz="UTC"): continue  # ya empezado
        if GEMINI and n_ai >= MAX_AI: break  # el resto en la siguiente ejecución
        div, home, away = row["Div"], row["HomeTeam"], row["AwayTeam"]
        if div not in models: models[div], inj[div] = strengths(df, div, today), injuries(div)
        squads = [inj[div].get(find_team(t, inj[div]), []) for t in (home, away)]
        absences = "\n".join(f"  [{(home, away)[i]}] {p['name']} ({p['pos']}, {p['games']} PJ, {p['goals']} goles) "
                             f"{p['status']}: {p['note']} · vuelve {p['ret']}" for i, s in enumerate(squads) for p in s)
        lh, la, P = predict(models[div], home, away)
        vb, hh = value_bets(P, row), h2h(df, home, away)
        fh, fa = form(df, home), form(df, away)
        odds = " / ".join(str(row.get(c)) for c in ("AvgH", "AvgD", "AvgA"))
        stats = (f"Goles esperados: {home} {lh:.2f} - {la:.2f} {away}\n"
                 f"Probabilidades: 1 {P['H']:.0%} | X {P['D']:.0%} | 2 {P['A']:.0%} | +2.5 {P['O25']:.0%} | ambos marcan {P['BTTS']:.0%}\n"
                 f"Cuotas medias 1X2: {odds}\n"
                 f"Forma (últimos 6, G/E/P): {home} {fh} | {away} {fa}\n"
                 f"Cara a cara: {h2h_summary(hh, home, away)}")
        ai = ai_report(LEAGUES[div], row["Date"], home, away, stats, absences or "Sin bajas conocidas")
        n_ai += ai is not None
        title = f"{home} vs {away}"
        msg = [f"⚽ {LEAGUES[div]} · {row['Date']:%d/%m} {row.get('Time') or ''} (hora UK)", title, "", stats]
        if absences: msg += ["", "🚑 Bajas y dudas:", absences]
        if ai and ai.get("pick"): msg += ["", f"🤖 IA: {ai['pick']} (confianza {ai.get('confidence', '?')}/5)"]
        if vb: msg += ["", "💰 SEÑALES DE VALOR:"] + [f"  {NAMES[v['market']]} @ {v['odds']} · prob {v['p']:.0%} · ventaja {v['edge']:+.0%}" for v in vb]
        msg += ["", "Pronóstico estadístico, no garantía. Juega con responsabilidad."]
        send_text("\n".join(msg))

        img = None
        if any(squads):
            img = tmp / f"{k.replace('|', '_')}.png"; player_graph(home, away, squads, img)
            send_file("sendPhoto", "photo", img, f"Estado de la plantilla · {title}")
        h2h_lines = "\n".join(f"{r.Date:%d/%m/%Y}  {r.HomeTeam} {int(r.FTHG)}-{int(r.FTAG)} {r.AwayTeam}"
                              for r in hh.iloc[::-1].head(25).itertuples()) or "—"
        pdf = tmp / f"{k.replace('|', '_')}.pdf"
        pdf_report(pdf, f"{title} · {LEAGUES[div]} · {row['Date']:%d/%m/%Y}", [
            ("Modelo y datos", stats),
            ("Señales de valor", "\n".join(msg[msg.index("💰 SEÑALES DE VALOR:") + 1:-2]) if vb else "Ninguna"),
            ("Análisis IA", (ai or {}).get("analysis", "IA no configurada")),
            ("Bajas y dudas", absences or "Sin bajas conocidas"),
            ("Últimos enfrentamientos", h2h_lines)], img)
        send_file("sendDocument", "document", pdf, f"Informe {title}")
        state["analyzed"].append(k)
        state["signals"] += [{**v, "key": k, "match": title, "start": start.isoformat()} for v in vb]


def main():
    today = pd.Timestamp.now().normalize()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"seen": [], "analyzed": [], "signals": []}
    df = update_history(today)
    print(f"Historial: {len(df)} partidos ({df.Date.min():%Y}-{df.Date.max():%Y})")
    report_results(df, state, today)
    analyze_fixtures(df, state, today)
    import esports, multisport
    multisport.run(state, today)
    esports.run(state, today)
    import bankroll
    bankroll.run(state, today)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
