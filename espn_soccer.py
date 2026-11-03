"""Jugadores, porteros y árbitro de fútbol desde el resumen público de ESPN (summary?event=<id>), para:
props de jugador (tiros, tiros a puerta, goles, asistencias, tarjetas), efecto del portero en córners, árbitro en
todas las ligas y liquidar las props de Unibet (data/kambi_player_props.csv).

data/espn_soccer/<liga>.jsonl, una línea por partido: {id, date, league, home, away, hs, as, referee,
team_stats: {local|visitante: {wonCorners, totalShots, shotsOnTarget, foulsCommitted, yellowCards, ...}},
players: [[id, nombre, equipo (h|a), titular, posición, goles, asist., tiros, tiros a puerta, paradas, tiros recibidos,
faltas, amarillas, rojas, fueras de juego]]}.
Uso: python espn_soccer.py [max_partidos] → incremental: primero los últimos 7 días, luego historial hacia atrás
(3 temporadas) hasta max_partidos por ejecución (lo lanza .github/workflows/datos_espn.yml)."""
import json, sys, time

import pandas as pd
import requests

import analyst as A

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
LEAGUES = {"E0": "eng.1", "E1": "eng.2", "SP1": "esp.1", "SP2": "esp.2", "I1": "ita.1", "I2": "ita.2", "D1": "ger.1", "D2": "ger.2",
           "F1": "fra.1", "F2": "fra.2", "N1": "ned.1", "P1": "por.1", "B1": "bel.1", "T1": "tur.1", "G1": "gre.1", "SC0": "sco.1",
           "ucl": "uefa.champions", "uel": "uefa.europa",
           # ligas «nuevas» del bot: football-data no trae sus estadísticas; ESPN sí (córners, tarjetas, tiros, jugadores)
           "ARG": "arg.1", "BRA": "bra.1", "MEX": "mex.1", "USA": "usa.1", "JPN": "jpn.1", "AUT": "aut.1", "DNK": "den.1",
           "NOR": "nor.1", "SWE": "swe.1", "SWZ": "sui.1", "POL": "pol.1"}
PSTATS = ["totalGoals", "goalAssists", "totalShots", "shotsOnTarget", "saves", "shotsFaced", "foulsCommitted", "yellowCards",
          "redCards", "offsides"]
TSTATS = ["wonCorners", "totalShots", "shotsOnTarget", "foulsCommitted", "yellowCards", "redCards", "saves", "possessionPct", "offsides"]
DIR = A.DATA / "espn_soccer"


def get(path, **params):
    r = requests.get(f"{BASE}/{path}", params=params, headers=A.UA, timeout=30)
    r.raise_for_status(); return r.json()


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return 0.0


def referee(d):
    for o in (d.get("gameInfo") or {}).get("officials") or []:
        pos = str((o.get("position") or {}).get("name") or o.get("position") or "").lower()
        if not pos or "referee" in pos and "assistant" not in pos and "var" not in pos and "video" not in pos:
            return o.get("displayName") or o.get("fullName")
    return None


def compact(d):
    """Resumen ESPN de fútbol → {referee, team_stats, players, home_id}."""
    comp = ((d.get("header") or {}).get("competitions") or [{}])[0]
    side = {str(c.get("team", {}).get("id")): ("h" if c.get("homeAway") == "home" else "a") for c in comp.get("competitors", [])}
    ts = {}
    for t in (d.get("boxscore") or {}).get("teams", []):
        sd = side.get(str(t.get("team", {}).get("id")))
        if sd: ts[sd] = {s["name"]: _f(s.get("displayValue", s.get("value"))) for s in t.get("statistics", []) if s.get("name") in TSTATS}
    players = []
    for ro in d.get("rosters") or []:
        sd = side.get(str((ro.get("team") or {}).get("id"))) or ("h" if ro.get("homeAway") == "home" else "a")
        for p in ro.get("roster") or []:
            ath = p.get("athlete") or {}
            if not ath.get("id"): continue
            st = {s.get("name"): _f(s.get("value")) for s in p.get("stats") or []}
            if not st and not p.get("starter") and not p.get("subbedIn"): continue  # banquillo sin jugar
            pos = (p.get("position") or {}).get("abbreviation") or (p.get("position") or {}).get("name") or ""
            players.append([ath["id"], ath.get("displayName"), sd, bool(p.get("starter")), pos, *[st.get(k, 0.0) for k in PSTATS]])
    return {"referee": referee(d), "team_stats": ts, "players": players}


def path(league): return DIR / f"{LEAGUES[league]}.jsonl"


def load(league):
    p = path(league)
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def recent_rows(codes=("ucl", "uel"), days=30):
    """{clave de multisport «liga|fecha|local|visitante»: fila con columnas de football-data (FTHG, HC, HY=amarillas+rojas,
    HF, HST…)} de los partidos de ESPN de los últimos días, para liquidar combinadas europeas (football_stats.settle_sgp)."""
    out, cut = {}, f"{pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days):%Y-%m-%d}"
    for code in codes:
        for g in load(code):
            if str(g["date"])[:10] < cut: continue
            ts = g.get("team_stats") or {}
            h, a = ts.get("h") or {}, ts.get("a") or {}
            if not h or not a: continue
            x = lambda d, k: d.get(k, float("nan"))
            out[f"{code}|{str(g['date'])[:10]}|{g['home']}|{g['away']}"] = {
                "FTHG": g["hs"], "FTAG": g["as"], "HC": x(h, "wonCorners"), "AC": x(a, "wonCorners"),
                "HY": x(h, "yellowCards") + h.get("redCards", 0), "AY": x(a, "yellowCards") + a.get("redCards", 0),
                "HF": x(h, "foulsCommitted"), "AF": x(a, "foulsCommitted"), "HST": x(h, "shotsOnTarget"), "AST": x(a, "shotsOnTarget")}
    return out


def events(slug, day):
    """[(id, fecha, local, visitante, goles local, goles visitante)] de los partidos TERMINADOS de un día."""
    out = []
    for e in get(f"{slug}/scoreboard", dates=f"{day:%Y%m%d}").get("events", []):
        c = (e.get("competitions") or [{}])[0]
        if not ((c.get("status") or e.get("status") or {}).get("type") or {}).get("completed"): continue
        t = {x.get("homeAway"): x for x in c.get("competitors", [])}
        if "home" in t and "away" in t:
            out.append((e["id"], e.get("date"), t["home"]["team"]["displayName"], t["away"]["team"]["displayName"],
                        _f(t["home"].get("score")), _f(t["away"].get("score"))))
    return out


def update(max_games=1500, seasons=3, pause=0.4):
    """Últimos 7 días de cada liga y después el historial hacia atrás, hasta max_games partidos nuevos."""
    DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    state_f = DIR / "progreso.json"
    prog = json.loads(state_f.read_text()) if state_f.exists() else {}
    n, probe = 0, False
    for back in (False, True):
        for code, slug in LEAGUES.items():
            p = path(code)
            done = {json.loads(l)["id"] for l in p.open(encoding="utf-8")} if p.exists() else set()
            if back:  # historial: desde el día en que se quedó, hacia atrás
                start = pd.Timestamp(prog.get(slug, f"{today - pd.Timedelta(days=8):%Y-%m-%d}"), tz="UTC")
                days = pd.date_range(end=start, periods=150, freq="D")[::-1]
                if start < today - pd.DateOffset(years=seasons): continue
            else: days = pd.date_range(end=today, periods=7, freq="D")[::-1]
            with p.open("a", encoding="utf-8") as f:
                for day in days:
                    if n >= max_games: break
                    try: evs = events(slug, day)
                    except (requests.RequestException, ValueError) as e: print("ESPN", slug, day.date(), e); continue
                    for eid, date, home, away, hs, as_ in evs:
                        if eid in done or n >= max_games: continue
                        try: d = get(f"{slug}/summary", event=eid)
                        except (requests.RequestException, ValueError) as e: print("ESPN", slug, eid, e); time.sleep(3); continue
                        c = compact(d)
                        if not probe:  # una vez por ejecución: qué trae de verdad (para revisar en el log)
                            print("sondeo", slug, eid, "árbitro:", c["referee"], "| equipo:", c["team_stats"].get("h"),
                                  "| jugador:", c["players"][:1]); probe = True
                        f.write(json.dumps({"id": eid, "date": date, "league": code, "home": home, "away": away, "hs": hs, "as": as_,
                                            **c}, ensure_ascii=False) + "\n")
                        done.add(eid); n += 1
                        time.sleep(pause)
                    if back: prog[slug] = f"{day - pd.Timedelta(days=1):%Y-%m-%d}"
                    time.sleep(pause)
            state_f.write_text(json.dumps(prog, indent=0))
    print(f"ESPN fútbol: {n} partidos nuevos")
    return n


if __name__ == "__main__":
    update(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
