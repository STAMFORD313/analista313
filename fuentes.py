"""Fuentes de datos extra (gratis, sin clave; sondeadas desde Actions el 4-nov):

  Understat  xG por partido de EPL, LaLiga, Bundesliga, Serie A, Ligue 1 y RFPL desde 2014 → data/understat/<liga>.csv
  OpenDota   partidas profesionales de Dota 2 (equipos, liga, serie, marcador) → data/dota2/opendota.csv
  FotMob     estadísticas por partido con xG, ocasiones claras, tiros, córners… de ~30 ligas → data/fotmob/AAAAMM.jsonl

Uso: python fuentes.py understat · python fuentes.py opendota [páginas]  (los lanza nutricion.py)."""
import json, re, sys, time

import pandas as pd
import requests

import analyst as A

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
UNDERSTAT = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1", "RUS": "RFPL"}
U_DIR = A.DATA / "understat"
U_COLS = ["id", "date", "season", "home", "away", "hg", "ag", "hxg", "axg", "forecast_w", "forecast_d", "forecast_l"]


def understat_season(league, season):
    """Partidos jugados de una temporada (año de inicio) de Understat."""
    r = requests.get(f"https://understat.com/league/{league}/{season}", headers=UA, timeout=40)
    r.raise_for_status()
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
    if not m: return []
    data = json.loads(m.group(1).encode().decode("unicode_escape"))
    out = []
    for g in data:
        if not g.get("isResult"): continue
        f = g.get("forecast") or {}
        out.append({"id": int(g["id"]), "date": g["datetime"], "season": season, "home": g["h"]["title"], "away": g["a"]["title"],
                    "hg": int(g["goals"]["h"]), "ag": int(g["goals"]["a"]), "hxg": float(g["xG"]["h"]), "axg": float(g["xG"]["a"]),
                    "forecast_w": float(f.get("w", "nan")), "forecast_d": float(f.get("d", "nan")), "forecast_l": float(f.get("l", "nan"))})
    return out


def update_understat(first=2014):
    """Temporadas cerradas: una vez; la actual (y la anterior) en cada pasada."""
    U_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC"); cur = now.year if now.month >= 7 else now.year - 1
    tot = 0
    for code, lg in UNDERSTAT.items():
        f = U_DIR / f"{code}.csv"
        d = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=U_COLS)
        have = set(d.season.unique()) if len(d) else set()
        rows = []
        for s in range(first, cur + 1):
            if s in have and s < cur - 1: continue
            try: rows += understat_season(lg, s)
            except (requests.RequestException, ValueError) as e: print("Understat", lg, s, repr(e)[:120])
            time.sleep(1)
        if rows:
            d = pd.concat([d, pd.DataFrame(rows, columns=U_COLS)]).drop_duplicates("id", keep="last").sort_values("date")
            d.to_csv(f, index=False); tot += len(rows)
        print(f"Understat {lg}: {len(d)} partidos")
    print(f"Understat: {tot} filas actualizadas")


OD_FILE = A.DATA / "dota2" / "opendota.csv"
OD_COLS = ["match_id", "start_time", "duration", "league", "series_id", "series_type", "radiant_id", "radiant", "dire_id", "dire",
           "radiant_score", "dire_score", "radiant_win"]


def update_opendota(pages=40):
    """Partidas pro nuevas (desde la más reciente hasta las ya guardadas) y, con lo que sobre, historial hacia atrás."""
    d = pd.read_csv(OD_FILE) if OD_FILE.exists() else pd.DataFrame(columns=OD_COLS)
    known = set(d.match_id) if len(d) else set()
    rows, cursor, back = [], None, False
    for _ in range(pages):  # API gratis: 60 peticiones/min
        url = "https://api.opendota.com/api/proMatches" + (f"?less_than_match_id={cursor}" if cursor else "")
        try: page = requests.get(url, headers=UA, timeout=40).json()
        except (requests.RequestException, ValueError) as e: print("OpenDota", repr(e)[:120]); break
        if not isinstance(page, list) or not page: break
        new = [m for m in page if m["match_id"] not in known]
        rows += [{"match_id": m["match_id"], "start_time": pd.Timestamp(m["start_time"], unit="s", tz="UTC").isoformat(),
                  "duration": m.get("duration"), "league": m.get("league_name"), "series_id": m.get("series_id"),
                  "series_type": m.get("series_type"), "radiant_id": m.get("radiant_team_id"), "radiant": m.get("radiant_name"),
                  "dire_id": m.get("dire_team_id"), "dire": m.get("dire_name"), "radiant_score": m.get("radiant_score"),
                  "dire_score": m.get("dire_score"), "radiant_win": m.get("radiant_win")} for m in new]
        cursor = min(m["match_id"] for m in page)
        if len(new) < len(page) and not back and len(d):  # alcanzadas las guardadas: seguir desde la más antigua
            back, cursor = True, int(d.match_id.min())
        time.sleep(1.1)
    if rows:
        OD_FILE.parent.mkdir(parents=True, exist_ok=True)
        d = pd.concat([d, pd.DataFrame(rows, columns=OD_COLS)]).drop_duplicates("match_id").sort_values("match_id")
        d.to_csv(OD_FILE, index=False)
    print(f"OpenDota: {len(rows)} partidas nuevas, {len(d)} guardadas")


# ── FotMob: estadísticas por partido con xG (data/fotmob/AAAAMM.jsonl) ──────────────────────────────────────────────
FM = "https://www.fotmob.com/api/data"
FM_DIR = A.DATA / "fotmob"
FM_LEAGUES = ("Premier League", "LaLiga", "Bundesliga", "Serie A", "Ligue 1", "Championship", "Eredivisie", "Liga Portugal",
              "Champions League", "Europa League", "Conference League", "Segunda", "Serie B", "2. Bundesliga", "Ligue 2",
              "Pro League", "Super Lig", "Premiership", "Super League", "MLS", "Liga MX", "Serie A Betano", "Liga Profesional",
              "Eliteserien", "Allsvenskan", "Superliga", "Ekstraklasa", "J. League", "J1 League", "Bundesliga 2")
# Solo países donde FotMob publica xG (medido: el resto llega sin estadísticas) y sin juveniles/femeninas/regionales.
FM_CC = {"ENG", "ESP", "ITA", "GER", "FRA", "POR", "NED", "BEL", "TUR", "SCO", "GRE", "AUT", "SUI", "DEN", "SWE", "NOR",
         "POL", "RUS", "USA", "MEX", "ARG", "BRA", "JPN", "INT", "INT-2"}
FM_SKIP = re.compile(r"U\d\d|Femen|Women|Frauen|Féminine|Feminine|Federaci|Group|Youth|Reserve|Primavera|Liga F\b|National League", re.I)


def fm_keep(name, ccode):
    return (ccode in FM_CC and not FM_SKIP.search(name or "")
            and any(name.startswith(x) or x in name for x in FM_LEAGUES + ("LaLiga2", "Superligaen", "Primera")))


def fm_compact(d):
    """Detalle de FotMob → {id, date, league, home, away, hs, as, stats: {clave: [local, visitante]}} (solo números)."""
    g = d.get("general") or {}; h = [t or {} for t in ((d.get("header") or {}).get("teams") or [{}, {}])] + [{}, {}]
    stats = {}
    for grp in ((((d.get("content") or {}).get("stats") or {}).get("Periods") or {}).get("All") or {}).get("stats") or []:
        for s in (grp or {}).get("stats") or []:
            if not isinstance(s, dict): continue
            v = s.get("stats")
            if s.get("key") and isinstance(v, list) and len(v) == 2:
                try: stats[s["key"]] = [float(str(x).split(" ")[0].replace("%", "")) for x in v]
                except (TypeError, ValueError): pass
    return {"id": g.get("matchId"), "date": g.get("matchTimeUTCDate") or g.get("matchTimeUTC"), "league": g.get("leagueName"),
            "ccode": g.get("countryCode"), "home": h[0].get("name"), "away": h[1].get("name"),
            "hs": h[0].get("score"), "as": h[1].get("score"), "stats": stats}


def fm_day(day, done):
    """Partidos terminados de un día en las ligas de FM_LEAGUES aún no guardados: [(id, liga)]."""
    r = requests.get(f"{FM}/matches?date={day:%Y%m%d}", headers=UA, timeout=40).json() or {}
    out = []
    for lg in r.get("leagues") or []:
        if not isinstance(lg, dict): continue
        name = lg.get("name") or ""
        if not fm_keep(name, lg.get("ccode") or ""): continue
        for m in lg.get("matches") or []:
            if (m.get("status") or {}).get("finished") and str(m["id"]) not in done: out.append(str(m["id"]))
    return out


def update_fotmob(max_details=400, days_back=3):
    """Últimos días y, con lo que sobre, historial hacia atrás (progreso en data/fotmob/progreso.json)."""
    FM_DIR.mkdir(parents=True, exist_ok=True)
    prog_f = FM_DIR / "progreso.json"; prog = json.loads(prog_f.read_text()) if prog_f.exists() else {}
    done = set()
    for f in FM_DIR.glob("*.jsonl"):
        done |= {json.loads(l)["id"] for l in f.open(encoding="utf-8")}
    today = pd.Timestamp.now(tz="UTC").normalize()
    back_from = pd.Timestamp(prog.get("atras", str(today - pd.Timedelta(days=days_back + 1))))
    days = list(pd.date_range(end=today, periods=days_back, freq="D")[::-1]) + list(pd.date_range(end=back_from, periods=400, freq="D")[::-1])
    n = fails = 0
    for i, day in enumerate(days):
        if fails >= 5 and n == 0: print("FotMob: 5 días seguidos sin respuesta válida, se deja para la próxima"); break
        if n >= max_details or day < today - pd.DateOffset(years=3): break
        try: ids = fm_day(day, done)
        except (requests.RequestException, ValueError, AttributeError, TypeError) as e: print("FotMob", day.date(), repr(e)[:100]); fails += 1; continue
        for mid in ids:
            if n >= max_details: break
            try: c = fm_compact(requests.get(f"{FM}/matchDetails?matchId={mid}", headers=UA, timeout=40).json() or {})
            except (requests.RequestException, ValueError, AttributeError, TypeError) as e: print("FotMob", mid, repr(e)[:100]); continue
            if not c["date"]: continue
            with (FM_DIR / f"{str(c['date'])[:7].replace('-', '')}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**c, "id": str(c["id"])}, ensure_ascii=False) + "\n")
            done.add(str(c["id"])); n += 1; time.sleep(0.5)
        if i >= days_back and n < max_details: prog["atras"] = str(day - pd.Timedelta(days=1))  # día completo
        time.sleep(0.5)
    prog_f.write_text(json.dumps(prog))
    print(f"FotMob: {n} partidos nuevos, {len(done)} guardados")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if what in ("understat", "todo"): update_understat()
    if what in ("opendota", "todo"): update_opendota(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    if what in ("fotmob", "todo"): update_fotmob(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
