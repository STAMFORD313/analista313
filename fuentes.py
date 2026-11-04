"""Fuentes de datos extra (gratis, sin clave; sondeadas desde Actions el 4-nov):

  Understat  xG por partido de EPL, LaLiga, Bundesliga, Serie A, Ligue 1 y RFPL desde 2014 → data/understat/<liga>.csv
  OpenDota   partidas profesionales de Dota 2 (equipos, liga, serie, marcador) → data/dota2/opendota.csv

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


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if what in ("understat", "todo"): update_understat()
    if what in ("opendota", "todo"): update_opendota(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
