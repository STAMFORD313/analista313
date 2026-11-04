"""Sonda temporal: fuentes de datos gratis y sin clave, ¿responden desde GitHub Actions?"""
import requests
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json,text/html"}
URLS = {
    "clubelo (Elo fútbol)": "http://api.clubelo.com/2026-11-04",
    "understat (xG)": "https://understat.com/league/EPL",
    "sofascore": "https://api.sofascore.com/api/v1/sport/football/scheduled-events/2026-11-04",
    "fotmob": "https://www.fotmob.com/api/data/matches?date=20261104",
    "thesportsdb": "https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d=2026-11-04&s=Soccer",
    "fbref": "https://fbref.com/en/comps/9/Premier-League-Stats",
    "statsbomb open": "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json",
    "nhl oficial": "https://api-web.nhle.com/v1/schedule/now",
    "mlb oficial": "https://statsapi.mlb.com/api/v1/schedule?sportId=1",
    "nba cdn": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    "nba stats": "https://stats.nba.com/stats/scoreboardv3?GameDate=2026-11-04&LeagueID=00",
    "opendota": "https://api.opendota.com/api/proMatches",
    "liquipedia cs": "https://liquipedia.net/counterstrike/api.php?action=query&meta=siteinfo&format=json",
    "oracle's elixir (LoL)": "https://oracleselixir.com/tools/downloads",
    "gol.gg (LoL)": "https://gol.gg/tournament/list/",
    "hltv": "https://www.hltv.org/matches",
    "pinnacle guest": "https://guest.api.arcadia.pinnacle.com/0.1/sports/29/matchups",
    "betfair exchange": "https://www.betfair.com/www/sports/exchange/readonly/v1/bymarket?marketIds=1.1",
    "football-data.org": "https://api.football-data.org/v4/matches",
    "openligadb (Bundesliga)": "https://api.openligadb.de/getmatchdata/bl1/2026",
}
for name, u in URLS.items():
    try:
        r = requests.get(u, headers=UA, timeout=20)
        print(f"{name:26} {r.status_code} {r.headers.get('content-type', '')[:30]:30} {len(r.content):>8} B  {r.text[:80]!r}")
    except Exception as e: print(f"{name:26} ERROR {repr(e)[:100]}")
