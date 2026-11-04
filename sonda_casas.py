"""Sonda temporal: casas con cuotas públicas sin clave desde GitHub Actions."""
import requests
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json"}
URLS = {}
for op in ("ub", "ubes", "paf", "pafes", "leo", "leoes", "lvse", "888", "bc", "ngse", "rsi", "bwse", "betcity", "unibetse", "kambi"):
    URLS[f"kambi {op}"] = f"https://eu-offering-api.kambicdn.com/offering/v2018/{op}/listView/football/all/all/all/matches.json?lang=en_GB&market=GB"
URLS.update({
    "1xbet": "https://1xbet.com/LineFeed/Get1x2_VZip?sports=1&count=20&lng=en&mode=4&country=1",
    "smarkets": "https://api.smarkets.com/v3/events/?state=upcoming&type=football_match&limit=5",
    "polymarket": "https://gamma-api.polymarket.com/events?tag_slug=soccer&closed=false&limit=5",
    "betsson obg": "https://www.betsson.com/api/sb/v1/widgets/events-table/v2?categoryIds=1&eventPhase=Prematch&maxMarketCount=1&pageSize=5",
    "marathonbet": "https://www.marathonbet.com/en/betting/Football+-+11",
    "bwin": "https://sports.bwin.com/cds-api/bettingoffer/fixtures?lang=en&country=GB&fixtureTypes=Standard&state=Latest&offerMapping=Filtered&sportIds=4&take=5",
    "williamhill": "https://sports.williamhill.com/betting/en-gb/football",
    "betfair sportsbook": "https://www.betfair.com/sport/football",
    "sportium": "https://sports.sportium.es/es/s/FOOT/Futbol",
    "codere": "https://apuestas.codere.es/NavigationService/Event/GetEventsBySport?sportHandle=soccer",
    "draftkings nash": "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1/leagues/40253",
    "betway": "https://sports.betway.com/en/sports/sct/soccer",
    "stake": "https://stake.com/api/graphql",
    "pinnacle (control)": "https://guest.api.arcadia.pinnacle.com/0.1/sports/29/matchups?withSpecials=false&brandId=0",
})
for name, u in URLS.items():
    try:
        r = requests.get(u, headers=UA, timeout=20)
        print(f"{name:20} {r.status_code} {r.headers.get('content-type', '')[:25]:25} {len(r.content):>9} B  {r.text[:90]!r}")
    except Exception as e: print(f"{name:20} ERROR {repr(e)[:90]}")
