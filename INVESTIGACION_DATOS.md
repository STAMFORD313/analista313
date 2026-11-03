# Investigación: fuentes de datos por jugador y cuotas de props (2026-11-03)

**Cómo se ha verificado.** El proxy de esta sesión bloquea ESPN, Kambi, bo3.gg, Pinnacle, The Odds API, FBref, Understat, nba.com, Sofascore y football-data (CONNECT 403), y WebFetch tampoco resuelve esos dominios. Solo había acceso a GitHub (raw). Por eso:
- **Comprobado con datos reales**: los mercados de Kambi que el propio bot ya ha visto (`data/kambi_labels.json`, run real del 03-nov), los campos de jugador de ESPN fútbol (salida del notebook de `soccerdata`, que llama a `site.api.espn.com/.../summary`, temporada EPL 2019/20), el catálogo de StatsBomb open data (`competitions.json` descargado) y la columna `Referee` de `data/hist`.
- **Según documentación o búsquedas** (marcado ⚠️ *sin probar*): todo lo demás. Al final hay un script de sondeo para ejecutarlo una vez en GitHub Actions, donde ESPN y Kambi sí responden.

## Tabla priorizada

| # | Fuente | Qué aporta | Historial | Límite / coste | ¿Permitido desde un bot? | Impacto |
|---|---|---|---|---|---|---|
| 1 | **Kambi (Unibet) `betoffer/event/{id}.json`**, ya en uso | **Cuotas reales de props de jugador de fútbol**: tiros a puerta del jugador, marcar, asistencia, marcar o asistir, tarjeta, 2+ goles, primer goleador. Probablemente también NBA/NHL/MLB ⚠️ | Ninguno (solo se ven en directo: hay que guardarlas) | Sin clave. Es la misma petición que el bot ya hace por partido | Zona gris: es la API pública de la web, sin cuenta y con poco volumen (igual que hoy) | **Muy alto**: es la única cuota real de props gratis confirmada |
| 2 | **ESPN fútbol `summary?event=`** (misma API que multisport/nba_players) | Alineación (titular, cambios con minuto, posición), y por jugador: goles, asistencias, tiros, tiros a puerta, paradas, tiros recibidos, goles encajados, faltas, tarjetas y fueras de juego. Por equipo: córners, tiros, posesión y paradas. Árbitro vía `officials` | Varias temporadas (confirmada la 2019/20; profundidad exacta ⚠️) | Sin clave ni límite publicado | No oficial y sin permiso explícito. Ya se usa en el bot para NBA/NHL/MLB | **Muy alto**: jugadores, porteros y árbitros en ~27 ligas |
| 3 | **ESPN NBA/NHL/MLB `summary`**, ya en uso | NBA: REB, AST, 3PT, STL, BLK, TO y PTS por jugador. NHL: tiros (S) por jugador y SA/GA del portero. MLB: K, IP, BB y H del lanzador | El bot ya tiene 3 temporadas descargadas (box.jsonl, pitchers.csv, goalies.csv); ESPN tiene muchas más | Igual que el n.º 2 | Igual que el n.º 2 | **Alto** y casi gratis: solo hay que guardar más columnas |
| 4 | **The Odds API** (`api.the-odds-api.com/v4`) | Props NBA/NHL/MLB/NFL (`player_points`, `player_rebounds`, `player_assists`, `player_threes`, `player_shots_on_goal`, `pitcher_strikeouts`…) y algo de fútbol (`player_shots_on_target`, `player_goal_scorer_anytime`), casi todo de casas de EE. UU. | Históricos solo de pago (unos 10 créditos por mercado y región) | Gratis: 500 créditos/mes; 1 crédito = 1 mercado × 1 región por llamada, y los props van por evento | **Sí**: API oficial con clave (secreto de GitHub) | Medio: da para unos 15 partidos NBA/mes con 1 mercado |
| 5 | **Kalshi** (`api.elections.kalshi.com/trade-api/v2`) / **Polymarket** (`gamma-api.polymarket.com`) | Precios de mercados de predicción: partidos NBA/NFL/fútbol/esports. Kalshi tiene algunos props de jugador ⚠️ | Kalshi: velas/trades históricos públicos | Sin clave para leer | **Sí**: lectura pública documentada | Medio: un «mercado» de referencia sin margen de casa |
| 6 | **OddsPapi / odds-api.io** | Agregadores con plan gratis: muchas casas (incluidas esports y Pinnacle en OddsPapi); props en el plan gratis ⚠️ | OddsPapi dice tener `/historical-odds` en el plan gratis (máx. 3 casas por llamada) | OddsPapi: 250-1000 peticiones/mes (sus propias páginas no coinciden). odds-api.io: 500/día, sin casas sharp | Sí, con clave | Medio: lo mejor para medir el ROI de esports contra una 2.ª casa |
| 7 | **StatsBomb open data** (GitHub) | Eventos completos (tiros con xG, paradas, alineaciones, árbitro) | Solo competiciones sueltas: las 5 grandes de 2015/16, EPL 03/04, Messi en La Liga 04-21, Mundiales, Euros, Ligue 1 21-23, Bundesliga 23/24 | Ilimitado (archivos en GitHub) | **Sí**: licencia libre citando a StatsBomb | Bajo: sirve para calibrar (p. ej. el efecto del portero), no para producción |
| 8 | **API NHL `api-web.nhle.com/v1`** / **MLB `statsapi.mlb.com`** | Boxscore oficial (tiros por jugador; K por lanzador, alineaciones y abridores probables) | Décadas | Sin clave | NHL: no oficial, sin términos. MLB: «uso individual, no comercial, no masivo» (aviso MLBAM) | Bajo-medio: ESPN ya lo cubre. Útil para el abridor probable de MLB |
| 9 | **OpenDota** (`api.opendota.com/api`) | Dota 2: K/D/A por jugador en partidas pro | Años | Gratis sin clave: unas 2000 llamadas/día y 60/min | Sí | Bajo: bo3.gg ya da K/D/A |
| — | **FBref / Basketball-Reference** | Desde el 20-ene-2026 FBref **ya no tiene datos Opta** (tiros, xG…) | — | 10 peticiones/min (FBref) o 20/min, o bloqueo de 1 día | Prohíben crear herramientas con lo que se rastrea | **Descartar** |
| — | **Understat** | xG y tiros por jugador, 6 ligas desde 2014/15 | — | Sin API: hay que rastrear el HTML/JSON | Sin términos claros. MEJORAS.md ya lo descartó por anti-bot | Descartar |
| — | **stats.nba.com**, **HLTV**, **Sofascore/FotMob/WhoScored**, **Pinnacle** | — | — | nba.com bloquea las IPs de nube (GitHub Actions); HLTV/WhoScored usan Cloudflare | Saltárselo violaría los términos. Pinnacle solo da API a cuentas con saldo aprobadas; el `guest.api.arcadia` usa una clave sacada de la web | **No usar** |

## Detalles

### 1. Kambi: props de jugador (ya los descargamos y no los leemos)
- Endpoint: `https://eu-offering-api.kambicdn.com/offering/v2018/ub/betoffer/event/{id}.json?lang=en_GB&market=GB` (`kambi.get("betoffer/event/{id}")`, que ya se cachea en `kambi._offer_cache`). Para otros deportes, la lista sería `listView/basketball/nba/all/all/matches`, `listView/ice_hockey/nhl/...`, `listView/baseball/mlb/...` y `listView/esports/...` ⚠️.
- **Mercados de jugador vistos de verdad** en el run del 03-nov (Championship; número de veces vistos en `data/kambi_labels.json`):
  `To Score` 292 · `To Get a Card` 292 · `To score or give an assist (Settled using Opta data)` 292 · `Player's shots on target (Settled using Opta data)` 199 · `To give an assist (Settled using Opta data)` 162 · `To score at least 2 goals` 110 · `To score at least 3 goals` 10 · `First Goal Scorer` 8. Por equipo: `Most Shots on Target`, `Most Cards`, `Total Cards - <equipo>`, `<equipo> given a Red Card`, `Total Corners By <equipo> - 1st Half`.
- Estructura (la que ya lee `kambi.parse`): `betOffers[].criterion.englishLabel` y `outcomes[]` con `odds`/1000, `line`/1000, `type` (`OT_OVER`/`OT_UNDER`/`OT_YES`), y en los props `participant` (nombre del jugador) y `participantId` ⚠️ (comprobar el nombre exacto del campo con el sondeo).
- **Liquidación**: Kambi liquida con **Opta**. Ya se vio que football-data cuenta menos tiros a puerta que Opta (kambi.py). Antes de apostar tiros a puerta de jugador hay que comprobar que ESPN cuenta igual que Opta: comparar los tiros a puerta por equipo de ESPN con los de football-data y con el resultado de Kambi en unos 100 partidos.
- NBA/NHL/MLB: Kambi publica props (puntos, rebotes, asistencias y triples del jugador; tiros del jugador en NHL; strikeouts del lanzador en MLB) según sus notas de producto. Las etiquetas exactas están ⚠️ *sin ver*: basta con ampliar `save_labels` a esos deportes durante un día.
- Términos: no hay API pública documentada para terceros. Es el mismo uso «zona gris» que el bot ya hace (sin cuenta, una petición por partido). Mantener ese volumen bajo.

### 2. ESPN fútbol: jugadores, porteros y árbitro
- `https://site.api.espn.com/apis/site/v2/sports/soccer/{liga}/scoreboard?dates=YYYYMMDD` → ids. Después, `.../soccer/{liga}/summary?event={id}`.
- Códigos de liga: `eng.1` `eng.2` `eng.3` `eng.4` `esp.1` `esp.2` `ita.1` `ita.2` `ger.1` `ger.2` `fra.1` `fra.2` `ned.1` `por.1` `bel.1` `tur.1` `gre.1` `sco.1` `aut.1` `sui.1` `den.1` `swe.1` `nor.1` `usa.1` `mex.1` `arg.1` `bra.1` `uefa.champions` `uefa.europa`.
- **Campos confirmados** (columnas de soccerdata `read_lineup` / `read_matchsheet` sobre la EPL 2019/20; en el JSON van en camelCase dentro de `rosters[i].roster[j].stats[] {name, value}`):
  - Por jugador: `appearances, subIns, totalGoals, goalAssists, totalShots, shotsOnTarget, saves, shotsFaced, goalsConceded, foulsCommitted, foulsSuffered, yellowCards, redCards, offsides, ownGoals`, más `starter`, `subbedIn/subbedOut` (con minuto en `plays[].clock`), `position`, `formationPlace` y `athlete.id/displayName`.
  - Por equipo (`boxscore.teams[i].statistics[]`): `wonCorners, totalShots, shotsOnTarget, saves, possessionPct, foulsCommitted, yellowCards, redCards, offsides`.
  - Partido: `gameInfo.venue`, `attendance` y `boxscore.form`.
- **Árbitro**: `gameInfo.officials` en el summary ⚠️, o el endpoint `https://sports.core.api.espn.com/v2/sports/soccer/leagues/{liga}/events/{id}/competitions/{id}/officials` (documentado en el WADL de ESPN). football-data solo trae `Referee` en E0, E1 y SC0 (comprobado en `data/hist`). ESPN cubriría las demás ligas, que es lo que le faltaba a train_referee.py para pasar el corte.
- **Efecto del portero en córners**: titular con `position == "Goalkeeper"` y `saves/shotsFaced` → el portero de cada partido se cruza con los HC/AC de football-data (mismo emparejamiento por fecha y equipos que hace espn_extra).
- Historial: hay temporadas con estadísticas de jugador al menos desde 2019/20 (comprobado). Las anteriores y las ligas menores pueden traer `rosters` sin `stats` ⚠️ (soccerdata lo trata como «No lineup info»). Coste: unas 380 peticiones por temporada y liga; a ~2 por segundo, ~3 min por liga y temporada.
- Ejemplo (forma, según soccerdata): `{"starter": true, "athlete": {"id": "...", "displayName": "Douglas Luiz"}, "position": {"name": "Defensive Midfielder"}, "stats": [{"name": "totalShots", "value": 2.0}, {"name": "shotsOnTarget", "value": 1.0}, ...]}`.

### 3. ESPN NBA/NHL/MLB: añadir columnas, no fuentes
- `nba_players.compact()` ya recorre `boxscore.players[].statistics[]` con `labels`. En la NBA esas etiquetas son `MIN, FG, 3PT, FT, OREB, DREB, REB, AST, STL, BLK, TO, PF, +/-, PTS` (`3PT` viene como «hechos-intentados»), pero solo se guardan MIN, PTS y +/-. Guardando también REB, AST, 3PT, STL y BLK se tendrían rebotes, asistencias y triples con el historial ya descargado (desde 2023-24; para más temporadas, volver a bajarlo).
- NHL: los patinadores traen `S` (tiros) y `TOI`; starters.py solo guarda el portero. MLB: el lanzador trae `K` (ya se usa dentro del FIP), así que guardar `K` e `IP` del abridor basta para el modelo de strikeouts.
- Abridor probable de MLB (antes del partido): `statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher` ⚠️, de uso no comercial (encaja con una banca virtual).

### 4. APIs de cuotas con plan gratis
- **The Odds API**: `GET /v4/sports/{deporte}/events/{eventId}/odds?regions=us&markets=player_points,player_rebounds&oddsFormat=decimal&apiKey=...`. Props de NBA, NFL, NHL y MLB (casas de EE. UU.); en fútbol EPL y algunas más (`player_shots_on_target`, `player_goal_scorer_anytime`, `player_to_receive_card`), «sobre todo en casas de EE. UU.», con `regions=eu` en el ejemplo oficial. Con 500 créditos al mes solo cabe una muestra (p. ej. 1 mercado NBA × 1 región en ~15 partidos al día durante 1 mes). No sirve para todo el calendario.
- **OddsPapi**: plan gratis con `/historical-odds` (es lo único gratuito con cuotas históricas, aunque con 3 casas por llamada) y casas de esports (GG.BET, Pinnacle). La cuota mensual no está clara (250 o 1000). Encaja para un backtest puntual de ROI en esports.
- **odds-api.io**: desde el 26-jul-2026 el plan gratis es «para pruebas»: 500 peticiones/día y sin casas sharp ni exchanges. Tiene CS2 y LoL.
- **Kalshi / Polymarket**: lectura pública sin clave y documentada. Polymarket tiene partidos de esports (CS2, LoL, Dota) y de NBA/NFL/fútbol. Kalshi tiene partidos y algunos props de jugador NBA/NFL. El precio es probabilidad pura: buen «mercado» para comparar, aunque no sea una casa de apuestas.

### 5. Esports y esoccer: otras cuotas públicas
- Lo actual (bo3.gg + Thunderpick y Kambi para esoccer) sigue siendo lo mejor sin clave. **No hay fuente gratis confirmada de cuotas de kills por jugador en CS2**: los props de esports son raros y casi siempre de proveedores de pago. Opciones: OddsPapi (dice tener props de GG.BET) ⚠️, y Kambi `listView/esports` (registrar etiquetas para ver si publica «kills del jugador») ⚠️.
- PandaScore: su plan gratis no da cuotas y sus términos prohíben usarlo para apostar. HLTV tiene Cloudflare y lo prohíbe. Liquipedia tiene API con normas estrictas (User-Agent propio, 1 petición cada 2 s y 1 cada 30 s para `parse`), pero no da cuotas.
- Esoccer: aparte de Kambi solo hay agregadores de pago (BetsAPI…) o Pinnacle (no permitido sin cuenta).

## Recomendación: las 3 primeras a integrar
1. **Props de fútbol de Kambi**: no cuesta peticiones nuevas. Parsear en `kambi.parse` los criterios `Player's shots on target`, `To Score`, `To give an assist` y `To Get a Card` (con jugador y línea) y guardarlos como `kambi_stats_odds.csv` (registro sin stake). Ampliar `save_labels` a baloncesto, hockey, béisbol y esports para descubrir sus props.
2. **ESPN fútbol summary**: crear `data/<liga>/players.csv` (jugador, minutos, tiros, tiros a puerta, goles, asistencias, tarjetas, paradas) y `officials.csv` para las ~27 ligas, 3 temporadas como en nba_players. Con eso: modelo de tiros a puerta y goles por jugador frente a las cuotas del n.º 1 (liquidación con ESPN, tras comprobar que cuenta igual que Opta), portero → córners y árbitro en todas las ligas.
3. **Columnas extra del boxscore ESPN NBA/NHL/MLB**: REB, AST, 3PT; S del patinador; K/IP del abridor. Con el n.º 1 en NBA, o una muestra de The Odds API con clave gratis como secreto, se mide el ROI de los props NBA.

## Script de sondeo (ejecutar una vez en GitHub Actions; aquí estaba bloqueado)
```python
import requests, json
E = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
ev = requests.get(f"{E}/scoreboard", params={"dates": "20251101"}).json()["events"][0]["id"]
d = requests.get(f"{E}/summary", params={"event": ev}).json()
print(d.keys(), d["gameInfo"].get("officials"))
p = d["rosters"][0]["roster"][0]; print(p["athlete"]["displayName"], [(s["name"], s["value"]) for s in p.get("stats", [])])
K = "https://eu-offering-api.kambicdn.com/offering/v2018/ub"
lv = requests.get(f"{K}/listView/basketball/nba/all/all/matches.json", params={"lang": "en_GB", "market": "GB"}).json()
eid = lv["events"][0]["event"]["id"]
bo = requests.get(f"{K}/betoffer/event/{eid}.json", params={"lang": "en_GB", "market": "GB"}).json()["betOffers"]
print(sorted({b["criterion"]["englishLabel"] for b in bo})); print(json.dumps(bo[-1]["outcomes"][0])[:400])
```

## Fuentes
- soccerdata (ESPN): https://github.com/probberechts/soccerdata (`soccerdata/espn.py`, `docs/datasources/ESPN.ipynb`)
- Endpoints ESPN no oficiales: https://github.com/pseudo-r/Public-ESPN-API (README, `docs/sports/soccer.md`, `PROJECT_SCOPE.md`)
- StatsBomb open data: https://github.com/statsbomb/open-data
- FBref sin Opta: https://awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html ; política de bots: https://www.sports-reference.com/bot-traffic.html , https://www.sports-reference.com/data_use.html
- The Odds API (EPL, props): https://the-odds-api.com/sports/epl-odds.html ; límites: https://oddspapi.io/blog/the-odds-api-free-tier-limits
- odds-api.io (cambio del plan gratis): https://docs.odds-api.io/changelog ; OddsPapi esports: https://oddspapi.io/blog/pandascore-api-alternative-free-esports-odds/
- Polymarket: https://docs.polymarket.com/market-data/overview ; Kalshi: https://the-obstacle-is-the-way.github.io/kalshi-starter-code-python/_vendor-docs/kalshi-api-reference
- NHL: https://github.com/Zmalski/NHL-API-Reference ; MLB (aviso MLBAM): https://github.com/timothyf/gameday_api/blob/master/README.rdoc
- nba.com bloquea IPs de nube: https://github.com/bttmly/nba/issues/41
- Kambi y los props NBA: https://sbcnews.co.uk/technology/2018/10/17/kambi-seeks-to-become-clutch-player-of-nba-betting/
- Esports props escasos: https://sportsapis.dev/esports-odds-api ; OpenDota: https://blog.opendota.com/2018/04/17/changes-to-the-api/
- Understat: https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html
