# Investigación GitHub: proyectos open source de apuestas y predicción deportiva

2026-11-03 · [agente-investigación] · Solo lectura: no toca el código del bot. Antes se leyeron MEJORAS.md y COORDINACION.md para no repetir lo ya probado. Ya estaba hecho o descartado: ClubElo API, understat/fbref, valor frente a Pinnacle, Platt sobre modelo+mercado, bajas NBA/NHL, abridores MLB, ProphitBet/penaltyblog/soccerdata (revisión superficial) y combinadas entrenadas.

**Red:** desde esta sesión el proxy bloquea Polymarket, Kalshi, OpenDota, api-web.nhle.com, statsapi.mlb.com, cdn.nba.com y MoneyPuck (403), así que no las he podido probar. Hay que hacerlo desde GitHub Actions, igual que pasó con Kambi. nflverse (releases de GitHub) sí se ha descargado y comprobado.

## 1. Repos más útiles

| Repo | ★ | Licencia | Última act. | Qué aporta al bot |
|---|---:|---|---|---|
| [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) | 248 | MIT | 2026-10-19 | 7 métodos para quitar el margen (multiplicativo, aditivo, power, **Shin**, DMW, odds-ratio, logarítmico); `goal_expectancy` (λ local/visitante deducidos del 1X2); pi-ratings, Massey y Colley; Poisson bivariante, cópula Weibull, ZIP y NB; RPS. Se puede copiar código. |
| [mberk/shin](https://github.com/mberk/shin) | 105 | MIT | 2026-08-04 | Método de Shin en ~50 líneas, con diagnóstico de convergencia. Con 2 resultados equivale al aditivo. |
| [noushiFlex/orion](https://github.com/noushiFlex/orion) | 1 | MIT | 2026-10-27 | Pocas estrellas, pero es el más **honesto** que he visto: walk-forward por temporada contra el **cierre de Pinnacle**, con tests de fuga y una tabla de qué aporta cada versión. Rating online con xG: 0.9801→0.9780. Cuotas previas como feature: →0.9687. «Desacuerdo entre casas»: →0.9647. Habituales lesionados: →0.9635. Árbitro, suspensiones y entrenador: nada. Conclusión suya: nivel de mercado, sin ventaja (+0.0004 ± 0.0021 frente al cierre). |
| [machina-sports/sports-skills](https://github.com/machina-sports/sports-skills) | 262 | MIT | 2026-11-02 | Catálogo de **endpoints gratuitos sin clave**: Polymarket (gamma + CLOB `prices-history`), Kalshi (candlesticks; series KXNBA/KXNFL/KXNHL/KXMLB, EPL/UCL y **esports CS2/LoL/Dota2**), ProphetX, NHL `api-web.nhle.com`, MLB `statsapi.mlb.com`, NBA CDN, OpenDota y Leaguepedia Cargo (CC-BY-SA). |
| [nflverse/nflverse-data](https://github.com/nflverse/nflverse-data) | 441 | CC-BY-4.0 (datos) | 2026-11-01 | `schedules/games.csv`, **verificado**: 7.548 partidos desde 1999. **Moneyline, spread y total de cierre** desde 2006 (100 % desde 2007), QB titular de cada equipo (id), descanso, techo y superficie, viento y temperatura (~67 %) y árbitro. |
| [nflverse/nflreadpy](https://github.com/nflverse/nflreadpy) | 249 | MIT | 2026-10-30 | Cliente Python de lo anterior (play-by-play, EPA, snaps y lesiones semanales). |
| [betcode-org/flumine](https://github.com/betcode-org/flumine) | 251 | MIT | 2026-10-19 | Controles de riesgo (`StrategyExposure`: tope por orden, por selección y por mercado; validación de órdenes), paper trading y simulación con histórico. |
| [betcode-org/betfair](https://github.com/betcode-org/betfair) | 515 | MIT | 2026-10-13 | Cliente de la API de Betfair. Necesita cuenta y app key, así que no nos sirve directamente. Las cuotas del exchange ya vienen en football-data (`BFE*`, ver mejora 5). |
| [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting) | 830 | MIT | 2026-09-24 | Dataloaders (football-data) y «bettors» sobre scikit-learn con backtest por series temporales. Es una referencia de arquitectura, no un modelo nuevo. |
| [gotoConversion/goto_conversion](https://github.com/gotoConversion/goto_conversion) | 117 | MIT | 2026-09-06 | Otro método para quitar el margen (ajusta según el error típico de cada prob.), usado en soluciones premiadas de Kaggle March Madness. Es una alternativa a Shin. |
| [xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data) | 220 | MIT | 2026-09-06 | Volcado de **ClubElo** en CSV (273.972 snapshots) junto a football-data. Sustituye a la API de ClubElo que daba 502, pero solo es real hasta jun-2025 (después es «provisional»). |
| [conorwalsh99/ml-for-sports-betting](https://github.com/conorwalsh99/ml-for-sports-betting) | – | MIT | 2025 (archivado) | Código del artículo de Walsh & Joshi (2024, *ML with Applications*): elegir el modelo por **calibración** en vez de por acierto da mucho más ROI en la NBA (según el artículo). Respalda nuestra regla de elegir por log-loss. |
| [plwp/afl-betting](https://github.com/plwp/afl-betting) | 1 | GPL-3.0 | 2026-04 | Resultado nulo publicado con honestidad: +12.8 % de ROI en 74 apuestas pero **CLV −0.009**, que atribuyen al ruido. Las features de movimiento de cuota (apertura→cierre) **empeoraron** el walk-forward. Sirve como método; el código es GPL y no se puede copiar. |
| [vivekjoshy/openskill.py](https://github.com/vivekjoshy/openskill.py) | 378 | MIT | 2026-05-12 | Rating bayesiano por jugador en equipos (media y σ), más rápido que TrueSkill. Podría servir para el rating de jugadores CS2/Dota/LoL con incertidumbre. |
| [odota/core](https://github.com/odota/core) | 1641 | MIT | 2026-10-30 | API pública de OpenDota (~60 pet./min sin clave): partidos pro con draft, KDA por jugador y patch. |
| [swar/nba_api](https://github.com/swar/nba_api) | 3858 | MIT | 2026-10-28 | stats.nba.com (tracking y props por jugador). Ojo: NBA suele bloquear las IP de la nube (Actions). |
| [sedemmler/WagerBrain](https://github.com/sedemmler/WagerBrain) | 319 | MIT | 2020 | Utilidades de Kelly, conversión de cuotas y arbitraje. Nada que no tengamos. |

**No aplicables o descartados:**
- **Necesitan cuenta o clave:** Betfair API, GRID (esports; lo usa `chdoyle1/lol-betting-pipeline`), The Odds API, Betsapi.
- **Términos de uso o anti-bot:** scrapers de oddsportal (`jordantete/OddsHarvester` MIT, `gingeleski/odds-portal-scraper`), soccerdata para FBref/Understat/WhoScored (Apache-2.0; ya descartado).
- **Datos congelados:** `dcaribou/transfermarkt-datasets` (CC0, sin actualizar desde jul-2026).
- **Ya no está en GitHub:** `JeffSackmann/tennis_atp`. Hay un espejo CC BY-NC-SA en `Aneeshers/tennis-sackmann-archive`.
- **Sin uso en vivo:** StatsBomb open-data y `ML-KULeuven/soccer_xg` (Apache-2.0) necesitan datos de eventos y solo cubren competiciones sueltas.

**⚠️ Marcados por prometer ROI sin backtest verificable:**
- `kyleskom/NBA-Machine-Learning-Sports-Betting` (1771★, sin licencia): acierto y EV, pero ni CLV ni walk-forward contra el cierre.
- `JoePeirson/esports_betting_model` («gané 20.000 £»).
- `llSourcell/ChatGPT_Sports_Betting_Bot`.
- `Ezee-Kits/AI-SportyBet-Bot-Phone`.
- Los «bots» de Polymarket con nombres repetidos (`huberco/...`).

## 2. Medidas rápidas hechas en esta investigación

Los scripts están en el scratchpad y no tocan el bot. Datos: `data/hist`, temporadas 24/25 y 25/26.

- **Quitar el margen del 1X2 (10.934 partidos), log-loss frente al multiplicativo (normalización proporcional, la que usa el bot):**

  | Cuota | Aditivo | Power | Shin |
  |---|---|---|---|
  | Media apertura | −0.00080 ± 0.00030 | −0.00074 ± 0.00032 | −0.00068 ± 0.00022 |
  | Media cierre | −0.00080 ± 0.00030 | — | −0.00068 ± 0.00021 |
  | Betfair Exchange | — | — | −0.00030 ± 0.00011 |
  | Pinnacle cierre | — | — | −0.00026 ± 0.00017 |

  Cuanto más margen tiene la casa, más se nota. En **Más/Menos 2.5 no cambia nada** (±0.0001), y en NFL a 2 vías (cierre, 3.137 partidos) tampoco (−0.0001 ± 0.0003).
- **Mercado de referencia (1X2 de apertura con Shin, mismos partidos):** Betfair Exchange 0.9900 frente a media de casas 0.9912. Pinnacle (subconjunto) 0.9885; Pinnacle cierre 0.9856.
- **Fuerza implícita en las cuotas como feature de estadísticas:** supremacía log(pH/pA), log-odds de Más 2.5 y |supremacía|.
  - Montaje: Poisson con medias móviles del equipo (EW 0.1, causales) más esas tres variables; ajuste en 22/23-23/24, prueba en 24/25-25/26 (10.932 partidos).
  - Log-verosimilitud por equipo: **córners −0.033 ± 0.003 (local) / −0.029 ± 0.003 (visitante)**, **tiros a puerta −0.033 / −0.027**, tarjetas −0.008 / −0.004, faltas −0.006 / −0.011 (todas > 3 errores típicos).
  - Ojo: la referencia es más simple que `football_stats` (fuerzas local/visitante, binomial negativa, forma). La mejora real será menor y hay que medirla contra el modelo de producción.

## 3. Mejoras propuestas (por prioridad)

1. **Cuotas 1X2 y Más/Menos como feature de córners, tiros a puerta, tarjetas y faltas** (idea de `penaltyblog.goal_expectancy` y orion v1.3).
   - **Qué:** en `football_stats`, multiplicar el esperado de cada equipo por exp(a·supremacía + b·log-odds O2.5 + c·|sup|). Las cuotas salen sin margen de la media de football-data en el histórico y de Kambi/fixtures en vivo.
   - **Por qué:** el mercado resume la diferencia de nivel mejor que las medias de 10 años. El favorito domina, saca más córners y tira más (§2: −0.03 por equipo con una referencia simple). Además es el único mercado de estadísticas que **sí apostamos** (córners de Unibet).
   - **Medir:** coeficientes en el año −3, log-loss más/menos a la línea mediana en las 2 últimas temporadas frente al modelo actual con forma (mismo protocolo que train_form.py). Después, ROI con `data/kambi_stats_odds.csv`.
   - **Coste:** bajo (≈ 1-2 h; las columnas ya están en data/hist).

2. **NFL con nflverse `games.csv`** (CC-BY-4.0; citar a nflverse).
   - **Qué:** (a) **cuotas de cierre históricas** desde 2006 para medir por fin el ROI y el peso modelo/mercado en NFL (MEJORAS: «ROI no medible»); (b) ajuste por **cambio de QB titular** (QB distinto del habitual → −X de Elo, como hacía el Elo de FiveThirtyEight); (c) viento, techo y temperatura para los totales.
   - **Medir:** walk-forward (años −4/−3 eligen, 2 últimas temporadas miden), log-loss frente al Elo de producción (0.6367) y frente al cierre sin margen, y ROI a cuota de cierre. Totales: log-loss más/menos y CRPS.
   - **Coste:** bajo-medio (CSV de 2 MB, sin clave, descarga verificada; el QB probable en vivo sale del scoreboard o del summary de ESPN).

3. **Polymarket y Kalshi como cuota histórica y segunda referencia** (endpoints de `sports-skills`, MIT).
   - **Qué:** precios previos e **historial de precios** (`clob.polymarket.com/prices-history`, candlesticks de Kalshi) de CS2, LoL, Dota 2, NBA, NFL, NHL y MLB.
   - **Por qué:** hoy los esports tienen 49 filas con cuota previa (`data/cs2/odds.csv`) y no hay forma de medir ROI ni elegir el peso de mezcla. Con los mercados ya resueltos se tendrían cientos de partidos de golpe, y también una referencia de cierre con poco margen para el CLV.
   - **Medir:** log-loss del modelo frente al último precio previo, peso óptimo elegido en el año −2 y medido en el −1, y ROI ± error típico.
   - **Antes:** comprobar acceso desde Actions, liquidez (descartar mercados con poco volumen) y términos de uso (lectura pública).
   - **Coste:** medio.

4. **Activar o apagar segmentos de la banca por CLV, no por ROI** (práctica de flumine, orion y afl-betting).
   - **Qué:** hoy `SEG_MIN=50` mira el IC del ROI, que necesita miles de apuestas. El CLV frente al cierre sin margen tiene mucha menos varianza.
   - **Regla propuesta:** un segmento solo recibe stake si el CLV medio es > 0 con 2 errores típicos y ≥ 30 apuestas, y se apaga si el CLV es < 0 con 2 errores típicos.
   - **Medir:** en data/hist, si el CLV del año −2 predice el signo del ROI del año −1 por segmento. Después, en `senales.csv` (ya guarda `close_p`/`clv`).
   - **Coste:** bajo.

5. **Quitar el margen con Shin/aditivo en mercados de 3 resultados** (`mberk/shin` o `penaltyblog.implied`, MIT).
   - **Dónde:** 1X2 de fútbol, Champions/Europa de DK y FC 27 de Unibet. Valen 15 líneas o copiar la función.
   - **Por qué:** medido −0.0007 ± 0.0002 de log-loss. Corrige el sesgo favorito-longshot de la normalización proporcional, que **infla la «ventaja» en longshots**: es justo el tramo 25-40 % donde la auditoría vio sobreconfianza.
   - **De paso:** usar **Betfair Exchange (`BFE*`)** o Pinnacle en vez de la media como «mercado sin margen» del histórico (0.9900 frente a 0.9912).
   - **Medir:** log-loss del mercado y de la mezcla, y ROI de la regla del bot en el año −1.
   - **Coste:** muy bajo.

6. **xG de football-data** (columnas `HxG`/`AxG`, nuevas en 26/27: 1.527 partidos ya, 0 en temporadas anteriores).
   - **Qué:** fuerzas de ataque y defensa online mezclando goles y xG (orion v1.1: −0.0021), y xG como feature de tiros a puerta y córners.
   - **Medir:** walk-forward dentro de la temporada (1.ª mitad ajusta, 2.ª mide) cuando haya ~3.000 partidos (≈ enero 2027).
   - **Coste:** bajo, pero hay que **esperar datos**. Mientras tanto basta con no descartar las columnas al cargar.

7. **Desacuerdo entre casas como feature y como filtro** (orion v1.5: 0.9653→0.9647).
   - **Qué:** dispersión Max−Avg (o B365 frente a BFE) por resultado. Si las casas discrepan mucho, hay información que no ha llegado a todas o el mercado está menos asentado.
   - **Uso:** como feature en la corrección del modelo o como filtro de stake (Kelly más bajo con mucha dispersión).
   - **Medir:** log-loss y ROI en el año −1 con lo ajustado en el −2.
   - **Coste:** bajo. **Esperable:** poco, porque el fútbol está apagado; probarlo en córners/Kambi no es posible (una sola casa).

8. **OpenSkill (MIT) para el rating de jugadores de esports con incertidumbre**.
   - **Qué:** sustituir o complementar el rating «sin olvido» de `cs2_players` por μ/σ por jugador. σ crece con la inactividad y encoge automáticamente a los recién llegados, cosa que hoy se hace con heurísticas.
   - **Medir:** walk-forward honesto ya montado (`cs2_players.py walkforward`), log-loss de serie S/A/B frente a 0.6089 (CS2), 0.5906 (Dota) y 0.5866 (LoL).
   - **Coste:** medio. **Ganancia incierta:** la fuerza de la plantilla ya aporta casi todo.

**Ideas vistas que ya están hechas o descartadas** (no se proponen):
- Pi-ratings, Elo con margen y forma en fútbol 1X2: el mercado gana y los mercados están apagados.
- Modelo «mercado + corrección» al estilo Platt: ya probado (+0.0005-0.0022, sin ROI).
- Movimiento de cuota apertura→cierre como feature: empeora en afl-betting.
- Valor frente a Pinnacle: ya descartado (`pinnacle_value.py`).
- Simulación de Monte Carlo de torneos: solo para mercados de campeón, que no usamos.
