# Plan de entrenamiento del bot (lo sigue el agente en cada «siguiente iteración», sin pedir nada al usuario)

Regla fija: solo entra lo que mejora fuera de muestra (walk-forward) y, para apostar, lo que gana a la cuota real.
Informe al usuario: una línea por mejora con 🟢 (conseguido) o 🔴 (no mejora / descartado).

## Ciclo de cada iteración (≤ 45 min, pocos tokens)
1. `git pull`; leer `data/expertos.json` (parte de los bots expertos) y el final de MEJORAS.md.
2. Si algún experto o workflow falla → arreglarlo primero.
3. Tomar la primera tarea pendiente de la cola, hacerla, test, MEJORAS.md, push.

## Cola (en orden; marcar [x] al terminar)
- [ ] Esports/FC 27 vs casa: repetir con ≥ 200 partidos (CS2/Dota) y ≥ 600 (FC 27); si el modelo bate a la casa en 2 mitades, quitar sin_stake.
- [x] Esports: Unibet como 2.ª casa (cuotas de CS2/LoL/Dota/Valorant) → más partidos con cuota real y mejor cuota.
- [x] Esports: sondeo → bo3.gg solo tiene CS2/LoL/Dota/R6/MLBB (ya cubiertas); Unibet además CoD, Overwatch 2, Honor of Kings (sin fuente gratuita de resultados).
- [ ] Props: con ≥ 200 liquidadas, decidir por estadística (NBA y fútbol) si se apuestan (ROI − 2 e.t. > 0).
- [x] Fútbol: más ligas de ESPN en espn_soccer.LEAGUES (las 11 «nuevas» del bot).
- [ ] Fútbol: modelo de estadísticas (córners/tarjetas/tiros) para las 11 ligas nuevas con los datos de ESPN.
- [x] Fútbol: portero → córners descartado (sin efecto). Árbitro: ESPN solo lo publica en ~20 %; reintentar con más historial.
- [ ] Props de CS2/Dota/LoL (kills) contra cuotas reales si Unibet/Thunderpick las publican.
- [x] NFL con nflverse: con 1.808 partidos la casa gana (la ventaja de 263 era ruido) → NFL sin apuestas. Champions: revalidar con más partidos.
- [ ] Revisar semanalmente el banco de pruebas y las estrategias; ajustar MAX_ODDS y segmentos con datos nuevos.

## Fuentes de datos nuevas (sondeo desde Actions, 4-nov: gratis y sin clave)
- [x] Pinnacle (cuotas de cierre, todos los deportes) → pinnacle.py, experto datos_pinnacle
- [x] Pinnacle como referencia del valor de las piernas «gana X» del chat (31 de 85 casan)
- [ ] Pinnacle en el juez del mercado (cierre de Pinnacle frente al modelo, por liga) cuando haya ≥ 200 partidos guardados
- [x] FotMob recolector (xG y estadísticas por partido) → [ ] usar el xG en el 1X2 y en córners/tiros · ~~Understat~~ (4-nov: sirve una página vacía a cualquier petición desde Actions)
- [x] OpenDota → fuentes.py, experto datos_opendota (500 partidas en la prueba; 40 páginas por pasada)
- [x] OpenDota en el Elo de Dota 2: medido, no mejora (0.6597 vs 0.5894 de bo3.gg; mezcla w=0). Reintentar con héroes/jugadores · gol.gg (LoL) · Liquipedia (calendarios esports, con límite de peticiones)
- [ ] API oficial NHL (api-web.nhle.com) · MLB statsapi · StatsBomb open data (histórico de eventos) · OpenLigaDB (Bundesliga)
- Bloqueadas desde Actions: SofaScore, FBref, HLTV, NBA (cdn/stats), Betfair, Oracle's Elixir, TheSportsDB (clave «3» retirada)

## Estrategias (laboratorio.py, semanal)
- [x] Laboratorio con ~30 estrategias y validación antes/después de 2021 + CLV
- [ ] Más casas en vivo para comparar cuotas (la ventaja viene de coger la mejor): Unibet + DraftKings (ESPN) + Thunderpick + las que sondee el agente de plataformas
- [ ] Vigilar el CLV real de «valor contra Pinnacle» (⚽💎) con ≥ 50 apuestas; si CLV < 0, apagarla
- [ ] Añadir al laboratorio: córners/tarjetas contra Pinnacle (si sus cuotas se guardan), NBA/NHL con odds_*.csv + cierre Pinnacle
