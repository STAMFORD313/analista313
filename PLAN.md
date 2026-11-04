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
