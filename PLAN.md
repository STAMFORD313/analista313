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
- [ ] Esports: añadir las disciplinas que confirme `data/sondeo.json` (Valorant, etc.) con su experto de datos y su Elo.
- [ ] Props: con ≥ 200 liquidadas, decidir por estadística (NBA y fútbol) si se apuestan (ROI − 2 e.t. > 0).
- [x] Fútbol: más ligas de ESPN en espn_soccer.LEAGUES (las 11 «nuevas» del bot).
- [ ] Fútbol: modelo de estadísticas (córners/tarjetas/tiros) para las 11 ligas nuevas con los datos de ESPN.
- [ ] Fútbol: efecto del portero en córners y árbitro en todas las ligas (con el historial de ESPN).
- [ ] Props de CS2/Dota/LoL (kills) contra cuotas reales si Unibet/Thunderpick las publican.
- [ ] NFL y Champions (único mercado US/Europa que bate al cierre): reforzar con datos de jugadores (QB titular, nflverse).
- [ ] Revisar semanalmente el banco de pruebas y las estrategias; ajustar MAX_ODDS y segmentos con datos nuevos.
