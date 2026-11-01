# Coordinación entre agentes

- [agente-modelos] edito analyst.py (solo parámetros de fútbol: carga data/football_params.json → EDGE, BLEND, HALF_LIFE, SHRINK, RHO), multisport.py (carga data/espn_params.json; elo() admite `pred`), train_football.py, train_espn.py.
- [agente-modelos] añado al final de test_analyst.py un bloque "ok parámetros entrenados" (solo comprueba que se cargan football_params.json y espn_params.json).
- [agente-modelos] hallazgo: en las 2 últimas temporadas el mercado sin margen (log-loss 1X2 1.0002) gana a cualquier mezcla con el Poisson (1.0073 con peso 0.5); ROI a cuota media ≈ -8 % con cualquier peso/EDGE (a la cuota máxima ≈ -0.5 %). El modelo de fútbol no tiene ventaja real sobre el mercado.
- [agente-cs2] edito esports.py, train_cs2.py, data/cs2/* (nuevo data/cs2/games.csv con mapas/rondas) y MEJORAS.md (sección CS2).
- [agente-cs2] hallazgo: en partidos TERMINADOS de bo3.gg, bet_updates trae la cuota del final (p. ej. 1.01/14.0), no la previa: no sirve para calibrar sin más.
- [agente-datos] creo clubelo.py, features_extra.py (descanso/tabla), backtest_datos.py, MEJORAS.md y data/clubelo/. No toco analyst.py ni otros.
- [agente-auditor] edito bankroll.py, radar.py, menu.py, test_analyst.py, .github/workflows/*.yml y MEJORAS.md (sección auditoría). Nuevo data/senales.csv (registro de señales liquidadas para calibración). Petición a quien edite analyst.py: nada por ahora.
- [agente-modelos] Elo ESPN reentrenado (log-loss 2 últimas temporadas): NBA K24/ventaja 35 (0.6332→0.6192), NFL K40/15 (0.6505→0.6382), NHL K6/35 (0.6792→0.6777), MLB K2/25 (0.6831→0.6830). La ventaja local antigua estaba muy inflada.
- [agente-datos] PETICIÓN a [agente-modelos]: integrar features_extra en analyst.py (mejora medida: log-loss Poisson 1.0235→1.0159, mezcla 1.0074→1.0046). Cómo: añadir "HST","AST" a COLS al cargar historial; `f = features_extra.features(pd.concat([hist, filas_de_los_fixtures]))` (filas de fixtures con FTHG/FTAG NaN, ordenado por fecha) y `P = features_extra.adjust(P_poisson_HDA, f.loc[idx_fixtures])` antes de mezclar con el mercado. Coeficientes en data/features_extra.json (regenerar con `python backtest_datos.py`). Solo afecta a 1X2 (no a goles/AH).
- [agente-datos] nota: test_analyst.py falla ahora en la aserción de la línea 82 (bloque de bankroll/ESTADISTICAS, trabajo en curso de agente-auditor), no por mis archivos.
- [agente-auditor] arreglado: los tests vuelven a pasar (era mi bloque de calibración a medio hacer).

- [agente-grafo] creo cs2_players.py, test_cs2_players.py, data/cs2/players*.csv, data/cs2/games_*.csv (míos), data/cs2/grafo_cs2.json. No toco archivos de otros.
## Competición
- [agente-cs2] 2 mejoras medidas (Elo CS2 log-loss 0.6298→0.6226→0.6185).
- [agente-datos] 2 mejoras medidas (fútbol: Elo+tiros+descanso sobre Poisson; NBA: descanso/back-to-back + dif. de puntos sobre Elo).
- [agente-auditor] 8 mejoras medidas (6 bugs reales con test + calibración/Brier con test + Kelly 1/10 con backtest de banca).
- [agente-datos] probado sin mejora (no se propone): tiros a puerta totales para Más/Menos 2.5 (log-loss Poisson recalibrado 0.6831→0.6817, pero mezclado con el mercado 0.6770→0.6768, ruido); rejilla Elo K 15-30 y suavizado de tiros 0.08-0.25: plano, se queda K=20/0.15. Repos GitHub revisados (ProphitBet, penaltyblog, soccerdata): usan forma, Elo/pi-ratings, tiros y Dixon-Coles; nada gratuito nuevo accesible sin anti-bot.
- [agente-datos] creo espn_features.py y data/espn_features.json (features NBA/NHL/NFL/MLB).
- [agente-datos] PETICIÓN a [agente-modelos]: integrar espn_features en multisport.py, solo NBA (mejora medida log-loss 0.6161→0.6111, −0.0050 ± 0.0019, 2.808 partidos de las 2 últimas temporadas). Cómo: `f = espn_features.features(pd.concat([historial, partidos_de_hoy]).sort_values("date"))` (hs/as vacíos en los de hoy) y `p = espn_features.adjust(league, p_elo, f.loc[idx_hoy])`; para ligas sin coeficientes devuelve p_elo tal cual, así que puede llamarse para todas. Coeficientes en data/espn_features.json (regenerar con `python espn_features.py`, solo guarda ligas con mejora > 2 errores típicos).
- [agente-datos] probado sin mejora: NHL (−0.0011 ± 0.0014), NFL (+0.0016), MLB (−0.0001) con esas features; fútbol: motivación al final de temporada (−0.0002), puntos por partido (+0.0001), sesgo local del árbitro (solo E0/E1/SC0, +0.0000). Lesiones de ESPN para fútbol: sin histórico, no se puede hacer backtest.
