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

## Competición
- [agente-datos] 1 mejora medida (features Elo+tiros+descanso sobre Poisson).
