# Coordinación entre agentes

- [agente-modelos] edito analyst.py (solo parámetros de fútbol: carga data/football_params.json → EDGE, BLEND, HALF_LIFE, SHRINK, RHO), multisport.py (carga data/espn_params.json; elo() admite `pred`), train_football.py, train_espn.py.
- [agente-modelos] hallazgo: en las 2 últimas temporadas el mercado sin margen (log-loss 1X2 1.0002) gana a cualquier mezcla con el Poisson (1.0073 con peso 0.5); ROI a cuota media ≈ -8 % con cualquier peso/EDGE (a la cuota máxima ≈ -0.5 %). El modelo de fútbol no tiene ventaja real sobre el mercado.
- [agente-cs2] edito esports.py, train_cs2.py, data/cs2/* (nuevo data/cs2/games.csv con mapas/rondas) y MEJORAS.md (sección CS2).
- [agente-cs2] hallazgo: en partidos TERMINADOS de bo3.gg, bet_updates trae la cuota del final (p. ej. 1.01/14.0), no la previa: no sirve para calibrar sin más.
