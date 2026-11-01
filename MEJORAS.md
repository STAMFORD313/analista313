# Mejoras del modelo (backtest sin mirar el futuro, 2 últimas temporadas)

- 2026-11-01 · Fútbol (train_football.py → data/football_params.json): vida media 365→540 d, k=5, RHO -0.10→-0.05. Log-loss 1X2 modelo 1.0243→1.0237 (mezcla 50/50 con mercado 1.0075→1.0073; mercado solo 1.0002). Peso 0.5 y EDGE 0.05 se mantienen: el mejor ROI (peso 0.5, EDGE 0.02: -6.7 %) no supera el ruido frente a -8.1 % (3.462 apuestas a cuota media). A cuota máxima el ROI es ≈ -0.5 %: sin ventaja real sobre el mercado.
- 2026-11-01 · Elo ESPN (train_espn.py → data/espn_params.json), log-loss antes→después: NBA K20/HFA100→K24/HFA35 0.6332→0.6192; NFL K20/55→K40/15 0.6505→0.6382; NHL K8/50→K6/35 0.6792→0.6777; MLB K4/24→K2/25 0.6831→0.6830. ROI no medible: el historial ESPN no guarda cuotas.
