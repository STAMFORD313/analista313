"""Features previas al partido sacadas de los propios CSV de football-data (sin fuentes nuevas ni peticiones):
Elo propio con margen de goles, media exponencial de tiros a puerta (HST/AST: proxy de xG, solo ligas de data/hist)
y días de descanso. Todo usa solo partidos anteriores, así que vale para backtest y para predecir."""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd

ELO_K, ELO_HFA, SOT_ALPHA = 20, 60, 0.15
COEF = Path(__file__).parent / "data" / "features_extra.json"  # lo escribe backtest_datos.py
FEATS = ["elo_diff", "sot_diff", "rest_diff"]


def design(P, f, std):
    """Matriz de la regresión: log-probs del Poisson (H, D, A), features escaladas y si hay datos de tiros."""
    return np.column_stack([np.log(np.clip(P, 1e-9, 1)), *(f[c].fillna(0).to_numpy() / std[c] for c in FEATS),
                            f.sot_diff.notna().to_numpy(float)])


def adjust(P, f, coef=None):
    """P (n, 3) probabilidades 1X2 del Poisson + f = features(...) de esos partidos → probabilidades corregidas (n, 3)."""
    c = coef or json.loads(COEF.read_text())
    z = design(np.asarray(P, float), f, c["std"]) @ np.array(c["coef"]).T + np.array(c["intercept"])
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def features(df):
    """df ordenado por fecha con HomeTeam, AwayTeam, FTHG, FTAG, Date y opcional HST/AST →
    DataFrame (mismo índice) con elo_diff, sot_diff, rest_diff (local menos visitante, antes del partido).
    shortcut: los nombres de equipo se tratan como globales (sirve entre divisiones de un país; colisiones entre países raras)."""
    elo, sf, sa, last = {}, {}, {}, {}
    hst = df["HST"].to_numpy(float) if "HST" in df else np.full(len(df), np.nan)
    ast = df["AST"].to_numpy(float) if "AST" in df else np.full(len(df), np.nan)
    out = np.full((len(df), 3), np.nan)
    for i, (h, a, hg, ag, d) in enumerate(zip(df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG, df.Date)):
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        sot = (sf[h] - sa[h]) - (sf[a] - sa[a]) if h in sf and a in sf else np.nan
        rest = min((d - last[h]).days, 21) - min((d - last[a]).days, 21) if h in last and a in last else np.nan
        out[i] = eh - ea, sot, rest
        if pd.isna(hg) or pd.isna(ag): continue
        exp = 1 / (1 + 10 ** (-(eh + ELO_HFA - ea) / 400))
        gd = abs(hg - ag)
        g = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = ELO_K * g * ((hg > ag) + 0.5 * (hg == ag) - exp)
        elo[h], elo[a] = eh + delta, ea - delta
        if not (math.isnan(hst[i]) or math.isnan(ast[i])):
            for t, f, c in ((h, hst[i], ast[i]), (a, ast[i], hst[i])):
                sf[t] = sf.get(t, f) + SOT_ALPHA * (f - sf.get(t, f))
                sa[t] = sa.get(t, c) + SOT_ALPHA * (c - sa.get(t, c))
        last[h] = last[a] = d
    return pd.DataFrame(out, index=df.index, columns=["elo_diff", "sot_diff", "rest_diff"])


if __name__ == "__main__":  # comprobación mínima
    t = pd.DataFrame({"HomeTeam": ["A", "B", "A"], "AwayTeam": ["B", "A", "B"], "FTHG": [3, 0, 1], "FTAG": [0, 2, 1],
                      "HST": [8, 2, 5], "AST": [1, 6, 5], "Date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-12"])})
    f = features(t)
    assert f.elo_diff[0] == 0 and f.elo_diff[2] > 0 and np.isnan(f.sot_diff[0]) and f.sot_diff[2] > 0 and f.rest_diff[2] == 0
    print(f)
