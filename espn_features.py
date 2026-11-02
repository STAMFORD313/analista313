"""Features para NBA/NHL/NFL/MLB desde data/espn (sin peticiones): descanso, back-to-back y diferencia de puntos media
reciente (local menos visitante, antes del partido). adjust() corrige la prob. local del Elo de multisport.py con una
regresión logística; `python espn_features.py` hace el backtest (ajusta con todo lo anterior a las 2 últimas temporadas,
evalúa en ellas) y guarda los coeficientes en data/espn_features.json."""
import json

import numpy as np
import pandas as pd

COEF = __import__("pathlib").Path(__file__).parent / "data" / "espn_features.json"
FEATS = ["rest_diff", "b2b_home", "b2b_away", "mov_diff"]
MOV_ALPHA = 0.1


def features(df):
    """df ordenado por fecha (home, away, hs, as, date) → DataFrame con FEATS. Filas sin marcador no actualizan nada."""
    last, mov = {}, {}
    out = np.zeros((len(df), 4))
    for i, (h, a, hs, as_, d) in enumerate(zip(df.home, df.away, df.hs, df["as"], df.date)):
        rh, ra = ((d - last[t]).total_seconds() / 86400 if t in last else 7.0 for t in (h, a))
        out[i] = min(rh, 7) - min(ra, 7), rh < 1.6, ra < 1.6, mov.get(h, 0.0) - mov.get(a, 0.0)
        if pd.isna(hs) or pd.isna(as_): continue
        for t, m in ((h, hs - as_), (a, as_ - hs)):
            mov[t] = mov.get(t, 0.0) + MOV_ALPHA * (m - mov.get(t, 0.0))
        last[h] = last[a] = d
    return pd.DataFrame(out, index=df.index, columns=FEATS)


def design(p, f, std):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), *(f[c].to_numpy(float) / std[c] for c in FEATS)])


def adjust(league, p, f, coef=None):
    """p = prob. local del Elo (array), f = features(...) de esos partidos → prob. local corregida.
    Ligas sin coeficientes guardados (tenis, UFC o sin mejora) devuelven p tal cual."""
    c = (coef if coef is not None else json.loads(COEF.read_text()) if COEF.exists() else {}).get(league)
    if not c: return np.asarray(p, float)
    return 1 / (1 + np.exp(-(design(p, f, c["std"]) @ np.array(c["coef"]) + c["intercept"])))


def main():
    from sklearn.linear_model import LogisticRegression
    import multisport as M
    out = {}
    for league in ("nba", "nhl", "nfl", "mlb"):
        _, _, _, hfa, k = M.SPORTS[league]
        P = M._P.get(league, {})
        df = pd.concat([x for f in sorted((M.DIR / league).glob("*.csv")) if not (x := pd.read_csv(f, dtype={"id": str})).empty], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.dropna(subset=["res"]).drop_duplicates("id").sort_values("date", ignore_index=True)
        pred = []
        M.elo(df, P.get("hfa", hfa), P.get("K", k), pred, P.get("keep", 0.75), P.get("kt", 250), P.get("mov", False))
        p, F = np.array(pred), features(df)
        ok = (df.res != 0.5).to_numpy()  # empates (NHL antiguos/NFL): fuera de la regresión binaria
        te = ok & (df.date >= df.date.max() - pd.DateOffset(years=2)).to_numpy()
        tr = ok & ~te
        tr &= (df.date >= df.date.min() + pd.DateOffset(years=1)).to_numpy()  # primer año: ratings sin asentar
        y = df.res.to_numpy(float)
        std = {c: float(F[c].std()) or 1.0 for c in FEATS}
        X = design(p, F, std)
        ll = lambda q: -(y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q))
        base = ll(np.clip(p[te], 1e-6, 1 - 1e-6))
        res = {}
        for name, cols in {"elo recalibrado": [0], "+descanso/b2b": [0, 1, 2, 3], "+dif. puntos": [0, 4], "+todo": [0, 1, 2, 3, 4]}.items():
            q = LogisticRegression(C=10, max_iter=1000).fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
            d = ll(q) - base
            res[name] = d.mean(), d.std() / np.sqrt(len(d))
            print(f"{league} {name:16s} log-loss {base.mean():.4f}→{ll(q).mean():.4f} ({d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}, {te.sum()} partidos)")
        d_all, se = res["+todo"]
        if d_all < -2 * se:  # solo ligas con mejora fuera del ruido (2 errores típicos); coeficientes con todos los datos
            m = LogisticRegression(C=10, max_iter=1000).fit(X[ok & (tr | te)], y[ok & (tr | te)])
            out[league] = {"std": std, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                           "logloss_gain": round(float(d_all), 4)}
            assert np.allclose(adjust(league, p[:20], F[:20], out), m.predict_proba(X[:20])[:, 1])
    COEF.write_text(json.dumps(out))


if __name__ == "__main__":
    t = pd.DataFrame({"home": ["A", "B"], "away": ["B", "A"], "hs": [100, 90], "as": [90, 95],
                      "date": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-02T00:00Z"])})
    f = features(t)
    assert f.b2b_home[1] == 1 and f.b2b_away[1] == 1 and f.mov_diff[1] == -2.0 and f.rest_diff[0] == 0
    main()
