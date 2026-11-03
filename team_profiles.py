"""Perfiles de estadísticas por equipo (10 años de data/hist, football-data.co.uk) para el grafo (data/grafo.json).

Por equipo y condición (local / visitante): partidos, media a favor y en contra de goles, tiros, tiros a puerta,
córners, faltas y tarjetas; % de partidos por encima de la línea de equipo de su liga (mediana de los 200 anteriores,
a .5); forma = media de los últimos 10 partidos menos la de 10 años (a favor). knowledge.Graph.save los pega en el
atributo `stats` de cada nodo «equipo».
`python team_profiles.py` → prueba fuera de muestra de si la forma reciente aporta algo sobre el modelo de
football_stats (logística sobre logit(p_modelo) ajustada en el año −3, medida en las 2 últimas temporadas)."""
import json

import numpy as np
import pandas as pd

import analyst as A
import football_stats as FS

STATS = {"goles": ("FTHG", "FTAG"), "tiros": ("HS", "AS"), "tiros_puerta": ("HST", "AST"), "corners": ("HC", "AC"),
         "faltas": ("HF", "AF"), "tarjetas": ("HY", "AY")}
YEARS, FORM = 10, 10


def long_format(df):
    """Una fila por equipo y partido: fecha, liga, equipo, condición, <stat>_f (a favor) y <stat>_c (en contra)."""
    df = FS.prep(df)
    for c in ("HS", "AS"): df[c] = pd.to_numeric(df.get(c), errors="coerce")
    out = []
    for cond, me, op, side in (("local", "HomeTeam", "AwayTeam", 0), ("visitante", "AwayTeam", "HomeTeam", 1)):
        x = pd.DataFrame({"Date": df.Date, "Div": df.Div, "team": df[me], "rival": df[op], "cond": cond, "match": df.index})
        for s, cols in STATS.items():
            x[f"{s}_f"], x[f"{s}_c"] = df[cols[side]].to_numpy(float), df[cols[1 - side]].to_numpy(float)
        out.append(x)
    return pd.concat(out, ignore_index=True).sort_values(["Date", "match"], kind="stable", ignore_index=True)


def profiles(df, today=None):
    """{equipo: {"local": {...}, "visitante": {...}, "forma10": {...}, "partidos": n}} con los últimos 10 años."""
    today = today or df.Date.max()
    L = long_format(df[(df.Date > today - pd.DateOffset(years=YEARS)) & (df.Date <= today)])
    # línea de equipo de cada liga (como la de las casas): mediana de los 200 valores anteriores, a .5
    for s in STATS:
        L[f"{s}_line"] = L.groupby("Div")[f"{s}_f"].transform(FS.median_line)
    out = {}
    for team, d in L.groupby("team"):
        p = {"partidos": int(len(d)), "desde": f"{d.Date.min():%Y-%m}", "liga": d.Div.iloc[-1]}
        for cond, c in d.groupby("cond"):
            q = {"n": int(len(c))}
            for s in STATS:
                f, ln = c[f"{s}_f"], c[f"{s}_line"]
                if f.notna().sum() < 10: continue
                ok = f.notna() & ln.notna()
                q[s] = [round(float(f.mean()), 2), round(float(c[f"{s}_c"].mean()), 2),
                        round(float((f[ok] > ln[ok]).mean()), 2) if ok.sum() >= 10 else None]  # a favor, en contra, % más de
            p[cond] = q
        last = d.tail(FORM)
        p["forma10"] = {s: round(float(last[f"{s}_f"].mean() - d[f"{s}_f"].mean()), 2) for s in STATS
                        if last[f"{s}_f"].notna().sum() >= 5}
        out[team] = p
    return out


OUT = A.DATA / "team_profiles.json"


def load_hist():
    return pd.concat([A.load_csv(f.read_bytes(), A.COLS + FS.COLS + ["HS", "AS"]) for f in sorted(A.HIST.glob("*.csv"))],
                     ignore_index=True).dropna(subset=["FTHG", "FTAG"]).sort_values("Date", ignore_index=True)


def refresh(max_age_h=20):
    """Perfiles de data/team_profiles.json, recalculados (~10 s) si tienen más de max_age_h horas. {} si falla."""
    import time
    try:
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        if time.time() - old.get("_ts", 0) > max_age_h * 3600:  # la hora va dentro: en Actions cada clon es «nuevo»
            old = {"_ts": int(time.time()), **profiles(load_hist())}
            OUT.write_text(json.dumps(old, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return old
    except Exception as e:
        print("perfiles de equipo", repr(e)); return {}


def card_line(prof, team):
    """Texto corto del perfil (para tarjetas): «Córners 5.8/4.1 (58 % más de) · …» de la condición más reciente."""
    p = prof.get(team)
    if not p: return ""
    parts = [f"{s.replace('_', ' ')} {v[0]:g}/{v[1]:g}" for s, v in p.get("local", {}).items() if isinstance(v, list)]
    return f"{team} en casa (10 años, a favor/en contra): " + " · ".join(parts)


def backtest():
    """¿Aporta la forma de 10 partidos sobre el modelo de football_stats? Más/menos por equipo a la línea mediana."""
    import train_sgp as S
    V, test, fit = S.load()
    df = FS.prep(pd.concat([A.load_csv(f.read_bytes(), A.COLS + FS.COLS) for f in sorted(A.HIST.glob("*.csv"))], ignore_index=True))
    d = df.dropna(subset=["FTHG", "FTAG", *FS.COLS[:4], "HF", "AF", "HST", "AST"]).sort_values("Date", ignore_index=True)
    since = d.Date.max() - pd.DateOffset(years=3)
    keep = (d.Date >= since).to_numpy()
    res = {}
    for st, (h, a, _) in {"goals": ("FTHG", "FTAG", ""), **FS.STATS}.items():
        # forma = media de los últimos 10 partidos del equipo (en cualquier condición) − media de sus 2 años previos
        x = pd.DataFrame({"team": np.r_[d.HomeTeam, d.AwayTeam], "v": np.r_[d[h], d[a]], "i": np.r_[d.index, d.index],
                          "home": np.r_[np.ones(len(d)), np.zeros(len(d))]}).sort_values(["i", "home"], kind="stable")
        g = x.groupby("team").v
        x["form"] = g.transform(lambda s: s.shift(1).rolling(FORM, min_periods=5).mean()) - \
                    g.transform(lambda s: s.shift(1).rolling(76, min_periods=20).mean())
        for sc in ("h", "a"):
            fm = x[x.home == (1 if sc == "h" else 0)].set_index("i").form.reindex(d.index).to_numpy()[keep]
            mu, r, y, line = V[f"{st}_{sc}"]
            p = np.clip(S.p_over(V[f"{st}_{sc}"]), 1e-6, 1 - 1e-6); o = (y > line).astype(float)
            ok = np.isfinite(fm); F_, T_ = fit & ok, test & ok
            sd = fm[F_].std()
            X = lambda m: np.c_[np.log(p[m] / (1 - p[m])), fm[m] / sd]
            from sklearn.linear_model import LogisticRegression
            m0 = LogisticRegression(C=1e6).fit(X(F_)[:, :1], o[F_]); m1 = LogisticRegression(C=1e6).fit(X(F_), o[F_])
            ll = lambda m, Z: -(o[T_] * np.log(m.predict_proba(Z)[:, 1]) + (1 - o[T_]) * np.log(m.predict_proba(Z)[:, 0]))
            l0, l1 = ll(m0, X(T_)[:, :1]), ll(m1, X(T_))
            res[f"{st}_{sc}"] = (round(float(l0.mean()), 4), round(float(l1.mean()), 4),
                                 round(float((l1 - l0).std() / np.sqrt(len(l0))), 4), round(float(m1.coef_[0][1]), 3))
            print(f"{st}_{sc}: log-loss modelo recalibrado {l0.mean():.4f} → + forma 10 {l1.mean():.4f} "
                  f"({l1.mean() - l0.mean():+.4f} ± {res[f'{st}_{sc}'][2]}), coef. forma {m1.coef_[0][1]:+.3f}")
    return res


if __name__ == "__main__":
    backtest()
