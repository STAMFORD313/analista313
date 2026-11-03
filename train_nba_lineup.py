"""NBA: fuerza de la alineación que juega cada partido frente al Elo y frente a las cuotas de CIERRE de DraftKings.

Por jugador, solo con partidos anteriores: +/- por minuto con olvido (encogido a 0 con K minutos) y minutos esperados.
Fuerza de un equipo en un partido = Σ (+/- por minuto × minutos esperados) de los que JUEGAN, escalada a 240 min; y
«bajas» = fuerza de los que juegan − fuerza de su plantilla habitual (sus 9 con más minutos esperados).
shortcut: «los que juegan» = los que tuvieron minutos (la alineación se conoce ~30 min antes; el cierre ya la recoge).
Modelos (logística, ajuste en los partidos SIN cuota previos, prueba en los partidos con cuota de cierre):
  Elo · Elo + alineación · mercado · mercado + alineación (¿aporta algo que la casa no sepa?).
Uso: PYTHONIOENCODING=utf-8 python train_nba_lineup.py → data/nba/lineup_params.json"""
import json, math

import numpy as np
import pandas as pd
from scipy import optimize

import market_check as MC
import multisport as M
import nba_players as N

OUT = M.A.DATA / "nba" / "lineup_params.json"


def features(games, a=0.03, k=300.0, a_min=0.15):
    """{id: (fuerza local, fuerza visitante, bajas local, bajas visitante)} antes de cada partido."""
    R, out = {}, {}
    for g in sorted(games, key=lambda g: g["date"]):
        teams = g.get("teams") or []
        if len(teams) != 2: continue
        f = {}
        for t in teams:
            pl = [p for p in t["players"] if p[2] > 0]
            def rate(pid):
                s = R.get(pid)
                return (s["pm"] / (s["m"] + k), s["em"]) if s else (0.0, 0.0)
            on = [rate(p[0]) for p in pl]
            tot = sum(m for _, m in on) or 1.0
            st = sum(r * m for r, m in on) * 240 / tot
            reg = sorted((s for pid, s in R.items() if s.get("team") == t["team"]), key=lambda s: -s["em"])[:9]
            tr = sum(s["em"] for s in reg) or 1.0
            sf = sum(s["pm"] / (s["m"] + k) * s["em"] for s in reg) * 240 / tr
            f[t["team"]] = (st, st - sf if reg else 0.0)
        if g["home"] in f and g["away"] in f:
            out[g["id"]] = (*[f[g["home"]][0], f[g["away"]][0]], f[g["home"]][1], f[g["away"]][1])
        for t in teams:  # actualizar después del partido
            for p in t["players"]:
                if p[2] <= 0: continue
                s = R.setdefault(p[0], {"pm": 0.0, "m": 0.0, "em": p[2], "team": t["team"]})
                s["pm"] = s["pm"] * (1 - a) + p[4]; s["m"] = s["m"] * (1 - a) + p[2]
                s["em"] += a_min * (p[2] - s["em"]); s["team"] = t["team"]
    return out


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def fit(X, y):
    f = lambda w: -(y * np.log(1 / (1 + np.exp(-X @ w))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-X @ w)))).sum()
    return optimize.minimize(f, np.r_[1.0, np.zeros(X.shape[1] - 1)], method="BFGS").x


def ll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    df = MC.history("nba")
    _, _, _, hfa, k = M.SPORTS["nba"]; c = M._P.get("nba", {})
    pred = []
    M.elo(df, hfa, k, pred, c.get("keep", 0.75), c.get("kt", 250), c.get("mov", False))
    pe = dict(zip(df.id, pred))
    games = N.load_box("nba")
    F = features(games)
    rows = []
    for g in games:
        if g["id"] not in F or g["id"] not in pe or g["hs"] == g["as"]: continue
        o = g.get("odds") or {}
        oh, oa = MC.dec(o.get("ml_h")), MC.dec(o.get("ml_a"))
        pm = (1 / oh) / (1 / oh + 1 / oa) if oh and oa else np.nan
        sh, sa, bh, ba = F[g["id"]]
        rows.append({"date": g["date"], "y": float(g["hs"] > g["as"]), "elo": pe[g["id"]], "mk": pm, "oh": oh, "oa": oa,
                     "fuerza": (sh - sa) / 10, "bajas": (bh - ba) / 10})
    d = pd.DataFrame(rows).sort_values("date", ignore_index=True)
    has = d.mk.notna().to_numpy()
    fi = ~has & (pd.to_datetime(d.date, utc=True) >= pd.to_datetime(d.date, utc=True).max() - pd.DateOffset(years=2)).to_numpy()
    te = has
    y = d.y.to_numpy()
    X_elo = np.c_[logit(d.elo)]; X_lin = np.c_[logit(d.elo), d.fuerza, d.bajas]
    w_elo, w_lin = fit(X_elo[fi], y[fi]), fit(X_lin[fi], y[fi])
    p_elo, p_lin = 1 / (1 + np.exp(-X_elo @ w_elo)), 1 / (1 + np.exp(-X_lin @ w_lin))
    # ¿aporta la alineación algo sobre el MERCADO? (ajuste en la 1.ª mitad de los partidos con cuota, prueba en la 2.ª)
    idx = np.flatnonzero(te); h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    X_mk = np.c_[logit(d.mk.fillna(0.5)), d.fuerza, d.bajas]
    w_mk = fit(X_mk[h1], y[h1]); p_mkl = 1 / (1 + np.exp(-X_mk @ w_mk))
    se = lambda u, v: float((u - v).std() / math.sqrt(len(u)))
    L = {n: ll(p[te], y[te]) for n, p in (("elo", p_elo), ("elo+alineación", p_lin), ("mercado", d.mk.to_numpy()))}
    l2_mk, l2_mkl = ll(d.mk.to_numpy()[h2], y[h2]), ll(p_mkl[h2], y[h2])
    print(f"NBA: ajuste {fi.sum()} partidos sin cuota, prueba {te.sum()} con cuota de cierre DK")
    print(f"  coef. alineación: fuerza {w_lin[1]:+.3f}, bajas {w_lin[2]:+.3f} (por 10 puntos de +/- en 240 min)")
    print(f"  log-loss: Elo {L['elo'].mean():.4f} · Elo+alineación {L['elo+alineación'].mean():.4f} "
          f"({L['elo+alineación'].mean() - L['elo'].mean():+.4f} ± {se(L['elo+alineación'], L['elo']):.4f}) · mercado {L['mercado'].mean():.4f}")
    print(f"  mercado + alineación (2.ª mitad, {len(h2)}): {l2_mk.mean():.4f} → {l2_mkl.mean():.4f} ({l2_mkl.mean() - l2_mk.mean():+.4f} ± {se(l2_mkl, l2_mk):.4f})")
    # ROI 2.ª mitad apostando donde mercado+alineación ve ventaja ≥ 3 % a la cuota de cierre
    q = p_mkl[h2]; oh, oa = d.oh.to_numpy()[h2], d.oa.to_numpy()[h2]
    eh, ea = q * oh - 1, (1 - q) * oa - 1
    side_h = eh >= ea; ev = np.where(side_h, eh, ea); bet = ev >= 0.03
    g = np.where(side_h, np.where(y[h2] == 1, oh - 1, -1), np.where(y[h2] == 0, oa - 1, -1))[bet]
    print(f"  ROI 2.ª mitad (ventaja ≥ 3 %, cuota de cierre): {g.mean() if len(g) else 0:+.1%} ± {g.std() / math.sqrt(max(len(g), 1)):.1%} en {len(g)}")
    res = {"w_elo_lineup": [round(float(x), 4) for x in w_lin], "w_mk_lineup": [round(float(x), 4) for x in w_mk],
           "ll_elo": round(float(L["elo"].mean()), 4), "ll_elo_lineup": round(float(L["elo+alineación"].mean()), 4),
           "ll_mercado": round(float(L["mercado"].mean()), 4), "ll_mk_2a": round(float(l2_mk.mean()), 4), "ll_mk_lineup_2a": round(float(l2_mkl.mean()), 4),
           "se_mk": round(se(l2_mkl, l2_mk), 4), "roi_2a": [round(float(g.mean()), 4) if len(g) else None, int(len(g))]}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
