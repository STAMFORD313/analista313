"""NBA (y NFL/NHL): combinada del mismo partido «ganador + Más/Menos del total» con la correlación entre margen y total.

Con medias móviles de cada equipo (últimos 20 partidos, sin mirar el partido) se predice margen y total; los residuos
(real − previsto) dan la correlación ρ entre margen del local y total. Cópula gaussiana con ρ (train_sgp.cells_copula)
frente al producto de probabilidades: log-loss de las 4 casillas (gana local/visitante × más/menos de la línea) en la
última temporada, con ρ y desviaciones del año anterior. Guarda data/sgp_us.json {liga: {rho, ...}} (lo usa multisport).
Uso: PYTHONIOENCODING=utf-8 python train_nba_sgp.py"""
import json, math

import numpy as np
import pandas as pd
from scipy import stats

import analyst as A
import train_sgp as S

OUT = A.DATA / "sgp_us.json"


def load(league):
    d = pd.concat([pd.read_csv(f) for f in sorted((A.DATA / "espn" / league).glob("*.csv"))]).dropna(subset=["hs", "as"])
    d["date"] = pd.to_datetime(d.date, utc=True)
    return d.drop_duplicates("id").sort_values("date", ignore_index=True)


def expected(d, n=20):
    """Margen y total previstos con las medias de los últimos n partidos de cada equipo (anteriores al partido)."""
    long = pd.concat([pd.DataFrame({"i": d.index, "t": d.home, "pf": d.hs, "pa": d["as"]}),
                      pd.DataFrame({"i": d.index, "t": d.away, "pf": d["as"], "pa": d.hs})]).sort_values("i", kind="stable")
    g = long.groupby("t")
    for c in ("pf", "pa"): long[c + "_m"] = g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=8).mean())
    h = long.drop_duplicates("i", keep="first").set_index("i"); a = long.drop_duplicates("i", keep="last").set_index("i")
    h, a = h.reindex(d.index), a.reindex(d.index)
    em = (h.pf_m - h.pa_m + a.pa_m - a.pf_m) / 2
    et = (h.pf_m + h.pa_m + a.pf_m + a.pa_m) / 2
    return em.to_numpy(), et.to_numpy()


def main():
    out = {}
    for league in ("nba", "nfl", "nhl"):
        d = load(league)
        em, et = expected(d)
        m, t = (d.hs - d["as"]).to_numpy(float), (d.hs + d["as"]).to_numpy(float)
        ok = np.isfinite(em) & np.isfinite(et) & (m != 0)
        end = d.date.max()
        te = ok & (d.date >= end - pd.DateOffset(years=1)).to_numpy()
        fi = ok & (d.date < end - pd.DateOffset(years=1)).to_numpy() & (d.date >= end - pd.DateOffset(years=2)).to_numpy()
        hfa = float((m - em)[fi].mean())
        rm, rt = m - em - hfa, t - et
        sd_m, sd_t = float(rm[fi].std()), float(rt[fi].std())
        rho = float(np.corrcoef(rm[fi], rt[fi])[0, 1])
        line = np.floor(et) + 0.5
        pa = stats.norm.cdf((em + hfa) / sd_m); pb = 1 - stats.norm.cdf((line - et) / sd_t)
        c = S.cell(m > 0, t > line)
        ix = np.arange(len(c))
        l0 = -np.log(S.cells_cov(pa, pb, 0.0)[ix, c])[te]; l1 = -np.log(S.cells_copula(pa, pb, rho)[ix, c])[te]
        se = float((l1 - l0).std() / math.sqrt(te.sum()))
        fav = pa >= 0.5
        hit = np.where(fav, m > 0, m < 0) & (t > line)
        pj = np.where(fav, S.cells_copula(pa, pb, rho)[:, 0], S.cells_copula(pa, pb, rho)[:, 2])
        pn = np.where(fav, pa, 1 - pa) * pb
        out[league] = {"rho": round(rho, 4), "sd_margin": round(sd_m, 2), "sd_total": round(sd_t, 2), "n_test": int(te.sum()),
                       "ll_indep": round(float(l0.mean()), 5), "ll_copula": round(float(l1.mean()), 5), "se": round(se, 5)}
        print(f"{league}: ρ(margen local, total) = {rho:+.3f} · test {te.sum()} partidos: log-loss 4 casillas independencia "
              f"{l0.mean():.5f} → cópula {l1.mean():.5f} ({l1.mean() - l0.mean():+.5f} ± {se:.5f}) · «favorito gana + más»: "
              f"cópula {pj[te].mean():.3f} / producto {pn[te].mean():.3f} / real {hit[te].mean():.3f}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
