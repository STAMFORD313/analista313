"""Buscador de estrategias (bot «nutrición»): con TODAS las candidatas liquidadas con cuota real (data/senales.csv)
prueba segmentos deporte × mercado × tramo de cuota × tramo de ventaja y los valida walk-forward: el segmento se
descubre en la mitad antigua y se confirma en la nueva. Resultado en data/estrategias.json:
  · «rentables»: ROI > 0 en las dos mitades y ROI total − 2 e.t. > 0 (n ≥ 30) → la banca les sube el stake ×1.5;
  · «bloqueados»: ROI + 2 e.t. < 0 (n ≥ 30) → la banca no los apuesta.
Uso: python estrategias.py (lo ejecuta nutricion.yml cada pocas horas)."""
import itertools, json, math

import numpy as np
import pandas as pd

import analyst as A

OUT = A.DATA / "estrategias.json"
MIN_N = 30


def market_group(m):
    m = str(m)
    if "_" in m: return m.split("_")[0]
    return {"H": "1X2", "D": "1X2", "A": "1X2", "home": "ganador", "away": "ganador", "t1": "ganador", "t2": "ganador",
            "over": "total", "under": "total", "O25": "total", "U25": "total"}.get(m, "hándicap")


def odds_band(o): return "≤1.5" if o <= 1.5 else "1.5-2" if o <= 2 else "2-2.5" if o <= 2.5 else ">2.5"


def edge_band(e): return "<0" if e < 0 else "0-5%" if e < 0.05 else "5-15%" if e < 0.15 else ">15%"


def segment_keys(r):
    """Todas las combinaciones de filtros (de más general a más concreto) en las que cae una señal."""
    f = {"deporte": str(r["deporte"]), "mercado": market_group(r["mercado"]), "cuota": odds_band(r["odds"]), "ventaja": edge_band(r["edge"])}
    keys = []
    for n in (1, 2, 3):
        for c in itertools.combinations(f, n): keys.append(" · ".join(f"{k}={f[k]}" for k in c))
    return keys


def stats(g):
    n = len(g); m = g.mean(); se = g.std() / math.sqrt(n) if n > 1 else 9.0
    return n, float(m), float(se)


def search(d=None):
    d = pd.read_csv(A.DATA / "senales.csv") if d is None else d
    d = d[~d.casa.astype(str).str.startswith("cuota justa")].copy()
    if d.empty: return {"rentables": [], "bloqueados": [], "n": 0}
    d["g"] = np.where(d.gano == 1, d.retorno.astype(float) - 1, -1.0)
    d = d.sort_values("fecha", kind="stable").reset_index(drop=True)
    half = len(d) // 2
    rows = [(k, i) for i, r in d.iterrows() for k in segment_keys(r)]
    by = pd.DataFrame(rows, columns=["seg", "i"]).groupby("seg").i.apply(list)
    good, bad = [], []
    for seg, idx in by.items():
        if len(idx) < MIN_N: continue
        g = d.g.iloc[idx]; n, m, se = stats(g)
        old, new = g[np.array(idx) < half], g[np.array(idx) >= half]
        row = {"segmento": seg, "n": n, "roi": round(m, 4), "se": round(se, 4), "roi_antigua": round(float(old.mean()), 4) if len(old) else None,
               "roi_nueva": round(float(new.mean()), 4) if len(new) else None}
        if m - 2 * se > 0 and len(old) >= 10 and len(new) >= 10 and old.mean() > 0 and new.mean() > 0: good.append(row)
        elif m + 2 * se < 0: bad.append(row)
    out = {"fecha": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}", "n": len(d),
           "rentables": sorted(good, key=lambda r: -(r["roi"] - 2 * r["se"])), "bloqueados": sorted(bad, key=lambda r: r["roi"] + 2 * r["se"])}
    return out


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {"rentables": [], "bloqueados": []}


def factor(s):
    """Multiplicador de stake de una señal según las estrategias: 0 si cae en un segmento bloqueado, 1.5 si en uno
    rentable (confirmado en las dos mitades), 1 si no."""
    E = load()
    r = {"deporte": s.get("sport", "?"), "mercado": s["market"], "odds": s["odds"], "edge": s["edge"]}
    keys = set(segment_keys(r))
    if any(b["segmento"] in keys for b in E.get("bloqueados", [])): return 0.0
    if any(g["segmento"] in keys for g in E.get("rentables", [])): return 1.5
    return 1.0


if __name__ == "__main__":
    res = search()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{res['n']} candidatas: {len(res['rentables'])} estrategias rentables, {len(res['bloqueados'])} bloqueadas")
    for r in res["rentables"][:5] + res["bloqueados"][:5]: print(" ", r)
