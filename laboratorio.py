"""Laboratorio de estrategias: prueba muchas estrategias a la vez con el historial de football-data (data/hist, desde
2015) y las ordena por lo que ganarían FUERA de donde se eligieron.

Validación: cada estrategia (con sus parámetros fijados de antemano, sin optimizar sobre la prueba) se mide en dos
periodos separados: «antes» (2015-2020) y «después» (2021-hoy). Solo cuenta como buena si gana en los dos y en «después»
ROI − 2 e.t. > 0. CLV = cuota cogida frente al cierre de Pinnacle sin margen (la señal de ventaja más estable).
Cuotas: «b365» = una sola casa blanda (lo más parecido a Unibet); «mejor» = la mejor entre bet365, Bwin, William Hill,
Interwetten y VC Bet (comparar casas). Pinnacle = PS* (cuota de cuando football-data recoge, días antes) y PSC* (cierre).
Resultado: data/laboratorio.json. Uso: python laboratorio.py (experto «laboratorio» de nutricion.py, semanal)."""
import glob, json, math

import numpy as np
import pandas as pd

import analyst as A

OUT = A.DATA / "laboratorio.json"
SOFT = ("B365", "BW", "WH", "IW", "VC")
SPLIT = pd.Timestamp("2021-01-01")
TOP5 = ("E0", "SP1", "D1", "I1", "F1")


def load():
    cols = ["Div", "Date", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA", "P>2.5", "P<2.5", "PC>2.5", "PC<2.5",
            "B365>2.5", "B365<2.5", "Max>2.5", "Max<2.5", "AHh", "PAHH", "PAHA", "PCAHH", "PCAHA", "B365AHH", "B365AHA",
            *[f"{b}{k}" for b in SOFT for k in "HDA"]]
    fr = []
    for f in glob.glob(str(A.DATA / "hist" / "*.csv")):
        try: d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip", usecols=lambda c: c in cols)
        except (ValueError, pd.errors.ParserError): continue
        if "PSH" in d and "B365H" in d: fr.append(d)
    D = pd.concat(fr, ignore_index=True)
    D["Date"] = pd.to_datetime(D.Date, dayfirst=True, format="mixed", errors="coerce")
    D = D.dropna(subset=["Date", "FTR", "PSH", "PSD", "PSA", "B365H", "B365D", "B365A"])
    return D[D.Date >= "2015-01-01"].reset_index(drop=True)


def fair3(D, pre="PS"):
    """1X2 sin margen de Pinnacle: PS (cuando lo recoge football-data) o PSC (cierre)."""
    i = {k: 1 / D[f"{pre}{k}"] for k in "HDA"}; t = i["H"] + i["D"] + i["A"]
    return {k: i[k] / t for k in "HDA"}


def best(D, k):
    return pd.concat([D[f"{b}{k}"] for b in SOFT if f"{b}{k}" in D], axis=1).max(axis=1)


def run_strategy(D, name, sides):
    """sides: [(máscara, cuota, gana, prob. de cierre)] → métricas por periodo."""
    rows = []
    for mask, odds, win, pclose in sides:
        m = mask.fillna(False).to_numpy()
        if not m.any(): continue
        g = np.where(win[m], odds[m] - 1, -1.0)
        clv = (odds[m] * pclose[m] - 1).to_numpy()
        rows.append(pd.DataFrame({"date": D.Date[m].to_numpy(), "g": g, "clv": clv}))
    if not rows: return None
    r = pd.concat(rows)
    res = {"estrategia": name, "n": int(len(r))}
    for per, sel in (("antes", r.date < SPLIT), ("despues", r.date >= SPLIT)):
        x = r[sel]
        if len(x) < 30: res[per] = None; continue
        res[per] = {"n": int(len(x)), "roi": round(float(x.g.mean()), 4), "se": round(float(x.g.std() / math.sqrt(len(x))), 4),
                    "clv": round(float(np.nanmean(x.clv)), 4)}
    a, d = res.get("antes"), res.get("despues")
    tot_se = float(r.g.std() / math.sqrt(len(r)))
    res["total"] = {"roi": round(float(r.g.mean()), 4), "se": round(tot_se, 4), "clv": round(float(np.nanmean(r.clv)), 4)}
    res["buena"] = bool(a and d and a["roi"] > 0 and d["roi"] - 2 * d["se"] > 0)
    # prometedora: gana en los dos periodos y el total es claramente positivo (el periodo reciente aún tiene poca muestra)
    res["prometedora"] = bool(a and d and a["roi"] > 0 and d["roi"] > 0 and r.g.mean() - 2 * tot_se > 0 and np.nanmean(r.clv) > 0)
    return res


def strategies(D):
    fair, close = fair3(D, "PS"), fair3(D, "PSC")
    win = {k: D.FTR == k for k in "HDA"}
    odds = {("b365", k): D[f"B365{k}"] for k in "HDA"} | {("mejor", k): best(D, k) for k in "HDA"}
    top5 = D.Div.isin(TOP5)
    S = {}
    # 1) valor contra Pinnacle (1X2): la casa paga más que el precio justo
    for src in ("b365", "mejor"):
        for m in (0.02, 0.03, 0.05, 0.08):
            for cap, cname in ((2.5, "≤2.5"), (99, "todas")):
                S[f"valor 1X2 {src} >{m:.0%} cuotas {cname}"] = [((odds[src, k] * fair[k] - 1 > m) & (odds[src, k] <= cap), odds[src, k], win[k], close[k]) for k in "HDA"]
        S[f"valor 1X2 {src} >3% solo top-5"] = [((odds[src, k] * fair[k] - 1 > 0.03) & (odds[src, k] <= 2.5) & top5, odds[src, k], win[k], close[k]) for k in "HDA"]
        S[f"valor 1X2 {src} >3% sin top-5"] = [((odds[src, k] * fair[k] - 1 > 0.03) & (odds[src, k] <= 2.5) & ~top5, odds[src, k], win[k], close[k]) for k in "HDA"]
        S[f"valor 1X2 {src} >3% solo empates"] = [((odds[src, "D"] * fair["D"] - 1 > 0.03), odds[src, "D"], win["D"], close["D"])]
    # 2) estrategias «de manual» sin valor (para comparar: suelen perder el margen)
    S["favorito local (Pinnacle ≥ 60 %) a b365"] = [(fair["H"] >= 0.6, odds["b365", "H"], win["H"], close["H"])]
    S["empate en partidos igualados a b365"] = [((fair["H"] - fair["A"]).abs() < 0.08, odds["b365", "D"], win["D"], close["D"])]
    S["visitante no favorito 2.5-4 a b365"] = [((odds["b365", "A"] > 2.5) & (odds["b365", "A"] <= 4), odds["b365", "A"], win["A"], close["A"])]
    S["siempre el favorito a la mejor cuota"] = [((fair[k] >= fair["H"]) & (fair[k] >= fair["A"]) & (fair[k] >= fair["D"]), odds["mejor", k], win[k], close[k]) for k in "HA"]
    # 3) valor contra Pinnacle en más/menos 2.5 goles
    if "P>2.5" in D and "B365>2.5" in D:
        io, iu = 1 / D["P>2.5"], 1 / D["P<2.5"]; fo, fu = io / (io + iu), iu / (io + iu)
        co, cu = (1 / D["PC>2.5"]) / (1 / D["PC>2.5"] + 1 / D["PC<2.5"]), (1 / D["PC<2.5"]) / (1 / D["PC>2.5"] + 1 / D["PC<2.5"])
        tg = D.FTHG + D.FTAG
        for m in (0.02, 0.04):
            S[f"valor más/menos 2.5 b365 >{m:.0%}"] = [(D["B365>2.5"] * fo - 1 > m, D["B365>2.5"], tg > 2.5, co), (D["B365<2.5"] * fu - 1 > m, D["B365<2.5"], tg < 2.5, cu)]
            if "Max>2.5" in D:
                S[f"valor más/menos 2.5 mejor >{m:.0%}"] = [(D["Max>2.5"] * fo - 1 > m, D["Max>2.5"], tg > 2.5, co), (D["Max<2.5"] * fu - 1 > m, D["Max<2.5"], tg < 2.5, cu)]
    # 4) valor contra Pinnacle en hándicap asiático (sin medios puntos: se excluyen líneas .25/.75)
    if "PAHH" in D and "B365AHH" in D:
        ih, ia = 1 / D.PAHH, 1 / D.PAHA; fh, fa = ih / (ih + ia), ia / (ih + ia)
        ch, ca = (1 / D.PCAHH) / (1 / D.PCAHH + 1 / D.PCAHA), (1 / D.PCAHA) / (1 / D.PCAHH + 1 / D.PCAHA)
        half = (D.AHh * 2) % 1 != 0.5
        margin = D.FTHG - D.FTAG + D.AHh
        S["valor hándicap asiático b365 >3%"] = [((D.B365AHH * fh - 1 > 0.03) & half & (margin != 0), D.B365AHH, margin > 0, ch),
                                                 ((D.B365AHA * fa - 1 > 0.03) & half & (margin != 0), D.B365AHA, margin < 0, ca)]
    return S


def main():
    D = load()
    print(f"Laboratorio: {len(D)} partidos ({D.Date.min():%Y}-{D.Date.max():%Y})")
    res = [r for name, sides in strategies(D).items() if (r := run_strategy(D, name, sides)) and r["n"] >= 30]
    key = lambda r: (r["despues"]["roi"] - 2 * r["despues"]["se"]) if r.get("despues") else -9
    res.sort(key=key, reverse=True)
    for r in res:
        a, d = r.get("antes") or {}, r.get("despues") or {}
        print(f"{'🟢' if r['buena'] else '🟡' if r['prometedora'] else '🔴'} {r['estrategia']:42} antes {a.get('roi', 0):+.3f}±{a.get('se', 0):.3f} (n{a.get('n', 0)}) · "
              f"después {d.get('roi', 0):+.3f}±{d.get('se', 0):.3f} (n{d.get('n', 0)}) · CLV {d.get('clv', 0):+.3f}")
    OUT.write_text(json.dumps({"fecha": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}", "partidos": len(D), "estrategias": res}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
