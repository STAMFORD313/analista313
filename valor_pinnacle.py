"""Estrategia «valor contra Pinnacle»: apostar en Unibet cuando paga más que el precio justo (sin margen) de Pinnacle,
la casa más afinada. No necesita predecir mejor que nadie: aprovecha que las casas blandas tardan en ajustar sus cuotas.

Backtest (football-data, 60.192 partidos 2015-2026, mejor cuota entre bet365/Bwin/WH/Interwetten/VC frente al 1X2 de
Pinnacle sin margen, cuota ≤ 2.5): ventaja > 3 % → 942 apuestas, ROI +11.0 % ± 3.3 %, CLV +5.4 % frente al cierre de
Pinnacle; > 5 % → 396, ROI +14.2 % ± 5.3 %, CLV +7.2 %. Con una sola casa el ROI es más ruidoso (bet365 > 4 %: +0.9 % ± 5.3 %,
CLV +3.7 %): el CLV positivo es la señal estable.

En vivo (cada run de analista): 1X2 de Unibet (Kambi) frente a data/pinnacle/futbol.csv (pinnacle.py), partidos de las
próximas 36 h; señal si cuota × prob. justa − 1 ≥ 3 % y cuota ≤ 2.5 (la banca solo apuesta ≥ 5 %: su EDGE). Se liquida con
los resultados de FotMob/ESPN y su CLV con el cierre de Pinnacle (estrategias.py lo vigila)."""
import json

import pandas as pd

import analyst as A

MIN_EDGE, MAX_ODDS, HOURS = 0.03, 2.5, 36
SPORT = "⚽💎"  # segmento propio en la banca y en estrategias.py (no se mezcla con el 1X2 del modelo)


def fair_rows(now):
    try: d = pd.read_csv(A.DATA / "pinnacle" / "futbol.csv")
    except FileNotFoundError: return pd.DataFrame()
    d = d[(d.market == "moneyline")].dropna(subset=["o_home", "o_away", "o_draw"])
    t = pd.to_datetime(d.start, utc=True)
    return d[(t > now + pd.Timedelta(minutes=10)) & (t < now + pd.Timedelta(hours=HOURS))]


def candidates(now=None, uni=None, min_edge=MIN_EDGE, max_odds=MAX_ODDS):
    """Señales de valor: Unibet paga más que Pinnacle sin margen (min_edge/max_odds más laxos para piernas de combinada)."""
    import kambi
    now = now or pd.Timestamp.now(tz="UTC")
    pin = fair_rows(now)
    if not len(pin): return []
    if uni is None:  # la mejor cuota entre Unibet, Paf, LeoVegas (Kambi) y Smarkets (bolsa, tras su comisión)
        uni = kambi.football_1x2_best()
        try:
            import smarkets
            uni += smarkets.football_1x2(HOURS)
        except Exception as e: print("Smarkets", repr(e)[:120])
    out, matched, edges = [], 0, []
    for r in pin.itertuples():
        t = pd.Timestamp(r.start)
        us = [e for e in uni if abs((e["start"] - t).total_seconds()) < 2 * 3600
              and A.find_team(r.home, [e["home"]]) and A.find_team(r.away, [e["away"]])]
        if not us: continue
        u = {"odds": {}, "books": {}}  # la mejor cuota de cada resultado entre todas las casas que tienen el partido
        for e in us:
            for mm, o in e["odds"].items():
                if o > u["odds"].get(mm, 0): u["odds"][mm], u["books"][mm] = o, (e.get("books") or {}).get(mm, e.get("book", "?"))
        matched += 1
        inv = {"H": 1 / r.o_home, "D": 1 / r.o_draw, "A": 1 / r.o_away}; tot = sum(inv.values())
        edges += [u["odds"][m] * inv[m] / tot - 1 for m in "HDA"]
        for m, name in (("H", r.home), ("D", "Empate"), ("A", r.away)):
            p, o = inv[m] / tot, u["odds"][m]
            edge = o * p - 1
            if edge < min_edge or o > max_odds: continue
            book = (u.get("books") or {}).get(m, u.get("book", "Unibet"))
            out.append({"key": f"vp|{t:%Y-%m-%d}|{r.home}|{r.away}", "market": m, "pick": name, "odds": round(o, 2), "book": book,
                        "p": round(p, 4), "implied": round(p, 4), "edge": round(edge, 4), "match": f"{r.home} vs {r.away}",
                        "start": t.isoformat(), "league": f"💎 Valor vs Pinnacle · {r.league}", "sport": SPORT,
                        "info": [f"💎 <b>Valor contra Pinnacle</b>: {book} paga {o:.2f} y el precio justo de Pinnacle es {1 / p:.2f} "
                                 f"(ventaja {edge:+.1%})", "Backtest 2015-2026 (cuota ≤ 2.5, > 3 %): ROI +11 % ± 3 %, CLV +5 %."],
                        "context": f"{r.home} vs {r.away} ({r.league}). Estrategia de valor: {book} {o:.2f} frente a Pinnacle justo {1 / p:.2f}."})
    if edges: print(f"valor Pinnacle: {len(pin)} partidos de Pinnacle, {len(uni)} de las casas Kambi, {matched} emparejados; "
                    f"ventaja de la mejor casa: mediana {sorted(edges)[len(edges) // 2]:+.1%}, máxima {max(edges):+.1%}")
    else: print(f"valor Pinnacle: {len(pin)} partidos de Pinnacle, {len(uni)} de Unibet, 0 emparejados")
    return out


def run(state):
    try: cands = candidates()
    except Exception as e: print("valor Pinnacle", repr(e)); return
    have = {(s["key"], s["market"]) for s in state["signals"]}
    new = [c for c in cands if (c["key"], c["market"]) not in have]
    state["signals"] += new
    print(f"valor Pinnacle: {len(cands)} con valor, {len(new)} nuevas")


# ── liquidación con FotMob / ESPN (nombres de Pinnacle emparejados por parecido) ─────────────────────────────────
def _results(days=6):
    out = []
    for f in sorted((A.DATA / "fotmob").glob("*.jsonl"))[-2:]:
        out += [json.loads(l) for l in f.open(encoding="utf-8")]
    import espn_soccer as ES
    for code in ES.LEAGUES:
        out += [{"date": g["date"], "home": g["home"], "away": g["away"], "hs": g["hs"], "as": g["as"]} for g in ES.load(code)[-80:]]
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + 2)
    return [g for g in out if g.get("hs") is not None and pd.Timestamp(g["date"]) >= cut]


def settle(state):
    pend = [s for s in state["signals"] if s["key"].startswith("vp|") and s.get("result") is None
            and pd.Timestamp(s["start"]) < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)]
    if not pend: return 0
    res, n = _results(), 0
    for s in pend:
        home, away = s["match"].split(" vs ", 1); t = pd.Timestamp(s["start"])
        g = next((g for g in res if abs((pd.Timestamp(g["date"]) - t).total_seconds()) < 12 * 3600
                  and A.find_team(home, [g["home"]]) and A.find_team(away, [g["away"]])), None)
        if g:
            hs, as_ = float(g["hs"]), float(g["as"])
            s["result"] = {"H": hs > as_, "D": hs == as_, "A": as_ > hs}[s["market"]]; s["score"] = f"{hs:g}-{as_:g}"; n += 1
        elif pd.Timestamp.now(tz="UTC") - t > pd.Timedelta(days=6): s["result"] = "nula"
    print(f"valor Pinnacle: {n} liquidadas")
    return n
