"""Combinada de FC 27 (Esoccer Battle) con los mercados de LeoVegas España (Kambi): goles totales, goles de cada jugador,
hándicap y ambos marcan, además del ganador.

Probabilidad de cada pierna: goles esperados de cada jugador del modelo del bot (esoccer.py: Poisson por jugador y liga;
medido en 17.831 partidos de los últimos 15 días: total > 4.5 previsto 70 % / real 73 %, > 6.5 42 % / 42 %, jugador > 2.5
57 % / 60 %) mezclada con la de la casa sin margen (30 % modelo + 70 % casa, como en la combinada de córners: la casa
cobra ~12-15 % de margen y el modelo aún no le gana en el 1X2). Una pierna por partido, la de más ventaja, con
probabilidad 40-92 %; primero las que tienen ventaja. Las cuotas que se ven se guardan en data/fc/odds_leo.csv para
medir más adelante si el modelo gana a la casa en estos mercados."""
import math, re, time

import pandas as pd
import requests

import analyst as A
import esoccer as E
import kambi as K

OP, BOOK = "leoes", "LeoVegas España"
W, P_MIN, P_MAX, EDGE = 0.30, 0.40, 0.92, 0.02
LOG = E.DIR / "odds_leo.csv"
_MODEL = {}


def model():
    """Modelo de goles por jugador con los últimos 120 días (se rehace como mucho cada 20 min)."""
    if _MODEL.get("t", 0) > time.time() - 1200: return _MODEL["m"], _MODEL["df"]
    df = pd.read_csv(E.HIST); df["date"] = pd.to_datetime(df["date"], utc=True); df = df.sort_values(["date", "id"], ignore_index=True)
    m = E.load_model()  # el de analista (cada 30 min); si no hay, se ajusta aquí (tarda minutos)
    if m is None: m, *_ = E.fit(df[df.date >= df.date.max() - pd.Timedelta(days=120)])
    _MODEL.update(t=time.time(), m=m, df=df)
    return m, df


def scheduled(hours=3):
    """Partidos de Esoccer Battle programados o en juego en las próximas `hours` h: [{id, date, league, p1, p2, c1, c2}]."""
    from concurrent.futures import ThreadPoolExecutor
    now = pd.Timestamp.now(tz="UTC")
    ts = [t for t in E.tournaments(now, now + pd.Timedelta(hours=hours)) if t.get("status_id") in (1, 2, 3)
          and now - pd.Timedelta(hours=3) < pd.Timestamp(t.get("start_date") or now) < now + pd.Timedelta(hours=hours)]  # solo los de ahora
    def rows(t):
        try: return [g for g in E.match_rows(t, finished=False) if now < pd.Timestamp(g["date"]) < now + pd.Timedelta(hours=hours)]
        except (requests.RequestException, ValueError) as e: print("esportsbattle", t.get("id"), e); return []
    with ThreadPoolExecutor(6) as ex: return [g for gs in ex.map(rows, ts) for g in gs]  # en paralelo: antes 66 s


def offers(eid, op=OP):
    """Mercados de un partido de esoccer en Kambi: [{k: tot|ptot|hcp|btts, who: 'h'|'a'|None, line, odds: {lado: cuota}}]
    (lados O/U, Y/N o h/a; `who` y el hándicap en el orden local/visitante de Kambi). Sin 1.ª parte ni par/impar."""
    r = requests.get(f"https://eu-offering-api.kambicdn.com/offering/v2018/{op}/betoffer/event/{eid}.json",
                     params=K.Q, headers=A.UA, timeout=30).json()
    ev = (r.get("events") or [{}])[0]
    hn, an = K.nick(ev.get("homeName")), K.nick(ev.get("awayName"))
    out = []
    for b in r.get("betOffers", []):
        lab = b["criterion"].get("englishLabel", "")
        oc = [o for o in b.get("outcomes", []) if o.get("odds") and o.get("status", "OPEN") == "OPEN"]
        if "Half" in lab or "Odd/Even" in lab or len(oc) != 2: continue
        side = lambda o: (o.get("englishLabel") or o.get("label") or "").lower()
        if lab.startswith("Total Goals"):
            who = None if lab == "Total Goals" else "h" if K.nick(lab.replace("Total Goals by", "")) == hn else \
                "a" if K.nick(lab.replace("Total Goals by", "")) == an else "?"
            if who == "?" or not oc[0].get("line"): continue
            od = {("O" if side(o).startswith("over") else "U"): o["odds"] / 1000 for o in oc}
            out.append({"k": "ptot" if who else "tot", "who": who, "line": oc[0]["line"] / 1000, "odds": od})
        elif lab == "Both Teams To Score":
            out.append({"k": "btts", "who": None, "line": None, "odds": {("Y" if side(o).startswith("yes") else "N"): o["odds"] / 1000 for o in oc}})
        elif lab == "Handicap":
            od, line = {}, None
            for o in oc:
                w = "h" if K.nick(o.get("participant")) == hn else "a" if K.nick(o.get("participant")) == an else None
                if w: od[w] = o["odds"] / 1000; line = o["line"] / 1000 if w == "h" else line
            if len(od) == 2 and line is not None and line % 1: out.append({"k": "hcp", "who": None, "line": line, "odds": od})
    return out, hn, an


def _over(lam, line): return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(line) + 1))


def model_p(mk, s, l1, l2, diff):
    """Probabilidad del modelo de un lado de un mercado (l1/l2 = goles esperados del local/visitante de Kambi)."""
    if mk["k"] == "tot": p = _over(l1 + l2, mk["line"]); return p if s == "O" else 1 - p
    if mk["k"] == "ptot": p = _over(l1 if mk["who"] == "h" else l2, mk["line"]); return p if s == "O" else 1 - p
    if mk["k"] == "btts": p = (1 - math.exp(-l1)) * (1 - math.exp(-l2)); return p if s == "Y" else 1 - p
    ph = sum(v for d, v in diff.items() if d + mk["line"] > 0)  # local con su hándicap
    return ph if s == "h" else 1 - ph


def leg_text(mk, s, home, away):
    """Texto de la pierna en el orden de Esoccer Battle (home/away ya colocados por el llamador)."""
    if mk["k"] == "tot": return f"{'Más' if s == 'O' else 'Menos'} de {mk['line']:g} goles"
    if mk["k"] == "ptot": return f"{'Más' if s == 'O' else 'Menos'} de {mk['line']:g} goles de {home if mk['who'] == 'h' else away}"
    if mk["k"] == "btts": return f"Ambos marcan: {'Sí' if s == 'Y' else 'No'}"
    h = mk["line"] if s == "h" else -mk["line"]
    return f"{home if s == 'h' else away} {h:+g}".replace("-", "−")


def judge(leg, hs, as_):
    """(acierto, porqué) de una pierna de estos mercados con el marcador (hs, as_) en el orden home/away de la pierna."""
    f = leg.get("fc") or {}
    k, line, s, who = f.get("k"), f.get("line"), f.get("s"), f.get("who")
    if k == "tot": v = hs + as_; return (v > line) == (s == "O"), f"hubo {v:g} goles ({hs:g}-{as_:g})"
    if k == "ptot":
        v = hs if who == "h" else as_; return (v > line) == (s == "O"), f"{leg['home'] if who == 'h' else leg['away']} marcó {v:g} ({hs:g}-{as_:g})"
    if k == "btts": ok = hs > 0 and as_ > 0; return ok == (s == "Y"), f"acabó {hs:g}-{as_:g}"
    if k == "hcp": d = (hs - as_) + line; return (d > 0) == (s == "h"), f"acabó {hs:g}-{as_:g}"
    return None, "pierna no reconocida"


def candidates(hours=3):
    """(piernas candidatas, nº de partidos con cuotas). Cada candidata: dict con lo necesario para mostrar y simular."""
    t0 = time.time()
    m, df = model(); t1 = time.time()
    games = scheduled(hours); t2 = time.time()
    evs = K.esoccer_events(OP); t3 = time.time()
    print(f"combinada FC 27: modelo {t1 - t0:.0f} s · calendario {t2 - t1:.0f} s ({len(games)}) · {BOOK} {t3 - t2:.0f} s ({len(evs)})", flush=True)
    cands, priced, log = [], 0, []
    now = pd.Timestamp.now(tz="UTC")
    for g in games:
        e = K.find_esoccer(g["p1"], g["p2"], pd.Timestamp(g["date"]), evs)
        if not e: continue
        try: mks, hn, an = offers(e["id"])
        except (requests.RequestException, ValueError) as ex: print("LeoVegas", e["id"], ex); continue
        if not mks: continue
        priced += 1
        swap = hn != g["p1"].lower()  # Kambi pone al revés local y visitante
        lh, la = m.lam(g["league"], g["p1"], g["p2"]), m.lam(g["league"], g["p2"], g["p1"])
        l1, l2 = (la, lh) if swap else (lh, la)  # en el orden de Kambi
        _, _, diff = E.markets(l1, l2)
        best = None
        for mk in mks:
            inv = {s: 1 / o for s, o in mk["odds"].items()}; t = sum(inv.values())
            for s, o in mk["odds"].items():
                pm, pc = model_p(mk, s, l1, l2, diff), inv[s] / t
                q = W * pm + (1 - W) * pc
                log.append({"seen": now.isoformat(), "id": g["id"], "start": g["date"], "p1": g["p1"], "p2": g["p2"], "liga": g["league"],
                            "k": mk["k"], "who": mk["who"], "line": mk["line"], "s": s, "odds": o, "p_modelo": round(pm, 4), "p_casa": round(pc, 4)})
                if not (P_MIN <= q <= P_MAX): continue
                c = (q * o - 1, q, o, mk, s, pm, pc)
                if best is None or c[0] > best[0]: best = c
        if not best: continue
        edge, q, o, mk, s, pm, pc = best
        # al orden de Esoccer Battle (el de los resultados guardados)
        flip = {"h": "a", "a": "h"}
        mk2 = {**mk, "who": flip.get(mk["who"], mk["who"]) if swap else mk["who"], "line": (-mk["line"] if swap and mk["k"] == "hcp" else mk["line"])}
        s2 = flip.get(s, s) if swap and mk["k"] == "hcp" else s
        lp1, lp2 = lh, la
        cands.append({"edge": edge, "q": q, "odds": o, "pm": pm, "pc": pc, "g": g, "l1": lp1, "l2": lp2,
                      "txt": leg_text(mk2, s2, g["p1"], g["p2"]), "fc": {"k": mk2["k"], "who": mk2["who"], "line": mk2["line"], "s": s2},
                      "form": (E.form(df, g["p1"]), E.form(df, g["p2"]))})
    if log:
        try:
            old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame(log)]).drop_duplicates(["id", "k", "who", "line", "s"], keep="last").to_csv(LOG, index=False)
        except OSError as ex: print("odds_leo", ex)
    print(f"combinada FC 27: {len(games)} partidos, {priced} con cuotas de {BOOK}, {len(cands)} piernas")
    return cands, priced


def combo(target=10.0, hours=3, cands=None):
    """(texto HTML, piernas para simular, cuota, prob.) de la combinada de FC 27; (texto, None, …) si no hay."""
    if cands is None: cands, priced = candidates(hours)
    else: priced = len(cands)
    value = [c for c in cands if c["edge"] >= EDGE]
    pool = sorted(value, key=lambda c: -c["edge"]) + sorted([c for c in cands if c["edge"] < EDGE], key=lambda c: -c["edge"])
    legs, odds, p = [], 1.0, 1.0
    for c in pool:
        if odds >= target: break
        legs.append(c); odds *= c["odds"]; p *= c["q"]
    if len(legs) < 2:
        return (f"Ahora mismo no hay partidos de FC 27 con mercados de {BOOK} suficientes ({priced} con cuotas en las "
                f"próximas {hours} h). Hay partidos casi cada hora: prueba en un rato.", None, None, None)
    legs.sort(key=lambda c: c["g"]["date"]); ev = p * odds - 1
    TZ = "Europe/Madrid"
    out = [f"🎮⚽ <b>Combinada FC 27 · objetivo {target:g}</b> (cuotas de {BOOK}; una pierna por partido)"]
    nb = sum(c["edge"] < EDGE for c in legs)
    if nb: out.append(f"⚠️ {nb} de las {len(legs)} piernas no tienen ventaja clara: son las menos malas para llegar a la cuota.")
    for c in legs:
        g = c["g"]; t = pd.Timestamp(g["date"]).tz_convert(TZ)
        out.append(f"<b>{A.esc(g['p1'])} – {A.esc(g['p2'])}</b> · {t:%H:%M} · {A.esc(g['league'])}\n"
                   f"  📊 Goles esperados: {A.esc(g['p1'])} {c['l1']:.1f} · {A.esc(g['p2'])} {c['l2']:.1f} · forma {c['form'][0]} | {c['form'][1]}\n"
                   f"  • {A.esc(c['txt'])} @ {c['odds']:.2f} (bot {c['q']:.0%}: modelo {c['pm']:.0%}, casa {c['pc']:.0%}; ventaja {c['edge']:+.0%})")
    out.append(f"{len(legs)} piernas · cuota <b>{odds:.1f}</b>: 1 € → {odds:.0f} € · acierta el <b>{p:.1%}</b> de las veces.")
    if odds < target: out.append(f"⚠️ Con estos partidos llego a cuota {odds:.1f}.")
    out.append(f"{'🟢' if ev > 0 else '🔴'} Según el bot, a la larga {'ganas' if ev > 0 else 'pierdes'} {abs(ev):.0%}. "
               "La casa cobra mucho margen en FC 27: es lotería, simúlala antes de poner dinero.")
    sim = [dict(src="fc", side=None, home=c["g"]["p1"], away=c["g"]["p2"], start=str(pd.Timestamp(c["g"]["date"])),
                txt=c["txt"], odds=c["odds"], p=round(c["q"], 4), fc=c["fc"]) for c in legs]
    return "\n".join(out), sim, odds, p
