"""Cuotas de Pinnacle (la casa más afinada: referencia del «precio justo») con su API pública de invitado, sin clave.

Guarda en data/pinnacle/<deporte>.csv una fila por partido y mercado principal (ganador, total y hándicap del partido
completo) con la ÚLTIMA cuota vista: como se sobrescribe en cada pasada hasta que empieza, al final queda la de cierre.
Lo lanza nutricion.py (experto «datos_pinnacle»). Uso: python pinnacle.py"""
import json, re, sys

import pandas as pd
import requests

import analyst as A

API = "https://guest.api.arcadia.pinnacle.com/0.1"
SPORTS = {"futbol": 29, "baloncesto": 4, "hockey": 19, "esports": 12, "beisbol": 3, "americano": 15, "tenis": 33}
DIR = A.DATA / "pinnacle"
HEAD = {**A.UA, "Accept": "application/json"}
COLS = ["matchup", "start", "league", "home", "away", "market", "line", "o_home", "o_away", "o_draw", "o_over", "o_under", "seen"]


def get(path):
    r = requests.get(f"{API}/{path}", headers=HEAD, timeout=60)
    r.raise_for_status(); return r.json()


def dec(american):
    """Cuota americana → decimal."""
    if american is None: return None
    a = float(american)
    return round(1 + a / 100 if a > 0 else 1 + 100 / -a, 4)


def snapshot(sport_id, hours=72):
    """Filas de los partidos que empiezan en las próximas `hours` horas (mercados principales, partido completo)."""
    now = pd.Timestamp.now(tz="UTC")
    mus = {m["id"]: m for m in get(f"sports/{sport_id}/matchups?withSpecials=false&brandId=0")
           if m.get("type") == "matchup" and not m.get("parent") and len(m.get("participants", [])) >= 2}
    keep = {}
    for i, m in mus.items():
        t = pd.Timestamp(m["startTime"])
        if now < t < now + pd.Timedelta(hours=hours): keep[i] = (m, t)  # solo antes de empezar: la última = la de cierre (no en vivo)
    rows = []
    for mk in get(f"sports/{sport_id}/markets/straight?primaryOnly=true&withSpecials=false"):
        if mk.get("matchupId") not in keep or mk.get("period") != 0 or mk.get("status") != "open": continue
        m, t = keep[mk["matchupId"]]
        side = {p.get("alignment"): p["name"] for p in m["participants"]}
        pr = {p.get("designation"): p for p in mk.get("prices", [])}
        typ = mk.get("type")
        if typ not in ("moneyline", "total", "spread"): continue
        line = (pr.get("over") or pr.get("home") or {}).get("points")
        rows.append({"matchup": m["id"], "start": str(t), "league": (m.get("league") or {}).get("name"),
                     "home": side.get("home"), "away": side.get("away"), "market": typ, "line": line,
                     "o_home": dec((pr.get("home") or {}).get("price")), "o_away": dec((pr.get("away") or {}).get("price")),
                     "o_draw": dec((pr.get("draw") or {}).get("price")), "o_over": dec((pr.get("over") or {}).get("price")),
                     "o_under": dec((pr.get("under") or {}).get("price")), "seen": str(now)})
    return rows


# juveniles, reservas y femenino: Pinnacle los nombra igual que al primer equipo («PSV – Club Brugge» de la Youth League)
# → se emparejarían con el partido de los mayores (ventajas y CLV falsos). Fuera al guardar y al leer.
NOT_SENIOR = re.compile(r"U\d\d|Youth|Reserve|Women|Femen|Frauen|Primavera|Juvenil|Amateur", re.I)


def senior(d):
    """Filas de primeros equipos (sin juveniles/reservas/femenino) de un DataFrame de Pinnacle."""
    return d[~d["league"].fillna("").astype(str).str.contains(NOT_SENIOR)] if "league" in d and len(d) else d


def read(name):
    """data/pinnacle/<name>.csv sin juveniles/reservas/femenino (DataFrame vacío con columnas si no existe)."""
    f = DIR / f"{name}.csv"
    return senior(pd.read_csv(f)) if f.exists() else pd.DataFrame(columns=COLS)


def update(hours=72):
    DIR.mkdir(parents=True, exist_ok=True)
    tot = 0
    for name, sid in SPORTS.items():
        try: rows = snapshot(sid, hours)
        except (requests.RequestException, ValueError) as e: print("Pinnacle", name, repr(e)[:150]); continue
        f = DIR / f"{name}.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLS)
        new = pd.DataFrame(rows, columns=COLS)
        # la fila más reciente de cada partido y mercado: antes de empezar se sobrescribe → queda la cuota de cierre
        d = senior(pd.concat([old, new]).drop_duplicates(["matchup", "market"], keep="last"))
        d.to_csv(f, index=False); tot += len(new)
        print(f"Pinnacle {name}: {len(new)} mercados ahora, {len(d)} guardados")
    print(f"Pinnacle: {tot} mercados")


if __name__ == "__main__":
    update(int(sys.argv[1]) if len(sys.argv) > 1 else 72)


# ── CLV: la cuota que cogió el bot frente al cierre de Pinnacle sin margen ──────────────────────────────────────────
FILE_OF = {"nba": "baloncesto", "cbb": "baloncesto", "nhl": "hockey", "nfl": "americano", "cfb": "americano", "mlb": "beisbol",
           "cs2": "esports", "lol": "esports", "dota2": "esports", "r6": "esports", "mlbb": "esports", "atp": "tenis", "wta": "tenis", "vp": "futbol"}
_CLOSE = {}


def close_p(s):
    """Probabilidad sin margen del cierre de Pinnacle del lado apostado por la señal s (ganador o más/menos), o None."""
    import analyst as A
    lg = s["key"].split("|")[0]
    f = FILE_OF.get(lg, "futbol" if lg in A.ALL_LEAGUES or lg in ("ucl", "uel", "uecl") else None)
    if not f or " vs " not in s.get("match", ""): return None
    if f not in _CLOSE:
        _CLOSE[f] = read(f)
    d, (home, away), t = _CLOSE[f], s["match"].split(" vs ", 1), pd.Timestamp(s["start"])
    m = s["market"]
    side = {"home": "h", "H": "h", "t1": "h", "away": "a", "A": "a", "t2": "a", "D": "d",
            "over": "o", "O25": "o", "under": "u", "U25": "u", "sph": "sh", "spa": "sa"}.get(m)
    if not side: return None
    mk = "total" if side in ("o", "u") else "spread" if side in ("sh", "sa") else "moneyline"
    for r in d[d.market == mk].itertuples():
        if abs((pd.Timestamp(r.start) - t).total_seconds()) > 3 * 3600: continue
        if not (A.find_team(home, [r.home]) and A.find_team(away, [r.away])): continue
        if mk == "spread":  # hándicap: solo si Pinnacle cerró en la misma línea (la del local; la del visitante es la opuesta)
            if s.get("line") is None or pd.isna(r.line) or pd.isna(r.o_home) or pd.isna(r.o_away): return None
            if abs(r.line - (s["line"] if side == "sh" else -s["line"])) > 1e-9: return None
            ih, ia = 1 / r.o_home, 1 / r.o_away
            return round((ih if side == "sh" else ia) / (ih + ia), 4)
        if mk == "total":
            line = s.get("line") or (2.5 if m in ("O25", "U25") else None)
            if line is None or r.line != line or pd.isna(r.o_over) or pd.isna(r.o_under): return None
            io, iu = 1 / r.o_over, 1 / r.o_under
            return round((io if side == "o" else iu) / (io + iu), 4)
        inv = {"h": 1 / r.o_home, "a": 1 / r.o_away, "d": (1 / r.o_draw) if r.o_draw == r.o_draw and r.o_draw else 0.0}
        if side == "d" and not inv["d"]: return None
        return round(inv[side] / sum(inv.values()), 4)
    return None
