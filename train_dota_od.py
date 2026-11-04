"""¿Mejora el modelo de Dota 2 con las partidas de OpenDota? (fuentes.py → data/dota2/opendota.csv)

Elo por mapa con OpenDota (cada partida = un mapa, K con margen de kills), solo con partidas anteriores a cada serie;
prueba = series de bo3.gg (data/dota2/matches.csv) del último año con los dos equipos emparejados por nombre. Compara el
log-loss de serie del modelo actual (esports.map_elo) con el de OpenDota y con su mezcla (peso elegido en la 1.ª mitad,
medido en la 2.ª). Guarda data/dota2/opendota_params.json; el modelo de producción solo lo usará si mejora > 2 e.t.
Uso: python train_dota_od.py"""
import json, math, re, unicodedata

import numpy as np
import pandas as pd

import esports as E

OUT = E.A.DATA / "dota2" / "opendota_params.json"
K, START = 24.0, 1500.0


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(team|esports?|gaming|club|the)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def od_elo(od):
    """[(fecha, {equipo normalizado: rating})] tras cada día, para consultar el rating previo a una serie."""
    R, snaps, day = {}, [], None
    for r in od.sort_values("start_time").itertuples():
        h, a = norm(r.radiant), norm(r.dire)
        if not h or not a: continue
        d = r.start_time.normalize()
        if day is not None and d != day: snaps.append((day, dict(R)))
        day = d
        rh, ra = R.get(h, START), R.get(a, START)
        e = 1 / (1 + 10 ** ((ra - rh) / 400))
        w = 1.0 if r.radiant_win else 0.0
        mov = math.log(abs((r.radiant_score or 0) - (r.dire_score or 0)) + 2) / math.log(12)
        R[h] = rh + K * mov * (w - e); R[a] = ra - K * mov * (w - e)
    if day is not None: snaps.append((day, dict(R)))
    return snaps


def main():
    od = pd.read_csv(E.A.DATA / "dota2" / "opendota.csv")
    od["start_time"] = pd.to_datetime(od.start_time, utc=True)
    E.configure("dota2")
    df = pd.read_csv(E.HIST); df["date"] = pd.to_datetime(df.date, utc=True)
    df = df.dropna(subset=["winner_id"]).sort_values("date", ignore_index=True); df = df[df.bo % 2 == 1]
    since = df.date.max() - pd.DateOffset(years=1)
    preds = {}
    E.map_elo(df[df.date >= df.date.max() - pd.DateOffset(years=4)], since=since, games=E.load_games(), preds=preds)
    snaps = od_elo(od); days = [s[0] for s in snaps]
    rows = []
    for r in df[df.id.isin(preds)].itertuples():
        i = np.searchsorted(days, r.date.normalize()) - 1  # ratings del día anterior a la serie
        if i < 0: continue
        R = snaps[i][1]; a, b = norm(r.team1), norm(r.team2)
        if a not in R or b not in R: continue
        pm = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))
        rows.append((r.date, preds[r.id][0], E.series_prob(pm, int(r.bo)), float(r.winner_id == r.t1_id)))
    if len(rows) < 100: print(f"OpenDota: solo {len(rows)} series emparejadas; hace falta más historial"); return
    d = pd.DataFrame(rows, columns=["date", "p_bo3", "p_od", "y"]).sort_values("date", ignore_index=True)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    h = len(d) // 2; fi, te = d.iloc[:h], d.iloc[h:]
    mix = lambda x, w: 1 / (1 + np.exp(-((1 - w) * lg(x.p_bo3) + w * lg(x.p_od))))
    w = min(np.linspace(0, 1, 11), key=lambda w: ll(mix(fi, w), fi.y).mean())
    l0, l1, l2 = ll(te.p_bo3, te.y), ll(te.p_od, te.y), ll(mix(te, w), te.y)
    se = float((l2 - l0).std() / math.sqrt(len(te)))
    print(f"Dota 2: {len(d)} series emparejadas (OpenDota {len(od)} mapas); prueba {len(te)}")
    print(f"  log-loss serie: bo3.gg {l0.mean():.4f} · OpenDota {l1.mean():.4f} · mezcla w={w:.1f} {l2.mean():.4f} ({l2.mean() - l0.mean():+.4f} ± {se:.4f})")
    res = {"w": float(w), "ll_bo3": round(float(l0.mean()), 4), "ll_od": round(float(l1.mean()), 4), "ll_mix": round(float(l2.mean()), 4),
           "se": round(se, 4), "n": int(len(te)), "usar": bool(l2.mean() < l0.mean() - 2 * se)}
    OUT.write_text(json.dumps(res, indent=1)); print("  usar en producción:", res["usar"])


if __name__ == "__main__":
    main()
