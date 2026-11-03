"""Props de jugador en CS2: kills de un jugador en la serie (más/menos de una línea), walk-forward.

Datos: data/cs2/players_matches.csv (kills por jugador y serie, bo3.gg) + players_games.csv (rondas de cada mapa) +
Elo de producción por mapa (esports.map_elo, prob. antes de jugar). Modelo:
  kills ~ BinNeg(media = KPR_jugador · rondas esperadas · exp(c · logit(p_mapa de su equipo)))
  KPR_jugador = kills/ronda con olvido (vida media H series) encogida hacia la media con K rondas;
  rondas esperadas = mapas esperados (BO1 1, BO3 2 + P(3er mapa) de bo3_scores) × rondas medias por mapa.
Referencia (lo que haría un apostador sin modelo): media de kills por serie de sus últimas series.
Elección de H, K, c y r en el año −2; medido en el año −1 con log-loss de más/menos a una línea tipo casa
(mediana de sus 10 series anteriores del mismo BO, a .5) y log-verosimilitud del número exacto.
Uso: PYTHONIOENCODING=utf-8 python train_cs2_props.py → data/cs2/props_params.json"""
import itertools, json, math

import numpy as np
import pandas as pd
from scipy import optimize, stats

import esports as E

OUT = E.A.DATA / "cs2" / "props_params.json"


def dataset():
    E.configure("cs2")
    m = pd.read_csv(E.HIST); m["date"] = pd.to_datetime(m.date, utc=True); m = m.sort_values("date", ignore_index=True)
    preds = {}
    E.map_elo(m, m.date.max() - pd.DateOffset(years=2, months=1), E.load_games(), preds)
    g = pd.read_csv(E.A.DATA / "cs2" / "players_games.csv")
    rounds = (g.ws + g.ls).groupby(g.match_id).sum()
    pm = pd.read_csv(E.A.DATA / "cs2" / "players_matches.csv")
    pm = pm[(pm.kills >= 0) & pm.match_id.isin(preds) & pm.match_id.isin(rounds.index)]
    d = pm.merge(m[["id", "date", "bo", "t1_id", "t2_id", "tier"]], left_on="match_id", right_on="id")
    d = d[d.bo.isin([1, 3])].sort_values(["date", "match_id"], ignore_index=True)
    d["rounds"] = d.match_id.map(rounds)
    pmap = d.match_id.map(lambda i: preds[i][1])
    d["p_team"] = np.where(d.team_id == d.t1_id, pmap, 1 - pmap)
    q3 = [E.bo3_scores(min(max(p, 1e-3), 1 - 1e-3)) for p in d.p_team]
    d["e_maps"] = np.where(d.bo == 1, 1.0, [2 + s[2, 1] + s[1, 2] for s in q3])
    return d


def features(d, hl, k):
    """KPR previa con olvido por serie del jugador, media de kills de sus 10 series previas y línea tipo casa."""
    a = 0.5 ** (1 / hl)
    kpr, base_kpr, last = np.zeros(len(d)), np.zeros(len(d)), {}
    glob_k = glob_r = 0.0
    for i, (pid, kl, rd) in enumerate(zip(d.player_id, d.kills, d.rounds)):
        K, R = last.get(pid, (0.0, 0.0))
        g0 = glob_k / glob_r if glob_r else 0.68
        kpr[i] = (K + k * g0) / (R + k); base_kpr[i] = g0
        last[pid] = (K * a + kl, R * a + rd); glob_k += kl; glob_r += rd
    d = d.assign(kpr=kpr)
    grp = d.groupby(["player_id", "bo"]).kills
    d["mean10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    d["line"] = np.floor(grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).median())) + 0.5
    return d


def nb_ll(y, mu, r): return stats.nbinom.logpmf(y, r, r / (r + mu))


def over_ll(y, mu, r, line):
    p = np.clip(1 - stats.nbinom.cdf(np.floor(line), r, r / (r + mu)), 1e-9, 1 - 1e-9); o = y > line
    return -(o * np.log(p) + (1 - o) * np.log(1 - p)), p


def main():
    d0 = dataset()
    end = d0.date.max()
    fit = ((d0.date >= end - pd.DateOffset(years=2)) & (d0.date < end - pd.DateOffset(years=1))).to_numpy()
    te = (d0.date >= end - pd.DateOffset(years=1)).to_numpy()
    rpm = float((d0.rounds / np.where(d0.bo == 1, 1, d0.rounds * 0 + 1)).where(d0.bo == 1).mean())  # rondas por mapa (BO1)
    best = None
    for hl, k in itertools.product([10, 25, 50, 100], [100, 300, 1000]):
        d = features(d0, hl, k)
        ok = d.line.notna().to_numpy() & d.mean10.notna().to_numpy()
        F_ = fit & ok
        lp = np.log(np.clip(d.p_team, 1e-3, 1 - 1e-3) / (1 - np.clip(d.p_team, 1e-3, 1 - 1e-3))).to_numpy()
        base = (d.kpr * d.e_maps * rpm).to_numpy(); y = d.kills.to_numpy()

        def nll(w):
            mu = base * np.exp(w[0] * lp[F_]); r = math.exp(w[1])
            return -nb_ll(y[F_], mu, r).sum()
        base_f = base[F_]
        nll_f = lambda w: -nb_ll(y[F_], base_f * np.exp(w[0] * lp[F_]), math.exp(w[1])).sum()
        w = optimize.minimize(nll_f, [0.1, math.log(10)], method="Nelder-Mead").x
        score = nll_f(w) / F_.sum()
        if best is None or score < best[0]: best = (score, hl, k, w, d, ok, lp, base, y)
    _, hl, k, w, d, ok, lp, base, y = best
    T_ = te & ok; F_ = fit & ok
    mu = base * np.exp(w[0] * lp); r = math.exp(w[1])
    # referencia: media de sus 10 últimas series del mismo BO con la misma dispersión ajustada en el año −2
    r0 = math.exp(optimize.minimize_scalar(lambda lr: -nb_ll(y[F_], d.mean10.to_numpy()[F_], math.exp(lr)).sum(), bounds=(0, 6), method="bounded").x)
    m10 = d.mean10.to_numpy()
    l_mod, p_mod = over_ll(y[T_], mu[T_], r, d.line.to_numpy()[T_])
    l_ref, _ = over_ll(y[T_], m10[T_], r0, d.line.to_numpy()[T_])
    e_mod, e_ref = -nb_ll(y[T_], mu[T_], r), -nb_ll(y[T_], m10[T_], r0)
    se = lambda a, b: float((a - b).std() / math.sqrt(len(a)))
    res = {"half_life": hl, "k": k, "c": round(float(w[0]), 4), "r": round(r, 2), "rounds_per_map": round(rpm, 2), "n_test": int(T_.sum()),
           "ou": round(float(l_mod.mean()), 4), "ou_ref": round(float(l_ref.mean()), 4), "ou_se": round(se(l_mod, l_ref), 4),
           "nll": round(float(e_mod.mean()), 4), "nll_ref": round(float(e_ref.mean()), 4), "nll_se": round(se(e_mod, e_ref), 4)}
    print(f"CS2 kills por serie: vida media {hl} series, K {k} rondas, c={w[0]:+.3f} (Elo del equipo), r={r:.1f}; test {T_.sum()} jugador-serie")
    print(f"  más/menos a la línea tipo casa (mediana de sus 10 últimas): referencia (media 10) {l_ref.mean():.4f} → modelo {l_mod.mean():.4f} "
          f"({l_mod.mean() - l_ref.mean():+.4f} ± {res['ou_se']})")
    print(f"  log-verosimilitud del número exacto: {e_ref.mean():.4f} → {e_mod.mean():.4f} ({e_mod.mean() - e_ref.mean():+.4f} ± {res['nll_se']})")
    hit = (y[T_] > d.line.to_numpy()[T_]) == (p_mod >= 0.5); conf = np.maximum(p_mod, 1 - p_mod)
    for lo in (0.55, 0.6, 0.65):
        s = conf >= lo
        print(f"  picks con prob. ≥ {lo:.0%}: {s.sum()} · prevista {conf[s].mean():.3f} → acierto {hit[s].mean():.3f}")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
