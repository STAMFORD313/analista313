"""Coordinador del equipo de bots expertos que nutren y entrenan al bot principal (sin IA: solo scripts, 0 tokens).

Cada experto es un script con UNA tarea y una frecuencia. En cada pasada (lo lanza el reloj cada 30 min, workflow
datos_espn.yml «nutricion») el coordinador ejecuta los que tocan, cada uno con su límite de tiempo y aislado (si uno
falla, los demás siguen), y deja el parte en data/expertos.json (última ejecución, duración, estado y última línea de
salida). Los entrenadores solo cambian parámetros si no empeoran fuera de muestra (lo comprueba cada script).
Uso: python nutricion.py [nombre ...]  (sin nombres: los que tocan) · python nutricion.py --lista"""
import json, subprocess, sys, time

import pandas as pd

import analyst as A

STATE = A.DATA / "expertos.json"
PY = sys.executable

# nombre: (cada cuántas horas, minutos máximos, comando, qué hace)
EXPERTS = {
    # ── recopiladores de datos ───────────────────────────────────────────────────────────────────────────────
    "datos_futbol_jugadores": (0, 40, [PY, "espn_soccer.py", "3000"], "ESPN fútbol: alineaciones, jugadores, porteros y árbitro de 18 competiciones"),
    "datos_nba_nhl": (6, 60, [PY, "-c", "import nba_players as N; [N.download(l, refetch=True) for l in ('nba', 'nhl')]"],
                      "ESPN NBA/NHL: boxscores con puntos, rebotes, asistencias, triples y tiros por jugador"),
    "sondeo_esports": (12, 10, [PY, "sondeo.py"], "Disciplinas nuevas de bo3.gg y esports con cuota en Unibet"),
    # ── evaluadores (comparan con la casa usando lo ya jugado) ─────────────────────────────────────────────────
    "props_futbol": (0, 15, [PY, "football_players.py"], "Props de jugador de fútbol: backtest y liquidación de las cuotas de Unibet"),
    "props_nba": (0, 10, [PY, "nba_props.py"], "Props de jugador NBA: liquida las cuotas de Unibet y mide el modelo contra la casa"),
    "estrategias": (0, 10, [PY, "estrategias.py"], "Busca segmentos rentables/perdedores con validación (ajusta el stake de la banca)"),
    "juez_mercado": (24, 40, [PY, "market_check.py"], "Modelo vs cuotas de CIERRE de DraftKings por liga y mercado: fija el peso del modelo (0 = no apostar)"),
    # ── entrenadores (diarios; solo guardan si no empeoran fuera de muestra) ──────────────────────────────────
    "entrena_esports": (24, 90, ["bash", "-c", "for g in cs2 dota2 lol r6 mlbb; do python train_cs2.py $g auto || echo fallo $g; done"],
                        "Elo por mapa/serie de CS2, Dota 2, LoL, R6 y MLBB"),
    "entrena_fc27": (24, 30, ["bash", "-c", "python train_fc.py; python train_fc_calib.py"], "Modelo y recalibración de FC 27"),
    "entrena_props": (24, 30, ["bash", "-c", "python train_cs2_props.py; python train_nba_props.py"], "Props de kills CS2 y de jugadores NBA"),
    "entrena_espn": (24, 90, ["bash", "-c", "pip install -q scikit-learn; python train_espn.py; python train_espn.py ncaa"],
                     "Elo y totales de NBA, NFL, NHL, MLB, tenis, UFC y NCAA"),
    "entrena_futbol": (24, 90, ["bash", "-c", "pip install -q scikit-learn; python train_football.py; python football_stats.py; "
                                              "python train_form.py; python train_market_stats.py; python train_sgp.py"],
                       "Fútbol: 1X2/goles, córners/tarjetas/faltas/tiros, forma, cuotas como dato y cópula de combinadas"),
    "entrena_combinadas_us": (24, 10, [PY, "train_nba_sgp.py"], "Correlación ganador-total de NBA/NFL/NHL"),
}


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def due(name, st, now):
    every = EXPERTS[name][0]
    last = st.get(name, {}).get("ultima")
    return every == 0 or not last or now - pd.Timestamp(last) >= pd.Timedelta(hours=every) - pd.Timedelta(minutes=5)


def run(names=None, budget_min=240):
    st, now, t_start = load(), pd.Timestamp.now(tz="UTC"), time.time()
    todo = names or [n for n in EXPERTS if due(n, st, now)]
    for n in todo:
        every, limit, cmd, what = EXPERTS[n]
        left = budget_min - (time.time() - t_start) / 60
        if left < 5: print("sin tiempo para", n); break
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=min(limit, left) * 60)
            ok, out = r.returncode == 0, (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired as e:
            ok, out = False, f"tiempo agotado ({limit} min)"
        tail = [l for l in out.splitlines() if l.strip()][-1:] or [""]
        st[n] = {"ultima": now.isoformat(), "ok": ok, "segundos": round(time.time() - t0), "salida": tail[0][:200], "tarea": what}
        print(f"{'✅' if ok else '❌'} {n} ({st[n]['segundos']} s): {tail[0][:150]}")
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return st


def report():
    """Líneas del parte del equipo (para Telegram o el log)."""
    st = load()
    return [f"{'✅' if v.get('ok') else '❌'} {n}: {v.get('ultima', '')[:16]} · {v.get('segundos')} s" for n, v in st.items()]


if __name__ == "__main__":
    if "--lista" in sys.argv:
        for n, (e, m, _, w) in EXPERTS.items(): print(f"{n:24} cada {e or 0.5:>4} h · máx {m:>3} min · {w}")
    else: run([a for a in sys.argv[1:] if a in EXPERTS] or None)
