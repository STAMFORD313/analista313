"""Valor contra Pinnacle ("line shopping"): la prob. sin margen de Pinnacle es la verdad y se apuesta cuando la mejor
cuota de las casas blandas la supera en al menos X. `python pinnacle_value.py` hace el backtest con data/hist
(apertura: PS* contra casas blandas / Max*; cierre: PSC* contra lo mismo de cierre), por mercado, liga y umbral."""
import numpy as np
import pandas as pd

SOFT = ["B365", "BW", "WH", "IW", "VC", "1XB", "BF"]  # casas blandas con columnas propias en football-data
MARKETS = {"1X2": ("H", "D", "A"), "OU": (">2.5", "<2.5")}


def fair(pin):
    """Cuotas de Pinnacle de un mercado (lista) → probabilidades sin margen (normalización proporcional)."""
    inv = 1 / np.asarray(pin, float)
    return inv / inv.sum()


def value(pin, best, x):
    """pin = cuotas de Pinnacle del mercado, best = mejor cuota disponible de cada resultado (mismo orden) →
    [(índice, ventaja)] de los resultados con prob_Pinnacle * cuota - 1 >= x."""
    ev = fair(pin) * np.asarray(best, float) - 1
    return [(i, float(e)) for i, e in enumerate(ev) if e >= x]


def _cols(prefix, s, close):  # nombre de columna football-data: PSH / PSCH, P>2.5 / PC>2.5, B365>2.5 / B365C>2.5
    c = "C" if close else ""
    return f"{prefix}{c}{s}"


def load():
    from analyst import HIST, load_csv
    frames = [load_csv(f.read_bytes(), None).assign(Div=f.stem.split("_")[0]) for f in sorted(HIST.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["FTHG", "FTAG"])
    for c in df.columns[df.columns.str.contains(r"^(?:B365|BW|WH|IW|VC|1XB|BF|PS|P|Max|BbMx)C?(?:[HDA]|[<>]2\.5)$")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("Date", ignore_index=True)


def backtest(df, market, close, book, xs):
    """book: 'soft' (mejor de SOFT) o 'max' (Max* de football-data, incluye exchanges). → DataFrame de apuestas."""
    sides = MARKETS[market]
    pin_p = "PS" if market == "1X2" else "P"
    pin = np.column_stack([df.get(_cols(pin_p, s, close), pd.Series(np.nan, df.index)) for s in sides])
    if book == "soft":
        best = np.column_stack([np.nanmax(np.column_stack([df.get(_cols(b, s, close), pd.Series(np.nan, df.index))
                                                          for b in SOFT]), 1) for s in sides])
    else:
        mx = [df.get(_cols("Max", s, close), pd.Series(np.nan, df.index)) for s in sides]
        if not close:  # temporadas antiguas: BbMx (apertura)
            mx = [m.fillna(df.get(f"BbMx{s}", pd.Series(np.nan, df.index))) for m, s in zip(mx, sides)]
        best = np.column_stack(mx)
    ok = np.isfinite(pin).all(1) & np.isfinite(best).all(1) & (pin > 1).all(1)
    p = np.where(ok[:, None], 1 / np.where(ok[:, None], pin, 1), np.nan)
    p /= p.sum(1, keepdims=True)
    ev = p * best - 1
    g = df.FTHG.to_numpy() + df.FTAG.to_numpy()
    win = (np.column_stack([df.FTHG > df.FTAG, df.FTHG == df.FTAG, df.FTHG < df.FTAG]) if market == "1X2"
           else np.column_stack([g > 2.5, g < 2.5]))
    out = []
    for x in xs:
        r, c = np.nonzero(ok[:, None] & (ev >= x) & (ev <= 0.5))  # >50 %: casi seguro error de datos
        out.append(pd.DataFrame({"x": x, "Div": df.Div.to_numpy()[r], "season": df.Date.dt.year.to_numpy()[r],
                                 "ev": ev[r, c], "odds": best[r, c], "gain": np.where(win[r, c], best[r, c] - 1, -1.0)}))
    return pd.concat(out, ignore_index=True)


def summary(b, by):
    s = b.groupby(by).gain.agg(["mean", "std", "count"])
    s["se"] = s["std"] / np.sqrt(s["count"])
    s["z"] = s["mean"] / s["se"]
    return s[["count", "mean", "se", "z"]].rename(columns={"count": "apuestas", "mean": "ROI", "se": "error"})


def main():
    df = load()
    pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
    xs = [0.0, 0.01, 0.02, 0.03, 0.05]
    for market in MARKETS:
        for close in (False, True):
            for book in ("soft", "max"):
                b = backtest(df, market, close, book, xs)
                print(f"\n== {market} {'cierre' if close else 'apertura'} vs {book} ==")
                print(summary(b, "x").to_string())
                if book == "soft" and not close:
                    print(summary(b[b.x == 0.02], "Div").to_string())
                    print(summary(b[b.x == 0.02], "season").tail(8).to_string())


if __name__ == "__main__":
    assert abs(fair([2.0, 2.0]).sum() - 1) < 1e-12 and value([1.9, 1.9], [2.1, 1.8], 0.02) == [(0, value([1.9, 1.9], [2.1, 1.8], 0.02)[0][1])]
    main()
