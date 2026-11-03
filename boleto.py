"""Formato único de boleto para todas las tarjetas de Telegram (apuestas, combinadas, simulaciones y predicciones):

<b>Madrid 🆚 Sevilla</b>
  ↳ Gana Madrid ✅
  ↳ Marca Mbappé ✅
  ↳ Más de 2.5 córners ❌   ← la que falló
Cuota: 5.60 · Apuesta: 1.90 € · Ganado: 10.64 €   (o «Perdido: 1.90 €»)

Las predicciones (aún sin jugar) van igual pero sin marcas y con «Si acierta: x €»."""
import analyst as A

CUR = "€"
MARK = {True: " ✅", False: " ❌", "nula": " ➖", None: ""}


def legs_block(legs, match=None, when=None):
    """legs: [(texto, acierto True/False/'nula'/None)] o [(partido, texto, acierto)] en combinadas de varios partidos."""
    out = []
    if match: out.append(f"<b>{A.esc(match).replace(' vs ', ' 🆚 ')}</b>" + (f" · 🕒 {when}" if when else ""))
    last = None
    for l in legs:
        if len(l) == 3:  # combinada de varios partidos: el partido encima de su selección
            m, t, ok = l
            if m != last: out.append(f"<b>{A.esc(m).replace(' vs ', ' 🆚 ')}</b>"); last = m
        else: t, ok = l
        out.append(f"  ↳ {A.esc(t)}{MARK.get(ok, '')}")
    return out


def money(odds=None, stake=None, result=None, odds_label="Cuota"):
    """result: True ganada · False perdida · 'nula' · None pendiente."""
    parts = []
    if odds: parts.append(f"{odds_label}: <b>{odds:.2f}</b>")
    if stake is not None:
        parts.append(f"Apuesta: {stake:.2f} {CUR}")
        if result is True and odds: parts.append(f"✅ Ganado: <b>{stake * odds:.2f} {CUR}</b>")
        elif result is False: parts.append(f"❌ Perdido: <b>{stake:.2f} {CUR}</b>")
        elif result == "nula": parts.append(f"➖ Nula: se devuelven {stake:.2f} {CUR}")
        elif odds: parts.append(f"Si acierta: {stake * odds:.2f} {CUR}")
    return " · ".join(parts)


def boleto(match, legs, odds=None, stake=None, result=None, when=None, odds_label="Cuota"):
    """Bloque completo (lista de líneas). result=None y legs sin marcas = predicción."""
    if result is None and legs and all((l[-1] is not None) for l in legs):
        result = "nula" if all(l[-1] == "nula" for l in legs) else False if any(l[-1] is False for l in legs) else True
    m = money(odds, stake, result, odds_label)
    return [*legs_block(legs, match, when), *([m] if m else [])]
