"""Sonda temporal: nombres de ESPN frente a los de Kambi (Unibet) en los partidos de 48 h."""
import chat, kambi
ev = kambi.events()
print("Kambi fútbol:", len(ev), "partidos; ej.:", [(e["home"], e["away"], str(e["start"])[:16]) for e in ev[:8]])
for code, evs in chat.upcoming(chat.DAY_LEAGUES).items():
    for t, h, a, _ in evs:
        print(code, h, "-", a, "→", kambi.find(h, a, t))
