"""Intérprete de lenguaje natural del chat de Telegram: solo cuando las reglas de chat.py no entienden la frase.

Recibe la frase (escrita o transcrita de un audio) y las últimas líneas de la conversación, y la traduce a UNA orden que
chat.py sabe ejecutar («combinada Girona Almería», «estadísticas de Mbappé»…) o a una respuesta corta si hay que
preguntar algo. Los números (probabilidades, cuotas) los pone siempre chat.py con sus modelos medidos: el intérprete
no inventa datos. Necesita el secreto ANTHROPIC_API_KEY en GitHub (menu.yml); sin él, chat.py sigue con sus reglas.
Modelo: CHAT_MODEL (por defecto claude-opus-5-5, esfuerzo bajo); unos pocos céntimos al día con uso normal."""
import json, os

MODEL = os.getenv("CHAT_MODEL", "claude-opus-5-5")
SYSTEM = """Eres el intérprete de un bot de apuestas deportivas en Telegram. El usuario escribe o dicta en español coloquial \
(a veces con errores de transcripción de voz). Tu único trabajo es traducir su último mensaje a UNA orden del bot, usando la \
conversación previa para resolver referencias («sí», «ese partido», «y del otro equipo», «súbele la cuota»).

Órdenes que entiende el bot (escríbelas exactamente con este estilo, con nombres reales de equipos/jugadores):
- «estadísticas de <jugador o equipo>»
- «próximo partido de <equipo>» · «último resultado de <equipo>»
- «quién gana <local> <visitante>» · «gana <equipo> a <cuota>»
- «<jugador o equipo> más de <línea> <estadística>» (estadísticas: goles, tiros, tiros a puerta, córners, tarjetas, faltas, \
fueras de juego, puntos, rebotes, asistencias, triples); con cuota: «… a 1.90»; total de un partido: «<local> <visitante> más de 9.5 córners»
- varias piernas en una combinada: unirlas con « y » («Mbappé más de 0.5 tiros a puerta y Madrid Barcelona más de 7.5 córners @2.1»)
- «combinada <local> <visitante>» (combinada segura de un partido) · «combinada del día» (varios partidos, para subir la cuota con seguridad)
- «ayuda»

Si el último mensaje responde a una pregunta del bot («sí», «vale», «venga», «ese»), devuelve la orden que el bot ofrecía. \
Si de verdad falta un dato imprescindible, no adivines: pon orden vacía y en «respuesta» una pregunta corta y concreta \
(en español, tuteando). Si el usuario charla o pregunta algo fuera de lo que hace el bot, responde en «respuesta» en una o \
dos frases y sugiere lo más parecido que sí puede hacer. Nunca inventes probabilidades, cuotas ni estadísticas."""
SCHEMA = {"type": "object", "properties": {"orden": {"type": "string"}, "respuesta": {"type": "string"}},
          "required": ["orden", "respuesta"], "additionalProperties": False}


def enabled(): return bool(os.getenv("ANTHROPIC_API_KEY"))


def route(text, history=()):
    """{"orden": str, "respuesta": str} o None (sin clave, error de la API o rechazo)."""
    if not enabled(): return None
    try: import anthropic
    except ImportError: print("intérprete: falta el paquete anthropic"); return None
    convo = "\n".join(f"{who}: {msg}" for who, msg in list(history)[-8:]) or "(sin conversación previa)"
    try:
        r = anthropic.Anthropic(timeout=30.0, max_retries=1).beta.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # si el modelo rechaza, otro lo reintenta
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": f"Conversación reciente:\n{convo}\n\nÚltimo mensaje del usuario: {text}"}])
    except anthropic.APIStatusError as e: print("intérprete", e.status_code, str(e)[:200]); return None
    except anthropic.APIConnectionError as e: print("intérprete sin conexión", repr(e)); return None
    if r.stop_reason in ("refusal", "max_tokens"): print("intérprete", r.stop_reason); return None
    txt = next((b.text for b in r.content if b.type == "text"), "")
    try: out = json.loads(txt)
    except ValueError: return None
    return {"orden": str(out.get("orden") or "").strip(), "respuesta": str(out.get("respuesta") or "").strip()}
