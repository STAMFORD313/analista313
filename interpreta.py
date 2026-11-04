"""Intérprete de lenguaje natural del chat de Telegram: solo cuando las reglas de chat.py no entienden la frase.

Recibe la frase (escrita o transcrita de un audio) y las últimas líneas de la conversación, y la traduce a UNA orden que
chat.py sabe ejecutar («combinada Girona Almería», «estadísticas de Mbappé»…) o a una respuesta corta si hay que
preguntar algo. Los números (probabilidades, cuotas) los pone siempre chat.py con sus modelos medidos: el intérprete
no inventa datos. Proveedores, en orden: Claude si hay ANTHROPIC_API_KEY (de pago, opcional); Gemini con el secreto GEMINI_API_KEY que ya
usa el bot (plan gratis: flash-lite y, si falla, flash); GitHub Models con el token del workflow
(GH_MODELS_TOKEN; probado el 4-nov: aún no responde). Sin ninguno, chat.py sigue con sus reglas. `python interpreta.py` hace una prueba (workflow probar_interprete.yml)."""
import json, os

MODEL = os.getenv("CHAT_MODEL", "claude-opus-5-5")
GH_MODELS = os.getenv("GH_MODELS_MODEL", "openai/gpt-4.1-mini").split(",")  # se prueban en orden
GH_URL = "https://models.github.ai/inference/chat/completions"
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


GEMINI_MODELS = os.getenv("CHAT_GEMINI_MODELS", "gemini-flash-lite-latest,gemini-flash-latest")  # lite: ~0,6 s y acertó 4/4 en la prueba; flash daba 503.split(",")
_DEAD = set()  # modelos de Gemini con la cuota del día agotada (429)


def enabled(): return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GH_MODELS_TOKEN"))


def route_gemini(text, history=()):
    """Gemini (google-genai, salida JSON). None si no hay clave, se agota la cuota o falla."""
    key = os.getenv("GEMINI_API_KEY")
    if not key: return None
    try: from google import genai
    except ImportError: print("intérprete: falta google-genai"); return None
    client = genai.Client(api_key=key, http_options={"timeout": 15_000})
    for model in [m.strip() for m in GEMINI_MODELS if m.strip() not in _DEAD]:
        try:
            r = client.models.generate_content(model=model, contents=_prompt(text, history),
                                               config={"system_instruction": SYSTEM + "\n\nResponde SOLO con un JSON: "
                                                       "{\"orden\": \"...\", \"respuesta\": \"...\"}.",
                                                       "response_mime_type": "application/json", "temperature": 0})
            out = _parse(r.text)
            if out: return out
            print("Gemini respuesta rara", model, (r.text or "")[:200])
        except Exception as e:  # cuota, red o respuesta mala: al siguiente modelo
            print("Gemini", model, str(e)[:200])
            if getattr(e, "code", None) == 429: _DEAD.add(model.strip())
    return None


def _prompt(text, history):
    convo = "\n".join(f"{who}: {msg}" for who, msg in list(history)[-8:]) or "(sin conversación previa)"
    return f"Conversación reciente:\n{convo}\n\nÚltimo mensaje del usuario: {text}"


def _parse(txt):
    import re
    m = re.search(r"\{.*\}", txt or "", re.S)
    try: out = json.loads(m.group(0)) if m else None
    except ValueError: return None
    if not isinstance(out, dict): return None
    return {"orden": str(out.get("orden") or "").strip(), "respuesta": str(out.get("respuesta") or "").strip()}


def route_github(text, history=()):
    """GitHub Models (API tipo chat completions) con el token del workflow. None si falla o se agota el cupo."""
    import requests
    tok = os.getenv("GH_MODELS_TOKEN")
    if not tok: return None
    msgs = [{"role": "system", "content": SYSTEM + "\n\nResponde SOLO con un JSON: {\"orden\": \"...\", \"respuesta\": \"...\"}."},
            {"role": "user", "content": _prompt(text, history)}]
    head = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    for model in GH_MODELS:
        for fmt in ({"type": "json_schema", "json_schema": {"name": "orden", "schema": SCHEMA, "strict": True}}, {"type": "json_object"}):
            try: r = requests.post(GH_URL, headers=head, json={"model": model.strip(), "messages": msgs, "temperature": 0,
                                                                  "max_tokens": 300, "response_format": fmt}, timeout=30)
            except requests.RequestException as e: print("GitHub Models", repr(e)); return None
            if r.status_code == 400: print("GitHub Models 400", model, r.text[:300]); continue  # formato no admitido: el siguiente
            if not r.ok: print("GitHub Models", model, r.status_code, r.text[:200]); break
            try: out = _parse(r.json()["choices"][0]["message"]["content"])
            except (ValueError, KeyError, IndexError): out = None
            if not out: print("GitHub Models respuesta rara", r.text[:300])
            if out: return out
    return None


def route(text, history=()):
    """{"orden": str, "respuesta": str} o None (sin proveedor, error o rechazo)."""
    if not os.getenv("ANTHROPIC_API_KEY"): return route_gemini(text, history) or route_github(text, history)
    try: import anthropic
    except ImportError: print("intérprete: falta el paquete anthropic"); return None
    try:
        r = anthropic.Anthropic(timeout=30.0, max_retries=1).beta.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # si el modelo rechaza, otro lo reintenta
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": _prompt(text, history)}])
    except anthropic.APIStatusError as e: print("intérprete", e.status_code, str(e)[:200]); return None
    except anthropic.APIConnectionError as e: print("intérprete sin conexión", repr(e)); return None
    if r.stop_reason in ("refusal", "max_tokens"): print("intérprete", r.stop_reason); return None
    return _parse(next((b.text for b in r.content if b.type == "text"), ""))


if __name__ == "__main__":  # prueba: python interpreta.py
    for msg, hist in (("sí", [("bot", "¿Quieres la combinada segura de Girona – Almería?")]),
                      ("oye y qué tal le va últimamente al equipo blanco de la capital", []),
                      ("Añádeles más para subir la cuota, lo máximo que pueda pero de forma segura", [("bot", "Combinada segura · Girona – Almería")]),
                      ("cuánto suele tirar a puerta mbappe", [])):
        print(msg, "→", route(msg, hist))
