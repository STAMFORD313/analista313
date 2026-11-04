"""Escucha los botones de Telegram casi sin pausa (cada ejecución de GitHub Actions dura ~5 h 45 min y la
siguiente la releva): al pulsar un deporte, el bot muestra solo las apuestas de ese deporte en un único mensaje
que se va editando; y si le escribes, contesta con estadísticas o su opinión de una apuesta (chat.py); también a notas de voz (voz.py, Whisper local). Lee el estado que publica la ejecución horaria (git fetch cada 5 min)."""
import json, os, subprocess, time

import requests

import analyst as A
import chat as charla
import voz
import radar
import simulaciones

RUN_SECONDS, REFRESH, KICK = 5 * 3600 + 45 * 60, 300, 1800
LIVE_EVERY, LIVE_MAX = 180, 4 * 3600  # tarjeta «📡 en juego»: se refresca cada 3 min hasta que acaben (máx. 4 h)
SWEEP = 3 * 3600  # barrido de combinadas buenas (chat.sweep) cada 3 h
API = f"https://api.telegram.org/bot{A.TG}"


def load_state():
    subprocess.run("git fetch -q --depth 1 origin main && git reset -q --hard origin/main", shell=True)
    return json.loads(A.STATE.read_text(encoding="utf-8"))


def keyboard(): return json.dumps({"inline_keyboard": radar.sport_buttons() + [[{"text": "🔄 Actualizar", "callback_data": "dep:refresh"}]]})


def target_message(cq, state):
    """Mensaje a editar: el de la vista pulsada (también el de un relevo anterior), nunca el marcador fijado."""
    mid = ((cq or {}).get("message") or {}).get("message_id")
    return mid if mid and mid != state.get("scoreboard_id") else None


def kick_analysis():
    """Lanza el workflow `analista` cada 30 min (el cron de GitHub se salta horas). Necesita GITHUB_TOKEN con actions: write."""
    repo, token = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_TOKEN")
    if not (repo and token): return
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/actions/workflows/analista.yml/dispatches",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          json={"ref": "main"}, timeout=30)
        if r.status_code != 204: print("dispatch analista", r.status_code, r.text[:200])
    except requests.RequestException as e: print("dispatch analista", e)


def boleto_imagen(msg):
    """Foto de un boleto → Gemini lee las piernas y cuotas (interpreta.read_slip) → el bot las analiza (chat.analyze_slip)."""
    import interpreta, os
    try:
        fid = msg["photo"][-1]["file_id"] if msg.get("photo") else msg["document"]["file_id"]
        path = voz.download(fid)
        if not path: return "No he podido descargar la imagen."
        try: data = open(path, "rb").read()
        finally: os.unlink(path)
        slip = interpreta.read_slip(data, (msg.get("document") or {}).get("mime_type") or "image/jpeg")
        if not slip: return "No he sabido leer el boleto de la imagen 🤔 ¿Me escribes las apuestas?"
        if msg.get("caption"): print("boleto con texto:", msg["caption"][:100])
        return charla.analyze_slip(slip)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"No he podido analizar la imagen ({A.esc(type(e).__name__)})."


def refresh_live(cards, now=None):
    """Reedita las tarjetas de apuestas en directo; quita las que ya no tienen nada en juego o llevan demasiado."""
    now = now or time.time()
    for mid, c in list(cards.items()):
        if now - c["t"] < LIVE_EVERY: continue
        try: txt = simulaciones.live(filtro=c["filtro"])
        except Exception as e: print("directo", repr(e)); continue
        c["t"] = now
        done = not txt or not any(x in txt for x in ("⏳", "🟢", "🔴", "🕒"))
        if txt:
            foot = "\n\n🔄 " + time.strftime("%H:%M", time.gmtime()) + " UTC · " + ("terminado" if done else "se actualiza cada 3 min")
            A.tg("editMessageText", message_id=mid, text=(txt + foot)[:4000], parse_mode="HTML")
        if done or now - c["start"] > LIVE_MAX: cards.pop(mid)
        if done:  # acabado: liquidar ya y mandar el resultado con el porqué
            try:
                for m in simulaciones.settle(): A.tg("sendMessage", text=m[:4000], parse_mode="HTML")
            except Exception as e: print("simuladas", repr(e))


def main():
    voz.warm()  # carga Whisper en segundo plano: el primer audio no espera la descarga del modelo
    state, loaded, view, last, offset, shown = load_state(), time.time(), None, "futbol", None, None
    end = time.time() + RUN_SECONDS
    live_cards = {}  # message_id → {filtro, t, start}
    kicked, swept = 0, time.time() - SWEEP + 600  # primer barrido a los 10 min de arrancar
    while time.time() < end:
        if time.time() - kicked > KICK: kick_analysis(); kicked = time.time()
        if time.time() - swept > SWEEP:
            swept = time.time()
            try:
                msg = charla.sweep()
                if msg: A.tg("sendMessage", text=msg[:4000], parse_mode="HTML")
            except Exception as e: print("barrido", repr(e))
        refresh_live(live_cards)
        if time.time() - loaded > REFRESH:
            state, loaded = load_state(), time.time()
            try:  # simuladas del chat: liquidar con ESPN y avisar
                for msg in simulaciones.settle(): A.tg("sendMessage", text=msg[:4000], parse_mode="HTML")
            except Exception as e: print("simuladas", repr(e))
        try:
            ups = requests.get(f"{API}/getUpdates", params={"timeout": 50, "offset": offset,
                                                             "allowed_updates": '["callback_query","message"]'}, timeout=70).json()
        except (requests.RequestException, ValueError) as e:
            print("getUpdates", e); time.sleep(5); continue
        for u in ups.get("result", []):
            offset = u["update_id"] + 1
            cq, msg = u.get("callback_query"), u.get("message")
            chat = str(((cq or {}).get("message") or msg or {}).get("chat", {}).get("id"))
            if chat != str(A.CHAT): continue  # solo responde al dueño
            slug = None
            if cq and cq.get("data", "").startswith("dep:"):
                A.tg("answerCallbackQuery", callback_query_id=cq["id"])
                slug = cq["data"][4:]
                if slug == "refresh": state, loaded, slug = load_state(), time.time(), last
            elif msg and msg.get("text", "").startswith(("/start", "/menu")):
                slug = last
            elif msg and (msg.get("photo") or str((msg.get("document") or {}).get("mime_type", "")).startswith("image/")):
                reply = boleto_imagen(msg)  # foto de un boleto: la IA lo lee y el bot lo analiza
                A.tg("sendMessage", text=reply[:4000], parse_mode="HTML", reply_to_message_id=msg.get("message_id"))
                continue
            elif msg and (msg.get("text") or msg.get("voice") or msg.get("audio")):  # frase libre o nota de voz (chat.py, sin IA)
                quoted = said = None
                try:
                    said = msg.get("text") or voz.from_message(msg)
                    quoted = (msg.get("reply_to_message") or {}).get("text") or (msg.get("reply_to_message") or {}).get("caption")
                    reply = charla.answer(said, quoted=quoted) if said else "🎤 No he entendido el audio, ¿me lo repites?"
                    live = reply.startswith("📡")
                    if not msg.get("text") and said: reply = f"🎤 <i>«{A.esc(said)}»</i>\n\n{reply}"
                except Exception as e:
                    import traceback; traceback.print_exc()
                    reply, live = f"No he podido mirarlo ahora ({A.esc(type(e).__name__)}: {A.esc(str(e)[:150])}).", False
                if live: reply += "\n\n🔄 Te la voy actualizando aquí cada 3 min."
                sent = A.tg("sendMessage", text=reply[:4000], parse_mode="HTML", reply_to_message_id=msg.get("message_id"))
                if live and sent:
                    for k in [k for k, c in live_cards.items() if c["filtro"] == (quoted or said)]: live_cards.pop(k)  # la vieja se queda quieta
                    live_cards[sent["message_id"]] = {"filtro": quoted or said, "t": time.time(), "start": time.time()}
                continue
            if not slug: continue
            last, text = slug, radar.sport_view(state, slug)
            target = target_message(cq, state)
            if target == view and text == shown: continue  # igual que lo que ya se ve: Telegram daría "not modified"
            if target and A.tg("editMessageText", message_id=target, text=text, parse_mode="HTML", reply_markup=keyboard()):
                if view and view != target: A.tg("deleteMessage", message_id=view)
                view, shown = target, text
                continue
            if view: A.tg("deleteMessage", message_id=view)  # un solo mensaje de vista: no llenar el chat
            sent = A.tg("sendMessage", text=text, parse_mode="HTML", reply_markup=keyboard())
            view, shown = sent and sent["message_id"], text


if __name__ == "__main__":
    main()
