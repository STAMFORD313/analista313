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


def main():
    voz.warm()  # carga Whisper en segundo plano: el primer audio no espera la descarga del modelo
    state, loaded, view, last, offset, shown = load_state(), time.time(), None, "futbol", None, None
    end = time.time() + RUN_SECONDS
    kicked = 0
    while time.time() < end:
        if time.time() - kicked > KICK: kick_analysis(); kicked = time.time()
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
            elif msg and (msg.get("text") or msg.get("voice") or msg.get("audio")):  # frase libre o nota de voz (chat.py, sin IA)
                try:
                    said = msg.get("text") or voz.from_message(msg)
                    reply = charla.answer(said) if said else "🎤 No he entendido el audio, ¿me lo repites?"
                    if not msg.get("text") and said: reply = f"🎤 <i>«{A.esc(said)}»</i>\n\n{reply}"
                except Exception as e:
                    import traceback; traceback.print_exc()
                    reply = f"No he podido mirarlo ahora ({A.esc(type(e).__name__)}: {A.esc(str(e)[:150])})."
                A.tg("sendMessage", text=reply[:4000], parse_mode="HTML", reply_to_message_id=msg.get("message_id"))
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
