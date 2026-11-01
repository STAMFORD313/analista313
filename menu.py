"""Escucha los botones de Telegram casi sin pausa (cada ejecución de GitHub Actions dura ~5 h 45 min y la
siguiente la releva): al pulsar un deporte, el bot muestra solo las apuestas de ese deporte en un único mensaje
que se va editando. Lee el estado que publica la ejecución horaria (git fetch cada 5 min)."""
import json, subprocess, time

import requests

import analyst as A
import radar

RUN_SECONDS, REFRESH = 5 * 3600 + 45 * 60, 300
API = f"https://api.telegram.org/bot{A.TG}"


def load_state():
    subprocess.run("git fetch -q --depth 1 origin main && git reset -q --hard origin/main", shell=True)
    return json.loads(A.STATE.read_text(encoding="utf-8"))


def keyboard(): return json.dumps({"inline_keyboard": radar.sport_buttons() + [[{"text": "🔄 Actualizar", "callback_data": "dep:refresh"}]]})


def main():
    state, loaded, view, last, offset = load_state(), time.time(), None, "futbol", None
    end = time.time() + RUN_SECONDS
    while time.time() < end:
        if time.time() - loaded > REFRESH: state, loaded = load_state(), time.time()
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
            if not slug: continue
            last, text = slug, radar.sport_view(state, slug)
            in_view = cq and cq["message"]["message_id"] == view
            if not (in_view and A.tg("editMessageText", message_id=view, text=text, parse_mode="HTML", reply_markup=keyboard())):
                if view: A.tg("deleteMessage", message_id=view)  # un solo mensaje de vista: no llenar el chat
                sent = A.tg("sendMessage", text=text, parse_mode="HTML", reply_markup=keyboard())
                view = sent and sent["message_id"]


if __name__ == "__main__":
    main()
