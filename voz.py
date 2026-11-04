"""Notas de voz de Telegram → texto, sin IA de pago (0 tokens): Whisper «small» en la CPU del runner de GitHub
(faster-whisper, int8; el modelo, ~250 MB, se descarga de Hugging Face la primera vez en cada escucha de menu.py).
El texto pasa al chat (chat.answer) como si lo hubieras escrito."""
import os, tempfile

import requests

import analyst as A

MODEL = os.getenv("WHISPER_MODEL", "small")  # «base» es el doble de rápido pero entiende peor los nombres
_M = {}


def model():
    if "m" not in _M:
        from faster_whisper import WhisperModel
        _M["m"] = WhisperModel(MODEL, device="cpu", compute_type="int8")
    return _M["m"]


def transcribe(path):
    segs, _ = model().transcribe(path, language="es", beam_size=5, vad_filter=True,
                                 initial_prompt="Apuestas de fútbol, NBA y esports: córners, tarjetas, tiros a puerta, cuota, combinada.")
    return " ".join(s.text.strip() for s in segs).strip()


def download(file_id):
    """Ruta local del audio de Telegram (voice/audio), o None."""
    info = A.tg("getFile", file_id=file_id)
    if not info or not info.get("file_path"): return None
    r = requests.get(f"https://api.telegram.org/file/bot{A.TG}/{info['file_path']}", timeout=60)
    if not r.ok: return None
    f = tempfile.NamedTemporaryFile(suffix=os.path.splitext(info["file_path"])[1] or ".oga", delete=False)
    f.write(r.content); f.close()
    return f.name


def from_message(msg):
    """Texto de una nota de voz o audio del mensaje, o None si no lleva."""
    a = msg.get("voice") or msg.get("audio")
    if not a: return None
    p = download(a["file_id"])
    if not p: return ""
    try: return transcribe(p)
    finally: os.unlink(p)
