"""Sonda temporal: modelo de lenguaje local (llama.cpp + Qwen2.5 GGUF) en la CPU del runner: ¿instala, carga, cuánto tarda?"""
import json, time
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import interpreta as I
for repo, fn in (("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
                 ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf")):
    t0 = time.time(); p = hf_hub_download(repo, fn); t1 = time.time()
    llm = Llama(model_path=p, n_ctx=4096, n_threads=4, verbose=False); t2 = time.time()
    print(fn, f"descarga {t1 - t0:.0f} s, carga {t2 - t1:.0f} s", flush=True)
    for msg, hist in (("sí", [("bot", "¿Quieres la combinada segura de Girona – Almería?")]),
                      ("oye y qué tal le va últimamente al equipo blanco de la capital", []),
                      ("Añádeles más para subir la cuota, lo máximo que pueda pero de forma segura", [("bot", "Combinada segura · Girona – Almería")]),
                      ("cuánto suele tirar a puerta mbappe", []),
                      ("y el barça cuantos corners saca", [])):
        t = time.time()
        r = llm.create_chat_completion(messages=[{"role": "system", "content": I.SYSTEM}, {"role": "user", "content": I._prompt(msg, hist)}],
                                       response_format={"type": "json_object", "schema": I.SCHEMA}, temperature=0, max_tokens=120)
        print(f"  {time.time() - t:.1f} s · {msg} → {r['choices'][0]['message']['content']}", flush=True)
