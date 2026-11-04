"""Sonda temporal: qué dirección de GitHub Models responde con el token del workflow (probar_interprete.yml)."""
import os, requests
tok = os.environ["GH_MODELS_TOKEN"]
H = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
     "Content-Type": "application/json"}
body = {"model": "openai/gpt-4.1-mini", "messages": [{"role": "user", "content": "Di solo: hola"}], "max_tokens": 10}
for url, b in (("https://models.github.ai/inference/chat/completions", body),
               ("https://models.github.ai/orgs/STAMFORD313/inference/chat/completions", body),
               ("https://models.inference.ai.azure.com/chat/completions", dict(body, model="gpt-4o-mini"))):
    try:
        r = requests.post(url, headers=H, json=b, timeout=30, allow_redirects=False)
        print(url, r.status_code, r.headers.get("content-type"), r.headers.get("location"), repr(r.text[:300]))
    except Exception as e: print(url, repr(e))
r = requests.get("https://models.github.ai/catalog/models", headers=H, timeout=30)
print("catálogo", r.status_code, r.text[:300])
