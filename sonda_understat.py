import requests, re
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "X-Requested-With": "XMLHttpRequest"}
r = requests.get("https://understat.com/league/EPL/2025", headers=UA, timeout=30)
print("página", r.status_code, len(r.text), re.findall(r"(?:var \w+|JSON\.parse|getLeagueData|fetch\([^)]*\)|/[a-zA-Z]+Data/[^'\" ]*)", r.text)[:20])
print([s for s in re.findall(r'<script[^>]*src="([^"]+)"', r.text)][:10])
for u in ("https://understat.com/getLeagueData/EPL/2025", "https://understat.com/main/getLeagueData/EPL/2025"):
    x = requests.get(u, headers={**UA, "Referer": "https://understat.com/league/EPL/2025"}, timeout=30)
    print(u, x.status_code, x.headers.get("content-type"), x.text[:200].replace("\n", " "))
