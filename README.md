# Analista deportivo (Telegram)

Cada hora, gratis en GitHub Actions (fútbol, NBA, NFL, NHL, MLB, tenis ATP/WTA y esports con `PANDASCORE_TOKEN`).
Por defecto solo manda por Telegram las apuestas de la banca virtual de 100 $ y sus resultados; con `VERBOSE=1` manda también análisis, grafos y PDF:
- Historial de 5 grandes ligas desde 2005 (~38.000 partidos, football-data.co.uk) con cuotas.
- Modelo Poisson (vida media 1 año) mezclado con el mercado → probabilidades y **señales de valor**.
- IA Gemini con búsqueda en Google: bajas, dudas, sancionados, alineación probable y análisis experto.
- Grafo por equipo con cada jugador coloreado según su estado + informe PDF con cara a cara completo.
- Resultados nuevos y aciertos/ROI de las señales enviadas.

## Puesta en marcha
1. Telegram: habla con @BotFather → `/newbot` → copia el token. Escribe algo a tu bot y abre
   `https://api.telegram.org/bot<TOKEN>/getUpdates` para ver tu `chat.id`.
2. Gemini: crea una clave gratis en https://aistudio.google.com/apikey
3. GitHub: crea un repo **público** (minutos ilimitados), sube esta carpeta y en
   Settings → Secrets and variables → Actions añade `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY`.
4. Actions → analista → Run workflow (la primera vez descarga el historial).

Prueba local sin tokens: `python analyst.py` (imprime en consola en vez de enviar).
