from PIL import Image, ImageDraw
import io, interpreta, chat
img = Image.new("RGB", (720, 420), "white"); d = ImageDraw.Draw(img)
y = 20
for line in ["Unibet - Apuesta combinada", "", "AEK Athens - Real Madrid", "  Real Madrid gana   1.40",
             "Slavia Prague - Arsenal", "  Mas de 4.5 tarjetas   2.75", "FC Porto - Napoli", "  Ambos marcan: Si   1.80",
             "", "Cuota total: 6.93    Importe: 1,00 EUR"]:
    d.text((20, y), line, fill="black"); y += 36
b = io.BytesIO(); img.resize((1440, 840)).save(b, "PNG")
s = interpreta.read_slip(b.getvalue(), "image/png"); print("LEÍDO:", s)
if s: print(chat.analyze_slip(s))
