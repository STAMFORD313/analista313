import interpreta, chat
data = open("tmp_sonda/slip.jpg", "rb").read()
s = interpreta.read_slip(data, "image/jpeg"); print("LEÍDO:", s)
if s: print(chat.analyze_slip(s))
