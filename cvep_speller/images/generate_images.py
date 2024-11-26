from PIL import Image, ImageDraw, ImageFont

WIDTH = 150
HEIGHT = 150

TEXT_COLOR = (128, 128, 128)
FONT_SIZE = 30

KEYS = [
	"!", "@", "#", "$", "%", "^", "&", "asterisk", "(", ")", "_", "+",  # 12
	"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=",  # 12
	"Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]",  # 12
	"A", "S", "D", "F", "G", "H", "J", "K", "L", "colon", "quote", "bar",  # 12
	"tilde", "Z", "X", "C", "V", "B", "N", "M", "comma", ".", "question", "slash",  # 12
	"smaller", "space", "larger"]  # 3
KEY_COLORS = ["black", "white", "green", "blue"]
KEY_MAPPING = {  # Windows does not allow / , : * ? " < > | ~ in file names
	"slash": "/",
	"comma": ",",
	"colon": ":",
	"asterisk": "*",
	"question": "?",
	"quote": '"',
	"smaller": "<",
	"larger": ">",
	"bar": "|",
	"tilde": "~",
}

# Create images with symbols
for key in KEYS:
	if key == "space":
		for color in KEY_COLORS:
			img = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=color)
			img_draw = ImageDraw.Draw(img)
			img.save(f"{color}.png")
	else:
		if key in KEY_MAPPING:
			symbol = KEY_MAPPING[key]
		else:
			symbol = key
		for color in KEY_COLORS:
			img = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=color)
			img_draw = ImageDraw.Draw(img)
			_, _, text_width, text_height = img_draw.textbbox(xy=(0, 0), text=symbol, font_size=FONT_SIZE)
			x_pos = (WIDTH - text_width) / 2
			y_pos = (HEIGHT - text_height) / 2
			img_draw.text(xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE)
			img.save(f"{key}_{color}.png")
