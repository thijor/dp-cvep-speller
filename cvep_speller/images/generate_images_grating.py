import numpy as np
from PIL import Image, ImageDraw

np.random.seed(2)

WIDTH = 150
HEIGHT = 150

PATCH_HEIGHT = 20
PATCH_WIDTH = 20
N_PATCHES = 75

TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 30
GRAY_COLOR = (127, 127, 127)

KEYS = [
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "asterisk",
    "(",
    ")",
    "_",
    "+",
    "backspace",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "-",
    "=",
    "Q",
    "W",
    "E",
    "R",
    "T",
    "Y",
    "U",
    "I",
    "O",
    "P",
    "[",
    "]",
    "{",
    "}",
    "shift",
    "A",
    "S",
    "D",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "colon",
    "quote",
    "bar",
    ";",
    "'",
    "backslash",
    "tilde",
    "`",
    "Z",
    "X",
    "C",
    "V",
    "B",
    "N",
    "M",
    "comma",
    ".",
    "slash",
    "smaller",
    "larger",
    "question",
    "clear",
    "space",
    "autocomplete",
    "speaker",
]

# Windows does not allow / , : * ? " < > | ~ in file names
KEY_MAPPING = {
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
    "backslash": "\\",
    "backspace": "<-",
    "clear": "<<",
    "autocomplete": ">>",
    "shift": "sh",
    "speaker": "sp",
}


def generate_gabor_patch(
    size=(60, 60), theta=np.pi / 2, gamma=0.6, lamda=5, phi=0.0, sigma=2
):
    x, y = np.meshgrid(
        np.linspace(-size[1] // 2, size[1] // 2, size[1]),
        np.linspace(-size[0] // 2, size[0] // 2, size[0]),
    )

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(
        2 * np.pi * x_theta / lamda + phi
    )

    return gabor


grating_image = np.zeros(shape=(WIDTH, HEIGHT), dtype="float32")
for i in range(N_PATCHES):
    patch = generate_gabor_patch(
        size=(PATCH_HEIGHT, PATCH_WIDTH), theta=np.random.rand() * np.pi
    )
    while True:
        x_pos = int(np.random.rand() * (WIDTH - PATCH_WIDTH))
        y_pos = int(np.random.rand() * (HEIGHT - PATCH_HEIGHT))
        if (
            np.sqrt(
                (x_pos + PATCH_WIDTH // 2 - WIDTH // 2) ** 2
                + (y_pos + PATCH_HEIGHT // 2 - HEIGHT // 2) ** 2
            )
            > FONT_SIZE // 2
        ):
            break
    grating_image[y_pos : y_pos + PATCH_HEIGHT, x_pos : x_pos + PATCH_WIDTH] += patch
grating_image *= 127
grating_image += 127
grating_image = np.clip(grating_image, a_min=0, a_max=255)

for key in KEYS:
    if key == "space":
        # No symbol gray
        img = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=GRAY_COLOR)
        img_draw = ImageDraw.Draw(img)
        img.save("gray.png")

        # No symbol grating
        img = Image.fromarray(
            np.repeat(grating_image[:, :, np.newaxis], repeats=3, axis=2).astype(
                "uint8"
            )
        )
        img_draw = ImageDraw.Draw(img)
        img.save("grating.png")

    else:
        if key in KEY_MAPPING:
            symbol = KEY_MAPPING[key]
        else:
            symbol = key

        # Symbol gray uppercase
        img = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=GRAY_COLOR)
        img_draw = ImageDraw.Draw(img)
        _, _, text_width, text_height = img_draw.textbbox(
            xy=(0, 0), text=symbol, font_size=FONT_SIZE
        )
        x_pos = (WIDTH - text_width) / 2
        y_pos = (HEIGHT - text_height) / 2
        img_draw.text(
            xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
        )
        img.save(f"{key}_gray.png")

        # Symbol grating uppercase
        img = Image.fromarray(
            np.repeat(grating_image[:, :, np.newaxis], repeats=3, axis=2).astype(
                "uint8"
            )
        )
        img_draw = ImageDraw.Draw(img)
        img_draw.text(
            xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
        )
        img.save(f"{key}_grating.png")

        if key.isalpha() and len(key) == 1:
            # Symbol gray uppercase
            img = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=GRAY_COLOR)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol.lower(), font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text(
                xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
            )
            img.save(f"{key}_lower_gray.png")

            # Symbol grating uppercase
            img = Image.fromarray(
                np.repeat(grating_image[:, :, np.newaxis], repeats=3, axis=2).astype(
                    "uint8"
                )
            )
            img_draw = ImageDraw.Draw(img)
            img_draw.text(
                xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
            )
            img.save(f"{key}_lower_grating.png")
