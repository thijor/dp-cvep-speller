from PIL import Image, ImageDraw
import numpy as np

WIDTH = 150
HEIGHT = 150

TEXT_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (127, 127, 127)


def generate_gabor_patch(size=(40, 40), theta=np.pi / 2, gamma=0.6, lamda=2.5, phi=0.0, sigma=1.5):
    x, y = np.meshgrid(
        np.linspace(-size[1] // 2, size[1] // 2, size[1]),
        np.linspace(-size[0] // 2, size[0] // 2, size[0]))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lamda + phi)

    return gabor

# Add gabor patches
image = np.zeros(shape=(WIDTH, HEIGHT), dtype="float32")
patch_height = 30
patch_width = 30
n_patches = 75
for i in range(n_patches):
    theta = np.random.rand() * np.pi
    patch = generate_gabor_patch(size=(patch_height, patch_width), theta=theta)
    x_pos = int(np.random.rand() * (WIDTH - patch_width))
    y_pos = int(np.random.rand() * (HEIGHT - patch_height))
    image[y_pos:y_pos+patch_height, x_pos:x_pos+patch_width] += patch
image *= 127
image += 127
image = np.clip(image, 0, 255)
img = Image.fromarray(np.repeat(image[:, :, np.newaxis], repeats=3, axis=2).astype("uint8"))
img_draw = ImageDraw.Draw(img)

# Add text
_, _, text_width, text_height = img_draw.textbbox(xy=(0, 0), text="A", font_size=30)
x_pos = (WIDTH - text_width) / 2
y_pos = (HEIGHT - text_height) / 2
img_draw.text(xy=(x_pos, y_pos), fill=TEXT_COLOR, text="A", font_size=30)

# Save
img.save(f"A_grating.png")
