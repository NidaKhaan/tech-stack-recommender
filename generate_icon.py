"""
generate_icon.py
One-time script to generate a PNG favicon from the StackMatch logo shape.
Run once: python generate_icon.py
"""

from PIL import Image, ImageDraw
import os

size = 256
scale = size / 40

img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

outer = [(20 * scale, 2 * scale), (35 * scale, 11 * scale), (35 * scale, 29 * scale),
         (20 * scale, 38 * scale), (5 * scale, 29 * scale), (5 * scale, 11 * scale)]
inner = [(20 * scale, 9 * scale), (29 * scale, 14.5 * scale), (29 * scale, 25.5 * scale),
         (20 * scale, 31 * scale), (11 * scale, 25.5 * scale), (11 * scale, 14.5 * scale)]

draw.polygon(outer, fill="#14161A")
draw.polygon(inner, fill="#0E7C7B")
draw.ellipse(
    [(20 * scale - 4 * scale, 20 * scale - 4 * scale), (20 * scale + 4 * scale, 20 * scale + 4 * scale)],
    fill="#FAFAF8",
)

os.makedirs("assets", exist_ok=True)
img.save("assets/icon.png")
print("Saved assets/icon.png")