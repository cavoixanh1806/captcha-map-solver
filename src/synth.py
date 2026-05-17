"""Synthetic CAPTCHA generator that mimics the map_*.png style.

The real dataset has these visual properties (see research.md section 1.4):
- 128x128 RGBA on pastel ground (greyish/peach/lavender/sage)
- Map-like background texture: light irregular patches + thin curved lines
- 5 uppercase glyphs centred horizontally, each in its OWN saturated colour
  (red, navy, green, purple, yellow, brown, teal, ...)
- Glyphs are slightly rotated, scaled differently, and overlap at edges
- Yellow/orange speckle noise plus a few thin curved lines drawn over the
  whole image
- A bold sans-serif font with chunky strokes (looks hand-painted)

We reproduce all of that with PIL + numpy so we can sample unlimited
training data with known labels. The goal is NOT photo-realism: it's to
create images that share the *low-level statistics* the model needs in
order to learn glyph shapes that generalise to the real test split.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ALPHABET = "3479ACDEFHJKLMNPQRTUVWXY"
IMG_SIZE = 128


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
GLYPH_COLOURS = [
    (180, 30, 40),    # red
    (40, 60, 150),    # navy
    (30, 110, 60),    # forest green
    (110, 30, 130),   # purple
    (200, 165, 30),   # mustard yellow
    (130, 80, 30),    # brown
    (30, 130, 130),   # teal
    (200, 80, 80),    # coral
    (90, 30, 30),     # dark red
    (30, 30, 30),     # almost black
    (90, 130, 30),    # olive
    (200, 100, 30),   # orange
]

# Pastel base colours seen in the real corpus
BG_COLOURS = [
    (200, 200, 200),  # neutral grey
    (220, 200, 195),  # peach
    (185, 200, 180),  # sage green
    (190, 195, 215),  # lavender
    (215, 200, 200),  # warm pink
]


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------
DEFAULT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\trebucbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def discover_fonts(candidates: List[str] | None = None) -> List[str]:
    candidates = candidates or DEFAULT_FONT_CANDIDATES
    return [p for p in candidates if os.path.exists(p)]


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------
def _pastel_background(rng: random.Random) -> Image.Image:
    base = rng.choice(BG_COLOURS)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    arr = np.array(img, dtype=np.int16)

    # Add map-like splotches: a few softly coloured irregular regions
    blob_layer = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    blob_draw = ImageDraw.Draw(blob_layer)
    for _ in range(rng.randint(4, 8)):
        cx = rng.randint(0, IMG_SIZE)
        cy = rng.randint(0, IMG_SIZE)
        r = rng.randint(15, 45)
        delta = tuple(rng.randint(-25, 25) for _ in range(3))
        col = tuple(max(0, min(255, base[i] + delta[i])) for i in range(3))
        blob_draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    blob_layer = blob_layer.filter(ImageFilter.GaussianBlur(radius=6))
    arr = np.array(blob_layer, dtype=np.int16)

    # Light per-pixel noise for the speckled feel
    noise = np.random.randint(-6, 6, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _draw_curves(img: Image.Image, rng: random.Random) -> None:
    """Thin curved lines like the streets on the real backgrounds."""
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 5)):
        # Pick three control points for a quadratic curve
        x0 = rng.randint(-10, IMG_SIZE)
        y0 = rng.randint(-10, IMG_SIZE)
        x1 = rng.randint(0, IMG_SIZE)
        y1 = rng.randint(0, IMG_SIZE)
        x2 = rng.randint(0, IMG_SIZE + 10)
        y2 = rng.randint(0, IMG_SIZE + 10)
        steps = 30
        prev = None
        col = (
            rng.randint(180, 230),
            rng.randint(180, 230),
            rng.randint(180, 230),
        )
        for s in range(steps + 1):
            t = s / steps
            # Quadratic Bezier
            xt = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t * t * x2
            yt = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t * t * y2
            cur = (xt, yt)
            if prev is not None:
                draw.line([prev, cur], fill=col, width=rng.choice([1, 1, 2]))
            prev = cur


def _yellow_speckle(img: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(40, 90)):
        x = rng.randint(0, IMG_SIZE - 1)
        y = rng.randint(0, IMG_SIZE - 1)
        col = (
            rng.randint(200, 255),
            rng.randint(180, 230),
            rng.randint(20, 80),
        )
        draw.point((x, y), fill=col)


# ---------------------------------------------------------------------------
# Glyph drawing
# ---------------------------------------------------------------------------
def _draw_glyph(
    canvas: Image.Image,
    char: str,
    cx: int,
    cy: int,
    font_paths: List[str],
    rng: random.Random,
) -> None:
    """Render a single rotated/scaled glyph onto the canvas at (cx, cy)."""
    font_path = rng.choice(font_paths)
    size = rng.randint(38, 56)
    font = ImageFont.truetype(font_path, size)

    # Render the char on a transparent square that's larger than needed
    pad = 12
    bbox = font.getbbox(char)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    glyph = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glyph)
    colour = rng.choice(GLYPH_COLOURS) + (255,)
    gd.text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=colour)

    # Random rotation and tiny shear (mimic the wobble in the real captchas)
    angle = rng.uniform(-15, 15)
    glyph = glyph.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Random horizontal scale to vary character width
    sx = rng.uniform(0.85, 1.15)
    sy = rng.uniform(0.85, 1.15)
    glyph = glyph.resize((max(1, int(glyph.width * sx)), max(1, int(glyph.height * sy))), Image.BICUBIC)

    # Paste centred on (cx, cy) using the alpha channel as mask
    top_left = (cx - glyph.width // 2, cy - glyph.height // 2)
    canvas.paste(glyph, top_left, glyph)


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------
@dataclass
class SynthSample:
    image: Image.Image
    text: str


def render(
    text: str | None,
    font_paths: List[str],
    rng: random.Random | None = None,
) -> SynthSample:
    rng = rng or random.Random()
    if text is None:
        text = "".join(rng.choice(ALPHABET) for _ in range(5))

    img = _pastel_background(rng)
    _draw_curves(img, rng)

    # Place 5 glyphs roughly evenly across the centre band, with jitter
    n = len(text)
    band_y = rng.randint(IMG_SIZE // 2 - 5, IMG_SIZE // 2 + 5)
    spacing = rng.randint(18, 22)
    total_w = spacing * (n - 1)
    start_x = IMG_SIZE // 2 - total_w // 2
    for i, ch in enumerate(text):
        cx = start_x + i * spacing + rng.randint(-3, 3)
        cy = band_y + rng.randint(-6, 6)
        _draw_glyph(img, ch, cx, cy, font_paths, rng)

    _yellow_speckle(img, rng)

    # Slight overall blur to mimic the upscaled / aliased look
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return SynthSample(image=img, text=text)


# ---------------------------------------------------------------------------
# Convenience CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fonts = discover_fonts()
    if not fonts:
        raise SystemExit("No usable TrueType fonts found.")
    out_dir = "synth_preview"
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    for i in range(20):
        s = render(None, fonts, rng)
        s.image.save(os.path.join(out_dir, f"synth_{i:02d}_{s.text}.png"))
    print(f"Wrote 20 preview images to {out_dir}/")
