"""Synthetic CAPTCHA generator matching the real ``map_*.png`` distribution.

Designed from the 35-item checklist in ``research.md`` sections 8-12, derived
from direct pixel-vs-label analysis of 83 real images (batches 1-4 + spot
check). The implementation targets the empirical distribution rather than
photo realism. Highlights:

* 8 background presets (BG-A..BG-H) sampled with the proportions observed in
  the real corpus (lavender 22%, peach 22%, sage 18%, water/land split 18%,
  khaki/olive 5%, yellow pastel 5%, peach+cyan tile 4%, pink+sage 3%).
* Negative-offset rendering: ~80% of samples are forced into "Cao+" overlap.
* Two missing manifolds covered: ``9`` rendered as lowercase ``g`` (~30% of
  samples containing 9) and ``Q`` rendered as lowercase ``q`` (~15%).
* Glyph outline at ~25% (cyan 0.50, teal 0.30, black 0.15, white 0.05) with
  the black share matching the ~8-10% real frequency.
* Per-glyph scale/height/stroke disparity, optional baseline wave, optional
  whole-string shift and inter-glyph gaps.
* Same-family hue clustering for ~5% of samples (no colour shortcut).
* Multi-style noise: single arcs, X-grids, dense web of lines, plus dot
  speckles. Line thickness sampled from {1, 1, 2, 3} so dense backgrounds
  with thick strokes are reachable.
* Serif font variant available for letters that occasionally show feet/hooks
  in the real data (N, M, R, T, F, E, H).

Public API kept stable for ``src/dataset.py``::

    ALPHABET, IMG_SIZE, discover_fonts(), render(text, font_paths, rng)

Returns ``SynthSample`` with ``image`` (PIL RGB) and ``text`` (str).
"""
from __future__ import annotations

import colorsys
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ALPHABET = "3479ACDEFHJKLMNPQRTUVWXY"
IMG_SIZE = 128
CHAR_LEN = 5

# Background preset weights — calibrated from 83-image visual survey
# (research.md section 12.4).
BG_PRESETS: List[Tuple[str, float]] = [
    ("lavender", 0.22),       # BG-A
    ("peach", 0.22),          # BG-B
    ("sage", 0.18),           # BG-C
    ("water_land", 0.18),     # BG-D
    ("khaki_flat", 0.05),     # BG-E
    ("yellow_pastel", 0.05),  # BG-F
    ("peach_cyan", 0.05),     # BG-G
    ("pink_sage", 0.05),      # BG-H
]

# ---------------------------------------------------------------------------
# Font discovery — three categories with empirical weights derived from a
# visual font analysis of the real corpus (research notes, batch 4 review):
#
#   sans     — bold uniform stroke (Verdana Bold, Arial Bold, ...)   ~55%
#   serif    — slab serif / heavy serif (Rockwell, Cooper, ...)      ~30%
#   display  — extra-heavy display fonts (Impact, Arial Black)       ~15%
#
# We also detected ~85% of real images render all 5 glyphs in the same font
# but the remaining ~15% mix fonts per glyph. That mixed-font mode is
# implemented in :func:`_plan_string_fonts`.
# ---------------------------------------------------------------------------
FONT_CATEGORIES = {
    "sans": {
        "weight": 0.55,
        "files": [
            r"C:\Windows\Fonts\verdanab.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\bahnschrift.ttf",
            r"C:\Windows\Fonts\trebucbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\tahomabd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ],
    },
    "serif": {
        "weight": 0.30,
        "files": [
            r"C:\Windows\Fonts\rockb.ttf",       # Rockwell Bold (slab)
            r"C:\Windows\Fonts\coopbl.ttf",      # Cooper Black (display slab)
            r"C:\Windows\Fonts\bookosb.ttf",     # Bookman Bold
            r"C:\Windows\Fonts\georgiab.ttf",
            r"C:\Windows\Fonts\cambriab.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        ],
    },
    "display": {
        "weight": 0.15,
        "files": [
            r"C:\Windows\Fonts\impact.ttf",
            r"C:\Windows\Fonts\ariblk.ttf",      # Arial Black
        ],
    },
}

# 15% of images mix fonts across the 5 glyphs (research observation)
MIXED_FONT_PROB = 0.15


def discover_fonts(
    custom: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return the flat list of installed fonts across all categories.

    A flat list keeps the public API stable for ``src/dataset.py``. Custom
    paths supplied via ``custom`` are placed first; everything is filtered
    against the filesystem so the caller never sees a missing file.
    """
    fonts: List[str] = []
    if custom:
        fonts.extend(p for p in custom if os.path.exists(p))
    for info in FONT_CATEGORIES.values():
        fonts.extend(p for p in info["files"] if os.path.exists(p))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in fonts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _categorize_available(font_paths: Sequence[str]) -> dict:
    """Group ``font_paths`` by FONT_CATEGORIES membership."""
    available = set(font_paths)
    return {
        cat: [p for p in info["files"] if p in available]
        for cat, info in FONT_CATEGORIES.items()
    }


def _pick_string_font(font_paths: Sequence[str], rng: random.Random) -> str:
    """Sample a font path with the empirical category weights."""
    cats = _categorize_available(font_paths)
    weighted_cats = [(name, FONT_CATEGORIES[name]["weight"])
                     for name in cats if cats[name]]
    if not weighted_cats:
        # Nothing matched — fall back to whatever is available
        return rng.choice(list(font_paths))
    names, weights = zip(*weighted_cats)
    chosen_cat = rng.choices(names, weights=weights, k=1)[0]
    return rng.choice(cats[chosen_cat])


def _plan_string_fonts(
    text: str,
    font_paths: Sequence[str],
    rng: random.Random,
) -> List[str]:
    """Plan which font each glyph uses.

    * Default (~85%): pick one font and repeat it for all glyphs.
    * ``MIXED_FONT_PROB`` chance: pick a fresh font per glyph. This mirrors
      the small minority of real images that show slight font drift across
      glyph positions.
    """
    n = len(text)
    primary = _pick_string_font(font_paths, rng)
    if rng.random() < MIXED_FONT_PROB:
        return [_pick_string_font(font_paths, rng) for _ in range(n)]
    return [primary] * n


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


def _rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _sample_glyph_colour(
    rng: random.Random,
    base_hue: Optional[float] = None,
    hue_window_deg: float = 360.0,
    saturation_range: Tuple[float, float] = (0.55, 0.95),
    value_range: Tuple[float, float] = (0.35, 0.80),
) -> Tuple[int, int, int]:
    """Sample a saturated glyph colour, optionally constrained near a base hue."""
    if base_hue is None:
        h = rng.random()
    else:
        delta = (rng.random() - 0.5) * (hue_window_deg / 360.0)
        h = (base_hue + delta) % 1.0
    s = rng.uniform(*saturation_range)
    v = rng.uniform(*value_range)
    return _hsv_to_rgb(h, s, v)


def _hue_distance(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    """Smallest absolute hue difference in [0, 0.5]."""
    ha, _, _ = _rgb_to_hsv(rgb_a)
    hb, _, _ = _rgb_to_hsv(rgb_b)
    d = abs(ha - hb)
    return min(d, 1.0 - d)


# ---------------------------------------------------------------------------
# Background presets — 8 functions, each returning a 128x128 RGB image.
# ---------------------------------------------------------------------------
def _add_speckles(arr: np.ndarray, rng: random.Random, density: float = 0.005,
                  hue_choices: Optional[Sequence[Tuple[int, int, int]]] = None) -> None:
    """Sprinkle coloured 1-px dots in-place to mimic the speckle texture."""
    n = int(IMG_SIZE * IMG_SIZE * density)
    if n <= 0:
        return
    ys = np.random.randint(0, IMG_SIZE, n)
    xs = np.random.randint(0, IMG_SIZE, n)
    if hue_choices:
        for y, x in zip(ys, xs):
            arr[y, x] = rng.choice(hue_choices)
    else:
        # Default: warm yellow/cream speckle (matches BG-A)
        for y, x in zip(ys, xs):
            arr[y, x] = (
                rng.randint(200, 255),
                rng.randint(180, 230),
                rng.randint(40, 110),
            )


def _bg_lavender(rng: random.Random) -> Image.Image:
    base = (190, 195, 215)
    img = _solid_bg_with_blobs(base, rng, blob_count=(4, 8), blob_delta=(-25, 25))
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.003, 0.008),
                  hue_choices=[(220, 195, 80), (215, 165, 95)])
    return Image.fromarray(arr, "RGB")


def _bg_peach(rng: random.Random) -> Image.Image:
    base = (220, 200, 195)
    img = _solid_bg_with_blobs(base, rng, blob_count=(2, 6), blob_delta=(-18, 18))
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.002, 0.006),
                  hue_choices=[(160, 165, 200), (200, 200, 200)])
    return Image.fromarray(arr, "RGB")


def _bg_sage(rng: random.Random) -> Image.Image:
    base = (185, 200, 180)
    img = _solid_bg_with_blobs(base, rng, blob_count=(3, 7), blob_delta=(-20, 20))
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.004, 0.010),
                  hue_choices=[(220, 130, 150), (215, 170, 140)])
    return Image.fromarray(arr, "RGB")


def _bg_water_land(rng: random.Random) -> Image.Image:
    """BG-D: peach + blue/lavender split with water/land tile feel."""
    peach = (220, 200, 190)
    blue = rng.choice([(160, 175, 210), (170, 185, 200), (180, 195, 220)])
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), peach)
    draw = ImageDraw.Draw(img, "RGBA")
    # Diagonal split — pick a random angle, fill one side with the other tone
    angle = rng.uniform(20, 70)  # degrees
    if rng.random() < 0.5:
        angle = 180 - angle
    # Convert to a polygon that covers half the image roughly
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    span = IMG_SIZE * 1.5
    poly = [
        (cx + dx * span - dy * span, cy + dy * span + dx * span),
        (cx + dx * span + dy * span, cy + dy * span - dx * span),
        (cx - dx * span + dy * span, cy - dy * span - dx * span),
        (cx - dx * span - dy * span, cy - dy * span + dx * span),
    ]
    # Mask the right half
    half_poly = [poly[1], poly[2], (cx + dx * span, cy + dy * span)]
    draw.polygon(poly[:3] + [poly[3]], fill=(*blue, 255))
    # Sprinkle small "tiles" (rectangles) on both regions
    for _ in range(rng.randint(8, 18)):
        rx, ry = rng.randint(0, IMG_SIZE - 8), rng.randint(0, IMG_SIZE - 8)
        rw, rh = rng.randint(4, 12), rng.randint(3, 8)
        tile_col = rng.choice([peach, blue, (200, 195, 200)])
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=tile_col)
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.004, 0.010),
                  hue_choices=[(160, 175, 210), (220, 200, 190), (210, 200, 230)])
    return Image.fromarray(arr, "RGB")


def _bg_khaki_flat(rng: random.Random) -> Image.Image:
    """BG-E: solid olive/khaki, very few features."""
    base = rng.choice([(195, 180, 90), (185, 175, 100), (175, 165, 80)])
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    arr = np.array(img, dtype=np.int16)
    arr += np.random.randint(-6, 6, arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_yellow_pastel(rng: random.Random) -> Image.Image:
    """BG-F: clean yellow pastel, almost no lines."""
    base = rng.choice([(235, 225, 170), (225, 215, 165), (240, 230, 180)])
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    arr = np.array(img, dtype=np.int16)
    arr += np.random.randint(-5, 5, arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _bg_peach_cyan(rng: random.Random) -> Image.Image:
    """BG-G: peach with cyan tile patches and dotted cyan noise."""
    base = (225, 200, 190)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    draw = ImageDraw.Draw(img)
    cyan = (140, 215, 215)
    for _ in range(rng.randint(6, 12)):
        rx, ry = rng.randint(0, IMG_SIZE - 10), rng.randint(0, IMG_SIZE - 10)
        rw, rh = rng.randint(6, 18), rng.randint(4, 12)
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=cyan)
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.008, 0.020),
                  hue_choices=[cyan, (155, 220, 220)])
    return Image.fromarray(arr, "RGB")


def _bg_pink_sage(rng: random.Random) -> Image.Image:
    """BG-H: pink/magenta + sage split, very colourful."""
    pink = rng.choice([(220, 130, 160), (210, 120, 145), (230, 140, 170)])
    sage = rng.choice([(165, 195, 175), (170, 200, 175)])
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), sage)
    draw = ImageDraw.Draw(img)
    # Large irregular pink region
    for _ in range(rng.randint(2, 4)):
        cx = rng.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        cy = rng.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        r = rng.randint(35, 65)
        draw.ellipse((cx - r, cy - r * 1.2, cx + r * 1.4, cy + r), fill=pink)
    # Add a few dense lines
    for _ in range(rng.randint(4, 8)):
        x0, y0 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        x1, y1 = rng.randint(0, IMG_SIZE), rng.randint(0, IMG_SIZE)
        draw.line([(x0, y0), (x1, y1)], fill=(220, 100, 140), width=rng.choice([1, 1, 2]))
    arr = np.array(img, dtype=np.uint8)
    _add_speckles(arr, rng, density=rng.uniform(0.004, 0.010),
                  hue_choices=[(220, 100, 140), (170, 200, 175)])
    return Image.fromarray(arr, "RGB")


def _solid_bg_with_blobs(
    base: Tuple[int, int, int],
    rng: random.Random,
    blob_count: Tuple[int, int],
    blob_delta: Tuple[int, int],
) -> Image.Image:
    blob_layer = Image.new("RGB", (IMG_SIZE, IMG_SIZE), base)
    bd = ImageDraw.Draw(blob_layer)
    for _ in range(rng.randint(*blob_count)):
        cx = rng.randint(0, IMG_SIZE)
        cy = rng.randint(0, IMG_SIZE)
        r = rng.randint(15, 45)
        delta = tuple(rng.randint(*blob_delta) for _ in range(3))
        col = tuple(max(0, min(255, base[i] + delta[i])) for i in range(3))
        bd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    blob_layer = blob_layer.filter(ImageFilter.GaussianBlur(radius=6))
    return blob_layer


_BG_DISPATCH = {
    "lavender": _bg_lavender,
    "peach": _bg_peach,
    "sage": _bg_sage,
    "water_land": _bg_water_land,
    "khaki_flat": _bg_khaki_flat,
    "yellow_pastel": _bg_yellow_pastel,
    "peach_cyan": _bg_peach_cyan,
    "pink_sage": _bg_pink_sage,
}


def _sample_background(rng: random.Random) -> Tuple[Image.Image, str]:
    """Pick a preset by weight, return ``(image, preset_name)``."""
    r = rng.random()
    cum = 0.0
    for name, w in BG_PRESETS:
        cum += w
        if r <= cum:
            return _BG_DISPATCH[name](rng), name
    return _BG_DISPATCH[BG_PRESETS[-1][0]](rng), BG_PRESETS[-1][0]


# ---------------------------------------------------------------------------
# Line / curve noise — multiple styles to match the real diversity.
# ---------------------------------------------------------------------------
def _draw_quadratic_curve(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    colour: Tuple[int, int, int],
    width: int,
    margin: int = 10,
) -> None:
    x0 = rng.randint(-margin, IMG_SIZE + margin)
    y0 = rng.randint(-margin, IMG_SIZE + margin)
    x1 = rng.randint(0, IMG_SIZE)
    y1 = rng.randint(0, IMG_SIZE)
    x2 = rng.randint(-margin, IMG_SIZE + margin)
    y2 = rng.randint(-margin, IMG_SIZE + margin)
    steps = 32
    prev = None
    for s in range(steps + 1):
        t = s / steps
        xt = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t * t * x2
        yt = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t * t * y2
        cur = (xt, yt)
        if prev is not None:
            draw.line([prev, cur], fill=colour, width=width)
        prev = cur


def _draw_lines(
    img: Image.Image,
    rng: random.Random,
    style: str,
    glyph_hues: Sequence[float],
) -> None:
    """Draw line noise with a given style.

    Styles:
        single  — 1-2 thin curves (default sparse).
        x_grid  — 2-3 long crossing lines.
        dense   — 4-7 mixed curves and lines (heavy mat).
        cluster — 1-2 thick arcs near the text band.
    """
    draw = ImageDraw.Draw(img)
    n = {
        "single": rng.randint(1, 2),
        "x_grid": rng.randint(2, 3),
        "dense": rng.randint(4, 7),
        "cluster": rng.randint(1, 2),
    }[style]

    # Optionally make ONE line share a glyph's hue (research mục 24 / line=glyph trick)
    glyph_hue_match_idx = -1
    if glyph_hues and rng.random() < 0.05:
        glyph_hue_match_idx = rng.randrange(n)

    for i in range(n):
        if glyph_hue_match_idx == i:
            h = rng.choice(list(glyph_hues))
            colour = _hsv_to_rgb(h, rng.uniform(0.5, 0.85), rng.uniform(0.45, 0.75))
        else:
            colour = (
                rng.randint(150, 230),
                rng.randint(150, 230),
                rng.randint(150, 230),
            )

        # Thickness distribution — ~10% chance of a fat (3px) line per spec mục 17/23.
        width = rng.choices([1, 1, 2, 3], weights=[0.55, 0.25, 0.10, 0.10])[0]

        if style == "x_grid":
            x0 = rng.randint(-20, IMG_SIZE + 20)
            y0 = rng.randint(-20, IMG_SIZE + 20)
            x1 = IMG_SIZE - x0 + rng.randint(-30, 30)
            y1 = IMG_SIZE - y0 + rng.randint(-30, 30)
            draw.line([(x0, y0), (x1, y1)], fill=colour, width=width)
        elif style == "dense":
            if rng.random() < 0.5:
                _draw_quadratic_curve(draw, rng, colour, width)
            else:
                x0 = rng.randint(-20, IMG_SIZE + 20)
                y0 = rng.randint(-20, IMG_SIZE + 20)
                x1 = rng.randint(-20, IMG_SIZE + 20)
                y1 = rng.randint(-20, IMG_SIZE + 20)
                draw.line([(x0, y0), (x1, y1)], fill=colour, width=width)
        else:  # single, cluster
            _draw_quadratic_curve(draw, rng, colour, width)


def _draw_dots(img: Image.Image, rng: random.Random) -> None:
    """Tiny dot speckles overlaid on top of the (already speckled) bg."""
    arr = np.array(img, dtype=np.uint8)
    density = rng.uniform(0.003, 0.010)
    n = int(IMG_SIZE * IMG_SIZE * density)
    if n <= 0:
        img.paste(Image.fromarray(arr, "RGB"))
        return
    ys = np.random.randint(0, IMG_SIZE, n)
    xs = np.random.randint(0, IMG_SIZE, n)
    palette = [
        (220, 195, 80),
        (215, 165, 95),
        (220, 130, 150),
        (140, 215, 215),
        (160, 175, 210),
    ]
    for y, x in zip(ys, xs):
        arr[y, x] = rng.choice(palette)
    img.paste(Image.fromarray(arr, "RGB"))


# ---------------------------------------------------------------------------
# Glyph drawing
# ---------------------------------------------------------------------------
# Mục 1, 2: ALL glyphs are rendered UPPERCASE only. The real dataset does NOT
# contain any lowercase letters — the "two-storey 9" and "lowercase q" shapes
# observed in the research were actually just font variants of the uppercase
# characters (e.g. certain serif fonts render "9" with a curly tail, "Q" with
# a long descender). We must NOT substitute lowercase `g` or `q`.
#
# CORRECTION (section 13.7): earlier research sections (10, 11, 12) incorrectly
# identified these as "lowercase g/q variants". After direct comparison with
# synth output, it's clear the real data uses UPPERCASE ONLY with varied fonts.
G_VARIANT_PROB = 0.0  # DISABLED — no lowercase g
Q_VARIANT_PROB = 0.0  # DISABLED — no lowercase q


def _glyph_render_char(real_char: str, rng: random.Random) -> Tuple[str, bool]:
    """Always render the character as-is (uppercase). No lowercase substitution."""
    return real_char, False


# Removed _pick_font_for_string / _maybe_serif_font: the new
# _plan_string_fonts handles per-image font selection with category weights
# and the optional 15% mixed-font mode.


def _sample_outline(rng: random.Random) -> Optional[Tuple[Tuple[int, int, int], int]]:
    """25% chance to return ``(rgb, width_px)`` for an outline; else None.

    Colour palette weights match the real distribution: cyan dominant,
    teal next, then black ~10% (mục 5/35), white ~5%.
    """
    if rng.random() >= 0.25:
        return None
    colour = rng.choices(
        [
            (90, 215, 215),    # cyan
            (35, 130, 130),    # teal
            (10, 10, 10),      # black
            (245, 245, 245),   # white
        ],
        weights=[0.50, 0.30, 0.15, 0.05],
    )[0]
    width = rng.choice([1, 1, 2])
    return colour, width


def _draw_one_glyph(
    canvas: Image.Image,
    label_char: str,
    cx: int,
    cy: int,
    font_path: str,
    rng: random.Random,
    base_hue: Optional[float],
    near_hue_window_deg: float,
    forbid_colour_close_to: Optional[Tuple[int, int, int]] = None,
    base_size: int = 38,
    forced_colour: Optional[Tuple[int, int, int]] = None,
    use_strong_disparity: bool = False,
) -> Tuple[int, int, Tuple[int, int, int]]:
    """Render one glyph and return ``(width, height, colour)`` of the pasted shape.

    Real-data calibration (research.md sec 12.3):
    * Glyph height ~28-40px on a 128px canvas (~25-32%). Base ``base_size``
      should land in that band; per-glyph variation is small (±10%) for the
      majority of samples.
    * Only ~13% of samples show "Cao" letter-scale disparity — hence the
      ``use_strong_disparity`` flag set on the per-IMAGE level (not per
      glyph). When False, we keep variation small to match the typical
      uniform-look of real captchas.
    """
    visual_char, is_lowercase_variant = _glyph_render_char(label_char, rng)

    # Per-glyph size: narrow band by default, wider only when the planner
    # marked this image as "scale disparity" mode.
    if use_strong_disparity:
        size_scale = rng.uniform(0.78, 1.22)
    else:
        size_scale = rng.uniform(0.94, 1.06)
    size = max(20, int(base_size * size_scale))

    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        # Should never happen since the caller validated the path, but be safe.
        font = ImageFont.load_default()

    # Pick colour, possibly constrained
    for _ in range(8):
        if forced_colour is not None:
            colour = forced_colour
            break
        colour = _sample_glyph_colour(
            rng, base_hue=base_hue, hue_window_deg=near_hue_window_deg
        )
        if forbid_colour_close_to is None:
            break
        if _hue_distance(colour, forbid_colour_close_to) >= 60.0 / 360.0:
            break

    outline = _sample_outline(rng)

    # Render to a transparent square
    pad = 14
    bbox = font.getbbox(visual_char)
    w_glyph = (bbox[2] - bbox[0]) + pad * 2
    h_glyph = (bbox[3] - bbox[1]) + pad * 2
    glyph = Image.new("RGBA", (w_glyph, h_glyph), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glyph)
    if outline is not None:
        gd.text(
            (pad - bbox[0], pad - bbox[1]),
            visual_char,
            font=font,
            fill=colour + (255,),
            stroke_width=outline[1],
            stroke_fill=outline[0] + (255,),
        )
    else:
        gd.text(
            (pad - bbox[0], pad - bbox[1]),
            visual_char,
            font=font,
            fill=colour + (255,),
        )

    # Per-glyph rotation small (real glyphs are mostly upright; only ~10% have
    # noticeable wobble). Reduced from ±12 to ±7 for the typical case.
    angle = rng.uniform(-7, 7)
    glyph = glyph.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Subtle non-uniform scale for stroke/aspect variation. Was ±20%; that
    # over-warped the glyphs. Tighten to ±8% by default.
    if use_strong_disparity:
        sx = rng.uniform(0.85, 1.15)
        sy = rng.uniform(0.85, 1.15)
    else:
        sx = rng.uniform(0.94, 1.06)
        sy = rng.uniform(0.94, 1.06)
    new_w = max(1, int(glyph.width * sx))
    new_h = max(1, int(glyph.height * sy))
    glyph = glyph.resize((new_w, new_h), Image.BICUBIC)

    canvas.paste(
        glyph,
        (cx - glyph.width // 2, cy - glyph.height // 2),
        glyph,
    )
    return glyph.width, glyph.height, colour


# ---------------------------------------------------------------------------
# Layout planning
# ---------------------------------------------------------------------------
def _plan_centres(
    text: str,
    rng: random.Random,
) -> Tuple[List[Tuple[int, int]], int]:
    """Plan the (cx, cy) of each glyph plus the global string anchor.

    Layout decisions (mục 4/6/12/16/27 of the checklist):

    * Spacing 12-22 px so ~80% of samples have visible overlap.
    * 12% chance to shift the whole string ±25% horizontally.
    * 12% chance to apply a per-glyph baseline wave.
    * 5% chance to inject one extra gap between two glyphs.
    * 5% chance to tilt the whole string by ±10° (later, by rotating the
      pasting positions around the string centre).
    """
    n = len(text)

    # Spacing calibrated for base_size ~31 (glyph width ~18-24px).
    # Real data ALWAYS has overlap or near-touching — never spaced out like
    # individual letters. Remove the "easy" mode entirely; all samples get
    # tight spacing to match the real 80%+ overlap rate.
    spacing = rng.randint(12, 17)

    # Mục 31: optional bigger gap between two glyphs
    extra_gap_at = -1
    if n >= 3 and rng.random() < 0.05:
        extra_gap_at = rng.randint(1, n - 2)
        gap_extra = rng.randint(4, 12)
    else:
        gap_extra = 0

    total_w = spacing * (n - 1) + gap_extra
    centre_x = IMG_SIZE // 2

    # Mục 6/22: layout shift left/right. Research shows ~90% centred, only
    # ~10% shifted. Reduce from 12% to 5% and cap shift at ±15%.
    if rng.random() < 0.05:
        centre_x += int(rng.uniform(-0.15, 0.15) * IMG_SIZE)

    start_x = centre_x - total_w // 2
    band_y = rng.randint(IMG_SIZE // 2 - 6, IMG_SIZE // 2 + 6)

    # Mục 12/27: per-glyph baseline wave amplitude
    use_wave = rng.random() < 0.12
    wave_amp = rng.uniform(3, 7) if use_wave else 0.0
    wave_phase = rng.uniform(0, 2 * math.pi)

    # Mục 25: whole-string rotation about the string centre
    use_string_tilt = rng.random() < 0.05
    tilt_angle = math.radians(rng.uniform(-10, 10)) if use_string_tilt else 0.0

    centres: List[Tuple[int, int]] = []
    cur_x = start_x
    for i in range(n):
        if i == extra_gap_at + 1:
            cur_x += gap_extra
        cy = band_y + rng.randint(-5, 5)
        if use_wave:
            cy += int(wave_amp * math.sin(wave_phase + i * 1.1))
        cx = cur_x
        if use_string_tilt:
            # Rotate (cx, cy) around (centre_x, band_y)
            ox, oy = cx - centre_x, cy - band_y
            cos_t, sin_t = math.cos(tilt_angle), math.sin(tilt_angle)
            cx = int(centre_x + ox * cos_t - oy * sin_t)
            cy = int(band_y + ox * sin_t + oy * cos_t)
        centres.append((cx, cy))
        cur_x += spacing
    return centres, band_y


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------
@dataclass
class SynthSample:
    image: Image.Image
    text: str


def _decide_line_style(rng: random.Random) -> str:
    """Pick a line-noise style with the empirical mix."""
    return rng.choices(
        ["single", "x_grid", "cluster", "dense", "none"],
        weights=[0.45, 0.18, 0.10, 0.07, 0.20],
    )[0]


def render(
    text: Optional[str],
    font_paths: Sequence[str],
    rng: Optional[random.Random] = None,
) -> SynthSample:
    rng = rng or random.Random()
    if text is None:
        text = "".join(rng.choice(ALPHABET) for _ in range(CHAR_LEN))

    if not font_paths:
        raise ValueError("font_paths is empty — discover_fonts() returned nothing.")

    # Per-glyph font plan: 85% same font, 15% mixed.
    glyph_fonts = _plan_string_fonts(text, font_paths, rng)

    # Per-IMAGE scale-disparity flag — set True for ~13% of samples.
    use_strong_disparity = rng.random() < 0.13

    # 1) background
    img, _bg_name = _sample_background(rng)

    # 2) line noise — drawn UNDER the glyphs.
    glyph_plan = _plan_glyph_colours(text, rng)
    line_style = _decide_line_style(rng)
    if line_style != "none":
        _draw_lines(
            img,
            rng,
            style=line_style,
            glyph_hues=[_rgb_to_hsv(c)[0] for c in glyph_plan],
        )

    # 3) layout
    centres, _band_y = _plan_centres(text, rng)

    # 4) glyphs — base_size 28-34 keeps total string width within ~80% canvas.
    base_size = rng.randint(28, 34)
    for i, ch in enumerate(text):
        cx, cy = centres[i]
        prev_colour = glyph_plan[i - 1] if i > 0 else None
        # Repeated-char nudge (mục 8/16/24/26)
        if i > 0 and text[i] == text[i - 1]:
            if _hue_distance(glyph_plan[i], glyph_plan[i - 1]) < 60 / 360:
                cx += rng.randint(2, 5)
        _draw_one_glyph(
            img,
            label_char=ch,
            cx=cx,
            cy=cy,
            font_path=glyph_fonts[i],
            rng=rng,
            base_hue=_rgb_to_hsv(glyph_plan[i])[0],
            near_hue_window_deg=10,
            forbid_colour_close_to=prev_colour,
            base_size=base_size,
            forced_colour=glyph_plan[i],
            use_strong_disparity=use_strong_disparity,
        )

    # 5) top-layer dot speckles
    _draw_dots(img, rng)

    # 6) optional final blur (mimic upscale aliasing)
    if rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.8)))

    return SynthSample(image=img, text=text)


# ---------------------------------------------------------------------------
# Glyph colour planning — separate so the line drawer can also reference it.
# ---------------------------------------------------------------------------
def _plan_glyph_colours(text: str, rng: random.Random) -> List[Tuple[int, int, int]]:
    """Return a 5-element list of glyph RGB colours covering several modes:

    * 5%   — same-family multi-glyph (mục 7/33): all 5 share base hue ±25°.
    * 10%  — same-family pair (mục 32): 2 adjacent glyphs near-hue.
    * 5%   — exact same hue between adjacent (mục 21/24): copy hue.
    * Otherwise — diverse hues, with hard distance constraint between pairs.
    """
    n = len(text)
    mode = rng.choices(
        ["same_family_all", "same_family_pair", "same_hue_pair", "diverse"],
        weights=[0.05, 0.10, 0.05, 0.80],
    )[0]

    if mode == "same_family_all":
        base_hue = rng.random()
        return [
            _sample_glyph_colour(rng, base_hue=base_hue, hue_window_deg=50)
            for _ in range(n)
        ]

    # Build diverse base list first
    used_hues: List[float] = []
    colours: List[Tuple[int, int, int]] = []
    for i in range(n):
        for _ in range(12):
            c = _sample_glyph_colour(rng)
            h, _, _ = _rgb_to_hsv(c)
            if all(min(abs(h - u), 1 - abs(h - u)) >= 50 / 360 for u in used_hues):
                colours.append(c)
                used_hues.append(h)
                break
        else:
            colours.append(_sample_glyph_colour(rng))
            used_hues.append(_rgb_to_hsv(colours[-1])[0])

    if mode == "same_family_pair" and n >= 2:
        i = rng.randrange(n - 1)
        h, _, _ = _rgb_to_hsv(colours[i])
        colours[i + 1] = _sample_glyph_colour(rng, base_hue=h, hue_window_deg=40)

    if mode == "same_hue_pair" and n >= 2:
        i = rng.randrange(n - 1)
        colours[i + 1] = colours[i]

    # Repeated-character protection (mục 8/16): if label[i]==label[i+1] and
    # the colours are too close, force them apart UNLESS we explicitly chose
    # same-hue mode.
    return colours


# ---------------------------------------------------------------------------
# CLI preview helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fonts = discover_fonts()
    if not fonts:
        raise SystemExit("No usable TrueType fonts found.")
    out_dir = "synth_preview"
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    for i in range(40):
        s = render(None, fonts, rng)
        s.image.save(os.path.join(out_dir, f"synth_{i:02d}_{s.text}.png"))
    print(f"Wrote 40 preview images to {out_dir}/")
