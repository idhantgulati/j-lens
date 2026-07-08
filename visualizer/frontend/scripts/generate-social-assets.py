#!/usr/bin/env python3
"""Generate PNG favicons and OG image for link previews."""

import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT
EMOJI_SRC = ROOT / "emoji-source.png"
TWEMOJI_MAGNIFIER = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f50e.png"

# Brand palette (light theme)
BG = (255, 253, 250)       # #fffdfa
INK = (31, 41, 55)         # #1f2937
MUTED = (75, 85, 99)       # #4b5563
ACCENT = (179, 86, 27)     # #b3561b
ACCENT_BRIGHT = (219, 122, 42)  # #db7a2a
LINE = (231, 224, 209)     # #e7e0d1
PANEL = (248, 243, 234)    # #f8f3ea
MOTOR = (45, 94, 157)      # #2d5e9d


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _sans(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Avenir Next.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size, index=1 if bold and path.endswith(".ttc") else 0)
    return ImageFont.load_default()


def _emoji_src() -> Image.Image:
    if EMOJI_SRC.exists():
        return Image.open(EMOJI_SRC).convert("RGBA")
    with urllib.request.urlopen(TWEMOJI_MAGNIFIER) as resp:
        data = resp.read()
    EMOJI_SRC.write_bytes(data)
    return Image.open(BytesIO(data)).convert("RGBA")


def make_favicon_png(size: int, src: Image.Image | None = None) -> Image.Image:
    if src is None:
        src = _emoji_src()
    img = Image.new("RGBA", (size, size), BG + (255,))
    pad = max(1, size // 10)
    inner = size - 2 * pad
    emoji = src.resize((inner, inner), Image.Resampling.LANCZOS)
    img.paste(emoji, (pad, pad), emoji)
    return img


def paste_emoji(img: Image.Image, cx: int, cy: int, size: int, src: Image.Image) -> None:
    emoji = src.resize((size, size), Image.Resampling.LANCZOS)
    img.paste(emoji, (cx - size // 2, cy - size // 2), emoji)


def _blend(bg: tuple[int, int, int], fg: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return tuple(int(fg[i] * a + bg[i] * (1 - a)) for i in range(3))


def _warm_background(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), BG)
    glow = Image.new("RGB", (w, h), BG)
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse((w - 520, -180, w + 120, 420), fill=(255, 244, 230))
    gdraw.ellipse((w - 760, 80, w - 40, 620), fill=(248, 236, 220))
    gdraw.ellipse((-80, h - 220, 360, h + 80), fill=(245, 240, 232))
    return Image.blend(img, glow, 0.42)


def _mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size, index=0)
    return ImageFont.load_default()


def _draw_card_shadow(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 16) -> None:
    x0, y0, x1, y1 = box
    for i, alpha in enumerate((0.04, 0.03, 0.02, 0.01)):
        offset = 10 - i * 2
        draw.rounded_rectangle(
            (x0, y0 + offset, x1, y1 + offset),
            radius=radius,
            fill=_blend(BG, (26, 25, 23), alpha),
        )


def _draw_mini_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=LINE, width=1)

    # Layer bands: sensory (grey), workspace (warm), motor (cool)
    layers = [
        ("sensory", 0.04, None),
        ("sensory", 0.04, None),
        ("sensory", 0.05, None),
        ("workspace", 0.10, " Italy"),
        ("workspace", 0.12, " euro"),
        ("workspace", 0.14, " Rome"),
        ("workspace", 0.16, " tennis"),
        ("workspace", 0.18, " rugby"),
        ("workspace", 0.16, " sport"),
        ("workspace", 0.14, " ball"),
        ("motor", 0.09, " the"),
        ("motor", 0.11, " is"),
    ]
    cols = 8
    pad_x, pad_y = 16, 34
    inner_w = x1 - x0 - pad_x * 2
    inner_h = y1 - y0 - pad_y * 2 - 18
    row_h = inner_h // len(layers)
    col_w = inner_w // cols
    mono = _mono(13)
    label_font = _mono(10)
    header_font = _sans(13, bold=True)

    draw.text((x0 + pad_x, y0 + 12), "argmax · layer × position", font=header_font, fill=MUTED)

    sample_tokens = [
        [" the", " fact", " Italy", " is", " shaped", " like", " a", " boot"],
        [" the", " currency", " used", " in", " that", " country", " is", " euro"],
        [" Italy", " uses", " the", " euro", " as", " its", " currency", "."],
        [" the", " capital", " of", " Italy", " is", " Rome", ",", " not"],
        [" Rome", " sits", " in", " the", " center", " of", " Italy", "."],
        [" tennis", " and", " rugby", " are", " both", " popular", " sports", ""],
        [" rugby", " is", " often", " compared", " to", " football", " in", ""],
        [" the", " sport", " I", " chose", " is", " tennis", ".", ""],
        [" tennis", " uses", " a", " racket", " and", " a", " yellow", " ball"],
        [" ball", " sports", " include", " tennis", ",", " rugby", ",", " soccer", ""],
        [" the", " answer", " is", " the", " euro", ".", " Italy", " uses", " it"],
        [" the", " next", " token", " is", " likely", " euro", ".", ""],
    ]

    for row, (band, tint, label) in enumerate(layers):
        ry = y0 + pad_y + row * row_h
        if band == "workspace":
            band_fill = _blend((255, 255, 255), ACCENT_BRIGHT, tint)
        elif band == "motor":
            band_fill = _blend((255, 255, 255), MOTOR, tint)
        else:
            band_fill = _blend((255, 255, 255), (156, 148, 132), tint)
        draw.rectangle((x0 + 1, ry, x1 - 1, ry + row_h), fill=band_fill)
        draw.text((x0 + 8, ry + 5), f"L{row + 8}", font=label_font, fill=_blend(MUTED, INK, 0.15))

        tokens = sample_tokens[row]
        for col in range(cols):
            cx = x0 + pad_x + col * col_w
            cy = ry + 4
            token = tokens[col] if col < len(tokens) else ""
            if not token:
                continue
            chip_w = min(col_w - 4, int(draw.textlength(token, font=mono) + 12))
            chip_fill = (255, 255, 255) if band != "sensory" else _blend((255, 255, 255), (240, 236, 228), 0.5)
            outline = LINE if band == "sensory" else _blend(LINE, ACCENT_BRIGHT, 0.15 if band == "workspace" else 0.0)
            draw.rounded_rectangle((cx, cy, cx + chip_w, cy + row_h - 8), radius=5, fill=chip_fill, outline=outline, width=1)
            ink = INK if band != "sensory" else _blend(INK, MUTED, 0.45)
            if label and token.strip() == label.strip():
                ink = ACCENT
                draw.rounded_rectangle((cx, cy, cx + chip_w, cy + row_h - 8), radius=5, outline=ACCENT_BRIGHT, width=1)
            draw.text((cx + 5, cy + 3), token, font=mono, fill=ink)


def make_og_image(src: Image.Image | None = None) -> Image.Image:
    if src is None:
        src = _emoji_src()
    w, h = 1200, 630
    img = _warm_background(w, h)
    draw = ImageDraw.Draw(img)

    left = 72
    card = (668, 54, 1148, 558)
    text_right = card[0] - 36

    paste_emoji(img, left + 30, 78, 56, src)

    title_font = _font(58, bold=True)
    sub_font = _font(26)
    badge_font = _mono(19)
    tag_font = _sans(20)
    foot_font = _sans(18)

    title_x = left + 74
    title_y = 52
    draw.text((title_x, title_y), "J-", font=title_font, fill=INK)
    j_w = draw.textlength("J-", font=title_font)
    draw.text((title_x + j_w, title_y), "Lens", font=title_font, fill=ACCENT)
    lens_w = draw.textlength("Lens", font=title_font)
    draw.text((title_x + j_w + lens_w + 6, title_y), "Visualizer", font=title_font, fill=INK)

    draw.line((left, 126, text_right, 126), fill=LINE, width=1)

    draw.text((left, 148), "What a language model is", font=sub_font, fill=MUTED)
    draw.text((left, 182), "holding at every layer", font=sub_font, fill=MUTED)

    badge = "Qwen3.5-4B"
    bx, by = left, 244
    bw = int(draw.textlength(badge, font=badge_font) + 24)
    draw.rounded_rectangle((bx, by, bx + bw, by + 34), radius=17, fill=(255, 255, 255), outline=LINE, width=1)
    draw.text((bx + 12, by + 6), badge, font=badge_font, fill=MUTED)

    draw.text((left, 302), "Jacobian lens · logit-lens baseline", font=tag_font, fill=_blend(MUTED, INK, 0.12))
    draw.text((left, 332), "swap / steer interventions · J-Space", font=tag_font, fill=_blend(MUTED, INK, 0.12))
    draw.text((left, 374), "Based on Gurnee et al. (2026)", font=foot_font, fill=_blend(MUTED, INK, 0.05))

    _draw_card_shadow(draw, card, radius=18)
    _draw_mini_grid(draw, card)

    draw.text((left, h - 46), "j-lens.idhant.xyz", font=_mono(18), fill=_blend(MUTED, INK, 0.1))

    return img


def _sync_hero_from_og_jpg() -> bool:
    """Copy og-image.jpg → hero-art.png when the manual OG asset is present."""
    og_jpg = OUT / "og-image.jpg"
    if not og_jpg.is_file():
        return False
    Image.open(og_jpg).convert("RGB").save(OUT / "hero-art.png", optimize=True)
    print("synced hero-art.png from og-image.jpg")
    return True


def main() -> None:
    src = _emoji_src()
    for size, name in ((16, "favicon-16x16.png"), (32, "favicon-32x32.png"), (180, "apple-touch-icon.png")):
        make_favicon_png(size, src).save(OUT / name, optimize=True)
        print(f"wrote {name}")

    icons = [make_favicon_png(s, src) for s in (16, 32, 48)]
    icons[0].save(OUT / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])

    if _sync_hero_from_og_jpg():
        return

    og = make_og_image(src)
    og.save(OUT / "og-image.png", optimize=True)
    print("wrote og-image.png")


if __name__ == "__main__":
    main()
