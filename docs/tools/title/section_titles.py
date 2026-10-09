#!/usr/bin/env python
"""Section titles as cloth patchwork, in the hero title's spirit but with no rendering: each
letter of the word is cut into a few rectangles and every rectangle is filled with a crop of a
real material (the rectified preview thumbnails), with thin seams and a dark outline so the
word stays readable on the page's light background.

  python docs/tools/title/section_titles.py                 # all titles -> images/titles/<slug>.png (3x)
  python docs/tools/title/section_titles.py --titles Overview Method --scale 3

CSS shows them at the h3 size (`.cloth-title img { height: 2.6rem }`); the h3 keeps the text in a
visually-hidden span (section navigator, search engines, screen readers).
"""
import argparse, glob, os, random, re
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, "..", ".."))
FONT = os.path.join(HERE, "Jost-Bold.ttf")
THUMBS = sorted(glob.glob(os.path.join(DOCS, "images", "thumbs", "*.jpg")))
OUT = os.path.join(DOCS, "images", "titles")
TITLES = ["Overview", "Method", "Results", "Dataset Preview", "Citation"]
OUTLINE = (43, 47, 54)          # #2b2f36, as the hero title
SEAM = (43, 47, 54, 150)
PALETTE = None


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


TONE_MEAN, TONE_STD = 92.0, 14.0          # every patch is graded to this sRGB band: the word reads as
                                            # one dark shape, the cloth shows as texture, not as colour


def calm_materials():
    """Thumbnails whose own tone is mid-dark and not strongly coloured (readable as letters)."""
    keep = []
    for m in THUMBS:
        a = np.asarray(Image.open(m).convert("RGB").resize((64, 64)), np.float32)
        lum = a.mean(axis=2); sat = (a.max(axis=2) - a.min(axis=2)).mean()
        if 45 <= lum.mean() <= 170 and sat < 55:
            keep.append(m)
    return keep or THUMBS


def patch(mat, w, h, rng, texel_px):
    """A crop of material `mat` (~texel_px texture per output pixel), graded to the common tone band."""
    src = Image.open(mat).convert("RGB")
    cw, ch = int(w * texel_px), int(h * texel_px)
    cw, ch = min(cw, src.width), min(ch, src.height)
    x = rng.randint(0, src.width - cw); y = rng.randint(0, src.height - ch)
    im = src.crop((x, y, x + cw, y + ch)).resize((max(1, w), max(1, h)), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    lum = a.mean(axis=2, keepdims=True)
    chroma = a - lum                                   # keep a hint of the cloth's colour
    z = (lum - lum.mean()) / max(lum.std(), 1e-3)
    graded = TONE_MEAN + np.clip(z, -2.5, 2.5) * TONE_STD + 0.45 * chroma
    return Image.fromarray(np.clip(graded, 0, 255).astype(np.uint8))


def render(text, px, scale, seed):
    rng = random.Random(seed)
    global PALETTE
    PALETTE = calm_materials()
    font = ImageFont.truetype(FONT, px)
    pad = int(px * 0.12)
    # layout: per-glyph boxes
    x = pad; boxes = []
    for ch in text:
        if ch == " ":
            x += int(px * 0.28); continue
        bb = font.getbbox(ch)                     # (l, t, r, b) relative to origin at (0, 0)
        adv = font.getlength(ch)
        boxes.append((ch, x, bb))
        x += adv * 0.965                          # slight negative tracking (bold caps sit tighter)
    W = int(x + pad); asc, desc = font.getmetrics(); H = asc + desc + 2 * pad
    mask = Image.new("L", (W, H), 0); md = ImageDraw.Draw(mask)
    for ch, gx, bb in boxes:
        md.text((int(gx), pad), ch, font=font, fill=255)
    fill = Image.new("RGB", (W, H), (0, 0, 0))
    seams = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sd = ImageDraw.Draw(seams)
    last = None
    for ch, gx, bb in boxes:
        l, t, r, b = int(gx + bb[0]), int(pad + bb[1]), int(gx + bb[2]), int(pad + bb[3])
        gw, gh = r - l, b - t
        # cells: narrow glyphs 1 column, others 2-3; 2-4 rows; jittered splits
        ncol = 1 if gw < px * 0.28 else rng.choice([1, 2, 2])
        nrow = rng.choice([2, 2, 3]) if gh > px * 0.5 else 2
        xs = [l] + sorted(rng.randint(int(l + gw * 0.25), int(r - gw * 0.25)) for _ in range(ncol - 1)) + [r] if ncol > 1 else [l, r]
        ys = [t] + sorted(rng.randint(int(t + gh * 0.2), int(b - gh * 0.2)) for _ in range(nrow - 1)) + [b]
        for i in range(ncol):
            for j in range(nrow):
                cw, chh = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
                if cw <= 0 or chh <= 0: continue
                mat = rng.choice([m for m in PALETTE if m != last] or PALETTE); last = mat
                fill.paste(patch(mat, cw, chh, rng, texel_px=rng.uniform(1.6, 2.6) / scale * 3), (xs[i], ys[j]))
        for xx in xs[1:-1]: sd.line([(xx, t), (xx, b)], fill=SEAM, width=max(1, scale))
        for yy in ys[1:-1]: sd.line([(l, yy), (r, yy)], fill=SEAM, width=max(1, scale))
    # compose: outline (dilated mask in dark) under the patchwork letters
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ring = mask.filter(ImageFilter.MaxFilter(2 * max(1, int(1.2 * scale)) + 1))
    outline = Image.new("RGBA", (W, H), OUTLINE + (255,)); outline.putalpha(ring)
    out.alpha_composite(outline)
    letters = fill.convert("RGBA"); letters.putalpha(mask)
    # seams only inside the letters
    sm = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sm.paste(seams, (0, 0), Image.fromarray(np.minimum(np.asarray(seams.split()[3]), np.asarray(mask))))
    letters.alpha_composite(sm)
    out.alpha_composite(letters)
    # trim transparent margins
    a = np.asarray(out.split()[3]); ys_, xs_ = np.where(a > 0)
    return out.crop((xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", nargs="*", default=TITLES)
    ap.add_argument("--scale", type=int, default=3, help="device-pixel multiple of the CSS size")
    ap.add_argument("--css-px", type=float, default=41.6, help="h3 font size in CSS px (2.6rem)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k, t in enumerate(a.titles):
        im = render(t, int(a.css_px * a.scale), a.scale, a.seed + k)
        p = os.path.join(OUT, slug(t) + ".png"); im.save(p, optimize=True)
        print(f"{t:16s} -> {os.path.relpath(p, DOCS)} {im.size} {os.path.getsize(p)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
