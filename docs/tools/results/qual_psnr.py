#!/usr/bin/env python3
"""Overlay the per-cell PSNR numbers on the qualitative figures of results.html.

Numbers come from data/qual_psnr.json (see its _source field). Cell positions are
measured from the white gutters of the full-resolution masters in
images/results/qual/fullres/, so the badges land on the cells whatever the montage
geometry. Idempotent: an existing <div class="qual-badges"> is replaced.

    python3 docs/tools/results/qual_psnr.py          # rewrite results.html in place
"""
import json, os, re, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
PAGE = os.path.join(DOCS, "results.html")
FULL = os.path.join(DOCS, "images", "results", "qual", "fullres")
DATA = json.load(open(os.path.join(DOCS, "data", "qual_psnr.json")))

# figure name -> (row keys, column keys in the figure's column order); None = no number (GT)
UBO_COLS = ["Bonn", "MERL", "PBR", "Ours", None]
IMG_COLS = [None, "Ours", "Bonn", "MERL", "PBR"]
FIGS = {
    "ubo2014_1-6":  (["carpet02", "carpet07", "carpet09", "carpet12", "fabric02", "fabric04"], UBO_COLS, "ubo2014"),
    "ubo2014_7-12": (["fabric09", "fabric11", "felt01", "felt03", "felt05", "felt10"], UBO_COLS, "ubo2014"),
}
for mat in ("226", "452", "370", "314", "145"):
    FIGS[f"imgspace_{mat}"] = ([0, 1, 2, 3], IMG_COLS, f"image_space:{mat}")


def cells(name, thr=250, min_run=20):
    """Content runs (fractions) along x and y, found from all-white gutters."""
    im = np.asarray(Image.open(f"{FULL}/{name}.jpg").convert("L"))
    H, W = im.shape

    def runs(mask):
        out, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            if not v and start is not None:
                out.append((start, i)); start = None
        if start is not None:
            out.append((start, len(mask)))
        return out

    cols = [(a / W, b / W) for a, b in runs(im.min(axis=0) < thr) if b - a > min_run]
    rows = [(a / H, b / H) for a, b in runs(im.min(axis=1) < thr) if b - a > min_run]
    return cols, rows


def value(src, row, col):
    if col is None:
        return None
    if src == "ubo2014":
        return DATA["ubo2014"][row][col]
    mat = src.split(":")[1]
    return DATA["image_space"][mat][col][row]


def badges(name):
    rows, colkeys, src = FIGS[name]
    cols, rws = cells(name)
    assert len(cols) == len(colkeys) and len(rws) == len(rows), (name, len(cols), len(rws))
    out = []
    for r, rk in enumerate(rows):
        for c, ck in enumerate(colkeys):
            v = value(src, rk, ck)
            if v is None:
                continue
            x0, _ = cols[c]; y0, _ = rws[r]
            out.append(f'<span style="left:{100*x0:.2f}%;top:{100*y0:.2f}%">{v:.2f}</span>')
    return '<div class="qual-badges" aria-hidden="true">' + "".join(out) + "</div>"


def main():
    html = open(PAGE).read()
    n = 0
    for name in FIGS:
        pat = re.compile(r'(<a class="res-card qual-card qual-img" href="images/results/qual/fullres/' + re.escape(name)
                         + r'\.jpg"[^>]*>\s*<img[^>]*>)(\s*<div class="qual-badges".*?</div>)?', re.S)
        m = pat.search(html)
        if not m:
            sys.exit(f"figure {name} not found in results.html")
        html = html[:m.start()] + m.group(1) + "\n                " + badges(name) + html[m.end():]
        n += 1
    open(PAGE, "w").write(html)
    print(f"results.html: badges written for {n} figures")


if __name__ == "__main__":
    main()
