#!/usr/bin/env python3
"""
Copy the real camera / LED cut-outs used by the 3-D capture viewer
(obj.html, js/viz.js) into docs/images/viz/.

Source: the launch-video gizmo assets (option H3b) — photographic cut-outs of
the rig's camera head and LED head from the 4K rig footage, matte-cleaned and
regraded (see the teaser repo, tools/elements/gizmo_v2_cutout.py). Each sprite
is padded (right / bottom, transparent) to power-of-two dimensions so WebGL 1
can mip-map it without three.js resizing it; the pixel geometry in viz.js
(anchor / axis / content size) is unchanged by the padding.

    python3 docs/tools/build_viz_assets.py [--src <cutouts dir>]
"""
import argparse
import os

from PIL import Image

SRC = "/media/raid/cloth/output/teaser_video/elements/gizmo_v2/assets/cutouts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "viz")
FILES = {  # out name: source name
    "cam_side.png": "cam_side_graded.png",
    "cam_above.png": "cam_above_graded.png",
    "led_front.png": "led_front_h3b_graded.png",
    "led_side.png": "led_side_h3b_graded.png",
    "led_rear.png": "led_rear_h3b_graded.png",
}


def pot(n):
    p = 1
    while p < n:
        p *= 2
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for out, src in FILES.items():
        im = Image.open(os.path.join(a.src, src)).convert("RGBA")
        w, h = im.size
        canvas = Image.new("RGBA", (pot(w), pot(h)), (0, 0, 0, 0))
        canvas.paste(im, (0, 0))
        p = os.path.join(OUT, out)
        canvas.save(p, optimize=True)
        print(f"{out}: {w}x{h} -> {canvas.size[0]}x{canvas.size[1]}, {os.path.getsize(p) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
