#!/usr/bin/env python
"""The page's composite of the fully dressed test room vs a real path-traced render of it.

Builds xtalk.DRESS on the live page (headless Chrome), reads the composite back
(roomDemo.core.snapshot(), no highlights), with and without the crosstalk term, and compares
with xtalk.py's c1920 (the same room path-traced with every piece dressed at once, 1920x1080,
256 spp, OptiX-denoised). Metric as xtalk.py: per-pixel max over RGB of |difference| in 8-bit
sRGB DN; 'lp' = both blurred (sigma 1.5 px) first.

  /data/colin/envs/teaser/bin/python docs/tools/room/measure_page.py [URL]
"""
import base64
import json
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtalk as X  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
P5 = "/media/raid/cloth/output/teaser_video/room_demo/passes_r5"
#: xtalk.DRESS pieces -> the page's pieces (r5: the pillow pairs are single pillows)
SPLIT = {"pillowsL": ["pillow55", "pillow59"], "pillowsR": ["pillow3", "pillow5"]}
ARGS = ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"]
TAG = os.environ.get("MEASURE_TAG", "")          # suffix for the output files


def snap(pg, crosstalk):
    pg.evaluate(f"window.__roomNoCrosstalk = {'false' if crosstalk else 'true'}")
    d = pg.evaluate("roomDemo.core.snapshot()")
    buf = np.frombuffer(base64.b64decode(d.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=ARGS)
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.goto(URL + "?measure=1")
        pg.evaluate("document.getElementById('apply').scrollIntoView({block:'center'})")
        pg.wait_for_function("window.roomDemo && document.querySelector('.room-ready')", timeout=60000)
        pg.evaluate("roomDemo.demo.stop()")
        ids = pg.evaluate("roomDemo.core.TGT.map(t => t.id)")
        mats = pg.evaluate("roomDemo.core.MATS.map(m => m.id)")
        pg.evaluate("roomDemo.demo.stop()")
        pg.wait_for_function("!roomDemo.core.busy()", timeout=60000)
        for r0, m in X.DRESS.items():
            for r in SPLIT.get(r0, [r0]):
                k, mi = ids.index(r), mats.index(m)
                pg.evaluate(f"(async () => {{ await roomDemo.core.need({mi}); const d = roomDemo.core.TGT[{k}].drop;"
                            f" roomDemo.core.apply({k}, {mi}, d[0], d[1]); }})()")
                pg.wait_for_timeout(300)
        pg.wait_for_function("!roomDemo.core.busy()", timeout=60000)
        pg.wait_for_timeout(500)
        state = dict(zip(ids, pg.evaluate("roomDemo.core.regionMat")))
        page = snap(pg, True)
        page_nox = snap(pg, False)
        b.close()
    out_dir = X.OUTX
    cv2.imwrite(f"{out_dir}/page_dressed_1920{TAG}.png", page[:, :, ::-1].astype(np.uint8))
    H, W = page.shape[:2]
    c = X.srgb8(cv2.imread(f"{X.RD}/c1920.exr", cv2.IMREAD_UNCHANGED)[:H, :, ::-1].astype(np.float32))
    cov4 = np.load(f"{P5}/masks.npy")[:2 * H, :2 * W]
    cov = cv2.resize(cov4, (W, H), interpolation=cv2.INTER_AREA) if cov4.shape[1] != W else cov4
    names = json.load(open(f"{P5}/masks.json"))
    pieces = np.clip(sum(cov[:, :, names[r]["channel"]] for r0 in X.DRESS for r in SPLIT.get(r0, [r0])), 0, 1)
    edge = ((cov > 0.02) & (cov < 0.98)).any(2).astype(np.uint8)
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8)) > 0
    lp = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    res = {"state": {k: (mats[v] if v >= 0 else None) for k, v in state.items()}}
    for nm, img in (("page", page), ("page_without_crosstalk", page_nox)):
        o = {}
        for tag, f in (("raw", lambda z: z), ("lp", lp)):
            v = np.abs(f(c) - f(img)).max(2)
            st = lambda m: [round(float(v[m].mean()), 2), round(float(np.percentile(v[m], 99)), 1),
                            round(float((v[m] > 4).mean()), 3)]
            o[tag] = {"frame": st(np.ones_like(edge)), "inside_pieces": st(pieces > 0.5),
                      "outside_pieces": st(pieces < 0.01), "edge_band": st(edge), "away_from_edges": st(~edge)}
        o["bias_inside_pieces_rgb"] = (c - img)[pieces > 0.5].mean(0).round(2).tolist()
        o["bias_outside_rgb"] = (c - img)[pieces < 0.01].mean(0).round(2).tolist()
        res[nm] = o
        cv2.imwrite(f"{out_dir}/err_c1920_minus_{nm}{TAG}_lowpass_x8.png",
                    np.clip(np.abs(lp(c) - lp(img)) * 8, 0, 255).astype(np.uint8)[:, :, ::-1])
    cv2.imwrite(f"{out_dir}/c1920_render.png", np.clip(c, 0, 255).astype(np.uint8)[:, :, ::-1])
    for nm2, (x0, y0, x1, y1) in {"left_sofa": (80, 240, 920, 600), "cushions_right_sofa": (840, 260, 1920, 840)}.items():
        tiles = []
        for lab, im in (("render", c), ("page", page), ("|render-page| lowpass x8", np.abs(lp(c) - lp(page)) * 8)):
            t = np.ascontiguousarray(np.clip(im[y0:y1, x0:x1], 0, 255).astype(np.uint8)[:, :, ::-1])
            cv2.putText(t, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(t)
        cv2.imwrite(f"{out_dir}/final{TAG}_crop_{nm2}.png", np.concatenate(tiles, 1))
    json.dump(res, open(f"{out_dir}/final{TAG}_page_vs_render.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
