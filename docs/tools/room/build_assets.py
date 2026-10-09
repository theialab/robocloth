#!/usr/bin/env python
"""Pack the room render passes into the web assets of the "apply a trained material" panel.

  export PYTHONNOUSERSITE=1
  /data/colin/envs/teaser/bin/python docs/tools/room/build_assets.py [--partial]

r5: one region per piece (12: each sofa body, each of the 4 throw pillows, the 4 floor
cushions, the rug, the left curtain), ten materials, two resolutions from one set of 4K renders:
  page = base + sum_r cov_r * (C_r - G_r)            (C_r / G_r: piece r's colour in its current
                                                      material / in grey, from the interior of
                                                      its own pass pushed over its AA band)
         * prod_r [1 + (ratio_r - 1) * (1 - cov_r)]   (ratio_r = lo_<r>_<m> / lo_grey: how piece r
                                                      in m changes the light reaching the rest)
Regions that touch go into different "classes" (graph colouring) so one image per class can
hold every region's colour, edges included.
Inputs (/media/raid/cloth/output/teaser_video/room_demo/):
  passes_r5/   masks.npy (12 regions at 3840x2160, passes.py REGIONS order, 256 spp AOV),
               position.npy, neural_<region>_<m>.exr crops (3840x2160 frame, 1024 spp, even
               offsets), lo_<region>_<m>.exr + lo_grey.exr (480x270), swatch_<m>.exr, times.json
  passes_r2/r3 swatch_<m>.exr of the earlier materials (flat 512^2 patches, resolution-free)
  M1's grey room (thread/method_assets/work/room_webpage_grey_spp2048.exr, 3840x2160)
  thread/latent_decoded/cells/<m>_ccm_web512.webp  the decoded latent tiles (latdec_<m>)
Output: docs/media/room/
  per resolution set ("" = 1920x900, the 4K renders downsampled 2x2; "4k/" = 3840x1800):
      base.webp, masks_{a,b,c,d}.webp, grey_<k>.webp (per class),
      mat_<m>_<k>.webp (per material and class, cropped to the class box)
  shared (resolution-free, 1920 px): glow_{a..d}.webp, pos.webp, hdr.webp, ratio_<m>.webp
  (4x4 atlas of 480x225 crosstalk tiles), swatch_<m>.webp, latdec_<m>.webp, meta.json
The page picks the 4K set when devicePixelRatio * panel width > meta.sets[1].min_px.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SRC = "/media/raid/cloth/output/teaser_video/room_demo"
P5 = f"{SRC}/passes_r5"
SWATCH_DIRS = [f"{SRC}/passes_r5", f"{SRC}/passes_r3", f"{SRC}/passes_r2"]
M1_GREY = ("/media/raid/cloth/output/teaser_video/thread/method_assets/work/"
           "room_webpage_grey_spp2048.exr")
LATDEC = "/media/raid/cloth/output/teaser_video/thread/latent_decoded/cells/{}_ccm_web512.webp"
DST = os.environ.get("ROOM_DST") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "media", "room")
W4, H4 = 3840, 1800              # shown part of the 3840x2160 frame
W, H = 1920, 900                 # the 1x set and the hit-test / layout space
SETS = [dict(dir="", size=[W, H], min_px=0), dict(dir="4k/", size=[W4, H4], min_px=2000)]
PARTIAL = "--partial" in sys.argv          # development: missing passes stay grey
MATS = [(453, "red weave"), (8, "mustard"), (145, "quatrefoil"), (226, "houndstooth"), (9, "stripes"),
        (50, "coral"), (367, "lime"), (311, "blue weave"), (314, "blue-grey tweed"), (370, "satin")]
PRELOAD = [453]                  # the demo's first material; the others load on first grab
RATIO_SIGMA = (3.0, 12.0, 10.0)          # near sigma, far sigma, transition (lo px)
REGIONS = ["sofaL", "pillow59", "pillow55", "sofaR", "pillow3", "pillow5",
           "cush58", "cush2", "cush56", "cush57", "rug", "curtainL"]
#: id -> label, label when the drop also dresses the companions, companions, drop px (1920 space)
TARGETS = {
    "sofaL": ("Left sofa", "Left sofa + pillows", ["pillow55", "pillow59"], [520, 487]),
    "pillow55": ("Left sofa, left pillow", None, [], None),
    "pillow59": ("Left sofa, right pillow", None, [], None),
    "sofaR": ("Right sofa", "Right sofa + pillows", ["pillow3", "pillow5"], [1335, 505]),
    "pillow3": ("Right sofa, left pillow", None, [], None),
    "pillow5": ("Right sofa, right pillow", None, [], None),
    "cush58": ("Long floor pillow", None, [], [419, 637]),
    "cush2": ("Floor pillow", None, [], [854, 591]),
    "cush56": ("Round cushion", None, [], [1058, 644]),
    "cush57": ("Tufted floor cushion", None, [], [1322, 667]),
    "rug": ("Rug", None, [], [791, 760]),
    "curtainL": ("Curtain", None, [], None),        # the drape at the left edge (deepest point)
}
#: regions that get a class of their own: the curtain sits at the far left edge, away from every
#: other class's box, so folding it into one would about double that class's crop (and change
#: every mat_<m>_<k>.webp); as its own class it only adds small grey_3 / mat_<m>_3 images
OWN_CLASS = ["curtainL"]


def srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def wr(name, rgb01, q):
    """q > 100: lossless WebP."""
    p = os.path.join(DST, name)
    cv2.imwrite(p, np.ascontiguousarray(u8(rgb01)[:, :, ::-1]), [cv2.IMWRITE_WEBP_QUALITY, q])
    return os.path.getsize(p)


def defirefly(lin, k=2.5):
    med = np.stack([cv2.medianBlur(np.ascontiguousarray(lin[:, :, c]), 5) for c in range(3)], -1)
    bad = lin.mean(2) > k * med.mean(2) + 0.02
    out = lin.copy()
    out[bad] = med[bad]
    return out, float(bad.mean())


def pure(img, a, iters=6):
    """img with the region's interior colour (a > 0.95) pushed outwards over its antialiased
    band (0 < a <= 0.95), so the band no longer carries the neighbours' colour."""
    out = img.copy()
    valid = a > 0.95
    band = (a > 0.004) & ~valid
    k = np.ones((3, 3), np.float32)
    for _ in range(iters):
        todo = band & ~valid
        if not todo.any():
            break
        num = cv2.filter2D(out * valid[:, :, None], -1, k, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(valid.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        fill = todo & (den > 0)
        out[fill] = num[fill] / den[fill][:, None]
        valid = valid | fill
    return out


def halo(cov, fill_holes=True):
    m8 = (cov > 0.5).astype(np.uint8)
    if fill_holes:      # a sofa's pillows are holes in its mask and must not glow
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        ff = m8.copy()
        cv2.floodFill(ff, np.zeros((m8.shape[0] + 2, m8.shape[1] + 2), np.uint8), (0, 0), 2)
        filled = np.maximum((ff != 2).astype(np.float32), cov)
    else:
        filled = cov
    b = cv2.GaussianBlur(cv2.dilate(filled, np.ones((3, 3), np.uint8)), (0, 0), 10)
    return np.clip((b - filled) * 2.2, 0, 1)


def load_pass(region, m):
    t = json.load(open(f"{P5}/times.json"))
    tag = f"neural_{region}_{m}"
    if tag in t and os.path.exists(f"{P5}/{tag}.exr"):
        img = cv2.imread(f"{P5}/{tag}.exr", cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)
        return t[tag]["crop"], img, t[tag]
    raise FileNotFoundError(f"no pass for {region} {m}")


def down2(img):
    """2x2 area average (the 1920 set from the 4K one)"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def deepest(m):
    """a drop point deep inside a region (1920 px); the frame's edge counts as a boundary (the
    curtain runs along the left edge: its middle, not its first column)"""
    d = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return [int(x), int(y)]


def main():
    os.makedirs(DST, exist_ok=True)
    os.makedirs(os.path.join(DST, "4k"), exist_ok=True)
    sizes = {}
    nreg = len(REGIONS)
    nt = (nreg + 2) // 3
    letters = "abcdefgh"[:nt]
    grey4 = cv2.imread(M1_GREY, cv2.IMREAD_UNCHANGED)[:H4, :, ::-1].astype(np.float32)
    cov4 = np.load(f"{P5}/masks.npy")[:H4]
    s4 = cov4.sum(2, keepdims=True)
    cov4 = np.where(s4 > 1, cov4 / np.maximum(s4, 1e-6), cov4).astype(np.float32)
    data = {"": dict(grey=down2(grey4), cov=down2(cov4), f=0.5), "4k/": dict(grey=grey4, cov=cov4, f=1.0)}
    cov = data[""]["cov"]                                         # 1920 space: layout, glows, ...
    pos = down2(np.load(f"{P5}/position.npy")[:H4])

    # ---- shared, 1920 px
    head = np.clip(np.log2(np.maximum(data[""]["grey"], 1.0)) / 2.0, 0, 1)
    sizes["hdr.webp"] = wr("hdr.webp", cv2.resize(head, (W // 2, H // 2), interpolation=cv2.INTER_AREA), 90)
    cov_p = np.concatenate([cov, np.zeros((H, W, 3 * nt - nreg), np.float32)], -1)
    for i in range(nt):
        g = np.stack([halo(cov_p[:, :, 3 * i + c], REGIONS[3 * i + c].startswith("sofa"))
                      if 3 * i + c < nreg else np.zeros((H, W), np.float32) for c in range(3)], -1)
        sizes[f"glow_{letters[i]}.webp"] = wr(f"glow_{letters[i]}.webp", g, 101)
    anym = cov.max(2) > 0.02
    pmin = pos[anym].min(0) - 0.05
    pmax = pos[anym].max(0) + 0.05
    sizes["pos.webp"] = wr("pos.webp", (pos - pmin) / (pmax - pmin), 101)

    # classes: regions that share pixels must not share an image (decided once, at 4K)
    band4 = cov4 > 0.004
    adj = {i: {j for j in range(nreg) if j != i and (band4[:, :, i] & band4[:, :, j]).any()} for i in range(nreg)}
    own = [REGIONS.index(r) for r in OWN_CLASS if r in REGIONS]
    cls = {}
    for i in sorted((i for i in range(nreg) if i not in own), key=lambda i: -len(adj[i])):
        used = {cls[j] for j in adj[i] if j in cls}
        cls[i] = min(k for k in range(nreg) if k not in used)
    for i in own:
        cls[i] = max(cls.values()) + 1
    ncls = max(cls.values()) + 1
    print("classes:", {REGIONS[i]: cls[i] for i in range(nreg)}, "adjacent to the curtain:",
          [REGIONS[j] for i in own for j in sorted(adj[i])])
    for d in data.values():
        d["band"] = d["cov"] > 0.004
        hh, ww = d["cov"].shape[:2]
        d["boxes"] = []
        for k in range(ncls):
            m = np.zeros((hh, ww), bool)
            for i in range(nreg):
                if cls[i] == k:
                    m |= d["band"][:, :, i]
            ys, xs = np.where(m)
            x0, y0 = max(0, xs.min() - 2), max(0, ys.min() - 2)
            x1, y1 = min(ww, xs.max() + 3), min(hh, ys.max() + 3)
            d["boxes"].append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])

    def class_image(d, colour_of, k):
        """sRGB image of class k (cropped to its box): each region's colour on its pixels."""
        x0, y0, w, h = d["boxes"][k]
        img = np.full((h, w, 3), 0.5, np.float32)
        for i in range(nreg):
            if cls[i] != k:
                continue
            col = colour_of(i)                     # (y0c, x0c, patch) or None
            if col is None:
                continue
            cy, cx, patch = col
            ph, pw = patch.shape[:2]
            # intersect the patch with the class box
            ax0, ay0 = max(x0, cx), max(y0, cy)
            ax1, ay1 = min(x0 + w, cx + pw), min(y0 + h, cy + ph)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            m = d["band"][ay0:ay1, ax0:ax1, i]
            sub = img[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
            sub[m] = srgb(patch[ay0 - cy:ay1 - cy, ax0 - cx:ax1 - cx])[m]
        return img

    def region_window(d, i, pad=6):
        hh, ww = d["cov"].shape[:2]
        ys, xs = np.where(d["band"][:, :, i])
        return max(0, ys.min() - pad), max(0, xs.min() - pad), min(hh, ys.max() + pad + 1), min(ww, xs.max() + pad + 1)

    for sd, d in data.items():
        sizes[f"{sd}base.webp"] = wr(f"{sd}base.webp", srgb(d["grey"]), 92)
        cp = np.concatenate([d["cov"], np.zeros(d["cov"].shape[:2] + (3 * nt - nreg,), np.float32)], -1)
        for i in range(nt):
            sizes[f"{sd}masks_{letters[i]}.webp"] = wr(f"{sd}masks_{letters[i]}.webp", cp[:, :, 3 * i:3 * i + 3], 101)

        def grey_col(i, d=d):
            y0, x0, y1, x1 = region_window(d, i)
            return y0, x0, pure(d["grey"][y0:y1, x0:x1], d["cov"][y0:y1, x0:x1, i])
        for k in range(ncls):
            sizes[f"{sd}grey_{k}.webp"] = wr(f"{sd}grey_{k}.webp", class_image(d, grey_col, k), 92)

    lo_grey = cv2.imread(f"{P5}/lo_grey.exr", cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)
    LH, LW = lo_grey.shape[:2]
    TH = int(round(LH * H4 / 2160))                     # shown rows of the lo frame (225)
    RC = int(np.ceil(np.sqrt(nreg)))                    # atlas: RC x RC tiles
    cov_lo = cv2.resize(np.load(f"{P5}/masks.npy"), (LW, LH), interpolation=cv2.INTER_AREA)

    passes, stats, swatch_gain, missing, ratio_stats = {}, {}, {}, [], {}
    for m, _ in MATS:
        cols = {sd: {} for sd in data}
        for ri, r in enumerate(REGIONS):
            try:
                (x0, y0, w, h), img, rec = load_pass(r, m)
            except FileNotFoundError as e:
                if not PARTIAL:
                    raise
                missing.append(str(e))
                continue
            passes[f"neural_{r}_{m}"] = {k: rec[k] for k in ("crop", "spp", "render_s", "px")}
            img, frac = defirefly(img)
            img = cv2.bilateralFilter(img, 5, 0.06, 1.5)
            stats[f"{r}_{m}"] = round(frac, 5)
            y1 = min(y0 + h, H4)
            img = img[:y1 - y0]
            for sd, d in data.items():
                pimg = img if sd else down2(img)
                f = d["f"]
                cx, cy = int(x0 * f), int(y0 * f)
                ph, pw = pimg.shape[:2]
                cols[sd][ri] = (cy, cx, pure(pimg, d["cov"][cy:cy + ph, cx:cx + pw, ri]))
        for sd, d in data.items():
            for k in range(ncls):
                sizes[f"{sd}mat_{m}_{k}.webp"] = wr(f"{sd}mat_{m}_{k}.webp",
                                                    class_image(d, lambda i: cols[sd].get(i), k), 94)
        # crosstalk: ratio_r = smooth(lo_r) / smooth(lo_grey) away from r itself (its own change
        # would smear over its silhouette at this resolution), filled back by normalized convolution
        atlas = np.full((RC * TH, RC * LW, 3), 0.5, np.float32)
        for ri, r in enumerate(REGIONS):
            lp = f"{P5}/lo_{r}_{m}.exr"
            if not os.path.exists(lp):
                missing.append(f"lo {r} {m}")
                continue
            lo = cv2.imread(lp, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)
            excl = cv2.dilate((cov_lo[:, :, ri] > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            wgt = (~excl).astype(np.float32)[:, :, None]

            def smooth_ratio(sg):
                num = cv2.GaussianBlur(lo * wgt, (0, 0), sg)
                den = cv2.GaussianBlur(lo_grey * wgt, (0, 0), sg)
                return np.clip(np.where(den > 1e-4, num / np.maximum(den, 1e-4), 1.0), 0.5, 1.5)
            # two scales: lo renders are not per-pixel correlated (most paths touch some cloth),
            # so the ratio is smoothed with sigma 3 lo px near the piece, where the bleed is
            # sharp, and sigma 12 from ~10 lo px on (measured best vs the full render, xtalk.py)
            dist = cv2.distanceTransform((~excl).astype(np.uint8), cv2.DIST_L2, 5)
            kf = np.clip((dist - RATIO_SIGMA[2]) / RATIO_SIGMA[2], 0, 1)[:, :, None]
            rt = (smooth_ratio(RATIO_SIGMA[0]) * (1 - kf) + smooth_ratio(RATIO_SIGMA[1]) * kf)[:TH]
            ratio_stats[f"{r}_{m}"] = [round(float(np.percentile(rt, 1)), 3), round(float(np.percentile(rt, 99)), 3)]
            cy, cx = divmod(ri, RC)
            atlas[cy * TH:(cy + 1) * TH, cx * LW:(cx + 1) * LW] = rt - 0.5
        sizes[f"ratio_{m}.webp"] = wr(f"ratio_{m}.webp", atlas, 95)
        sp = next((f"{dd}/swatch_{m}.exr" for dd in SWATCH_DIRS if os.path.exists(f"{dd}/swatch_{m}.exr")), None)
        if sp:
            sw = cv2.imread(sp, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)[::-1]   # rows = v
            p95 = float(np.percentile(sw @ np.array([0.2126, 0.7152, 0.0722], np.float32), 95))
            gain = float(np.clip(0.6 / max(p95, 1e-6), 1.0, 1.5))     # lift dim cards a little
            swatch_gain[m] = round(gain, 3)
            sizes[f"swatch_{m}.webp"] = wr(f"swatch_{m}.webp", srgb(sw * gain), 90)
        else:
            missing.append(f"swatch {m}")
        ld = LATDEC.format(m)
        if os.path.exists(ld):
            shutil.copyfile(ld, os.path.join(DST, f"latdec_{m}.webp"))
            sizes[f"latdec_{m}.webp"] = os.path.getsize(ld)

    targets = []
    for r in REGIONS:
        label, label_with, comp, drop = TARGETS[r]
        ri = REGIONS.index(r)
        ys, xs = np.where(cov[:, :, ri] > 0.05)
        if drop is None:
            drop = deepest(cov[:, :, ri] > 0.99)
        x, y = drop
        assert cov[y, x, ri] > 0.5, (r, cov[y, x])
        # touch: a floor cushion within reach of the finger wins over the rug / bare floor under
        # it ("grow"); the rug "yields" to them; sofas and pillows keep exact hits
        t = dict(id=r, label=label, region=ri, companions=[REGIONS.index(c) for c in comp],
                 small=r.startswith(("pillow", "cush")), grow=r.startswith("cush"), yields=(r == "rug"),
                 bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], drop=drop)
        if label_with:
            t["labelWith"] = label_with
        targets.append(t)
    times = json.load(open(f"{P5}/times.json"))
    meta = dict(size=[W, H], pos_min=pmin.tolist(), pos_max=pmax.tolist(),
                regions=REGIONS, targets=targets,
                sets=[dict(st, boxes=data[st["dir"]]["boxes"]) for st in SETS],
                classes=dict(of=[cls[i] for i in range(nreg)],
                             boxes_uv=[[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in data[""]["boxes"]]),
                ratio=dict(cols=RC, tile=[LW, TH], offset=0.5, lo_res=[LW, LH]),
                materials=[dict(id=m, name=n) for m, n in MATS], preload=PRELOAD,
                # latent tile per material (js/room_core.js makeTile): swap paths here to show
                # other tile images ({"kind": "frames", "frames": [...], "fps": 8},
                # {"kind": "pca", "a": ..., "b": ...}); the card in hand decodes to swatch_<m>
                tiles={str(m): dict(kind="image", src=f"latdec_{m}.webp") for m, _ in MATS},
                passes=passes, swatches={k: v for k, v in times.items() if k.startswith("swatch_")},
                swatch_gain=swatch_gain, defirefly=stats, ratio_p1_p99=ratio_stats)
    json.dump(meta, open(os.path.join(DST, "meta.json"), "w"), indent=1)
    for f in missing:
        print("MISSING:", f)
    tot = sum(sizes.values())
    for sd in ("", "4k/"):
        per = {m: sum(v for f, v in sizes.items() if f.startswith(f"{sd}mat_{m}_") or f == f"ratio_{m}.webp")
               for m, _ in MATS}
        shared = sum(v for f, v in sizes.items() if ("/" not in f if sd == "" else True) and not f.startswith(
            ("mat_", "4k/mat_", "ratio_")) and (f.startswith(sd) if sd else "/" not in f or f.startswith("4k/") is False))
        print(f"set '{sd or '1920'}': per material {', '.join(f'{m}: {v / 1e6:.2f}' for m, v in per.items())} MB")
    print(json.dumps(sizes, indent=0))
    print(f"total {tot / 1e6:.2f} MB in {len(sizes)} files")


if __name__ == "__main__":
    main()
