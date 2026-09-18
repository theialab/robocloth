#!/usr/bin/env python3
"""
Precompute per-material BRDF stats (per-channel min/max + mean colour)
for the RoboCloth preview materials.

Why
---
The webpage previously computed these stats per-frame off the canvas
(8-bit decoded), which gave a different colour and range every time
the slider moved. The user wants ONE set of values per material that
represents the whole capture.

What it does
------------
For each material:
1. Streams `hdr.tar` from HuggingFace.
2. Decodes each 16-bit RGB PNG.
3. Projects the **shrunken** material rectangle (bbox minus 5 mm per
   side = 1 cm inset) into image space using rotated_camera.json and
   the SIMPLE_RADIAL intrinsics from hdr_crop_bboxes.json.
4. Masks the image to the projected polygon, drops any (0,0,0)
   left-over from the boundary, and accumulates per-channel
   min / max / sum / count in 16-bit linear space.
5. Writes `webpage/data/material_stats/<id>.json`.

Per-pixel cost is ~150 ms (PNG decode dominates), so a full material
with ~580 images is ~2 min. Sample-folder materials (12 images) are
seconds.

Usage
-----
    python3 webpage/tools/build_material_stats.py             # all 20
    python3 webpage/tools/build_material_stats.py --ids 156 249
    python3 webpage/tools/build_material_stats.py --force     # rebuild existing
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    import requests
except ImportError:
    print("Needs `pip install opencv-python requests`")
    sys.exit(1)


HF = "https://huggingface.co/datasets/koalapenguin/RoboCloth/resolve/main"

DEFAULT_IDS = [2, 8, 20, 35, 50, 84, 96, 109, 130, 145, 156, 170, 190,
               220, 249, 309, 350, 400, 428, 483]
SAMPLE_IDS  = {156, 249, 309, 428, 483}


# ---------------------------------------------------------------------------
# Projection helpers — mirror webpage/js/transforms.js::makeWorld0Projector
# ---------------------------------------------------------------------------

def project_world0(P_world_m, cam_entry, intr):
    """
    Project a world0 point (metres) to image pixel (u, v).

    rotated_camera.json stores camera-to-world0 in OpenGL convention with
    the Umeyama scale `s` baked into the rotation matrix. We extract `s`
    from the column norm, divide it out, flip OpenGL→OpenCV camera axes,
    and apply the SIMPLE_RADIAL distortion model.

    Returns None if the point is behind the camera or the rotation is
    degenerate.
    """
    R = np.asarray(cam_entry["rotation_matrix"], dtype=np.float64)
    s = np.linalg.norm(R[:, 0])
    if s < 1e-12:
        return None
    Rn = R / s
    t_m = np.asarray(cam_entry["position"], dtype=np.float64) / 1000.0

    # w2c rotation = Rn^T, translation = -Rn^T · t
    RT = Rn.T
    wt = -RT @ t_m

    p_cam_gl = RT @ P_world_m + wt
    X, Y, Z = p_cam_gl[0], -p_cam_gl[1], -p_cam_gl[2]  # OpenGL→OpenCV
    if Z <= 1e-9:
        return None

    x_n = X / Z
    y_n = Y / Z
    r2 = x_n * x_n + y_n * y_n
    dist = 1.0 + float(intr["distortion"]) * r2
    u = float(intr["focal_length"]) * dist * x_n + float(intr["cx"])
    v = float(intr["focal_length"]) * dist * y_n + float(intr["cy"])
    return (u, v)


def make_polygon_mask(shape, polygon):
    """uint8 mask of pixels inside the polygon."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------

def fetch_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def stream_tar(tar_url):
    """
    Open the remote tar as a stream, handling HF's 429 rate-limit with
    exponential backoff. Returns the response so callers can use
    `with response: ...`.
    """
    sess = requests.Session()
    delay = 5
    while True:
        try:
            r = sess.get(tar_url, stream=True, allow_redirects=True, timeout=600)
            if r.status_code == 429:
                # Honour `ratelimit: r=N;t=secs` if present
                hdr = r.headers.get("ratelimit", "")
                m = re.search(r"t=(\d+)", hdr)
                wait = max(int(m.group(1)) + 2 if m else 30, delay)
                print(f"   429 throttled — sleep {wait}s", flush=True)
                r.close()
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise


# ---------------------------------------------------------------------------
# Per-material pipeline
# ---------------------------------------------------------------------------

CAM_RE = re.compile(r"camera-(\d+)\.png$")


def process_material(mid, sample, shrink_m, out_dir):
    folder = f"sample/material_{mid}" if sample else f"materials/{mid}"
    label = f"[{mid:>4}{' SAMPLE' if sample else '       '}]"
    print(f"{label} starting", flush=True)

    # Load metadata
    bbox   = fetch_json(f"{HF}/{folder}/bbox.json")
    rotcam = fetch_json(f"{HF}/{folder}/rotated_camera.json")
    crop   = fetch_json(f"{HF}/{folder}/hdr_crop_bboxes.json")
    intr   = crop["intrinsics"]
    rot_index = {int(c["camera_id"]): c for c in rotcam}

    # Inset top-surface corners in world0
    mn = bbox["bbox_min"]; mx = bbox["bbox_max"]
    cx = (mn[0] + mx[0]) / 2.0
    cy = (mn[1] + mx[1]) / 2.0
    w_half = max(0.005, (mx[0] - mn[0]) / 2.0 - shrink_m)
    l_half = max(0.005, (mx[1] - mn[1]) / 2.0 - shrink_m)
    z      = mx[2]
    corners3d = np.array([
        [cx - w_half, cy - l_half, z],
        [cx + w_half, cy - l_half, z],
        [cx + w_half, cy + l_half, z],
        [cx - w_half, cy + l_half, z],
    ])

    # Accumulators (R, G, B order).
    #
    # We keep:
    #   - per-channel running min / max (absolute extremes; informative
    #     for debugging but very noisy — even a single dark or saturated
    #     pixel sets these)
    #   - per-channel sum + count (for the mean)
    #   - per-channel histogram across the 0..65535 capture range
    #     (used to derive robust 0.5 / 99.5 percentiles as the "range"
    #     displayed in the UI; immune to one-pixel outliers).
    minRGB = np.array([65535.0, 65535.0, 65535.0])
    maxRGB = np.array([0.0, 0.0, 0.0])
    sumRGB = np.array([0.0, 0.0, 0.0])
    histRGB = np.zeros((3, 65536), dtype=np.int64)
    count   = 0
    frames  = 0
    skipped = 0

    tar_url = f"{HF}/{folder}/hdr.tar"
    t0 = time.time()
    with stream_tar(tar_url) as r:
        tf = tarfile.open(fileobj=r.raw, mode="r|", bufsize=4 * 1024 * 1024)
        for member in tf:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            m = CAM_RE.search(member.name)
            if not m:
                continue
            camera_id = int(m.group(1))

            # Always read the data to advance the tar pointer
            data_obj = tf.extractfile(member)
            if data_obj is None:
                continue
            png_bytes = data_obj.read()

            cam_entry = rot_index.get(camera_id)
            if cam_entry is None:
                skipped += 1
                continue

            # Decode 16-bit PNG (cv2 returns BGR for 3-channel; flip to RGB)
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None or img.dtype != np.uint16:
                skipped += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Project + mask
            pixels = [project_world0(p, cam_entry, intr) for p in corners3d]
            if any(p is None for p in pixels):
                skipped += 1
                continue
            mask = make_polygon_mask(img.shape, pixels)
            inside = img[mask == 1]                                 # N×3
            if inside.size == 0:
                skipped += 1
                continue
            # Drop fully-zero leftovers (boundary mask leakage)
            nonzero = inside.sum(axis=1) > 0
            inside = inside[nonzero]
            if inside.size == 0:
                skipped += 1
                continue

            ch_min = inside.min(axis=0).astype(np.float64)
            ch_max = inside.max(axis=0).astype(np.float64)
            ch_sum = inside.sum(axis=0).astype(np.float64)
            minRGB = np.minimum(minRGB, ch_min)
            maxRGB = np.maximum(maxRGB, ch_max)
            sumRGB += ch_sum
            # Per-channel histogram for percentile-based range later.
            for ch in range(3):
                histRGB[ch] += np.bincount(inside[:, ch].astype(np.int64),
                                           minlength=65536)
            count  += len(inside)
            frames += 1

            if frames % 50 == 0:
                print(f"{label}   {frames} frames in {time.time()-t0:.0f}s", flush=True)
        tf.close()

    if count == 0:
        print(f"{label} no valid pixels", flush=True)
        return None

    mean_rgb = (sumRGB / count).tolist()

    # Per-channel percentiles from the histogram. These define the
    # "range bar" shown in the UI — robust to one-pixel outliers (a
    # single saturated specular highlight or a stray dark-mask leak
    # doesn't blow the bar to 0..65535).
    def pct(hist, p):
        cs = np.cumsum(hist)
        thresh = cs[-1] * p
        return int(np.searchsorted(cs, thresh))

    lo_rgb = [pct(histRGB[c], 0.005) for c in range(3)]
    hi_rgb = [pct(histRGB[c], 0.995) for c in range(3)]

    result = {
        "id":             mid,
        # 0.5 / 99.5 percentiles — what the UI bars actually display.
        "lo_rgb_16bit":   lo_rgb,
        "hi_rgb_16bit":   hi_rgb,
        # Mean colour (sRGB-encoded in JS for display).
        "mean_rgb_16bit": mean_rgb,
        # Absolute extremes — kept for debugging / future use; not shown.
        "min_rgb_16bit":  [int(v) for v in minRGB.tolist()],
        "max_rgb_16bit":  [int(v) for v in maxRGB.tolist()],
        "pixel_count":    int(count),
        "frames_used":    int(frames),
        "frames_skipped": int(skipped),
        "shrink_m":       shrink_m,
        "source":         "sample" if sample else "full",
    }

    out_path = out_dir / f"{mid}.json"
    out_path.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - t0
    print(f"{label} DONE {elapsed:.1f}s frames={frames} skip={skipped} "
          f"lo={lo_rgb} hi={hi_rgb} "
          f"mean={[int(v) for v in mean_rgb]}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--samples", nargs="*", type=int, default=sorted(SAMPLE_IDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shrink-mm", type=float, default=5.0,
                    help="Per-side inset in mm (default 5mm = 1cm shrink per dim).")
    ap.add_argument("--out", default="webpage/data/material_stats")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids = sorted(set(args.ids if args.ids is not None else DEFAULT_IDS))
    samples = set(args.samples)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for mid in ids:
        out_path = out_dir / f"{mid}.json"
        if out_path.exists() and not args.force:
            print(f"[{mid:>4}] exists, skipping (use --force to rebuild)")
        else:
            todo.append(mid)
    if not todo:
        print("Nothing to do.")
        return

    print(f"Processing {len(todo)} materials with {args.workers} workers, "
          f"inset = {args.shrink_mm} mm per side")
    t0 = time.time()
    succ = fail = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_material, mid, mid in samples,
                      args.shrink_mm / 1000.0, out_dir): mid
            for mid in todo
        }
        for fut in cf.as_completed(futs):
            mid = futs[fut]
            try:
                if fut.result():
                    succ += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"[{mid:>4}] FAILED: {e}")
                import traceback; traceback.print_exc()
                fail += 1

    print(f"\nDone in {time.time()-t0:.0f}s — {succ} OK, {fail} failed")


if __name__ == "__main__":
    main()
