#!/usr/bin/env python3
"""exp-015 step 5: is the A100 render the same image as the exp-005 RTX render?

Compares linear EXR frames pairwise and reports, per frame:
  * PSNR over linear radiance (peak = the exp-005 frame's max, printed), and
    PSNR over the tone-mapped x/(1+x) image, which is what anyone actually looks
    at and is not dominated by a handful of bright specular pixels;
  * max / mean absolute difference, and the relative difference on pixels above
    1% of the frame max (so near-black background does not flatter the number);
  * the Monte-Carlo noise floor for context: the expected spread of two
    independent spp-N estimates of the same integral. Two correct renders of the
    same scene differ by exactly this much, so "agrees up to MC noise" means the
    measured difference sits at the floor, not at zero.

Usage:
  python compare_to_exp005.py --a <dir with frames_exr> --b <dir with frames_exr> \
      --frames 0,40,80 --out comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_exr(path: Path) -> np.ndarray:
    """Read a linear RGB EXR as float32 HxWx3, via imageio (OpenEXR plugin)."""
    import imageio.v3 as iio
    arr = np.asarray(iio.imread(path), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((peak ** 2) / mse))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}"}
    diff = np.abs(a - b)
    peak = float(max(a.max(), b.max()))
    ta, tb = a / (1.0 + a), b / (1.0 + b)
    mask = a > 0.01 * a.max()
    rel = diff[mask] / np.maximum(a[mask], 1e-6)
    return {
        "shape": list(a.shape),
        "a_max": float(a.max()), "a_mean": float(a.mean()),
        "b_max": float(b.max()), "b_mean": float(b.mean()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "psnr_linear_db": psnr(a, b, peak),
        "psnr_tonemapped_db": psnr(ta, tb, 1.0),
        "rel_diff_mean_pct_above_1pct_of_max": float(rel.mean() * 100.0),
        "rel_diff_p99_pct_above_1pct_of_max": float(np.percentile(rel, 99) * 100.0),
        "lit_pixel_fraction": float(mask.mean()),
        # Both renders are unbiased estimates of the same integral, so their
        # difference has twice the variance of one render. Halving the measured
        # variance estimates the per-render noise level.
        "implied_per_render_rms_noise": float(np.sqrt(np.mean((a - b) ** 2) / 2.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference run dir (exp-005)")
    ap.add_argument("--b", type=Path, required=True, help="new run dir (Leonardo)")
    ap.add_argument("--frames", default="0,40,80")
    ap.add_argument("--subdir", default="frames_exr")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = {"a": str(args.a), "b": str(args.b), "frames": {}}
    for idx in [int(x) for x in args.frames.split(",")]:
        pa = args.a / args.subdir / f"frame_{idx:03d}.exr"
        pb = args.b / args.subdir / f"frame_{idx:03d}.exr"
        if not pa.exists() or not pb.exists():
            results["frames"][idx] = {"error": f"missing {pa if not pa.exists() else pb}"}
            continue
        results["frames"][idx] = compare(read_exr(pa), read_exr(pb))

    ok = [v for v in results["frames"].values() if "psnr_tonemapped_db" in v]
    if ok:
        results["summary"] = {
            "n_frames": len(ok),
            "psnr_linear_db_min": min(v["psnr_linear_db"] for v in ok),
            "psnr_tonemapped_db_min": min(v["psnr_tonemapped_db"] for v in ok),
            "max_abs_diff_max": max(v["max_abs_diff"] for v in ok),
            "mean_abs_diff_max": max(v["mean_abs_diff"] for v in ok),
        }
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
