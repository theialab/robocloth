#!/usr/bin/env python3
"""Build the B1 dataset manifest (exp-015): 900 train + 100 test sequences.

    python make_manifest.py --out .../B1_Fixed_camera/manifest.json

Deterministic given ``--master-seed``: every trajectory seed is a fixed offset from it, so the
same command always produces the same 1,000 sequences.  Train paths are chosen by
``trajectories.greedy_select`` (coverage-maximising, 85 % spline / 15 % highlight_sweep) out of a
pool of 3,000 spline + 500 highlight_sweep candidates; the test split uses seeds from disjoint
blocks, so no test trajectory can have been seen in training.

Sequence directories are ``<split>/<class>_<NNNN>`` — no seeds and no dates in names (SPEC section 1);
the seed of each sequence lives in the manifest and in the sequence's ``metadata.json``.
"""
from __future__ import annotations

import argparse, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_sequences import trajectories as T
from synthetic_sequences import seqlib as L  # noqa: E402

# Disjoint seed blocks. A block base is 1e8 apart so adding the master seed can never make two
# blocks overlap; test spline/highlight seeds therefore never coincide with a training seed.
BLOCK = {"train_spline": 100_000_000, "train_highlight_sweep": 200_000_000,
         "test_spline": 300_000_000, "test_highlight_sweep": 400_000_000,
         "test_ring": 500_000_000, "test_spiral": 600_000_000,
         "test_lissajous": 700_000_000, "test_spiral_reference": 800_000_000}

POOL = {"spline": 3000, "highlight_sweep": 500}          # candidates sampled for the train selection
TRAIN_QUOTAS = {"spline": 765, "highlight_sweep": 135}   # 85 % / 15 % of 900
TEST_COUNTS = {"spline": 30, "highlight_sweep": 10, "ring": 20, "spiral": 20,
               "lissajous": 19, "spiral_reference": 1}
EXR_EVERY_N_TRAIN = 10                                   # material EXR kept for every 10th train sequence


def git_info(repo):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"repo": str(repo), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def build(master_seed=20260922, frames=81, verbose=True, setup="B1"):
    setup_dir, spec_fn = L.SETUPS[setup]
    spec = spec_fn(frames=frames)
    t0 = time.time()

    # ---- train: sample the candidate pool, then pick 900 by greedy coverage
    pool, pool_seeds = [], []
    for cls, n in POOL.items():
        base = BLOCK[f"train_{cls}"] + master_seed
        for i in range(n):
            seed = base + i
            pool.append((cls, T.sample_trajectory(cls, seed, spec).directions))
            pool_seeds.append(seed)
    if verbose:
        print(f"pool: {len(pool)} candidates sampled in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    chosen = T.greedy_select(pool, spec, sum(TRAIN_QUOTAS.values()), quotas=TRAIN_QUOTAS)
    if len(chosen) != sum(TRAIN_QUOTAS.values()):
        raise RuntimeError(f"greedy_select returned {len(chosen)} of {sum(TRAIN_QUOTAS.values())}")
    if verbose:
        print(f"greedy_select: {len(chosen)} of {len(pool)} in {time.time()-t1:.1f}s", flush=True)

    # coverage of the chosen set vs a random draw with the same class quotas, and vs the whole pool
    rng = np.random.default_rng(master_seed)
    by_cls = {c: [i for i, (cc, _) in enumerate(pool) if cc == c] for c in POOL}
    rand = [int(i) for c, k in TRAIN_QUOTAS.items() for i in rng.choice(by_cls[c], k, replace=False)]
    cov = {"greedy": T.coverage_stats([pool[i][1] for i in chosen], spec),
           "random_baseline": T.coverage_stats([pool[i][1] for i in rand], spec),
           "pool_all": T.coverage_stats([v for _, v in pool], spec)}

    sequences, counters = [], {}
    for k, idx in enumerate(chosen):
        cls = pool[idx][0]
        counters[cls] = counters.get(cls, 0) + 1
        sequences.append({"index": k, "split": "train", "class": cls, "seed": pool_seeds[idx],
                          "id": f"{cls}_{counters[cls]:04d}", "dir": f"train/{cls}_{counters[cls]:04d}",
                          "keep_exr": k % EXR_EVERY_N_TRAIN == 0})

    # ---- test: fixed counts per class, seeds from blocks the training pool never touches
    n_train = len(sequences)
    for cls, n in TEST_COUNTS.items():
        base = BLOCK[f"test_{cls}"] + master_seed
        for i in range(n):
            seed = base + i
            T.sample_trajectory(cls, seed, spec)      # fail here, not on the cluster
            k = len(sequences)
            sequences.append({"index": k, "split": "test", "class": cls, "seed": seed,
                              "id": f"{cls}_{i+1:04d}", "dir": f"test/{cls}_{i+1:04d}", "keep_exr": True})
    cov["test"] = T.coverage_stats(
        [T.sample_trajectory(s["class"], s["seed"], spec).directions for s in sequences[n_train:]], spec)

    manifest = {
        "schema_version": 1,
        "setup": setup, "setup_dir": setup_dir,
        "description": (f"exp-015 {setup} dataset: " +
                        ("fixed camera (theta 45 deg, phi 90 deg, r 3.0, fov_y 35 deg), moving point light" if setup == "B1"
                         else "fixed point light (theta 45 deg, phi 90 deg, r 3.0), moving camera (fov_y 35 deg)") +
                        ", 832x480, 81 frames at 15 fps. Every sequence is rendered twice: material 314 "
                        "(frames_png, frames_exr when keep_exr) and the homogeneous PBR-patch twin (pbr_patch_png)."),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {**git_info(HERE.parents[1]), "script": str(Path(__file__).resolve()),
                      "python": platform.python_version(), "numpy": np.__version__},
        "master_seed": master_seed,
        "frames": frames, "fps": 15, "resolution": [832, 480],
        "trajectory_spec": {k: v for k, v in vars(spec).items() if not k.startswith("_")},
        "selection": {"method": "greedy maximisation of sum log(1+count) over 8x24 equal-area cells "
                                "(trajectories.greedy_select), per-class quotas",
                      "cells": {"rings": 8, "sectors": 24},
                      "pool": POOL, "train_quotas": TRAIN_QUOTAS, "test_counts": TEST_COUNTS,
                      "seed_blocks": {k: v + master_seed for k, v in BLOCK.items()},
                      "coverage_stats": cov},
        "storage": {"material_png": "always", "material_exr": f"every {EXR_EVERY_N_TRAIN}th train sequence and all test sequences",
                    "pbr_patch_png": "always", "pbr_patch_exr": "never"},
        "twin": "pbr_patch",
        "render": {"material": {"mode": "material", "spp": 64, "batch_spp": 16, "png_dir": "frames_png", "exr_dir": "frames_exr"},
                   "pbr_patch": {"mode": "pbr_patch", "spp": 32, "batch_spp": 16, "png_dir": "pbr_patch_png", "exr_dir": None},
                   "max_depth": 4, "intensity": 20.0},
        "counts": {"total": len(sequences),
                   "train": sum(s["split"] == "train" for s in sequences),
                   "test": sum(s["split"] == "test" for s in sequences),
                   "keep_exr": sum(s["keep_exr"] for s in sequences),
                   "by_split_class": {sp: {c: sum(s["split"] == sp and s["class"] == c for s in sequences)
                                           for c in T.ALL_CLASSES
                                           if any(s["split"] == sp and s["class"] == c for s in sequences)}
                                      for sp in ("train", "test")}},
        "sequences": sequences,
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-seed", type=int, default=20260922)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--setup", choices=list(L.SETUPS), default="B1")
    args = ap.parse_args()

    m = build(args.master_seed, args.frames, setup=args.setup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=2) + "\n")

    c = m["counts"]
    print(f"\n{c['total']} sequences -> {args.out}")
    print(f"  train {c['train']}  test {c['test']}  material EXR kept for {c['keep_exr']}")
    for sp, d in c["by_split_class"].items():
        print(f"  {sp}: " + ", ".join(f"{k} {v}" for k, v in d.items()))
    print("\ncoverage over 8x24 = 192 equal-area cells (81 poses per sequence):")
    print(f"  {'set':16s} {'empty':>6s} {'min':>6s} {'median':>8s} {'max':>6s} {'cv':>6s}")
    for k, v in m["selection"]["coverage_stats"].items():
        print(f"  {k:16s} {v['empty']:6d} {v['min']:6d} {v['median']:8.0f} {v['max']:6d} {v['cv']:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
