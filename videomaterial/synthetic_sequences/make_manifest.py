#!/usr/bin/env python3
"""Build a dataset manifest (exp-015): 900 train + 100 test sequences.

    python make_manifest.py --out .../B1_Fixed_camera/manifest.json
    python make_manifest.py --setup B2 --master-seed 20260923 --out .../B2_Fixed_light/manifest.json
    python make_manifest.py --setup B3 --master-seed 20261001 --out .../B3_Joint/manifest.json \
        --light-manifest .../B1_Fixed_camera/manifest.json --camera-manifest .../B2_Fixed_light/manifest.json

B3 (jointly moving camera and light) is built from the B1 and B2 manifests, see ``build_b3``.

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

DEFAULT_MASTER_SEED = 20260922                           # B1 (and the historical CLI default for B2)
B3_DEFAULT_MASTER_SEED = 20261001                        # B3 pairing (requested by Zhen 2026-10-01)
B3_TEST_GROUPS = {"unseen_light": 34, "unseen_camera": 33, "unseen_pair": 33}
B3_TRAIN_GROUP = "pair"


def git_info(repo):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"repo": str(repo), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def render_block():
    """Render settings shared by every setup's manifest (material 64 spp + PBR-patch twin 32 spp)."""
    return {"material": {"mode": "material", "spp": 64, "batch_spp": 16, "png_dir": "frames_png", "exr_dir": "frames_exr"},
            "pbr_patch": {"mode": "pbr_patch", "spp": 32, "batch_spp": 16, "png_dir": "pbr_patch_png", "exr_dir": None},
            "max_depth": 4, "intensity": 20.0}


def build(master_seed=DEFAULT_MASTER_SEED, frames=81, verbose=True, setup="B1"):
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
        "render": render_block(),
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


# ============================================================================ B3: joint camera + light
def _source_ref(seq, man):
    """Compact reference to a sequence of a source (B1/B2) manifest: enough to regenerate its poses."""
    return {"setup": man["setup"], "setup_dir": man["setup_dir"], "dir": seq["dir"], "id": seq["id"],
            "split": seq["split"], "class": seq["class"], "seed": seq["seed"], "manifest_index": seq["index"]}


def _draw(rng, n, pool_size):
    """n indices uniformly without replacement where the pool allows (with replacement otherwise)."""
    return [int(i) for i in rng.choice(pool_size, n, replace=n > pool_size)]


def _mirror_offset_deg(light_dirs, cam_dirs):
    """Angle between the camera direction and the mirror direction of the light about +Y (0 = the
    camera sits exactly in the specular lobe's centre)."""
    m = light_dirs * np.array([-1.0, 1.0, -1.0])
    return np.degrees(np.arccos(np.clip((m * cam_dirs).sum(-1), -1.0, 1.0)))


def _half_theta_deg(light_dirs, cam_dirs):
    h = light_dirs + cam_dirs
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(h[..., 1], -1.0, 1.0)))


def _geometry_stats(pairs_dirs):
    """Distribution of the relative light/camera geometry over all frames of a set of sequences."""
    if not pairs_dirs:
        return None
    off = np.stack([_mirror_offset_deg(l, c) for l, c in pairs_dirs])        # (S, F)
    th_h = np.stack([_half_theta_deg(l, c) for l, c in pairs_dirs])
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)}
    return {"sequences": len(pairs_dirs), "frames": int(off.size),
            "mirror_offset_deg": pct(off), "half_vector_theta_deg": pct(th_h),
            "frac_frames_mirror_offset_lt": {f"{t}deg": float((off < t).mean()) for t in (5, 10, 20)},
            "frac_sequences_min_mirror_offset_lt": {f"{t}deg": float((off.min(1) < t).mean()) for t in (5, 10, 20)}}


def _joint_coverage(pairs_dirs, lspec, cspec, n_rings=4, n_sectors=8):
    """Frames per (light cell x camera cell) on a coarse 4x8 equal-area grid per element (1,024 joint cells)."""
    n = n_rings * n_sectors
    cnt = np.zeros(n * n)
    for l, c in pairs_dirs:
        jid = T.cell_ids(l, lspec, n_rings, n_sectors) * n + T.cell_ids(c, cspec, n_rings, n_sectors)
        cnt += np.bincount(jid, minlength=n * n)
    return {"cells": n * n, "grid_per_element": f"{n_rings}x{n_sectors}", "empty": int((cnt == 0).sum()),
            "min": int(cnt.min()), "median": float(np.median(cnt)), "max": int(cnt.max()),
            "cv": float(cnt.std() / max(cnt.mean(), 1e-9))}


def b3_statistics(sequences, lspec, cspec, frames):
    """Coverage of light x camera directions and the distinct source paths each group uses."""
    cache = {}

    def dirs(src, spec):
        key = (src["setup"], src["class"], src["seed"])
        if key not in cache:
            cache[key] = T.sample_trajectory(src["class"], src["seed"], spec).directions
        return cache[key]

    def pairs(sel):
        return [(dirs(s["light"], lspec), dirs(s["camera"], cspec)) for s in sel]

    by_group = {}
    for s in sequences:
        by_group.setdefault(s["group"], []).append(s)
    train = by_group.get(B3_TRAIN_GROUP, [])
    test = [s for s in sequences if s["split"] == "test"]
    stats = {
        "light_coverage_192_cells": {"train": T.coverage_stats([dirs(s["light"], lspec) for s in train], lspec),
                                     "test": T.coverage_stats([dirs(s["light"], lspec) for s in test], lspec)},
        "camera_coverage_192_cells": {"train": T.coverage_stats([dirs(s["camera"], cspec) for s in train], cspec),
                                      "test": T.coverage_stats([dirs(s["camera"], cspec) for s in test], cspec)},
        "joint_coverage": {"train": _joint_coverage(pairs(train), lspec, cspec),
                           "test": _joint_coverage(pairs(test), lspec, cspec)},
        "relative_geometry": {g: _geometry_stats(pairs(sel)) for g, sel in by_group.items()},
        "distinct_paths": {},
    }
    stats["relative_geometry"]["test_all"] = _geometry_stats(pairs(test))
    # the fixed-element datasets for reference: B1 = moving light, camera fixed; B2 = the reverse
    lfix = np.repeat(T.ang_to_vec(cspec.fixed_theta_deg, cspec.fixed_phi_deg)[None], frames, 0)
    cfix = np.repeat(T.ang_to_vec(lspec.fixed_theta_deg, lspec.fixed_phi_deg)[None], frames, 0)
    stats["relative_geometry"]["reference_B1_train_lights_fixed_camera"] = _geometry_stats(
        [(dirs(s["light"], lspec), cfix) for s in train])
    stats["relative_geometry"]["reference_B2_train_cameras_fixed_light"] = _geometry_stats(
        [(lfix, dirs(s["camera"], cspec)) for s in train])
    for g, sel in list(by_group.items()) + [("test_all", test)]:
        d = {}
        for el in ("light", "camera"):
            for sp in ("train", "test"):
                used = [s[el]["dir"] for s in sel if s[el]["split"] == sp]
                if used:
                    d[f"{el}_{sp}_paths"] = {"distinct": len(set(used)), "uses": len(used)}
            cls = {}
            for s in sel:
                cls[s[el]["class"]] = cls.get(s[el]["class"], 0) + 1
            d[f"{el}_classes"] = dict(sorted(cls.items()))
        stats["distinct_paths"][g] = d
    return stats


def build_b3(light_manifest, camera_manifest, master_seed=B3_DEFAULT_MASTER_SEED, verbose=True,
             stats=True, test_groups=None, source_files=None):
    """B3 = uniform random pairs of a B1 light path and a B2 camera path (Zhen, 2026-09-24).

    Frame k of a B3 sequence has the light at frame k of its B1 sequence and the camera pose of frame
    k of its B2 sequence. Train: a uniformly random one-to-one pairing of the B1 train light paths with
    the B2 train camera paths (each used exactly once). Test, three groups: ``unseen_light`` (B1 TEST
    light x B2 TRAIN camera), ``unseen_camera`` (B1 TRAIN light x B2 TEST camera), ``unseen_pair``
    (both from the TRAIN pools, a pair that is not in the training set). Test paths are drawn
    uniformly without replacement inside each group. One rng, ``default_rng(master_seed)``, draws in a
    fixed order (train permutation; unseen_light light then camera; unseen_camera light then camera;
    unseen_pair light then camera, the camera draw repeated while any pair is a training pair), so
    the manifest is deterministic given the two source manifests and ``master_seed``."""
    js = L.JOINT_SETUPS["B3"]
    lm, cm = light_manifest, camera_manifest
    if lm["setup"] != js["light"] or cm["setup"] != js["camera"]:
        raise ValueError(f"B3 needs a {js['light']} light manifest and a {js['camera']} camera manifest; "
                         f"got {lm['setup']} / {cm['setup']}")
    for k in ("frames", "fps", "resolution"):
        if lm[k] != cm[k]:
            raise ValueError(f"source manifests disagree on {k}: {lm[k]} vs {cm[k]}")
    frames = lm["frames"]
    groups = dict(B3_TEST_GROUPS if test_groups is None else test_groups)
    l_tr = [s for s in lm["sequences"] if s["split"] == "train"]
    l_te = [s for s in lm["sequences"] if s["split"] == "test"]
    c_tr = [s for s in cm["sequences"] if s["split"] == "train"]
    c_te = [s for s in cm["sequences"] if s["split"] == "test"]
    if len(l_tr) != len(c_tr):
        raise ValueError(f"one-to-one train pairing needs equal pools; got {len(l_tr)} lights, {len(c_tr)} cameras")

    rng = np.random.default_rng(master_seed)
    perm = [int(i) for i in rng.permutation(len(c_tr))]
    train_pairs = [(l_tr[i], c_tr[perm[i]]) for i in range(len(l_tr))]
    train_keys = {(l["dir"], c["dir"]) for l, c in train_pairs}

    test_pairs = {}
    n = groups["unseen_light"]
    li, ci = _draw(rng, n, len(l_te)), _draw(rng, n, len(c_tr))
    test_pairs["unseen_light"] = [(l_te[a], c_tr[b]) for a, b in zip(li, ci)]
    n = groups["unseen_camera"]
    li, ci = _draw(rng, n, len(l_tr)), _draw(rng, n, len(c_te))
    test_pairs["unseen_camera"] = [(l_tr[a], c_te[b]) for a, b in zip(li, ci)]
    n = groups["unseen_pair"]
    li = _draw(rng, n, len(l_tr))
    redraws = 0
    while True:
        ci = _draw(rng, n, len(c_tr))
        cand = [(l_tr[a], c_tr[b]) for a, b in zip(li, ci)]
        if not any((l["dir"], c["dir"]) in train_keys for l, c in cand):
            break
        redraws += 1
        if redraws > 10000:
            raise RuntimeError("could not draw unseen pairs outside the training pairing")
    test_pairs["unseen_pair"] = cand

    sequences = []
    for k, (l, c) in enumerate(train_pairs):
        sid = f"{B3_TRAIN_GROUP}_{k+1:04d}"
        sequences.append({"index": k, "split": "train", "group": B3_TRAIN_GROUP, "class": B3_TRAIN_GROUP,
                          "id": sid, "dir": f"train/{sid}", "keep_exr": k % EXR_EVERY_N_TRAIN == 0,
                          "light": _source_ref(l, lm), "camera": _source_ref(c, cm)})
    for g in B3_TEST_GROUPS:
        for i, (l, c) in enumerate(test_pairs[g]):
            sid = f"{g}_{i+1:04d}"
            sequences.append({"index": len(sequences), "split": "test", "group": g, "class": g,
                              "id": sid, "dir": f"test/{sid}", "keep_exr": True,
                              "light": _source_ref(l, lm), "camera": _source_ref(c, cm)})

    lspec = L.SETUPS[js["light"]][1](frames=frames)
    cspec = L.SETUPS[js["camera"]][1](frames=frames)
    t0 = time.time()
    statistics = b3_statistics(sequences, lspec, cspec, frames) if stats else None
    if verbose and stats:
        print(f"pairing statistics in {time.time()-t0:.1f}s", flush=True)

    def src_info(man, role):
        info = {"setup": man["setup"], "setup_dir": man["setup_dir"], "master_seed": man["master_seed"],
                "manifest_created_utc": man.get("created_utc"),
                "manifest_generator_commit": man.get("generator", {}).get("commit"),
                "train": sum(s["split"] == "train" for s in man["sequences"]),
                "test": sum(s["split"] == "test" for s in man["sequences"])}
        if source_files and role in source_files:
            info.update(source_files[role])
        return info

    manifest = {
        "schema_version": 1,
        "setup": "B3", "setup_dir": js["setup_dir"],
        "description": ("B3 dataset: jointly moving point light AND camera. Frame k of every sequence has the light "
                        "at frame k of a B1_Fixed_camera sequence and the camera pose of frame k of a B2_Fixed_light "
                        "sequence (uniform random pairs, both paths at their native speed). 832x480, 81 frames at "
                        "15 fps, fov_y 35 deg, camera and light radius 3.0, intensity 20. Every sequence is rendered "
                        "twice: material 314 (frames_png, frames_exr when keep_exr) and the homogeneous PBR-patch "
                        "twin (pbr_patch_png)."),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {**git_info(HERE.parents[1]), "script": str(Path(__file__).resolve()),
                      "python": platform.python_version(), "numpy": np.__version__},
        "master_seed": master_seed,
        "frames": frames, "fps": lm["fps"], "resolution": lm["resolution"],
        "sources": {"light": src_info(lm, "light"), "camera": src_info(cm, "camera")},
        "trajectory_spec": {"light": {k: v for k, v in vars(lspec).items() if not k.startswith("_")},
                            "camera": {k: v for k, v in vars(cspec).items() if not k.startswith("_")}},
        "pairing": {"rule": "frame k: light = frame k of the B1 light path, camera = frame k of the B2 camera path",
                    "train": f"uniform random one-to-one pairing of the {len(l_tr)} B1 train light paths with the "
                             f"{len(c_tr)} B2 train camera paths (each path used exactly once)",
                    "test_groups": {"unseen_light": "B1 TEST light path x B2 TRAIN camera path",
                                    "unseen_camera": "B1 TRAIN light path x B2 TEST camera path",
                                    "unseen_pair": "B1 TRAIN light x B2 TRAIN camera, pair not in the training set"},
                    "test_group_sizes": groups,
                    "draws": "uniform without replacement inside each group",
                    "rng": "numpy.random.default_rng(master_seed); order: train permutation, unseen_light (light, "
                           "camera), unseen_camera (light, camera), unseen_pair (light, camera redrawn while any "
                           "pair is a training pair)",
                    "unseen_pair_camera_redraws": redraws,
                    "statistics": statistics},
        "storage": {"material_png": "always", "material_exr": f"every {EXR_EVERY_N_TRAIN}th train sequence and all test sequences",
                    "pbr_patch_png": "always", "pbr_patch_exr": "never"},
        "twin": "pbr_patch",
        "render": render_block(),
        "counts": {"total": len(sequences),
                   "train": sum(s["split"] == "train" for s in sequences),
                   "test": sum(s["split"] == "test" for s in sequences),
                   "keep_exr": sum(s["keep_exr"] for s in sequences),
                   "by_split_group": {sp: {g: sum(s["split"] == sp and s["group"] == g for s in sequences)
                                           for g in [B3_TRAIN_GROUP, *B3_TEST_GROUPS]
                                           if any(s["split"] == sp and s["group"] == g for s in sequences)}
                                      for sp in ("train", "test")}},
        "sequences": sequences,
    }
    return manifest


def _print_b3(m, out):
    c = m["counts"]
    print(f"\n{c['total']} sequences -> {out}")
    print(f"  train {c['train']}  test {c['test']}  material EXR kept for {c['keep_exr']}")
    for sp, d in c["by_split_group"].items():
        print(f"  {sp}: " + ", ".join(f"{k} {v}" for k, v in d.items()))
    st = m["pairing"]["statistics"]
    if not st:
        return
    print(f"  unseen_pair camera redraws: {m['pairing']['unseen_pair_camera_redraws']}")
    for key in ("light_coverage_192_cells", "camera_coverage_192_cells", "joint_coverage"):
        print(f"\n{key}:")
        print(f"  {'set':8s} {'cells':>6s} {'empty':>6s} {'min':>6s} {'median':>8s} {'max':>6s} {'cv':>6s}")
        for k, v in st[key].items():
            print(f"  {k:8s} {v['cells']:6d} {v['empty']:6d} {v['min']:6d} {v['median']:8.0f} {v['max']:6d} {v['cv']:6.3f}")
    print("\nrelative geometry (mirror offset = angle camera <-> mirror of light):")
    for g, v in st["relative_geometry"].items():
        if v:
            print(f"  {g:40s} median offset {v['mirror_offset_deg']['p50']:6.1f} deg  frames<10deg "
                  f"{v['frac_frames_mirror_offset_lt']['10deg']:.3f}  seqs min<10deg "
                  f"{v['frac_sequences_min_mirror_offset_lt']['10deg']:.3f}")
    print("\ndistinct source paths:")
    for g, d in st["distinct_paths"].items():
        print(f"  {g:14s} " + "  ".join(f"{k}={v['distinct']}/{v['uses']}" for k, v in d.items() if k.endswith("_paths")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--master-seed", type=int, default=None,
                    help=f"default {DEFAULT_MASTER_SEED} for B1/B2, {B3_DEFAULT_MASTER_SEED} for B3")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--setup", choices=list(L.SETUPS) + list(L.JOINT_SETUPS), default="B1")
    ap.add_argument("--light-manifest", type=Path, default=None, help="B3: the B1_Fixed_camera manifest.json")
    ap.add_argument("--camera-manifest", type=Path, default=None, help="B3: the B2_Fixed_light manifest.json")
    args = ap.parse_args()

    if args.setup in L.JOINT_SETUPS:
        if args.light_manifest is None or args.camera_manifest is None:
            ap.error("--setup B3 needs --light-manifest and --camera-manifest")
        files = {role: {"manifest_path": str(p.resolve()), "manifest_sha256": L.sha256(p)}
                 for role, p in (("light", args.light_manifest), ("camera", args.camera_manifest))}
        m = build_b3(json.loads(args.light_manifest.read_text()), json.loads(args.camera_manifest.read_text()),
                     B3_DEFAULT_MASTER_SEED if args.master_seed is None else args.master_seed, source_files=files)
        if m["frames"] != args.frames:
            ap.error(f"source manifests have {m['frames']} frames, --frames says {args.frames}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2) + "\n")
        _print_b3(m, args.out)
        return 0

    m = build(DEFAULT_MASTER_SEED if args.master_seed is None else args.master_seed, args.frames, setup=args.setup)
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
