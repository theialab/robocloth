#!/usr/bin/env python
"""Offline stand-in for the `hf` CLI, for dry-running scripts/ci_stage1_datapath.sh.

Understands exactly the invocation scripts/download_dataset_stage1.sh issues,

    hf download <repo> --repo-type dataset --include <glob> [--include ...] --local-dir <dest>

and materialises a tiny but schema-correct copy of the parts of the
koalapenguin/RoboCloth layout that match the include globs:

    globals/{emitter_calibration.json, camera_factor.json, sample_size.json,
             training_list_500.txt, test_list_500.txt}
    materials/<id>/{observations_structured.npz, scan_log.json,
                    rotated_camera.json, point_metadata.json}      id in 0..499

Like the real CLI, a pattern that matches nothing downloads nothing.  Any flag
or positional outside `hf download --help` is rejected (exit 64) so a drift in
the downloader's CLI usage is caught.  Schemas follow docs/data_formats.md and
the loaders in training/datasets/points.py + training/utils/io.py.

Knobs (environment variables, all optional):
    FAKE_HF_OMIT              space-separated repo paths to leave out
                              (simulates a file missing upstream)
    FAKE_HF_NPZ_DROP_KEY      array name to drop from every observations_structured.npz
    FAKE_HF_META_POINTS_DELTA int added to point_metadata.json num_points
    FAKE_HF_LOG               file to append the JSON-encoded argv to (one line per call)
    FAKE_HF_K / FAKE_HF_V     frames / points per fake material (default 12 / 8)
"""
import fnmatch
import json
import os
import sys

import numpy as np

REPO = "koalapenguin/RoboCloth"
MATERIAL_FILES = ["observations_structured.npz", "scan_log.json", "rotated_camera.json", "point_metadata.json"]
GLOBAL_FILES = ["emitter_calibration.json", "camera_factor.json", "sample_size.json",
                "training_list_500.txt", "test_list_500.txt"]
# `hf download --help` (huggingface_hub 1.29): options taking a value / bare flags
VALUE_OPTS = {"--repo-type", "--type", "--include", "--exclude", "--local-dir", "--revision",
              "--cache-dir", "--token", "--max-workers", "--format"}
FLAG_OPTS = {"--quiet", "-q", "--json", "--force-download", "--no-force-download",
             "--dry-run", "--no-dry-run", "--no-truncate"}


def fail(msg, code=64):
    print(f"fake_hf: {msg}", file=sys.stderr)
    sys.exit(code)


def parse(argv):
    if not argv or argv[0] != "download":
        fail(f"only 'download' is supported (got {argv[:1]})")
    positionals, includes, excludes, opts = [], [], [], {}
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in VALUE_OPTS:
            vals = []
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                vals.append(argv[i])
                i += 1
            if not vals:
                fail(f"option {tok} needs a value")
            if tok == "--include":
                includes.extend(vals)
            elif tok == "--exclude":
                excludes.extend(vals)
            else:
                if len(vals) != 1:
                    fail(f"option {tok} takes exactly one value (got {vals})")
                opts["--repo-type" if tok == "--type" else tok] = vals[0]
            continue
        if tok in FLAG_OPTS:
            opts[tok] = True
        elif tok.startswith("-"):
            fail(f"unknown option {tok}")
        else:
            positionals.append(tok)
        i += 1
    if len(positionals) != 1:
        fail(f"expected exactly one REPO_ID positional (got {positionals})")
    return positionals[0], includes, excludes, opts


def virtual_listing():
    paths = [f"globals/{f}" for f in GLOBAL_FILES]
    for mid in range(500):
        paths.extend(f"materials/{mid}/{f}" for f in MATERIAL_FILES)
    return paths


def rot_z(deg):
    t = np.deg2rad(deg)
    c, s = float(np.cos(t)), float(np.sin(t))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def material_payload(mid, K, V):
    """Deterministic tiny material: K frames x V points, schema as in docs/data_formats.md."""
    rng = np.random.default_rng(mid)
    xyz = (np.array([0.16, -0.11, -0.061], dtype=np.float32)
           + rng.uniform(-0.05, 0.05, size=(V, 3)).astype(np.float32))
    point_ids = np.arange(V, dtype=np.int32)
    rgbs = rng.integers(200, 40000, size=(K, V, 3), dtype=np.uint16)
    rgbs[rng.random((K, V)) < 0.3] = 0                  # 0 = unobserved
    rgbs[0, 0] = 1                                       # keep >=1 valid observation
    cam_pos = rng.uniform(100.0, 400.0, size=(K, 3)).astype(np.float32)     # mm
    light_pos = rng.uniform(100.0, 400.0, size=(K, 3)).astype(np.float32)   # mm
    npz = {"xyz": xyz, "point_ids": point_ids, "rgbs": rgbs, "cam_pos": cam_pos, "light_pos": light_pos}
    drop = os.environ.get("FAKE_HF_NPZ_DROP_KEY")
    if drop:
        npz.pop(drop, None)

    scan_log, rotated_camera = [], []
    for k in range(K):
        angle = 360.0 * k / K
        scan_log.append({
            "filename": f"{k:04d}.png", "scan_id": k, "overall_id": k, "camera_id": k,
            "light_id": k % 120, "emitter_id": k % 120, "turn_angle": angle,
            "position": cam_pos[k].tolist(), "rotation_matrix": rot_z(angle),
            "position_light": light_pos[k].tolist(), "rotation_matrix_light": rot_z(-angle),
        })
        rotated_camera.append({"camera_id": k, "position": cam_pos[k].tolist(), "rotation_matrix": rot_z(angle)})
    n_obs = int((rgbs != 0).any(axis=2).sum())
    meta = {"num_points": V + int(os.environ.get("FAKE_HF_META_POINTS_DELTA", "0")),
            "num_observations": n_obs, "observations_file": "observations_structured.npz"}
    return {"observations_structured.npz": npz, "scan_log.json": scan_log,
            "rotated_camera.json": rotated_camera, "point_metadata.json": meta}


def global_payload(name):
    if name == "emitter_calibration.json":
        table = {f"{a:.1f}": round(float(np.cos(np.deg2rad(a))), 4) for a in range(0, 91, 5)}
        return {"resolution_degrees": 5.0, "angle_range": [0, 90], "max_cam_rad_ratio": 1.0, "data": table}
    if name == "camera_factor.json":
        return {"camera_factor_segments": [{"id_start": 0, "id_end": 99, "factor": "linear_factor1"},
                                           {"id_start": 100, "id_end": 499, "factor": "linear_factor2"}]}
    if name == "sample_size.json":
        return {"sample_sizes": [{"id_start": 0, "id_end": 499, "width": 0.14, "length": 0.14}]}
    if name == "training_list_500.txt":
        return "".join(f"{i}\n" for i in range(0, 500, 5))
    if name == "test_list_500.txt":
        return "".join(f"{i}\n" for i in range(3, 500, 50))
    raise KeyError(name)


def write(dest, rel, payload):
    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rel.endswith(".npz"):
        np.savez(out, **payload)
    elif rel.endswith(".json"):
        with open(out, "w") as f:
            json.dump(payload, f, indent=1)
    else:
        with open(out, "w") as f:
            f.write(payload)


def main(argv):
    log = os.environ.get("FAKE_HF_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(argv) + "\n")
    repo, includes, excludes, opts = parse(argv)
    if repo != REPO:
        fail(f"unknown repo {repo!r} (fake serves only {REPO})")
    if opts.get("--repo-type") != "dataset":
        fail(f"{REPO} is a dataset repo; --repo-type dataset is required (got {opts.get('--repo-type')!r})")
    dest = opts.get("--local-dir")
    if not dest:
        fail("--local-dir is required by this fake (the downloader must assemble a local tree)")
    if not includes:
        fail("no --include patterns given; refusing to fake a full-corpus download")

    omit = set(os.environ.get("FAKE_HF_OMIT", "").split())
    K, V = int(os.environ.get("FAKE_HF_K", "12")), int(os.environ.get("FAKE_HF_V", "8"))
    selected = [p for p in virtual_listing()
                if any(fnmatch.fnmatch(p, pat) for pat in includes)
                and not any(fnmatch.fnmatch(p, pat) for pat in excludes)
                and p not in omit]
    os.makedirs(os.path.join(dest, ".cache", "huggingface", "download"), exist_ok=True)  # like the real CLI
    materials = {}
    for rel in selected:
        parts = rel.split("/")
        if parts[0] == "globals":
            write(dest, rel, global_payload(parts[1]))
        else:
            mid = int(parts[1])
            if mid not in materials:
                materials[mid] = material_payload(mid, K, V)
            write(dest, rel, materials[mid][parts[2]])
    print(f"fake_hf: wrote {len(selected)} file(s) under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
