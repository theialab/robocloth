#!/usr/bin/env python3
"""Collect run_table_bonn.sh results and check them against the paper table.

Usage:
  python collect_eval_results_bonn.py <eval_results_bonn_dir> [--materials "318 377"]
      [--models "Ours Bonn"] [--tolerance-db 0.05] [--missing "318/Bonn ..."]
  python collect_eval_results_bonn.py --reuse-check <result.json> <checkpoint>
      [--expect material=318 model=Ours experiment=stage2_bonn dataset_folder=... overrides=]
  python collect_eval_results_bonn.py --metrics-snapshot <csv_logger_dir>
  python collect_eval_results_bonn.py --check-data <Bonn_val> --materials "318 377 32 226 37"
  python collect_eval_results_bonn.py --val-split-counts <Bonn_val> <mat_id>

Prints the reproduced "Per-material reconstruction PSNR" table, bottom block
(cross-domain: the held-out Bonn / UBOFAB19 test materials 318, 377, 32,
226, 37) next to the values reported in the paper with a PASS/FAIL verdict
per cell.  Same fail-closed contract and exit codes as
scripts/collect_eval_results.py (6 = expected cell missing, 7 = beyond
tolerance, 2 = bad arguments), which implements the table, the result-JSON
schema and the --reuse-check.

Bonn-specific parts, used by run_table_bonn.sh:
  --check-data         fail (exit 2) before any evaluation if the Bonn folder
                       lacks bonn_point_metadata.json, an entry for a requested
                       material, or one of the per-material files the loader
                       opens (REQUIRED_SUFFIXES);
  --val-split-counts   how many poly (RGB) and gray (pan + LLS) images the
                       fixed held-out split of a material contains;
  bonn_val_psnr()      the paper metric: the image-count-weighted mean of the
                       trainer's val/poly_psnr and val/gray_psnr, i.e. the mean
                       per-view PSNR over ALL held-out views (val/all_psnr, a
                       pooled-MSE PSNR, is recorded but is not the table metric).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from collect_eval_results import EXIT_BAD_ARGS, Table, main  # noqa: E402  (scripts/collect_eval_results.py)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
BONN_DATASET_PY = os.path.join(REPO_ROOT, "training", "datasets", "bonn.py")

# Paper: Table "Per-material reconstruction", bottom block (cross-domain: Bonn test set).
# Columns: stage-1 decoder training source (+ Disney-PBR baseline); "Ours" = RoboCloth.
PAPER = {
    "318": {"Ours": 44.38, "Bonn": 45.83, "MERL": 40.26, "PBR": 43.16},
    "377": {"Ours": 16.29, "Bonn": 16.78, "MERL": 15.57, "PBR": 15.93},
    "32":  {"Ours": 15.31, "Bonn": 16.42, "MERL": 14.18, "PBR": 15.17},
    "226": {"Ours": 24.14, "Bonn": 25.09, "MERL": 22.46, "PBR": 23.92},
    "37":  {"Ours": 17.03, "Bonn": 18.22, "MERL": 16.36, "PBR": 16.75},
}
PAPER_AVG = {"Ours": 23.43, "Bonn": 24.47, "MERL": 21.77, "PBR": 22.99}
MODELS = ["Ours", "Bonn", "MERL", "PBR"]

BONN = Table("Per-material reconstruction PSNR (dB) — held-out Bonn (UBOFAB19) test set", PAPER, PAPER_AVG,
             default_dir="eval_results_bonn")

# Per-material files the stage-2 loader opens (datasets/bonn.py, _load_single_material_full,
# with the shipped data config use_pan=True, use_lls=True) plus the metadata the model needs.
REQUIRED_SUFFIXES = ("_calibration.mat", "_xyz_rot000.exr", "_poly.exr", "_pan.exr", "_lls.exr")
METADATA_JSON = "bonn_point_metadata.json"
# Split parameters of configs/data/bonn.yaml (BonnSingleMaterialValDataset).
VAL_VIEW_RATIO, VAL_SEED = 0.2, 42


def material_prefix(data_root: str, mat_id) -> str:
    return os.path.join(data_root, f"mat{int(mat_id):04d}")


def check_data(data_root: str, materials) -> list:
    """Problems that would make every evaluation of these materials fail (empty list = fine)."""
    problems = []
    if not os.path.isdir(data_root):
        return [f"{data_root} is not a directory"]
    meta_path = os.path.join(data_root, METADATA_JSON)
    meta = None
    if not os.path.isfile(meta_path):
        problems.append(f"{meta_path} is missing — run: python scripts/comparisons/generate_bonn_metadata.py {data_root}")
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{meta_path} is unreadable ({e})")
    for mat in materials:
        prefix = material_prefix(data_root, mat)
        for suffix in REQUIRED_SUFFIXES:
            if not os.path.isfile(prefix + suffix):
                problems.append(f"material {mat}: missing {prefix + suffix}")
        if isinstance(meta, dict) and str(int(mat)) not in meta:
            problems.append(f"material {mat}: no entry in {meta_path} — re-run generate_bonn_metadata.py "
                            f"after downloading the material")
    return problems


def _channel_names(exr_path: str) -> list:
    """Channel names of an EXR as the dataset sees them (pyexr: sorted); header only, no pixels."""
    import pyexr  # dependency of the training environment (envs/training.txt)
    return pyexr.open(exr_path).channel_map["all"]


def _dataset_parsers():
    """The channel-name parsers of training/datasets/bonn.py — the loader's own, not a copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("robocloth_bonn_dataset", BONN_DATASET_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._parse_poly_channels, module._parse_pan_channels, module._parse_lls_channels


def val_split_counts(data_root: str, mat_id, use_pan: bool = True, use_lls: bool = True,
                     val_view_ratio: float = VAL_VIEW_RATIO, val_seed: int = VAL_SEED):
    """(n_poly, n_gray): images of each kind in the material's held-out split.

    Mirrors BonnSingleMaterialValDataset: images are enumerated poly, then pan
    (LEDs il001-il024 only), then LLS, from the EXR channel names; a
    RandomState(val_seed) permutation holds out the first
    max(1, int(n_images * val_view_ratio)) of them.  Only EXR headers are read.
    """
    import numpy as np
    parse_poly, parse_pan, parse_lls = _dataset_parsers()
    prefix = material_prefix(data_root, mat_id)
    n_poly = len(parse_poly(_channel_names(prefix + "_poly.exr")))
    n_pan = len([im for im in parse_pan(_channel_names(prefix + "_pan.exr")) if int(im["led"][2:]) <= 24]) if use_pan else 0
    n_lls = len(parse_lls(_channel_names(prefix + "_lls.exr"))) if use_lls else 0
    n_images = n_poly + n_pan + n_lls
    if n_images == 0:
        raise ValueError(f"no poly/pan/lls images found for material {mat_id} under {data_root}")
    perm = np.random.RandomState(val_seed).permutation(n_images)
    val = perm[:max(1, int(n_images * val_view_ratio))]
    n_val_poly = int((val < n_poly).sum())
    return n_val_poly, int(len(val)) - n_val_poly


def bonn_val_psnr(poly_psnr, gray_psnr, n_poly: int, n_gray: int) -> float:
    """The table metric: per-view PSNR averaged over all held-out views.

    The trainer logs val/poly_psnr (mean over the poly views) and val/gray_psnr
    (mean over the pan + LLS views) — every view has the same pixel count, so
    weighting the two by their image counts gives the plain mean over views.
    """
    if n_poly < 0 or n_gray < 0 or n_poly + n_gray == 0:
        raise ValueError(f"invalid split counts poly={n_poly} gray={n_gray}")
    if n_poly and poly_psnr is None:
        raise ValueError(f"{n_poly} poly views in the split but no val/poly_psnr was logged")
    if n_gray and gray_psnr is None:
        raise ValueError(f"{n_gray} gray views in the split but no val/gray_psnr was logged")
    total = (n_poly * float(poly_psnr) if n_poly else 0.0) + (n_gray * float(gray_psnr) if n_gray else 0.0)
    return total / (n_poly + n_gray)


def main_bonn(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--check-data":
        if len(argv) != 4 or argv[2] != "--materials":
            print("usage: collect_eval_results_bonn.py --check-data <Bonn_val> --materials \"318 377 ...\"", file=sys.stderr)
            return EXIT_BAD_ARGS
        problems = check_data(argv[1], argv[3].split())
        for p in problems:
            print(f"[check-data] {p}", file=sys.stderr)
        if problems:
            print(f"[check-data] {len(problems)} problem(s) under {argv[1]}", file=sys.stderr)
            return EXIT_BAD_ARGS
        print(f"[check-data] {argv[1]}: {METADATA_JSON} and all per-material files present for {argv[3]}")
        return 0
    if argv and argv[0] == "--val-split-counts":
        if len(argv) != 3:
            print("usage: collect_eval_results_bonn.py --val-split-counts <Bonn_val> <mat_id>", file=sys.stderr)
            return EXIT_BAD_ARGS
        n_poly, n_gray = val_split_counts(argv[1], argv[2])
        print(json.dumps({"material": str(int(argv[2])), "n_val_poly": n_poly, "n_val_gray": n_gray,
                          "n_val_views": n_poly + n_gray}))
        return 0
    return main(argv, table=BONN)


if __name__ == "__main__":
    sys.exit(main_bonn())
