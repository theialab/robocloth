#!/usr/bin/env python3
"""Generate bonn_point_metadata.json for a Bonn SVBRDF dataset folder.

Scans all matXXXX_xyz_rot000.exr files, reads (H, W) from the EXR header,
and writes a single JSON file mapping mat_id -> num_points (= H * W).

Usage:
    python scripts/comparisons/generate_bonn_metadata.py /absolute/path/to/Bonn_train
    python scripts/comparisons/generate_bonn_metadata.py /absolute/path/to/Bonn_val

Run it once per downloaded folder (stage 1 reads Bonn_train, stage 2 and the
evaluation read Bonn_val); the models (BonnLatentBRDF / BonnPBRLatentBRDF)
and the renderer's Bonn wrapper size the per-material latent bank from it.
Re-run it after adding materials to a folder.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_exr_dimensions(filepath):
    """Read (H, W) from an EXR file header without loading pixel data."""
    import OpenEXR
    f = OpenEXR.InputFile(str(filepath))
    dw = f.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    return H, W


def main():
    parser = argparse.ArgumentParser(description="Generate bonn_point_metadata.json")
    parser.add_argument("data_folder", type=str,
                        help="Path to a Bonn dataset folder (e.g. /absolute/path/to/Bonn_train)")
    args = parser.parse_args()

    root = Path(args.data_folder)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    xyz_files = sorted(root.glob("mat*_xyz_rot000.exr"))
    if not xyz_files:
        print(f"Error: no mat*_xyz_rot000.exr files found in {root}")
        sys.exit(1)

    metadata = {}
    for xyz_path in xyz_files:
        mat_str = xyz_path.stem.split("_")[0]   # 'mat0001'
        mat_id = int(mat_str[3:])               # 1
        H, W = get_exr_dimensions(xyz_path)
        num_points = H * W
        metadata[mat_id] = {"H": H, "W": W, "num_points": num_points}
        print(f"  mat{mat_id:04d}: {H} x {W} = {num_points:,} points")

    out_path = root / "bonn_point_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"\nWrote {len(metadata)} materials to {out_path}")
    total = sum(v["num_points"] for v in metadata.values())
    print(f"Total points across all materials: {total:,}")


if __name__ == "__main__":
    main()
