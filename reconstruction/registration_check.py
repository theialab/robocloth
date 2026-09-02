"""
Single source of truth for COLMAP registration health checks.

A material is considered "well-registered" when COLMAP successfully placed
at least REGISTRATION_THRESHOLD * len(scan_log) images in sparse/0/images.bin.
Materials below the threshold need to be re-run with exhaustive_matcher (see
colmap_exhaustive.sh).

Used by:
  - scheduler.py                       (live + restart sanity check)
  - colmap.sh / colmap_exhaustive.sh   (CLI below: validate the *staged* model
                                        before it replaces a published sparse/)
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

# Make read_write_model importable regardless of CWD
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from read_write_model import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)

# Materials with registered/K below this are unhealthy and must be re-run.
# Empirical distribution across 299 materials in Dataset_Nov11:
#   - 137 register >=99%, 141 register 97-99%, 8 register 95-97%
#   - 1 marginal at 94.6%, 1 marginal at 92.9%
#   - then a cliff: 0 in 80-90%, 11 truly broken cases all below 80%
# Picking 0.90 catches the entire failure cliff while letting the two
# 92-94% marginals through (avoids unnecessary exhaustive reruns).
DEFAULT_REGISTRATION_THRESHOLD = 0.90

# Override with COLMAP_REGISTRATION_THRESHOLD=<float in (0, 1]>. It is read
# once at import, so the scheduler (which hands its environment to the
# colmap.sh children) and the staged-model validator apply the same gate.
REGISTRATION_THRESHOLD_ENV = "COLMAP_REGISTRATION_THRESHOLD"


def _threshold_from_env() -> float:
    raw = os.environ.get(REGISTRATION_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_REGISTRATION_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{REGISTRATION_THRESHOLD_ENV}={raw!r} is not a number") from None
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{REGISTRATION_THRESHOLD_ENV}={raw!r} must be in (0, 1]")
    return value


REGISTRATION_THRESHOLD = _threshold_from_env()

# Exit codes of the CLI — and therefore of the "validate" stage of colmap.sh.
VALIDATE_OK = 0
VALIDATE_INVALID_MODEL = 2
VALIDATE_LOW_REGISTRATION = 3


def _images_bin_header_count(img_bin) -> int:
    """Return num_reg_images from a COLMAP images.bin, or -1 if unreadable.

    Reads only the 8-byte uint64 header (num_reg_images) — COLMAP's binary
    format starts with that count. Avoids parsing the full file (~100-200 MB
    over NFS) when callers only need the count.
    """
    img_bin = Path(img_bin)
    if not img_bin.exists():
        return -1
    try:
        with open(img_bin, "rb") as fid:
            header = fid.read(8)
            if len(header) < 8:
                return -1
            return struct.unpack("<Q", header)[0]
    except Exception:
        return -1


def count_registered_images(material_dir) -> int:
    """Return the number of images in sparse/0/images.bin, or -1 if unreadable."""
    return _images_bin_header_count(Path(material_dir) / "sparse" / "0" / "images.bin")


def _scan_log_length(scan_log_path) -> int:
    """Return the number of entries in a scan_log.json file, or -1 if unreadable."""
    sl = Path(scan_log_path)
    if not sl.exists():
        return -1
    try:
        with open(sl) as f:
            return len(json.load(f))
    except Exception:
        return -1


def count_scan_log_entries(material_dir) -> int:
    """Return the number of entries in scan_log.json, or -1 if unreadable."""
    return _scan_log_length(Path(material_dir) / "scan_log.json")


def registration_ratio(material_dir) -> Tuple[int, int, float]:
    """
    Return (n_registered, n_scans, ratio).
    Ratio is in [0, 1], or -1.0 if either side is missing/unreadable.
    """
    n_reg = count_registered_images(material_dir)
    n_scans = count_scan_log_entries(material_dir)
    if n_reg < 0 or n_scans <= 0:
        return n_reg, n_scans, -1.0
    return n_reg, n_scans, n_reg / n_scans


def is_well_registered(material_dir, threshold: Optional[float] = None) -> bool:
    """Return True iff registration ratio >= threshold (default REGISTRATION_THRESHOLD)."""
    if threshold is None:
        threshold = REGISTRATION_THRESHOLD
    _, _, ratio = registration_ratio(material_dir)
    return ratio >= threshold


# ---------------------------------------------------------------------------
# Staged-model validation (colmap.sh / colmap_exhaustive.sh, stage "validate")
# ---------------------------------------------------------------------------

_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def validate_sparse_model(sparse_dir, scan_log_path,
                          threshold: Optional[float] = None) -> Tuple[int, str]:
    """
    Validate a *staged* COLMAP sparse model before it replaces a published one.

    `sparse_dir` must have the layout colmap.sh produces: the mapper's submodel
    in sparse_dir/0/{cameras,images,points3D}.bin plus a flattened copy of those
    three files and points3D.ply in sparse_dir itself (what reconstruct.py and
    the scheduler read).

    Checks, in order (fail-closed — any exception counts as invalid):
      1. every required file exists and is non-empty;
      2. the flattened model loads with read_write_model, cameras / images /
         points3D are all non-empty and every image references a known camera;
      3. sparse_dir/0/images.bin registers the same number of images (that is
         the file registration_ratio() reads later);
      4. registered / len(scan_log) >= threshold (default REGISTRATION_THRESHOLD).

    Returns (code, message) with code VALIDATE_OK, VALIDATE_INVALID_MODEL or
    VALIDATE_LOW_REGISTRATION.
    """
    if threshold is None:
        threshold = REGISTRATION_THRESHOLD
    sparse_dir = Path(sparse_dir)

    # 1. Required files: submodel 0, the flattened copy, the PLY.
    required = [sparse_dir / "0" / f for f in _MODEL_FILES]
    required += [sparse_dir / f for f in _MODEL_FILES]
    required.append(sparse_dir / "points3D.ply")
    for path in required:
        if not path.is_file():
            return VALIDATE_INVALID_MODEL, f"missing {path}"
        if path.stat().st_size == 0:
            return VALIDATE_INVALID_MODEL, f"empty file {path}"

    # 2. The flattened model must load and be non-empty.
    try:
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))
        points3D = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    except Exception as e:  # truncated / corrupt binary
        return VALIDATE_INVALID_MODEL, (f"model in {sparse_dir} does not load: "
                                        f"{type(e).__name__}: {e}")
    for name, table in (("cameras", cameras), ("images", images), ("points3D", points3D)):
        if len(table) == 0:
            return VALIDATE_INVALID_MODEL, f"{name}.bin in {sparse_dir} is empty"
    unknown = sorted({img.camera_id for img in images.values()} - set(cameras))
    if unknown:
        return VALIDATE_INVALID_MODEL, f"images reference unknown camera ids {unknown}"

    # 3. sparse/0/images.bin must agree with the flattened copy.
    n_sub = _images_bin_header_count(sparse_dir / "0" / "images.bin")
    if n_sub != len(images):
        return VALIDATE_INVALID_MODEL, (f"{sparse_dir}/0/images.bin registers {n_sub} images "
                                        f"but the flattened copy {len(images)}")

    # 4. Registration gate (same rule as registration_ratio()).
    n_scans = _scan_log_length(scan_log_path)
    if n_scans <= 0:
        return VALIDATE_INVALID_MODEL, f"cannot read scan_log {scan_log_path}"
    ratio = len(images) / n_scans
    msg = (f"registered {len(images)}/{n_scans} ({ratio*100:.1f}%), "
           f"threshold {threshold*100:.0f}%")
    if ratio < threshold:
        return VALIDATE_LOW_REGISTRATION, msg + " — below threshold"
    return VALIDATE_OK, msg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a staged COLMAP sparse model before it is published "
                    "(used by colmap.sh / colmap_exhaustive.sh).")
    parser.add_argument("--sparse-dir", required=True,
                        help="staged sparse/ dir (contains 0/ and the flattened *.bin)")
    parser.add_argument("--scan-log", required=True, help="the material's scan_log.json")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"registration gate (default: ${REGISTRATION_THRESHOLD_ENV} "
                             f"or {DEFAULT_REGISTRATION_THRESHOLD})")
    args = parser.parse_args(argv)
    code, msg = validate_sparse_model(args.sparse_dir, args.scan_log, args.threshold)
    print(f"[validate] {'OK' if code == VALIDATE_OK else 'FAIL'} ({code}): {msg}")
    return code


if __name__ == "__main__":
    sys.exit(main())
