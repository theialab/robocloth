"""
Shared fixtures for tests/reconstruction — stdlib + numpy only.

Everything generated goes under ROBOCLOTH_TEST_TMP (default: the system temp
dir, <tempdir>/robocloth-tests), never inside the repository. The fixtures are
tiny (a few 4x4 PNGs and 10-image models), so any writable directory will do.
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RECON_DIR = REPO_ROOT / "reconstruction"
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))
import read_write_model as rwm  # noqa: E402

CI_ROOT = Path(os.environ.get("ROBOCLOTH_TEST_TMP") or (Path(tempfile.gettempdir()) / "robocloth-tests"))


def make_tmpdir(prefix: str) -> Path:
    CI_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + "-", dir=CI_ROOT))


def write_png(path, width=4, height=4, value=128):
    """Write a minimal valid 8-bit grayscale PNG (no image library needed)."""
    raw = b"".join(b"\x00" + bytes([value]) * width for _ in range(height))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def frame_names(n):
    return [f"scan-{i:04d}.png" for i in range(n)]


def build_model(n_images, names=None, n_points=6):
    """Tiny well-formed COLMAP model: 1 camera, n_images images, n_points points, every image sees every point."""
    names = names or frame_names(n_images)
    cameras = {1: rwm.Camera(id=1, model="SIMPLE_RADIAL", width=8, height=8,
                             params=np.array([10.0, 4.0, 4.0, 0.0]))}
    images = {}
    for i in range(n_images):
        img_id = i + 1
        images[img_id] = rwm.Image(
            id=img_id, qvec=np.array([1.0, 0.0, 0.0, 0.0]), tvec=np.array([0.1 * i, 0.0, 0.0]),
            camera_id=1, name=names[i],
            xys=np.array([[1.0 + k, 2.0 + k] for k in range(n_points)], dtype=float),
            point3D_ids=np.arange(1, n_points + 1, dtype=np.int64))
    points3D = {}
    for k in range(n_points):
        pid = k + 1
        points3D[pid] = rwm.Point3D(
            id=pid, xyz=np.array([0.1 * k, 0.2, 1.0 + k]), rgb=np.array([10, 20, 30], dtype=np.uint8),
            error=0.5, image_ids=np.arange(1, n_images + 1, dtype=np.int64),
            point2D_idxs=np.full(n_images, k, dtype=np.int64))
    return cameras, images, points3D


def write_ply(path, points3D):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points3D)}",
             "property float x", "property float y", "property float z", "end_header"]
    lines += [" ".join(f"{v:.6f}" for v in pt.xyz) for pt in points3D.values()]
    Path(path).write_text("\n".join(lines) + "\n")


def write_model_dir(sparse_dir, n_images, names=None, n_points=6):
    """Write the layout colmap.sh publishes: 0/*.bin + flattened *.bin + *.txt + points3D.ply."""
    sparse_dir = Path(sparse_dir)
    sub = sparse_dir / "0"
    sub.mkdir(parents=True, exist_ok=True)
    cameras, images, points3D = build_model(n_images, names, n_points)
    rwm.write_model(cameras, images, points3D, str(sub), ext=".bin")
    rwm.write_model(cameras, images, points3D, str(sparse_dir), ext=".bin")
    rwm.write_model(cameras, images, points3D, str(sparse_dir), ext=".txt")
    write_ply(sparse_dir / "points3D.ply", points3D)
    return cameras, images, points3D


def make_project(root, name="mat_001", n_frames=10, existing_sparse=True, sentinel="SENTINEL.txt"):
    """Material dir with ldr/ (tiny PNGs), scan_log.json and optionally a valid sparse/ carrying a sentinel."""
    project = Path(root) / name
    (project / "ldr").mkdir(parents=True)
    for fname in frame_names(n_frames):
        write_png(project / "ldr" / fname)
    (project / "scan_log.json").write_text(
        json.dumps([{"scan_id": i, "angle": 3.6 * i} for i in range(n_frames)]))
    if existing_sparse:
        write_model_dir(project / "sparse", n_frames)
        (project / "sparse" / sentinel).write_text("previous valid reconstruction\n")
    return project


def snapshot(directory):
    """{relative path: sha256} for every file under directory ({} if it does not exist)."""
    directory = Path(directory)
    out = {}
    if not directory.exists():
        return out
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(directory))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
