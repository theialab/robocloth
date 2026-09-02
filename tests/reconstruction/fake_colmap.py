"""
Stand-in for the `colmap` binary used by tests/reconstruction (no real COLMAP).

Emulates exactly the subcommands colmap.sh / colmap_exhaustive.sh call and
writes its outputs with reconstruction/read_write_model.py. The mapper's
behaviour is selected with FAKE_COLMAP_MODE:

  valid        (default) register every image found in --image_path
  low_reg      register only FAKE_COLMAP_REGISTERED images (default: half)
  empty        write a syntactically valid model with 0 cameras/images/points
  truncated    write a 4-byte images.bin (unreadable by read_write_model)
  mapper_fail  write a partial output (0/cameras.bin only), print an error, exit 1
  hang         write 0/cameras.bin, then sleep (<= 60 s) until killed — signal tests

Every invocation appends "<subcommand> CUDA_VISIBLE_DEVICES=<value|unset>" to
$FAKE_COLMAP_LOG so tests can assert the pipeline order and the GPU id. The
tests install it as an executable named `colmap` first on PATH.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixtures  # noqa: E402

rwm = fixtures.rwm


def parse_args(argv):
    """COLMAP accepts both `--key value` and `--key=value`."""
    opts = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
                opts[k] = v
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[a[2:]] = argv[i + 1]
                i += 1
            else:
                opts[a[2:]] = "true"
        i += 1
    return opts


def log_call(sub):
    log = os.environ.get("FAKE_COLMAP_LOG")
    if log:
        with open(log, "a") as f:
            f.write(f"{sub} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}\n")


def mapper(opts):
    mode = os.environ.get("FAKE_COLMAP_MODE", "valid")
    out = Path(opts["output_path"]) / "0"
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(f for f in os.listdir(opts["image_path"]) if f.lower().endswith((".png", ".jpg")))
    if mode == "mapper_fail":
        cams, _, _ = fixtures.build_model(1, names[:1])
        rwm.write_cameras_binary(cams, str(out / "cameras.bin"))
        print("fake colmap mapper: simulated crash", file=sys.stderr)
        return 1
    if mode == "hang":
        # signal tests: leave a partial output, then block until killed
        cams, _, _ = fixtures.build_model(1, names[:1])
        rwm.write_cameras_binary(cams, str(out / "cameras.bin"))
        for _ in range(600):  # <= 60 s, so a failed test cannot leak a process for long
            time.sleep(0.1)
        return 1
    if mode == "empty":
        rwm.write_model({}, {}, {}, str(out), ext=".bin")
        return 0
    if mode == "truncated":
        cams, imgs, pts = fixtures.build_model(len(names), names)
        rwm.write_model(cams, imgs, pts, str(out), ext=".bin")
        (out / "images.bin").write_bytes(b"\x00\x01\x02\x03")
        return 0
    n = len(names)
    if mode == "low_reg":
        n = int(os.environ.get("FAKE_COLMAP_REGISTERED", max(1, len(names) // 2)))
    cams, imgs, pts = fixtures.build_model(n, names[:n])
    rwm.write_model(cams, imgs, pts, str(out), ext=".bin")
    return 0


def _read_model_lenient(path):
    """Read what can be read (a truncated file yields an empty table) — the
    validator, not the converter, is what must catch a corrupt model."""
    tables = []
    for reader, name in ((rwm.read_cameras_binary, "cameras.bin"),
                         (rwm.read_images_binary, "images.bin"),
                         (rwm.read_points3D_binary, "points3D.bin")):
        try:
            tables.append(reader(str(path / name)))
        except Exception:
            tables.append({})
    return tables


def model_converter(opts):
    inp = Path(opts["input_path"])
    outp = Path(opts["output_path"])
    kind = opts.get("output_type", "").upper()
    cams, imgs, pts = _read_model_lenient(inp)
    if kind == "PLY":
        fixtures.write_ply(outp, pts)
        return 0
    if kind == "TXT":
        outp.mkdir(parents=True, exist_ok=True)
        rwm.write_model(cams, imgs, pts, str(outp), ext=".txt")
        return 0
    print(f"fake colmap: unsupported output_type {kind}", file=sys.stderr)
    return 2


def main(argv):
    if not argv:
        print("fake colmap: missing subcommand", file=sys.stderr)
        return 2
    sub, opts = argv[0], parse_args(argv[1:])
    log_call(sub)
    if sub == "feature_extractor":
        Path(opts["database_path"]).write_bytes(b"fake-colmap-db")
        return 0
    if sub in ("sequential_matcher", "exhaustive_matcher"):
        return 0 if Path(opts["database_path"]).exists() else 1
    if sub == "mapper":
        return mapper(opts)
    if sub == "model_converter":
        return model_converter(opts)
    print(f"fake colmap: unsupported subcommand {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
