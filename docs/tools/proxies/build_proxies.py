#!/usr/bin/env python
"""Low-res scrubbing proxies for the material viewer (obj.html).

For every material, each capture of Dataset_submission/<id>/hdr/*.png (16-bit linear,
2048 x 3072) becomes a small 8-bit WebP: the frame is area-downsampled to LONG px on its
long side and stored as sqrt(linear) ("sqrt" encoding), which keeps the dark linear
values the viewer's exposure / gamma LUT lifts.  The WebPs are packed into one plain
ustar archive per material (same member order as hdr.tar, names <stem>.webp), so the
page reads a single view with one small HTTP range request through the same TarReader
it already uses for hdr.tar.

Output per material:  <out>/<id>/proxy.tar, <out>/<id>/index.json ({name: [offset, size]}),
<out>/<id>/meta.json.  A material whose outputs exist and match its hdr/ listing is skipped.

  python docs/tools/proxies/build_proxies.py --ids 314,226 --out /media/raid/cloth/output/webpage/proxies
  python docs/tools/proxies/build_proxies.py --all --out ... --workers 8
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC = Path("/media/raid/cloth/Dataset_submission")
HERE = Path(__file__).resolve().parent
TAR_INDEX_DIR = HERE.parent.parent / "data" / "tar_index"
LONG = 640
QUALITY = 90
METHOD = 4


def encode_one(args):
    """-> (webp bytes, source h, w) for one capture."""
    path, long_px, quality = args
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.dtype != np.uint16 or im.ndim != 3:
        # a few published captures are truncated files (libpng "Read Error"): no proxy for them,
        # the viewer simply shows none while the full capture loads
        return None
    h, w = im.shape[:2]
    s = long_px / max(h, w)
    small = cv2.resize(im.astype(np.float32), (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    enc = np.clip(np.sqrt(small / 65535.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, :, ::-1]   # BGR -> RGB
    buf = io.BytesIO()
    Image.fromarray(enc).save(buf, "WEBP", quality=quality, method=METHOD)
    return buf.getvalue(), h, w


def listing(mid):
    d = SRC / str(mid) / "hdr"
    names = sorted(p.name for p in d.iterdir() if p.suffix == ".png")
    if not names:
        raise RuntimeError(f"no PNGs in {d}")
    idx = TAR_INDEX_DIR / f"{mid}.json"
    if idx.exists():
        keys = sorted(k for k in json.load(open(idx)) if "/" not in k)
        if keys != names:
            raise RuntimeError(f"{mid}: hdr/ listing differs from data/tar_index ({len(names)} vs {len(keys)} entries)")
    return names


def done(mid, out, names):
    m = out / str(mid) / "meta.json"
    if not m.exists():
        return False
    meta = json.load(open(m))
    idx = out / str(mid) / "index.json"
    has_meta = idx.exists() and "__meta" in json.load(open(idx))
    return (meta.get("count") == len(names) and meta.get("long") == LONG and has_meta
            and (out / str(mid) / "proxy.tar").exists())


def build(mid, out, pool, workers):
    names = listing(mid)
    if done(mid, out, names):
        return "skip", len(names), 0.0
    t0 = time.time()
    d = out / str(mid)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "proxy.tar.part"
    index = {}
    sizes_in = 0
    jobs = [(SRC / str(mid) / "hdr" / n, LONG, QUALITY) for n in names]
    skipped = []
    h = w = None
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for n, res in zip(names, pool.map(encode_one, jobs, chunksize=4)):
            if res is None:
                skipped.append(n)
                continue
            data, h, w = res
            info = tarfile.TarInfo(n[:-4] + ".webp")
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
            # offset of the member's data = current archive position - its padded size
            end = tar.fileobj.tell()
            index[info.name] = [end - ((len(data) + 511) // 512) * 512, len(data)]
            sizes_in += 1
    os.replace(tmp, d / "proxy.tar")
    # the page reads the geometry it needs from the index itself (keys starting with "__"
    # are not members)
    index["__meta"] = dict(long=LONG, source_hw=[h, w], encoding="sqrt", count=len(names) - len(skipped))
    json.dump(index, open(d / "index.json", "w"), separators=(",", ":"))
    json.dump(dict(count=len(names), long=LONG, quality=QUALITY, encoding="sqrt", source_hw=[h, w],
                   source="Dataset_submission/<id>/hdr", bytes=os.path.getsize(d / "proxy.tar"),
                   skipped_unreadable=skipped),
              open(d / "meta.json", "w"), indent=1)
    if skipped:
        print(f"  {mid}: {len(skipped)} unreadable capture(s) skipped: {', '.join(skipped)}", flush=True)
    return "built", len(names), time.time() - t0


def verify(mid, out):
    """Every index entry reads back as a decodable WebP at the recorded offset."""
    d = out / str(mid)
    index = {k: v for k, v in json.load(open(d / "index.json")).items() if not k.startswith("__")}
    with open(d / "proxy.tar", "rb") as fh, tarfile.open(d / "proxy.tar") as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert set(members) == set(index), "member set differs from index"
        for name, (off, size) in index.items():
            m = members[name]
            assert (m.offset_data, m.size) == (off, size), f"{name}: index {off, size} vs tar {m.offset_data, m.size}"
        for name in list(index)[:: max(1, len(index) // 8)]:
            off, size = index[name]
            fh.seek(off)
            Image.open(io.BytesIO(fh.read(size))).load()
    return len(index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma-separated material ids")
    ap.add_argument("--all", action="store_true", help="every material that has a data/tar_index entry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.all:
        ids = sorted(int(p.stem) for p in TAR_INDEX_DIR.glob("*.json"))
    elif a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    else:
        sys.exit("--ids or --all")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total_b = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for k, mid in enumerate(ids):
            try:
                st, n, dt = build(mid, out, pool, a.workers)
                b = os.path.getsize(out / str(mid) / "proxy.tar")
                total_b += b
                extra = f" verified {verify(mid, out)}" if a.verify and st == "built" else ""
                print(f"[{k + 1}/{len(ids)}] {mid}: {st} {n} views {b / 1e6:.1f} MB {dt:.0f} s{extra}", flush=True)
            except Exception as e:                      # keep going; the failure is in the log
                print(f"[{k + 1}/{len(ids)}] {mid}: FAILED {e}", flush=True)
    print(f"total {total_b / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
