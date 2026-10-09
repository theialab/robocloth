#!/usr/bin/env python3
"""
Build data/tar_index/<id>.json for every material WITHOUT touching hdr.tar.

The published materials/<id>/hdr.tar files were written by Python's tarfile
(pax format) from the sorted files of Dataset_submission/<id>/hdr/: every
member is a 512-byte pax header + 512 bytes of pax data (the float mtime) +
a 512-byte ustar header, then the PNG padded to 512 bytes; the archive ends
with two zero blocks, padded to a 10240-byte record. So each PNG's offset
follows from the local file sizes alone:

    offset_k = sum_{j<k} (1536 + pad512(size_j)) + 1536

This reproduces all 25 shipped indexes byte for byte. To guard against a
local tree that drifted from what was uploaded, the predicted archive size
is compared with the size Hugging Face reports for hdr.tar (`--verify`:
one batched /paths-info API call per 100 materials), and `--spot-check N`
reads N member headers per material over HTTP Range (1 resolve + N small
CDN requests per material; leave at 0 for a full run, use it on a sample).

Usage
-----
    python3 docs/tools/build_tar_indexes_local.py --ids 314 --verify
    python3 docs/tools/build_tar_indexes_local.py --all --verify --write
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "koalapenguin/RoboCloth"
HF = "https://huggingface.co"
LOCAL = Path("/media/raid/cloth/Dataset_submission")
HEADER = 1536          # pax header + pax data + ustar header
RECORD = 10240


def pad512(n):
    return (n + 511) // 512 * 512


def predict(mid, root):
    hdr = root / str(mid) / "hdr"
    names = sorted(n for n in os.listdir(hdr) if not n.startswith("."))
    index, off = {}, 0
    for n in names:
        size = (hdr / n).stat().st_size
        index[n] = [off + HEADER, size]
        off += HEADER + pad512(size)
    total = (off + 1024 + RECORD - 1) // RECORD * RECORD
    return index, total


def hf_sizes(ids):
    """{id: size of materials/<id>/hdr.tar} via batched /paths-info calls."""
    out = {}
    for k in range(0, len(ids), 100):
        batch = ids[k:k + 100]
        body = urllib.parse.urlencode(
            [("paths", f"materials/{m}/hdr.tar") for m in batch]).encode()
        req = urllib.request.Request(
            f"{HF}/api/datasets/{REPO}/paths-info/main", data=body, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            for item in json.load(r):
                m = int(item["path"].split("/")[1])
                out[m] = item.get("size")
        time.sleep(1.0)
    return out


def spot_check(mid, index, n):
    """Read the ustar header in front of n random members (polite: 0.2 s apart)."""
    url = f"{HF}/datasets/{REPO}/resolve/main/materials/{mid}/hdr.tar"
    names = random.sample(sorted(index), min(n, len(index)))
    for name in names:
        off = index[name][0] - 512
        req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 511}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            url = r.geturl()            # reuse the signed CDN URL afterwards
            block = r.read()
        got = block[:100].split(b"\0", 1)[0].decode()
        size = int(block[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
        if got != f"hdr/{name}" or size != index[name][1]:
            return f"header mismatch at {name}: {got!r} size {size}"
        time.sleep(0.2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=[])
    ap.add_argument("--all", action="store_true", help="every material under --local")
    ap.add_argument("--local", type=Path, default=LOCAL)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "tar_index")
    ap.add_argument("--verify", action="store_true", help="compare archive size with HF")
    ap.add_argument("--spot-check", type=int, default=0, metavar="N")
    ap.add_argument("--write", action="store_true", help="write the JSON files (else dry run)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = sorted({int(p.name) for p in args.local.iterdir() if p.name.isdigit()}
                 if args.all else set(args.ids))
    if not ids:
        ap.error("give --ids or --all")
    sizes = hf_sizes(ids) if args.verify else {}

    ok = bad = 0
    for mid in ids:
        try:
            index, total = predict(mid, args.local)
        except FileNotFoundError:
            print(f"[{mid}] no local hdr/ folder"); bad += 1; continue
        problem = None
        if args.verify and sizes.get(mid) != total:
            problem = f"size mismatch: predicted {total}, HF {sizes.get(mid)}"
        if not problem and args.spot_check:
            problem = spot_check(mid, index, args.spot_check)
        if problem:
            print(f"[{mid}] SKIP ({problem})"); bad += 1; continue
        ok += 1
        dst = args.out / f"{mid}.json"
        if args.write and (args.overwrite or not dst.exists()):
            # basename keys only: tar_reader.js also resolves "hdr/<name>"
            dst.write_text(json.dumps(index, separators=(",", ":")))
        print(f"[{mid}] {len(index)} files, {total} bytes"
              + (" (verified)" if args.verify else "")
              + (f" -> {dst}" if args.write else ""))
    print(f"done: {ok} ok, {bad} skipped")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
