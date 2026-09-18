#!/usr/bin/env python3
"""
Build per-material `hdr.tar` indexes by stream-parsing each remote tar.

This is the fast path: we issue ONE HTTPS request per material instead of
one per file (the naive walker hits HuggingFace's 3000-req/5-min rate
limit instantly with 500 files per material). The trade-off is bandwidth
— we download the full ~5 GB tar per material — but we never buffer the
file data, so RAM stays flat.

Throughput is then network-bound. On a typical HF CDN connection that is
~30-80 MB/s; expect 1-3 min per full-capture material and a few seconds
per sample-folder material.

Usage
-----
    python3 webpage/tools/build_tar_indexes.py [--ids ...] [--samples ...]
                                               [--workers 2]
                                               [--out webpage/data]
"""

import argparse
import concurrent.futures as cf
import json
import sys
import tarfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script needs `requests`. Install with: pip install requests")
    sys.exit(1)


HF_BASE = "https://huggingface.co/datasets/koalapenguin/RoboCloth/resolve/main"

DEFAULT_SAMPLE_IDS = [156, 249, 309, 428, 483]
DEFAULT_FULL_IDS   = [2, 8, 20, 35, 50, 84, 96, 109, 130, 145, 170, 190,
                      220, 350, 400]


def build_index_streaming(tar_url: str, mid: int) -> dict:
    """
    Stream the tar over HTTP, parse each member's metadata via tarfile,
    and skip data blocks. Returns {filename: [offset, size]}.

    Handles HF's "resolvers" rate limit (3000 requests / 5 min) with
    exponential backoff — we only need ONE successful HTTP request per
    material, but if a previous run blew the budget we need to wait for
    the window to reset.
    """
    sess = requests.Session()
    delay = 5
    while True:
        try:
            r = sess.get(tar_url, stream=True, allow_redirects=True, timeout=600)
            if r.status_code == 429:
                # Honour the ratelimit headers when available
                wait = int(r.headers.get("ratelimit", "").split(";t=")[-1] or "30")
                wait = max(wait + 2, delay)
                print(f"  [{mid}] 429 throttled — sleeping {wait}s")
                r.close()
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            r.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise

    with r:
        # `mode='r|'` -> streaming, uncompressed. tarfile won't try to seek().
        # `bufsize=4MB` so each read pulls a sizable chunk of network.
        tf = tarfile.open(fileobj=r.raw, mode="r|", bufsize=4 * 1024 * 1024)

        index = {}
        n_files = 0
        t0 = time.time()
        for member in tf:
            if member.isfile():
                # member.offset_data == byte offset of the file's data in
                # the tar; member.size is the data length in bytes.
                index[member.name] = [member.offset_data, member.size]
                base = member.name.rsplit("/", 1)[-1]
                if base != member.name and base not in index:
                    index[base] = [member.offset_data, member.size]
                n_files += 1
        elapsed = time.time() - t0
        print(f"  [{mid}] {n_files} files, {elapsed:.1f}s "
              f"({tf.offset / 1e9:.2f} GB streamed)")
        tf.close()
    return index


def process(mid: int, sample: bool, out_dir: Path, force: bool):
    folder = f"sample/material_{mid}" if sample else f"materials/{mid}"
    tar_url = f"{HF_BASE}/{folder}/hdr.tar"
    out_path = out_dir / "tar_index" / f"{mid}.json"

    if out_path.exists() and not force:
        try:
            data = json.loads(out_path.read_text())
            if data:
                print(f"  [{mid}] index already exists ({len(data)} keys); "
                      f"use --force to rebuild")
                return mid, True
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{mid}] streaming {tar_url}…")
    try:
        index = build_index_streaming(tar_url, mid)
        out_path.write_text(json.dumps(index))
        size_kb = out_path.stat().st_size / 1024
        print(f"  [{mid}] -> {out_path.name} ({size_kb:.1f} KB)")
        return mid, True
    except Exception as e:
        print(f"  [{mid}] FAILED: {e}")
        return mid, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--samples", nargs="*", type=int,
                    default=DEFAULT_SAMPLE_IDS)
    ap.add_argument("--out", default="webpage/data")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent materials to stream. Each consumes ~5GB "
                         "of bandwidth so don't go too high.")
    ap.add_argument("--force", action="store_true",
                    help="Re-build even if the index file already exists.")
    args = ap.parse_args()

    if args.ids is None:
        ids = sorted(set(DEFAULT_SAMPLE_IDS + DEFAULT_FULL_IDS))
    else:
        ids = sorted(set(args.ids))
    samples = set(args.samples)
    out_dir = Path(args.out)

    print(f"Building tar indexes for {len(ids)} materials -> {out_dir}/tar_index/")
    print(f"  workers: {args.workers}, force-rebuild: {args.force}")

    succ = fail = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process, mid, mid in samples, out_dir, args.force): mid
            for mid in ids
        }
        for fut in cf.as_completed(futs):
            _, ok = fut.result()
            if ok: succ += 1
            else:  fail += 1
    print(f"\nDone in {time.time()-t0:.1f}s — {succ} OK, {fail} failed")


if __name__ == "__main__":
    main()
