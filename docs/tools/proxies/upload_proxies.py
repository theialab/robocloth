#!/usr/bin/env python
"""Upload the viewer's scrubbing proxies (build_proxies.py output) to the assets repo,
batch by batch, as materials finish building.

  python docs/tools/proxies/upload_proxies.py --out /media/raid/cloth/output/webpage/proxies
  python docs/tools/proxies/upload_proxies.py --out ... --verify      # compare remote sizes

A material is uploaded once its meta.json exists (written last by the builder) and its
index.json carries the "__meta" record; each commit holds --batch materials (3 files each).
State: <out>/_logs/uploaded.json.  Needs a Hugging Face login with write access
(hf auth login as koalapenguin).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "koalapenguin/RoboCloth-assets"
PREFIX = "web/proxies"
FILES = ("proxy.tar", "index.json", "meta.json")
HERE = Path(__file__).resolve().parent
TAR_INDEX_DIR = HERE.parent.parent / "data" / "tar_index"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ready(out, mid):
    d = out / str(mid)
    if not all((d / f).exists() for f in FILES):
        return False
    try:
        return "__meta" in json.load(open(d / "index.json"))
    except Exception:
        return False


def verify(api, out, ids):
    remote = {}
    for e in api.list_repo_tree(REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True):
        if getattr(e, "size", None) is not None:
            remote[e.path] = e.size
    bad = []
    for mid in ids:
        for f in FILES:
            p = f"{PREFIX}/{mid}/{f}"
            local = out / str(mid) / f
            if not local.exists():
                bad.append((p, "no local file")); continue
            if p not in remote:
                bad.append((p, "missing remotely")); continue
            if remote[p] != local.stat().st_size:
                bad.append((p, f"size {remote[p]} != {local.stat().st_size}"))
    log(f"verify: {len(ids)} materials, {len(remote)} remote files, {len(bad)} problems")
    for p, why in bad[:40]:
        print("  ", p, why)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--poll", type=int, default=120, help="seconds between looks for new materials")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    ids = sorted(int(p.stem) for p in TAR_INDEX_DIR.glob("*.json"))
    api = HfApi()
    log(f"login: {api.whoami()['name']}; {len(ids)} materials expected")
    if a.verify:
        sys.exit(0 if verify(api, out, ids) else 1)
    state_p = out / "_logs" / "uploaded.json"
    state_p.parent.mkdir(exist_ok=True)
    done = set(json.load(open(state_p))) if state_p.exists() else set()
    total = 0
    while True:
        todo = [m for m in ids if m not in done and ready(out, m)]
        for k in range(0, len(todo), a.batch):
            batch = todo[k:k + a.batch]
            ops = [CommitOperationAdd(path_in_repo=f"{PREFIX}/{m}/{f}", path_or_fileobj=str(out / str(m) / f))
                   for m in batch for f in FILES]
            size = sum((out / str(m) / "proxy.tar").stat().st_size for m in batch)
            for attempt in range(4):
                try:
                    api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                                      commit_message=f"viewer proxies: {len(batch)} materials ({batch[0]}..{batch[-1]})")
                    break
                except Exception as e:                    # transient: wait and retry
                    log(f"commit failed ({type(e).__name__}: {str(e)[:120]}), retry {attempt + 1}")
                    time.sleep(30 * (attempt + 1))
            else:
                log("giving up on this batch for now"); break
            done |= set(batch)
            total += size
            json.dump(sorted(done), open(state_p, "w"))
            log(f"uploaded {len(batch)} ({batch[0]}..{batch[-1]}, {size / 1e6:.0f} MB); {len(done)}/{len(ids)} done")
        if len(done) >= len(ids):
            log("all materials uploaded")
            verify(api, out, ids)
            break
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
