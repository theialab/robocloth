#!/usr/bin/env python3
"""Audit every sequence's metadata.json against a frozen reference of the APPROVED render settings.

Usage: python audit_dataset_settings.py <B1_root> --reference reference_settings_B1.json [--out audit.json]
Prints one line per deviating sequence and a summary. Exit code 1 if anything deviates.
The reference file is the human-readable contract; change it only with Zhen's explicit approval.
"""
import argparse, glob, json, os, sys


def get(d, path):
    for k in path.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--reference", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = json.load(open(a.reference))
    files = sorted(glob.glob(os.path.join(a.root, "*", "*", "metadata.json")))
    deviations = {}; checked = 0; seen_commits = set(); seen_ckpt = set()
    for f in files:
        m = json.load(open(f)); seq = os.path.relpath(os.path.dirname(f), a.root); checked += 1
        seen_commits.add(get(m, "generator.commit")); seen_ckpt.add(get(m, "material.checkpoint.sha256"))
        bad = []
        for path, expected in ref["must_equal"].items():
            got = get(m, path)
            if got != expected:
                bad.append(f"{path}: got {got!r}, expected {expected!r}")
        if m.get("status") != "complete":
            bad.append(f"status: {m.get('status')}")
        if bad:
            deviations[seq] = bad
    summary = {"root": a.root, "reference": ref.get("name"), "sequences_checked": checked,
               "sequences_deviating": len(deviations), "generator_commits_seen": sorted(c for c in seen_commits if c),
               "checkpoint_sha256_seen": sorted(c for c in seen_ckpt if c), "deviations": deviations}
    for seq, bad in deviations.items():
        print(f"DEVIATION {seq}: " + "; ".join(bad))
    print(f"checked {checked} sequences against '{ref.get('name')}': {len(deviations)} deviating; "
          f"generator commits {summary['generator_commits_seen']}; checkpoint sha {[c[:12] for c in summary['checkpoint_sha256_seen']]}")
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
    return 1 if deviations else 0


if __name__ == "__main__":
    sys.exit(main())
