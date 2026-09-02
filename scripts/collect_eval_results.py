#!/usr/bin/env python3
"""Collect eval_stage2.sh results and check them against the paper table.

Usage:
  python collect_eval_results.py <eval_results_dir> [--materials "226 314"]
      [--models "Ours Bonn"] [--tolerance-db 0.05] [--missing "226/Bonn ..."]
  python collect_eval_results.py --reuse-check <result.json> <checkpoint>
      [--expect material=226 model=Ours experiment=eval_stage2 dataset_folder=... overrides=]
  python collect_eval_results.py --metrics-snapshot <csv_logger_dir>

Reads every <mat>_<model>.json produced by eval_stage2.sh and prints the
reproduced "Per-material reconstruction PSNR" table (top block: in-domain on
our held-out test set) next to the values reported in the paper, with a
PASS/FAIL verdict per cell.  The collector fails closed:
  exit 6  an expected cell (MATERIALS x MODELS; default the whole table) has
          no valid result,
  exit 7  a cell differs from the paper by more than the tolerance
          (default 0.05 dB, env TOLERANCE_DB; 7 takes precedence over 6),
  exit 2  bad arguments (e.g. a material that has no paper value).

This module also owns the result-JSON contract shared by eval_stage2.sh /
eval_stage2_ubo.sh and the table drivers (run_table_ours.sh / run_table_ubo.sh):
  write_result()  writes <name>.json atomically (<name>.json.tmp + rename) with
                  provenance: checkpoint path / size / mtime (+ sha256 when
                  CKPT_SHA256=1), git HEAD of this repo, timestamp, host and
                  the config identifiers of the run;
  check_reuse()   (--reuse-check) decides whether an existing JSON may stand in
                  for re-evaluating a checkpoint: only if the recorded checkpoint
                  path, size and mtime all match the file on disk AND every
                  identifier passed with --expect (material, model, experiment,
                  dataset_folder, overrides, ...) matches what was recorded;
  metrics_files() / fresh_metrics_csv()  (--metrics-snapshot) let the evaluators
                  tell THIS run's CSVLogger metrics.csv from files left by earlier
                  runs of the same experiment name.
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

EXIT_BAD_ARGS, EXIT_MISSING_CELLS, EXIT_TOLERANCE = 2, 6, 7
DEFAULT_TOLERANCE_DB = 0.05   # the milestone-3 reproduction saw at most +0.04 dB per cell
AVG_ROUNDING_SLACK_DB = 0.01  # paper averages are means of unrounded values printed to 0.01 dB
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_KEYS = ("dataset_folder", "btf_path")   # --expect identifiers compared by realpath

# Paper: Table "Per-material reconstruction PSNR", top block (our test set).
# Rows: material; columns: stage-1 decoder source (+ PBR baseline).
PAPER = {
    "226": {"Ours": 29.24, "Bonn": 25.63, "MERL": 24.24, "PBR": 25.35},
    "314": {"Ours": 26.60, "Bonn": 24.42, "MERL": 22.93, "PBR": 23.98},
    "370": {"Ours": 29.71, "Bonn": 29.21, "MERL": 27.75, "PBR": 29.29},
    "145": {"Ours": 28.44, "Bonn": 24.45, "MERL": 23.39, "PBR": 24.28},
    "452": {"Ours": 34.15, "Bonn": 32.91, "MERL": 30.89, "PBR": 32.06},
}
PAPER_AVG = {"Ours": 29.63, "Bonn": 27.32, "MERL": 25.84, "PBR": 26.99}
MODELS = ["Ours", "Bonn", "MERL", "PBR"]


@dataclass(frozen=True)
class Table:
    """One paper table: material -> model -> PSNR (dB), plus the average row."""
    title: str
    paper: dict
    paper_avg: dict
    name_width: int = 8            # width of the material column
    default_dir: str = "eval_results"

    @property
    def materials(self):
        return list(self.paper)

    @property
    def models(self):
        return list(self.paper_avg)


OURS = Table("Per-material reconstruction PSNR (dB) — our held-out test set", PAPER, PAPER_AVG)


# ---- result-JSON contract ------------------------------------------------------
def checkpoint_fingerprint(ckpt: str, sha256: bool = False) -> dict:
    """Identity of a checkpoint file: resolved path, size and mtime (+ optional sha256)."""
    st = os.stat(ckpt)
    fp = {
        "ckpt_abs": os.path.realpath(ckpt),
        "ckpt_size": st.st_size,
        "ckpt_mtime_ns": st.st_mtime_ns,
        "ckpt_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if sha256:
        h = hashlib.sha256()
        with open(ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        fp["ckpt_sha256"] = h.hexdigest()
    return fp


def git_state(repo: str = REPO_ROOT):
    """(HEAD sha, dirty?) of the repo; ("unknown", None) when git is unavailable."""
    def run(*args):
        return subprocess.run(["git", "--no-optional-locks", "-C", repo, *args],
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def write_result(out_path: str, material, model: str, ckpt: str, val_psnr: float,
                 val_loss=None, **extra) -> dict:
    """Write one <mat>_<model>.json result with provenance; atomic (.json.tmp + rename)."""
    head, dirty = git_state()
    res = {"material": str(material), "model": model, "ckpt": ckpt,
           "val_psnr": val_psnr, "val_loss": val_loss}
    res.update(checkpoint_fingerprint(ckpt, sha256=os.environ.get("CKPT_SHA256") == "1"))
    res.update({"git_head": head, "git_dirty": dirty,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hostname": socket.gethostname(), "provenance_version": 1})
    res.update(extra)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return res


def _identity(value) -> str:
    """Normalise a recorded / expected identifier: lists space-joined, None -> ''."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return "" if value is None else str(value)


def check_reuse(json_path: str, ckpt: str, expect=None):
    """May json_path stand in for re-evaluating ckpt?  -> (ok, human-readable reason).

    A result is reusable only if it has a val_psnr, every identifier in `expect`
    ({key: value} — material, model, experiment, dataset_folder, overrides, ...)
    matches what it recorded (PATH_KEYS by realpath, lists space-joined; a key it
    does not record is a mismatch), and its recorded checkpoint path, size and
    mtime (and sha256, when both recorded and CKPT_SHA256=1) match the checkpoint
    on disk.  Anything else -> re-evaluate.
    """
    if not os.path.isfile(json_path):
        return False, f"no previous result ({json_path})"
    try:
        with open(json_path) as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        return False, f"previous result is unreadable ({e})"
    if not isinstance(r, dict) or r.get("val_psnr") is None:
        return False, "previous result has no val_psnr"
    if any(k not in r for k in ("ckpt_abs", "ckpt_size", "ckpt_mtime_ns")):
        return False, "previous result records no checkpoint provenance (written before the fail-closed drivers)"
    for key, want in (expect or {}).items():
        if key not in r:
            return False, f"previous result does not record '{key}' (written by an older evaluator or another tool)"
        have, want = _identity(r[key]), _identity(want)
        if key in PATH_KEYS:
            have, want = os.path.realpath(have), os.path.realpath(want)
        if have != want:
            return False, f"{key} differs from this run: recorded {have!r}, now {want!r}"
    want_sha = os.environ.get("CKPT_SHA256") == "1" and bool(r.get("ckpt_sha256"))
    try:
        fp = checkpoint_fingerprint(ckpt, sha256=want_sha)
    except OSError as e:
        return False, f"checkpoint not readable ({e})"
    if r["ckpt_abs"] != fp["ckpt_abs"]:
        return False, f"checkpoint path changed: recorded {r['ckpt_abs']}, now {fp['ckpt_abs']}"
    if r["ckpt_size"] != fp["ckpt_size"]:
        return False, f"checkpoint size changed: recorded {r['ckpt_size']} bytes, now {fp['ckpt_size']}"
    if r["ckpt_mtime_ns"] != fp["ckpt_mtime_ns"]:
        return False, (f"checkpoint mtime changed: recorded {r.get('ckpt_mtime', r['ckpt_mtime_ns'])}, "
                       f"now {fp['ckpt_mtime']}")
    if want_sha and r["ckpt_sha256"] != fp["ckpt_sha256"]:
        return False, "checkpoint sha256 changed"
    return True, (f"checkpoint unchanged (path/size/mtime match); val/psnr {r['val_psnr']:.2f} dB "
                  f"from {r.get('timestamp', 'unknown time')}")


# ---- CSVLogger output of one evaluation run ----------------------------------------
def metrics_files(log_dir: str) -> dict:
    """Every metrics.csv under a CSVLogger tree (save_dir/name/version_*/) -> {path: [inode, mtime_ns, size]}."""
    found = {}
    for root, _dirs, files in os.walk(log_dir):
        if "metrics.csv" in files:
            path = os.path.join(root, "metrics.csv")
            st = os.stat(path)
            found[path] = [st.st_ino, st.st_mtime_ns, st.st_size]
    return found


def fresh_metrics_csv(log_dir: str, prior: dict):
    """The newest metrics.csv under log_dir that was NOT in `prior` (a metrics_files() snapshot
    taken before train.py ran) with the same inode/mtime/size — i.e. one this run wrote.
    None when the run left no new file (the caller must fail rather than read a stale one)."""
    now = metrics_files(log_dir)
    fresh = [p for p, ident in now.items() if prior.get(p) != ident]
    return max(fresh, key=lambda p: now[p][1]) if fresh else None


# ---- table -----------------------------------------------------------------------
def load_results(results_dir: str) -> dict:
    """<mat>_<model>.json files -> {material: {model: record}}; unreadable files are reported and skipped."""
    got = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
            mat, model = str(r["material"]), r["model"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"warning: ignoring unreadable result {path}: {e!r}", file=sys.stderr)
            continue
        got.setdefault(mat, {})[model] = r
    return got


def collect(results_dir: str, table: Table, materials=None, models=None,
            tolerance_db: float = DEFAULT_TOLERANCE_DB, missing_cells=()) -> int:
    """Print the repro-vs-paper table for MATERIALS x MODELS and return the exit code (0/6/7/2)."""
    materials = list(materials or table.materials)
    models = list(models or table.models)
    unknown = [m for m in materials if m not in table.paper] + [m for m in models if m not in table.models]
    if unknown or tolerance_db < 0:
        print(f"error: no paper value for {unknown}; known materials {table.materials}, models {table.models}"
              if unknown else f"error: negative tolerance {tolerance_db}", file=sys.stderr)
        return EXIT_BAD_ARGS
    expected = {(mat, m) for mat in materials for m in models}
    forced_missing = set(missing_cells)          # cells whose checkpoint the driver could not find
    avg_tol = tolerance_db + AVG_ROUNDING_SLACK_DB
    got = load_results(results_dir)

    w, cw = table.name_width, 25
    header = f"{'material':>{w}} | " + " | ".join(f"{m:>{cw}}" for m in table.models)
    sub = f"{'':>{w}} | " + " | ".join(f"{'repro/paper/diff  status':>{cw}}" for _ in table.models)
    print(f"\n=== {table.title} ===")
    print(f"results: {results_dir}")
    print(f"tolerance: |repro - paper| <= {tolerance_db:.2f} dB per cell "
          f"(average row: <= {avg_tol:.2f} dB, incl. rounding slack of the paper's 2-decimal averages)")
    print(header)
    print(sub)
    print("-" * len(header))

    n_pass, failed, missing, ignored, leftover = 0, [], [], [], []
    sums, counts = {m: 0.0 for m in table.models}, {m: 0 for m in table.models}
    for mat in table.materials:
        cells = []
        for m in table.models:
            paper = table.paper[mat][m]
            rec = got.get(mat, {}).get(m)
            repro = rec.get("val_psnr") if isinstance(rec, dict) else None
            if (mat, m) not in expected:
                if repro is not None:
                    ignored.append(f"{mat}/{m}")
                cells.append(f"{'--':>5}/{paper:5.2f}/{'--':>5} {'n/a':>7}")
            elif f"{mat}/{m}" in forced_missing or repro is None:
                if repro is not None:
                    leftover.append(f"{mat}/{m}")
                missing.append(f"{mat}/{m}")
                cells.append(f"{'--':>5}/{paper:5.2f}/{'--':>5} {'MISSING':>7}")
            else:
                diff = repro - paper
                ok = abs(diff) <= tolerance_db + 1e-9
                if ok:
                    n_pass += 1
                else:
                    failed.append((f"{mat}/{m}", repro, paper, diff, tolerance_db))
                sums[m] += repro
                counts[m] += 1
                cells.append(f"{repro:5.2f}/{paper:5.2f}/{diff:+5.2f} {'PASS' if ok else 'FAIL':>7}")
        print(f"{mat:>{w}} | " + " | ".join(cells))

    avg_cells, n = [], len(table.materials)
    for m in table.models:
        if counts[m] == n:
            avg = sums[m] / n
            diff = avg - table.paper_avg[m]
            ok = abs(diff) <= avg_tol + 1e-9
            if not ok:
                failed.append((f"average/{m}", avg, table.paper_avg[m], diff, avg_tol))
            avg_cells.append(f"{avg:5.2f}/{table.paper_avg[m]:5.2f}/{diff:+5.2f} {'PASS' if ok else 'FAIL':>7}")
        else:
            avg_cells.append(f"{'--':>5}/{table.paper_avg[m]:5.2f}/{'--':>5} {f'({counts[m]}/{n})':>7}")
    print("-" * len(header))
    print(f"{'average':>{w}} | " + " | ".join(avg_cells))
    print()

    n_cell_fail = sum(1 for c in failed if not c[0].startswith("average/"))
    print(f"cells: {len(expected)} expected, {len(expected) - len(missing)} present, {len(missing)} missing; "
          f"{n_pass} PASS, {n_cell_fail} FAIL (tolerance {tolerance_db:.2f} dB)")
    if ignored:
        print(f"note: {len(ignored)} result(s) outside the requested MATERIALS x MODELS were not checked: "
              + " ".join(ignored))
    if leftover:
        print("note: leftover result(s) ignored because the driver found no checkpoint for: " + " ".join(leftover))
    if missing:
        print(f"MISSING cells ({len(missing)}): " + " ".join(missing))
    for cell, repro, paper, diff, tol in failed:
        print(f"FAIL {cell}: repro {repro:.4f} vs paper {paper:.2f} ({diff:+.4f} dB, |diff| > {tol:.2f} dB)")
    if failed:
        print(f"RESULT: FAIL — {len(failed)} cell(s) beyond tolerance (exit {EXIT_TOLERANCE})")
        return EXIT_TOLERANCE
    if missing:
        print(f"RESULT: FAIL — {len(missing)} expected cell(s) missing (exit {EXIT_MISSING_CELLS})")
        return EXIT_MISSING_CELLS
    print("RESULT: PASS — all expected cells within tolerance")
    return 0


def main(argv=None, table: Table = OURS) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=table.default_dir)
    p.add_argument("--materials", default="", help="space-separated subset (default: every paper material)")
    p.add_argument("--models", default="", help="space-separated subset (default: every paper column)")
    p.add_argument("--tolerance-db", type=float, default=None,
                   help=f"max |repro - paper| per cell (default: env TOLERANCE_DB or {DEFAULT_TOLERANCE_DB})")
    p.add_argument("--missing", default="",
                   help="'mat/model' cells whose checkpoint was missing: reported MISSING even if a leftover JSON exists")
    p.add_argument("--reuse-check", nargs=2, metavar=("RESULT_JSON", "CKPT"),
                   help="print why RESULT_JSON may (exit 0) or may not (exit 1) be reused for CKPT")
    p.add_argument("--expect", nargs="*", default=[], metavar="KEY=VALUE",
                   help="with --reuse-check: identifiers the result must record (material=, model=, "
                        "experiment=, dataset_folder=, overrides=, ...); any mismatch -> exit 1")
    p.add_argument("--metrics-snapshot", metavar="CSV_LOGGER_DIR",
                   help="print metrics_files(CSV_LOGGER_DIR) as JSON (eval_stage2*.sh take it before train.py)")
    a = p.parse_args(argv)
    if a.metrics_snapshot is not None:
        print(json.dumps(metrics_files(a.metrics_snapshot)))
        return 0
    if a.reuse_check:
        bad = [e for e in a.expect if "=" not in e]
        if bad:
            print(f"error: --expect takes KEY=VALUE pairs, got {bad}", file=sys.stderr)
            return EXIT_BAD_ARGS
        ok, reason = check_reuse(*a.reuse_check, expect=dict(e.split("=", 1) for e in a.expect))
        print(reason)
        return 0 if ok else 1
    if a.expect:
        print("error: --expect is only meaningful with --reuse-check", file=sys.stderr)
        return EXIT_BAD_ARGS
    tol = a.tolerance_db
    if tol is None:
        try:
            tol = float(os.environ.get("TOLERANCE_DB", DEFAULT_TOLERANCE_DB))
        except ValueError:
            print(f"error: TOLERANCE_DB={os.environ['TOLERANCE_DB']!r} is not a number", file=sys.stderr)
            return EXIT_BAD_ARGS
    return collect(a.results_dir, table, a.materials.split() or None, a.models.split() or None,
                   tol, a.missing.split())


if __name__ == "__main__":
    sys.exit(main())
