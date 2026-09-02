#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end CI check for the STAGE-1 DATA PATH.
#
# scripts/smoke_test.sh assumes DATA_ROOT already exists, so it cannot catch a
# broken downloader or a changed upstream layout.  This script starts from an
# EMPTY directory and exercises the real chain:
#
#   1. scripts/download_dataset_stage1.sh  — exactly the materials in MATERIALS
#   2. structural validation               — every file stage 1 opens is present,
#      non-empty and well-formed (npz arrays/dtypes/shapes, JSON records,
#      point_metadata vs. npz consistency, globals copied to the root)
#   3. one short stage-1 epoch via scripts/train_stage1.sh (CSVLogger)
#      — train/total_loss must be finite and a checkpoint must be written
#
# Requirements
#   * network access to huggingface.co and the `hf` CLI (huggingface_hub, part
#     of envs/training.txt) on PATH or next to ROBOCLOTH_PYTHON
#   * ~5 GB free disk under CI_DATA_ROOT (two sparse materials + globals)
#   * step 3 only: one CUDA GPU (~48 GB with the paper batch size; pass e.g.
#     CI_TRAIN_ARGS="data.rays_num=100000" on smaller cards) and the robocloth
#     training environment (torch 2.5.1 / pytorch-lightning 1.9.5)
#
# Inputs (environment variables)
#   CI_ROOT           parent of every default location below
#                     (default $TMPDIR/robocloth-ci, i.e. /tmp/robocloth-ci)
#   CI_DATA_ROOT      download target; must be absent or EMPTY.  Symlinks are
#                     resolved before the check (a link to a populated
#                     directory counts as non-empty)      (default $CI_ROOT/data)
#   CI_OUTPUT_ROOT    logs, training list, training outputs (default $CI_ROOT/out)
#   MATERIALS         space-separated numeric material IDs (default "145 226")
#   ROBOCLOTH_PYTHON  python of the training env — a path, or a bare command
#                     name looked up on PATH (default `python`); also runs the
#                     numpy-based validation.  Step 3 puts a job-local `python`
#                     wrapper that execs it first on PATH, so a conda prefix, a
#                     venv or a bare /usr/bin/python3.x all work
#   CI_FRESH=1        empty a non-empty CI_DATA_ROOT first (its contents, the
#                     directory / symlink itself stays).  Refused unless the
#                     resolved path is strictly below CI_WIPE_PREFIX
#                     (default $CI_ROOT)
#   CI_SKIP_TRAIN=1   stop after validation (no GPU / training env needed)
#   CI_TRAIN_BATCHES  limit_train_batches for the single epoch (default 50)
#   CI_TRAIN_ARGS     extra Hydra overrides appended to the training command
#   HF_HOME           hf cache; defaults to $CI_OUTPUT_ROOT/hf_home if unset
#
# Re-running into the same CI_OUTPUT_ROOT is safe: step 3 first removes the
# previous run's $CI_OUTPUT_ROOT/ci_stage1 (and the python wrapper) and only
# accepts a metrics.csv / checkpoint written after this run's start marker, so
# a run that trains nothing cannot pass on an earlier run's files.
#
# Usage
#   bash scripts/ci_stage1_datapath.sh                     # full run
#   CI_SKIP_TRAIN=1 bash scripts/ci_stage1_datapath.sh     # download + validate only
#   CUDA_VISIBLE_DEVICES=1 ROBOCLOTH_PYTHON=/path/to/robocloth-env/bin/python \
#       CI_FRESH=1 bash scripts/ci_stage1_datapath.sh
#
# Exit codes: 0 PASS · 1 precondition/download failure · 2 validation failure
#             · 3 training failure
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
CI_ROOT=${CI_ROOT:-${TMPDIR:-/tmp}/robocloth-ci}
CI_DATA_ROOT=${CI_DATA_ROOT:-$CI_ROOT/data}
CI_OUTPUT_ROOT=${CI_OUTPUT_ROOT:-$CI_ROOT/out}
MATERIALS=${MATERIALS:-"145 226"}
CI_WIPE_PREFIX=${CI_WIPE_PREFIX:-$CI_ROOT}
CI_TRAIN_BATCHES=${CI_TRAIN_BATCHES:-50}
CI_TRAIN_ARGS=${CI_TRAIN_ARGS:-}
EXP_NAME=ci_stage1

log() { printf '[ci_stage1_datapath] %s\n' "$*"; }
die() { printf '[ci_stage1_datapath] ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

# ---- preconditions ---------------------------------------------------------
for v in CI_DATA_ROOT CI_OUTPUT_ROOT; do
    [[ ${!v} == /* ]] || die "$v must be an absolute path (got '${!v}')"
done
read -r -a IDS <<< "$MATERIALS"
(( ${#IDS[@]} > 0 )) || die "MATERIALS is empty"
for id in "${IDS[@]}"; do
    [[ $id =~ ^[0-9]+$ ]] || die "MATERIALS must be numeric material IDs (got '$id')"
done

# ROBOCLOTH_PYTHON: an executable path, or a bare command name looked up on PATH
ROBOCLOTH_PYTHON=${ROBOCLOTH_PYTHON:-python}
if [[ $ROBOCLOTH_PYTHON != */* ]]; then
    resolved=$(command -v -- "$ROBOCLOTH_PYTHON" || true)   # a function/builtin has no slash
    [[ $resolved == */* ]] || die "ROBOCLOTH_PYTHON='$ROBOCLOTH_PYTHON' is not an executable on PATH"
    ROBOCLOTH_PYTHON=$resolved
fi
[[ -x $ROBOCLOTH_PYTHON && ! -d $ROBOCLOTH_PYTHON ]] \
    || die "ROBOCLOTH_PYTHON is not an executable python (got '$ROBOCLOTH_PYTHON')"
"$ROBOCLOTH_PYTHON" -c 'import numpy' 2>/dev/null || die "$ROBOCLOTH_PYTHON cannot import numpy"
PY_BIN_DIR=$(cd "$(dirname "$ROBOCLOTH_PYTHON")" && pwd -P)
# absolute from here on (train_stage1.sh cd's away); the basename is kept
# unresolved on purpose — a venv's bin/python is a symlink to its base python
ROBOCLOTH_PYTHON=$PY_BIN_DIR/$(basename "$ROBOCLOTH_PYTHON")

# `hf` CLI: whatever is on PATH wins, then the training env's own bin dir.
if command -v hf >/dev/null 2>&1; then
    HF_BIN=$(command -v hf)
elif [[ -x $PY_BIN_DIR/hf ]]; then
    HF_BIN=$PY_BIN_DIR/hf
else
    die "'hf' CLI not found on PATH or in $PY_BIN_DIR (pip install huggingface_hub)"
fi

# ---- (a) the test must start from an empty data directory ------------------
# Every check runs on the RESOLVED path: `[[ -d ]]` follows a symlinked root
# but `find <link>` does not descend into it (no -H), so an unresolved check
# lets a link to a populated directory through.
if [[ -L $CI_DATA_ROOT && ! -e $CI_DATA_ROOT ]]; then
    die "CI_DATA_ROOT is a dangling symlink: $CI_DATA_ROOT -> $(readlink -- "$CI_DATA_ROOT")"
fi
if [[ -e $CI_DATA_ROOT && ! -d $CI_DATA_ROOT ]]; then
    die "CI_DATA_ROOT exists and is not a directory: $CI_DATA_ROOT"
fi
REAL_DATA_ROOT=$(realpath -m -- "$CI_DATA_ROOT")
shown_root=$CI_DATA_ROOT
[[ $REAL_DATA_ROOT == "$CI_DATA_ROOT" ]] || shown_root="$CI_DATA_ROOT -> $REAL_DATA_ROOT"
if [[ -d $REAL_DATA_ROOT && -n $(find -H "$REAL_DATA_ROOT" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    if [[ ${CI_FRESH:-0} != 1 ]]; then
        die "CI_DATA_ROOT is non-empty: $shown_root — this test must start from an empty directory. Point CI_DATA_ROOT at an empty/absent directory, or set CI_FRESH=1 to wipe it (only allowed below $CI_WIPE_PREFIX)."
    fi
    real_prefix=$(realpath -m -- "$CI_WIPE_PREFIX")
    case $REAL_DATA_ROOT in
        "$real_prefix"/?*) ;;
        *) die "CI_FRESH=1 refused: $REAL_DATA_ROOT is not below $real_prefix (set CI_WIPE_PREFIX explicitly to allow it)" ;;
    esac
    # contents only: the directory may be a symlink target or a mount point
    log "CI_FRESH=1: wiping $REAL_DATA_ROOT"
    find -H "$REAL_DATA_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi
mkdir -p "$REAL_DATA_ROOT" "$CI_OUTPUT_ROOT"
export HF_HOME=${HF_HOME:-$CI_OUTPUT_ROOT/hf_home}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

# ---- (b) real downloader, explicit material list ---------------------------
log "== [1/3] download: materials ${IDS[*]} -> $CI_DATA_ROOT (hf: $HF_BIN) =="
DL_LOG=$CI_OUTPUT_ROOT/download.log
if ! PATH="$(dirname "$HF_BIN"):$PATH" ROBOCLOTH_MATERIALS="${IDS[*]}" \
        bash "$REPO_ROOT/scripts/download_dataset_stage1.sh" "$CI_DATA_ROOT" 2>&1 | tee "$DL_LOG"; then
    die "download_dataset_stage1.sh failed (see $DL_LOG)"
fi

# ---- (c) validate the assembled layout -------------------------------------
# Mirrors what stage 1 actually opens: MultiMaterialDenseDataset
# (training/datasets/points.py) reads observations_structured.npz +
# scan_log.json + rotated_camera.json, the latent bank
# (training/models/neural_brdf_refactored.py) sizes itself from
# point_metadata.json, MultiAreaEmitter reads scan_log.json +
# emitter_calibration.json.
log "== [2/3] validate downloaded layout =="
VAL_LOG=$CI_OUTPUT_ROOT/validate.log
set +e
"$ROBOCLOTH_PYTHON" - "$CI_DATA_ROOT" "${IDS[@]}" <<'PY' 2>&1 | tee "$VAL_LOG"
import filecmp, json, os, sys
import numpy as np

root, ids = sys.argv[1], sys.argv[2:]
problems = []   # (kind, path, detail)

def missing(path, detail): problems.append(("MISSING", path, detail))
def invalid(path, detail): problems.append(("INVALID", path, detail))

REQUIRED = ["observations_structured.npz", "scan_log.json", "rotated_camera.json", "point_metadata.json"]
# array -> allowed dtype kinds; shapes are checked against K/V taken from rgbs
NPZ_KINDS = {"xyz": "f", "point_ids": "iu", "rgbs": "u", "cam_pos": "f", "light_pos": "f"}
# utils/io.load_camera_turntable_light_metadata + read_light_transforms
SCAN_LOG_KEYS = ["scan_id", "camera_id", "light_id", "filename", "position", "rotation_matrix",
                 "position_light", "rotation_matrix_light"]
ROTCAM_KEYS = ["camera_id", "position", "rotation_matrix"]   # utils/io.load_camera_metadata

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — report, don't crash the validator
        invalid(path, f"not valid JSON: {e}")
        return None

def check_records(path, keys, label):
    d = load_json(path)
    if d is None:
        return None
    if not isinstance(d, list) or not d:
        invalid(path, f"expected a non-empty JSON list of {label} records")
        return None
    for i, rec in enumerate(d):
        miss = [k for k in keys if not isinstance(rec, dict) or k not in rec]
        if miss:
            invalid(path, f"record {i} is missing keys {miss}")
            break
    first = d[0]
    if isinstance(first, dict) and all(k in first for k in ("position", "rotation_matrix")):
        if np.shape(first["position"]) != (3,) or np.shape(first["rotation_matrix"]) != (3, 3):
            invalid(path, f"record 0: position {np.shape(first['position'])} / rotation_matrix "
                          f"{np.shape(first['rotation_matrix'])}, expected (3,) / (3, 3)")
    return d

def check_npz(path):
    try:
        with np.load(path) as data:
            keys = set(data.files)
            miss = [k for k in NPZ_KINDS if k not in keys]
            if miss:
                invalid(path, f"missing arrays {miss} (found {sorted(keys)})")
                return None
            arrs = {k: data[k] for k in NPZ_KINDS}
    except Exception as e:  # noqa: BLE001
        invalid(path, f"numpy could not load it: {e}")
        return None
    rgbs = arrs["rgbs"]
    if rgbs.ndim != 3 or rgbs.shape[2] != 3:
        invalid(path, f"rgbs shape {rgbs.shape}, expected (K, V, 3)")
        return None
    K, V, _ = rgbs.shape
    if K == 0 or V == 0:
        invalid(path, f"empty observation tensor (K={K}, V={V})")
    for k, shp in {"xyz": (V, 3), "point_ids": (V,), "cam_pos": (K, 3), "light_pos": (K, 3)}.items():
        if arrs[k].shape != shp:
            invalid(path, f"{k} shape {arrs[k].shape} != {shp} (K={K}, V={V} from rgbs)")
    for k, kinds in NPZ_KINDS.items():
        if arrs[k].dtype.kind not in kinds:
            invalid(path, f"{k} dtype {arrs[k].dtype}, expected kind '{kinds}'")
    if rgbs.dtype != np.uint16:
        invalid(path, f"rgbs dtype {rgbs.dtype} != uint16")
    for k in ("xyz", "cam_pos", "light_pos"):
        if arrs[k].dtype.kind == "f" and not np.isfinite(arrs[k]).all():
            invalid(path, f"{k} contains NaN/inf")
    # 0 = unobserved; the loader silently drops a material with no valid observation
    n_obs = 0
    for s in range(0, K, 64):
        n_obs += int((rgbs[s:s + 64] != 0).any(axis=2).sum())
    if n_obs == 0:
        invalid(path, "no valid observations (every rgbs entry is 0)")
    return {"K": K, "V": V, "n_obs": n_obs, "bytes": os.path.getsize(path)}

summary = []
for mid in ids:
    mdir = os.path.join(root, mid)
    if not os.path.isdir(mdir):
        missing(mdir, f"material directory for requested id {mid}")
        continue
    ok = True
    for name in REQUIRED:
        p = os.path.join(mdir, name)
        if not os.path.isfile(p):
            missing(p, "required by stage 1"); ok = False
        elif os.path.getsize(p) == 0:
            invalid(p, "empty file"); ok = False
    if not ok:
        continue
    info = check_npz(os.path.join(mdir, "observations_structured.npz"))
    check_records(os.path.join(mdir, "scan_log.json"), SCAN_LOG_KEYS, "scan_log")
    check_records(os.path.join(mdir, "rotated_camera.json"), ROTCAM_KEYS, "rotated_camera")
    pm_path = os.path.join(mdir, "point_metadata.json")
    pm = load_json(pm_path)
    if pm is not None:
        n_pts = pm.get("num_points") if isinstance(pm, dict) else None
        if not isinstance(n_pts, int) or n_pts <= 0:
            invalid(pm_path, f"num_points must be a positive int (got {n_pts!r})")
        elif info is not None and n_pts != info["V"]:
            invalid(pm_path, f"num_points={n_pts} but observations_structured.npz has V={info['V']} "
                             f"points (latent bank would be mis-sized)")
    if info is not None:
        summary.append(f"  {mid}: K={info['K']} frames, V={info['V']} points, {info['n_obs']:,} valid obs, "
                       f"npz {info['bytes'] / 1e9:.2f} GB, point_metadata={pm}")

# globals/* must have been copied to the dataset root by the downloader
gdir = os.path.join(root, "globals")
if not os.path.isdir(gdir):
    missing(gdir, "globals/ directory (hf --include 'globals/*')")
else:
    gfiles = sorted(f for f in os.listdir(gdir) if os.path.isfile(os.path.join(gdir, f)))
    if not gfiles:
        invalid(gdir, "globals/ is empty")
    for f in gfiles:
        dst = os.path.join(root, f)
        if not os.path.isfile(dst):
            missing(dst, f"copy of globals/{f} at the dataset root")
        elif not filecmp.cmp(os.path.join(gdir, f), dst, shallow=False):
            invalid(dst, f"differs from globals/{f}")
    summary.append(f"  globals: {gfiles}")
emit = os.path.join(root, "emitter_calibration.json")
if not os.path.isfile(emit):
    missing(emit, "LED falloff table (renderer.emitter.direction_json)")
else:
    d = load_json(emit)
    if isinstance(d, dict):
        miss = [k for k in ("resolution_degrees", "max_cam_rad_ratio", "data") if k not in d]
        if miss:
            invalid(emit, f"missing keys {miss}")
        elif not isinstance(d["data"], dict) or not d["data"]:
            invalid(emit, "'data' must be a non-empty {angle: ratio} table")
leftover = os.path.join(root, "materials")
if os.path.isdir(leftover) and os.listdir(leftover):
    invalid(leftover, f"still holds {sorted(os.listdir(leftover))} — downloader did not move them to the root")

print("\n".join(summary) if summary else "  (no material summary)")
if problems:
    print(f"VALIDATION FAILED — {len(problems)} problem(s) under {root}:")
    for kind, path, detail in problems:
        print(f"  {kind:8s} {path}  ({detail})")
    sys.exit(2)
print(f"validation OK: {len(ids)} material(s) + globals under {root}")
PY
rc=${PIPESTATUS[0]}
set -e
(( rc == 0 )) || die "validation failed (see $VAL_LOG)" 2

if [[ ${CI_SKIP_TRAIN:-0} == 1 ]]; then
    log "CI_SKIP_TRAIN=1 — skipping the training step"
    log "PASS (download + validation): data=$CI_DATA_ROOT materials=${IDS[*]} logs=$CI_OUTPUT_ROOT"
    exit 0
fi

# ---- (d) one short stage-1 epoch on the downloaded data --------------------
log "== [3/3] one stage-1 epoch (limit_train_batches=$CI_TRAIN_BATCHES) =="
EXP_DIR=$CI_OUTPUT_ROOT/$EXP_NAME
SHIM_DIR=$CI_OUTPUT_ROOT/pybin
# A rerun into the same CI_OUTPUT_ROOT is judged on its own files, never on a
# previous run's checkpoint / metrics.csv: remove that run's experiment dir and
# wrapper first — only ever strictly below the resolved CI_OUTPUT_ROOT, as with
# the CI_FRESH gate — and stamp a start marker; the assertions after training
# accept nothing older than it.
REAL_OUTPUT_ROOT=$(realpath -m -- "$CI_OUTPUT_ROOT")
for stale in "$EXP_DIR" "$SHIM_DIR"; do
    [[ -e $stale || -L $stale ]] || continue
    real_stale=$(realpath -m -- "$stale")
    case $real_stale in
        "$REAL_OUTPUT_ROOT"/?*) ;;
        *) die "refusing to remove $stale: it resolves to $real_stale, not below $REAL_OUTPUT_ROOT" ;;
    esac
    log "removing previous run's $stale"
    rm -rf -- "$stale"
    [[ ! -e $stale && ! -L $stale ]] || die "could not remove previous run's $stale"
done
START_MARK=$CI_OUTPUT_ROOT/train_start.stamp
touch -- "$START_MARK"
LIST=$CI_OUTPUT_ROOT/ci_training_list.txt
printf '%s\n' "${IDS[@]}" > "$LIST"
TRAIN_LOG=$CI_OUTPUT_ROOT/train.log
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi
read -r -a EXTRA_ARGS <<< "$CI_TRAIN_ARGS"
# train_stage1.sh calls a bare `python`: a job-local wrapper that execs
# ROBOCLOTH_PYTHON goes first on PATH.  A wrapper script, not a symlink — a
# `python` symlink with no pyvenv.cfg beside it starts a venv interpreter as
# its base python.  PY_BIN_DIR follows so the env's own hf/wandb still resolve.
mkdir -p "$SHIM_DIR"
printf '#!/bin/bash\nexec %q "$@"\n' "$ROBOCLOTH_PYTHON" > "$SHIM_DIR/python"
chmod +x "$SHIM_DIR/python"
set +e
PATH="$SHIM_DIR:$PY_BIN_DIR:$PATH" DATA_ROOT=$CI_DATA_ROOT OUTPUT_ROOT=$CI_OUTPUT_ROOT \
TRAINING_LIST=$LIST EXP_NAME=$EXP_NAME \
bash "$REPO_ROOT/scripts/train_stage1.sh" \
    model.trainer.max_epochs=1 model.trainer.check_val_every_n_epoch=1 \
    model.trainer.limit_train_batches="$CI_TRAIN_BATCHES" \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' '~model.logger.project' \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$TRAIN_LOG"
rc=${PIPESTATUS[0]}
set -e
(( rc == 0 )) || die "stage-1 training exited with $rc (see $TRAIN_LOG)" 3

if ! "$ROBOCLOTH_PYTHON" - "$EXP_DIR" "$START_MARK" <<'PY'
import csv, glob, math, os, sys
exp, t0 = sys.argv[1], os.stat(sys.argv[2]).st_mtime_ns   # t0: this run's start marker

def this_run(pattern):
    """exp/**/pattern files written since t0, oldest first, plus how many older ones were ignored."""
    found = glob.glob(os.path.join(exp, "**", pattern), recursive=True)
    fresh = sorted((p for p in found if os.stat(p).st_mtime_ns >= t0), key=lambda p: os.stat(p).st_mtime_ns)
    return fresh, len(found) - len(fresh)

def ignored(n):
    return f" ({n} older file(s) from an earlier run ignored)" if n else ""

csvs, n_old = this_run("metrics.csv")
if not csvs:
    print(f"FAIL: no metrics.csv written by this run under {exp}{ignored(n_old)}"); sys.exit(3)
csvf = csvs[-1]
with open(csvf) as f:
    vals = [float(r["train/total_loss"]) for r in csv.DictReader(f) if r.get("train/total_loss")]
if not vals:
    print(f"FAIL: no train/total_loss rows in {csvf}"); sys.exit(3)
bad = [v for v in vals if not math.isfinite(v)]
if bad:
    print(f"FAIL: non-finite train/total_loss in {csvf}: {bad[:5]}"); sys.exit(3)
ckpts, n_old = this_run("*.ckpt")
if not ckpts:
    print(f"FAIL: no checkpoint (*.ckpt) written by this run under {exp}{ignored(n_old)}"); sys.exit(3)
print(f"train/total_loss: {len(vals)} values, first {vals[0]:.4f} -> last {vals[-1]:.4f} (all finite)")
print(f"metrics    : {csvf}")
print("checkpoints: " + ", ".join(ckpts))
PY
then
    die "post-training assertions failed (see $TRAIN_LOG)" 3
fi

# ---- (e) summary -----------------------------------------------------------
log "PASS: stage-1 data path OK"
log "  data root     : $CI_DATA_ROOT (materials ${IDS[*]})"
log "  training list : $LIST"
log "  outputs/logs  : $CI_OUTPUT_ROOT ($EXP_NAME/, download.log, validate.log, train.log)"
