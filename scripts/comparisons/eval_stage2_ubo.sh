#!/bin/bash
# ---------------------------------------------------------------------------
# Stage-2 checkpoint evaluation on a UBO2014 BTF material.
#
# Runs main.py with model.test=True: loads ALL weights from the trained
# stage-2 checkpoint (except emitter buffers) and executes the exact
# training-time validation step over the fixed held-out BTF angle combinations
# (val_view_ratio 0.2, seed 42) — producing the val/psnr metric reported in
# the cross-dataset transfer table of the paper, plus the same GT/prediction
# visualizations the trainer saves during validation.
#
# Usage:
#   DATA_ROOT=/path/to/BTF bash eval_stage2_ubo.sh <material> <CKPT> [TAG]
#   e.g. eval_stage2_ubo.sh felt01 checkpoints/stage2/UBO/felt01/Bonn_epoch60.ckpt
#
#   DATA_ROOT is the flat folder of UBO2014 .btf files, as produced by
#   scripts/comparisons/download_ubo2014.sh (-> <root>/UBO2014).
#
# TAG defaults to the checkpoint filename prefix (Ours/Bonn/MERL/PBR);
# TAG=PBR selects the Disney-PBR architecture.
# Results: $OUTPUT_ROOT/eval_results_ubo/<material>_<TAG>.json, written
# atomically (<name>.json.tmp + rename) with provenance — checkpoint
# path/size/mtime, git HEAD, timestamp, host, config identifiers — so
# run_table_ubo.sh can tell a stale result from a reusable one.
#
# Knobs: CKPT_SHA256=1 (also record the checkpoint sha256),
#        ROBOCLOTH_PYTHON=... (interpreter; default `python`, the generic PYTHON
#        variable is deliberately ignored).
# Fails closed: interpreter not executable -> exit 2; checkpoint missing -> exit 4;
# a non-zero train.py exits this script with the same code; no metrics.csv written
# by THIS run -> exit 3 (EXP_NAME is reused across runs, files left by earlier runs
# are never read); no val/psnr in it -> exit 1. No JSON is written in these cases.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPTS_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$SCRIPTS_DIR/../training"

MAT=${1:?usage: eval_stage2_ubo.sh <material> <CKPT> [TAG]}
CKPT=${2:?usage: eval_stage2_ubo.sh <material> <CKPT> [TAG]}
TAG=${3:-$(basename "$CKPT" | sed 's/_epoch.*//;s/\.ckpt//')}

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/full_experiments}
EXP_NAME=${EXP_NAME:-Eval_UBO_${MAT}_${TAG}}
BTF_FILE=${MAT}_W400xH400_L151xV151.btf
EXPERIMENT=$([ "$TAG" = "PBR" ] && echo eval_ubo_pbr || echo eval_ubo)
PYTHON=${ROBOCLOTH_PYTHON:-python}
COLLECT=$SCRIPTS_DIR/collect_eval_results.py   # owns the result-JSON contract

[ -x "$(command -v "$PYTHON" || true)" ] \
    || { echo "[eval_stage2_ubo] ERROR: interpreter not found or not executable: $PYTHON (set ROBOCLOTH_PYTHON)" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "[eval_stage2_ubo] ERROR: checkpoint not found: $CKPT" >&2; exit 4; }

EXP_DIR=$OUTPUT_ROOT/$EXP_NAME
mkdir -p "$EXP_DIR"
# The CSV logger writes $EXP_DIR/$EXP_NAME/version_<N>/metrics.csv, and EXP_NAME is reused
# across runs (possibly with another checkpoint). Snapshot what is there now so that only
# a metrics.csv that appears or changes after train.py can count as this run's output.
PRIOR_METRICS=$("$PYTHON" "$COLLECT" --metrics-snapshot "$EXP_DIR/$EXP_NAME")

"$PYTHON" train.py +experiment=$EXPERIMENT \
    dataset_folder="$DATA_ROOT" data.btf_filename=$BTF_FILE \
    output_folder="$OUTPUT_ROOT" exp_output_root_path="$EXP_DIR" \
    experiment_name="$EXP_NAME" model.ckpt_path="$CKPT" \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
    '~model.logger.project' \
    "${@:4}"

# ---- collect the scalar metrics from THIS run's CSV logger output ------------
RESULTS_DIR=$OUTPUT_ROOT/eval_results_ubo
mkdir -p "$RESULTS_DIR"
# The JSON is written by collect_eval_results.write_result (scripts/), which owns the
# result schema shared with run_table_ubo.sh's stale-result check.
PRIOR_METRICS=$PRIOR_METRICS "$PYTHON" - "$EXP_DIR/$EXP_NAME" "$RESULTS_DIR/${MAT}_${TAG}.json" "$MAT" "$TAG" \
    "$CKPT" "$SCRIPTS_DIR" "$EXPERIMENT" "$DATA_ROOT" "$BTF_FILE" "$EXP_NAME" "${@:4}" <<'EOF'
import csv, json, os, sys
log_dir, out_path, mat, tag, ckpt, scripts_dir, experiment, data_root, btf_file, exp_name = sys.argv[1:11]
sys.path.insert(0, scripts_dir)
from collect_eval_results import fresh_metrics_csv, write_result
prior = json.loads(os.environ["PRIOR_METRICS"])          # metrics.csv files that predate train.py
csv_path = fresh_metrics_csv(log_dir, prior)
if csv_path is None:
    print(f"[eval_stage2_ubo] ERROR: train.py exited 0 but wrote no new metrics.csv under {log_dir} "
          f"({len(prior)} metrics.csv from earlier runs of {exp_name} ignored)", file=sys.stderr)
    sys.exit(3)
vals = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            if v not in (None, "") and k.startswith("val/"):
                vals[k] = float(v)
if vals.get("val/psnr") is None:
    sys.exit(f"[eval_stage2_ubo] ERROR: no val/psnr logged in {csv_path} (val/ columns seen: {sorted(vals)})")
data_root = os.path.abspath(data_root)
res = write_result(out_path, material=mat, model=tag, ckpt=ckpt,
                   val_psnr=vals["val/psnr"], val_loss=vals.get("val/loss"),
                   experiment=experiment, dataset_folder=data_root, btf_path=os.path.join(data_root, btf_file),
                   exp_name=exp_name, metrics_csv=csv_path, overrides=sys.argv[11:])
print(f"[eval_stage2_ubo] {mat} / {tag}: val/psnr = {res['val_psnr']:.2f} dB -> {out_path}")
EOF
