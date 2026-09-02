#!/bin/bash
# ---------------------------------------------------------------------------
# Stage-2 checkpoint evaluation on a RoboCloth material.
#
# Runs main.py with model.test=True: this loads ALL weights from the trained
# stage-2 checkpoint (except the emitter buffers) and executes the exact
# Stage2Trainer.validation_step over the full held-out validation split
# (the same fixed 80/20 split, seed 42, used during training):
#   - renders every validation view with the trained model (spp = 16),
#   - saves GT / prediction images exactly like training-time validation
#     (images/gt_view_<i>_0.png and images/result_view_<i>_0_psnr<PSNR>.png),
#   - computes per-view PSNR (peak = per-view GT max over visible pixels) and
#     averages it over the full validation set -> the paper's val/psnr metric.
#
# The scalar result is parsed from the CSV logger output and written to
#   $OUTPUT_ROOT/eval_results/<MAT_ID>_<TAG>.json
# (atomically, via <name>.json.tmp + rename) together with provenance —
# checkpoint path/size/mtime, git HEAD, timestamp, host, config identifiers —
# so run_table_ours.sh can tell a stale result from a reusable one.
#
# Usage:
#   DATA_ROOT=/path/to/capture_data bash scripts/eval_stage2.sh <MAT_ID> <CKPT> [TAG]
#
#   <CKPT> is a trained stage-2 checkpoint, e.g.
#     checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt   (released checkpoints), or
#     $OUTPUT_ROOT/Stage2_Ours145_from_Ours_run_1/training/model_0.20_0.20/last.ckpt
#   [TAG] labels the result file (default: checkpoint filename prefix before
#     "_epoch", i.e. Ours/Bonn/MERL/PBR for the released checkpoints).
#     TAG=PBR selects the Disney-PBR architecture; anything else the neural one.
#
# Knobs:
#   SAVE_ALL_VIEWS=1      save images for every validation view (default: first 20)
#   CKPT_SHA256=1         also record the checkpoint's sha256 in the result JSON
#   ROBOCLOTH_PYTHON=...  interpreter (default: `python` of the active env; the
#                         generic PYTHON variable is deliberately ignored)
#
# Fails closed: interpreter not executable -> exit 2 (before anything runs);
# checkpoint missing -> exit 4; a non-zero train.py exits this script with the
# same code; no metrics.csv written by THIS run -> exit 3 (EXP_NAME is reused
# across runs, so metrics.csv files left by earlier runs are never read);
# no val/psnr in it -> exit 1. No JSON is written in any of these cases.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPTS_DIR/../training"

MAT_ID=${1:?usage: eval_stage2.sh <MAT_ID> <CKPT> [TAG]}
CKPT=${2:?usage: eval_stage2.sh <MAT_ID> <CKPT> [TAG]}
TAG=${3:-$(basename "$CKPT" | sed 's/_epoch.*//;s/\.ckpt//')}

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root (folder containing <mat_id>/ subfolders)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/milestone3}
EXP_NAME=${EXP_NAME:-Eval_Ours${MAT_ID}_${TAG}}
VALID_NUM=$([ "${SAVE_ALL_VIEWS:-0}" = "1" ] && echo -1 || echo 20)
EXPERIMENT=$([ "$TAG" = "PBR" ] && echo eval_stage2_pbr || echo eval_stage2)
PYTHON=${ROBOCLOTH_PYTHON:-python}
COLLECT=$SCRIPTS_DIR/collect_eval_results.py   # owns the result-JSON contract

[ -x "$(command -v "$PYTHON" || true)" ] \
    || { echo "[eval_stage2] ERROR: interpreter not found or not executable: $PYTHON (set ROBOCLOTH_PYTHON)" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "[eval_stage2] ERROR: checkpoint not found: $CKPT" >&2; exit 4; }

DATASET_FOLDER=$DATA_ROOT/$MAT_ID
EMITTER_CALIB=${EMITTER_CALIB:-$DATASET_FOLDER/emitter_calibration.json}
[ -f "$EMITTER_CALIB" ] || EMITTER_CALIB=$DATA_ROOT/emitter_calibration.json

EXP_DIR=$OUTPUT_ROOT/$EXP_NAME
mkdir -p "$EXP_DIR"
# The CSV logger writes $EXP_DIR/$EXP_NAME/version_<N>/metrics.csv, and EXP_NAME is reused
# across runs (possibly with another checkpoint). Snapshot what is there now so that only
# a metrics.csv that appears or changes after train.py can count as this run's output.
PRIOR_METRICS=$("$PYTHON" "$COLLECT" --metrics-snapshot "$EXP_DIR/$EXP_NAME")

"$PYTHON" train.py +experiment=$EXPERIMENT \
    dataset_folder="$DATASET_FOLDER" \
    renderer.emitter.direction_json="$EMITTER_CALIB" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$EXP_DIR" \
    experiment_name="$EXP_NAME" \
    data.valid_num=$VALID_NUM \
    model.ckpt_path="$CKPT" \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
    '~model.logger.project' \
    "${@:4}"

# ---- collect the scalar metrics from THIS run's CSV logger output ------------
RESULTS_DIR=$OUTPUT_ROOT/eval_results
mkdir -p "$RESULTS_DIR"
# The JSON is written by collect_eval_results.write_result (scripts/), which owns the
# result schema shared with run_table_ours.sh's stale-result check.
PRIOR_METRICS=$PRIOR_METRICS "$PYTHON" - "$EXP_DIR/$EXP_NAME" "$RESULTS_DIR/${MAT_ID}_${TAG}.json" "$MAT_ID" "$TAG" \
    "$CKPT" "$SCRIPTS_DIR" "$EXPERIMENT" "$DATASET_FOLDER" "$EXP_NAME" "$VALID_NUM" "${@:4}" <<'EOF'
import csv, json, os, sys
log_dir, out_path, mat, tag, ckpt, scripts_dir, experiment, dataset_folder, exp_name, valid_num = sys.argv[1:11]
sys.path.insert(0, scripts_dir)
from collect_eval_results import fresh_metrics_csv, write_result
prior = json.loads(os.environ["PRIOR_METRICS"])          # metrics.csv files that predate train.py
csv_path = fresh_metrics_csv(log_dir, prior)
if csv_path is None:
    print(f"[eval_stage2] ERROR: train.py exited 0 but wrote no new metrics.csv under {log_dir} "
          f"({len(prior)} metrics.csv from earlier runs of {exp_name} ignored)", file=sys.stderr)
    sys.exit(3)
vals = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            if v not in (None, "") and k.startswith("val/"):
                vals[k] = float(v)
if vals.get("val/psnr") is None:
    sys.exit(f"[eval_stage2] ERROR: no val/psnr logged in {csv_path} (val/ columns seen: {sorted(vals)})")
res = write_result(out_path, material=mat, model=tag, ckpt=ckpt,
                   val_psnr=vals["val/psnr"], val_loss=vals.get("val/loss"),
                   experiment=experiment, dataset_folder=os.path.abspath(dataset_folder), exp_name=exp_name,
                   valid_num=int(valid_num), metrics_csv=csv_path, overrides=sys.argv[11:])
loss = f"{res['val_loss']:.4f}" if res["val_loss"] is not None else "n/a"
print(f"[eval_stage2] material {mat} / {tag}: val/psnr = {res['val_psnr']:.2f} dB "
      f"(val/loss = {loss})  ->  {out_path}")
EOF
