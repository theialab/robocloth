#!/bin/bash
# ---------------------------------------------------------------------------
# Reproduce Table "Per-material reconstruction PSNR" (bottom block: the
# held-out Bonn / UBOFAB19 test materials 318, 377, 32, 226, 37) from the
# released stage-2 checkpoints.
#
# Evaluates 5 materials x 4 models (RoboCloth/Bonn/MERL decoder warm-starts +
# Disney-PBR baseline), then runs collect_eval_results_bonn.py, which prints
# the reproduced table next to the paper numbers with a PASS/FAIL verdict per
# cell.
#
# Usage:
#   DATA_ROOT=/path/to/Bonn_val CKPT_ROOT=/path/to/checkpoints/stage2/Bonn \
#       bash scripts/comparisons/run_table_bonn.sh
#
#   DATA_ROOT: the UBOFAB19 *validation measurements* (bonn_data/get_UBOFAB19_val_meas.sh)
#     with bonn_point_metadata.json generated in it (generate_bonn_metadata.py);
#     see docs/bonn.md. Checked before anything is evaluated.
#   CKPT_ROOT layout: $CKPT_ROOT/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
#     (exactly one checkpoint per material and model)
#   MATERIALS / MODELS can be overridden to evaluate a subset, e.g.
#     MATERIALS="318" MODELS="Ours" bash scripts/comparisons/run_table_bonn.sh
#
# Knobs:
#   OUTPUT_ROOT=...     results: $OUTPUT_ROOT/eval_results_bonn/<mat>_<model>.json
#   TOLERANCE_DB=0.05   max |repro - paper| per cell before the table FAILs
#   FORCE=1             re-evaluate every cell, ignoring existing results
#   ALLOW_MISSING=1     report missing checkpoints, evaluate the remaining
#                       cells anyway; those cells are marked MISSING in the
#                       table and the run still exits nonzero at the end
#   SAVE_ALL_VIEWS=1    save GT/prediction PNGs for every held-out view
#                       (default: the first 20; the metric always uses all)
#   LLS_SPP=<n>         pass model.lls_spp=<n> to train.py (Monte-Carlo samples
#                       of the linear-light-source views; the shipped
#                       stage2_bonn config resolves to 16). Recorded as an
#                       override, so results made with another value are
#                       never reused for this one.
#   CKPT_SHA256=1       also record / compare the checkpoint sha256
#   ROBOCLOTH_PYTHON=.. interpreter for train.py and the collector (default:
#                       `python`; the generic PYTHON variable is deliberately
#                       ignored)
#
# There is no separate eval_*.sh for Bonn: the evaluation of one cell is the
# evaluate_cell function below. It runs train.py with +experiment=stage2_bonn
# (stage2_bonn_pbr for PBR) plus model.test=true — the same switch
# eval_stage2.yaml applies to stage2.yaml — which loads ALL weights from the
# checkpoint (except emitter buffers) and executes Stage1Trainer_Bonn's
# validation step over the full fixed held-out split (val_view_ratio 0.2,
# seed 42; 133 of 668 images per material). The trainer logs val/poly_psnr
# (mean over the RGB views) and val/gray_psnr (mean over the pan + LLS views);
# the paper metric is their image-count-weighted mean = the mean per-view
# PSNR over all held-out views. The counts come from the material's EXR
# headers (collect_eval_results_bonn.val_split_counts). Everything is written
# to $RESULTS_DIR/<mat>_<model>.json (atomically, with provenance: checkpoint
# path/size/mtime, git HEAD, timestamp, host, config identifiers, both
# component PSNRs and the counts).
#
# The script can be interrupted and re-run: an existing result JSON is reused
# only if the checkpoint it records (path, size, mtime) is unchanged AND it was
# produced for this cell's configuration (material, model, experiment, dataset
# folder, same overrides); otherwise the cell is re-evaluated, the reason is
# logged, and the superseded JSON is kept aside as <name>.json.prev so that
# nothing but this run's own output can be accepted for the cell.
#
# Fails closed — exit codes:
#   2  bad environment: DATA_ROOT is not a directory / interpreter not found /
#      bonn_point_metadata.json or a required per-material file is missing
#   4  a checkpoint is missing or ambiguous (all cells are checked BEFORE any
#      evaluation starts; every offending (mat, model) is listed)
#   5  an evaluation failed or did not leave a valid result JSON for this
#      checkpoint (the re-run command is printed; the run stops there)
#   6  an expected table cell has no result           (collector)
#   7  a cell differs from the paper by more than TOLERANCE_DB (collector)
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$SCRIPT_DIR"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Bonn_val folder (UBOFAB19 validation measurements + bonn_point_metadata.json)}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the released stage-2 Bonn checkpoints (checkpoints/stage2/Bonn)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/full_experiments}
MATERIALS=${MATERIALS:-"318 377 32 226 37"}
MODELS=${MODELS:-"Ours Bonn MERL PBR"}
TOLERANCE_DB=${TOLERANCE_DB:-0.05}
FORCE=${FORCE:-0}
ALLOW_MISSING=${ALLOW_MISSING:-0}
VALID_NUM=$([ "${SAVE_ALL_VIEWS:-0}" = "1" ] && echo -1 || echo 20)
LLS_SPP=${LLS_SPP:-}
PYTHON=${ROBOCLOTH_PYTHON:-python}
# Test hook: an alternative evaluator with the <mat> <ckpt> <tag> CLI + result-JSON
# contract of evaluate_cell (tests/bonn_driver). Empty = evaluate_cell below.
EVAL_SCRIPT=${ROBOCLOTH_EVAL_SCRIPT:-}
COLLECT=$SCRIPT_DIR/collect_eval_results_bonn.py
RESULTS_DIR=$OUTPUT_ROOT/eval_results_bonn
# Extra Hydra overrides of this run (recorded in the result JSON as "overrides").
EXTRA=()
[ -n "$LLS_SPP" ] && EXTRA+=("model.lls_spp=$LLS_SPP")

[ -x "$(command -v "$PYTHON" || true)" ] \
    || { echo "[run_table_bonn] ERROR: interpreter not found or not executable: $PYTHON (set ROBOCLOTH_PYTHON)" >&2; exit 2; }
[ -d "$DATA_ROOT" ] || { echo "[run_table_bonn] ERROR: DATA_ROOT is not a directory: $DATA_ROOT" >&2; exit 2; }
DATA_ROOT=$(cd "$DATA_ROOT" && pwd -P)   # results record the dataset folder; compare it unambiguously
# The loader needs the per-material files and the model needs bonn_point_metadata.json for
# every requested material; refuse now rather than after the first multi-minute load.
"$PYTHON" "$COLLECT" --check-data "$DATA_ROOT" --materials "$MATERIALS" \
    || { echo "[run_table_bonn] ERROR: DATA_ROOT is not a usable Bonn folder (see docs/bonn.md)" >&2; exit 2; }
export OUTPUT_ROOT DATA_ROOT

# ---- evaluation of one cell -------------------------------------------------
# evaluate_cell <mat> <ckpt> <tag>: train.py (model.test=true) + result JSON.
# Called with `|| rc=$?`, so errexit is off inside: every step returns explicitly.
# Exit codes: train.py's own code; 3 = no new metrics.csv; 1 = metric missing.
evaluate_cell() {
  local mat=$1 ckpt=$2 tag=$3 experiment exp_name exp_dir prior rc
  experiment=$([ "$tag" = PBR ] && echo stage2_bonn_pbr || echo stage2_bonn)
  exp_name=Eval_Bonn_${mat}_${tag}
  exp_dir=$OUTPUT_ROOT/$exp_name
  mkdir -p "$exp_dir" || return 1
  # The CSV logger writes $exp_dir/$exp_name/version_<N>/metrics.csv, and exp_name is reused
  # across runs (possibly with another checkpoint). Snapshot what is there now so that only
  # a metrics.csv that appears or changes after train.py can count as this run's output.
  prior=$("$PYTHON" "$COLLECT" --metrics-snapshot "$exp_dir/$exp_name") || return 1
  rc=0
  (cd "$REPO_ROOT/training" && "$PYTHON" train.py +experiment=$experiment \
      dataset_folder="$DATA_ROOT" \
      data.overfit_mat_id=$mat \
      data.valid_num=$VALID_NUM \
      model.test=true \
      model.trainer.enable_checkpointing=false \
      model.ckpt_path="$ckpt" \
      output_folder="$OUTPUT_ROOT" \
      exp_output_root_path="$exp_dir" \
      experiment_name="$exp_name" \
      'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
      '~model.logger.project' \
      ${EXTRA[@]+"${EXTRA[@]}"}) || rc=$?
  [ $rc -eq 0 ] || return $rc
  mkdir -p "$RESULTS_DIR" || return 1
  # The JSON is written by collect_eval_results.write_result (scripts/), which owns the
  # result schema shared with the stale-result check below.
  PRIOR_METRICS=$prior "$PYTHON" - "$exp_dir/$exp_name" "$RESULTS_DIR/${mat}_${tag}.json" "$mat" "$tag" \
      "$ckpt" "$SCRIPT_DIR" "$experiment" "$DATA_ROOT" "$exp_name" "$VALID_NUM" ${EXTRA[@]+"${EXTRA[@]}"} <<'EOF'
import csv, json, os, sys
log_dir, out_path, mat, tag, ckpt, scripts_dir, experiment, data_root, exp_name, valid_num = sys.argv[1:11]
sys.path.insert(0, scripts_dir)
from collect_eval_results_bonn import bonn_val_psnr, val_split_counts   # + scripts/collect_eval_results.py
from collect_eval_results import fresh_metrics_csv, write_result
prior = json.loads(os.environ["PRIOR_METRICS"])          # metrics.csv files that predate train.py
csv_path = fresh_metrics_csv(log_dir, prior)
if csv_path is None:
    print(f"[run_table_bonn] ERROR: train.py exited 0 but wrote no new metrics.csv under {log_dir} "
          f"({len(prior)} metrics.csv from earlier runs of {exp_name} ignored)", file=sys.stderr)
    sys.exit(3)
vals = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            if v not in (None, "") and k.startswith("val/"):
                vals[k] = float(v)
n_poly, n_gray = val_split_counts(data_root, mat)        # from the EXR headers, same split as the loader
try:
    psnr = bonn_val_psnr(vals.get("val/poly_psnr"), vals.get("val/gray_psnr"), n_poly, n_gray)
except ValueError as e:
    sys.exit(f"[run_table_bonn] ERROR: {e} in {csv_path} (val/ columns seen: {sorted(vals)})")
res = write_result(out_path, material=mat, model=tag, ckpt=ckpt, val_psnr=psnr, val_loss=None,
                   experiment=experiment, dataset_folder=os.path.abspath(data_root), exp_name=exp_name,
                   valid_num=int(valid_num), metrics_csv=csv_path, overrides=sys.argv[11:],
                   val_poly_psnr=vals.get("val/poly_psnr"), val_gray_psnr=vals.get("val/gray_psnr"),
                   val_all_psnr=vals.get("val/all_psnr"), n_val_poly=n_poly, n_val_gray=n_gray,
                   n_val_views=n_poly + n_gray,
                   val_poly_loss=vals.get("val/poly_loss"), val_gray_loss=vals.get("val/gray_loss"))
print(f"[run_table_bonn] material {mat} / {tag}: val psnr = {res['val_psnr']:.2f} dB "
      f"(poly {n_poly} views: {vals.get('val/poly_psnr', float('nan')):.2f}, "
      f"gray {n_gray} views: {vals.get('val/gray_psnr', float('nan')):.2f})  ->  {out_path}")
EOF
}

# ---- 1. resolve every checkpoint before evaluating anything -------------------
cells=(); ckpts=(); missing=()
[ -d "$CKPT_ROOT" ] || echo "[run_table_bonn] ERROR: CKPT_ROOT is not a directory: $CKPT_ROOT" >&2
for mat in $MATERIALS; do
  for model in $MODELS; do
    matches=("$CKPT_ROOT/$mat/${model}"_epoch*.ckpt)
    if [ ! -f "${matches[0]}" ]; then
      missing+=("$mat/$model")
      echo "[run_table_bonn] MISSING checkpoint for material $mat / $model: expected $CKPT_ROOT/$mat/${model}_epoch<N>.ckpt" >&2
    elif [ ${#matches[@]} -gt 1 ]; then
      missing+=("$mat/$model")
      echo "[run_table_bonn] AMBIGUOUS checkpoint for material $mat / $model (keep exactly one): ${matches[*]}" >&2
    else
      cells+=("$mat/$model"); ckpts+=("${matches[0]}")
    fi
  done
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "[run_table_bonn] ${#missing[@]} cell(s) without a usable checkpoint under $CKPT_ROOT: ${missing[*]}" >&2
  if [ "$ALLOW_MISSING" != "1" ]; then
    echo "[run_table_bonn] aborting before any evaluation (exit 4). Set ALLOW_MISSING=1 to evaluate the remaining cells; the run still exits nonzero." >&2
    exit 4
  fi
  echo "[run_table_bonn] ALLOW_MISSING=1: evaluating the remaining ${#cells[@]} cell(s); missing cells will be marked MISSING and the run exits nonzero" >&2
fi

# ---- 2. evaluate (reusing a result only when its checkpoint is unchanged) -----
mkdir -p "$RESULTS_DIR"
for i in "${!cells[@]}"; do
  mat=${cells[$i]%/*}; model=${cells[$i]#*/}; ckpt=${ckpts[$i]}
  result=$RESULTS_DIR/${mat}_${model}.json
  # What a result must record to stand in for this cell: the identifiers evaluate_cell
  # writes for exactly this material / model / dataset with this run's overrides.
  expect=("material=$mat" "model=$model" "dataset_folder=$DATA_ROOT" "overrides=${EXTRA[*]-}"
          "experiment=$([ "$model" = PBR ] && echo stage2_bonn_pbr || echo stage2_bonn)")
  if [ "$FORCE" = "1" ]; then
    echo "[run_table_bonn] FORCE=1: re-evaluating material $mat / $model"
  elif reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table_bonn] reuse $result: $reason"
    continue
  else
    echo "[run_table_bonn] evaluating material $mat / $model: $reason"
  fi
  if [ -e "$result" ]; then
    # A superseded result must not stay where the post-evaluation check could accept it
    # (e.g. when the evaluator exits 0 without writing anything); keep it aside instead.
    mv -f "$result" "$result.prev"
    echo "[run_table_bonn] previous result kept as $result.prev"
  fi
  echo "[run_table_bonn] evaluating material $mat / $model ($ckpt)"
  rc=0
  if [ -n "$EVAL_SCRIPT" ]; then
    bash "$EVAL_SCRIPT" "$mat" "$ckpt" "$model" || rc=$?
  else
    evaluate_cell "$mat" "$ckpt" "$model" || rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "[run_table_bonn] ERROR: evaluation of material $mat / $model failed (exit $rc). Command:" >&2
    if [ -n "$EVAL_SCRIPT" ]; then
      echo "    DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT bash $EVAL_SCRIPT $mat $ckpt $model" >&2
    else
      echo "    MATERIALS=$mat MODELS=$model FORCE=1 DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT CKPT_ROOT=$CKPT_ROOT${LLS_SPP:+ LLS_SPP=$LLS_SPP} bash $SCRIPT_DIR/run_table_bonn.sh" >&2
    fi
    exit 5
  fi
  if ! reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table_bonn] ERROR: evaluation of material $mat / $model exited 0 but left no valid result for $ckpt at $result: $reason" >&2
    exit 5
  fi
done

# ---- 3. collect: every expected cell present and within tolerance -------------
rc=0
"$PYTHON" "$COLLECT" "$RESULTS_DIR" --materials "$MATERIALS" --models "$MODELS" \
    --tolerance-db "$TOLERANCE_DB" --missing "${missing[*]:-}" || rc=$?
if [ $rc -ne 0 ]; then
  echo "[run_table_bonn] FAILED (exit $rc) — see the table above" >&2
  exit $rc
fi
if [ ${#missing[@]} -gt 0 ]; then   # not reached in practice: the collector already exits 6
  echo "[run_table_bonn] FAILED (exit 4): ${#missing[@]} checkpoint(s) were missing" >&2
  exit 4
fi
echo "[run_table_bonn] PASS: all ${#cells[@]} cells reproduced within $TOLERANCE_DB dB of the paper"
