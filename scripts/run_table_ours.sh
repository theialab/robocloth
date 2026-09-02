#!/bin/bash
# ---------------------------------------------------------------------------
# Reproduce Table "Per-material reconstruction PSNR" (top block, our test set)
# from the released stage-2 checkpoints.
#
# Evaluates 5 materials x 4 models (RoboCloth/Bonn/MERL decoder warm-starts +
# Disney-PBR baseline) with eval_stage2.sh, then runs collect_eval_results.py,
# which prints the reproduced table next to the paper numbers with a
# PASS/FAIL verdict per cell.
#
# Usage:
#   DATA_ROOT=/path/to/capture_data CKPT_ROOT=/path/to/checkpoints/stage2/RoboCloth \
#       bash scripts/run_table_ours.sh
#
#   CKPT_ROOT layout: $CKPT_ROOT/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
#     (exactly one checkpoint per material and model)
#   MATERIALS / MODELS can be overridden to evaluate a subset, e.g.
#     MATERIALS="145" MODELS="Ours" bash scripts/run_table_ours.sh
#
# Knobs:
#   OUTPUT_ROOT=...     results: $OUTPUT_ROOT/eval_results/<mat>_<model>.json
#   TOLERANCE_DB=0.05   max |repro - paper| per cell before the table FAILs
#   FORCE=1             re-evaluate every cell, ignoring existing results
#   ALLOW_MISSING=1     report missing checkpoints, evaluate the remaining
#                       cells anyway; those cells are marked MISSING in the
#                       table and the run still exits nonzero at the end
#   ROBOCLOTH_PYTHON=.. interpreter for the collector (default: `python`; also
#                       honoured by eval_stage2.sh — the generic PYTHON variable
#                       is deliberately ignored)
#
# Each evaluation renders ~118 held-out views at spp 16 (roughly 10-30 min on
# a modern GPU); the full table is 20 evaluations. The script can be
# interrupted and re-run: an existing result JSON is reused only if the
# checkpoint it records (path, size, mtime) is unchanged AND it was produced
# for this cell's configuration (material, model, experiment, dataset folder,
# no extra overrides); otherwise the cell is re-evaluated, the reason is
# logged, and the superseded JSON is kept aside as <name>.json.prev so that
# nothing but this run's own output can be accepted for the cell.
#
# Fails closed — exit codes:
#   2  bad environment: DATA_ROOT is not a directory / interpreter not found
#   4  a checkpoint is missing or ambiguous (all cells are checked BEFORE any
#      evaluation starts; every offending (mat, model) is listed)
#   5  an evaluation failed or did not leave a valid result JSON for this
#      checkpoint (the failing command is printed; the run stops there)
#   6  an expected table cell has no result           (collector)
#   7  a cell differs from the paper by more than TOLERANCE_DB (collector)
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the released stage-2 checkpoints (checkpoints/stage2/RoboCloth)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/milestone3}
MATERIALS=${MATERIALS:-"226 314 370 145 452"}
MODELS=${MODELS:-"Ours Bonn MERL PBR"}
TOLERANCE_DB=${TOLERANCE_DB:-0.05}
FORCE=${FORCE:-0}
ALLOW_MISSING=${ALLOW_MISSING:-0}
PYTHON=${ROBOCLOTH_PYTHON:-python}
# Test hook: an alternative evaluator with eval_stage2.sh's CLI + result-JSON contract.
EVAL_SCRIPT=${ROBOCLOTH_EVAL_SCRIPT:-$SCRIPT_DIR/eval_stage2.sh}
COLLECT=$SCRIPT_DIR/collect_eval_results.py
RESULTS_DIR=$OUTPUT_ROOT/eval_results

[ -x "$(command -v "$PYTHON" || true)" ] \
    || { echo "[run_table] ERROR: interpreter not found or not executable: $PYTHON (set ROBOCLOTH_PYTHON)" >&2; exit 2; }
[ -d "$DATA_ROOT" ] || { echo "[run_table] ERROR: DATA_ROOT is not a directory: $DATA_ROOT" >&2; exit 2; }
DATA_ROOT=$(cd "$DATA_ROOT" && pwd -P)   # results record the dataset folder; compare it unambiguously
export OUTPUT_ROOT DATA_ROOT

# ---- 1. resolve every checkpoint before evaluating anything -------------------
cells=(); ckpts=(); missing=()
[ -d "$CKPT_ROOT" ] || echo "[run_table] ERROR: CKPT_ROOT is not a directory: $CKPT_ROOT" >&2
for mat in $MATERIALS; do
  for model in $MODELS; do
    matches=("$CKPT_ROOT/$mat/${model}"_epoch*.ckpt)
    if [ ! -f "${matches[0]}" ]; then
      missing+=("$mat/$model")
      echo "[run_table] MISSING checkpoint for material $mat / $model: expected $CKPT_ROOT/$mat/${model}_epoch<N>.ckpt" >&2
    elif [ ${#matches[@]} -gt 1 ]; then
      missing+=("$mat/$model")
      echo "[run_table] AMBIGUOUS checkpoint for material $mat / $model (keep exactly one): ${matches[*]}" >&2
    else
      cells+=("$mat/$model"); ckpts+=("${matches[0]}")
    fi
  done
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "[run_table] ${#missing[@]} cell(s) without a usable checkpoint under $CKPT_ROOT: ${missing[*]}" >&2
  if [ "$ALLOW_MISSING" != "1" ]; then
    echo "[run_table] aborting before any evaluation (exit 4). Set ALLOW_MISSING=1 to evaluate the remaining cells; the run still exits nonzero." >&2
    exit 4
  fi
  echo "[run_table] ALLOW_MISSING=1: evaluating the remaining ${#cells[@]} cell(s); missing cells will be marked MISSING and the run exits nonzero" >&2
fi

# ---- 2. evaluate (reusing a result only when its checkpoint is unchanged) -----
mkdir -p "$RESULTS_DIR"
for i in "${!cells[@]}"; do
  mat=${cells[$i]%/*}; model=${cells[$i]#*/}; ckpt=${ckpts[$i]}
  result=$RESULTS_DIR/${mat}_${model}.json
  # What a result must record to stand in for this cell: the identifiers eval_stage2.sh
  # writes for exactly this material / model / dataset with no extra Hydra overrides.
  expect=("material=$mat" "model=$model" "dataset_folder=$DATA_ROOT/$mat" "overrides="
          "experiment=$([ "$model" = PBR ] && echo eval_stage2_pbr || echo eval_stage2)")
  if [ "$FORCE" = "1" ]; then
    echo "[run_table] FORCE=1: re-evaluating material $mat / $model"
  elif reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table] reuse $result: $reason"
    continue
  else
    echo "[run_table] evaluating material $mat / $model: $reason"
  fi
  if [ -e "$result" ]; then
    # A superseded result must not stay where the post-evaluation check could accept it
    # (e.g. when the evaluator exits 0 without writing anything); keep it aside instead.
    mv -f "$result" "$result.prev"
    echo "[run_table] previous result kept as $result.prev"
  fi
  echo "[run_table] evaluating material $mat / $model ($ckpt)"
  rc=0
  bash "$EVAL_SCRIPT" "$mat" "$ckpt" "$model" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "[run_table] ERROR: evaluation of material $mat / $model failed (exit $rc). Command:" >&2
    echo "    DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT bash $EVAL_SCRIPT $mat $ckpt $model" >&2
    exit 5
  fi
  if ! reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table] ERROR: evaluation of material $mat / $model exited 0 but left no valid result for $ckpt at $result: $reason" >&2
    exit 5
  fi
done

# ---- 3. collect: every expected cell present and within tolerance -------------
rc=0
"$PYTHON" "$COLLECT" "$RESULTS_DIR" --materials "$MATERIALS" --models "$MODELS" \
    --tolerance-db "$TOLERANCE_DB" --missing "${missing[*]:-}" || rc=$?
if [ $rc -ne 0 ]; then
  echo "[run_table] FAILED (exit $rc) — see the table above" >&2
  exit $rc
fi
if [ ${#missing[@]} -gt 0 ]; then   # not reached in practice: the collector already exits 6
  echo "[run_table] FAILED (exit 4): ${#missing[@]} checkpoint(s) were missing" >&2
  exit 4
fi
echo "[run_table] PASS: all ${#cells[@]} cells reproduced within $TOLERANCE_DB dB of the paper"
