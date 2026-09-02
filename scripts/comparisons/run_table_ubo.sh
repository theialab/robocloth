#!/bin/bash
# ---------------------------------------------------------------------------
# Evaluate released UBO2014 stage-2 checkpoints and print the reproduced
# "Cross-dataset transfer to UBO2014" table next to the paper values, with a
# PASS/FAIL verdict per cell (12 materials x 4 models = 48 evaluations).
#
# Usage:
#   DATA_ROOT=/path/to/BTF CKPT_ROOT=/path/to/checkpoints/stage2/UBO \
#       bash scripts/comparisons/run_table_ubo.sh
#
#   DATA_ROOT is the flat folder of UBO2014 .btf files. The BTFs are NOT
#   redistributed by us: get them from the University of Bonn BTFDBB with
#   scripts/comparisons/download_ubo2014.sh (-> <root>/UBO2014).
#
#   CKPT_ROOT layout: $CKPT_ROOT/<material>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
#     (exactly one checkpoint per material and model)
#   Subsets: MATERIALS="felt01 felt03" MODELS="Bonn" bash scripts/comparisons/run_table_ubo.sh
#
# Knobs:
#   OUTPUT_ROOT=...     results: $OUTPUT_ROOT/eval_results_ubo/<material>_<model>.json
#   TOLERANCE_DB=0.05   max |repro - paper| per cell before the table FAILs
#   FORCE=1             re-evaluate every cell, ignoring existing results
#   ALLOW_MISSING=1     report missing checkpoints, evaluate the remaining
#                       cells anyway; those cells are marked MISSING in the
#                       table and the run still exits nonzero at the end
#   ROBOCLOTH_PYTHON=.. interpreter for the collector (default: `python`; also
#                       honoured by eval_stage2_ubo.sh — the generic PYTHON
#                       variable is deliberately ignored)
#
# The script can be interrupted and re-run: an existing result JSON is reused
# only if the checkpoint it records (path, size, mtime) is unchanged AND it was
# produced for this cell's configuration (material, model, experiment, BTF
# folder, no extra overrides); otherwise the cell is re-evaluated, the reason
# is logged, and the superseded JSON is kept aside as <name>.json.prev so that
# nothing but this run's own output can be accepted for the cell.
#
# Fails closed — exit codes:
#   2  bad environment: DATA_ROOT is not a directory / interpreter not found
#   4  a checkpoint is missing or ambiguous (all cells are checked BEFORE any
#      evaluation starts; every offending (material, model) is listed)
#   5  an evaluation failed or did not leave a valid result JSON for this
#      checkpoint (the failing command is printed; the run stops there)
#   6  an expected table cell has no result           (collector)
#   7  a cell differs from the paper by more than TOLERANCE_DB (collector)
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$SCRIPT_DIR"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the released stage-2 UBO checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/full_experiments}
MATERIALS=${MATERIALS:-"fabric02 fabric04 fabric09 fabric11 felt01 felt03 felt05 felt10 carpet02 carpet07 carpet09 carpet12"}
MODELS=${MODELS:-"Ours Bonn MERL PBR"}
TOLERANCE_DB=${TOLERANCE_DB:-0.05}
FORCE=${FORCE:-0}
ALLOW_MISSING=${ALLOW_MISSING:-0}
PYTHON=${ROBOCLOTH_PYTHON:-python}
# Test hook: an alternative evaluator with eval_stage2_ubo.sh's CLI + result-JSON contract.
EVAL_SCRIPT=${ROBOCLOTH_EVAL_SCRIPT:-$SCRIPT_DIR/eval_stage2_ubo.sh}
COLLECT=$SCRIPT_DIR/collect_eval_results_ubo.py
RESULTS_DIR=$OUTPUT_ROOT/eval_results_ubo

[ -x "$(command -v "$PYTHON" || true)" ] \
    || { echo "[run_table_ubo] ERROR: interpreter not found or not executable: $PYTHON (set ROBOCLOTH_PYTHON)" >&2; exit 2; }
[ -d "$DATA_ROOT" ] || { echo "[run_table_ubo] ERROR: DATA_ROOT is not a directory: $DATA_ROOT" >&2; exit 2; }
DATA_ROOT=$(cd "$DATA_ROOT" && pwd -P)   # results record the BTF folder; compare it unambiguously
export OUTPUT_ROOT DATA_ROOT

# ---- 1. resolve every checkpoint before evaluating anything -------------------
cells=(); ckpts=(); missing=()
[ -d "$CKPT_ROOT" ] || echo "[run_table_ubo] ERROR: CKPT_ROOT is not a directory: $CKPT_ROOT" >&2
for mat in $MATERIALS; do
  for model in $MODELS; do
    matches=("$CKPT_ROOT/$mat/${model}"_epoch*.ckpt)
    if [ ! -f "${matches[0]}" ]; then
      missing+=("$mat/$model")
      echo "[run_table_ubo] MISSING checkpoint for $mat / $model: expected $CKPT_ROOT/$mat/${model}_epoch<N>.ckpt" >&2
    elif [ ${#matches[@]} -gt 1 ]; then
      missing+=("$mat/$model")
      echo "[run_table_ubo] AMBIGUOUS checkpoint for $mat / $model (keep exactly one): ${matches[*]}" >&2
    else
      cells+=("$mat/$model"); ckpts+=("${matches[0]}")
    fi
  done
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "[run_table_ubo] ${#missing[@]} cell(s) without a usable checkpoint under $CKPT_ROOT: ${missing[*]}" >&2
  if [ "$ALLOW_MISSING" != "1" ]; then
    echo "[run_table_ubo] aborting before any evaluation (exit 4). Set ALLOW_MISSING=1 to evaluate the remaining cells; the run still exits nonzero." >&2
    exit 4
  fi
  echo "[run_table_ubo] ALLOW_MISSING=1: evaluating the remaining ${#cells[@]} cell(s); missing cells will be marked MISSING and the run exits nonzero" >&2
fi

# ---- 2. evaluate (reusing a result only when its checkpoint is unchanged) -----
mkdir -p "$RESULTS_DIR"
for i in "${!cells[@]}"; do
  mat=${cells[$i]%/*}; model=${cells[$i]#*/}; ckpt=${ckpts[$i]}
  result=$RESULTS_DIR/${mat}_${model}.json
  # What a result must record to stand in for this cell: the identifiers eval_stage2_ubo.sh
  # writes for exactly this material / model / BTF folder with no extra Hydra overrides.
  expect=("material=$mat" "model=$model" "dataset_folder=$DATA_ROOT" "overrides="
          "experiment=$([ "$model" = PBR ] && echo eval_ubo_pbr || echo eval_ubo)")
  if [ "$FORCE" = "1" ]; then
    echo "[run_table_ubo] FORCE=1: re-evaluating $mat / $model"
  elif reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table_ubo] reuse $result: $reason"
    continue
  else
    echo "[run_table_ubo] evaluating $mat / $model: $reason"
  fi
  if [ -e "$result" ]; then
    # A superseded result must not stay where the post-evaluation check could accept it
    # (e.g. when the evaluator exits 0 without writing anything); keep it aside instead.
    mv -f "$result" "$result.prev"
    echo "[run_table_ubo] previous result kept as $result.prev"
  fi
  echo "[run_table_ubo] evaluating $mat / $model ($ckpt)"
  rc=0
  bash "$EVAL_SCRIPT" "$mat" "$ckpt" "$model" || rc=$?
  if [ $rc -ne 0 ]; then
    echo "[run_table_ubo] ERROR: evaluation of $mat / $model failed (exit $rc). Command:" >&2
    echo "    DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT bash $EVAL_SCRIPT $mat $ckpt $model" >&2
    exit 5
  fi
  if ! reason=$("$PYTHON" "$COLLECT" --reuse-check "$result" "$ckpt" --expect "${expect[@]}"); then
    echo "[run_table_ubo] ERROR: evaluation of $mat / $model exited 0 but left no valid result for $ckpt at $result: $reason" >&2
    exit 5
  fi
done

# ---- 3. collect: every expected cell present and within tolerance -------------
rc=0
"$PYTHON" "$COLLECT" "$RESULTS_DIR" --materials "$MATERIALS" --models "$MODELS" \
    --tolerance-db "$TOLERANCE_DB" --missing "${missing[*]:-}" || rc=$?
if [ $rc -ne 0 ]; then
  echo "[run_table_ubo] FAILED (exit $rc) — see the table above" >&2
  exit $rc
fi
if [ ${#missing[@]} -gt 0 ]; then   # not reached in practice: the collector already exits 6
  echo "[run_table_ubo] FAILED (exit 4): ${#missing[@]} checkpoint(s) were missing" >&2
  exit 4
fi
echo "[run_table_ubo] PASS: all ${#cells[@]} cells reproduced within $TOLERANCE_DB dB of the paper"
