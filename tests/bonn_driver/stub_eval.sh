#!/bin/bash
# Stub evaluator for tests/bonn_driver: same CLI (<mat> <ckpt> <tag>) and
# result-JSON contract as evaluate_cell in scripts/comparisons/run_table_bonn.sh,
# but instant and GPU-free. The driver picks it up via ROBOCLOTH_EVAL_SCRIPT.
# Appends "<mat>/<tag>" to $STUB_CALL_LOG on every call; behaviour knobs
# (STUB_PSNR_OFFSETS, STUB_FAIL_CELLS, ...) are documented in stub_eval.py.
# -B: never leave __pycache__ under scripts/ when importing the collectors.
set -euo pipefail
echo "$1/$3" >> "${STUB_CALL_LOG:?set STUB_CALL_LOG}"
exec "${ROBOCLOTH_PYTHON:-python}" -B "$(dirname "$0")/stub_eval.py" "$1" "$2" "$3"
