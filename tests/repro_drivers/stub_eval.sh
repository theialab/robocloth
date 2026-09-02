#!/bin/bash
# Stub evaluator for tests/repro_drivers: same CLI (<mat> <ckpt> <tag>) and
# result-JSON contract as scripts/eval_stage2.sh / eval_stage2_ubo.sh, but
# instant and GPU-free. The drivers pick it up via ROBOCLOTH_EVAL_SCRIPT.
# Appends "<mat>/<tag>" to $STUB_CALL_LOG on every call; behaviour knobs
# (STUB_TABLE, STUB_PSNR_OFFSETS, STUB_FAIL_CELLS, ...) are documented in stub_eval.py.
# -B: never leave __pycache__ under scripts/ when importing the collector.
set -euo pipefail
echo "$1/$3" >> "${STUB_CALL_LOG:?set STUB_CALL_LOG}"
exec "${ROBOCLOTH_PYTHON:-python}" -B "$(dirname "$0")/stub_eval.py" "$1" "$2" "$3"
