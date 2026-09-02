#!/bin/bash
# Stand-in for `python` inside scripts/eval_stage2.sh / eval_stage2_ubo.sh
# (selected with ROBOCLOTH_PYTHON=...): emulates `train.py` by writing the
# CSV-logger metrics.csv those scripts parse — into the next free version_<N>/
# directory, like pytorch_lightning's CSVLogger — and forwards every other
# invocation (the metrics snapshot, the result-writing heredoc, the collector)
# to the real interpreter $REAL_PYTHON.
#
# Knobs:
#   FAKE_TRAIN_LOG        file that receives train.py's argv (one line per call)
#   FAKE_TRAIN_RC         exit code of train.py (default 0)
#   FAKE_TRAIN_NO_CSV=1   train.py exits 0 without writing metrics.csv
#   FAKE_TRAIN_PSNR       logged val/psnr (default: paper value of 145/Ours);
#                         an empty string logs no val/psnr value at all
set -euo pipefail
if [ "${1:-}" = "train.py" ]; then
  printf '%s\n' "$*" >> "${FAKE_TRAIN_LOG:?set FAKE_TRAIN_LOG}"
  if [ "${FAKE_TRAIN_RC:-0}" != "0" ]; then exit "$FAKE_TRAIN_RC"; fi
  if [ "${FAKE_TRAIN_NO_CSV:-0}" = "1" ]; then exit 0; fi
  exp_dir=; exp_name=
  for a in "$@"; do
    case $a in
      exp_output_root_path=*) exp_dir=${a#*=} ;;
      experiment_name=*) exp_name=${a#*=} ;;
    esac
  done
  n=0
  while [ -e "$exp_dir/$exp_name/version_$n" ]; do n=$((n + 1)); done
  d=$exp_dir/$exp_name/version_$n
  mkdir -p "$d"
  printf 'epoch,step,val/loss,val/psnr\n0,0,,\n1,10,0.0123,%s\n' \
      "${FAKE_TRAIN_PSNR-28.442256927490234}" > "$d/metrics.csv"
  exit 0
fi
exec "${REAL_PYTHON:?set REAL_PYTHON}" "$@"
