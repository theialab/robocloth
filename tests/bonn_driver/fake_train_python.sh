#!/bin/bash
# Stand-in for `python` inside scripts/comparisons/run_table_bonn.sh (selected
# with ROBOCLOTH_PYTHON=...): emulates `train.py` by writing the CSV-logger
# metrics.csv the driver's evaluate_cell parses — with the columns
# Stage1Trainer_Bonn logs during validation — into the next free version_<N>/
# directory, like pytorch_lightning's CSVLogger — and forwards every other
# invocation (the data check, the metrics snapshot, the result-writing
# heredoc, the collector) to the real interpreter $REAL_PYTHON.
#
# Knobs:
#   FAKE_TRAIN_LOG          file that receives train.py's argv (one line per call)
#   FAKE_TRAIN_RC           exit code of train.py (default 0)
#   FAKE_TRAIN_NO_CSV=1     train.py exits 0 without writing metrics.csv
#   FAKE_TRAIN_POLY_PSNR    logged val/poly_psnr (default 40.0)
#   FAKE_TRAIN_GRAY_PSNR    logged val/gray_psnr (default 45.4957); an empty
#                           string logs no gray metrics at all
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
  poly=${FAKE_TRAIN_POLY_PSNR-40.0}
  gray=${FAKE_TRAIN_GRAY_PSNR-45.4957}
  if [ -n "$gray" ]; then
    printf 'epoch,step,val/gray_loss,val/gray_mse,val/gray_psnr,val/poly_loss,val/poly_mse,val/poly_psnr,val/all_psnr,val/all_mse\n' > "$d/metrics.csv"
    printf '0,0,0.0111,0.0002,%s,0.0123,0.0003,%s,44.0,0.0004\n' "$gray" "$poly" >> "$d/metrics.csv"
  else
    printf 'epoch,step,val/poly_loss,val/poly_mse,val/poly_psnr\n0,0,0.0123,0.0003,%s\n' "$poly" > "$d/metrics.csv"
  fi
  exit 0
fi
exec "${REAL_PYTHON:?set REAL_PYTHON}" "$@"
