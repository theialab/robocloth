#!/bin/bash
# Stage 1 on the MERL database. Hyperparameters: configs/experiment/stage1_merl.yaml.
#
#   DATA_ROOT=/path/to/MERL/brdfs bash train_stage1_merl.sh
#
#   The MERL .binary files are NOT redistributed by us: download them from
#   Zenodo with scripts/comparisons/download_merl.sh (-> <root>/MERL/brdfs).
set -euo pipefail
cd "$(dirname "$0")/../../training"
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the MERL brdfs folder}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
EXP_NAME=${EXP_NAME:-stage1_merl}
export WANDB_MODE=${WANDB_MODE:-offline}
python train.py +experiment=stage1_merl \
    dataset_folder="$DATA_ROOT" output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" experiment_name="$EXP_NAME" "$@"
