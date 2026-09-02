#!/bin/bash
# Stage 2 on a held-out UBO2014 BTF material. Hyperparameters:
# configs/experiment/stage2_ubo*.yaml.
#
#   DATA_ROOT=/path/to/UBO2014 STAGE1_CKPT=.../stage1/Bonn.ckpt MODEL=Bonn \
#       bash train_stage2_ubo.sh <material>              # e.g. felt01
#
#   DATA_ROOT is the flat folder of UBO2014 .btf files; download it from the
#   University of Bonn BTFDBB with scripts/comparisons/download_ubo2014.sh.
set -euo pipefail
cd "$(dirname "$0")/../../training"
MAT=${1:?usage: train_stage2_ubo.sh <material>}; shift || true
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
MODEL=${MODEL:-Bonn}
EXP_NAME=${EXP_NAME:-stage2_ubo_${MAT}_from_${MODEL}}
export WANDB_MODE=${WANDB_MODE:-offline}
EXTRA=()
if [ "$MODEL" = "PBR" ]; then EXPERIMENT=stage2_ubo_pbr; CKPT_OVERRIDE=()
else
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT for MODEL=$MODEL}
    EXPERIMENT=stage2_ubo; CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
    [ "$MODEL" = "MERL" ] && EXTRA+=(model.factor_init=0.1)
fi
python train.py +experiment=$EXPERIMENT \
    dataset_folder="$DATA_ROOT" data.btf_filename=${MAT}_W400xH400_L151xV151.btf \
    output_folder="$OUTPUT_ROOT" exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    experiment_name="$EXP_NAME" "${EXTRA[@]}" "${CKPT_OVERRIDE[@]}" "$@"
