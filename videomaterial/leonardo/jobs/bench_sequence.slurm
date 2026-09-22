#!/bin/bash
#SBATCH --job-name=vm-exp15-bench
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# exp-015 step 4: time the exp-005 sequence on one A100.
#
# Parameters arrive through --export (all optional except VM_RUN_NAME):
#   VM_RUN_NAME      output dir name under $VM_OUT_ROOT/runs
#   VM_SPP           samples per pixel (default 64)
#   VM_FRAME_INDICES subset of the 81-pose schedule, e.g. "0,40,80" (default all)
#   VM_VARIANT       mitsuba variant (default cuda_ad_rgb)
#
#   sbatch --export=ALL,VM_RUN_NAME=spp064,VM_SPP=64 \
#          --output=.../spp064-%j.out --error=.../spp064-%j.err \
#          videomaterial/leonardo/jobs/bench_sequence.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-leonardo-render}"
source "$REPO/videomaterial/leonardo/config/paths.sh"

RUN_NAME="${VM_RUN_NAME:?VM_RUN_NAME must be set}"
SPP="${VM_SPP:-64}"
BATCH="${VM_BATCH_SPP:-4}"
VARIANT="${VM_VARIANT:-cuda_ad_rgb}"
OUT="$VM_OUT_ROOT/runs/$RUN_NAME"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "run=$RUN_NAME spp=$SPP variant=$VARIANT indices=${VM_FRAME_INDICES:-all}"
nvidia-smi -L || true

test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
mkdir -p "$VM_OUT_ROOT/runs"
export PYTHONUNBUFFERED=1

python "$REPO/videomaterial/leonardo/bench/run_benchmark.py" \
    --rendering-root "$VM_RENDERING_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$(dirname "$(dirname "$VM_CKPT")")" \
    --output "$OUT" \
    --frames 81 \
    --spp "$SPP" \
    --batch-spp "$BATCH" \
    --seed 20260821 \
    --variant "$VARIANT" \
    --tag "$RUN_NAME" \
    ${VM_FRAME_INDICES:+--frame-indices "$VM_FRAME_INDICES"}

echo "done: $OUT"
