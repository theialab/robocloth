#!/bin/bash
#SBATCH --job-name=vm-exp15-validate-gpu
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
# exp-015 step 2: prove Mitsuba's cuda_ad_rgb variant works on an A100 and that
# the GPU is actually busy during a render. Debug-sized on purpose (cluster rule:
# a minimal 1-GPU short test before any long job).
#
#   sbatch --output=.../validate-%j.out --error=.../validate-%j.err \
#          videomaterial/leonardo/jobs/validate_gpu.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-leonardo-render}"
source "$REPO/videomaterial/leonardo/config/paths.sh"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$(python -V 2>&1) env=$VM_ENV_DIR"
nvidia-smi -L || true

mkdir -p "$VM_OUT_ROOT/logs"
export PYTHONUNBUFFERED=1
python "$REPO/videomaterial/leonardo/env/validate_gpu.py" \
    --spp "${VM_VALIDATE_SPP:-4}" \
    --out "$VM_OUT_ROOT/logs/validate_gpu_${SLURM_JOB_ID:-local}.json" \
    ${VM_ALLOW_LLVM:+--allow-llvm-fallback}
