#!/bin/bash
#SBATCH --job-name=vm-b1-debug
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
# exp-015 B1 debug run (cluster rule: a minimal 1-GPU short job before any long one).
# Three real manifest sequences, material + ball, into a DEBUG root so the production
# dataset tree stays untouched by possibly-wrong code.
#
# What the run must show before the array may be submitted:
#   * 3x RENDER_COMPLETE, frames_png + ball_png present (frames_exr where keep_exr),
#   * per-sequence wall time near 190 s (material) + ball time,
#   * gap between consecutive sequences < 5 s -> the checkpoint is NOT reloaded,
#   * the saldo -b delta, which gives the local-h per A100-hour factor.
#
#   sbatch --output=$VM_RUN_ROOT/logs/debug-%j.out --error=$VM_RUN_ROOT/logs/debug-%j.err \
#          videomaterial/leonardo_dataset/jobs/debug_b1.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_MANIFEST" || { echo "missing manifest $VM_MANIFEST"; exit 1; }
mkdir -p "$VM_DEBUG_ROOT" "$VM_RUN_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_MANIFEST" \
    --shard "${VM_SHARD:-0}" --nshards "${VM_NSHARDS:-16}" \
    --limit "${VM_LIMIT:-3}" \
    --root "$VM_DEBUG_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
end=$(date +%s)
echo "debug wall seconds: $((end - start))"
echo "RENDER_COMPLETE files: $(find "$VM_DEBUG_ROOT" -name RENDER_COMPLETE | wc -l)"
