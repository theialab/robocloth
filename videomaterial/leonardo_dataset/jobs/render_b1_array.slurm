#!/bin/bash
#SBATCH --job-name=vm-b1-render
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
# exp-015 B1 production render: 1,000 sequences over 16 independent single-GPU tasks
# (63 sequences each; round-robin shards, so every task has the same class mix).
# One task = one material-scene load for its whole shard; ~190 s material + ~30 s ball
# per sequence puts a 63-sequence shard near 4 h, inside the 8 h limit.
#
# Resumable by design: a sequence directory with RENDER_COMPLETE is skipped, so a
# failed or time-limited task is fixed by resubmitting the same array index.
#
#   sbatch --array=0-15 --output=$VM_RUN_ROOT/logs/render-%A_%a.out \
#          --error=$VM_RUN_ROOT/logs/render-%A_%a.err \
#          videomaterial/leonardo_dataset/jobs/render_b1_array.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${VM_NSHARDS:-16}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_MANIFEST" || { echo "missing manifest $VM_MANIFEST"; exit 1; }
mkdir -p "$VM_B1_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_MANIFEST" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --root "$VM_B1_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
end=$(date +%s)
echo "shard $SHARD wall seconds: $((end - start)) exit=$status"
exit $status
