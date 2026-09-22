#!/bin/bash
#SBATCH --job-name=vm-pbr-patch
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/logs/pbr_shard_%a-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/logs/pbr_shard_%a-%j.err
# exp-015 pbr_patch twin production pass: add the Zhen-approved homogeneous PBR patch
# twin to all 1,000 existing B1 sequences, over 4 independent single-GPU tasks
# (250 round-robin sequences each). One task = one patch-scene load for its whole
# shard; ~25-35 s per sequence on an A100 puts a shard near 2.5 h, inside the 3 h limit.
#
# Additive only: it writes pbr_patch_png/ (81 PNGs), pbr_patch_frames.jsonl, a
# pbr_patch_render block in metadata.json, a refreshed MANIFEST.sha256 and
# PBR_PATCH_COMPLETE. frames_png/, frames_exr/ and ball_png/ are never read or written.
#
# Resumable by design: a sequence with PBR_PATCH_COMPLETE is skipped, so a failed or
# time-limited task is fixed by resubmitting the same array index.
#
#   sbatch --array=0-3 videomaterial/leonardo_dataset/jobs/render_pbr_patch_array.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${VM_NSHARDS:-4}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_MANIFEST" || { echo "missing manifest $VM_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
mkdir -p "$VM_B1_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_MANIFEST" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --root "$VM_B1_ROOT" \
    --twin-only pbr_patch --twin-spp 32 --twin-batch-spp 16 \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
end=$(date +%s)
echo "shard $SHARD wall seconds: $((end - start)) exit=$status"
exit $status
