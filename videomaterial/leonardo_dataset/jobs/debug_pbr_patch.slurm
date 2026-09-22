#!/bin/bash
#SBATCH --job-name=vm-pbr-debug
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/logs/pbr_debug-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/logs/pbr_debug-%j.err
# exp-015 pbr_patch twin debug run (cluster rule: a minimal 1-GPU short job before any
# long one). Three real B1 sequences get the Zhen-approved trivial-case twin added by
# the --twin-only pass. It only ADDS pbr_patch_png/, pbr_patch_frames.jsonl, a
# pbr_patch_render block in metadata.json and PBR_PATCH_COMPLETE; frames_png/,
# frames_exr/ and ball_png/ are never touched, so running on the real dataset root is
# safe and the three sequences are then skipped by the production array.
#
# What the run must show before the array may be submitted:
#   * 3x PBR_PATCH_COMPLETE, 81 PNGs in each pbr_patch_png/,
#   * sha256sum -c MANIFEST.sha256 clean in the touched sequences,
#   * metadata.json pbr_patch_render.generator.commit == the local HEAD sha (CODE_VERSION stamp),
#   * per-sequence wall near 25-35 s and gap between sequences < 5 s (scene stays resident),
#   * sacct Elapsed/AllocTRES -> billing-h per sequence, projected x1000 must stay under 80.
#
#   sbatch videomaterial/leonardo_dataset/jobs/debug_pbr_patch.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_MANIFEST" || { echo "missing manifest $VM_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
mkdir -p "$VM_B1_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_MANIFEST" \
    --shard "${VM_SHARD:-0}" --nshards "${VM_NSHARDS:-4}" \
    --limit "${VM_LIMIT:-3}" \
    --root "$VM_B1_ROOT" \
    --twin-only pbr_patch --twin-spp 32 --twin-batch-spp 16 \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
end=$(date +%s)
echo "debug wall seconds: $((end - start)) exit=$status"
echo "PBR_PATCH_COMPLETE files: $(find "$VM_B1_ROOT" -name PBR_PATCH_COMPLETE | wc -l)"
exit $status
