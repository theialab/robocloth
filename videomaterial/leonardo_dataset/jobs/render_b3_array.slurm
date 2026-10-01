#!/bin/bash
#SBATCH --job-name=vm-b3-render
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B3_Joint/logs/b3_shard_%a-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B3_Joint/logs/b3_shard_%a-%j.err
# B3 production render: 1,000 sequences (900 train pairs + 100 test: 34 unseen_light, 33
# unseen_camera, 33 unseen_pair) over 16 independent single-GPU tasks (62-63 sequences each;
# round-robin shards, so every task has the same train/test mix). B3 = jointly moving point
# light AND camera: frame k = light of frame k of a B1 sequence, camera pose of frame k of a
# B2 sequence; 832x480, 81 frames, material 314 at 64 spp + pbr_patch twin at 32 spp.
#
# One task = one material-scene load for its whole shard. Every sequence first compares its
# regenerated poses with the two source sequences' frames.jsonl (--source-root), exactly; a
# mismatch fails that sequence instead of rendering wrong poses.
#
# Resumable by design: a sequence directory with RENDER_COMPLETE is skipped, so a failed or
# time-limited task is fixed by resubmitting the same array index. --time 08:00:00 (never the
# full 24 h: those requests do not backfill); B2's tasks took ~5 h.
#
#   sbatch --array=0-15 videomaterial/leonardo_dataset/jobs/render_b3_array.slurm   (from $VM_REPO_ROOT)
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-b3-joint}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

# B3 paths. paths.sh describes the B1 render and is not edited by this experiment, so the B3
# root / manifest are defined here and never inherit a B1 or B2 value.
export VM_DATA_ROOT="${VM_DATA_ROOT:-$VM_WORK_ROOT/data/Robocloth_synthetic_sequence}"
export VM_B3_ROOT="${VM_B3_ROOT:-$VM_DATA_ROOT/B3_Joint}"
export VM_B3_MANIFEST="${VM_B3_MANIFEST:-$VM_B3_ROOT/manifest.json}"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${VM_NSHARDS:-16}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_B3_MANIFEST" || { echo "missing manifest $VM_B3_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
# hard guard: this job must only ever render a B3 manifest into the B3 tree.
python - "$VM_B3_MANIFEST" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
assert m["setup"] == "B3", f'manifest setup is {m["setup"]!r}, refusing to render'
print(f'manifest ok: setup={m["setup"]} twin={m["twin"]} sequences={m["counts"]["total"]} master_seed={m["master_seed"]}')
PY
case "$VM_B3_ROOT" in *B1_Fixed_camera*|*B2_Fixed_light*) echo "refusing: VM_B3_ROOT points at a source tree ($VM_B3_ROOT)"; exit 1;; esac
mkdir -p "$VM_B3_ROOT/logs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

start=$(date +%s)
set +e
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_B3_MANIFEST" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --root "$VM_B3_ROOT" \
    --source-root "$VM_DATA_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
set -e
end=$(date +%s)
echo "shard $SHARD wall seconds: $((end - start)) exit=$status"
exit $status
