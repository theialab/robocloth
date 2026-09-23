#!/bin/bash
#SBATCH --job-name=vm-b2-render
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B2_Fixed_light/logs/b2_shard_%a-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B2_Fixed_light/logs/b2_shard_%a-%j.err
# exp-015 B2 production render: 1,000 sequences (900 train + 100 test) over 16
# independent single-GPU tasks (~63 sequences each; round-robin shards, so every task
# has the same class mix). B2 = fixed point light (theta 45 deg, phi 90 deg, r 3.0),
# moving camera (polar 0.5-80 deg, <= 3 deg/frame, fov_y 35 deg), 832x480, 81 frames.
#
# One task = one material-scene load for its whole shard; ~190 s material (64 spp) +
# ~15 s pbr_patch twin (32 spp) per sequence puts a 63-sequence shard near 3.6 h,
# inside the 8 h limit.
#
# Resumable by design: a sequence directory with RENDER_COMPLETE is skipped, so a
# failed or time-limited task is fixed by resubmitting the same array index.
#
# Budget: expected ~580 billing-h, HARD CAP 700 billing-h (the supervisor scancels
# the array if the running total crosses it).
#
#   sbatch --array=0-15 videomaterial/leonardo_dataset/jobs/render_b2_array.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

# B2 paths. paths.sh describes the B1 render and is not edited by this experiment, so
# the B2 root / manifest are defined here and never inherit a B1 value.
export VM_B2_ROOT="${VM_B2_ROOT:-$VM_WORK_ROOT/data/Robocloth_synthetic_sequence/B2_Fixed_light}"
export VM_B2_MANIFEST="${VM_B2_MANIFEST:-$VM_B2_ROOT/manifest.json}"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${VM_NSHARDS:-16}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}_${SHARD} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_B2_MANIFEST" || { echo "missing manifest $VM_B2_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
# hard guard: this job must never be pointed at the B1 manifest or root.
python - "$VM_B2_MANIFEST" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
assert m["setup"] == "B2", f'manifest setup is {m["setup"]!r}, refusing to render'
print(f'manifest ok: setup={m["setup"]} twin={m["twin"]} sequences={m["counts"]["total"]}')
PY
case "$VM_B2_ROOT" in *B1*) echo "refusing: VM_B2_ROOT points at a B1 tree ($VM_B2_ROOT)"; exit 1;; esac
mkdir -p "$VM_B2_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
set +e
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_B2_MANIFEST" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --root "$VM_B2_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
set -e
end=$(date +%s)
echo "shard $SHARD wall seconds: $((end - start)) exit=$status"
exit $status
