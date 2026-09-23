#!/bin/bash
#SBATCH --job-name=vm-b2-debug
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B2_Fixed_light/logs/b2_debug-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/data/Robocloth_synthetic_sequence/B2_Fixed_light/logs/b2_debug-%j.err
# exp-015 B2 debug run (cluster rule: a minimal 1-GPU short job before any long one).
# B2 = fixed point light (theta 45 deg, phi 90 deg, r 3.0), moving camera (fov_y 35 deg).
# Three real manifest sequences, material 64 spp + pbr_patch twin 32 spp, into a DEBUG
# root so the production B2 tree stays untouched by possibly-wrong code.
#
# What the run must show before the array may be submitted:
#   * 3x RENDER_COMPLETE, 81 frames_png + 81 pbr_patch_png each (frames_exr where keep_exr),
#   * sha256sum -c MANIFEST.sha256 clean,
#   * metadata setup == "B2", light.mode == "fixed", camera.mode == "trajectory",
#     generator.commit == the local HEAD sha (CODE_VERSION stamp),
#   * per-sequence wall near 190 s (material) + ~15 s (twin),
#   * gap between consecutive sequences < 5 s -> the checkpoint is NOT reloaded,
#   * sacct Elapsed x billing=8 -> billing-h per sequence; x1000 must stay under 700.
#
#   sbatch videomaterial/leonardo_dataset/jobs/debug_b2.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

# B2 paths. paths.sh describes the B1 render and is not edited by this experiment, so
# the B2 root / manifest / debug root are defined here and never inherit a B1 value.
export VM_B2_ROOT="${VM_B2_ROOT:-$VM_WORK_ROOT/data/Robocloth_synthetic_sequence/B2_Fixed_light}"
export VM_B2_MANIFEST="${VM_B2_MANIFEST:-$VM_B2_ROOT/manifest.json}"
export VM_B2_RUN_ROOT="${VM_B2_RUN_ROOT:-$VM_WORK_ROOT/outputs/synthetic_experiments/exp15_b2_render}"
export VM_B2_DEBUG_ROOT="${VM_B2_DEBUG_ROOT:-$VM_B2_RUN_ROOT/debug/B2_Fixed_light}"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_B2_MANIFEST" || { echo "missing manifest $VM_B2_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
# hard guard: this job must never be pointed at the B1 manifest.
python - "$VM_B2_MANIFEST" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
assert m["setup"] == "B2", f'manifest setup is {m["setup"]!r}, refusing to render'
print(f'manifest ok: setup={m["setup"]} twin={m["twin"]} sequences={m["counts"]["total"]} '
      f'material_spp={m["render"]["material"]["spp"]} twin_spp={m["render"][m["twin"]]["spp"]}')
PY
mkdir -p "$VM_B2_DEBUG_ROOT" "$VM_B2_ROOT/logs"
export PYTHONUNBUFFERED=1

start=$(date +%s)
set +e
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_B2_MANIFEST" \
    --shard "${VM_SHARD:-0}" --nshards "${VM_NSHARDS:-16}" \
    --limit "${VM_LIMIT:-3}" \
    --root "$VM_B2_DEBUG_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
set -e
end=$(date +%s)
echo "debug wall seconds: $((end - start)) exit=$status"
echo "RENDER_COMPLETE files: $(find "$VM_B2_DEBUG_ROOT" -name RENDER_COMPLETE | wc -l)"
exit $status
