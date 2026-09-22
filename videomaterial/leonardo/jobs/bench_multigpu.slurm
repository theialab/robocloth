#!/bin/bash
#SBATCH --job-name=vm-exp15-bench-4gpu
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:40:00
# exp-015 step 4c: scaling test. Four independent 3-frame renders, one per GPU
# of a single Booster node, launched concurrently with CUDA_VISIBLE_DEVICES.
# If a Leonardo node renders four sequences at the single-GPU per-frame time,
# the dataset cost is (sequences / 4) x single-GPU sequence time.
#
#   sbatch --export=ALL,VM_SPP=64 --output=... --error=... \
#          videomaterial/leonardo/jobs/bench_multigpu.slurm
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-leonardo-render}"
source "$REPO/videomaterial/leonardo/config/paths.sh"

SPP="${VM_SPP:-64}"
INDICES="${VM_FRAME_INDICES:-0,40,80}"
RUN_ROOT="$VM_OUT_ROOT/runs/${VM_RUN_NAME:-scaling4gpu_spp${SPP}}"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1

start=$(date +%s)
pids=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python "$REPO/videomaterial/leonardo/bench/run_benchmark.py" \
        --rendering-root "$VM_RENDERING_ROOT" \
        --scene-src "$VM_EXP005_DIR" \
        --checkpoint-root "$(dirname "$(dirname "$VM_CKPT")")" \
        --output "$RUN_ROOT/gpu$gpu" \
        --frames 81 \
        --frame-indices "$INDICES" \
        --spp "$SPP" \
        --batch-spp 4 \
        --seed 20260821 \
        --variant "${VM_VARIANT:-cuda_ad_rgb}" \
        --tag "scaling_gpu$gpu" \
        > "$RUN_ROOT/gpu$gpu.stdout" 2>&1 &
    pids+=($!)
    echo "launched gpu$gpu pid ${pids[-1]}"
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "gpu$i finished ok"
    else
        echo "gpu$i FAILED"
        status=1
    fi
done
end=$(date +%s)
echo "all four concurrent renders wall seconds: $((end - start))"
echo "$((end - start))" > "$RUN_ROOT/concurrent_wall_seconds.txt"
exit $status
