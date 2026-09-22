#!/usr/bin/env bash
# Shared Leonardo paths for the exp-015 Mitsuba render benchmark.
# Source this from every job script and from the env builder.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
#
# Machine-specific overrides (VM_REPO_ROOT above all) live in paths.local.sh,
# which is sourced at the end if present (dec-010: one remote code dir per
# experiment worktree so concurrent agents never collide).

set -u

export VM_ACCOUNT="IscrB_OVER"
export VM_PARTITION="boost_usr_prod"

export WORK="${WORK:-/leonardo_work/IscrB_OVER}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrB_OVER}"

export VM_WORK_ROOT="$WORK/zli00003/VideoMaterial"
export VM_FAST_ROOT="$FAST/zli00003/VideoMaterial"

# Python environment (built by env/build_env.sh on the LOGIN node; compute
# nodes may have no internet).
export VM_ENV_DIR="$VM_FAST_ROOT/envs/mitsuba-render-v1"
export VM_PYTHON="$VM_ENV_DIR/bin/python"

# Data.
export VM_CKPT="$VM_WORK_ROOT/models/robocloth/314/Ours_epoch80.ckpt"
export VM_EXP005_DIR="$VM_WORK_ROOT/data/robocloth_synthetic_v1/2026-08-21-mat314-wan832x480-81f-64spp"

# Outputs (mirrors the fileserver layout of exp-015 SPEC section 1).
export VM_OUT_ROOT="$VM_WORK_ROOT/outputs/synthetic_experiments/exp15_leonardo_mitsuba_render_benchmark"

# Code (default; paths.local.sh overrides).
export VM_REPO_ROOT="${VM_REPO_ROOT:-$VM_WORK_ROOT/code/robocloth-leonardo-render}"

_vm_paths_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_vm_paths_dir/paths.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$_vm_paths_dir/paths.local.sh"
fi
unset _vm_paths_dir

export VM_RENDERING_ROOT="$VM_REPO_ROOT/rendering"
