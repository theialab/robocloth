#!/usr/bin/env bash
# Shared Leonardo paths for the exp-015 B1 dataset render (1,000 sequences).
# Source from every job script:  source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
#
# This is the dataset render, NOT the benchmark: it has its own remote code dir
# (dec-010, one per experiment worktree) so it cannot collide with the
# robocloth-leonardo-render benchmark tree. Overrides go in paths.local.sh.
set -u

export VM_ACCOUNT="IscrB_OVER"
export VM_PARTITION="boost_usr_prod"

export WORK="${WORK:-/leonardo_work/IscrB_OVER}"
export FAST="${FAST:-/leonardo_scratch/fast/IscrB_OVER}"

export VM_WORK_ROOT="$WORK/zli00003/VideoMaterial"
export VM_FAST_ROOT="$FAST/zli00003/VideoMaterial"

# Python environment (built on a login node by the benchmark task; mitsuba 3.8.0,
# drjit 1.3.1, torch 2.4.1+cu124, cuda_ad_rgb via the driver's OptiX).
export VM_ENV_DIR="$VM_FAST_ROOT/envs/mitsuba-render-v1"

# Inputs (read-only: never written by this experiment).
export VM_CKPT_ROOT="$VM_WORK_ROOT/models/robocloth"
export VM_CKPT="$VM_CKPT_ROOT/314/Ours_epoch80.ckpt"
export VM_EXP005_DIR="$VM_WORK_ROOT/data/robocloth_synthetic_v1/2026-08-21-mat314-wan832x480-81f-64spp"

# Code: this worktree, rsynced without .git/assets.
export VM_REPO_ROOT="${VM_REPO_ROOT:-$VM_WORK_ROOT/code/robocloth-synthetic-sequences}"

# Outputs. The dataset itself, and the run's own logs/status.
export VM_B1_ROOT="$VM_WORK_ROOT/data/Robocloth_synthetic_sequence/B1_Fixed_camera"
export VM_MANIFEST="$VM_B1_ROOT/manifest.json"
export VM_RUN_ROOT="$VM_WORK_ROOT/outputs/synthetic_experiments/exp15_b1_render"
export VM_DEBUG_ROOT="$VM_RUN_ROOT/debug/B1_Fixed_camera"

_vm_paths_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_vm_paths_dir/paths.local.sh" ]; then
    # shellcheck disable=SC1091
    source "$_vm_paths_dir/paths.local.sh"
fi
unset _vm_paths_dir

export VM_RENDERING_ROOT="$VM_REPO_ROOT/rendering"
export VM_PKG_DIR="$VM_REPO_ROOT/videomaterial/synthetic_sequences"
