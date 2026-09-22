#!/usr/bin/env bash
# exp-015 B1 render orchestrator, run on g00s (NOT on Leonardo).
#
#   nohup setsid bash videomaterial/leonardo_dataset/scripts/orchestrate_b1.sh \
#       > /media/raid/cloth/VideoMaterial/output/synthetic_experiments/exp15_b1_render/orchestrate.log 2>&1 &
#
# setsid + nohup so it survives the agent session that started it. It submits the
# Slurm array, polls every 5 min, resubmits shards that came back short exactly once,
# rsyncs the dataset to the fileserver, verifies every MANIFEST.sha256 with
# sha256sum -c, and finally writes DONE or FAILED next to status.json.
#
# Stop it with:  pkill -f orchestrate_b1
set -euo pipefail

export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/tmp/vm-agent.sock}"

STATE_DIR="${VM_STATE_DIR:-/media/raid/cloth/VideoMaterial/output/synthetic_experiments/exp15_b1_render}"
LOCAL_ROOT="${VM_LOCAL_B1:-/media/raid/cloth/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera}"
MANIFEST="${VM_LOCAL_MANIFEST:-$LOCAL_ROOT/manifest.json}"
REPO_ROOT="${VM_REMOTE_REPO:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$STATE_DIR"
echo "orchestrator starting $(date -u +%Y-%m-%dT%H:%M:%SZ) state=$STATE_DIR"
exec python3 "$HERE/orchestrate_b1.py" \
    --state-dir "$STATE_DIR" \
    --manifest "$MANIFEST" \
    --local-root "$LOCAL_ROOT" \
    --repo-root "$REPO_ROOT" \
    --nshards "${VM_NSHARDS:-16}" \
    --array "${VM_ARRAY:-0-15}" \
    --poll "${VM_POLL:-300}" \
    "$@"
