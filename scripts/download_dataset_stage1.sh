#!/bin/bash
# ---------------------------------------------------------------------------
# Download the SPARSE training set for stage-1 decoder training: per-material
# observation tensors + pose metadata (NO dense HDR images; ~490 GB instead of
# ~3.5 TB), plus the dataset-level calibration/split files.
#
# Usage:
#   bash scripts/download_dataset_stage1.sh [DATA_ROOT]           # default ./DATA_ROOT
# ---------------------------------------------------------------------------
set -euo pipefail
DEST=${1:-$PWD/DATA_ROOT}
REPO=koalapenguin/RoboCloth
# The released corpus contains numeric material IDs 0..499.  Do not use a
# shell wildcard here: an unquoted '*' expands to files in the caller's
# working directory before it ever reaches the Hugging Face include filter.
if [[ -n ${ROBOCLOTH_MATERIALS:-} ]]; then
    read -r -a IDS <<< "$ROBOCLOTH_MATERIALS"
else
    IDS=({0..499})
fi

if (( ${#IDS[@]} == 0 )); then
    echo "[download_dataset_stage1] ERROR: no material IDs requested" >&2
    exit 1
fi

INCLUDES=(--include "globals/*")
for id in "${IDS[@]}"; do
    if [[ ! $id =~ ^[0-9]+$ ]]; then
        echo "[download_dataset_stage1] ERROR: invalid material ID: $id" >&2
        exit 1
    fi
    INCLUDES+=(--include "materials/$id/observations_structured.npz" \
               --include "materials/$id/scan_log.json" \
               --include "materials/$id/rotated_camera.json" \
               --include "materials/$id/point_metadata.json")
done

mkdir -p "$DEST"
hf download "$REPO" --repo-type dataset "${INCLUDES[@]}" --local-dir "$DEST"
cp -f "$DEST"/globals/* "$DEST"/
for id in "${IDS[@]}"; do
    src="$DEST/materials/$id"
    if [[ ! -d $src ]]; then
        echo "[download_dataset_stage1] ERROR: requested material $id was not downloaded" >&2
        exit 1
    fi
    rm -rf "$DEST/$id"
    mv "$src" "$DEST/$id"
done
rmdir "$DEST/materials" 2>/dev/null || true
echo "[download_dataset_stage1] ready: $DEST ($(ls -d "$DEST"/[0-9]* 2>/dev/null | wc -l) materials)"
