#!/bin/bash
# ---------------------------------------------------------------------------
# Download the FULL RoboCloth dataset (~3.5 TB: all 500 materials including
# dense HDR views) and assemble a training-ready DATA_ROOT. Each hdr.tar is
# extracted and removed after download.
#
# Usage:
#   bash scripts/download_dataset_full.sh [DATA_ROOT]             # default ./DATA_ROOT
# ---------------------------------------------------------------------------
set -euo pipefail
DEST=${1:-$PWD/DATA_ROOT}
REPO=koalapenguin/RoboCloth
if [[ -n ${ROBOCLOTH_MATERIALS:-} ]]; then
    read -r -a IDS <<< "$ROBOCLOTH_MATERIALS"
else
    IDS=({0..499})
fi

if (( ${#IDS[@]} == 0 )); then
    echo "[download_dataset_full] ERROR: no material IDs requested" >&2
    exit 1
fi

INCLUDES=(--include "globals/*")
for id in "${IDS[@]}"; do
    if [[ ! $id =~ ^[0-9]+$ ]]; then
        echo "[download_dataset_full] ERROR: invalid material ID: $id" >&2
        exit 1
    fi
    INCLUDES+=(--include "materials/$id/*")
done

echo "[download_dataset_full] full dataset is ~3.5 TB — ensure disk space."
mkdir -p "$DEST"
hf download "$REPO" --repo-type dataset "${INCLUDES[@]}" --local-dir "$DEST"
cp -f "$DEST"/globals/* "$DEST"/
for id in "${IDS[@]}"; do
    src="$DEST/materials/$id"
    if [[ ! -d $src ]]; then
        echo "[download_dataset_full] ERROR: requested material $id was not downloaded" >&2
        exit 1
    fi
    rm -rf "$DEST/$id"
    mv "$src" "$DEST/$id"
    if [ -f "$DEST/$id/hdr.tar" ]; then
        tar -xf "$DEST/$id/hdr.tar" -C "$DEST/$id" && rm "$DEST/$id/hdr.tar"
    fi
done
rmdir "$DEST/materials" 2>/dev/null || true
echo "[download_dataset_full] ready: $DEST ($(ls -d "$DEST"/[0-9]* 2>/dev/null | wc -l) materials)"
