#!/bin/bash
# ---------------------------------------------------------------------------
# Download ONE material (dense capture data: HDR views + reconstruction files)
# plus the dataset-level calibration/split files, and assemble a training-ready
# DATA_ROOT. The per-material hdr.tar is extracted and removed afterwards.
#
# Usage:
#   bash scripts/download_material.sh <material_id> [DATA_ROOT]   # default ./DATA_ROOT
# ---------------------------------------------------------------------------
set -euo pipefail
ID=${1:?usage: download_material.sh <material_id> [DATA_ROOT]}
DEST=${2:-$PWD/DATA_ROOT}
REPO=koalapenguin/RoboCloth

if [[ ! $ID =~ ^[0-9]+$ ]]; then
    echo "[download_material] ERROR: material_id must be numeric: $ID" >&2
    exit 1
fi

mkdir -p "$DEST"
hf download "$REPO" --repo-type dataset \
    --include "globals/*" --include "materials/$ID/*" --local-dir "$DEST"

cp -f "$DEST"/globals/* "$DEST"/
rm -rf "$DEST/$ID" && mv "$DEST/materials/$ID" "$DEST/$ID"
if [ -f "$DEST/$ID/hdr.tar" ]; then
    tar -xf "$DEST/$ID/hdr.tar" -C "$DEST/$ID" && rm "$DEST/$ID/hdr.tar"
fi
echo "[download_material] ready: $DEST/$ID ($(ls "$DEST/$ID/hdr" | wc -l) HDR views)"
