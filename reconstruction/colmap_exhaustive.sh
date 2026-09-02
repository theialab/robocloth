#!/usr/bin/env bash
# COLMAP sparse reconstruction of one captured material — exhaustive matcher.
# Retry variant of colmap.sh for materials whose sequential run registered
# fewer than COLMAP_REGISTRATION_THRESHOLD of the frames (scheduler.py
# dispatches it automatically). Only the matcher differs; the fail-closed
# contract is identical to colmap.sh:
#
#   * strict mode: the first failing stage aborts the run with a nonzero status;
#     the EXIT trap appends "== FAILED: stage=<name> exit=<code> ==" to the log;
#     SIGINT / SIGTERM are trapped into exit 130 / 143 and take the same path
#     (once the current COLMAP stage has exited — see colmap.sh);
#   * everything is built in a local staging dir under COLMAP_TMP (default
#     /tmp/robocloth_colmap) and the staging dir is removed on every exit path
#     except SIGKILL;
#   * the staged model is validated (registration_check.py) BEFORE anything
#     under <project_dir> is touched;
#   * on success the previous <project_dir>/sparse (if any) is renamed to
#     sparse.prev-<timestamp> — never deleted — and the new result is renamed in.
#     (This replaces the old `sparse_seq_failed` backup: a rejected sequential
#     run is no longer published at all, so there is normally nothing to back
#     up when the exhaustive retry starts.)
#   * on failure <project_dir>/sparse is left exactly as it was; whatever was
#     staged is kept at <project_dir>/sparse.failed-<timestamp> for post-mortem.
#
# Usage:  bash colmap_exhaustive.sh <project_dir> <gpu_id>
# Exit status: 0 ok · 2 staged model unreadable/empty · 3 registration below the
# threshold · 130 / 143 interrupted by SIGINT / SIGTERM · otherwise the failing
# stage's own status.
# Environment: COLMAP_TMP, COLMAP_REGISTRATION_THRESHOLD, PYTHON (see colmap.sh).
set -euo pipefail

PROJECT="${1:?usage: $0 <project_dir> <gpu_id>}"
gpu_id="${2:?usage: $0 <project_dir> <gpu_id>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# === LOCAL STORAGE SETUP (avoid slow NFS I/O) ===
TMP_BASE="${COLMAP_TMP:-/tmp/robocloth_colmap}"
PROJECT_NAME=$(basename "${PROJECT}")
TMP_PROJECT="${TMP_BASE}/${PROJECT_NAME}_$$"  # $$ = PID for uniqueness
RUN_TS="$(date +%Y%m%d-%H%M%S)"

# Name of the stage currently running, reported by the EXIT trap on failure.
STAGE="setup"

cleanup() {
    local rc="$1"     # $? at the moment the EXIT trap fired
    local out keep
    CLEANUP_STARTED=1
    trap '' INT TERM  # finish tidying up even if signalled again
    trap - EXIT
    set +e
    if [ "${rc}" -ne 0 ]; then
        echo "== FAILED: stage=${STAGE} exit=${rc} =="
        # Never touch ${PROJECT}/sparse here. Keep the staged output (if any)
        # for post-mortem and drop any half-copied publish dir of this run.
        for out in sparse undistorted; do
            rm -rf "${PROJECT}/.${out}.incoming-${RUN_TS}-$$"
            if [ -d "${TMP_PROJECT}/${out}" ] && [ -n "$(ls -A "${TMP_PROJECT}/${out}" 2>/dev/null)" ]; then
                keep="${PROJECT}/${out}.failed-${RUN_TS}"
                if [ -e "${keep}" ]; then keep="${keep}-$$"; fi
                if mv "${TMP_PROJECT}/${out}" "${keep}"; then
                    echo "Staged ${out} kept at ${keep}"
                else
                    echo "WARNING: could not keep staged ${out} at ${keep}"
                fi
            fi
        done
    fi
    remove_staging
    exit "${rc}"
}

remove_staging() {
    if [ -n "${TMP_PROJECT}" ] && [ "${TMP_PROJECT}" != "/" ] && [ -d "${TMP_PROJECT}" ]; then
        rm -rf "${TMP_PROJECT}"
        echo "Temporary directory removed: ${TMP_PROJECT}"
    fi
}

# Publish one staged output dir into the project. The staged copy is first
# duplicated next to its destination (same filesystem), then the previous
# result is renamed to <name>.prev-<timestamp> and the new one renamed in, so
# an interrupted publish can never leave a half-written <name>/.
publish_dir() {
    local name="$1"
    local staged="${TMP_PROJECT}/${name}"
    local dest="${PROJECT}/${name}"
    local incoming="${PROJECT}/.${name}.incoming-${RUN_TS}-$$"
    local prev="${PROJECT}/${name}.prev-${RUN_TS}"
    cp -a "${staged}" "${incoming}"
    # The two renames must not be split by a signal (<name>/ would be missing
    # in between): ignore INT/TERM around these two instant operations only.
    trap '' INT TERM
    if [ -e "${dest}" ]; then
        if [ -e "${prev}" ]; then prev="${prev}-$$"; fi
        mv "${dest}" "${prev}"
        echo "Previous ${name} kept at ${prev}"
    fi
    mv "${incoming}" "${dest}"
    arm_signal_traps
    echo "${name} output published to ${dest}"
}

# An interrupted run takes the same fail-closed exit path as a failed stage,
# with the 128+signal status (bash defers the trap until the current foreground
# COLMAP stage has exited — see colmap.sh). Once cleanup() is running, a
# further signal is ignored so the tidy-up completes.
CLEANUP_STARTED=""
on_signal() {
    trap '' INT TERM
    [ -n "${CLEANUP_STARTED}" ] || exit "$1"
}
arm_signal_traps() {
    trap 'on_signal 130' INT
    trap 'on_signal 143' TERM
}
arm_signal_traps
trap 'cleanup $?' EXIT

echo "== Setting up local storage =="
echo "Original project: ${PROJECT}"
echo "Temp project: ${TMP_PROJECT}"

STAGE="preflight"
[ -d "${PROJECT}/ldr" ] || { echo "ERROR: ${PROJECT}/ldr not found"; exit 1; }
[ -f "${PROJECT}/scan_log.json" ] || { echo "ERROR: ${PROJECT}/scan_log.json not found (needed by the registration gate)"; exit 1; }
command -v colmap >/dev/null 2>&1 || { echo "ERROR: colmap not found on PATH"; exit 1; }

# Create tmp project directory
mkdir -p "${TMP_PROJECT}"

# Copy images to local storage
STAGE="copy_images"
echo "== Copying images to local storage =="
cp -r "${PROJECT}/ldr" "${TMP_PROJECT}/ldr"
echo "Images copied to ${TMP_PROJECT}/ldr"

# Define paths using local storage
IMG_DIR="${TMP_PROJECT}/ldr"
DB="${TMP_PROJECT}/database.db"
OUT_SPARSE="${TMP_PROJECT}/sparse"
UNDIST_OUT="${TMP_PROJECT}/undistorted"  # published as well if a stage ever fills it

STAGE="feature_extractor"
echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES="${gpu_id}" colmap feature_extractor \
    --database_path "${DB}" \
    --image_path "${IMG_DIR}" \
    --ImageReader.single_camera=true

STAGE="exhaustive_matcher"
echo "== Exhaustive matching =="
CUDA_VISIBLE_DEVICES="${gpu_id}" colmap exhaustive_matcher --database_path "${DB}"

STAGE="mapper"
echo "== Mapping =="
mkdir -p "${OUT_SPARSE}"
CUDA_VISIBLE_DEVICES="${gpu_id}" colmap mapper \
    --database_path="${DB}" \
    --image_path="${IMG_DIR}" \
    --output_path="${OUT_SPARSE}"

STAGE="select_submodel"
echo "== Using submodel 0 (largest) as final reconstruction =="
# model_merger silently fails when submodels don't share images, producing a
# corrupt merged model. Keep only sparse/0 (the largest submodel).
[ -f "${OUT_SPARSE}/0/images.bin" ] || { echo "ERROR: mapper produced no submodel 0 in ${OUT_SPARSE}"; exit 1; }
cp "${OUT_SPARSE}"/0/*.bin "${OUT_SPARSE}/"

STAGE="model_converter_txt"
echo "== Convert sparse model to text =="
colmap model_converter \
    --input_path   "${OUT_SPARSE}" \
    --output_path  "${OUT_SPARSE}" \
    --output_type  TXT

STAGE="model_converter_ply"
echo "== Convert sparse model to ply =="
CUDA_VISIBLE_DEVICES="${gpu_id}" colmap model_converter \
    --input_path "${OUT_SPARSE}" \
    --output_path "${OUT_SPARSE}/points3D.ply" \
    --output_type PLY

# === VALIDATE BEFORE PUBLISHING ===
STAGE="validate"
echo "== Validating staged sparse model =="
"${PYTHON_BIN}" "${HERE}/registration_check.py" \
    --sparse-dir "${OUT_SPARSE}" \
    --scan-log "${PROJECT}/scan_log.json"

# === PUBLISH INTO THE PROJECT (previous result kept as *.prev-<timestamp>) ===
STAGE="publish"
echo "== Publishing sparse output to original project =="
publish_dir sparse
if [ -d "${UNDIST_OUT}" ]; then
    publish_dir undistorted
fi

# === CLEANUP ===
STAGE="cleanup"
echo "== Cleaning up temporary files =="
remove_staging

echo "== Finished COLMAP reconstruction =="
