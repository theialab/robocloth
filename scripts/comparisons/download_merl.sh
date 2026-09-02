#!/bin/bash
# ---------------------------------------------------------------------------
# Download the MERL BRDF database (the "MERL" decoder column of the paper) from
# its official archive on Zenodo and lay it out the way this code base expects.
#
# Source:  MERL BRDF Database, Zenodo record 8101681
#          https://zenodo.org/records/8101681   (DOI 10.5281/zenodo.8101680)
#          one archive, BRDFDatabase.zip (1.25 GB), holding
#          BRDFDatabase/brdfs/<material>.binary  (100 files, 34,992,012 B each)
#          License: CC-BY-SA-4.0 (as published on the Zenodo record).
# These files used to be mirrored in our Hugging Face bundle; they are not any
# more — fetch them from the original source with this script.
#
# Usage:
#   bash scripts/comparisons/download_merl.sh /absolute/path/to/DATA_ROOT
#   bash scripts/comparisons/download_merl.sh /absolute/path/to/DATA_ROOT \
#        --verify /absolute/path/to/external_checksums.json
#
# Result (the layout configs/data/merl.yaml + training/datasets/merl.py read):
#   DATA_ROOT/MERL/brdfs/<material>.binary          100 files, ~3.3 GB
#   DATA_ROOT/MERL/BRDFDatabase.zip                 the archive, kept for reuse
# Point stage-1 training at the brdfs folder:
#   DATA_ROOT=/absolute/path/to/DATA_ROOT/MERL/brdfs bash scripts/comparisons/train_stage1_merl.sh
#
# --verify FILE compares every extracted file against a JSON of sha256 digests
# ({"<path ending in MERL/brdfs/<name>.binary>": {"sha256": ..., "size": ...}}).
# Without it the script still checks the archive's md5, the file count and every
# file's size, and fails closed on any mismatch.
#
# Environment: MERL_ZENODO_URL (override the archive URL).
# Exit codes: 0 ok, 1 usage/environment, 2 download or integrity failure.
# ---------------------------------------------------------------------------
set -euo pipefail

ZENODO_URL=${MERL_ZENODO_URL:-https://zenodo.org/records/8101681/files/BRDFDatabase.zip}
ARCHIVE_NAME=BRDFDatabase.zip
ARCHIVE_MD5=7141af4c12b4c4feed299769260b3604
ARCHIVE_BYTES=1253117184
EXPECTED_COUNT=100
EXPECTED_FILE_BYTES=34992012

log() { printf '[download_merl] %s\n' "$*"; }
die() { printf '[download_merl] ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

# ---- arguments -------------------------------------------------------------
DATA_ROOT=${1:-}
[ -n "$DATA_ROOT" ] || die "usage: download_merl.sh /absolute/path/to/DATA_ROOT [--verify /absolute/path/to/external_checksums.json]"
case $DATA_ROOT in
    /*) ;;
    *) die "DATA_ROOT must be an absolute path (got '$DATA_ROOT')" ;;
esac
shift
VERIFY_JSON=
while [ $# -gt 0 ]; do
    case $1 in
        --verify)
            [ $# -ge 2 ] || die "--verify needs the path of a checksum JSON"
            VERIFY_JSON=$2
            case $VERIFY_JSON in /*) ;; *) die "--verify needs an absolute path (got '$VERIFY_JSON')" ;; esac
            [ -f "$VERIFY_JSON" ] || die "--verify file does not exist: $VERIFY_JSON"
            shift 2
            ;;
        *) die "unknown argument: $1" ;;
    esac
done

command -v unzip >/dev/null 2>&1 || die "'unzip' not found on PATH"
if command -v curl >/dev/null 2>&1; then
    FETCH=curl
elif command -v wget >/dev/null 2>&1; then
    FETCH=wget
else
    die "neither curl nor wget found on PATH"
fi
PYTHON_BIN=${PYTHON:-python3}
if [ -n "$VERIFY_JSON" ]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "--verify needs a python3 interpreter (set PYTHON=)"
fi

MERL_ROOT=$DATA_ROOT/MERL
BRDF_DIR=$MERL_ROOT/brdfs
ARCHIVE=$MERL_ROOT/$ARCHIVE_NAME
mkdir -p "$BRDF_DIR"

# ---- 1. archive -------------------------------------------------------------
if [ -f "$ARCHIVE" ] && [ "$(stat -c %s "$ARCHIVE")" = "$ARCHIVE_BYTES" ]; then
    log "archive already present: $ARCHIVE"
else
    log "downloading $ZENODO_URL -> $ARCHIVE (1.25 GB, resumable)"
    if [ "$FETCH" = curl ]; then
        curl -fL -C - --retry 3 --retry-delay 5 -o "$ARCHIVE" "$ZENODO_URL" \
            || die "download failed: $ZENODO_URL" 2
    else
        wget -c -O "$ARCHIVE" "$ZENODO_URL" || die "download failed: $ZENODO_URL" 2
    fi
fi

have_bytes=$(stat -c %s "$ARCHIVE")
[ "$have_bytes" = "$ARCHIVE_BYTES" ] \
    || die "archive size mismatch: $ARCHIVE is $have_bytes bytes, expected $ARCHIVE_BYTES (delete it and re-run)" 2
have_md5=$(md5sum "$ARCHIVE" | awk '{print $1}')
[ "$have_md5" = "$ARCHIVE_MD5" ] \
    || die "archive md5 mismatch: got $have_md5, Zenodo publishes $ARCHIVE_MD5 (delete it and re-run)" 2
log "archive md5 OK ($have_md5)"

# ---- 2. extract the .binary files ------------------------------------------
log "extracting BRDFDatabase/brdfs/*.binary -> $BRDF_DIR"
unzip -o -q -j "$ARCHIVE" 'BRDFDatabase/brdfs/*.binary' -d "$BRDF_DIR" \
    || die "unzip failed for $ARCHIVE" 2

# ---- 3. count and sizes ------------------------------------------------------
count=$(find "$BRDF_DIR" -maxdepth 1 -name '*.binary' -type f | wc -l)
[ "$count" -eq "$EXPECTED_COUNT" ] \
    || die "expected $EXPECTED_COUNT .binary files in $BRDF_DIR, found $count" 2
bad=$(find "$BRDF_DIR" -maxdepth 1 -name '*.binary' -type f -not -size "${EXPECTED_FILE_BYTES}c" -printf '%f ' || true)
[ -z "$bad" ] || die "wrong size (expected $EXPECTED_FILE_BYTES bytes): $bad" 2
log "$count .binary files, all $EXPECTED_FILE_BYTES bytes"

# ---- 4. optional sha256 verification ----------------------------------------
if [ -n "$VERIFY_JSON" ]; then
    log "verifying sha256 against $VERIFY_JSON"
    "$PYTHON_BIN" - "$BRDF_DIR" "$VERIFY_JSON" <<'PY' || exit 2
import hashlib, json, os, sys
brdf_dir, ref_path = sys.argv[1], sys.argv[2]
with open(ref_path) as f:
    ref = json.load(f)
# accept {"path": {"sha256": ...}} / {"path": "<sha256>"} / [{"path":..,"sha256":..}]
flat = {}
items = ref.items() if isinstance(ref, dict) else ((e.get("path"), e) for e in ref)
for key, val in items:
    if not key:
        continue
    digest = val if isinstance(val, str) else (val or {}).get("sha256")
    if digest:
        flat[os.path.basename(key)] = digest
problems, checked = [], 0
for name in sorted(os.listdir(brdf_dir)):
    if not name.endswith(".binary"):
        continue
    want = flat.get(name)
    if want is None:
        problems.append(f"NO REFERENCE  {name}")
        continue
    h = hashlib.sha256()
    with open(os.path.join(brdf_dir, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        problems.append(f"SHA256        {name}: {h.hexdigest()} != {want}")
    else:
        checked += 1
if problems:
    print(f"[download_merl] VERIFICATION FAILED - {len(problems)} problem(s):")
    for p in problems:
        print("  " + p)
    sys.exit(2)
print(f"[download_merl] sha256 OK for all {checked} files")
PY
fi

log "ready: $BRDF_DIR ($count materials)"
log "the archive is kept at $ARCHIVE — remove it yourself to free 1.25 GB"
