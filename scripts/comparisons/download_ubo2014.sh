#!/bin/bash
# ---------------------------------------------------------------------------
# Download the 12 held-out UBO2014 BTF materials used for Table 2 from the
# University of Bonn BTF Database (BTFDBB).
#
# Source:  BTF Database Bonn, UBO2014
#          page:  https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/
#          files: https://cg.cs.uni-bonn.de/btf/UBO2014/<category>/<material>_W400xH400_L151xV151.btf
#          (category is the material name without its number: carpet/fabric/felt)
#          Terms: see the BTFDBB terms of use on the page above — the download
#          page is JavaScript-only and its terms could not be quoted here, so
#          read them there before using the data, and cite the dataset as the
#          BTFDBB asks.
# These files used to be mirrored in our Hugging Face bundle; they are not any
# more — fetch them from the original source with this script.
#
# Usage:
#   bash scripts/comparisons/download_ubo2014.sh /absolute/path/to/DATA_ROOT
#   bash scripts/comparisons/download_ubo2014.sh /absolute/path/to/DATA_ROOT \
#        --verify /absolute/path/to/external_checksums.json
#
# Result (the layout run_table_ubo.sh / train_stage2_ubo.sh expect as DATA_ROOT):
#   DATA_ROOT/UBO2014/<material>_W400xH400_L151xV151.btf     12 files, ~1.0 GB
#   DATA_ROOT=/absolute/path/to/DATA_ROOT/UBO2014 bash scripts/comparisons/run_table_ubo.sh
# Reading .btf needs btf_extractor:
#   pip install Cython && pip install btf_extractor==1.7.0 --no-build-isolation
#
# --verify FILE compares every file against a JSON of sha256 digests
# ({"<path ending in <material>_W400xH400_L151xV151.btf>": {"sha256":..., "size":...}}).
# Without it the script still checks every file's exact size and fails closed.
#
# If the server cannot be reached (or the URL pattern has changed) the script
# prints manual-download instructions and exits 2 — it never leaves a partial
# tree behind as if it had succeeded.
#
# Environment: UBO2014_BASE_URL (override the file base URL).
# Exit codes: 0 ok, 1 usage/environment, 2 download unavailable or integrity failure.
# ---------------------------------------------------------------------------
set -euo pipefail

BASE_URL=${UBO2014_BASE_URL:-https://cg.cs.uni-bonn.de/btf/UBO2014}
PAGE_URL=https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/
SUFFIX=_W400xH400_L151xV151.btf

# the 12 held-out materials of Table 2, with the exact published file size
MATERIALS=(carpet02 carpet07 carpet09 carpet12
           fabric02 fabric04 fabric09 fabric11
           felt01 felt03 felt05 felt10)
SIZES=(83570143 83619582 83747666 83747666
       83553711 83553711 82878555 82878555
       84349585 84349585 83841579 83601338)

log() { printf '[download_ubo2014] %s\n' "$*"; }
die() { printf '[download_ubo2014] ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

manual_instructions() {
    cat >&2 <<EOF
[download_ubo2014] The UBO2014 BTF files could not be fetched automatically.

Download them by hand instead:
  1. open  $PAGE_URL
     and accept the BTFDBB terms of use;
  2. download these 12 files (the 400x400 px, 151x151 angle variant):
$(printf '       %s\n' "${MATERIALS[@]/%/$SUFFIX}")
  3. put them, unrenamed and flat (no per-material subfolders), into
       $UBO_DIR/
  4. re-run this script to check sizes (add --verify <json> for sha256).
EOF
}

# ---- arguments -------------------------------------------------------------
DATA_ROOT=${1:-}
[ -n "$DATA_ROOT" ] || die "usage: download_ubo2014.sh /absolute/path/to/DATA_ROOT [--verify /absolute/path/to/external_checksums.json]"
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

command -v curl >/dev/null 2>&1 || die "'curl' not found on PATH"
PYTHON_BIN=${PYTHON:-python3}
if [ -n "$VERIFY_JSON" ]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "--verify needs a python3 interpreter (set PYTHON=)"
fi

UBO_DIR=$DATA_ROOT/UBO2014
mkdir -p "$UBO_DIR"

# ---- 1. which files are still missing --------------------------------------
missing=()
for i in "${!MATERIALS[@]}"; do
    mat=${MATERIALS[$i]}
    dest=$UBO_DIR/$mat$SUFFIX
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "${SIZES[$i]}" ]; then
        continue
    fi
    missing+=("$i")
done

if [ ${#missing[@]} -eq 0 ]; then
    log "all ${#MATERIALS[@]} BTF files already present with the expected sizes"
else
    # ---- 2. probe the server before downloading anything --------------------
    probe_mat=${MATERIALS[${missing[0]}]}
    probe_cat=${probe_mat%%[0-9]*}
    probe_url=$BASE_URL/$probe_cat/$probe_mat$SUFFIX
    log "probing $probe_url"
    code=$(curl -sS -I -L --max-time 60 -o /dev/null -w '%{http_code}' "$probe_url" || echo 000)
    if [ "$code" != "200" ]; then
        log "probe returned HTTP $code — no verified direct download URL"
        manual_instructions
        exit 2
    fi
    log "probe OK (HTTP 200); downloading ${#missing[@]} file(s) -> $UBO_DIR"

    # ---- 3. download --------------------------------------------------------
    for i in "${missing[@]}"; do
        mat=${MATERIALS[$i]}
        cat=${mat%%[0-9]*}
        url=$BASE_URL/$cat/$mat$SUFFIX
        dest=$UBO_DIR/$mat$SUFFIX
        log "  $mat ($(( SIZES[i] / 1000000 )) MB)"
        curl -fL -C - --retry 3 --retry-delay 5 --max-time 3600 -o "$dest" "$url" || {
            log "download failed for $mat ($url)"
            manual_instructions
            exit 2
        }
    done
fi

# ---- 4. sizes ---------------------------------------------------------------
problems=0
for i in "${!MATERIALS[@]}"; do
    mat=${MATERIALS[$i]}
    dest=$UBO_DIR/$mat$SUFFIX
    if [ ! -f "$dest" ]; then
        printf '[download_ubo2014]   MISSING  %s\n' "$mat$SUFFIX" >&2
        problems=$((problems + 1))
        continue
    fi
    have=$(stat -c %s "$dest")
    if [ "$have" != "${SIZES[$i]}" ]; then
        printf '[download_ubo2014]   SIZE     %s: %s bytes, expected %s\n' "$mat$SUFFIX" "$have" "${SIZES[$i]}" >&2
        problems=$((problems + 1))
    fi
done
if [ "$problems" -ne 0 ]; then
    log "$problems file(s) missing or the wrong size"
    manual_instructions
    exit 2
fi
log "${#MATERIALS[@]} BTF files present with the expected sizes"

# ---- 5. optional sha256 verification ----------------------------------------
if [ -n "$VERIFY_JSON" ]; then
    log "verifying sha256 against $VERIFY_JSON"
    "$PYTHON_BIN" - "$UBO_DIR" "$VERIFY_JSON" <<'PY' || exit 2
import hashlib, json, os, sys
ubo_dir, ref_path = sys.argv[1], sys.argv[2]
with open(ref_path) as f:
    ref = json.load(f)
flat = {}
items = ref.items() if isinstance(ref, dict) else ((e.get("path"), e) for e in ref)
for key, val in items:
    if not key:
        continue
    digest = val if isinstance(val, str) else (val or {}).get("sha256")
    if digest:
        flat[os.path.basename(key)] = digest
problems, checked = [], 0
for name in sorted(os.listdir(ubo_dir)):
    if not name.endswith(".btf"):
        continue
    want = flat.get(name)
    if want is None:
        problems.append(f"NO REFERENCE  {name}")
        continue
    h = hashlib.sha256()
    with open(os.path.join(ubo_dir, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        problems.append(f"SHA256        {name}: {h.hexdigest()} != {want}")
    else:
        checked += 1
if problems:
    print(f"[download_ubo2014] VERIFICATION FAILED - {len(problems)} problem(s):")
    for p in problems:
        print("  " + p)
    sys.exit(2)
print(f"[download_ubo2014] sha256 OK for all {checked} files")
PY
fi

log "ready: $UBO_DIR"
log "use it as DATA_ROOT for scripts/comparisons/run_table_ubo.sh and train_stage2_ubo.sh"
