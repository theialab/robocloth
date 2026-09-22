#!/usr/bin/env bash
# Build the exp-015 Mitsuba GPU render environment on Leonardo.
#
# RUN THIS ON A LOGIN NODE: compute nodes have no outbound internet.
# The venv is built in place at $VM_ENV_DIR and is NOT relocatable
# (exp-006 lost an env to an atomic move; do not copy it afterwards).
#
#   bash videomaterial/leonardo/env/build_env.sh            # build (idempotent-ish)
#   VM_ENV_FORCE=1 bash .../build_env.sh                    # wipe and rebuild
#
# Everything is logged to $VM_ENV_DIR.build.log.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/../config/paths.sh"

TORCH_VERSION="2.4.1"
TORCHVISION_VERSION="0.19.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

LOG="${VM_ENV_DIR}.build.log"
mkdir -p "$(dirname "$VM_ENV_DIR")"

if [ "${VM_ENV_FORCE:-0}" = "1" ] && [ -d "$VM_ENV_DIR" ]; then
    echo "removing existing env $VM_ENV_DIR"
    rm -rf "$VM_ENV_DIR"
fi

if [ -x "$VM_PYTHON" ]; then
    echo "env already present at $VM_ENV_DIR (use VM_ENV_FORCE=1 to rebuild)"
else
    module purge
    module load python/3.11.7
    python3 -m venv "$VM_ENV_DIR"
fi

# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

{
    echo "=== build started $(date -Is) on $(hostname) ==="
    python -V
    python -m pip install --upgrade pip wheel setuptools
    python -m pip install --index-url "$TORCH_INDEX" \
        "torch==${TORCH_VERSION}+cu124" "torchvision==${TORCHVISION_VERSION}+cu124"
    python -m pip install -r "$_here/requirements.txt"
    # Optional: only used by the UBO BTF path, which this benchmark never
    # touches (brdf_plugin/material/BTF.py imports it inside a function).
    # It needs a C build, so a failure here is reported and tolerated.
    python -m pip install "btf_extractor==1.7.0" || \
        echo "WARN: btf_extractor not installed (lazy import only, benchmark unaffected)"
    python -m pip check || echo "WARN: pip check reported issues (see above)"
    echo "=== build finished $(date -Is) ==="
} 2>&1 | tee -a "$LOG"

echo
echo "--- recorded versions ---"
python - <<'PY' | tee -a "$LOG"
import importlib, json, platform, sys
out = {"python": platform.python_version(), "executable": sys.executable}
for name in ("mitsuba", "drjit", "torch", "torchvision", "numpy", "scipy",
             "imageio", "PIL", "cv2", "hydra", "omegaconf",
             "pytorch_lightning", "OpenEXR"):
    try:
        m = importlib.import_module(name)
        out[name] = getattr(m, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        out[name] = f"MISSING ({type(exc).__name__}: {exc})"
print(json.dumps(out, indent=2))
PY

python -m pip freeze > "${VM_ENV_DIR}.pip-freeze.txt"
du -sb "$VM_ENV_DIR" | tee -a "$LOG"
echo "env ready: $VM_ENV_DIR"
