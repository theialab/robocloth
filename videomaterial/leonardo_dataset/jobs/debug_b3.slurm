#!/bin/bash
#SBATCH --job-name=vm-b3-debug
#SBATCH --account=IscrB_OVER
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/outputs/synthetic_experiments/exp20_b3_render/logs/b3_debug-%j.out
#SBATCH --error=/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/outputs/synthetic_experiments/exp20_b3_render/logs/b3_debug-%j.err
# B3 debug run (cluster rule: a minimal 1-GPU short job before any long one).
# B3 = jointly moving point light AND camera: frame k = light of frame k of a B1 sequence,
# camera pose of frame k of a B2 sequence. Three FULL manifest sequences (81 frames, material
# 64 spp + pbr_patch twin 32 spp) - one train/pair, one test/unseen_light, one
# test/unseen_camera - into a DEBUG root, so the B3 dataset tree stays untouched by
# possibly-wrong code; then the white-Lambert G0 renders of the same poses and all gates.
#
# What the run must show before the array may be submitted:
#   * 3x RENDER_COMPLETE, 81 frames_png + 81 pbr_patch_png each (frames_exr: all three keep it),
#   * sha256sum -c MANIFEST.sha256 clean,
#   * source check exact (render_manifest refuses otherwise) and check_metadata --sources pass,
#   * white-Lambert gates: sub-pixel edge re-projection < 1 px, every mismatched pixel centre within
#     the box filter support (0.5 px per axis, L-inf) of the projected outline, interior irradiance PSNR > 40 dB,
#   * material (EXR) and twin vs the projected outline of the logged pose: spill <= 2.05 px L-inf (separable gaussian
#     filter radius 2 px), no holes in the material EXR or the twin (PNG holes = dark texels, reported),
#   * metadata setup == "B3", camera.mode == light.mode == "trajectory", generator.commit ==
#     SYNCED_COMMIT,
#   * gap between consecutive sequences < 5 s (the checkpoint is NOT reloaded),
#   * per-sequence wall -> projected billing-h of the 1,000-sequence array (billing=8 per A100).
#
#   sbatch videomaterial/leonardo_dataset/jobs/debug_b3.slurm      (from $VM_REPO_ROOT)
set -euo pipefail

REPO="${VM_REPO_ROOT:-/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-b3-joint}"
source "$REPO/videomaterial/leonardo_dataset/paths.sh"

# B3 paths. paths.sh describes the B1 render and is not edited by this experiment, so the B3
# root / manifest / run root are defined here and never inherit a B1 or B2 value.
export VM_DATA_ROOT="${VM_DATA_ROOT:-$VM_WORK_ROOT/data/Robocloth_synthetic_sequence}"
export VM_B3_ROOT="${VM_B3_ROOT:-$VM_DATA_ROOT/B3_Joint}"
export VM_B3_MANIFEST="${VM_B3_MANIFEST:-$VM_B3_ROOT/manifest.json}"
export VM_B3_RUN_ROOT="${VM_B3_RUN_ROOT:-$VM_WORK_ROOT/outputs/synthetic_experiments/exp20_b3_render}"
export VM_B3_DEBUG_ROOT="${VM_B3_DEBUG_ROOT:-$VM_B3_RUN_ROOT/debug/B3_Joint}"
DEBUG_SEQS="${VM_DEBUG_SEQS:-train/pair_0001,test/unseen_light_0001,test/unseen_camera_0001}"
CHECK_FRAMES="0,10,20,30,40,50,60,70,80"

module purge
module load python/3.11.7
# shellcheck disable=SC1091
source "$VM_ENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
test -f "$VM_CKPT" || { echo "missing checkpoint $VM_CKPT"; exit 1; }
test -f "$VM_B3_MANIFEST" || { echo "missing manifest $VM_B3_MANIFEST"; exit 1; }
test -f "$REPO/CODE_VERSION" || { echo "missing CODE_VERSION stamp at $REPO"; exit 1; }
cat "$REPO/CODE_VERSION"
# hard guard: this job must only ever render a B3 manifest.
python - "$VM_B3_MANIFEST" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
assert m["setup"] == "B3", f'manifest setup is {m["setup"]!r}, refusing to render'
print(f'manifest ok: setup={m["setup"]} twin={m["twin"]} sequences={m["counts"]["total"]} '
      f'material_spp={m["render"]["material"]["spp"]} twin_spp={m["render"][m["twin"]]["spp"]} master_seed={m["master_seed"]}')
PY
mkdir -p "$VM_B3_DEBUG_ROOT" "$VM_B3_RUN_ROOT/logs" "$VM_B3_RUN_ROOT/debug/white_lambert" "$VM_B3_RUN_ROOT/debug/gates"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# ---- 1. the three sequences, exactly as the array renders them (+ exact source check)
start=$(date +%s)
set +e
python "$VM_PKG_DIR/render_manifest.py" \
    --manifest "$VM_B3_MANIFEST" \
    --only "$DEBUG_SEQS" \
    --root "$VM_B3_DEBUG_ROOT" \
    --source-root "$VM_DATA_ROOT" \
    --scene-src "$VM_EXP005_DIR" \
    --checkpoint-root "$VM_CKPT_ROOT" \
    --rendering-root "$VM_RENDERING_ROOT"
status=$?
set -e
end=$(date +%s)
echo "render wall seconds: $((end - start)) exit=$status"
echo "RENDER_COMPLETE files: $(find "$VM_B3_DEBUG_ROOT" -name RENDER_COMPLETE | wc -l)"

# ---- 2. white-Lambert renders of the same poses (G0 silhouette / irradiance gates)
wl_start=$(date +%s)
for d in ${DEBUG_SEQS//,/ }; do
    name="$(basename "$d")"
    python "$VM_PKG_DIR/render_sequence.py" \
        --manifest "$VM_B3_MANIFEST" --sequence "$d" --source-root "$VM_DATA_ROOT" \
        --mode white_lambert --spp 64 --batch-spp 16 --frame-indices "$CHECK_FRAMES" \
        --output "$VM_B3_RUN_ROOT/debug/white_lambert/$name" \
        --scene-src "$VM_EXP005_DIR" --checkpoint-root "$VM_CKPT_ROOT" --rendering-root "$VM_RENDERING_ROOT" \
        | tail -1 || status=1
done
echo "white_lambert wall seconds: $(( $(date +%s) - wl_start ))"

# ---- 3. gates (CPU): checksums, derivability + alignment + sources, white-Lambert G0
for d in ${DEBUG_SEQS//,/ }; do
    name="$(basename "$d")"; seq="$VM_B3_DEBUG_ROOT/$d"
    if [ -f "$seq/MANIFEST.sha256" ]; then
        (cd "$seq" && sha256sum -c --quiet MANIFEST.sha256) && echo "sha256 ok: $d" || { echo "sha256 FAILED: $d"; status=1; }
        python "$VM_PKG_DIR/check_metadata.py" "$seq" --align --sources "$VM_DATA_ROOT" \
            --out "$VM_B3_RUN_ROOT/debug/gates/${name}_sequence.json" > /dev/null || status=1
    else
        echo "missing MANIFEST.sha256: $d"; status=1
    fi
    wl="$VM_B3_RUN_ROOT/debug/white_lambert/$name"
    if [ -f "$wl/frames.jsonl" ]; then
        python "$VM_PKG_DIR/check_metadata.py" "$wl" --outline-px 0.52 --edge-px 1.0 --psnr 40 \
            --out "$VM_B3_RUN_ROOT/debug/gates/${name}_white_lambert.json" > /dev/null || status=1
    fi
done
python - "$VM_B3_RUN_ROOT/debug/gates" "$VM_B3_DEBUG_ROOT" "$REPO/SYNCED_COMMIT" ${DEBUG_SEQS//,/ } <<'PY' || status=1
import json, sys
from pathlib import Path
gates, root, synced, seqs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).read_text().strip(), sys.argv[4:]
ok = True
for d in seqs:
    name = Path(d).name
    s = json.loads((gates / f"{name}_sequence.json").read_text())
    w = json.loads((gates / f"{name}_white_lambert.json").read_text())
    m = json.loads((root / d / "metadata.json").read_text())
    a, src = s["alignment"], s["sources"]
    row = {"seq": d, "setup": m["setup"], "cam_mode": m["camera"]["mode"], "light_mode": m["light"]["mode"],
           "commit_ok": m["generator"]["commit"] == synced, "source_check_exact": m["trajectory"]["source_check"]["exact"],
           "sources_pass": src["pass"], "spill_max_px": a["spill_max_px"], "holes_px_total": a["holes_px_total"],
           "align_pass": a["pass"], "twin_iou_min": a["material_twin_iou_min"],
           "edge_offset_px_max": w.get("edge_offset_px_max"), "outline_max_px": w.get("outline_max_px"),
           "pixel_centre_silhouette_px": w["silhouette_max_px"], "irr_psnr_interior_min": w.get("irradiance_psnr_db_interior_min"),
           "white_lambert_pass": w["pass"], "wall_s": m["hardware"]["wall_seconds"],
           "material_s": m["hardware"]["material_seconds"], "twin_s": m["pbr_patch_render"]["seconds"]}
    good = (row["setup"] == "B3" and row["cam_mode"] == row["light_mode"] == "trajectory" and row["commit_ok"]
            and row["source_check_exact"] and row["sources_pass"] and row["align_pass"] and row["white_lambert_pass"])
    ok &= bool(good)
    print(("GATES PASS " if good else "GATES FAIL ") + json.dumps(row))
sys.exit(0 if ok else 1)
PY
echo "debug total wall seconds: $(( $(date +%s) - start )) exit=$status"
exit $status
