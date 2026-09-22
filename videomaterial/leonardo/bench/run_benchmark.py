#!/usr/bin/env python3
"""exp-015 benchmark driver: time the exp-005 RoboCloth sequence on one A100.

This is the exp-005 production driver
(`/media/raid/cloth/output/VideoMaterial/robocloth_synthetic_v1/
  2026-08-21-mat314-wan832x480-81f-64spp/render_sequence.py`)
with three changes, and nothing else — the rendered pixels must stay comparable:

  * paths are CLI arguments (RoboCloth repo root, checkpoint root, scene source)
    instead of workstation constants;
  * `--frame-indices` renders a subset of the SAME 81-pose schedule, so a
    3-frame debug run uses the exact poses of frames 0 / 40 / 80;
  * per-frame timing is split into checkpoint-load / first-frame (JIT+OptiX
    pipeline build) / steady-state, and `nvidia-smi` is sampled throughout, so
    the reported seconds-per-frame is the steady-state number a long sequence
    would actually pay.

The pose schedule, scene, spp batching, tone map and EXR output are byte-for-byte
the exp-005 logic.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# exp-005 pose schedule (verbatim)
# ---------------------------------------------------------------------------
def light_pose(index: int, frames: int, radius: float) -> dict:
    t = index / max(frames - 1, 1)
    theta_deg = 80.0 + t * (5.0 - 80.0)
    phi_deg_unwrapped = t * 2.0 * 360.0
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg_unwrapped)
    position = [
        radius * math.sin(theta) * math.cos(phi),
        radius * math.cos(theta),
        radius * math.sin(theta) * math.sin(phi),
    ]
    return {
        "index": index,
        "t": t,
        "theta_deg": theta_deg,
        "phi_deg": phi_deg_unwrapped % 360.0,
        "phi_deg_unwrapped": phi_deg_unwrapped,
        "position": position,
    }


class SmiSampler(threading.Thread):
    """Background `nvidia-smi` sampler; proves the GPU is the one working."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.util: list[float] = []
        self.mem: list[float] = []
        self.gpu_name: str | None = None
        self._halt = threading.Event()

    def run(self) -> None:
        q = "utilization.gpu,memory.used,name"
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={q}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                for line in out.splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 3:
                        self.util.append(float(p[0]))
                        self.mem.append(float(p[1]))
                        self.gpu_name = p[2]
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        if not self.util:
            return {"n_samples": 0}
        return {
            "n_samples": len(self.util),
            "gpu_name": self.gpu_name,
            "util_gpu_pct_mean": sum(self.util) / len(self.util),
            "util_gpu_pct_max": max(self.util),
            "util_gpu_pct_p10": float(np.percentile(self.util, 10)),
            "mem_used_mib_max": max(self.mem),
        }


def write_bitmap_pair(mi, image, png_path: Path, exr_path: Path) -> dict:
    """exp-005: Reinhard-tonemapped PNG + linear EXR, refuse non-finite."""
    arr = np.asarray(image, dtype=np.float32)
    finite = bool(np.isfinite(arr).all())
    if not finite:
        raise RuntimeError(f"non-finite radiance in {png_path.name}")
    png = arr / (1.0 + arr)
    png = np.nan_to_num(png, nan=0.0, posinf=0.0, neginf=0.0)
    mi.util.write_bitmap(str(png_path), mi.Bitmap(png))
    mi.util.write_bitmap(str(exr_path), mi.Bitmap(arr))
    return {
        "linear_min": float(arr.min()),
        "linear_max": float(arr.max()),
        "linear_mean": float(arr.mean()),
        "finite": finite,
    }


def build_scene_dir(dest: Path, scene_src: Path, checkpoint_root: str) -> Path:
    """Copy the exp-005 scene.xml and rewrite materials.json's checkpoint_root.

    Nothing else in materials.json changes: same material id, same uv_tiling,
    same two_sided / uv_inset flags, so the BSDF is the exp-005 BSDF.
    """
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene_src / "scene.xml", dest / "scene.xml")
    spec = json.loads((scene_src / "materials.json").read_text())
    spec["checkpoint_root"] = checkpoint_root
    (dest / "materials.json").write_text(json.dumps(spec, indent=2) + "\n")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendering-root", type=Path, required=True,
                    help="<robocloth repo>/rendering")
    ap.add_argument("--scene-src", type=Path, required=True,
                    help="dir holding the exp-005 scene.xml + materials.json")
    ap.add_argument("--checkpoint-root", required=True,
                    help="dir whose <material-id>/Ours_epoch*.ckpt is the material")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=81,
                    help="length of the pose schedule (NOT how many are rendered)")
    ap.add_argument("--frame-indices", default=None,
                    help="comma list or A:B[:S] slice of schedule indices to render")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--batch-spp", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--light-radius", type=float, default=3.0)
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--no-exr", action="store_true",
                    help="skip EXR writing (timing-only runs)")
    ap.add_argument("--tag", default="", help="free-form label stored in timing.json")
    args = ap.parse_args()

    if args.frame_indices is None:
        indices = list(range(args.frames))
    elif ":" in args.frame_indices:
        parts = [int(p) if p else None for p in args.frame_indices.split(":")]
        indices = list(range(args.frames))[slice(*parts)]
    else:
        indices = [int(x) for x in args.frame_indices.split(",") if x != ""]
    for i in indices:
        if not 0 <= i < args.frames:
            raise ValueError(f"frame index {i} outside schedule of {args.frames}")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    png_dir = args.output / "frames_png"
    exr_dir = args.output / "frames_exr"
    png_dir.mkdir(exist_ok=True)
    exr_dir.mkdir(exist_ok=True)
    scene_dir = build_scene_dir(args.output / "scene", args.scene_src, args.checkpoint_root)
    shutil.copy2(Path(__file__), args.output / "run_benchmark.py")

    sys.path.insert(0, str(args.rendering_root))
    import torch
    import drjit as dr
    import mitsuba as mi
    from render import load_materials, render_with_batches  # noqa: E402

    log_path = args.output / "render.log"
    started = time.time()

    mi.set_variant(args.variant)
    from brdf_plugin.cook_torrancebrdf import CookTorranceBRDF  # noqa: E402
    from brdf_plugin.mlp import MLPBRDF  # noqa: E402
    from brdf_plugin.utils.scene_loader import load_scene_with_bsdf_overrides  # noqa: E402

    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))

    smi = SmiSampler()
    smi.start()

    load_started = time.time()
    with log_path.open("w", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        overrides, radiance = load_materials(str(scene_dir), default_two_sided=False)
        scene = load_scene_with_bsdf_overrides(
            str(scene_dir / "scene.xml"), overrides, radiance=radiance)
    load_seconds = time.time() - load_started

    params = mi.traverse(scene)
    if "light.position" not in params:
        raise KeyError(f"light.position missing; available keys: {list(params.keys())}")

    cuda_ok = torch.cuda.is_available()
    static = {
        "schema_version": 1,
        "experiment": "exp-015 leonardo mitsuba render benchmark",
        "tag": args.tag,
        "status": "rendering",
        "source_driver": "exp-005 render_sequence.py (pose schedule verbatim)",
        "rendering_root": str(args.rendering_root),
        "scene_src": str(args.scene_src),
        "checkpoint_root": args.checkpoint_root,
        "material_id": 314,
        "variant": args.variant,
        "resolution": [832, 480],
        "schedule_frames": args.frames,
        "rendered_indices": indices,
        "fps": 15,
        "spp": args.spp,
        "batch_spp": args.batch_spp,
        "max_depth": 4,
        "seed_base": args.seed,
        "checkpoint_load_seconds": load_seconds,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "mitsuba_version": mi.__version__,
        "drjit_version": dr.__version__,
        "torch_cuda_device_count": torch.cuda.device_count() if cuda_ok else 0,
        "torch_cuda_device_name": torch.cuda.get_device_name(0) if cuda_ok else None,
        "started_unix": started,
    }
    (args.output / "metadata.json").write_text(json.dumps(static, indent=2) + "\n")

    records = []
    frame_jsonl_path = args.output / "frames.jsonl"
    with frame_jsonl_path.open("w", buffering=1) as frame_jsonl, \
            log_path.open("a", buffering=1) as log:
        for n, index in enumerate(indices):
            pose = light_pose(index, args.frames, args.light_radius)
            params["light.position"] = mi.Point3f(pose["position"])
            params.update()
            seed = args.seed + index
            random.seed(seed)
            frame_started = time.time()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                image = render_with_batches(scene, args.spp, args.batch_spp)
            render_seconds = time.time() - frame_started
            png_path = png_dir / f"frame_{index:03d}.png"
            exr_path = exr_dir / f"frame_{index:03d}.exr"
            if args.no_exr:
                arr = np.asarray(image, dtype=np.float32)
                stats = {"linear_min": float(arr.min()), "linear_max": float(arr.max()),
                         "linear_mean": float(arr.mean()),
                         "finite": bool(np.isfinite(arr).all())}
                if not stats["finite"]:
                    raise RuntimeError(f"non-finite radiance in frame {index}")
            else:
                stats = write_bitmap_pair(mi, image, png_path, exr_path)
            elapsed = time.time() - frame_started
            record = {
                **pose,
                "order": n,
                "seed": seed,
                "render_seconds": render_seconds,
                "seconds": elapsed,
                "png": str(png_path.relative_to(args.output)),
                "exr": None if args.no_exr else str(exr_path.relative_to(args.output)),
                **stats,
            }
            records.append(record)
            frame_jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"frame {n + 1:02d}/{len(indices)} (schedule idx {index:03d}): "
                  f"theta={pose['theta_deg']:.3f} phi={pose['phi_deg']:.3f} "
                  f"render={render_seconds:.2f}s total={elapsed:.2f}s "
                  f"max={stats['linear_max']:.5f}", flush=True)
            del image
            if cuda_ok:
                torch.cuda.empty_cache()
            dr.flush_malloc_cache()
            gc.collect()

    finished = time.time()
    smi.stop()

    render_secs = [r["render_seconds"] for r in records]
    total_secs = [r["seconds"] for r in records]
    # Frame 1 pays DrJit/OptiX kernel compilation; steady state excludes it.
    steady_render = render_secs[1:] if len(render_secs) > 1 else render_secs
    steady_total = total_secs[1:] if len(total_secs) > 1 else total_secs

    timing = {
        "tag": args.tag,
        "variant": args.variant,
        "spp": args.spp,
        "batch_spp": args.batch_spp,
        "resolution": [832, 480],
        "schedule_frames": args.frames,
        "n_rendered": len(records),
        "rendered_indices": indices,
        "gpu_name": static["torch_cuda_device_name"],
        "checkpoint_load_seconds": load_seconds,
        "first_frame_render_seconds": render_secs[0],
        "render_seconds_per_frame_steady_mean": float(np.mean(steady_render)),
        "render_seconds_per_frame_steady_median": float(np.median(steady_render)),
        "render_seconds_per_frame_steady_min": float(np.min(steady_render)),
        "render_seconds_per_frame_steady_max": float(np.max(steady_render)),
        "total_seconds_per_frame_steady_mean": float(np.mean(steady_total)),
        "wall_seconds": finished - started,
        "wall_seconds_render_loop": sum(total_secs),
        # Projection to the full 81-frame sequence: model load once, first frame
        # pays JIT, remaining frames at steady-state total (render + file I/O).
        "projected_81frame_seconds": (
            load_seconds + total_secs[0] + 80.0 * float(np.mean(steady_total))
            if len(records) > 1 else None),
        "gpu_sampling": smi.summary(),
        "frames": records,
    }
    if timing["projected_81frame_seconds"] is not None:
        timing["projected_81frame_a100_gpu_hours"] = \
            timing["projected_81frame_seconds"] / 3600.0
    (args.output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")

    static.update({
        "status": "complete",
        "finished_unix": finished,
        "wall_seconds": finished - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda_ok else None,
    })
    (args.output / "metadata.json").write_text(json.dumps(static, indent=2) + "\n")
    (args.output / "RENDER_COMPLETE").write_text("complete\n")

    print(f"complete: {len(records)} frames in {finished - started:.2f}s "
          f"(load {load_seconds:.1f}s, first frame {render_secs[0]:.2f}s, "
          f"steady {np.mean(steady_render):.3f}s/frame render / "
          f"{np.mean(steady_total):.3f}s/frame total)", flush=True)
    if timing["projected_81frame_seconds"]:
        print(f"projected 81 frames @ spp {args.spp}: "
              f"{timing['projected_81frame_seconds']:.1f}s = "
              f"{timing['projected_81frame_a100_gpu_hours']:.4f} A100 GPU-hours",
              flush=True)
    print(f"gpu: {json.dumps(smi.summary())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
