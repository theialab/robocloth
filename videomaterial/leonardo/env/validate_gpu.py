#!/usr/bin/env python3
"""exp-015 gate: prove Mitsuba renders on the GPU on a Leonardo A100 node.

Checks, in order:
  1. torch sees CUDA; print device name / capability / driver + runtime CUDA.
  2. the OptiX runtime library the `cuda_ad_rgb` variant needs is loadable.
  3. `mi.set_variant('cuda_ad_rgb')` succeeds and `mi.variants()` lists it.
  4. a trivial sphere renders at 4 spp, while a background thread samples
     `nvidia-smi` utilisation so the log shows the GPU actually did work.

Exits non-zero if the GPU path fails; pass --allow-llvm-fallback to continue on
`llvm_ad_rgb` instead (every number produced that way is CPU and must be
labelled as such).

Writes a JSON summary to --out (default: stdout only).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time


def probe_optix() -> dict:
    """Try to dlopen the driver-provided OptiX library Mitsuba needs."""
    result = {"libnvoptix_loadable": False, "libcuda_loadable": False, "errors": {}}
    for key, soname in (("libcuda_loadable", "libcuda.so.1"),
                        ("libnvoptix_loadable", "libnvoptix.so.1")):
        try:
            ctypes.CDLL(soname)
            result[key] = True
        except OSError as exc:
            result["errors"][soname] = str(exc)
    # Where the loader would find them, for diagnosis.
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout
        result["ldconfig_matches"] = [
            line.strip() for line in out.splitlines()
            if "libnvoptix" in line or "libcuda.so" in line
        ][:8]
    except Exception as exc:  # pragma: no cover - diagnostic only
        result["ldconfig_matches"] = [f"ldconfig failed: {exc}"]
    result["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    return result


class SmiSampler(threading.Thread):
    """Sample `nvidia-smi` utilisation/memory every `interval` seconds."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._halt = threading.Event()

    def run(self) -> None:
        query = ("utilization.gpu,utilization.memory,memory.used,"
                 "temperature.gpu,power.draw,name")
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        self.samples.append({
                            "t": time.time(),
                            "util_gpu_pct": float(parts[0]),
                            "util_mem_pct": float(parts[1]),
                            "mem_used_mib": float(parts[2]),
                            "temp_c": float(parts[3]),
                            "power_w": float(parts[4]) if parts[4] not in ("N/A", "") else None,
                            "name": parts[5],
                        })
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        if not self.samples:
            return {"n_samples": 0}
        util = [s["util_gpu_pct"] for s in self.samples]
        mem = [s["mem_used_mib"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "gpu_name": self.samples[0]["name"],
            "util_gpu_pct_max": max(util),
            "util_gpu_pct_mean": sum(util) / len(util),
            "util_gpu_pct_nonzero_frac": sum(u > 0 for u in util) / len(util),
            "mem_used_mib_max": max(mem),
            "samples": self.samples,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-llvm-fallback", action="store_true")
    args = ap.parse_args()

    report: dict = {
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # --- nvidia-smi snapshot -------------------------------------------------
    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
        print(report["nvidia_smi"], flush=True)
    except Exception as exc:
        report["nvidia_smi"] = f"FAILED: {exc}"
        print(report["nvidia_smi"], flush=True)

    # --- torch ---------------------------------------------------------------
    import torch
    report["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        report["torch"]["device_name"] = torch.cuda.get_device_name(0)
        report["torch"]["capability"] = list(torch.cuda.get_device_capability(0))
        # A real allocation, so "available" is not just a driver query.
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        report["torch"]["matmul_ok"] = bool(torch.isfinite((x @ x).sum()).item())
        del x
        torch.cuda.empty_cache()
    print(json.dumps(report["torch"], indent=2), flush=True)

    # --- OptiX / driver libraries -------------------------------------------
    report["optix_probe"] = probe_optix()
    print(json.dumps(report["optix_probe"], indent=2), flush=True)

    # --- mitsuba variant -----------------------------------------------------
    import mitsuba as mi
    report["mitsuba_version"] = mi.__version__
    report["mitsuba_variants"] = list(mi.variants())
    print(f"mitsuba {mi.__version__} variants={report['mitsuba_variants']}", flush=True)

    variant = "cuda_ad_rgb"
    try:
        mi.set_variant(variant)
        report["variant_set"] = variant
        report["variant_error"] = None
    except Exception as exc:
        report["variant_error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED to set cuda_ad_rgb: {report['variant_error']}", flush=True)
        if not args.allow_llvm_fallback:
            report["status"] = "FAIL"
            _dump(report, args.out)
            return 2
        variant = "llvm_ad_rgb"
        mi.set_variant(variant)
        report["variant_set"] = variant
        print("FALLBACK: rendering on llvm_ad_rgb (CPU). "
              "All timings from this variant are CPU numbers.", flush=True)

    import drjit as dr
    report["drjit_version"] = dr.__version__

    # --- trivial sphere render ----------------------------------------------
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "fov": 35.0,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 2.0, 4.0], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": args.res, "height": args.res},
            "sampler": {"type": "independent"},
        },
        "sphere": {
            "type": "sphere",
            "radius": 1.0,
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": [0.4, 0.5, 0.8]}},
        },
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().translate([0, -1, 0])
                        .rotate([1, 0, 0], -90).scale(6),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.5}},
        },
        "light": {"type": "point", "position": [3.0, 3.0, 3.0],
                  "intensity": {"type": "rgb", "value": [60, 60, 60]}},
    })

    sampler = SmiSampler()
    sampler.start()
    # Warm-up (kernel compilation / OptiX pipeline build) is timed separately.
    t0 = time.time()
    img = mi.render(scene, spp=args.spp, seed=0)
    dr.eval(img)
    dr.sync_thread()
    t_first = time.time() - t0

    t0 = time.time()
    for i in range(5):
        img = mi.render(scene, spp=args.spp, seed=i + 1)
        dr.eval(img)
    dr.sync_thread()
    t_warm = (time.time() - t0) / 5
    time.sleep(0.5)
    sampler.stop()

    import numpy as np
    arr = np.asarray(img, dtype=np.float32)
    report["render"] = {
        "variant": variant,
        "resolution": [args.res, args.res],
        "spp": args.spp,
        "first_render_seconds_incl_jit": t_first,
        "warm_render_seconds_mean": t_warm,
        "finite": bool(np.isfinite(arr).all()),
        "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()),
    }
    report["gpu_utilisation_during_render"] = sampler.summary()
    # Keep the JSON small: drop the raw sample list from the printed summary.
    printed = dict(report["gpu_utilisation_during_render"])
    printed.pop("samples", None)
    print(json.dumps({"render": report["render"], "gpu": printed}, indent=2), flush=True)

    ok = (report["render"]["finite"] and report["render"]["max"] > 0.0
          and variant == "cuda_ad_rgb")
    if variant == "cuda_ad_rgb" and printed.get("util_gpu_pct_max", 0) <= 0:
        print("WARN: nvidia-smi never sampled non-zero utilisation "
              "(render may be too short to catch).", flush=True)
    report["status"] = "PASS" if ok else ("PASS_CPU_FALLBACK" if args.allow_llvm_fallback else "FAIL")
    _dump(report, args.out)
    print(f"STATUS={report['status']}", flush=True)
    return 0 if report["status"].startswith("PASS") else 3


def _dump(report: dict, out: str | None) -> None:
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
