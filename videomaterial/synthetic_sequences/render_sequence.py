#!/usr/bin/env python3
"""exp-015 one-load sequence renderer with schema-v2 metadata.

  python render_sequence.py --setup B1 --cls spline --seed 11 --mode ball --spp 16 \
      --output /media/raid/cloth/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/trajectory_visualizations/spline_01

Per frame the camera ``to_world`` and the light position are updated through ``mi.traverse`` (the
scene, and for ``--mode material`` the 1.16 GB checkpoint, are loaded once). Output chain is exactly
exp-005: linear float32 EXR, PNG = global Reinhard x/(1+x) → sRGB (by mi.util.write_bitmap) → 8-bit.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_sequences import trajectories as T  # noqa: E402
from synthetic_sequences import scene as S  # noqa: E402

SETUPS = {"B1": ("B1_Fixed_camera", T.TrajectorySpec.b1_light), "B2": ("B2_Fixed_light", T.TrajectorySpec.b2_camera)}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info(repo):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"repo": str(repo), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def write_bitmap_pair(mi, image, png_path, exr_path):
    arr = np.asarray(image, dtype=np.float32)
    finite = bool(np.isfinite(arr).all())
    if not finite:
        raise RuntimeError(f"non-finite radiance in {png_path.name}")
    png = np.nan_to_num(arr / (1.0 + arr), nan=0.0, posinf=0.0, neginf=0.0)
    mi.util.write_bitmap(str(png_path), mi.Bitmap(png))
    mi.util.write_bitmap(str(exr_path), mi.Bitmap(arr))
    return {"linear_min": float(arr.min()), "linear_max": float(arr.max()), "linear_mean": float(arr.mean()), "finite": finite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=list(SETUPS), required=True)
    ap.add_argument("--cls", choices=list(T.ALL_CLASSES), required=True)
    ap.add_argument("--seed", type=int, required=True, help="trajectory seed (stored in metadata, never in names)")
    ap.add_argument("--mode", choices=["material", "ball", "white_lambert"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--batch-spp", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--fov", type=float, default=35.0)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--frame-indices", default=None, help="comma list; renders a subset of the 81-pose schedule")
    ap.add_argument("--render-seed", type=int, default=20260922)
    ap.add_argument("--intensity", type=float, default=20.0)
    ap.add_argument("--ball-roughness", type=float, default=0.3, help="ball mode only")
    ap.add_argument("--ground-roughness", type=float, default=0.15, help="ball mode only: roughness of the glossy sample patch")
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--profile", default=None, choices=[None, *T.SPEED_PROFILES])
    ap.add_argument("--split", default="visualization")
    ap.add_argument("--scene-src", type=Path, default=Path("/media/raid/cloth/output/VideoMaterial/robocloth_synthetic_v1/2026-08-21-mat314-wan832x480-81f-64spp"))
    ap.add_argument("--checkpoint-root", default="/media/raid/cloth/output/BRDF/Stage-2-Finals/Ours")
    ap.add_argument("--rendering-root", type=Path, default=Path("/home/zla247/projects/robocloth/rendering"))
    args = ap.parse_args()

    out = args.output; out.mkdir(parents=True, exist_ok=True)
    (out / "frames_png").mkdir(exist_ok=True); (out / "frames_exr").mkdir(exist_ok=True)
    log = (out / "render.log").open("w", buffering=1)
    started = time.time()

    import drjit as dr, mitsuba as mi, torch
    mi.set_variant(args.variant)
    sys.path.insert(0, str(args.rendering_root))
    from render import render_with_batches  # noqa: E402  (RoboCloth rendering/render.py)

    setup_dir, spec_fn = SETUPS[args.setup]
    spec = spec_fn(frames=args.frames)
    traj = T.sample_trajectory(args.cls, args.seed, spec, profile=args.profile)
    moving_pos = traj.positions                                    # (F,3)
    fixed_pos = spec.fixed_direction * spec.radius
    if spec.moving_element == "light":
        light_pos = moving_pos; cam_pos = np.repeat(fixed_pos[None], args.frames, 0)
    else:
        cam_pos = moving_pos; light_pos = np.repeat(fixed_pos[None], args.frames, 0)

    # ---- scene (loaded once)
    material_meta = None; tmp_scene = None
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        if args.mode == "material":
            scene, spec_json, tmp_scene = S.load_material_scene(args.scene_src, args.checkpoint_root, args.rendering_root)
            ckpt = Path(args.checkpoint_root) / "314" / "Ours_epoch80.ckpt"
            material_meta = {"kind": "robocloth_neural", "id": 314, "checkpoint": {"path": str(ckpt), "sha256": sha256(ckpt) if ckpt.exists() else None},
                             "materials_json": spec_json, "bsdf_params": None}
        elif args.mode == "ball":
            scene = mi.load_dict(S.ball_scene_dict(cam_pos[0], light_pos[0], args.fov, args.width, args.height, args.intensity, ball_roughness=args.ball_roughness, ground_roughness=args.ground_roughness, max_depth=args.max_depth))
            material_meta = {"kind": "mitsuba_principled", "id": None, "checkpoint": None,
                             "bsdf_params": {"ball": {"base_color": [0.6, 0.6, 0.6], "roughness": args.ball_roughness, "specular": 0.5, "radius": S.BALL_RADIUS, "center": S.BALL_CENTER},
                                             "sample_patch": {"base_color": [0.5, 0.5, 0.5], "roughness": args.ground_roughness, "specular": 0.6, "half_extent": S.SAMPLE_HALF_EXTENT, "y": 0.002},
                                             "ground": {"diffuse_reflectance": [0.28, 0.28, 0.28], "y": 0.0}}}
        else:
            scene = mi.load_dict(S.white_lambert_scene_dict(cam_pos[0], light_pos[0], args.fov, args.width, args.height, args.intensity, max_depth=args.max_depth))
            material_meta = {"kind": "mitsuba_diffuse_white", "id": None, "checkpoint": None, "bsdf_params": {"cloth": {"reflectance": [1.0, 1.0, 1.0]}}}
    params = mi.traverse(scene)
    keys = list(params.keys())
    # the sensor is whichever object also exposes x_fov (scene.xml sensors have no id -> "object_<n>.to_world")
    sensor_prefixes = {k[: -len(".x_fov")] for k in keys if k.endswith(".x_fov")}
    cam_key = next((f"{p}.to_world" for p in sensor_prefixes if f"{p}.to_world" in keys), None)
    light_key = "light.position" if "light.position" in keys else None
    marker_key = None
    if cam_key is None or light_key is None:
        raise KeyError(f"need sensor to_world and light.position in traverse keys; got {keys[:40]}")
    print(f"traverse keys: camera={cam_key} light={light_key} marker={marker_key}", file=log)

    # ---- metadata (static part first, status=rendering)
    K = S.intrinsics(args.width, args.height, args.fov, "y")
    meta = {
        "schema_version": 2, "status": "rendering",
        "sequence_id": str(out.relative_to(out.parents[2])) if len(out.parents) > 2 else out.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "generator": {**git_info(HERE.parents[1]), "script": str(Path(__file__).resolve()), "python": platform.python_version(),
                      "mitsuba": mi.__version__, "drjit": dr.__version__, "torch": torch.__version__, "variant": args.variant},
        "seeds": {"trajectory": args.seed, "render_base": args.render_seed},
        "setup": args.setup, "mode": args.mode,
        "sample": {"shape": "rectangle", "half_extent_xz": [S.SAMPLE_HALF_EXTENT, S.SAMPLE_HALF_EXTENT],
                   "transform_to_world": "rotate(-90 deg about +X) then scale(0.75): the unit rectangle in the XY plane becomes the XZ plane with +Y normal",
                   "uv_convention": "u -> +X, v -> +Z (Mitsuba rectangle UVs after the -90 deg X rotation), origin at (-0.75, 0, -0.75)",
                   "uv_tiling": 1.0, "present_in_mode": args.mode != "ball"},
        "frame_convention": {"world": "sample frame: origin at sample centre, +Y = normal, +X = u, +Z = v",
                             "theta": "polar angle from +Y, degrees", "phi": "azimuth from +X towards +Z, degrees",
                             "camera_to_world": "Mitsuba perspective sensor: camera-space +Z is the viewing direction, +Y is up, +X points to the image LEFT (Mitsuba mirrors X); use check_metadata.py to regenerate rays",
                             "pixel_center": 0.5},
        "material": material_meta,
        "camera": {"mode": "trajectory" if spec.moving_element == "camera" else "fixed", "projection": "perspective",
                   "fov_deg": args.fov, "fov_axis": "y", "resolution": [args.width, args.height], "look_at": [0.0, 0.0, 0.0],
                   "up_hint": [0.0, 1.0, 0.0], "radius": spec.radius, "intrinsics_K": K,
                   "fixed_direction_deg": None if spec.moving_element == "camera" else {"theta": spec.fixed_theta_deg, "phi": spec.fixed_phi_deg}},
        "light": {"type": "point", "mode": "trajectory" if spec.moving_element == "light" else "fixed", "radius_from_origin": spec.radius,
                  "intensity_rgb": [args.intensity] * 3, "source_radius": 0.0,
                  "units": "Mitsuba point-light intensity I (W/sr); irradiance at distance r = I cos(theta) / r^2",
                  "fixed_direction_deg": None if spec.moving_element == "light" else {"theta": spec.fixed_theta_deg, "phi": spec.fixed_phi_deg},
                  "visual_marker": {"present": False, "note": "light position is drawn as an overlay by make_visualization.py"}},
        "trajectory": {"split": args.split, **traj.to_metadata()},
        "render": {"spp": args.spp, "batch_spp": args.batch_spp, "max_depth": args.max_depth, "integrator": "path", "sampler": "independent",
                   "denoiser": None, "exposure": 1.0, "tonemap": "global Reinhard x/(1+x) then sRGB (mi.util.write_bitmap), 8-bit PNG",
                   "exr": "linear RGB float32", "per_frame_seed": "render_base + frame index (random.seed before the batches)"},
        "frames": {"count": args.frames, "fps": args.fps, "jsonl": "frames.jsonl", "png_dir": "frames_png", "exr_dir": "frames_exr",
                   "rendered_indices": None},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": platform.node()},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    indices = list(range(args.frames)) if args.frame_indices is None else [int(x) for x in args.frame_indices.split(",")]
    records = []; secs = []
    with (out / "frames.jsonl").open("w", buffering=1) as fj:
        for idx in indices:
            cp, lp = cam_pos[idx], light_pos[idx]
            c2w = S.look_at_matrix(cp)
            params[cam_key] = mi.Transform4f(c2w)
            if light_key is not None:
                params[light_key] = mi.Point3f([float(x) for x in lp])
            if marker_key is not None:
                params[marker_key] = mi.Transform4f(mi.ScalarTransform4f().translate([float(x) for x in lp]).scale(args.light_radius))
            params.update()
            seed = args.render_seed + idx
            random.seed(seed)
            t0 = time.time()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                img = render_with_batches(scene, args.spp, args.batch_spp)
            png = out / "frames_png" / f"frame_{idx:03d}.png"; exr = out / "frames_exr" / f"frame_{idx:03d}.exr"
            stats = write_bitmap_pair(mi, img, png, exr)
            dt = time.time() - t0; secs.append(dt)
            cth, cph = T.vec_to_ang(cp / np.linalg.norm(cp)); lth, lph = T.vec_to_ang(lp / np.linalg.norm(lp))
            rec = {"index": idx, "t": idx / max(args.frames - 1, 1), "seed": seed, "seconds": dt,
                   "camera": {"position": cp.tolist(), "c2w": S.transform_to_list(c2w), "theta_deg": float(cth), "phi_deg": float(cph)},
                   "light": {"position": lp.tolist(), "theta_deg": float(lth), "phi_deg": float(lph), "distance": float(np.linalg.norm(lp))},
                   "files": {"png": str(png.relative_to(out)), "exr": str(exr.relative_to(out))}, "stats": stats}
            records.append(rec); fj.write(json.dumps(rec, sort_keys=True) + "\n")
            print(f"frame {idx:03d} cam θ={cth:6.2f} φ={cph:6.2f} light θ={lth:6.2f} φ={lph:6.2f} {dt:6.2f}s max={stats['linear_max']:.4f}", flush=True)

    meta["frames"]["rendered_indices"] = indices
    meta["hardware"].update({"wall_seconds": time.time() - started, "seconds_per_frame_mean": float(np.mean(secs)),
                             "seconds_per_frame": secs, "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
    meta["status"] = "complete"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    # manifest
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.sha256", "RENDER_COMPLETE", "render.log"):
            lines.append(f"{sha256(p)}  {p.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    (out / "RENDER_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"done: {len(indices)} frames, {np.mean(secs):.2f} s/frame, wall {time.time()-started:.1f}s -> {out}")
    if tmp_scene is not None:
        import shutil; shutil.rmtree(tmp_scene, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
