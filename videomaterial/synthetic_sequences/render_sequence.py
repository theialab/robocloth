#!/usr/bin/env python3
"""exp-015 one-load sequence renderer with schema-v2 metadata (single sequence, single process).

  python render_sequence.py --setup B1 --cls spline --seed 11 --mode ball --spp 16 \
      --output /media/raid/cloth/VideoMaterial/data/Robocloth_synthetic_sequence/B1_Fixed_camera/trajectory_visualizations/spline_01

Per frame the camera ``to_world`` and the light position are updated through ``mi.traverse`` (the
scene, and for ``--mode material`` the 1.16 GB checkpoint, are loaded once). Output chain is exactly
exp-005: linear float32 EXR, PNG = global Reinhard x/(1+x) -> sRGB (by mi.util.write_bitmap) -> 8-bit.

Everything but the argument parsing lives in ``seqlib.py``; ``render_manifest.py`` renders whole
dataset shards through the same functions, so the two drivers produce identical frames.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_sequences import seqlib as L  # noqa: E402
from synthetic_sequences import trajectories as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=list(L.SETUPS), required=True)
    ap.add_argument("--cls", choices=list(T.ALL_CLASSES), required=True)
    ap.add_argument("--seed", type=int, required=True, help="trajectory seed (stored in metadata, never in names)")
    ap.add_argument("--mode", choices=list(L.MODES), required=True)
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
    ap.add_argument("--no-exr", action="store_true", help="PNG only (the dataset keeps EXR for a subset)")
    ap.add_argument("--scene-src", type=Path, default=Path("/media/raid/cloth/output/VideoMaterial/robocloth_synthetic_v1/2026-08-21-mat314-wan832x480-81f-64spp"))
    ap.add_argument("--checkpoint-root", default="/media/raid/cloth/output/BRDF/Stage-2-Finals/Ours")
    ap.add_argument("--rendering-root", type=Path, default=Path("/home/zla247/projects/robocloth/rendering"))
    args = ap.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "render.log").open("w", buffering=1)
    started = time.time()

    import drjit as dr, mitsuba as mi, torch
    mi.set_variant(args.variant)
    render_fn = L.load_render_backend(args.rendering_root)

    cfg = L.RenderConfig(spp=args.spp, batch_spp=args.batch_spp, max_depth=args.max_depth, fov=args.fov,
                         width=args.width, height=args.height, fps=args.fps, frames=args.frames,
                         intensity=args.intensity, render_seed=args.render_seed, variant=args.variant,
                         ball_roughness=args.ball_roughness, ground_roughness=args.ground_roughness,
                         scene_src=args.scene_src, checkpoint_root=args.checkpoint_root,
                         rendering_root=args.rendering_root)
    poses = L.build_poses(args.setup, args.cls, args.seed, args.frames, profile=args.profile)
    handle = L.load_scene(mi, args.mode, poses.cam_pos[0], poses.light_pos[0], cfg, log=log)
    print(f"traverse keys: camera={handle.cam_key} light={handle.light_key}", file=log)

    meta = L.build_metadata(out, poses, args.mode, cfg, handle.material_meta, mi, dr, torch,
                            started, args.split, str(Path(__file__).resolve()))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    indices = list(range(args.frames)) if args.frame_indices is None else [int(x) for x in args.frame_indices.split(",")]
    exr_dir = None if args.no_exr else out / "frames_exr"

    def progress(rec):
        i = rec["index"]
        lth, lph = T.vec_to_ang(poses.light_pos[i] / np.linalg.norm(poses.light_pos[i]))
        cth, cph = T.vec_to_ang(poses.cam_pos[i] / np.linalg.norm(poses.cam_pos[i]))
        print(f"frame {i:03d} cam theta={cth:6.2f} phi={cph:6.2f} light theta={lth:6.2f} phi={lph:6.2f} "
              f"{rec['seconds']:6.2f}s max={rec['stats']['linear_max']:.4f}", flush=True)

    recs = L.render_poses(mi, handle, indices, poses.cam_pos, poses.light_pos, cfg, render_fn,
                          png_dir=out / "frames_png", exr_dir=exr_dir, log=log, on_frame=progress)

    with (out / "frames.jsonl").open("w") as fj:
        for rec in recs:
            fj.write(json.dumps(L.frame_record(rec, poses, cfg, out), sort_keys=True) + "\n")

    secs = [r["seconds"] for r in recs]
    meta["frames"]["rendered_indices"] = indices
    meta["frames"]["exr_dir"] = None if args.no_exr else "frames_exr"
    meta["hardware"].update({"wall_seconds": time.time() - started, "seconds_per_frame_mean": float(np.mean(secs)),
                             "seconds_per_frame": secs,
                             "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
    meta["status"] = "complete"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    L.finalize(out)
    print(f"done: {len(indices)} frames, {np.mean(secs):.2f} s/frame, wall {time.time()-started:.1f}s -> {out}")
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
