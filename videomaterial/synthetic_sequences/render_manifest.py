#!/usr/bin/env python3
"""Render one shard of the B1 dataset: material 314 + the PBR ball twin, one scene load for all.

    python render_manifest.py --manifest .../manifest.json --shard 3 --nshards 16 \
        --root $WORK/.../Robocloth_synthetic_sequence/B1_Fixed_camera

The expensive part of a sequence is loading the 1.16 GB neural-material checkpoint (~3-6 s) and
JIT-compiling the kernels (~1 s). This driver pays that once per Slurm task and then walks its whole
shard, which is what turns 1,000 sequences from "1,000 loads" into "16 loads". The per-sequence log
records the gap between consecutive sequences precisely so a regression to per-sequence loading
would be visible immediately.

Resumable: a sequence whose directory already has RENDER_COMPLETE is skipped, so a failed or
time-limited shard can simply be resubmitted.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_sequences import seqlib as L  # noqa: E402


def shard_of(sequences, shard, nshards):
    """Round-robin so every shard gets the same class mix and the same train/test ratio."""
    return sequences[shard::nshards]


def render_one(mi, dr, torch, seq, handles, cfgs, render_fn, root: Path, log, do_ball=True):
    out = root / seq["dir"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    poses = L.build_poses("B1", seq["class"], seq["seed"], cfgs["material"].frames)
    indices = list(range(cfgs["material"].frames))

    meta = L.build_metadata(out, poses, "material", cfgs["material"], handles["material"].material_meta,
                            mi, dr, torch, started, seq["split"], str(Path(__file__).resolve()))
    meta["dataset"] = {"manifest_index": seq["index"], "sequence_id_in_manifest": seq["id"],
                       "split": seq["split"], "keep_exr": bool(seq["keep_exr"])}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    exr_dir = (out / "frames_exr") if seq["keep_exr"] else None
    t_mat = time.time()
    mat = L.render_poses(mi, handles["material"], indices, poses.cam_pos, poses.light_pos,
                         cfgs["material"], render_fn, png_dir=out / "frames_png", exr_dir=exr_dir, log=log)
    t_mat = time.time() - t_mat

    ball, t_ball = [], 0.0
    if do_ball:
        t0 = time.time()
        ball = L.render_poses(mi, handles["ball"], indices, poses.cam_pos, poses.light_pos,
                              cfgs["ball"], render_fn, png_dir=out / "ball_png", exr_dir=None, log=log)
        t_ball = time.time() - t0
    ball_by_idx = {r["index"]: r for r in ball}

    with (out / "frames.jsonl").open("w") as fj:
        for rec in mat:
            b = ball_by_idx.get(rec["index"])
            extra = {"ball_png": None if b is None else str(b["png"].relative_to(out))}
            line = L.frame_record(rec, poses, cfgs["material"], out, extra_files=extra)
            if b is not None:
                line["ball_stats"] = b["stats"]
                line["ball_seconds"] = b["seconds"]
            fj.write(json.dumps(line, sort_keys=True) + "\n")

    secs = [r["seconds"] for r in mat]
    meta["frames"]["rendered_indices"] = indices
    meta["frames"]["exr_dir"] = "frames_exr" if seq["keep_exr"] else None
    meta["ball_render"] = {"rendered": do_ball, "mode": "ball", "spp": cfgs["ball"].spp,
                           "batch_spp": cfgs["ball"].batch_spp, "png_dir": "ball_png", "exr_dir": None,
                           "bsdf_params": handles["ball"].material_meta["bsdf_params"] if do_ball else None,
                           "note": "same 81 camera/light poses as the material render, cheap trivial-case twin",
                           "seconds": t_ball,
                           "seconds_per_frame_mean": float(np.mean([r["seconds"] for r in ball])) if ball else None}
    meta["hardware"].update({"wall_seconds": time.time() - started, "material_seconds": t_mat,
                             "seconds_per_frame_mean": float(np.mean(secs)), "seconds_per_frame": secs,
                             "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
    meta["status"] = "complete"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    L.finalize(out)
    return {"wall": time.time() - started, "material": t_mat, "ball": t_ball,
            "s_per_frame": float(np.mean(secs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--root", type=Path, required=True, help="the B1_Fixed_camera directory")
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="render at most this many sequences (debug runs)")
    ap.add_argument("--variant", default="cuda_ad_rgb")
    ap.add_argument("--render-seed", type=int, default=20260922)
    ap.add_argument("--scene-src", type=Path, required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--rendering-root", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    todo = shard_of(man["sequences"], args.shard, args.nshards)
    if args.limit:
        todo = todo[: args.limit]

    logdir = args.root / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = (logdir / f"shard_{args.shard:02d}.log").open("a", buffering=1)

    def say(msg):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[{stamp}] {msg}", file=log, flush=True)
        print(f"[{stamp}] {msg}", flush=True)

    say(f"shard {args.shard}/{args.nshards}: {len(todo)} sequences, root={args.root}, ball={not args.no_ball}")

    import drjit as dr, mitsuba as mi, torch  # noqa: E402
    mi.set_variant(args.variant)
    render_fn = L.load_render_backend(args.rendering_root)

    R = man["render"]
    common = dict(max_depth=R["max_depth"], intensity=R["intensity"], frames=man["frames"], fps=man["fps"],
                  width=man["resolution"][0], height=man["resolution"][1], render_seed=args.render_seed,
                  variant=args.variant, scene_src=args.scene_src, checkpoint_root=args.checkpoint_root,
                  rendering_root=args.rendering_root)
    cfgs = {"material": L.RenderConfig(spp=R["material"]["spp"], batch_spp=R["material"]["batch_spp"], **common),
            "ball": L.RenderConfig(spp=R["ball"]["spp"], batch_spp=R["ball"]["batch_spp"], **common)}

    t0 = time.time()
    poses0 = L.build_poses("B1", todo[0]["class"], todo[0]["seed"], man["frames"])
    handles = {"material": L.load_scene(mi, "material", poses0.cam_pos[0], poses0.light_pos[0], cfgs["material"], log=log)}
    say(f"material scene loaded in {time.time()-t0:.1f}s (checkpoint stays resident for the whole shard)")
    if not args.no_ball:
        t1 = time.time()
        handles["ball"] = L.load_scene(mi, "ball", poses0.cam_pos[0], poses0.light_pos[0], cfgs["ball"], log=log)
        say(f"ball scene loaded in {time.time()-t1:.1f}s")

    done = failed = skipped = 0
    prev_end = None
    for k, seq in enumerate(todo):
        out = args.root / seq["dir"]
        if (out / "RENDER_COMPLETE").exists():
            skipped += 1
            say(f"[{k+1}/{len(todo)}] skip {seq['dir']} (RENDER_COMPLETE present)")
            continue
        gap = None if prev_end is None else time.time() - prev_end
        try:
            r = render_one(mi, dr, torch, seq, handles, cfgs, render_fn, args.root, log, do_ball=not args.no_ball)
            done += 1
            say(f"[{k+1}/{len(todo)}] {seq['dir']} seed={seq['seed']} keep_exr={seq['keep_exr']} "
                f"wall={r['wall']:.1f}s material={r['material']:.1f}s ({r['s_per_frame']:.2f} s/frame) "
                f"ball={r['ball']:.1f}s gap_before={'n/a' if gap is None else f'{gap:.2f}s'}")
        except Exception as e:  # one bad sequence must not kill the rest of the shard
            failed += 1
            import traceback
            say(f"[{k+1}/{len(todo)}] FAILED {seq['dir']}: {e!r}\n{traceback.format_exc()}")
        prev_end = time.time()

    say(f"shard {args.shard} finished: {done} rendered, {skipped} skipped, {failed} failed, "
        f"wall {time.time()-t0:.1f}s")
    handles["material"].close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
