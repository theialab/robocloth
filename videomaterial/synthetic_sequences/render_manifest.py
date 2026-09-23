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


def render_one(mi, dr, torch, seq, handles, cfgs, render_fn, root: Path, log, do_ball=True, setup="B1", twin="ball"):
    out = root / seq["dir"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    poses = L.build_poses(setup, seq["class"], seq["seed"], cfgs["material"].frames)
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
        ball = L.render_poses(mi, handles[twin], indices, poses.cam_pos, poses.light_pos,
                              cfgs[twin], render_fn, png_dir=out / f"{twin}_png", exr_dir=None, log=log)
        t_ball = time.time() - t0
    ball_by_idx = {r["index"]: r for r in ball}

    with (out / "frames.jsonl").open("w") as fj:
        for rec in mat:
            b = ball_by_idx.get(rec["index"])
            extra = {f"{twin}_png": None if b is None else str(b["png"].relative_to(out))}
            line = L.frame_record(rec, poses, cfgs["material"], out, extra_files=extra)
            if b is not None:
                line[f"{twin}_stats"] = b["stats"]
                line[f"{twin}_seconds"] = b["seconds"]
            fj.write(json.dumps(line, sort_keys=True) + "\n")

    secs = [r["seconds"] for r in mat]
    meta["frames"]["rendered_indices"] = indices
    meta["frames"]["exr_dir"] = "frames_exr" if seq["keep_exr"] else None
    meta[f"{twin}_render"] = {"rendered": do_ball, "mode": twin, "spp": cfgs[twin].spp,
                           "batch_spp": cfgs[twin].batch_spp, "png_dir": f"{twin}_png", "exr_dir": None,
                           "bsdf_params": handles[twin].material_meta["bsdf_params"] if do_ball else None,
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


def render_twin_only(mi, dr, torch, seq, handle, cfg, render_fn, root: Path, log, twin: str, setup="B1"):
    """Add a `<twin>_png/` render (same poses) to an already-complete material sequence.
    Resumable via the `<TWIN>_COMPLETE` marker; PNG only; metadata gets a `<twin>_render` block."""
    out = root / seq["dir"]
    marker = out / f"{twin.upper()}_COMPLETE"
    if marker.exists():
        return None
    if not (out / "RENDER_COMPLETE").exists():
        raise RuntimeError(f"{seq['dir']}: material render not complete; twin-only pass refuses to run")
    started = time.time()
    poses = L.build_poses(setup, seq["class"], seq["seed"], cfg.frames)
    indices = list(range(cfg.frames))
    recs = L.render_poses(mi, handle, indices, poses.cam_pos, poses.light_pos, cfg, render_fn,
                          png_dir=out / f"{twin}_png", exr_dir=None, log=log)
    with (out / f"{twin}_frames.jsonl").open("w") as fj:
        for r in recs:
            fj.write(json.dumps({"index": r["index"], "seed": r["seed"], "seconds": r["seconds"],
                                 "png": str(r["png"].relative_to(out)), "stats": r["stats"]}, sort_keys=True) + "\n")
    meta = json.loads((out / "metadata.json").read_text())
    meta[f"{twin}_render"] = {
        "rendered": True, "mode": twin, "spp": cfg.spp, "batch_spp": cfg.batch_spp, "max_depth": cfg.max_depth,
        "png_dir": f"{twin}_png", "exr_dir": None, "frames_jsonl": f"{twin}_frames.jsonl",
        "material": handle.material_meta, "poses": "identical to the material sequence (same class/seed)",
        "generator": {**L.git_info(L.HERE.parents[1]), "script": str(Path(__file__).resolve()), "variant": cfg.variant},
        "render_seed_base": cfg.render_seed, "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": time.time() - started,
        "seconds_per_frame_mean": float(np.mean([r["seconds"] for r in recs])),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    L.finalize(out)                     # refresh MANIFEST.sha256 (RENDER_COMPLETE keeps its meaning)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return {"wall": time.time() - started, "s_per_frame": meta[f"{twin}_render"]["seconds_per_frame_mean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--root", type=Path, required=True, help="the B1_Fixed_camera directory")
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--twin-only", default=None, choices=["pbr_patch", "ball", "white_lambert"],
                    help="skip the material; only add <MODE>_png/ to sequences that already have RENDER_COMPLETE")
    ap.add_argument("--twin-spp", type=int, default=32)
    ap.add_argument("--twin-batch-spp", type=int, default=16)
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

    say(f"shard {args.shard}/{args.nshards}: {len(todo)} sequences, root={args.root}, setup={man.get('setup','B1')}, twin={man.get('twin','ball')} enabled={not args.no_ball}")

    import drjit as dr, mitsuba as mi, torch  # noqa: E402
    mi.set_variant(args.variant)
    render_fn = L.load_render_backend(args.rendering_root)

    setup = man.get("setup", "B1")
    twin = man.get("twin", "ball")
    R = man["render"]
    common = dict(max_depth=R["max_depth"], intensity=R["intensity"], frames=man["frames"], fps=man["fps"],
                  width=man["resolution"][0], height=man["resolution"][1], render_seed=args.render_seed,
                  variant=args.variant, scene_src=args.scene_src, checkpoint_root=args.checkpoint_root,
                  rendering_root=args.rendering_root)
    cfgs = {"material": L.RenderConfig(spp=R["material"]["spp"], batch_spp=R["material"]["batch_spp"], **common),
            twin: L.RenderConfig(spp=R[twin]["spp"], batch_spp=R[twin]["batch_spp"], **common)}

    t0 = time.time()
    poses0 = L.build_poses(setup, todo[0]["class"], todo[0]["seed"], man["frames"])
    if args.twin_only:
        twin = args.twin_only
        tcfg = L.RenderConfig(spp=args.twin_spp, batch_spp=args.twin_batch_spp, **common)
        handle = L.load_scene(mi, twin, poses0.cam_pos[0], poses0.light_pos[0], tcfg, log=log)
        say(f"twin-only pass: {twin} spp={tcfg.spp} batch={tcfg.batch_spp}; scene loaded in {time.time()-t0:.1f}s")
        done = failed = skipped = 0
        prev_end = None
        for k, seq in enumerate(todo):
            gap = None if prev_end is None else time.time() - prev_end
            try:
                r = render_twin_only(mi, dr, torch, seq, handle, tcfg, render_fn, args.root, log, twin, setup=setup)
                if r is None:
                    skipped += 1; say(f"[{k+1}/{len(todo)}] skip {seq['dir']} ({twin.upper()}_COMPLETE present)")
                else:
                    done += 1
                    say(f"[{k+1}/{len(todo)}] {seq['dir']} {twin} wall={r['wall']:.1f}s ({r['s_per_frame']:.2f} s/frame) "
                        f"gap_before={'n/a' if gap is None else f'{gap:.2f}s'}")
            except Exception as e:
                failed += 1
                import traceback
                say(f"[{k+1}/{len(todo)}] FAILED {seq['dir']}: {e!r}\n{traceback.format_exc()}")
            prev_end = time.time()
        say(f"shard {args.shard} twin-only finished: {done} rendered, {skipped} skipped, {failed} failed, wall {time.time()-t0:.1f}s")
        handle.close()
        return 1 if failed else 0
    handles = {"material": L.load_scene(mi, "material", poses0.cam_pos[0], poses0.light_pos[0], cfgs["material"], log=log)}
    say(f"material scene loaded in {time.time()-t0:.1f}s (checkpoint stays resident for the whole shard)")
    if not args.no_ball:
        t1 = time.time()
        handles[twin] = L.load_scene(mi, twin, poses0.cam_pos[0], poses0.light_pos[0], cfgs[twin], log=log)
        say(f"{twin} scene loaded in {time.time()-t1:.1f}s")

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
            r = render_one(mi, dr, torch, seq, handles, cfgs, render_fn, args.root, log, do_ball=not args.no_ball, setup=setup, twin=twin)
            done += 1
            say(f"[{k+1}/{len(todo)}] {seq['dir']} seed={seq['seed']} keep_exr={seq['keep_exr']} "
                f"wall={r['wall']:.1f}s material={r['material']:.1f}s ({r['s_per_frame']:.2f} s/frame) "
                f"{twin}={r['ball']:.1f}s gap_before={'n/a' if gap is None else f'{gap:.2f}s'}")
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
