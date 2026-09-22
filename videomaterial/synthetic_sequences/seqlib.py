"""Library half of the exp-015 sequence renderer: poses, scenes, per-frame rendering, metadata.

``render_sequence.py`` (one sequence, one process) and ``render_manifest.py`` (a shard of the B1
dataset, one scene load for hundreds of sequences) are both thin CLIs over this module, so the two
produce bit-identical frames: the per-frame chain is exactly

    handle.set_pose(cam, light)  ->  random.seed(render_base + index)  ->  render_with_batches(...)

``render_with_batches`` draws its per-batch Mitsuba seeds from Python's global RNG, which the
``random.seed`` call resets at every frame; nothing carries over from the previous frame or the
previous sequence, which is what makes the one-load driver safe.
"""
from __future__ import annotations

import contextlib, hashlib, json, os, platform, random, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_sequences import trajectories as T  # noqa: E402
from synthetic_sequences import scene as S  # noqa: E402

SETUPS = {"B1": ("B1_Fixed_camera", T.TrajectorySpec.b1_light),
          "B2": ("B2_Fixed_light", T.TrajectorySpec.b2_camera)}
MODES = ("material", "ball", "white_lambert")


# ------------------------------------------------------------------------------------- utilities
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
    return {"repo": str(repo), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


@dataclass
class RenderConfig:
    """Everything that is not the trajectory itself."""
    spp: int = 64
    batch_spp: int = 16
    max_depth: int = 4
    fov: float = 35.0
    width: int = 832
    height: int = 480
    fps: int = 15
    frames: int = 81
    intensity: float = 20.0
    render_seed: int = 20260922
    variant: str = "cuda_ad_rgb"
    ball_roughness: float = 0.3
    ground_roughness: float = 0.15
    scene_src: Optional[Path] = None
    checkpoint_root: Optional[str] = None
    rendering_root: Optional[Path] = None


# ----------------------------------------------------------------------------------------- poses
@dataclass
class PoseSet:
    setup: str
    setup_dir: str
    cls: str
    seed: int
    spec: "T.TrajectorySpec"
    traj: "T.Trajectory"
    cam_pos: np.ndarray            # (F, 3)
    light_pos: np.ndarray          # (F, 3)


def build_poses(setup: str, cls: str, seed: int, frames: int = 81, profile=None) -> PoseSet:
    """Deterministic camera/light positions for one sequence (no Mitsuba needed)."""
    setup_dir, spec_fn = SETUPS[setup]
    spec = spec_fn(frames=frames)
    traj = T.sample_trajectory(cls, seed, spec, profile=profile)
    moving = traj.positions
    fixed = spec.fixed_direction * spec.radius
    if spec.moving_element == "light":
        light_pos, cam_pos = moving, np.repeat(fixed[None], frames, 0)
    else:
        cam_pos, light_pos = moving, np.repeat(fixed[None], frames, 0)
    return PoseSet(setup, setup_dir, cls, seed, spec, traj, cam_pos, light_pos)


# ---------------------------------------------------------------------------------------- scenes
def load_render_backend(rendering_root):
    """RoboCloth's batched renderer (``rendering/render.py``); the BSDF plugins must already be
    registered for the active variant, which ``load_scene`` does for material mode."""
    sys.path.insert(0, str(rendering_root))
    from render import render_with_batches  # noqa: E402
    return render_with_batches


class SceneHandle:
    """A loaded Mitsuba scene whose camera and light can be moved through ``mi.traverse``.

    Loading is the expensive part (the material scene pulls in the 1.16 GB checkpoint), so a driver
    builds one handle per mode and then walks as many sequences as it likes through ``set_pose``.
    """

    def __init__(self, mi, scene, mode, material_meta, tmp_scene=None):
        self.mi, self.scene, self.mode = mi, scene, mode
        self.material_meta, self.tmp_scene = material_meta, tmp_scene
        self.params = mi.traverse(scene)
        keys = list(self.params.keys())
        # scene.xml sensors are anonymous ("object_<n>.to_world"); find the one that also has x_fov
        prefixes = {k[: -len(".x_fov")] for k in keys if k.endswith(".x_fov")}
        self.cam_key = next((f"{p}.to_world" for p in prefixes if f"{p}.to_world" in keys), None)
        self.light_key = "light.position" if "light.position" in keys else None
        if self.cam_key is None or self.light_key is None:
            raise KeyError(f"need a sensor to_world and light.position in traverse keys; got {keys[:40]}")

    def set_pose(self, cam_pos, light_pos):
        self.params[self.cam_key] = self.mi.Transform4f(S.look_at_matrix(cam_pos))
        self.params[self.light_key] = self.mi.Point3f([float(x) for x in light_pos])
        self.params.update()

    def close(self):
        if self.tmp_scene is not None:
            import shutil
            shutil.rmtree(self.tmp_scene, ignore_errors=True)
            self.tmp_scene = None


def load_scene(mi, mode: str, cam_pos0, light_pos0, cfg: RenderConfig, log=None) -> SceneHandle:
    """Load one of the three scenes. Noisy loaders are redirected into ``log`` when given."""
    sink = contextlib.nullcontext() if log is None else contextlib.redirect_stdout(log)
    sink2 = contextlib.nullcontext() if log is None else contextlib.redirect_stderr(log)
    with sink, sink2:
        if mode == "material":
            scene, spec_json, tmp = S.load_material_scene(cfg.scene_src, cfg.checkpoint_root, cfg.rendering_root)
            ckpt = Path(cfg.checkpoint_root) / "314" / "Ours_epoch80.ckpt"
            meta = {"kind": "robocloth_neural", "id": 314,
                    "checkpoint": {"path": str(ckpt), "sha256": sha256(ckpt) if ckpt.exists() else None},
                    "materials_json": spec_json, "bsdf_params": None}
            return SceneHandle(mi, scene, mode, meta, tmp)
        if mode == "ball":
            scene = mi.load_dict(S.ball_scene_dict(cam_pos0, light_pos0, cfg.fov, cfg.width, cfg.height,
                                                   cfg.intensity, ball_roughness=cfg.ball_roughness,
                                                   ground_roughness=cfg.ground_roughness, max_depth=cfg.max_depth))
            meta = {"kind": "mitsuba_principled", "id": None, "checkpoint": None,
                    "bsdf_params": {"ball": {"base_color": [0.6, 0.6, 0.6], "roughness": cfg.ball_roughness,
                                             "specular": 0.5, "radius": S.BALL_RADIUS, "center": S.BALL_CENTER},
                                    "sample_patch": {"base_color": [0.5, 0.5, 0.5], "roughness": cfg.ground_roughness,
                                                     "specular": 0.6, "half_extent": S.SAMPLE_HALF_EXTENT, "y": 0.002},
                                    "ground": {"diffuse_reflectance": [0.28, 0.28, 0.28], "y": 0.0}}}
            return SceneHandle(mi, scene, mode, meta)
        if mode == "white_lambert":
            scene = mi.load_dict(S.white_lambert_scene_dict(cam_pos0, light_pos0, cfg.fov, cfg.width,
                                                            cfg.height, cfg.intensity, max_depth=cfg.max_depth))
            meta = {"kind": "mitsuba_diffuse_white", "id": None, "checkpoint": None,
                    "bsdf_params": {"cloth": {"reflectance": [1.0, 1.0, 1.0]}}}
            return SceneHandle(mi, scene, mode, meta)
    raise ValueError(f"unknown mode {mode!r}")


# -------------------------------------------------------------------------------------- rendering
def write_bitmap_pair(mi, image, png_path, exr_path=None):
    """PNG = global Reinhard x/(1+x) -> sRGB by mi.util.write_bitmap -> 8-bit; EXR = linear float32."""
    arr = np.asarray(image, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite radiance in {Path(png_path).name}")
    png = np.nan_to_num(arr / (1.0 + arr), nan=0.0, posinf=0.0, neginf=0.0)
    mi.util.write_bitmap(str(png_path), mi.Bitmap(png))
    if exr_path is not None:
        mi.util.write_bitmap(str(exr_path), mi.Bitmap(arr))
    return {"linear_min": float(arr.min()), "linear_max": float(arr.max()),
            "linear_mean": float(arr.mean()), "finite": True}


def render_poses(mi, handle: SceneHandle, indices, cam_pos, light_pos, cfg: RenderConfig,
                 render_fn, png_dir: Path, exr_dir: Optional[Path] = None, log=None,
                 spp=None, batch_spp=None, on_frame=None):
    """Render ``indices`` of a pose schedule into ``png_dir`` (and ``exr_dir`` when given).

    Returns one record per frame: index, seconds, stats and the written paths. ``spp`` /
    ``batch_spp`` override ``cfg`` (the ball twin is cheaper than the material render)."""
    spp = cfg.spp if spp is None else spp
    batch_spp = cfg.batch_spp if batch_spp is None else batch_spp
    png_dir.mkdir(parents=True, exist_ok=True)
    if exr_dir is not None:
        exr_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx in indices:
        handle.set_pose(cam_pos[idx], light_pos[idx])
        seed = cfg.render_seed + idx
        random.seed(seed)
        t0 = time.time()
        sink = contextlib.nullcontext() if log is None else contextlib.redirect_stdout(log)
        sink2 = contextlib.nullcontext() if log is None else contextlib.redirect_stderr(log)
        with sink, sink2:
            img = render_fn(handle.scene, spp, batch_spp)
        png = png_dir / f"frame_{idx:03d}.png"
        exr = None if exr_dir is None else exr_dir / f"frame_{idx:03d}.exr"
        stats = write_bitmap_pair(mi, img, png, exr)
        rec = {"index": idx, "seed": seed, "seconds": time.time() - t0, "stats": stats,
               "png": png, "exr": exr}
        out.append(rec)
        if on_frame is not None:
            on_frame(rec)
    return out


# --------------------------------------------------------------------------------------- metadata
def build_metadata(out: Path, poses: PoseSet, mode: str, cfg: RenderConfig, material_meta: dict,
                   mi, dr, torch, started: float, split: str, script: str) -> dict:
    """Schema-v2 metadata (SPEC section 3), sufficient to re-derive every condition signal."""
    spec = poses.spec
    return {
        "schema_version": 2, "status": "rendering",
        "sequence_id": str(out.relative_to(out.parents[2])) if len(out.parents) > 2 else out.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "generator": {**git_info(HERE.parents[1]), "script": script, "python": platform.python_version(),
                      "mitsuba": mi.__version__, "drjit": dr.__version__, "torch": torch.__version__,
                      "variant": cfg.variant},
        "seeds": {"trajectory": poses.seed, "render_base": cfg.render_seed},
        "setup": poses.setup, "mode": mode,
        "sample": {"shape": "rectangle", "half_extent_xz": [S.SAMPLE_HALF_EXTENT, S.SAMPLE_HALF_EXTENT],
                   "transform_to_world": "rotate(-90 deg about +X) then scale(0.75): the unit rectangle in the XY plane becomes the XZ plane with +Y normal",
                   "uv_convention": "u -> +X, v -> +Z (Mitsuba rectangle UVs after the -90 deg X rotation), origin at (-0.75, 0, -0.75)",
                   "uv_tiling": 1.0, "present_in_mode": mode != "ball"},
        "frame_convention": {"world": "sample frame: origin at sample centre, +Y = normal, +X = u, +Z = v",
                             "theta": "polar angle from +Y, degrees", "phi": "azimuth from +X towards +Z, degrees",
                             "camera_to_world": "Mitsuba perspective sensor: camera-space +Z is the viewing direction, +Y is up, +X points to the image LEFT (Mitsuba mirrors X); use check_metadata.py to regenerate rays",
                             "pixel_center": 0.5},
        "material": material_meta,
        "camera": {"mode": "trajectory" if spec.moving_element == "camera" else "fixed", "projection": "perspective",
                   "fov_deg": cfg.fov, "fov_axis": "y", "resolution": [cfg.width, cfg.height],
                   "look_at": [0.0, 0.0, 0.0], "up_hint": [0.0, 1.0, 0.0], "radius": spec.radius,
                   "intrinsics_K": S.intrinsics(cfg.width, cfg.height, cfg.fov, "y"),
                   "fixed_direction_deg": None if spec.moving_element == "camera" else {"theta": spec.fixed_theta_deg, "phi": spec.fixed_phi_deg}},
        "light": {"type": "point", "mode": "trajectory" if spec.moving_element == "light" else "fixed",
                  "radius_from_origin": spec.radius, "intensity_rgb": [cfg.intensity] * 3, "source_radius": 0.0,
                  "units": "Mitsuba point-light intensity I (W/sr); irradiance at distance r = I cos(theta) / r^2",
                  "fixed_direction_deg": None if spec.moving_element == "light" else {"theta": spec.fixed_theta_deg, "phi": spec.fixed_phi_deg},
                  "visual_marker": {"present": False, "note": "light position is drawn as an overlay by make_visualization.py"}},
        "trajectory": {"split": split, **poses.traj.to_metadata()},
        "render": {"spp": cfg.spp, "batch_spp": cfg.batch_spp, "max_depth": cfg.max_depth, "integrator": "path",
                   "sampler": "independent", "denoiser": None, "exposure": 1.0,
                   "tonemap": "global Reinhard x/(1+x) then sRGB (mi.util.write_bitmap), 8-bit PNG",
                   "exr": "linear RGB float32",
                   "per_frame_seed": "render_base + frame index (random.seed before the batches)"},
        "frames": {"count": cfg.frames, "fps": cfg.fps, "jsonl": "frames.jsonl", "png_dir": "frames_png",
                   "exr_dir": "frames_exr", "rendered_indices": None},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": platform.node()},
    }


def frame_record(rec, poses: PoseSet, cfg: RenderConfig, out: Path, extra_files=None):
    """One ``frames.jsonl`` line: pose, files and per-frame statistics."""
    idx = rec["index"]
    cp, lp = poses.cam_pos[idx], poses.light_pos[idx]
    cth, cph = T.vec_to_ang(cp / np.linalg.norm(cp))
    lth, lph = T.vec_to_ang(lp / np.linalg.norm(lp))
    files = {"png": str(rec["png"].relative_to(out)),
             "exr": None if rec["exr"] is None else str(rec["exr"].relative_to(out))}
    files.update(extra_files or {})
    return {"index": idx, "t": idx / max(cfg.frames - 1, 1), "seed": rec["seed"], "seconds": rec["seconds"],
            "camera": {"position": cp.tolist(), "c2w": S.transform_to_list(S.look_at_matrix(cp)),
                       "theta_deg": float(cth), "phi_deg": float(cph)},
            "light": {"position": lp.tolist(), "theta_deg": float(lth), "phi_deg": float(lph),
                      "distance": float(np.linalg.norm(lp))},
            "files": files, "stats": rec["stats"]}


def finalize(out: Path, skip=("MANIFEST.sha256", "RENDER_COMPLETE", "render.log")):
    """Checksum every output file, then drop the RENDER_COMPLETE flag the resumable drivers look for."""
    lines = [f"{sha256(p)}  {p.relative_to(out)}" for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in skip]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    (out / "RENDER_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
