"""Render a scene folder (scene.xml + materials.json) with neural BRDFs.

A scene folder contains a Mitsuba 3 ``scene.xml`` (shapes carry ids), a
``materials.json`` mapping shape ids to Stage-2 checkpoints, and any meshes
referenced relatively by the XML.  Shapes not listed in ``materials.json``
keep their XML BSDF.  See ``README.md`` for the materials.json format.

Usage:

    python render.py scene=examples/cloth_on_bar
    python render.py scene=examples/teaser render.spp=64 render.width=1920 render.height=1080

Output goes to ``<output_base>/<output_name>.png`` (and ``.exr``, linear).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import random
import time

import hydra
from omegaconf import DictConfig

import drjit as dr

# Disable autodiff bookkeeping we don't need for forward rendering.
dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)

import numpy as np
import torch

import mitsuba as mi
# Variant is set from cfg.variant inside main(). Custom BSDF imports and
# mi.register_bsdf() must run AFTER set_variant because BSDF subclasses bind
# to the active variant at class-construction time.

# Our released Stage-2 checkpoints all use this wrapper; override per
# assignment with "material_type" for UBO / Bonn / PBR checkpoints.
DEFAULT_MATERIAL_TYPE = "AnisotropicLatentTexturedModel"

# Keys understood in a materials.json assignment; anything else is a typo.
_ASSIGNMENT_KEYS = {
    "material", "ckpt", "material_type", "uv_tiling", "uv_inset",
    "two_sided", "grazing_mask_deg", "apply_cosine_at_eval",
    "use_btf", "btf_path",
}


def _require_checkpoint_root(shape_id: str, checkpoint_root: str, what: str) -> None:
    """Fail closed on an unset or non-existent ``checkpoint_root``.

    A stale placeholder (the bundled examples default to
    ``${BRDF_CKPT_ROOT:-/absolute/path/to/...}``) or a typo must stop the
    render here, with a message that says how to fix it, rather than
    surfacing later as a confusing glob miss.
    """
    if not checkpoint_root:
        raise ValueError(f"[{shape_id}] {what} needs a checkpoint_root")
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(
            f"[{shape_id}] checkpoint_root is not a directory: {checkpoint_root!r} "
            "(materials.json \"checkpoint_root\" after env expansion). Download the "
            "released checkpoints with scripts/download_material.sh and point "
            "BRDF_CKPT_ROOT (or checkpoint_root) at .../checkpoints/stage2/RoboCloth.")


def resolve_checkpoint(shape_id: str, assignment: dict, checkpoint_root: str) -> str:
    """Resolve one assignment to a checkpoint file.

    ``{"material": "<id>"}`` globs ``<checkpoint_root>/<id>/Ours_epoch*.ckpt``
    (exactly one match expected).  ``{"ckpt": ...}`` is used verbatim if
    absolute, else joined onto ``checkpoint_root``.  Every failure mode — no
    root, root not a directory, no material folder, empty folder, missing
    file, ambiguous glob — raises; a shape is never silently left unassigned.
    """
    from brdf_plugin.utils.scene_loader import expand_env

    if "ckpt" in assignment:
        ckpt = expand_env(str(assignment["ckpt"]))
        if not os.path.isabs(ckpt):
            _require_checkpoint_root(shape_id, checkpoint_root, f"relative ckpt {ckpt!r}")
            ckpt = os.path.join(checkpoint_root, ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[{shape_id}] checkpoint not found: {ckpt}")
        return ckpt

    if "material" in assignment:
        _require_checkpoint_root(shape_id, checkpoint_root, "\"material\" assignments")
        material = str(assignment["material"])
        material_dir = os.path.join(checkpoint_root, material)
        pattern = os.path.join(material_dir, "Ours_epoch*.ckpt")
        if not os.path.isdir(material_dir):
            present = sorted(os.listdir(checkpoint_root))
            raise FileNotFoundError(
                f"[{shape_id}] no folder for material {material!r} under {checkpoint_root} "
                f"(it contains: {present[:10] if present else 'nothing — empty directory'}). "
                f"Download it with: bash scripts/download_material.sh {material}")
        matches = sorted(glob.glob(pattern))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"[{shape_id}] no checkpoint matches {pattern} "
                f"(folder contains: {sorted(os.listdir(material_dir))[:10]})")
        if len(matches) > 1:
            raise RuntimeError(
                f"[{shape_id}] ambiguous checkpoint glob {pattern}: {matches}")
        return matches[0]

    raise ValueError(f"[{shape_id}] assignment needs a \"material\" or \"ckpt\" key")


def load_materials(scene_dir: str, default_two_sided: bool):
    """Parse ``<scene_dir>/materials.json``.

    Returns ``(overrides, radiance)`` where ``overrides`` maps shape id to the
    mlpbrdf property dict consumed by ``load_scene_with_bsdf_overrides`` and
    ``radiance`` is an optional per-shape emitter-intensity override map.
    """
    from brdf_plugin.utils.scene_loader import expand_env

    path = os.path.join(scene_dir, "materials.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"materials.json not found: {path}")
    with open(path) as f:
        spec = json.load(f)

    checkpoint_root = expand_env(str(spec.get("checkpoint_root", "")))
    scene_two_sided = bool(spec.get("two_sided", default_two_sided))
    scene_uv_inset = bool(spec.get("uv_inset", False))
    scene_grazing = float(spec.get("grazing_mask_deg", 0.0))

    overrides = {}
    for shape_id, assignment in spec.get("assignments", {}).items():
        unknown = set(assignment) - _ASSIGNMENT_KEYS
        if unknown:
            raise KeyError(f"[{shape_id}] unknown materials.json key(s): {sorted(unknown)}")
        bsdf = {
            "type": "mlpbrdf",
            "model_path": resolve_checkpoint(shape_id, assignment, checkpoint_root),
            "material_type": str(assignment.get("material_type", DEFAULT_MATERIAL_TYPE)),
            "two_sided": bool(assignment.get("two_sided", scene_two_sided)),
        }
        if "uv_tiling" in assignment:
            bsdf["uv_tiling"] = float(assignment["uv_tiling"])
        if bool(assignment.get("uv_inset", scene_uv_inset)):
            bsdf["uv_inset"] = True
        grazing = float(assignment.get("grazing_mask_deg", scene_grazing))
        if grazing > 0:
            bsdf["grazing_mask_deg"] = grazing
        if "apply_cosine_at_eval" in assignment:
            bsdf["apply_cosine_at_eval"] = bool(assignment["apply_cosine_at_eval"])
        if bool(assignment.get("use_btf", False)):
            bsdf["use_btf"] = True
            bsdf["btf_path"] = expand_env(str(assignment.get("btf_path", "")))
        overrides[shape_id] = bsdf

    radiance = {k: float(v) for k, v in (spec.get("radiance") or {}).items()} or None
    return overrides, radiance


def render_with_batches(scene, total_spp: int, batch_spp: int):
    """Render ``total_spp`` in chunks, clearing CUDA/DrJit caches between batches."""
    from brdf_plugin.utils.cuda_manage import print_cuda_memory_info

    accumulated = None
    n_batches = (total_spp + batch_spp - 1) // batch_spp
    t0 = time.time()
    for b in range(n_batches):
        bs = min(batch_spp, total_spp - b * batch_spp)
        seed = random.randint(0, 2**31 - 1)
        print_cuda_memory_info(f"before batch {b + 1}/{n_batches}")
        # Forward render only: suspend AD so mi.render doesn't build/retain an
        # autodiff graph attached to scene params (second unbounded-memory source).
        with dr.suspend_grad():
            img = mi.render(scene, spp=bs, seed=seed)
            accumulated = img * bs if accumulated is None else accumulated + img * bs
        # Force evaluation so the lazy DrJit graph collapses to a concrete buffer
        # each batch. Without this the accumulator chains 1 add-node per batch and
        # GPU memory grows unboundedly over many batches (OOM at high spp).
        dr.eval(accumulated)
        dr.sync_thread()
        del img
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()
        print_cuda_memory_info(f"after batch  {b + 1}/{n_batches}")
    print(f"render time: {time.time() - t0:.1f}s")
    return accumulated / total_spp


@hydra.main(version_base=None, config_path="configs", config_name="render")
def main(cfg: DictConfig):
    scene_dir = hydra.utils.to_absolute_path(str(cfg.scene))
    scene_xml = os.path.join(scene_dir, "scene.xml")
    if not os.path.isfile(scene_xml):
        raise FileNotFoundError(f"Scene xml not found: {scene_xml}")

    mi.set_variant(str(cfg.variant))

    # Late imports: BSDF plugins must be registered after the variant is set.
    from brdf_plugin.cook_torrancebrdf import CookTorranceBRDF
    from brdf_plugin.mlp import MLPBRDF
    from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
    from brdf_plugin.utils.scene_loader import load_scene_with_bsdf_overrides

    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))

    overrides, radiance = load_materials(scene_dir, default_two_sided=bool(cfg.two_sided))

    print("=" * 64)
    print(f"Scene:     {scene_xml}")
    print(f"Variant:   {cfg.variant}")
    print(f"Render:    spp={cfg.render.spp}  batch_spp={cfg.render.batch_spp}")
    print("Shape -> checkpoint:")
    for sid, bsdf in overrides.items():
        print(f"  {sid:>10s}  two_sided={bsdf['two_sided']}  "
              f"({bsdf['material_type']}, {bsdf['model_path']})")
    print("=" * 64)

    # Optional fixed seed so unchanged regions are statistically comparable
    # across runs (per-batch sampler seeds come from this RNG).
    if cfg.seed is not None:
        random.seed(int(cfg.seed))

    scene = load_scene_with_bsdf_overrides(
        scene_xml, overrides,
        fov=cfg.render.fov,
        width=cfg.render.width,
        height=cfg.render.height,
        max_depth=cfg.render.max_depth,
        radiance=radiance,
    )
    print("loaded scene")

    print_cuda_memory_info("before render")
    image = render_with_batches(scene, int(cfg.render.spp), int(cfg.render.batch_spp))
    print_cuda_memory_info("after render")

    out_dir = hydra.utils.to_absolute_path(str(cfg.output_base))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg.output_name}.png")

    arr = np.array(image)
    # NO tonemapping by default: the PNG gets linear radiance (sRGB + clamp);
    # over-bright pixels clip to white, which honestly shows brightness bugs.
    # `tonemap=true` applies the legacy global Reinhard map instead.
    out_arr = arr / (1.0 + arr) if bool(cfg.tonemap) else arr
    # Guard against non-finite pixels (e.g. a diverged checkpoint whose extreme
    # latents make the decoder emit NaN/Inf). Identity for well-behaved renders.
    out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=0.0, neginf=0.0)
    mi.util.write_bitmap(out_path, mi.Bitmap(out_arr))
    # Always dump EXR for quantitative (pre-clip) analysis.
    mi.util.write_bitmap(os.path.join(out_dir, f"{cfg.output_name}.exr"), mi.Bitmap(arr))

    print(f"\nSaved {out_path}")
    print(f"  raw  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
