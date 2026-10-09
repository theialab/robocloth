#!/usr/bin/env python
"""Render passes for the "apply a trained material" room panel (fipt-mitsuba env).

r5 (full resolution): 3840x2160 frame (rows [0:1800] shown = the webpage hero), 1024 spp, one
region per piece of furniture (12 regions: each sofa body, each of the 4 throw pillows, the 4
floor cushions, the rug, and -- added after the first release -- the left curtain) and exactly
ONE mlpbrdf instance per pass (several shapes of a region
share it through a <ref>; the per-instance cost made the 4-cushion group CPU-bound).
Combinations are composited live on the page, never rendered; the 1920 px web set is the 4K
set downsampled (build_assets.py). The 480x270 crosstalk ("lo") passes are resolution-free.

(r3/r4: 1920x1080, 9 regions (pillow pairs), room_demo/passes_r3/; r2: 1280x720, passes_r2/.)

Extends the R1 room demo (~/projects/robocloth-room-demo/tools/passes.py, same scene, camera,
grey 0.55 cloth, 1280x720 frame of which rows [0:600] are shown, mlpbrdf with
uv' = frac(uv*tiling)*0.8+0.1 and one latent tile = TILE_M metres on every shape) with
  * the sofas' throw pillows as their own mask channels / passes (pillowsL, pillowsR), so a
    drop on a sofa textures the sofa and its pillows, a drop on a pillow only the pillows;
  * more materials (CKPT);
  * a flat top-down "swatch" of each material, texel-aligned with the latent tiles of
    room_demo/latent/lat_<m>_pc123.png (uv' = uv*0.8+0.1 on a unit square, i.e. texels
    0.1..0.9 = the region the latent tiles show) for the latent -> material cards.

stages
  masks            coverage of the REGIONS (one channel each, groups of three per AOV render)
                   + world position, at RES                      -> OUT/masks.npy, position.npy
  neural O M       room with region O in material M (other cloth grey), film cropped to O
  swatch M         512^2 flat patch of M under a directional light (+ dim sky), OptiX denoised
  grey             the all-grey room rendered like the neural passes (compositing reference)
  lo O M           480x270 full frame, region O in M (O=grey: none), correlated seed (crosstalk)
  uvcheck          uv AOV of the swatch camera (orientation of the swatch vs the latent tile)

  export PYTHONNOUSERSITE=1; cd ~/projects/SGHyperMaterials
  CUDA_VISIBLE_DEVICES=1 $R docs/tools/room/passes.py neural sofaL 453 --spp 128 --suffix _t128
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SGH = "/home/zla247/projects/SGHyperMaterials"
XML = f"{SGH}/data/room_new/room/room.xml"
ROOT = "/media/raid/cloth/output/teaser_video/room_demo"
OUT = f"{ROOT}/passes_r5"
FOV = 60.0
RES = (3840, 2160)
SHOW_H = 1800
GREY = 0.55
CLOTH = ["elm__52", "elm__54", "elm__43", "elm__2", "elm__3", "elm__5", "elm__55",
         "elm__56", "elm__57", "elm__58", "elm__59", "elm__29", "elm__30"]
REGIONS = {                    # mask channel order (= web order)
    "sofaL": ["elm__52"],                    # Cloud_Sofa-Fabric (image left)
    "pillow59": ["elm__59"],                 # its throw pillows, one target each
    "pillow55": ["elm__55"],
    "sofaR": ["elm__54"],                    # Cloud_Sofa001-Fabric.001
    "pillow3": ["elm__3"],
    "pillow5": ["elm__5"],
    "cush58": ["elm__58"],                   # LovePillow_002
    "cush2": ["elm__2"],                     # polstar_3 (floor pillow)
    "cush56": ["elm__56"],                   # Round_Pillow
    "cush57": ["elm__57"],                   # Pillow_002 (tufted floor cushion)
    "rug": ["elm__43"],                      # carpet-carpet (its white border elm__44 is not cloth)
    "curtainL": ["elm__30"],                 # curtains-curtain.green: the drape at the image's left
                                             # edge (appended last: the earlier channels keep their
                                             # meaning; elm__29, the right-hand curtain, stays grey)
}
#: metres per uv unit, sqrt(world area / uv area) of each PLY
M_PER_UV = {"elm__52": 3.665, "elm__54": 3.665, "elm__2": 1.23, "elm__56": 0.527,
            "elm__57": 0.166, "elm__58": 0.945, "elm__3": 0.972, "elm__5": 0.943,
            "elm__55": 0.818, "elm__59": 0.818, "elm__43": 5.091, "elm__30": 7.527}
TILE_M = 0.46
_S2 = "/media/raid/cloth/output/BRDF/Stage-2-Finals/Ours/"
_CI = "/data/colin/robocloth_ci/teaser_ckpts/checkpoints/stage2/RoboCloth/"
CKPT = {
    9: _S2 + "9/Ours_epoch92.ckpt", 50: _S2 + "50/Ours_epoch92.ckpt",
    145: _S2 + "145/Ours_epoch112.ckpt", 226: _S2 + "226/Ours_epoch100.ckpt",
    370: _S2 + "370/Ours_epoch80.ckpt", 452: _S2 + "452/Ours_epoch80.ckpt",
    453: _S2 + "453/Ours_epoch100.ckpt", 190: _CI + "190/Ours_epoch80.ckpt",
    8: _S2 + "8/Ours_epoch92.ckpt", 26: _S2 + "26/Ours_epoch92.ckpt",
    367: _S2 + "367/Ours_epoch96.ckpt", 311: _CI + "311/Ours_epoch94.ckpt",
    314: _S2 + "314/Ours_epoch80.ckpt",
}


def tiling(sid):
    return M_PER_UV[sid] / TILE_M


# ------------------------------------------------------------------ xml
def build_xml(out_xml, shape_bsdf, crop=None, integrator=None, shared=None):
    """shared: (bsdf_element) declared once at top level; shape_bsdf values may then be
    ref_bsdf(id) so that several shapes use ONE BSDF instance."""
    scene_dir = os.path.dirname(XML)
    tree = ET.parse(XML)
    root = tree.getroot()
    if shared is not None:
        first_shape = next(i for i, ch in enumerate(list(root)) if ch.tag == "shape")
        root.insert(first_shape, shared)
    for s in root.iter("string"):
        if s.get("name") == "filename" and not os.path.isabs(s.get("value")):
            s.set("value", os.path.join(scene_dir, s.get("value")))
    for sid in CLOTH:
        el = root.find(f".//shape[@id='{sid}']")
        for ch in list(el):
            if ch.tag in ("bsdf", "ref") and ch.get("name", "bsdf") == "bsdf":
                el.remove(ch)
        if sid in shape_bsdf:
            shape_bsdf[sid](el)
        else:
            b = ET.SubElement(el, "bsdf", {"type": "diffuse"})
            ET.SubElement(b, "rgb", {"name": "reflectance", "value": f"{GREY} {GREY} {GREY}"})
    root.find(".//sensor//float[@name='fov']").set("value", str(FOV))
    if crop is not None:
        film = root.find(".//sensor/film")
        for k, v in zip(("crop_offset_x", "crop_offset_y", "crop_width", "crop_height"), crop):
            ET.SubElement(film, "integer", {"name": k, "value": str(int(v))})
    if integrator is not None:
        old = root.find("./integrator")
        idx = list(root).index(old)
        root.remove(old)
        root.insert(idx, integrator)
    tree.write(out_xml)


def shared_neural(mid, til, bid="nbsdf"):
    b = ET.Element("bsdf", {"type": "mlpbrdf", "id": bid})
    for k, v in (("model_path", CKPT[mid]), ("material_type", "AnisotropicLatentTexturedModel"),
                 ("tiling", f"{til:.4f}")):
        ET.SubElement(b, "string", {"name": k, "value": v})
    return b


def ref_bsdf(bid="nbsdf"):
    def f(el):
        ET.SubElement(el, "ref", {"id": bid, "name": "bsdf"})
    return f


def neural_bsdf(mid, til):
    def f(el):
        b = ET.SubElement(el, "bsdf", {"type": "mlpbrdf"})
        for k, v in (("model_path", CKPT[mid]),
                     ("material_type", "AnisotropicLatentTexturedModel"),
                     ("tiling", f"{til:.4f}")):
            ET.SubElement(b, "string", {"name": k, "value": v})
    return f


def rgb_bsdf(rgb):
    def f(el):
        b = ET.SubElement(el, "bsdf", {"type": "diffuse"})
        ET.SubElement(b, "rgb", {"name": "reflectance", "value": " ".join(map(str, rgb))})
    return f


def aov_integrator(aovs):
    i = ET.Element("integrator", {"type": "aov"})
    ET.SubElement(i, "string", {"name": "aovs", "value": aovs})
    return i


# ------------------------------------------------------------------ mitsuba
def mi_init(neural=False):
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, False)
    if neural:
        if SGH not in sys.path:
            sys.path.insert(0, SGH)
        os.chdir(SGH)
        from custom_bsdf.mlp import MLPBRDF
        mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))
    return mi, dr


def render(mi, dr, scene, spp, batch, seed=1):
    random.seed(seed)
    acc = None
    nb = (spp + batch - 1) // batch
    for b in range(nb):
        bs = min(batch, spp - b * batch)
        with dr.suspend_grad():
            img = mi.render(scene, spp=bs, seed=random.randint(0, 2 ** 31 - 1))
            acc = img * bs if acc is None else acc + img * bs
        dr.eval(acc)
        dr.sync_thread()
        del img
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        dr.flush_malloc_cache()
    return acc / spp


def denoise(mi, img):
    h, w = img.shape[:2]
    dn = mi.OptixDenoiser(input_size=[w, h], albedo=False, normals=False, temporal=False)
    return np.array(dn(mi.TensorXf(img))).astype(np.float32)


def srgb8(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


def write_png(path, rgb8):
    cv2.imwrite(path, np.ascontiguousarray(rgb8[:, :, ::-1]))


def log_time(key, rec):
    """times.json (merged view) + one file per pass in times.d/ (safe with several GPUs)."""
    os.makedirs(f"{OUT}/times.d", exist_ok=True)
    json.dump(rec, open(f"{OUT}/times.d/{key}.json", "w"), indent=1)
    import fcntl
    with open(f"{OUT}/times.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = f"{OUT}/times.json"
        d = json.load(open(p)) if os.path.exists(p) else {}
        d[key] = rec
        json.dump(d, open(p + ".tmp", "w"), indent=1)
        os.replace(p + ".tmp", p)


# ------------------------------------------------------------------ stages
def stage_masks(a):
    """AOV renders: regions coded R,G,B in groups of three, then all regions black;
    coverage = coded - black (9 channels, REGIONS order) + world position."""
    mi, dr = mi_init()
    names = list(REGIONS)
    groups = [names[i:i + 3] for i in range(0, len(names), 3)] + [[]]
    imgs = []
    for gi, grp in enumerate(groups):
        sb = {}
        for name in names:
            c = [0.0, 0.0, 0.0]
            if name in grp:
                c[grp.index(name)] = 1.0
            for sid in REGIONS[name]:
                sb[sid] = rgb_bsdf(c)
        xml = f"{OUT}/_masks_{gi}.xml"
        build_xml(xml, sb, integrator=aov_integrator("alb:albedo,pos:position"))
        scene = mi.load_file(xml, resx=str(RES[0]), resy=str(RES[1]))
        t0 = time.time()
        img = np.array(render(mi, dr, scene, a.spp, 8, seed=7))
        print("aov", gi, img.shape, f"{time.time() - t0:.1f}s", flush=True)
        imgs.append(img)
    blk = imgs[-1][:, :, 0:3]
    cov = np.concatenate([np.clip(im[:, :, 0:3] - blk, 0, 1) for im in imgs[:-1]], -1)[:, :, :len(names)]
    pos = imgs[-1][:, :, 3:6]
    np.save(f"{OUT}/masks.npy", cov.astype(np.float32))
    np.save(f"{OUT}/position.npy", pos.astype(np.float32))
    for gi in range(len(groups) - 1):
        write_png(f"{OUT}/masks_{gi}.png", (np.clip(imgs[gi][:, :, 0:3] - blk, 0, 1) * 255 + 0.5).astype(np.uint8))
    info = {}
    for k, name in enumerate(names):
        ys, xs = np.where(cov[:, :, k] > 0.01)
        sel = cov[:, :, k] > 0.5
        info[name] = dict(channel=k, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                          px=int(sel.sum()))
        print(name, info[name])
    json.dump(info, open(f"{OUT}/masks.json", "w"), indent=1)


def crop_for(name, pad=64):
    """bbox + pad, on even pixels (the 1920 web set is the 4K crop downsampled 2x2)"""
    info = json.load(open(f"{OUT}/masks.json"))[name]
    x0, y0, x1, y1 = info["bbox"]
    x0, y0 = max(0, x0 - pad) // 2 * 2, max(0, y0 - pad) // 2 * 2
    x1, y1 = min(RES[0], x1 + pad + 1), min(RES[1], y1 + pad + 1)
    x1, y1 = x0 + (x1 - x0 + 1) // 2 * 2, y0 + (y1 - y0 + 1) // 2 * 2
    return (x0, y0, min(x1, RES[0]) - x0, min(y1, RES[1]) - y0)


def stage_neural(a):
    mi, dr = mi_init(neural=True)
    name, mid = a.obj, int(a.mat)
    crop = crop_for(name)
    til = float(np.mean([tiling(sid) for sid in REGIONS[name]]))   # one instance per pass
    sb = {sid: ref_bsdf() for sid in REGIONS[name]}
    tag = f"neural_{name}_{mid}" + (a.suffix or "")
    xml = f"{OUT}/_{tag}.xml"
    build_xml(xml, sb, crop=crop, shared=shared_neural(mid, til))
    t_load = time.time()
    scene = mi.load_file(xml, resx=str(RES[0]), resy=str(RES[1]))
    t_load = time.time() - t_load
    t0 = time.time()
    render(mi, dr, scene, 1, 1, seed=3)
    t_warm = time.time() - t0
    t0 = time.time()
    if a.batch <= 0:                        # auto: <= ~12 M samples per batch (GPU memory)
        a.batch = int(max(4, min(32, 12e6 / (crop[2] * crop[3]))))
    img = render(mi, dr, scene, a.spp, a.batch, seed=11)
    t_render = time.time() - t0
    noisy = np.array(img)[:, :, :3].astype(np.float32)
    t0 = time.time()
    den = denoise(mi, noisy)
    t_den = time.time() - t0
    cv2.imwrite(f"{OUT}/{tag}_noisy.exr", noisy[:, :, ::-1].copy())
    cv2.imwrite(f"{OUT}/{tag}.exr", den[:, :, ::-1].copy())
    write_png(f"{OUT}/{tag}.png", srgb8(den))
    h, w = noisy.shape[:2]
    rec = dict(obj=name, mat=mid, crop=list(crop), res=list(RES), px=w * h, spp=a.spp,
               batch=a.batch, load_s=round(t_load, 1), warmup_1spp_s=round(t_warm, 1),
               render_s=round(t_render, 1), denoise_s=round(t_den, 2),
               s_per_Mpx_per_spp=round(t_render / (w * h / 1e6) / a.spp, 4),
               tiling=round(til, 3), shapes=REGIONS[name],
               gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    log_time(tag, rec)
    print(json.dumps(rec), flush=True)


SW = 512


def swatch_scene(mi, bsdf):
    """Unit square (uv 0..1) seen top-down by an orthographic camera, film = the square."""
    el = 50.0                                         # light elevation (deg), from the upper left
    az = np.radians(235.0)              # lower left in world = upper left once flipped (row = v)
    d = np.array([np.cos(np.radians(el)) * np.cos(az), np.cos(np.radians(el)) * np.sin(az),
                  np.sin(np.radians(el))])
    return mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {"type": "orthographic",
                   "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, 3], target=[0, 0, 0],
                                                              up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": SW, "height": SW, "rfilter": {"type": "box"}},
                   "sampler": {"type": "independent", "sample_count": 16}},
        "patch": {"type": "rectangle", "bsdf": bsdf},
        "sun": {"type": "directional", "direction": (-d).tolist(), "irradiance": {"type": "rgb", "value": 2.6}},
        "sky": {"type": "constant", "radiance": {"type": "rgb", "value": 0.18}},
    })


def stage_swatch(a):
    mi, dr = mi_init(neural=True)
    from custom_bsdf.mlp import MLPBRDF
    mid = int(a.mat)
    props = mi.Properties("mlpbrdf")
    props["model_path"] = CKPT[mid]
    props["material_type"] = "AnisotropicLatentTexturedModel"
    props["tiling"] = "1.0"
    bsdf = MLPBRDF(props)
    scene = swatch_scene(mi, bsdf)
    t0 = time.time()
    img = np.array(render(mi, dr, scene, a.spp, a.batch, seed=13))[:, :, :3].astype(np.float32)
    t_render = time.time() - t0
    den = denoise(mi, img)
    cv2.imwrite(f"{OUT}/swatch_{mid}.exr", den[:, :, ::-1].copy())
    write_png(f"{OUT}/swatch_{mid}.png", srgb8(den))
    log_time(f"swatch_{mid}", dict(mat=mid, res=[SW, SW], spp=a.spp, render_s=round(t_render, 1),
                                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("swatch", mid, f"{t_render:.1f}s", den.mean((0, 1)), flush=True)


LO_RES = (480, 270)
LO_SPP, LO_BATCH, LO_SEED = 256, 32, 11


def stage_lo(a):
    """Low-res FULL frame of region O in material M (O = 'grey': no region dressed), same seed
    and batching for every lo render (correlated sampling), for the page's first-order
    crosstalk term: piece O's change of the light reaching the rest of the room, as the
    smooth irradiance ratio lo_<O>_<M> / lo_grey (build_assets.py)."""
    name = a.obj
    mid = int(a.mat) if a.mat else None
    mi, dr = mi_init(neural=True)          # same init as every lo pass (correlated sampling)
    tag = "lo_grey" if name == "grey" else f"lo_{name}_{mid}"
    xml = f"{OUT}/_{tag}.xml"
    if name == "grey":
        build_xml(xml, {})
    else:
        til = float(np.mean([tiling(sid) for sid in REGIONS[name]]))
        build_xml(xml, {sid: ref_bsdf() for sid in REGIONS[name]}, shared=shared_neural(mid, til))
    scene = mi.load_file(xml, resx=str(LO_RES[0]), resy=str(LO_RES[1]))
    t_load = time.time()
    render(mi, dr, scene, 1, 1, seed=3)
    t_warm = time.time() - t_load
    t0 = time.time()
    img = np.array(render(mi, dr, scene, LO_SPP, LO_BATCH, seed=LO_SEED))[:, :, :3].astype(np.float32)
    t_render = time.time() - t0
    den = denoise(mi, img)
    cv2.imwrite(f"{OUT}/{tag}_noisy.exr", img[:, :, ::-1].copy())
    cv2.imwrite(f"{OUT}/{tag}.exr", den[:, :, ::-1].copy())
    rec = dict(obj=name, mat=mid, res=list(LO_RES), spp=LO_SPP, seed=LO_SEED, render_s=round(t_render, 1),
               warmup_1spp_s=round(t_warm, 1), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    log_time(tag, rec)
    print(json.dumps(rec), flush=True)


def stage_grey(a):
    """The all-grey room rendered exactly like the neural passes (same film / filter / denoiser):
    the reference the passes are differenced against when compositing (build_assets.py)."""
    mi, dr = mi_init()
    xml = f"{OUT}/_grey.xml"
    build_xml(xml, {})
    scene = mi.load_file(xml, resx=str(RES[0]), resy=str(RES[1]))
    t0 = time.time()
    img = np.array(render(mi, dr, scene, a.spp, a.batch, seed=11))[:, :, :3].astype(np.float32)
    t_render = time.time() - t0
    den = denoise(mi, img)
    cv2.imwrite(f"{OUT}/grey_ref.exr", den[:, :, ::-1].copy())
    write_png(f"{OUT}/grey_ref.png", srgb8(den))
    log_time("grey_ref", dict(res=list(RES), spp=a.spp, render_s=round(t_render, 1),
                              gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("grey_ref", f"{t_render:.1f}s", flush=True)


def stage_uvcheck(a):
    mi, dr = mi_init()
    scene = swatch_scene(mi, {"type": "diffuse"})
    params = None  # noqa: F841
    s2 = mi.load_dict({
        "type": "scene", "integrator": {"type": "aov", "aovs": "uv:uv"},
        "sensor": scene.sensors()[0], "patch": {"type": "rectangle"}})
    img = np.array(mi.render(s2, spp=4))
    uv = img[:, :, -2:] if img.shape[2] >= 2 else img
    print("shape", img.shape)
    for (y, x) in ((0, 0), (0, SW - 1), (SW - 1, 0), (SW - 1, SW - 1)):
        print("pixel row", y, "col", x, "uv", uv[y, x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "neural", "swatch", "grey", "lo", "uvcheck"])
    ap.add_argument("obj", nargs="?")
    ap.add_argument("mat", nargs="?")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.stage == "swatch":
        a.mat = a.obj
    os.makedirs(OUT, exist_ok=True)
    dict(masks=stage_masks, neural=stage_neural, swatch=stage_swatch, grey=stage_grey, lo=stage_lo,
         uvcheck=stage_uvcheck)[a.stage](a)


if __name__ == "__main__":
    main()
