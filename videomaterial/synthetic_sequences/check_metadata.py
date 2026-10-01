#!/usr/bin/env python3
"""G0 checks for an exp-015 sequence, using ONLY metadata.json + frames.jsonl (+ the rendered EXRs of a
``--mode white_lambert`` sequence). No Mitsuba import.

  1. derivability: per-pixel camera rays (o ⊕ d) in the sample frame, hit points on y = 0, incident
     direction ω_i, cos θ_i / r², analytic Lambertian irradiance E = I cos θ_i / r²;
  2. silhouette: analytic inside-sample mask vs rendered mask (EXR > 0). Gate (``outline_max_px``):
     the centre of every mismatched pixel lies within the box filter's half-footprint (sqrt(2)/2 px,
     --outline-px default 0.75) of the projected sample outline (the quadrilateral through the four
     projected corners) - a pixel is lit iff its square touches the sample. Also reported, not
     gated: ``silhouette_max_px``, the distance to the boundary of the pixel-CENTRE mask, which has a
     sqrt(2) floor and reaches 2 px at the acute corners of strongly foreshortened views (B2/B3
     cameras go to 80 deg), so it cannot express the "< 1 px" re-projection gate;
  3. irradiance: analytic radiance E/π (albedo 1, direct light only) vs the rendered EXR; PSNR over the
     3-px-eroded interior must exceed --psnr dB (full-mask PSNR is also reported; edge pixels cap it).
  2b. edge re-projection (sub-pixel): the box-filtered white-Lambert render gives each pixel's
     covered fraction (render / analytic radiance); summed over a band around each projected sample
     edge and divided by the edge length it is that edge's offset in px against the analytic coverage
     (8x8 supersampled). ``edge_offset_px_max`` must stay below --edge-px (default 1.0); a 1-px yaw of
     the camera reads as 1.04 px.
  4. ``--align``: the material frames (EXR > 0 where kept, else PNG > 0) and the twin frames
     (``<twin>_png`` > 0) against the projected sample outline of the SAME logged pose. Both renders
     use Mitsuba's default gaussian filter (radius 4 sigma = 2 px), so a lit pixel outside the
     outline ("spill") must have its centre within --align-px (default 2.05) of it. Unlit pixels
     deeper inside than that ("holes") are reported separately: in the material they are genuinely
     dark texels that quantise to 0 in the 8-bit PNG (grazing light), not misalignment; the gate
     requires no holes in the material EXR and in the twin. Also the material/twin mask IoU.
  5. ``--sources DATA_ROOT`` (joint B3 sequences): per frame, camera position / c2w / angles must equal
     the B2 source's frames.jsonl and light position / angles / distance the B1 source's, exactly.
Works for every setup: the camera and the light are read per frame, so B3's jointly moving camera
AND light need nothing special (rays, hit points, omega_i and E change per frame).
Pixel→ray convention (verified against mi.Sensor.sample_ray on 2026-09-22):
  d_cam = normalize( -(u - cx)/fx , -(v - cy)/fy , 1 ),  d_world = R_c2w d_cam,  o = t_c2w,
  with (u, v) pixel-centre coordinates (integer index + 0.5), u to the right, v downward.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np


def load_exr(path):
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
    except Exception:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(str(path)); dw = f.header()["dataWindow"]
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        img = np.stack([np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(H, W) for c in "RGB"], -1)
    return np.asarray(img, dtype=np.float64)[..., :3]


def rays_from_metadata(meta, rec):
    W, H = meta["camera"]["resolution"]; K = np.array(meta["camera"]["intrinsics_K"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    c2w = np.array(rec["camera"]["c2w"], dtype=np.float64)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([-(u - cx) / fx, -(v - cy) / fy, np.ones_like(u)], -1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d = d_cam @ c2w[:3, :3].T
    o = np.broadcast_to(c2w[:3, 3], d.shape)
    return o, d


def derive(meta, rec):
    """Everything a conditioning branch could want, per pixel."""
    o, d = rays_from_metadata(meta, rec)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[..., 1] / d[..., 1]
    valid = (t > 0) & np.isfinite(t)
    hit = o + t[..., None] * d
    hx, hz = np.abs(hit[..., 0]), np.abs(hit[..., 2])
    he = meta["sample"]["half_extent_xz"]
    inside = valid & (hx <= he[0]) & (hz <= he[1])
    L = np.array(rec["light"]["position"], dtype=np.float64)
    to_l = L[None, None, :] - hit
    r2 = (to_l ** 2).sum(-1)
    wi = to_l / np.sqrt(r2)[..., None]
    cos_i = np.clip(wi[..., 1], 0.0, None)                     # normal is +Y
    I = float(meta["light"]["intensity_rgb"][0])
    E = np.where(inside, I * cos_i / r2, 0.0)
    return {"o": o, "d": d, "hit": hit, "inside": inside, "wi": wi, "cos_i": cos_i, "r2": r2, "E": E,
            "uv": np.stack([(hit[..., 0] + he[0]) / (2 * he[0]), (he[1] - hit[..., 2]) / (2 * he[1])], -1)}  # Mitsuba: v grows towards -Z


def _inside_at(meta, rec, u, v):
    """Analytic inside-sample test for rays through continuous pixel coordinates (u, v)."""
    K = np.array(meta["camera"]["intrinsics_K"]); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    c2w = np.array(rec["camera"]["c2w"], dtype=np.float64)
    d = np.stack([-(u - cx) / fx, -(v - cy) / fy, np.ones_like(u)], -1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True) @ c2w[:3, :3].T
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[1] / d[..., 1]
    hit = o + t[..., None] * d
    he = meta["sample"]["half_extent_xz"]
    return (t > 0) & np.isfinite(t) & (np.abs(hit[..., 0]) <= he[0]) & (np.abs(hit[..., 2]) <= he[1])


def edge_offsets_px(meta, rec, img, dv, n=8, band=2.5):
    """Signed sub-pixel offset of each projected sample edge (+ = rendered silhouette lies outside the
    analytic one), from the coverage of a box-filtered white-Lambert render. See check 2b."""
    inside = dv["inside"]
    dist = boundary_distance(inside)
    if dist is None:
        return None
    ys, xs = np.nonzero(dist <= band + 1.0)
    acc = np.zeros(len(ys))
    for sy in range(n):
        for sx in range(n):
            acc += _inside_at(meta, rec, xs + (sx + 0.5) / n, ys + (sy + 0.5) / n)
    alpha_a = inside.astype(np.float64); alpha_a[ys, xs] = acc / (n * n)
    I = float(meta["light"]["intensity_rgb"][0])
    pred = I * dv["cos_i"] / dv["r2"] / math.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_r = np.where(pred > 1e-9, img / pred, 0.0)
    diff = alpha_r - alpha_a
    cs = corners_px(meta, rec)
    if any(c is None for c in cs):
        return None
    px, py = xs + 0.5, ys + 0.5
    out = []
    for k in range(4):
        a, b = np.array(cs[k]), np.array(cs[(k + 1) % 4]); e = b - a; Lk = float(np.linalg.norm(e))
        t = ((px - a[0]) * e[0] + (py - a[1]) * e[1]) / (Lk * Lk)
        perp = np.abs((px - a[0]) * e[1] - (py - a[1]) * e[0]) / Lk
        sel = (t >= 0) & (t <= 1) & (perp <= band)
        out.append(float(diff[ys[sel], xs[sel]].sum() / Lk))
    return out


def corners_px(meta, rec):
    """Pixel coordinates of the four sample corners under the logged camera (convention above)."""
    K = np.array(meta["camera"]["intrinsics_K"]); c2w = np.array(rec["camera"]["c2w"], dtype=np.float64)
    he = meta["sample"]["half_extent_xz"]; w2c = np.linalg.inv(c2w); out = []
    for x, z in ((-he[0], he[1]), (he[0], he[1]), (he[0], -he[1]), (-he[0], -he[1])):
        pc = w2c[:3, :3] @ np.array([x, 0.0, z]) + w2c[:3, 3]
        out.append(None if pc[2] <= 1e-9 else [float(K[0, 2] - K[0, 0] * pc[0] / pc[2]),
                                                float(K[1, 2] - K[1, 1] * pc[1] / pc[2])])
    return out


def outline_distance_px(meta, rec, ys, xs):
    """Distance (px) of the pixel centres (ys, xs) to the projected sample outline (the quadrilateral
    through the four projected corners; lines stay lines under the pinhole projection)."""
    cs = corners_px(meta, rec)
    if any(c is None for c in cs):
        return None
    P = np.stack([np.asarray(xs) + 0.5, np.asarray(ys) + 0.5], -1).astype(np.float64)
    d = np.full(len(P), np.inf)
    for k in range(4):
        a, b = np.array(cs[k]), np.array(cs[(k + 1) % 4]); e = b - a
        t = np.clip(((P - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * e), axis=-1))
    return d


def load_png_mask(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int32).sum(-1) > 0


def check_alignment(seq_dir, px_tol=2.05, frames=None):
    """Material and twin frames vs the projected sample outline of the logged pose (``--align``)."""
    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    recs = [json.loads(l) for l in (seq_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    if frames is not None:
        recs = [r for r in recs if r["index"] in frames]
    twin_keys = sorted(k for k in recs[0]["files"] if k.endswith("_png") and k != "png" and recs[0]["files"][k]) if recs else []
    out = {"sequence": str(seq_dir), "px_tol": px_tol, "twins": twin_keys, "frames_checked": 0,
           "spill_max_px": {}, "holes_px_total": {}, "holes_frac_max": {}, "material_twin_iou_min": None,
           "frames": []}
    ious = []
    for rec in recs:
        dv = derive(meta, rec); inside = dv["inside"]
        f = {"index": rec["index"], "corners_px": corners_px(meta, rec)}
        masks = {}
        keys = []
        if rec["files"].get("exr") and (seq_dir / rec["files"]["exr"]).exists():
            keys.append(("exr", lambda p: load_exr(p).sum(-1) > 0))
        keys.append(("png", load_png_mask))
        keys += [(k, load_png_mask) for k in twin_keys]
        for key, loader in keys:
            rel = rec["files"].get(key)
            if not rel or not (seq_dir / rel).exists():
                f[key] = None; continue
            m = loader(seq_dir / rel); masks[key] = m
            ys, xs = np.nonzero(m & ~inside)                  # lit outside: spill
            spill = outline_distance_px(meta, rec, ys, xs) if len(ys) else np.zeros(0)
            hy, hx = np.nonzero(inside & ~m)                  # dark inside: near the edge or holes
            hd = outline_distance_px(meta, rec, hy, hx) if len(hy) else np.zeros(0)
            holes = int((hd > px_tol).sum()) if hd is not None else None
            f[key] = {"spill_max_px": float(spill.max()) if spill is not None and len(spill) else 0.0,
                      "holes_px": holes, "holes_frac": None if holes is None else holes / max(int(inside.sum()), 1)}
            out["spill_max_px"][key] = max(out["spill_max_px"].get(key, 0.0), f[key]["spill_max_px"])
            out["holes_px_total"][key] = out["holes_px_total"].get(key, 0) + (holes or 0)
            out["holes_frac_max"][key] = max(out["holes_frac_max"].get(key, 0.0), f[key]["holes_frac"] or 0.0)
        mat = "exr" if "exr" in masks else "png"
        for k in twin_keys:
            if mat in masks and k in masks:
                inter = (masks[mat] & masks[k]).sum(); union = (masks[mat] | masks[k]).sum()
                f[f"iou_{mat}_{k}"] = float(inter / max(union, 1)); ious.append(f[f"iou_{mat}_{k}"])
        out["frames"].append(f); out["frames_checked"] += 1
    out["material_twin_iou_min"] = float(min(ious)) if ious else None
    gated = [k for k in out["spill_max_px"]]
    out["pass_spill"] = bool(gated) and all(out["spill_max_px"][k] <= px_tol for k in gated)
    hole_keys = (["exr"] if "exr" in out["holes_px_total"] else []) + twin_keys
    out["pass_holes"] = all(out["holes_px_total"].get(k, 0) == 0 for k in hole_keys)
    out["pass"] = bool(out["frames_checked"]) and out["pass_spill"] and out["pass_holes"] \
        and all(k in out["spill_max_px"] for k in ["png"] + twin_keys)
    return out


def check_joint_sources(seq_dir, data_root):
    """B3: per-frame camera == B2 source and light == B1 source, exactly, from the files alone."""
    seq_dir, data_root = Path(seq_dir), Path(data_root)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    tr = meta["trajectory"]
    if tr.get("moving_element") != "camera+light":
        raise ValueError(f"{seq_dir} is not a joint (camera+light) sequence")
    load = lambda p: sorted((json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()), key=lambda r: r["index"])
    recs = load(seq_dir / "frames.jsonl")
    lsrc = load(data_root / tr["light_path"] / "frames.jsonl")
    csrc = load(data_root / tr["camera_path"] / "frames.jsonl")
    res = {"sequence": str(seq_dir), "light_path": tr["light_path"], "camera_path": tr["camera_path"],
           "frames": len(recs), "frames_in_sources": [len(lsrc), len(csrc)], "mismatches": []}
    by_l = {r["index"]: r for r in lsrc}; by_c = {r["index"]: r for r in csrc}
    for r in recs:
        i = r["index"]; l, c = by_l.get(i), by_c.get(i)
        if l is None or c is None:
            res["mismatches"].append({"index": i, "missing_in_source": True}); continue
        for k in ("position", "c2w", "theta_deg", "phi_deg"):
            if r["camera"][k] != c["camera"][k]:
                res["mismatches"].append({"index": i, "field": f"camera.{k}"})
        for k in ("position", "theta_deg", "phi_deg", "distance"):
            if r["light"][k] != l["light"][k]:
                res["mismatches"].append({"index": i, "field": f"light.{k}"})
    res["pass"] = not res["mismatches"] and len(recs) == len(lsrc) == len(csrc)
    return res


def boundary_distance(mask):
    """Distance (px) of every pixel to the nearest boundary pixel of ``mask`` (Chebyshev-ish via scipy if present)."""
    try:
        from scipy import ndimage
        edge = mask ^ ndimage.binary_erosion(mask)
        return ndimage.distance_transform_edt(~edge)
    except Exception:
        return None


def check_sequence(seq_dir, px_tol=1.0, psnr_min=40.0, frames=None, edge_px=1.0, outline_px=0.75):
    seq_dir = Path(seq_dir)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    recs = [json.loads(l) for l in (seq_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
    if frames is not None:
        recs = [r for r in recs if r["index"] in frames]
    is_white = meta["mode"] == "white_lambert"
    out = {"sequence": str(seq_dir), "mode": meta["mode"], "frames_checked": [], "silhouette_max_px": 0.0,
           "silhouette_mismatch_frac_max": 0.0, "irradiance_psnr_db_min": None, "irradiance_rel_err_median_max": 0.0}
    psnrs = []
    for rec in recs:
        dv = derive(meta, rec)
        f = {"index": rec["index"], "inside_pixels": int(dv["inside"].sum()),
             "E_max": float(dv["E"].max()), "wi_theta_deg_range": None}
        th = np.degrees(np.arccos(np.clip(dv["cos_i"][dv["inside"]], 0, 1)))
        if th.size:
            f["wi_theta_deg_range"] = [float(th.min()), float(th.max())]
        if is_white:
            img = load_exr(seq_dir / rec["files"]["exr"]).mean(-1)
            rendered_mask = img > 0.0
            mism = rendered_mask ^ dv["inside"]
            dist = boundary_distance(dv["inside"])
            f["silhouette_mismatch_frac"] = float(mism.mean())
            f["silhouette_max_px"] = float(dist[mism].max()) if (dist is not None and mism.any()) else 0.0
            my, mx = np.nonzero(mism)
            od = outline_distance_px(meta, rec, my, mx) if len(my) else np.zeros(0)
            f["outline_max_px"] = float(od.max()) if od is not None and len(od) else (0.0 if od is not None else None)
            if f["outline_max_px"] is not None:
                out["outline_max_px"] = max(out.get("outline_max_px", 0.0), f["outline_max_px"])
            f["corners_px"] = corners_px(meta, rec)
            f["edge_offset_px"] = edge_offsets_px(meta, rec, img, dv)
            if f["edge_offset_px"] is not None:
                out["edge_offset_px_max"] = max(out.get("edge_offset_px_max", 0.0), max(abs(x) for x in f["edge_offset_px"]))
            pred = dv["E"] / math.pi
            m = dv["inside"] & rendered_mask
            err = (img[m] - pred[m])
            mse = float((err ** 2).mean()); peak = float(pred[m].max())
            f["irradiance_psnr_db"] = 10 * math.log10(peak ** 2 / mse) if mse > 0 else float("inf")
            try:   # interior-only PSNR (3 px erosion) separates label errors from edge-filter effects
                from scipy import ndimage
                mi_ = ndimage.binary_erosion(m, iterations=3)
                e2 = img[mi_] - pred[mi_]; mse2 = float((e2 ** 2).mean())
                f["irradiance_psnr_db_interior"] = 10 * math.log10(peak ** 2 / mse2) if mse2 > 0 else float("inf")
            except Exception:
                f["irradiance_psnr_db_interior"] = None
            f["irradiance_rel_err_median"] = float(np.median(np.abs(err) / np.maximum(pred[m], 1e-6)))
            f["rendered_over_pred_mean_ratio"] = float(img[m].mean() / pred[m].mean())
            psnrs.append(f["irradiance_psnr_db"])
            out["silhouette_max_px"] = max(out["silhouette_max_px"], f["silhouette_max_px"])
            out["silhouette_mismatch_frac_max"] = max(out["silhouette_mismatch_frac_max"], f["silhouette_mismatch_frac"])
            out["irradiance_rel_err_median_max"] = max(out["irradiance_rel_err_median_max"], f["irradiance_rel_err_median"])
        out["frames_checked"].append(f)
    if psnrs:
        out["irradiance_psnr_db_min"] = float(min(psnrs))
        interior = [f.get("irradiance_psnr_db_interior") for f in out["frames_checked"] if f.get("irradiance_psnr_db_interior") is not None]
        out["irradiance_psnr_db_interior_min"] = float(min(interior)) if interior else None
        # A binary pixel-centre mask vs an anti-aliased render differs by partially covered boundary
        # pixels (<= 1 px, sqrt(2) on diagonals); the edge also caps the full-mask PSNR (~35-40 dB).
        out["pass_silhouette"] = out.get("outline_max_px") is not None and out["outline_max_px"] <= outline_px
        out["pass_irradiance"] = (out["irradiance_psnr_db_interior_min"] if interior else out["irradiance_psnr_db_min"]) >= psnr_min
        out["pass_edge_offset"] = out.get("edge_offset_px_max") is not None and out["edge_offset_px_max"] < edge_px
        out["pass"] = out["pass_silhouette"] and out["pass_irradiance"] and out["pass_edge_offset"]
    else:
        out["pass"] = None; out["note"] = "derivability only (not a white_lambert sequence)"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence_dir")
    ap.add_argument("--px", type=float, default=1.5, help="unused (kept for old command lines): the pixel-centre silhouette metric is reported, not gated")
    ap.add_argument("--outline-px", type=float, default=0.75, help="max distance of a mismatched pixel centre to the projected outline (box filter: sqrt(2)/2)")
    ap.add_argument("--psnr", type=float, default=40.0)
    ap.add_argument("--edge-px", type=float, default=1.0, help="max |sub-pixel edge offset| (check 2b)")
    ap.add_argument("--frames", default=None, help="comma list of frame indices to check (default all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--align", action="store_true", help="material + twin PNGs vs the analytic silhouette (check 4)")
    ap.add_argument("--align-px", type=float, default=2.05, help="gaussian filter radius (2 px) + margin")
    ap.add_argument("--sources", default=None, help="joint sequences: the Robocloth_synthetic_sequence dir (check 5)")
    a = ap.parse_args()
    frames = None if a.frames is None else {int(x) for x in a.frames.split(",")}
    res = check_sequence(a.sequence_dir, a.px, a.psnr, frames, a.edge_px, a.outline_px)
    if a.align:
        res["alignment"] = check_alignment(a.sequence_dir, a.align_px, frames)
    if a.sources:
        res["sources"] = check_joint_sources(a.sequence_dir, a.sources)
    s = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(s)
