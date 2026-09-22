#!/usr/bin/env python3
"""G0 checks for an exp-015 sequence, using ONLY metadata.json + frames.jsonl (+ the rendered EXRs of a
``--mode white_lambert`` sequence). No Mitsuba import.

  1. derivability: per-pixel camera rays (o ⊕ d) in the sample frame, hit points on y = 0, incident
     direction ω_i, cos θ_i / r², analytic Lambertian irradiance E = I cos θ_i / r²;
  2. silhouette: analytic inside-sample mask vs rendered mask (EXR > 0): every mismatched pixel
     must lie within --px (default 1.5 = diagonal neighbour) of the analytic boundary;
  3. irradiance: analytic radiance E/π (albedo 1, direct light only) vs the rendered EXR; PSNR over the
     3-px-eroded interior must exceed --psnr dB (full-mask PSNR is also reported; edge pixels cap it).
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
            "uv": np.stack([(hit[..., 0] + he[0]) / (2 * he[0]), (hit[..., 2] + he[1]) / (2 * he[1])], -1)}


def boundary_distance(mask):
    """Distance (px) of every pixel to the nearest boundary pixel of ``mask`` (Chebyshev-ish via scipy if present)."""
    try:
        from scipy import ndimage
        edge = mask ^ ndimage.binary_erosion(mask)
        return ndimage.distance_transform_edt(~edge)
    except Exception:
        return None


def check_sequence(seq_dir, px_tol=1.0, psnr_min=40.0, frames=None):
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
        out["pass_silhouette"] = out["silhouette_max_px"] <= px_tol
        out["pass_irradiance"] = (out["irradiance_psnr_db_interior_min"] if interior else out["irradiance_psnr_db_min"]) >= psnr_min
        out["pass"] = out["pass_silhouette"] and out["pass_irradiance"]
    else:
        out["pass"] = None; out["note"] = "derivability only (not a white_lambert sequence)"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence_dir")
    ap.add_argument("--px", type=float, default=1.5, help="max boundary offset in px (sqrt(2) = diagonal neighbour of a partially covered pixel)")
    ap.add_argument("--psnr", type=float, default=40.0)
    ap.add_argument("--frames", default=None, help="comma list of frame indices to check (default all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frames = None if a.frames is None else {int(x) for x in a.frames.split(",")}
    res = check_sequence(a.sequence_dir, a.px, a.psnr, frames)
    s = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(s)
