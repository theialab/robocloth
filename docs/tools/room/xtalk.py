#!/usr/bin/env python
"""How much realism does live per-object compositing lose? (crosstalk measurement)

Each page pass path-traces ONE piece in its material with every other cloth grey, so light
that the dressed pieces exchange (colour bleeding, secondary reflections) is missing from the
page's composite. This renders, at 960x540 (full frame, rows [0:450] = the shown region),
256 spp, the SAME seed and batching for every render (correlated sampling: paths that never
touch a changed surface are bit-identical):
  a      all cloth grey
  b      left sofa + its pillows in 453, the rest grey
  c      the fully dressed room (both sofas, both pillow pairs, 4 floor cushions; DRESS)
  obj_*  one piece at a time in its DRESS material, the rest grey   (-> the page composite d)
  lo_*   the same at 480x270 (for the first-order "low-res bleed" fix), + lo_a
Renders (noisy + OptiX-denoised EXR) and timings go to room_demo/variants/xtalk/renders/.

  export PYTHONNOUSERSITE=1; cd ~/projects/SGHyperMaterials
  CUDA_VISIBLE_DEVICES=1 $R docs/tools/room/xtalk.py render [names...]
  /data/colin/envs/teaser/bin/python docs/tools/room/xtalk.py analyse
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUTX = "/media/raid/cloth/output/teaser_video/room_demo/variants/xtalk"
RD = f"{OUTX}/renders"
HI = (960, 540)
LO = (480, 270)
SPP, SEED = 256, 11
BATCH = int(os.environ.get("XT_BATCH", "32"))
DRESS = {"sofaL": 50, "pillowsL": 145, "sofaR": 453, "pillowsR": 9,
         "cush58": 453, "cush2": 9, "cush56": 145, "cush57": 50}
PIECES = list(DRESS)


def scene_spec(name):
    """-> {region: material} for a render name."""
    if name in ("a", "lo_a"):
        return {}
    if name == "b":
        return {"sofaL": 453, "pillowsL": 453}
    if name in ("c", "c1920") or name.startswith("cseed"):   # cseed<N>: another seed (noise floor)
        return dict(DRESS)
    r = name.split("_")[-1]
    return {r: DRESS[r]}


def render_one(name):
    import xml.etree.ElementTree as ET
    import passes as P
    mi, dr = P.mi_init(neural=True)
    res = LO if name.startswith("lo_") else ((1920, 1080) if name == "c1920" else HI)
    spec = scene_spec(name)
    shared, sb = [], {}
    for k, (region, mid) in enumerate(spec.items()):      # one instance per region (own tiling)
        til = float(np.mean([P.tiling(s) for s in P.REGIONS[region]]))
        bid = f"nb{k}"
        shared.append(P.shared_neural(mid, til, bid))
        for sid in P.REGIONS[region]:
            sb[sid] = P.ref_bsdf(bid)
    xml = f"{RD}/_{name}.xml"
    P.build_xml(xml, sb)
    if shared:                                            # insert all shared BSDFs before the shapes
        tree = ET.parse(xml)
        root = tree.getroot()
        first = next(i for i, ch in enumerate(list(root)) if ch.tag == "shape")
        for b in reversed(shared):
            root.insert(first, b)
        tree.write(xml)
    scene = mi.load_file(xml, resx=str(res[0]), resy=str(res[1]))
    P.render(mi, dr, scene, 1, 1, seed=3)                 # warm-up, not timed
    t0 = time.time()
    seed = int(name[5:]) if name.startswith("cseed") else SEED
    img = np.array(P.render(mi, dr, scene, SPP, BATCH, seed=seed))[:, :, :3].astype(np.float32)
    t = time.time() - t0
    den = P.denoise(mi, img)
    cv2.imwrite(f"{RD}/{name}_noisy.exr", img[:, :, ::-1].copy())
    cv2.imwrite(f"{RD}/{name}.exr", den[:, :, ::-1].copy())
    p = f"{RD}/times.json"
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = dict(res=list(res), spp=SPP, instances=len(spec), render_s=round(t, 1),
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    json.dump(d, open(p, "w"), indent=1)
    print(name, d[name], flush=True)


# ------------------------------------------------------------------ analysis (teaser env)
def srgb8(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return s * 255.0


def rd(name, noisy=False):
    return cv2.imread(f"{RD}/{name}{'_noisy' if noisy else ''}.exr", cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)


def stats(d, m):
    v = d[m]
    return dict(mean=round(float(v.mean()), 2), p99=round(float(np.percentile(v, 99)), 1),
                max=round(float(v.max()), 1), frac_gt4=round(float((v > 4).mean()), 4),
                px_gt4=int((v > 4).sum()))


def pure(img, a, iters=6):
    out = img.copy()
    valid = a > 0.95
    band = (a > 0.004) & ~valid
    k = np.ones((3, 3), np.float32)
    for _ in range(iters):
        todo = band & ~valid
        if not todo.any():
            break
        num = cv2.filter2D(out * valid[:, :, None], -1, k, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(valid.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        fill = todo & (den > 0)
        out[fill] = num[fill] / den[fill][:, None]
        valid = valid | fill
    return out


def analyse():
    import passes as P
    H = 450
    names = list(P.REGIONS)
    cov = np.load(f"{P.OUT}/masks.npy")                                  # 1920x1080, 9 regions
    cov = cv2.resize(cov, HI, interpolation=cv2.INTER_AREA)[:H]
    ci = {r: cov[:, :, names.index(r)] for r in PIECES}
    a, b, c = rd("a")[:H], rd("b")[:H], rd("c")[:H]
    # (d) the page composite: base + per piece, the piece's own pass (interior colour pushed
    # over its antialiased band), blended by coverage -- as build_assets.py does
    tot = np.clip(sum(ci.values()), 0, 1)
    d = a * (1 - tot[:, :, None])
    for r in PIECES:
        d += pure(rd("obj_" + r)[:H], ci[r]) * ci[r][:, :, None]
    # (e) first-order fix: + each piece's low-res bleed onto the rest of the room. The low-res
    # difference (lo_obj_i - lo_a) is used only away from piece i (its own change would smear
    # over its silhouette), filled back in by normalized convolution, upsampled, and applied
    # outside piece i
    lo_a = rd("lo_a")
    e = d.copy()
    times = json.load(open(f"{RD}/times.json"))
    have_lo = all(f"lo_obj_{r}" in times for r in PIECES)
    cov_lo = cv2.resize(np.load(f"{P.OUT}/masks.npy"), LO, interpolation=cv2.INTER_AREA)
    if have_lo:
        # first-order, multiplicative: piece i changes the light arriving elsewhere by the
        # smooth ratio lo_obj_i / lo_a (irradiance ratio for diffuse-ish surfaces); applied to
        # everything outside piece i -- the grey room AND the other (dressed) pieces, whose own
        # albedo then scales the change correctly (an additive delta measured on grey would not)
        for r in PIECES:
            excl = cv2.dilate((cov_lo[:, :, names.index(r)] > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            w = (~excl).astype(np.float32)[:, :, None]
            num = cv2.GaussianBlur(rd("lo_obj_" + r) * w, (0, 0), 1.5)
            den = cv2.GaussianBlur(lo_a * w, (0, 0), 1.5)
            ratio = np.where(den > 1e-4, num / np.maximum(den, 1e-4), 1.0)
            ratio = np.clip(ratio, 0.5, 1.5)
            ratio = cv2.resize(ratio, (HI[0], HI[1]), interpolation=cv2.INTER_LINEAR)[:H]
            mi_ = ci[r][:, :, None]
            e *= ratio * (1 - mi_) + mi_
    e = np.maximum(e, 0)
    A, B, Cc, D, E = (srgb8(x) for x in (a, b, c, d, e))
    lp = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)          # crosstalk is smooth; noise is not
    m_obj = tot > 0.5
    m_ab = (ci["sofaL"] + ci["pillowsL"]) > 0.01
    m_ab = cv2.dilate(m_ab.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    D8 = lambda x, y: np.abs(x - y).max(2)
    out = {"setup": dict(res=list(HI), shown_rows=H, spp=SPP, seed=SEED, dress=DRESS,
                         metric="per-pixel max over RGB of |difference| in 8-bit sRGB DN, OptiX-denoised "
                                "renders; 'lp' = both images Gaussian-blurred sigma 1.5 px first")}
    ones = np.ones_like(m_obj)
    rug = cov[:, :, names.index("rug")] > 0.5
    other = (tot > 0.5) & ~m_ab
    rest = ~m_ab & ~rug & ~other
    for tag, x, y in (("raw", None, None), ("lp", lp, None)):
        f = (lambda z: z) if x is None else x
        Ab, Bb, Cb, Db, Eb = (f(z) for z in (A, B, Cc, D, E))
        db, dcd, dce = D8(Bb, Ab), D8(Cb, Db), D8(Cb, Eb)
        out[f"i_bleed_b_minus_a_outside_left_sofa_{tag}"] = stats(db, ~m_ab)
        out[f"i_bleed_where_{tag}"] = {"rug": stats(db, rug & ~m_ab), "other_pieces": stats(db, other),
                                       "wall_floor_rest": stats(db, rest)}
        out[f"ii_page_error_c_minus_d_{tag}"] = {"frame": stats(dcd, ones), "inside_pieces": stats(dcd, m_obj),
                                                 "outside_pieces": stats(dcd, ~m_obj)}
        if have_lo:
            out[f"fix_c_minus_e_{tag}"] = {"frame": stats(dce, ones), "inside_pieces": stats(dce, m_obj),
                                           "outside_pieces": stats(dce, ~m_obj)}
        if "cseed77" in times:
            Cn = srgb8(rd("cseed77")[:H])
            Cn = f(Cn)
            out[f"noise_floor_c_vs_c_other_seed_{tag}"] = {"frame": stats(D8(Cb, Cn), ones),
                                                           "inside_pieces": stats(D8(Cb, Cn), m_obj)}
    out["signed_mean_c_minus_d_per_piece_rgb"] = {r: (Cc - D)[ci[r] > 0.5].mean(0).round(2).tolist() for r in PIECES}
    out["signed_mean_c_minus_d_outside_rgb"] = (Cc - D)[tot < 0.01].mean(0).round(2).tolist()
    if have_lo:
        out["signed_mean_c_minus_e_per_piece_rgb"] = {r: (Cc - E)[ci[r] > 0.5].mean(0).round(2).tolist() for r in PIECES}
        out["signed_mean_c_minus_e_outside_rgb"] = (Cc - E)[tot < 0.01].mean(0).round(2).tolist()
    out["times"] = times
    json.dump(out, open(f"{OUTX}/xtalk.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "times"}))
    # images
    u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(f"{OUTX}/a_grey.png", u8(A))
    cv2.imwrite(f"{OUTX}/b_left_sofa_453.png", u8(B))
    cv2.imwrite(f"{OUTX}/c_full_render.png", u8(Cc))
    cv2.imwrite(f"{OUTX}/d_page_composite.png", u8(D))
    cv2.imwrite(f"{OUTX}/bleed_b_minus_a_x8.png", u8(np.abs(B - A) * 8))
    cv2.imwrite(f"{OUTX}/err_c_minus_d_lowpass_x8.png", u8(np.abs(lp(Cc) - lp(D)) * 8))
    if have_lo:
        cv2.imwrite(f"{OUTX}/err_c_minus_e_lowpass_x8.png", u8(np.abs(lp(Cc) - lp(E)) * 8))
    cv2.imwrite(f"{OUTX}/err_c_minus_d_x8.png", u8(np.abs(Cc - D) * 8))
    if have_lo:
        cv2.imwrite(f"{OUTX}/e_composite_with_lowres_bleed.png", u8(E))
        cv2.imwrite(f"{OUTX}/err_c_minus_e_x8.png", u8(np.abs(Cc - E) * 8))
    # side by side crops: c | d | |c-d| x8 (| e | |c-e| x8)
    crops = {"left_sofa": (40, 120, 460, 300), "cushions_right_sofa": (420, 130, 960, 420)}
    for nm, (x0, y0, x1, y1) in crops.items():
        row = [Cc, D, np.abs(Cc - D) * 8] + ([E, np.abs(Cc - E) * 8] if have_lo else [])
        tiles = [np.ascontiguousarray(u8(t[y0:y1, x0:x1])) for t in row]
        for t, lab in zip(tiles, ["c render", "d page", "|c-d| x8", "e +bleed", "|c-e| x8"]):
            cv2.putText(t, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(f"{OUTX}/crop_{nm}.png", np.concatenate(tiles, 1))


if __name__ == "__main__":
    os.makedirs(RD, exist_ok=True)
    if sys.argv[1] == "render":
        for n in sys.argv[2:]:
            render_one(n)
    else:
        analyse()
