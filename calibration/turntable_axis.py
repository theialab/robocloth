#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turntable axis & center estimation (COLMAP world ↔ robot base), NO global delta.

- Angles are expected CCW (right-hand rule). If your log is CW, pass --flip_cw.
- Alternating refinement:
    1) Umeyama Sim(3) from W -> expected centers (given n, p)
    2) Solve p from stacked (I - R_i) p = s R W_i + t - R_i C_i
    3) Small LM step on the axis n (unit vector on S^2)
- Optional polish: Sim(3) frozen; optimize (n, p) only with robust loss.

Outputs: dict(s, R, t, n, p, rmse, Wmap, Cexp) and printed metrics.
"""

import os
import re
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from numpy.linalg import svd, norm
from scipy.optimize import least_squares

# -------------------- COLMAP model I/O (from this repository) --------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECONSTRUCTION_DIR = _REPO_ROOT / "reconstruction"
if str(_RECONSTRUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(_RECONSTRUCTION_DIR))
from read_write_model import read_model, qvec2rotmat

# -------------------- Calibration constants (yours) --------------------
R_CAMERA2GRIPPER = np.array(
    [[ 6.28318646e-05,  9.99760635e-01,  2.18784947e-02],
     [-1.66959884e-04,  2.18785049e-02, -9.99760623e-01],
     [-9.99999984e-01,  5.91639931e-05,  1.68294587e-04]], dtype=float
)
t_CAMERA2GRIPPER = np.array([3.28324263e-2, 9.41618540e-3, 4.63816395e-02], dtype=float)  # meters

# ================================================================
# Math / Utility
# ================================================================

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T

def project_to_SO3(M: np.ndarray) -> np.ndarray:
    U,_,Vt = svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    return R

def rodrigues_axis_angle(n: np.ndarray, angle_rad: float | np.ndarray) -> np.ndarray:
    """
    Right-handed CCW rotation about axis n by angle_rad (rad).
    Supports scalar or vector of angles → returns (..,3,3).
    """
    n = np.asarray(n, float)
    n = n / (norm(n) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=float)
    I = np.eye(3)
    angle = np.asarray(angle_rad, float)[..., None, None]
    Sa = np.sin(angle); Ca = np.cos(angle)
    return I + Sa * K + (1.0 - Ca) * (K @ K)

def umeyama_sim3(P, Q, with_scale=True):
    """
    P (N,3) → Q (N,3): find s, R, t s.t. Q ≈ s R P + t
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q
    Sigma = (Q0.T @ P0) / len(P)
    U, D, VT = svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT
    if with_scale:
        var = (P0**2).sum() / len(P)
        s = np.trace(np.diag(D) @ S) / (var + 1e-15)
    else:
        s = 1.0
    t = mu_Q - s * (R @ mu_P)
    return s, R, t

def tangent_update_on_sphere(n: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Update unit vector n by tiny 2D tangent step u (R^2) and renormalize.
    """
    n = n / (norm(n) + 1e-15)
    a = np.array([1,0,0], float) if abs(n[0]) < 0.9 else np.array([0,1,0], float)
    e1 = np.cross(n, a); e1 /= (norm(e1) + 1e-15)
    e2 = np.cross(n, e1); e2 /= (norm(e2) + 1e-15)
    n2 = n + u[0]*e1 + u[1]*e2
    return n2 / (norm(n2) + 1e-15)

# ================================================================
# I/O glue (COLMAP ↔ log)
# ================================================================

def parse_colmap_images_txt(images):
    """
    images (read_model output) → dict[name] = 4x4 c2w
    """
    cam_c2w_dict = {}
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = c2w
    return cam_c2w_dict

def find_matching_entry(fname, scan_log):
    """
    Expect filenames like ...scan-<id>....*  → match scan_log entry with e['id']==<id>
    """
    base = fname.replace(".png", "").replace(".jpg", "")
    m = re.search(r'scan-(\d+)', base)
    if not m:
        raise ValueError(f"No 'scan-<id>' in filename: {fname}")
    scan_id = int(m.group(1))
    for i, e in enumerate(scan_log):
        if int(e["id"]) == scan_id:
            return i
    raise ValueError(f"No match for scan_id={scan_id}")

def collect_W_C_theta(scan_log_path, images, R_c2g, t_c2g, flip_angles_cw_to_ccw: bool):
    """
    Returns:
      W: (N,3)  COLMAP camera centers (world)
      C: (N,3)  camera centers from robot (base) for same captures
      theta_deg: (N,) angle per capture (CCW, degrees)
      entries: list of matched scan_log entries
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    colmap_c2w = parse_colmap_images_txt(images)

    W, C, theta, entries = [], [], [], []
    for fname, c2w in colmap_c2w.items():
        if "theta" not in fname:
            continue
        idx = find_matching_entry(fname, scan_log)
        e = scan_log[idx]

        # COLMAP camera center (world)
        W.append(c2w[:3, 3])

        # camera->base from robot (no undo)
        R_g2b = np.asarray(e["rotation_matrix"], float)
        t_g2b = np.asarray(e["position"], float) / 1000.0  # mm → m
        T_g2b = build_4x4(R_g2b, t_g2b)
        T_c2g = build_4x4(R_c2g, t_c2g)
        T_c2b = T_g2b @ T_c2g
        C.append(T_c2b[:3, 3])

        # angle → enforce CCW once here
        th = float(e.get("turn_angle", 0.0))
        theta.append(-th if flip_angles_cw_to_ccw else th)
        entries.append(e)

    return (np.asarray(W, float),
            np.asarray(C, float),
            np.asarray(theta, float),
            entries)

# ================================================================
# Estimator (alternating + optional polish) — NO DELTA
# ================================================================

@dataclass
class FitConfig:
    huber: float = 0.0005          # robust (meters); ~1 mm
    tikhonov_p: float = 1e-9       # tiny reg for center LS
    iters_alt: int = 40
    lr_axis: float = 0.1
    do_polish: bool = True
    polish_max_nfev: int = 250
    dtype: type = np.float64
    freeze_sim3_in_polish: bool = True  # polish only (n, p)

class TurntableAxisEstimator:
    def __init__(self, cfg: FitConfig = FitConfig()):
        self.cfg = cfg

    def _expected_centers(self, C_n, n, p, alphas_rad):
        """
        C_n: (N,3); alphas_rad: (N,)
        Returns: (N,3): R_i C_i + (I - R_i) p
        """
        Rn  = rodrigues_axis_angle(n, alphas_rad)     # (N,3,3)
        RC  = (Rn @ C_n[..., None])[..., 0]           # (N,3)
        trans = ((np.eye(3) - Rn) @ p)                # (N,3)
        return RC + trans

    def _rmse(self, A, B):
        r = A - B
        return float(np.sqrt(np.mean(np.sum(r*r, axis=-1))))

    def fit_alternating(self, W_n, C_n, alphas_deg, init=None):
        """
        Inputs:
          W_n: (N,3), C_n: (N,3), alphas_deg: (N,) CCW degrees
        Returns dict with params and metrics.
        """
        dtype = self.cfg.dtype
        W = np.asarray(W_n, dtype=dtype)
        C = np.asarray(C_n, dtype=dtype)
        ang_rad = np.deg2rad(np.asarray(alphas_deg, dtype=dtype))
        N = W.shape[0]

        # init
        if init is None: init = {}
        n = init.get("n", np.array([0.,0.,1.], dtype=dtype))
        p = init.get("p", W.mean(0))

        # initial Sim(3) ignoring rotation model
        s, R, t = umeyama_sim3(W, C, with_scale=True)
        huber = self.cfg.huber

        for _ in range(self.cfg.iters_alt):
            # 1) Update (s,R,t) to match expected centers
            Cexp = self._expected_centers(C, n, p, ang_rad)
            s, R, t = umeyama_sim3(W, Cexp, with_scale=True)

            # 2) Update center p by linear LS over stacked (I - R_i) p = s R W_i + t - R_i C_i
            Rn = rodrigues_axis_angle(n, ang_rad)
            A_stack, b_stack = [], []
            for i in range(N):
                Ri = Rn[i]
                Ai = np.eye(3) - Ri
                bi = s * (R @ W[i]) + t - (Ri @ C[i])
                A_stack.append(Ai); b_stack.append(bi)
            A = np.concatenate(A_stack, axis=0)
            B = np.concatenate(b_stack, axis=0)
            lam = self.cfg.tikhonov_p
            p = np.linalg.solve(A.T @ A + lam*np.eye(3), A.T @ B)

            # 3) Small update for axis n only
            def local_res(u2):
                n2 = tangent_update_on_sphere(n, self.cfg.lr_axis * u2)
                Cexp2 = self._expected_centers(C, n2, p, ang_rad)
                Wmap = (s * (W @ R.T)) + t
                r = (Wmap - Cexp2)
                rr = np.linalg.norm(r, axis=1)
                w = np.ones_like(rr)
                m = rr > huber
                w[m] = huber / (rr[m] + 1e-15)
                return (w[:,None] * r).ravel()

            sol = least_squares(local_res, np.zeros(2, dtype=dtype), method="lm", max_nfev=25)
            n = tangent_update_on_sphere(n, self.cfg.lr_axis * sol.x)

        # metrics
        Wmap = (s * (W @ R.T)) + t
        Cexp = self._expected_centers(C, n, p, ang_rad)
        rmse = self._rmse(Wmap, Cexp)

        return {
            "mode": "alternating-flat",
            "s": s, "R": project_to_SO3(R), "t": t,
            "n": n / (norm(n) + 1e-15),
            "p": p,
            "rmse": float(rmse),
            "Wmap": Wmap, "Cexp": Cexp,
        }

    def polish_joint_lm(self, W_n, C_n, alphas_deg, seed):
        """
        Final LM polish on (n, p) with Sim(3) frozen.
        """
        assert self.cfg.freeze_sim3_in_polish, "Polish assumes Sim(3) is frozen."
        dtype = self.cfg.dtype
        W = np.asarray(W_n, dtype=dtype)
        C = np.asarray(C_n, dtype=dtype)
        ang_rad0 = np.deg2rad(np.asarray(alphas_deg, dtype=dtype))
        huber = self.cfg.huber

        s0, R0, t0 = seed["s"], seed["R"], seed["t"]
        n0, p0 = seed["n"], seed["p"]

        # parameters: x = [p(3), n_tangent(2)]
        x0 = np.zeros(5, dtype=dtype)
        x0[:3] = p0

        def unpack(x):
            p = x[:3]
            u = x[3:5]
            n = tangent_update_on_sphere(n0, u)
            return p, n

        def residuals(x):
            p, n = unpack(x)
            Cexp = self._expected_centers(C, n, p, ang_rad0)
            Wmap = (s0 * (W @ R0.T)) + t0
            r = (Wmap - Cexp)
            rr = np.linalg.norm(r, axis=1)
            w = np.ones_like(rr)
            m = rr > huber
            w[m] = huber / (rr[m] + 1e-15)
            return (w[:,None] * r).ravel()

        sol = least_squares(residuals, x0, method="lm",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=self.cfg.polish_max_nfev)
        p, n = unpack(sol.x)
        Cexp = self._expected_centers(C, n, p, ang_rad0)
        Wmap = (s0 * (W @ R0.T)) + t0
        rmse = self._rmse(Wmap, Cexp)

        return {
            "mode": "polished-flat",
            "success": sol.success,
            "nfev": sol.nfev,
            "s": s0, "R": project_to_SO3(R0), "t": t0,
            "n": n / (norm(n) + 1e-15),
            "p": p,
            "rmse": float(rmse),
            "Wmap": Wmap, "Cexp": Cexp,
        }

    def fit(self, W_n, C_n, alphas_deg, do_polish=True, init=None):
        alt = self.fit_alternating(W_n, C_n, alphas_deg, init=init)
        if do_polish and self.cfg.do_polish:
            pol = self.polish_joint_lm(W_n, C_n, alphas_deg, seed=alt)
            return {"alternating": alt, "polished": pol}
        return {"alternating": alt}

# ================================================================
# Reporting / Debug
# ================================================================

def summarize_metrics(result):
    r = np.linalg.norm(result["Wmap"] - result["Cexp"], axis=1)
    print(f"Mode: {result['mode']}")
    print(f"  rmse (m):     {result['rmse']:.6e}")
    print(f"  axis n:       {result['n']}")
    print(f"  center p (m): {result['p']}")
    print(f"  scale s:      {result['s']:.9f}")
    print(f"  trans t (m):  {result['t']}")
    print(f"  residual |Wmap-Cexp| stats (m): min {r.min():.3e}, mean {r.mean():.3e}, max {r.max():.3e}")

def debug_rmse_with_center_z(W_n, C_n, alphas_deg, center_xyz, angles_are_ccw=True):
    """
    Debug: assume axis is +Z and center is provided; compute Umeyama RMSE.
    """
    W = np.asarray(W_n, float)
    C = np.asarray(C_n, float)
    p = np.asarray(center_xyz, float).reshape(3)

    a_deg = np.asarray(alphas_deg, float).reshape(-1)
    if not angles_are_ccw:
        a_deg = -a_deg
    ang = np.deg2rad(a_deg)

    def Rz(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)

    I = np.eye(3)
    R_all = np.stack([Rz(t) for t in ang], axis=0)
    Cexp  = (R_all @ C[..., None])[..., 0] + ((I - R_all) @ p)

    s, R, t = umeyama_sim3(W, Cexp, with_scale=True)
    Wmap    = (s * (W @ R.T)) + t

    resid = Wmap - Cexp
    errs  = np.linalg.norm(resid, axis=1)
    rmse  = float(np.sqrt(np.mean(errs**2)))
    stats = {"min": float(errs.min()), "mean": float(errs.mean()), "max": float(errs.max())}
    return {"rmse": rmse, "s": s, "R": project_to_SO3(R), "t": t, "Wmap": Wmap, "Cexp": Cexp, "residual_stats": stats}

def find_best_t_c2g_by_rmse(
    scan_log_path,
    images,
    R_c2g,
    t_c2g_init,
    flip_angles_cw_to_ccw=False,
    range_mm=3.0,     # ± window per axis (meters = mm/1000)
    steps=7,          # grid points per axis (steps^3 fits)
    fit_cfg=None,     # optional FitConfig to reuse across runs
):
    """
    Brute-force grid search over x,y,z of t_c2g (camera->gripper translation)
    to minimize final RMSE of the axis/center fit.

    Returns:
      best_t_c2g : (3,) meters
      best_out   : dict from estimator.fit (contains 'alternating' and maybe 'polished')
      tried      : list of (t_try (3,), rmse_float)
    """
    # Estimator config (delta-free model)
    if fit_cfg is None:
        fit_cfg = FitConfig(
            huber=0.001, iters_alt=40, do_polish=True, polish_max_nfev=250,
            freeze_sim3_in_polish=True
        )
    est = TurntableAxisEstimator(fit_cfg)

    # Build the grid (meters)
    offsets = np.linspace(-range_mm/1000.0, range_mm/1000.0, steps)
    best_rmse = np.inf
    best_t = t_c2g_init.copy()
    best_out = None
    tried = []

    total = len(offsets)**3
    c = 0
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                c += 1
                t_try = t_c2g_init + np.array([dx, dy, dz], float)

                # Assemble W, C, θ with this t_c2g
                W, C, thetas_deg, _ = collect_W_C_theta(
                    scan_log_path, images, R_c2g, t_try,
                    flip_angles_cw_to_ccw=flip_angles_cw_to_ccw
                )

                # Fit and score
                out = est.fit(W, C, thetas_deg, do_polish=True)
                rmse = (out["polished"]["rmse"]
                        if "polished" in out else out["alternating"]["rmse"])
                tried.append((t_try.copy(), float(rmse)))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_t = t_try.copy()
                    best_out = out

                # (optional) tiny progress ping
                if c % max(1, total // 10) == 0:
                    print(f"[t_c2g grid] {c}/{total}  current best RMSE: {best_rmse:.6e}")

    print(f"[t_c2g grid] BEST t_c2g (m): {best_t}  BEST RMSE (m): {best_rmse:.6e}")
    return best_t, best_out, tried

def evaluate_custom_center(
    scan_log_path,
    images,
    R_c2g,
    t_c2g,
    center_xyz,                  # (3,) meters, in BASE frame
    axis=np.array([0, 0, 1.0]),  # default +Z
    flip_angles_cw_to_ccw=False
):
    """
    Use a PROVIDED center and axis to compute expected centers and the
    optimal Sim(3) from W -> expected, then report RMSE.

    Returns: dict(rmse, s, R, t, residual_stats, Wmap, Cexp)
    """
    # 1) Assemble data
    W, C, thetas_deg, _ = collect_W_C_theta(
        scan_log_path, images, R_c2g, t_c2g,
        flip_angles_cw_to_ccw=flip_angles_cw_to_ccw
    )
    W = np.asarray(W, float)
    C = np.asarray(C, float)
    p = np.asarray(center_xyz, float).reshape(3)

    # 2) Build expected centers using provided axis + center
    ang = np.deg2rad(np.asarray(thetas_deg, float).reshape(-1))  # (N,)
    R_all = rodrigues_axis_angle(axis, ang)                      # (N,3,3)
    Cexp  = (R_all @ C[..., None])[..., 0] + ((np.eye(3) - R_all) @ p)

    # 3) Umeyama: W -> Cexp
    s, R, t = umeyama_sim3(W, Cexp, with_scale=True)
    Wmap    = (s * (W @ R.T)) + t
    # Transform the axis from colmap space to world space using Wmap transformation
    colmap_center = np.array([-0.01678844, -0.77130875, 2.25983005])
    colmap_axis = np.array([-0.01307955, 0.29506558, -0.95538748])
    
    # Transform center: apply same transformation as W -> Wmap
    world_center = (s * (colmap_center @ R.T)) + t
    
    # Transform axis: only apply rotation (no translation or scaling for direction vectors)
    world_axis = colmap_axis @ R.T
    world_axis = world_axis / (np.linalg.norm(world_axis) + 1e-15)  # normalize
    
    print(f"[Axis Transform] Colmap center: {colmap_center}")
    print(f"[Axis Transform] World center:  {world_center}")
    print(f"[Axis Transform] Colmap axis:   {colmap_axis}")
    print(f"[Axis Transform] World axis:    {world_axis}")
    # 4) Metrics
    resid = Wmap - Cexp
    errs  = np.linalg.norm(resid, axis=1)
    rmse  = float(np.sqrt(np.mean(errs**2)))
    stats = {"min": float(errs.min()), "mean": float(errs.mean()), "max": float(errs.max())}

    print("=== Custom-axis/center evaluation ===")
    print(f"  axis n:       {axis / (norm(axis) + 1e-15)}")
    print(f"  center p (m): {p}")
    print(f"  rmse (m):     {rmse:.6e}")
    print(f"  residual |Wmap-Cexp| stats (m): min {stats['min']:.3e}, "
          f"mean {stats['mean']:.3e}, max {stats['max']:.3e}")

    return {
        "rmse": rmse,
        "s": s, "R": project_to_SO3(R), "t": t,
        "Wmap": Wmap, "Cexp": Cexp,
        "residual_stats": stats
    }

def estimate_world2base(
    scan_log_path,
    images,
    flip_angles_cw_to_ccw=False,
    do_t_c2g_grid=True,
    grid_range_mm=3.0,
    grid_steps=7,
    custom_center_xyz=None,       # <- if provided, evaluate this center (+Z axis) after choosing t_c2g
):
    """
    1) (optional) Grid-search around t_CAMERA2GRIPPER to minimize RMSE.
    2) Re-fit once with the best t_c2g and print clean metrics.
    3) (optional) Evaluate a PROVIDED center (axis +Z) and print its Umeyama RMSE.

    Returns:
      {
        "best_t_c2g": (3,) meters,
        "fit":        dict from estimator.fit on the best_t_c2g,
        "tried":      list of (t_try, rmse) from the grid (if do_t_c2g_grid),
        "custom_eval": dict from evaluate_custom_center_z(...) or None
      }
    """
    # 1) Optional grid search for hand–eye translation
    if do_t_c2g_grid:
        best_t_c2g, _, tried = find_best_t_c2g_by_rmse(
            scan_log_path, images,
            R_CAMERA2GRIPPER, t_CAMERA2GRIPPER,
            flip_angles_cw_to_ccw=flip_angles_cw_to_ccw,
            range_mm=grid_range_mm,
            steps=grid_steps,
            fit_cfg=FitConfig(
                huber=0.001, iters_alt=40, do_polish=True, polish_max_nfev=250,
                freeze_sim3_in_polish=True
            ),
        )
    else:
        best_t_c2g = t_CAMERA2GRIPPER.copy()
        tried = []
        print("No grid search performed")

    # 2) Re-run once with the chosen t_c2g for clean reporting (full estimator, free n and p)
    W, C, thetas_deg, _ = collect_W_C_theta(
        scan_log_path, images, R_CAMERA2GRIPPER, best_t_c2g,
        flip_angles_cw_to_ccw=flip_angles_cw_to_ccw
    )
    cfg = FitConfig(
        huber=0.001, iters_alt=40, do_polish=True, polish_max_nfev=300,
        freeze_sim3_in_polish=True
    )
    est = TurntableAxisEstimator(cfg)
    out = est.fit(W, C, thetas_deg, do_polish=True)

    print("=== Alternating result (best t_c2g) ===")
    summarize_metrics(out["alternating"])
    if "polished" in out:
        print("=== Polished result (best t_c2g) ===")
        summarize_metrics(out["polished"])

    # 3) Optional: evaluate a provided center with axis fixed to +Z
    custom_eval = None
    custom_center_xyz = np.array([0.1662, -0.224, 3.4938], float)
    if custom_center_xyz is not None:
        custom_eval = evaluate_custom_center(
            scan_log_path=scan_log_path,
            images=images,
            R_c2g=R_CAMERA2GRIPPER,
            t_c2g=best_t_c2g,
            center_xyz=np.asarray(custom_center_xyz, float),
            axis=np.array([0.0048, 0.00126, 0.99998], float),
            flip_angles_cw_to_ccw=flip_angles_cw_to_ccw
        )

    return {"best_t_c2g": best_t_c2g, "fit": out, "tried": tried, "custom_eval": custom_eval}



def main(scan_log_path, model_path, flip_angles_cw_to_ccw=False, do_t_c2g_grid=False):
    cameras, images = read_model(model_path, ext=".bin")
    estimate_world2base(scan_log_path, images, flip_angles_cw_to_ccw=flip_angles_cw_to_ccw, do_t_c2g_grid=do_t_c2g_grid)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Axis/center estimation with CCW angles (no global delta).")
    parser.add_argument("--scan_log_path", type=str, required=True, help="Path to robot scan log JSON")
    parser.add_argument("--model_path", type=str, required=True, help="Path to COLMAP sparse model dir")
    parser.add_argument("--flip_cw", action="store_true", help="If your logged angles are CW, flip to CCW once")
    parser.add_argument("--mesh_path")
    args = parser.parse_args()
    
    do_t_c2g_grid = False
    main(args.scan_log_path, args.model_path, flip_angles_cw_to_ccw=args.flip_cw, do_t_c2g_grid=do_t_c2g_grid)
