#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand–eye calibration with metric upgrade.

*   Matches robot‐arm poses (scan_log.json) to camera poses (COLMAP
    ``images.txt`` or NeRF-style ``transforms.json``) via the φ/θ encoded in
    the image filenames
*   Runs OpenCV Tsai hand‐eye to obtain the camera-to-gripper rotation, then
    solves the camera-to-gripper translation **jointly with the metric scale**
    of COLMAP's world when the pose set makes that scale observable (the
    Umeyama 1991 centre fit initialises the scale and is the fallback)
*   Estimates the global **similarity** (scale + R + t) that maps COLMAP's world
    to the robot base

Conventions (shared with reconstruction/reconstruct.py)
-------------------------------------------------------
*   scan_log entries carry ``rotation_matrix`` / ``position`` = gripper→base
    (T_g2b). Positions are used in the log's own units (mm on our rig), so all
    translations returned here are in those units.
*   COLMAP ``images.txt`` stores WORLD→CAMERA (``qvec = [qw qx qy qz]``,
    ``tvec``); ``transforms.json`` stores CAMERA→WORLD in the OpenGL camera
    convention. Internally every camera pose is camera→world in the OpenCV
    camera convention.
*   Camera pose in the robot base: ``T_c2b = T_g2b @ T_c2g``.
*   The returned ``R_c2g`` therefore maps **OpenCV** camera-frame coordinates
    (x right, y down, z forward — the frame of COLMAP and ``cv2.solvePnP``) to
    the gripper. ``configs/renderer/rig_constants.yaml`` and the built-in
    constants of charuco_calibration.py express the same transform for the
    **OpenGL** camera frame (y up, z backward); the two differ by
    ``R_c2g_opengl = R_c2g_opencv @ diag(1, -1, -1)`` (``t_c2g`` is identical).
    ``--out`` records ``camera_convention`` and both matrices.

Usage
-----
    python calibration/hand_eye.py --scan_log_path scan_log.json \\
        --poses_path sparse/0/images.txt [--tol 1e-3] [--out hand_eye.json] \\
        [--scale_mode auto|joint|umeyama] [--c2g_json prior.json] \\
        [--mesh_path in.ply --mesh_out out.ply]
"""

import argparse
import json
import os
import warnings
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = np.asarray(t, dtype=float).reshape(3)
    return T

def quat_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix (pure NumPy)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])

def colmap_qvec_to_rotmat(qvec_wxyz):
    """
    Rotation matrix from a COLMAP quaternion ``[qw, qx, qy, qz]`` (scalar-FIRST,
    as written in images.txt and parsed by reconstruction/read_write_model.py).

    SciPy's ``Rotation.from_quat`` expects scalar-LAST ``[x, y, z, w]``, so the
    components are reordered explicitly here. Passing the COLMAP order straight
    into SciPy silently yields a different rotation.
    """
    qw, qx, qy, qz = np.asarray(qvec_wxyz, dtype=float).reshape(4)
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def rotmat_to_colmap_qvec(rotmat):
    """Inverse of :func:`colmap_qvec_to_rotmat`: 3x3 → ``[qw, qx, qy, qz]``."""
    x, y, z, w = R.from_matrix(np.asarray(rotmat, dtype=float)).as_quat()
    return np.array([w, x, y, z])

_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # flips the camera's y and z axes
CAMERA_CONVENTION = "opencv"                     # convention of every R_c2g produced by this module
CAMERA_CONVENTIONS = ("opencv", "opengl")

def flip_camera_convention_c2g(R_c2g):
    """
    Re-express a camera→gripper rotation for the other camera-axis convention
    (OpenCV ↔ OpenGL: 180° about the camera x axis, an involution). Only the
    camera's local axes change, so ``t_c2g`` (the camera centre in the gripper
    frame) is the same in both conventions. Use it to compare this module's
    output with configs/renderer/rig_constants.yaml / the charuco_calibration.py
    built-ins (OpenGL convention).
    """
    return np.asarray(R_c2g, dtype=float) @ _GL_CV_FLIP[:3, :3]

def _gl_to_cv(c2w_gl):
    """camera→world in the OpenGL camera convention (NeRF / iNGP transforms.json,
    camera looks down -z, y up) → OpenCV camera convention (COLMAP, calibrateHandEye).
    Right-multiplication only changes the camera's local axes; the camera centre
    is untouched."""
    return np.asarray(c2w_gl, dtype=float) @ _GL_CV_FLIP

def _cv_to_gl(c2w_cv):
    """OpenCV camera convention → OpenGL camera convention (same involution)."""
    return np.asarray(c2w_cv, dtype=float) @ _GL_CV_FLIP

def average_rotations(R_list):
    """Average rotations via quaternions (equal weights)."""
    qs = R.from_matrix(R_list).as_quat()          # (N,4)  [x y z w]
    # resolve antipodal symmetry so that dot(q_i,q_0) >= 0
    qs *= np.sign((qs * qs[0]).sum(-1, keepdims=True))
    q_mean = qs.mean(0)
    q_mean /= np.linalg.norm(q_mean)
    return R.from_quat(q_mean).as_matrix()

def official_umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot gripper positions
    Finds s, R, t such that  Q ≈ s · R @ P + t.

    Returns
    -------
    s : float, R : (3,3), t : (3,)   — **in this order** (same as
    ``umeyama_sim3`` in calibration/turntable_axis.py and reconstruction/reconstruct.py).
    """
    P, Q = np.asarray(P), np.asarray(Q)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q

    # equation 12 in Umeyama
    Sigma = Q0.T @ P0 / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT

    if with_scale:
        var_P = (P0**2).sum() / len(P)
        s = np.trace(np.diag(D) @ S) / var_P
    else:
        s = 1.0

    t = mu_Q - s * R @ mu_P
    return s, R, t


def camera_centres_from_c2w(c2w_list):
    """Return (N,3) array of camera centres from camera-to-world matrices."""
    return np.array([m[:3, 3] for m in c2w_list])

# -----------------------------------------------------------------------------#
#  Matching images ↔ robot log
# -----------------------------------------------------------------------------#
def extract_phi_theta_from_filename(filename):
    """Extract φ and θ from `scan-phi0.583_theta4.608.png`."""
    # Handle format: 'scan-0-phi0.079_theta3.142.png'
    base = (
        filename.replace(".png", "")
        .replace(".jpg", "")
        .replace("scan-", "")
    )
    if "-phi" in base and "_theta" in base:
        # Split on the phi part
        phi_part = base.split("-phi")[1]
        phi_theta_parts = phi_part.split("_theta")
        phi = float(phi_theta_parts[0])
        theta = float(phi_theta_parts[1])
        return phi, theta

    parts = base.split("_")
    phi = float([p for p in parts if p.startswith("phi")][0][3:]) # 3 if previous names
    theta = float([p for p in parts if p.startswith("theta")][0][5:]) # 5 if previous names
    return phi, theta


def find_matching_robot_pose(trg_phi, trg_theta, scan_log, tol=1e-3):
    """Return index in scan_log that best matches (φ,θ)."""
    best_i, best_d = None, float("inf")
    for i, e in enumerate(scan_log):
        d = max(abs(trg_phi - e["phi"]), abs(trg_theta - e["theta"]))
        if d < best_d:
            best_d, best_i = d, i
    return (best_i, best_d) if best_d <= tol else (None, best_d)


def _robot_pose_from_entry(entry):
    """scan_log entry → 4x4 gripper→base (positions kept in the log's units)."""
    return build_4x4(np.array(entry["rotation_matrix"], dtype=float),
                     np.array(entry["position"], dtype=float))


def _match_frames_to_scan_log(named_c2w, scan_log, tol):
    """
    named_c2w : iterable of (camera_idx, filename, 4x4 c2w in OpenCV convention)
    Returns robot_poses (list 4x4 g2b), cam_c2w (list 4x4), info (list dict),
    matched by the φ/θ encoded in the filename.
    """
    robot_poses, cam_c2w, info = [], [], []
    for cam_idx, fname, c2w in named_c2w:
        # Skip files that don't contain "theta" in the filename
        if "theta" not in fname:
            continue

        phi, theta = extract_phi_theta_from_filename(fname)
        idx, dist = find_matching_robot_pose(phi, theta, scan_log, tol)
        if idx is None:
            continue

        entry = scan_log[idx]
        cam_c2w.append(np.asarray(c2w, dtype=float))
        robot_poses.append(_robot_pose_from_entry(entry))   # gripper → base
        info.append(
            dict(camera_idx=cam_idx, robot_idx=idx, filename=fname,
                 target_phi=phi, target_theta=theta,
                 robot_phi=entry["phi"], robot_theta=entry["theta"],
                 distance=dist)
        )
    return robot_poses, cam_c2w, info


def load_matched_robot_poses(scan_log_path, transforms_json_path, tol=1e-3):
    """Match camera frames of a NeRF-style transforms.json to robot poses via φ/θ.

    ``transform_matrix`` is CAMERA→WORLD in the OpenGL camera convention; it is
    converted to the OpenCV convention used by the rest of this module."""
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    with open(transforms_json_path, "r") as f:
        tjson = json.load(f)

    named_c2w = [
        (i, frame["file_path"].removeprefix("images/"),
         _gl_to_cv(np.array(frame["transform_matrix"], dtype=float)))
        for i, frame in enumerate(tjson["frames"])
    ]
    robot_poses, cam_c2w, info = _match_frames_to_scan_log(named_c2w, scan_log, tol)
    print(f"Matched {len(robot_poses)}/{len(tjson['frames'])} frames.")
    return robot_poses, cam_c2w, info


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt and return a dictionary of image_name → 4x4 cam-to-world transform.

    COLMAP stores WORLD→CAMERA per image (``x_cam = R(qvec) @ x_world + tvec``,
    ``qvec = [qw qx qy qz]``); the pose is inverted here so the returned matrix
    is camera→world (camera centre ``= -Rᵀ t``), exactly as
    ``parse_colmap_images_txt`` in reconstruction/reconstruct.py does for the
    binary model.
    """
    cam_c2w_dict = {}
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        # Parse camera line
        tokens = line.split()
        if len(tokens) < 10:
            i += 1
            continue
        image_id = int(tokens[0])
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])
        cam_id = int(tokens[8])
        image_name = tokens[9]

        # world → camera, then invert to camera → world
        R_w2c = colmap_qvec_to_rotmat([qw, qx, qy, qz])
        t_w2c = np.array([tx, ty, tz])
        cam_c2w_dict[image_name] = build_4x4(R_w2c.T, -R_w2c.T @ t_w2c)
        i += 2  # skip the 2D point line
    return cam_c2w_dict


def load_matched_robot_poses_colmap_images_txt(scan_log_path, images_txt_path, tol=1e-3):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images_txt_path)

    named_c2w = [(None, fname, c2w) for fname, c2w in colmap_c2w_dict.items()]
    robot_poses, cam_c2w, info = _match_frames_to_scan_log(named_c2w, scan_log, tol)
    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, info


def load_matched_poses(scan_log_path, poses_path, tol=1e-3):
    """Dispatch on the camera-pose file type: ``*.json`` → NeRF transforms.json,
    ``*.txt`` → COLMAP images.txt."""
    ext = os.path.splitext(str(poses_path))[1].lower()
    if ext == ".json":
        return load_matched_robot_poses(scan_log_path, poses_path, tol)
    if ext == ".txt":
        return load_matched_robot_poses_colmap_images_txt(scan_log_path, poses_path, tol)
    raise ValueError(f"Unrecognised camera-pose file {poses_path!r}: "
                     "expected a transforms.json or a COLMAP images.txt")

# -----------------------------------------------------------------------------#
#  Hand–eye
# -----------------------------------------------------------------------------#
def hand_eye_calibration(robot_T, cam_w2c):
    """
    Tsai hand-eye calibration.

    Parameters
    ----------
    robot_T : list of 4×4   gripper → base
    cam_w2c : list of 4×4   world → camera (metric; "world" plays OpenCV's "target")

    Returns
    -------
    R_cam2grip (3,3), t_cam2grip (3,)
    """
    if len(robot_T) < 3 or len(robot_T) != len(cam_w2c):
        raise ValueError(f"Need ≥3 matched poses, got {len(robot_T)} robot / {len(cam_w2c)} camera")
    # OpenCV expects:
    #   R_gripper2base, t_gripper2base
    #   R_target2cam ,  t_target2cam
    Rg2b = np.array([T[:3, :3] for T in robot_T])
    tg2b = np.array([T[:3, 3]  for T in robot_T])
    Rw2c = np.array([T[:3, :3] for T in cam_w2c])
    tw2c = np.array([T[:3, 3]  for T in cam_w2c])

    R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
        Rg2b, tg2b, Rw2c, tw2c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_cam2grip, t_cam2grip.reshape(3)


FIXED_POINT_REL_TOL = 1e-3       # gripper poses share a fixed point if the LS residual is below this × rms|t_A|
SCALE_OBSERVABILITY_TOL = 1e-6   # secondary guard: σ_min/σ_max of the (t_X, s) system
SCALE_ATTENUATION_TOL = 1e-2     # max. estimated noise-to-signal ratio ρ of the joint scale (bias ≈ ρ/(1+ρ))
SCALE_MODES = ("auto", "joint", "umeyama")


def solve_translation_and_scale(robot_T, cam_w2c, R_c2g, s_fallback=None, mode="auto",
                                fixed_point_rel_tol=FIXED_POINT_REL_TOL,
                                rel_sv_tol=SCALE_OBSERVABILITY_TOL,
                                attenuation_tol=SCALE_ATTENUATION_TOL):
    """
    Exact metric upgrade: given the camera→gripper rotation, solve the
    camera→gripper translation and the scale ``s`` (log units per COLMAP unit)
    of the COLMAP world jointly by linear least squares.

    For every pose pair (i, j) the hand–eye equation ``A X = X B`` with
    ``A = T_g2b_j⁻¹ T_g2b_i`` (gripper motion) and ``B = T_w2c_j T_w2c_i⁻¹``
    (camera motion, translation in COLMAP units) gives

        (R_A − I) t_X − s · R_X t_B = −t_A ,

    which is linear in ``(t_X, s)``. Tsai's rotation step does not depend on
    the scale, but its translation step does — and the Umeyama fit of camera
    centres against *gripper* positions only approximates ``s`` because the
    centres are offset from the gripper by ``t_c2g``.

    **Degeneracy.** If every gripper pose keeps one gripper-frame point ``p_f``
    fixed in the base (a purely spherical scan — the camera orbiting a single
    focus point, as the rig's φ/θ pattern does), then
    ``(t_X, s) = (p_f + λ (t_X* − p_f), λ s*)`` fits exactly for every λ: the
    scale is unobservable from ``A X = X B``. Worse, when the robot poses are
    exactly spherical (positions generated analytically from φ/θ) while the
    COLMAP poses carry noise, ``(p_f, 0)`` fits with zero residual and the
    joint least squares *always* returns it. The degeneracy is therefore
    detected on the gripper poses alone (do they admit a common fixed point?),
    with the singular values of the joint system as a secondary guard.

    **Near-degeneracy.** Only the camera column ``m_s = −R_X t_B`` carries
    measurement noise (the robot poses are exact to repeatability), so the
    joint least squares is an errors-in-variables problem: its scale is
    *attenuated* towards zero by ``1/(1+ρ)``, ``ρ = |noise|²/|signal|²`` in the
    part ``m_⊥`` of ``m_s`` that the ``t_X`` columns cannot explain. On a nearly
    spherical scan ``m_⊥`` is tiny, ρ is large and the joint scale is badly
    biased long before the fixed-point test fires. ρ is estimated from the fit
    itself — ``ρ ≈ |r|² / (s² |m_⊥|²)`` with ``r`` the residual (exact to first
    order) — and the joint solution is accepted only if ``ρ ≤ attenuation_tol``
    (bias ≲ 1 %).

    Whenever the scale is judged unobservable, ``s`` is fixed to ``s_fallback``
    (the Umeyama centre fit) and only ``t_X`` is solved — which is Tsai's
    translation step at that scale, i.e. the original algorithm — with a
    warning.

    Parameters
    ----------
    robot_T    : list of 4×4   gripper → base
    cam_w2c    : list of 4×4   world → camera, translations in COLMAP units
    R_c2g      : (3,3)         camera → gripper rotation (e.g. from Tsai)
    s_fallback : float | None  scale used when the scale is unobservable / mode="umeyama"
                               (None → raise in that case)
    mode       : "auto" (joint solve if observable, else fallback), "joint"
                 (always joint), "umeyama" (always fixed to ``s_fallback``)
    fixed_point_rel_tol, rel_sv_tol, attenuation_tol : detection thresholds (see module constants)

    Returns
    -------
    t_c2g : (3,), s : float, info : dict with keys
        observable (bool), fixed_point_rel_residual, sv_ratio, scale_joint (the raw
        joint-LS scale, always computed), scale_attenuation_est (ρ),
        mode_used ("joint" | "umeyama")
    """
    if mode not in SCALE_MODES:
        raise ValueError(f"mode must be one of {SCALE_MODES}, got {mode!r}")
    robot_T = [np.asarray(T, dtype=float) for T in robot_T]
    cam_w2c = [np.asarray(T, dtype=float) for T in cam_w2c]
    n = len(robot_T)
    I3 = np.eye(3)
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(robot_T[j]) @ robot_T[i]
            B = cam_w2c[j] @ np.linalg.inv(cam_w2c[i])
            rows.append(np.hstack([A[:3, :3] - I3, -(R_c2g @ B[:3, 3])[:, None]]))
            rhs.append(-A[:3, 3])
    M = np.vstack(rows)          # [M_t | m_s]
    b = np.concatenate(rhs)      # = −t_A
    M_t, m_s = M[:, :3], M[:, 3]

    # --- observability 1: do the gripper motions alone admit a fixed point p (M_t p = b)? ---
    p_fix = np.linalg.lstsq(M_t, b, rcond=None)[0]
    fixed_rel = float(np.linalg.norm(M_t @ p_fix - b) / max(np.linalg.norm(b), 1e-300))
    # --- observability 2: conditioning of the joint system ---
    sv = np.linalg.svd(M, compute_uv=False)
    sv_ratio = float(sv[-1] / sv[0])
    # --- joint solve (always computed so the diagnostics can be reported) ---
    sol = np.linalg.lstsq(M, b, rcond=None)[0]
    t_joint, s_joint = sol[:3], float(sol[3])
    # --- observability 3: errors-in-variables attenuation of s_joint, ρ ≈ |r|² / (s² |m_⊥|²) ---
    m_perp = m_s - M_t @ np.linalg.lstsq(M_t, m_s, rcond=None)[0]
    resid = M @ sol - b
    signal = s_joint ** 2 * float(m_perp @ m_perp)
    attenuation = float(resid @ resid) / signal if signal > 0.0 else float("inf")
    degenerate = fixed_rel <= fixed_point_rel_tol or sv_ratio <= rel_sv_tol
    observable = not degenerate and attenuation <= attenuation_tol
    info = dict(observable=observable, fixed_point_rel_residual=fixed_rel, sv_ratio=sv_ratio,
                scale_joint=s_joint, scale_attenuation_est=attenuation)

    use_joint = mode == "joint" or (mode == "auto" and observable)
    if use_joint:
        info["mode_used"] = "joint"
        return t_joint, s_joint, info

    if degenerate:
        why = ("every gripper pose keeps one point fixed, i.e. a purely spherical scan (relative "
               f"fixed-point residual of the gripper motions {fixed_rel:.2e}, σ_min/σ_max {sv_ratio:.2e})")
    else:
        why = ("the pose set is nearly spherical or too noisy for the joint (t_c2g, s) solve: its scale "
               f"{s_joint:.6g} would be attenuated by an estimated {100.0 * attenuation / (1.0 + attenuation):.1f} % "
               f"(noise-to-signal ratio {attenuation:.2e} > {attenuation_tol:g})")
    if s_fallback is None:
        raise ValueError(f"Metric scale is unobservable from these poses — {why}; pass s_fallback")
    if mode == "auto":
        warnings.warn(
            f"hand_eye: metric scale is unobservable from AX=XB — {why}. Falling back to the Umeyama "
            f"centre-fit scale s = {s_fallback:.6g}, which is biased by the camera↔gripper lever arm; "
            "add poses at different focus distances to make the scale observable.",
            RuntimeWarning, stacklevel=2)
    # M = [M_t | m_s]; with s fixed:  M_t t = b − m_s·s
    t = np.linalg.lstsq(M_t, b - m_s * s_fallback, rcond=None)[0]
    info["mode_used"] = "umeyama"
    return t, float(s_fallback), info


def hand_eye_rms_residual(robot_T, cam_w2c_metric, R_c2g, t_c2g):
    """RMS of the translational ``A X − X B`` residual over consecutive pose pairs."""
    X = build_4x4(R_c2g, t_c2g)
    res = []
    for i in range(len(robot_T) - 1):
        A = np.linalg.inv(robot_T[i + 1]) @ robot_T[i]
        B = cam_w2c_metric[i + 1] @ np.linalg.inv(cam_w2c_metric[i])
        res.append((A @ X - X @ B)[:3, 3])
    return float(np.sqrt(np.mean(np.square(res)))) if res else 0.0

# -----------------------------------------------------------------------------#
#  Build world-to-camera transforms with metric scale only
# -----------------------------------------------------------------------------#
def build_scaled_w2c_using_scale_only(c2w_list, s):
    """
    Convert COLMAP camera-to-world poses (arbitrary scale) to
    *metric* world-to-camera poses.

    Only the uniform scale is applied: the hand–eye equation ``A X = X B`` is
    invariant to a rigid change of the world frame (it cancels in the relative
    camera motions ``B``), so the Umeyama rotation/translation are not needed.

    Parameters
    ----------
    c2w_list : list | (N,4,4) array
        Original COLMAP camera-to-world poses.
    s : float
        Uniform scale factor   (log units / COLMAP-unit).

    Returns
    -------
    list of (4,4) ndarray
        **world → camera** (a.k.a. W2C) matrices with metric translations.
    """
    w2c = []
    for C2W in c2w_list:
        C2W = np.asarray(C2W, dtype=float)
        C2W_metric = build_4x4(C2W[:3, :3], s * C2W[:3, 3])
        w2c.append(np.linalg.inv(C2W_metric))
    return w2c


def solve_hand_eye(robot_T, cam_c2w, scale_mode="auto"):
    """
    Full camera→gripper solve on matched pose lists.

    1. Umeyama fit of COLMAP camera centres against gripper positions → initial
       scale ``s0``  (approximate: centres are offset from the gripper).
    2. Tsai (``cv2.calibrateHandEye``) on the ``s0``-scaled poses → ``R_c2g``
       (Tsai's rotation is independent of the scale).
    3. Exact joint linear solve of ``(t_c2g, s)`` given ``R_c2g``; on a purely
       or nearly spherical scan the scale is unobservable and ``s0`` is kept (see
       :func:`solve_translation_and_scale`; ``scale_mode`` forces either branch).

    Returns dict with R_c2g (3,3) [OpenCV camera convention], t_c2g (3,), scale,
    scale_umeyama, scale_joint, scale_attenuation_est, scale_observable (bool),
    scale_mode_used, fixed_point_rel_residual, sv_ratio, n_poses, rms_residual.
    """
    if len(robot_T) < 3:
        raise ValueError(f"Need ≥3 matched poses for hand–eye, got {len(robot_T)}")
    cam_centres_world = camera_centres_from_c2w(cam_c2w)
    gripper_positions = np.array([T[:3, 3] for T in robot_T])

    s0, _, _ = sim3_umeyama(cam_centres_world, gripper_positions)

    cam_w2c_s0 = build_scaled_w2c_using_scale_only(cam_c2w, s0)
    R_c2g, _ = hand_eye_calibration(robot_T, cam_w2c_s0)

    cam_w2c_colmap = build_scaled_w2c_using_scale_only(cam_c2w, 1.0)
    t_c2g, s, info = solve_translation_and_scale(robot_T, cam_w2c_colmap, R_c2g,
                                                 s_fallback=s0, mode=scale_mode)

    rms = hand_eye_rms_residual(robot_T, build_scaled_w2c_using_scale_only(cam_c2w, s), R_c2g, t_c2g)
    return dict(R_c2g=R_c2g, t_c2g=t_c2g, scale=s, scale_umeyama=float(s0),
                scale_joint=info["scale_joint"], scale_attenuation_est=info["scale_attenuation_est"],
                scale_observable=bool(info["observable"]), scale_mode_used=info["mode_used"],
                fixed_point_rel_residual=info["fixed_point_rel_residual"], sv_ratio=info["sv_ratio"],
                n_poses=len(robot_T), rms_residual=rms)


def estimate_camera2gripper(scan_log_path, poses_path, phi_theta_tol=1e-3, scale_mode="auto"):
    """
    Estimate camera-to-gripper transformation using hand-eye calibration.

    Parameters
    ----------
    scan_log_path : str | Path
        Path to robot log with gripper→base poses
    poses_path : str | Path
        COLMAP ``images.txt`` or NeRF style ``transforms.json``
    phi_theta_tol : float, optional
        Tolerance for matching phi/theta angles (default: 1e-3)
    scale_mode : {"auto", "joint", "umeyama"}
        Metric-scale strategy, see :func:`solve_translation_and_scale` (default "auto")

    Returns
    -------
    R_c2g : (3,3) array
        Rotation matrix from camera to gripper (OpenCV camera convention)
    t_c2g : (3,) array
        Translation vector from camera to gripper (scan_log units)
    s : float
        Metric scale of the COLMAP world (scan_log units per COLMAP unit)
    """
    robot_T, cam_c2w, _ = load_matched_poses(scan_log_path, poses_path, phi_theta_tol)
    res = solve_hand_eye(robot_T, cam_c2w, scale_mode=scale_mode)
    return res["R_c2g"], res["t_c2g"], res["scale"]


def world_to_base_sim3(robot_T, cam_c2w, R_c2g, t_c2g):
    """
    Similarity COLMAP-world → robot-base from matched pose lists and a fixed
    camera→gripper transform. Returns (T_BW (4,4) = [[s·R, t],[0, 1]], s).
    """
    if len(robot_T) < 3 or len(robot_T) != len(cam_c2w):
        raise ValueError("Need ≥3 matched poses for the world→base similarity, got "
                         f"{len(robot_T)} robot / {len(cam_c2w)} camera")
    T_c2g = build_4x4(R_c2g, t_c2g)
    cam_centres_base, cam_centres_world = [], []
    for Tg2b, C2W in zip(robot_T, cam_c2w):
        Tc2b = Tg2b @ T_c2g                      # camera→base
        cam_centres_base .append(Tc2b[:3, 3])
        cam_centres_world.append(C2W[:3, 3])    # still arbitrary units

    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # ---------- similarity (scale,R,t)  world → base --------------------------
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)

    T_BW = np.eye(4)
    T_BW[:3, :3] = s * R
    T_BW[:3,  3] = t
    return T_BW, float(s)


def estimate_world2base(scan_log_path, poses_path,
                        R_c2g, t_c2g, phi_theta_tol=1e-3):
    """
    Parameters
    ----------
    scan_log_path : str | Path   (robot log with gripper→base poses)
    poses_path    : str | Path   (COLMAP ``images.txt`` or NeRF ``transforms.json``)
    R_c2g, t_c2g  : (3,3) and (3,)   fixed camera→gripper transform

    Returns
    -------
    T_BW : (4,4)  homogeneous matrix [  s·R   t ]
                                    [   0     1 ]
           that maps COLMAP‑world coordinates to robot‑base coordinates.
    """
    robot_T, cam_c2w, _ = load_matched_poses(scan_log_path, poses_path, phi_theta_tol)
    T_BW, _ = world_to_base_sim3(robot_T, cam_c2w, R_c2g, t_c2g)
    return T_BW


def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    """
    Load a mesh and transform it from COLMAP world coordinates to robot base coordinates.

    Parameters
    ----------
    mesh_path : str | Path
        Path to the input mesh file (e.g., .ply, .obj)
    T_BW : (4,4) array
        Transformation matrix from COLMAP world to robot base coordinates
    output_path : str | Path, optional
        Path to save the transformed mesh. If None, returns the transformed mesh object.

    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The transformed mesh object
    """
    import open3d as o3d

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")

    # Apply the transformation
    mesh.transform(T_BW)

    # Save if output path is provided
    if output_path is not None:
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not success:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")

    return mesh


# -----------------------------------------------------------------------------#
#  Main
# -----------------------------------------------------------------------------#
def load_c2g_json(path):
    """
    Read ``R_c2g`` (3×3) and ``t_c2g`` (3,) from a JSON file (e.g. a previous
    ``--out``) and return them in this module's OpenCV camera convention. A
    ``camera_convention`` field of ``"opengl"`` (rig_constants.yaml style) is
    converted; when the field is absent the file is assumed to be ``"opencv"``,
    which is what ``--out`` writes.
    """
    with open(path, "r") as f:
        d = json.load(f)
    R_c2g = np.asarray(d["R_c2g"], dtype=float).reshape(3, 3)
    t_c2g = np.asarray(d["t_c2g"], dtype=float).reshape(3)
    convention = str(d.get("camera_convention", CAMERA_CONVENTION)).lower()
    if convention not in CAMERA_CONVENTIONS:
        raise ValueError(f"{path}: camera_convention must be one of {CAMERA_CONVENTIONS}, got {convention!r}")
    if convention == "opengl":
        R_c2g = flip_camera_convention_c2g(R_c2g)
    return R_c2g, t_c2g


def main(args):
    robot_T, cam_c2w, _ = load_matched_poses(args.scan_log_path, args.poses_path, args.tol)
    if len(robot_T) < 3:
        raise SystemExit(
            f"hand_eye: only {len(robot_T)} camera frame(s) of {args.poses_path} matched an entry of "
            f"{args.scan_log_path} (need ≥ 3). Frames are matched to log entries through the φ/θ encoded in "
            "the image filename (scan-<id>-phi<φ>_theta<θ>.png|jpg, compared with the log's 'phi'/'theta' "
            f"fields within --tol = {args.tol:g} rad); check the filenames and the log, or loosen --tol.")

    result = dict(n_poses=len(robot_T), units="scan_log position units", camera_convention=CAMERA_CONVENTION,
                  note="R_c2g maps OpenCV camera-frame coordinates (x right, y down, z forward) to the gripper; "
                       "R_c2g_opengl is the same transform for the OpenGL camera frame "
                       "(configs/renderer/rig_constants.yaml, charuco_calibration.py built-ins)")
    if args.c2g_json:
        R_c2g, t_c2g = load_c2g_json(args.c2g_json)
        print(f"Using camera→gripper transform from {args.c2g_json}")
    else:
        he = solve_hand_eye(robot_T, cam_c2w, scale_mode=args.scale_mode)
        R_c2g, t_c2g = he["R_c2g"], he["t_c2g"]
        result.update(scale_world_to_base=he["scale"], scale_umeyama_init=he["scale_umeyama"],
                      scale_joint=he["scale_joint"], scale_attenuation_est=he["scale_attenuation_est"],
                      scale_observable=he["scale_observable"], scale_mode_used=he["scale_mode_used"],
                      fixed_point_rel_residual=he["fixed_point_rel_residual"], sv_ratio=he["sv_ratio"],
                      hand_eye_rms_residual=he["rms_residual"])
        print(f"Umeyama initial scale: {he['scale_umeyama']:.6g}   joint-LS scale: {he['scale_joint']:.6g} "
              f"(noise-to-signal {he['scale_attenuation_est']:.2e})   "
              f"metric scale used: {he['scale']:.6g} (mode {he['scale_mode_used']}; scale "
              f"{'observable' if he['scale_observable'] else 'UNOBSERVABLE from AX=XB'}, "
              f"fixed-point residual {he['fixed_point_rel_residual']:.2e})   "
              f"AX=XB translational RMS: {he['rms_residual']:.4g}")
    print("R_c2g (OpenCV camera convention):\n", R_c2g)
    print("t_c2g:", t_c2g)

    T_BW, s_bw = world_to_base_sim3(robot_T, cam_c2w, R_c2g, t_c2g)
    print(f"T_BW (COLMAP world → robot base, scale {s_bw:.6g}):\n", T_BW)
    result.update(R_c2g=R_c2g.tolist(), R_c2g_opengl=flip_camera_convention_c2g(R_c2g).tolist(),
                  t_c2g=t_c2g.tolist(), T_BW=T_BW.tolist(), scale_T_BW=s_bw)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {args.out}")

    if args.mesh_path:
        mesh_out = args.mesh_out or os.path.splitext(args.mesh_path)[0] + "_base.ply"
        transform_mesh_to_base(args.mesh_path, T_BW, output_path=mesh_out)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hand–eye calibration (camera→gripper, Tsai + metric upgrade) and "
                    "COLMAP-world→robot-base similarity from a robot scan_log and camera poses.")
    parser.add_argument("--scan_log_path", type=str, required=True,
                        help="Robot scan log JSON (entries with phi, theta, rotation_matrix, position)")
    parser.add_argument("--poses_path", type=str, required=True,
                        help="Camera poses: COLMAP sparse images.txt or NeRF-style transforms.json")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Max |Δφ|,|Δθ| (rad) for matching image filenames to log entries")
    parser.add_argument("--scale_mode", choices=SCALE_MODES, default="auto",
                        help="Metric scale of the COLMAP world: 'auto' = joint (t_c2g, s) solve when the poses make "
                             "it observable (non-spherical scan, noise-to-signal ratio of the scale <= "
                             f"{SCALE_ATTENUATION_TOL:g}), else Umeyama centre-fit fallback = the original "
                             "algorithm; 'joint' / 'umeyama' force one branch")
    parser.add_argument("--c2g_json", type=str, default=None,
                        help="Skip the hand–eye solve and use R_c2g/t_c2g from this JSON (e.g. a previous --out; "
                             "its 'camera_convention' field is honoured, default opencv)")
    parser.add_argument("--out", type=str, default=None,
                        help="Write R_c2g (OpenCV camera convention; R_c2g_opengl alongside), t_c2g, scale "
                             "diagnostics and T_BW to this JSON")
    parser.add_argument("--mesh_path", type=str, default=None, help="Optional mesh in COLMAP world to transform to the base frame")
    parser.add_argument("--mesh_out", type=str, default=None, help="Output mesh path (default: <mesh_path>_base.ply)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
