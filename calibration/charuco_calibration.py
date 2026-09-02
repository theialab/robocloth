# -*- coding: utf-8 -*-
"""
Turn‑table axis estimation
==========================

A self‑contained script that can work with **either** a classic checkerboard
or an OpenCV **Charuco** board.  Set `BOARD_TYPE` below to "checkerboard" or
"charuco" before running.

Pipeline
--------
1.  *Intrinsic calibration*   (Zhang for checkerboard / Charuco for Charuco)
2.  *Per–frame pose*          (`solvePnP` for the chosen board)
3.  *Axis fit*                (common eigen‑axis + least‑squares centre)

Only NumPy & OpenCV (≥ 4.7 for AprilTag/Charuco) are required.
"""

from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import json

# ──────────────────────── CONFIG ────────────────────────
DICT_NAME  = cv2.aruco.DICT_4X4_50
CU_COLS    = 11           # squares across (chessboard squares, not markers)
CU_ROWS    = 8           # squares down
CU_SQUARE  = 15       # square side length in mm (same units as the robot log / t_c2g)
CU_MARKER  = 11       # marker side length in mm
USE_LEGACY = True       # True if your PDF was generated with legacy pattern

# Input paths (calibration images, turntable scans, robot logs) are CLI
# arguments — see parse_args() at the bottom.

# Hand–eye (camera → gripper) result, mm, expressed for the **OpenGL** camera
# frame (x right, y up, z backward — the convention of
# configs/renderer/rig_constants.yaml and reconstruction/reconstruct.py).
# Override with --c2g_json; hand_eye.py --out writes the OpenCV convention and
# says so in its "camera_convention" field (see load_camera_to_gripper).
R_c2g = np.array([[-0.00369406,  0.99992083,  0.01202885],
                  [-0.00272167,  0.01201883, -0.99992407],
                  [-0.99998947, -0.00372652,  0.00267706]])
t_c2g = np.array([36.68630125, -24.61733549, 27.64501449])
BUILTIN_C2G_CONVENTION = "opengl"
CAMERA_CONVENTIONS = ("opencv", "opengl")

# ────────────────────── Helper functions ──────────────────────

def make_obj_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Generate (rows × cols) object points in the Z = 0 plane."""
    jj, ii = np.meshgrid(np.arange(rows), np.arange(cols), indexing="xy")
    pts = np.stack([ii.ravel() * square, jj.ravel() * square, np.zeros(cols * rows)], axis=1)
    return pts.astype(np.float32)


def rodrigues_to_Rt(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return R, t[:, 0], T


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 4×4 transform from R (3×3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# OpenCV camera frame (x right, y down, z forward: solvePnP, COLMAP, hand_eye.py)
# vs OpenGL camera frame (x right, y up, z backward: transforms.json,
# rig_constants.yaml, the built-in constants above) differ by a 180° rotation
# about the camera x axis. Right-multiplying a camera→X transform by T_CV2GL
# re-expresses it for the other camera frame (an involution); the camera
# centre, hence t_c2g, is the same in both.
T_CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def load_camera_to_gripper(c2g_json: str | None = None, convention: str | None = None) -> np.ndarray:
    """
    Camera→gripper transform T_c2g (4×4, log units) expressed for the **OpenCV**
    camera frame — the frame the solvePnP board poses live in, so that
    ``T_board2base = T_g2b @ T_c2g @ T_board2cam`` holds without further flips.

    c2g_json   : JSON with "R_c2g"/"t_c2g" (e.g. ``hand_eye.py --out``);
                 None → the built-in constants above.
    convention : "opencv" | "opengl" — the camera frame R_c2g is expressed in.
                 None → the file's "camera_convention" field (written by
                 hand_eye.py), else "opencv" for a JSON and "opengl" for the
                 built-ins.
    """
    if c2g_json:
        with open(c2g_json, "r") as f:
            d = json.load(f)
        R = np.asarray(d["R_c2g"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(d["t_c2g"], dtype=np.float64).reshape(3)
        if convention is None:
            convention = str(d.get("camera_convention", "opencv")).lower()
    else:
        R, t = R_c2g, t_c2g
        if convention is None:
            convention = BUILTIN_C2G_CONVENTION
    if convention not in CAMERA_CONVENTIONS:
        raise ValueError(f"camera convention must be one of {CAMERA_CONVENTIONS}, got {convention!r}")
    T = make_T(R, t)
    if convention == "opengl":
        T = T @ T_CV2GL          # OpenGL-frame camera→gripper → OpenCV-frame camera→gripper
    return T


def load_robot_pose_from_scan_log(scan_log_path: str) -> np.ndarray:
    """Load a single (base→gripper) pose from a scan_log JSON.

    This follows the same field usage as camera_calibration.load_matched_robot_poses:
    expects keys "rotation_matrix" (3×3) and "position" (3,).
    Returns a 4×4 transform. Uses the first valid entry.
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    if not scan_log:
        raise RuntimeError(f"No entries in {scan_log_path}")
    entry = scan_log[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(entry["rotation_matrix"], dtype=np.float64)
    T[:3, 3] = np.array(entry["position"], dtype=np.float64)
    return T

# ────────────────────── Axis estimation ──────────────────────
def estimate_axis_from_poses(R_list: List[np.ndarray], p_list: List[np.ndarray]):
    """Return axis direction d, point c, mean radius and RMS residual."""
    d_vecs = []
    R0 = R_list[0]
    for R in R_list[1:]:
        R_rel = R0.T @ R
        rvec, _ = cv2.Rodrigues(R_rel)
        ang = np.linalg.norm(rvec)
        if ang < 1e-6:
            continue
        axis = rvec[:, 0] / ang
        if d_vecs and np.dot(axis, d_vecs[0]) < 0:
            axis = -axis
        d_vecs.append(axis)
    d = np.mean(d_vecs, axis=0)
    d /= np.linalg.norm(d)
    # point on axis (least‑squares)
    P = np.eye(3) - np.outer(d, d)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p in p_list:
        A += P
        b += P @ p
    c = np.linalg.solve(A, b)
    radii = [np.linalg.norm(np.cross(p - c, d)) for p in p_list]
    residuals = [np.linalg.norm(P @ (p - c)) for p in p_list]
    return d, c, float(np.mean(radii)), float(np.sqrt(np.mean(np.square(residuals))))

# ------------ Build dictionary, board, and detector ------------
def make_charuco(detector_params: cv2.aruco.DetectorParameters = None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
    board = cv2.aruco.CharucoBoard((CU_COLS, CU_ROWS), CU_SQUARE, CU_MARKER, aruco_dict)
    if USE_LEGACY:
        board.setLegacyPattern(True)
    ch_params = cv2.aruco.CharucoParameters()
    if detector_params is None:
        detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charucoParams=ch_params, detectorParams=detector_params)
    return aruco_dict, board, detector

# ------------ Intrinsic calibration from ChArUco images ------------
def calibrate_intrinsics_charuco(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    _, board, detector = make_charuco()

    objpoints, imgpoints = [], []
    imsize = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imsize = img.shape[::-1]

        # Detect markers + interpolate ChArUco corners
        ch_corners, ch_ids, _, _ = detector.detectBoard(img)
        # Draw detected ChArUco corners for debugging
        # if ch_corners is not None and ch_ids is not None and len(ch_corners) > 0:
        #     debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     debug_img = cv2.aruco.drawDetectedCornersCharuco(debug_img, ch_corners, ch_ids)
        #     debug_path = f"debug_charuco_{Path(p).stem}.png"
        #     cv2.imwrite(debug_path, debug_img)
        if ch_ids is None or len(ch_ids) < 4:
            continue

        # Convert to matched 3D/2D points for calibration
        objPts, imgPts = board.matchImagePoints(ch_corners, ch_ids)
        if objPts is None or imgPts is None or len(objPts) == 0:
            continue

        objpoints.append(np.asarray(objPts, np.float32))
        imgpoints.append(np.asarray(imgPts, np.float32))

    if len(objpoints) < 8:
        raise RuntimeError("Need ≥ 8 views with valid ChArUco correspondences for calibration.")

    # Standard Zhang calibration
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=imsize,
        cameraMatrix=None,
        distCoeffs=None
    )
    return K, D, float(ret)

# ------------ Per-image pose from the same board ------------
# Returns (R, t, T_cam_board) with T 4x4 (camera->board)
def board_pose_charuco(img_bgr: np.ndarray, K: np.ndarray, D: np.ndarray):
    """
    Pose of the ChArUco board via matchImagePoints + solvePnP.
    Returns (R, t, T_cam_board) or None if detection fails.
    """
    # Reuse the same board/detector you used for intrinsics
    _, board, detector = make_charuco()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 4:
        return None

    # Map detected ChArUco corners to object/image points
    objPts, imgPts = board.matchImagePoints(ch_corners, ch_ids)
    if objPts is None or imgPts is None or len(objPts) < 4:
        return None

    # OpenCV solvePnP expects shapes (N,1,3) and (N,1,2), float32/float64
    obj = np.asarray(objPts, np.float32).reshape(-1, 1, 3)
    img = np.asarray(imgPts, np.float32).reshape(-1, 1, 2)

    # Robust default for many coplanar points
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:  
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    return R, t, T


# ------------ (Optional) debug overlay writer ------------
def save_charuco_debug(img_bgr: np.ndarray, out_path: str):
    _, _, detector = make_charuco()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    dbg = img_bgr.copy()
    if marker_corners is not None and marker_ids is not None and len(marker_corners) > 0:
        dbg = cv2.aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
    if ch_corners is not None and ch_ids is not None and len(ch_corners) > 0:
        dbg = cv2.aruco.drawDetectedCornersCharuco(dbg, ch_corners, ch_ids)
    cv2.imwrite(out_path, dbg)


def main(args: argparse.Namespace) -> None:
    calib_imgs = sorted(glob.glob(args.calib_glob))
    turn_imgs1 = sorted(glob.glob(args.top_glob)) if args.top_glob else []
    turn_imgs2 = sorted(glob.glob(args.tilt_glob)) if args.tilt_glob else []
    if not calib_imgs:
        sys.exit("No calibration images found – check --calib_glob pattern.")
    if not turn_imgs1 and not turn_imgs2:
        sys.exit("No turn‑table images found – check --top_glob/--tilt_glob patterns.")
    if turn_imgs1 and not args.scan_log_top:
        sys.exit("--scan_log_top is required with --top_glob.")
    if turn_imgs2 and not args.scan_log_tilt:
        sys.exit("--scan_log_tilt is required with --tilt_glob.")

    K, D, rms = calibrate_intrinsics_charuco(calib_imgs)

    print("\n=== Intrinsic calibration ===")
    print("K:\n", K)
    print("Distortion D:", D.ravel())
    print(f"Mean reprojection error: {rms:.3f} px\n")
    # camera→gripper for the OpenCV camera frame (the frame of the solvePnP board poses)
    T_c2g = load_camera_to_gripper(args.c2g_json, args.c2g_convention)
    T_g2c = np.linalg.inv(T_c2g)

    T_c2b_top  = load_robot_pose_from_scan_log(args.scan_log_top)  @ T_c2g if turn_imgs1 else None
    T_c2b_tilt = load_robot_pose_from_scan_log(args.scan_log_tilt) @ T_c2g if turn_imgs2 else None

    Rb_list: List[np.ndarray] = []
    pb_list: List[np.ndarray] = []

    # --- Process scans1 (top pose) ---
    for pth in turn_imgs1:
        img = cv2.imread(pth)
        if img is None:
            continue
        pose = board_pose_charuco(img, K, D)
        if pose is None:
            print(f"[warn] pose failed for {pth}")
            continue

        # board_pose_charuco returns board -> camera
        _, _, T_board2camera = pose

        # base <- board = (base <- camera) @ (camera <- board)
        T_board2b = T_c2b_top @ T_board2camera  
        Rb_list.append(T_board2b[:3, :3])
        pb_list.append(T_board2b[:3, 3])

    # --- Process scans2 (tilt pose) ---
    for pth in turn_imgs2:
        img = cv2.imread(pth)
        if img is None:
            continue
        pose = board_pose_charuco(img, K, D)
        if pose is None:
            print(f"[warn] pose failed for {pth}")
            continue

        # board -> camera
        _, _, T_board2camera = pose

        # base <- board
        T_board2b = T_c2b_tilt @ T_board2camera
        Rb_list.append(T_board2b[:3, :3])
        pb_list.append(T_board2b[:3, 3])

    if len(Rb_list) < 6:
        sys.exit("Need at least 6 valid board poses to fit an axis.")

    d, c, r_mean, rms_fit = estimate_axis_from_poses(Rb_list, pb_list)

    print("=== Axis result ===")
    print("Axis direction d (unit):", d)
    print("Axis point c (base):   ", c)
    print(f"Mean radius:           {r_mean:.4f} (log units, mm on our rig)")
    print(f"Fit RMS residual:      {rms_fit:.4f} (log units, mm on our rig)\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChArUco intrinsic calibration + turntable-axis estimate from board poses "
                    "seen by the robot-mounted camera at a fixed (top / tilt) gripper pose.")
    parser.add_argument("--calib_glob", type=str, required=True,
                        help="Glob of ChArUco images for intrinsic calibration, e.g. 'calib/*.png'")
    parser.add_argument("--top_glob", type=str, default=None,
                        help="Glob of turntable scans taken from the 'top' gripper pose")
    parser.add_argument("--tilt_glob", type=str, default=None,
                        help="Glob of turntable scans taken from the 'tilt' gripper pose")
    parser.add_argument("--scan_log_top", type=str, default=None,
                        help="Robot scan_log JSON of the 'top' pose (first entry is used); required with --top_glob")
    parser.add_argument("--scan_log_tilt", type=str, default=None,
                        help="Robot scan_log JSON of the 'tilt' pose (first entry is used); required with --tilt_glob")
    parser.add_argument("--c2g_json", type=str, default=None,
                        help="JSON with R_c2g/t_c2g (e.g. hand_eye.py --out) overriding the built-in hand–eye constants")
    parser.add_argument("--c2g_convention", choices=CAMERA_CONVENTIONS, default=None,
                        help="Camera-axis convention R_c2g is expressed in. Default: the JSON's 'camera_convention' "
                             "field (hand_eye.py --out writes 'opencv'), 'opencv' for a JSON without it, "
                             f"'{BUILTIN_C2G_CONVENTION}' for the built-in constants")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
