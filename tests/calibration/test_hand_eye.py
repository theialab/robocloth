# -*- coding: utf-8 -*-
"""
Numerical tests for calibration/hand_eye.py and the camera→gripper handling of
calibration/charuco_calibration.py (NumPy + SciPy + OpenCV only).

Run from the repository root:
    python -m unittest discover -s tests/calibration -v

Fixtures (scan_log JSON, COLMAP images.txt, transforms.json — a few KB) are
written to a temporary directory under $ROBOCLOTH_TEST_TMP if that is set, else
under the system temp dir, and removed afterwards unless
ROBOCLOTH_KEEP_FIXTURES=1. Nothing is written inside the repository.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO / "calibration"
sys.path.insert(0, str(CALIB_DIR))
import hand_eye as he  # noqa: E402
import charuco_calibration as cc  # noqa: E402

try:  # reference COLMAP quaternion convention used by the reconstruction pipeline
    sys.path.insert(0, str(REPO / "reconstruction"))
    from read_write_model import qvec2rotmat as ref_qvec2rotmat  # noqa: E402
except Exception:  # pragma: no cover - parity check is skipped if unavailable
    ref_qvec2rotmat = None

TOL = 1e-6          # required by the release checklist; observed errors are ~1e-13


def make_tmpdir(test_case):
    base = os.environ.get("ROBOCLOTH_TEST_TMP") or tempfile.gettempdir()
    if not os.path.isdir(base):
        base = None
    d = tempfile.mkdtemp(prefix="hand_eye_test_", dir=base)
    if not os.environ.get("ROBOCLOTH_KEEP_FIXTURES"):
        test_case.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return d


def random_rotations(n, seed):
    return Rot.random(n, random_state=np.random.default_rng(seed)).as_matrix()


# ----------------------------------------------------------------------------- #
#  Synthetic rig
# ----------------------------------------------------------------------------- #
def synthetic_rig(n=25, seed=0, scale_world=1.0 / 237.0, t_c2g=(36.7, -24.6, 27.6)):
    """
    Sample N gripper→base poses, a known camera→gripper transform and a known
    Sim(3) base→world (COLMAP world = scaled/rotated/shifted robot base).
    Returns everything needed to write fixtures in the formats hand_eye.py parses.
    """
    rng = np.random.default_rng(seed)
    R_all = random_rotations(n + 2, seed)
    T_c2g = he.build_4x4(R_all[0], np.asarray(t_c2g, float))
    robot_T = [he.build_4x4(R_all[2 + i], rng.uniform(-300, 300, 3) + [600.0, 0.0, 300.0])
               for i in range(n)]
    R_wb, t_wb = R_all[1], rng.uniform(-2, 2, 3)
    T_WB = np.eye(4)                       # base → world similarity
    T_WB[:3, :3] = scale_world * R_wb
    T_WB[:3, 3] = t_wb

    cam_c2w = []
    for Tg in robot_T:
        Tc2b = Tg @ T_c2g
        cam_c2w.append(he.build_4x4(R_wb @ Tc2b[:3, :3],
                                    scale_world * (R_wb @ Tc2b[:3, 3]) + t_wb))

    phis = 0.05 * np.arange(n)
    thetas = 0.1 * np.arange(n) + 0.3
    names = [f"scan-{i}-phi{phis[i]:.3f}_theta{thetas[i]:.3f}.jpg" for i in range(n)]
    scan_log = [dict(id=i, phi=float(phis[i]), theta=float(thetas[i]), euler=[0.0, 0.0, 0.0],
                     position=robot_T[i][:3, 3].tolist(),
                     rotation_matrix=robot_T[i][:3, :3].tolist()) for i in range(n)]
    return dict(T_c2g=T_c2g, robot_T=robot_T, cam_c2w=cam_c2w, T_WB=T_WB,
                T_BW=np.linalg.inv(T_WB), scale_world=scale_world, names=names,
                scan_log=scan_log, n=n)


def _f(x):
    return format(float(x), ".17g")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_fixtures(rig, out_dir, shuffle_seed=3):
    """scan_log.json + COLMAP-style images.txt + NeRF-style transforms.json."""
    out_dir = Path(out_dir)
    scan_log_path = out_dir / "scan_log.json"
    with open(scan_log_path, "w") as f:
        json.dump(rig["scan_log"], f)

    order = np.random.default_rng(shuffle_seed).permutation(rig["n"])   # file order ≠ log order
    images_txt = out_dir / "images.txt"
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {rig['n']}, mean observations per image: 2\n")
        for k, i in enumerate(order):
            w2c = np.linalg.inv(rig["cam_c2w"][i])
            q = he.rotmat_to_colmap_qvec(w2c[:3, :3])
            t = w2c[:3, 3]
            f.write(f"{k + 1} {_f(q[0])} {_f(q[1])} {_f(q[2])} {_f(q[3])} "
                    f"{_f(t[0])} {_f(t[1])} {_f(t[2])} 1 {rig['names'][i]}\n")
            f.write("100.5 200.25 -1 300.0 400.0 7\n")

    frames = [dict(file_path="images/" + rig["names"][i],
                   transform_matrix=he._cv_to_gl(rig["cam_c2w"][i]).tolist())
              for i in order]
    frames.append(dict(file_path="images/calib_extra.jpg",           # must be skipped (no theta)
                       transform_matrix=np.eye(4).tolist()))
    frames.append(dict(file_path="images/scan-999-phi9.000_theta9.000.jpg",   # no log match
                       transform_matrix=np.eye(4).tolist()))
    transforms = out_dir / "transforms.json"
    with open(transforms, "w") as f:
        json.dump(dict(fl_x=1000.0, fl_y=1000.0, frames=frames), f)
    return str(scan_log_path), str(images_txt), str(transforms)


def rotation_error(Ra, Rb):
    return float(np.abs(np.asarray(Ra) - np.asarray(Rb)).max())


# ----------------------------------------------------------------------------- #
#  (1) quaternion order
# ----------------------------------------------------------------------------- #
class TestQuaternionOrder(unittest.TestCase):
    def test_colmap_wxyz_helper_roundtrip_and_reference_parity(self):
        Rs = random_rotations(200, seed=11)
        for R_true in Rs:
            q_wxyz = he.rotmat_to_colmap_qvec(R_true)                 # COLMAP order [w x y z]
            self.assertAlmostEqual(np.linalg.norm(q_wxyz), 1.0, places=12)
            R_back = he.colmap_qvec_to_rotmat(q_wxyz)
            self.assertLess(rotation_error(R_back, R_true), 1e-12)
            # pure-NumPy [w,x,y,z] formula in the same module agrees
            self.assertLess(rotation_error(he.quat_to_rot_matrix(q_wxyz), R_true), 1e-12)
            if ref_qvec2rotmat is not None:                           # reconstruction pipeline parity
                self.assertLess(rotation_error(ref_qvec2rotmat(q_wxyz), R_true), 1e-12)

    def test_swapped_order_is_wrong(self):
        """Feeding the COLMAP order straight into SciPy (the old bug) gives a different rotation."""
        Rs = random_rotations(200, seed=12)
        for R_true in Rs:
            q_wxyz = he.rotmat_to_colmap_qvec(R_true)
            R_bug = Rot.from_quat(q_wxyz).as_matrix()                 # interprets [w x y z] as [x y z w]
            self.assertGreater(rotation_error(R_bug, R_true), 1e-3)
            # and the helper is order-sensitive: handing it scalar-last input also fails
            self.assertGreater(rotation_error(he.colmap_qvec_to_rotmat(np.roll(q_wxyz, -1)), R_true), 1e-3)
        # identity is the textbook case: [1,0,0,0] misread as [x y z w] is a 180° flip about x
        self.assertLess(rotation_error(he.colmap_qvec_to_rotmat([1, 0, 0, 0]), np.eye(3)), 1e-15)
        self.assertLess(rotation_error(Rot.from_quat([1, 0, 0, 0]).as_matrix(), np.diag([1, -1, -1])), 1e-15)


# ----------------------------------------------------------------------------- #
#  (2) w2c ↔ c2w
# ----------------------------------------------------------------------------- #
class TestWorldToCameraHandling(unittest.TestCase):
    def test_parse_colmap_images_txt_inverts_world_to_camera(self):
        tmp = make_tmpdir(self)
        Rs = random_rotations(5, seed=21)
        rng = np.random.default_rng(22)
        truth = {}
        path = Path(tmp) / "images.txt"
        with open(path, "w") as f:
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\n")
            for k, R_w2c in enumerate(Rs):
                t_w2c = rng.normal(size=3) * 3
                name = f"scan-{k}-phi0.{k:03d}_theta1.000.png"
                truth[name] = (R_w2c, t_w2c)
                q = he.rotmat_to_colmap_qvec(R_w2c)
                f.write(f"{k + 1} {_f(q[0])} {_f(q[1])} {_f(q[2])} {_f(q[3])} "
                        f"{_f(t_w2c[0])} {_f(t_w2c[1])} {_f(t_w2c[2])} 1 {name}\n")
                f.write("1.0 2.0 -1\n" if k % 2 else "\n")            # points line may be empty
        c2w = he.parse_colmap_images_txt(str(path))
        self.assertEqual(set(c2w), set(truth))
        for name, (R_w2c, t_w2c) in truth.items():
            T = c2w[name]
            centre = -R_w2c.T @ t_w2c
            self.assertLess(np.abs(T[:3, 3] - centre).max(), 1e-12)
            self.assertLess(rotation_error(T[:3, :3], R_w2c.T), 1e-12)
            w2c = he.build_4x4(R_w2c, t_w2c)
            self.assertLess(np.abs(T @ w2c - np.eye(4)).max(), 1e-12)
            self.assertLess(np.abs(np.linalg.inv(T) - w2c).max(), 1e-12)
            self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0, places=12)

    def test_transforms_json_loader_is_camera_to_world_in_cv_convention(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=6, seed=31)
        scan_log, _, transforms = write_fixtures(rig, tmp)
        robot_T, cam_c2w, info = he.load_matched_robot_poses(scan_log, transforms, tol=1e-3)
        self.assertEqual(len(robot_T), rig["n"])                     # extra frames skipped
        self.assertEqual(len(info), rig["n"])
        for Tg, C2W, inf in zip(robot_T, cam_c2w, info):
            i = rig["names"].index(inf["filename"])
            self.assertLess(np.abs(Tg - rig["robot_T"][i]).max(), 1e-12)
            # camera centre untouched by the GL→CV flip, axes y/z flipped
            self.assertLess(np.abs(C2W[:3, 3] - rig["cam_c2w"][i][:3, 3]).max(), 1e-12)
            self.assertLess(rotation_error(C2W[:3, :3], rig["cam_c2w"][i][:3, :3]), 1e-12)
            gl = np.array(load_json(transforms)["frames"][0]["transform_matrix"])
            self.assertLess(np.abs(he._gl_to_cv(gl)[:3, :3] - gl[:3, :3] @ np.diag([1, -1, -1])).max(), 1e-15)

    def test_build_scaled_w2c_returns_world_to_camera(self):
        rig = synthetic_rig(n=4, seed=41)
        s = 237.0
        w2c = he.build_scaled_w2c_using_scale_only(rig["cam_c2w"], s)
        self.assertEqual(len(w2c), rig["n"])
        for W2C, C2W in zip(w2c, rig["cam_c2w"]):
            C2W_metric = he.build_4x4(C2W[:3, :3], s * C2W[:3, 3])
            self.assertLess(np.abs(W2C @ C2W_metric - np.eye(4)).max(), 1e-9)
            self.assertLess(np.abs(W2C[:3, 3] + W2C[:3, :3] @ (s * C2W[:3, 3])).max(), 1e-9)


# ----------------------------------------------------------------------------- #
#  (3) Umeyama return order
# ----------------------------------------------------------------------------- #
class TestUmeyama(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.P = rng.normal(size=(50, 3)) * 2.0
        self.s = 3.7
        self.R = random_rotations(1, seed=52)[0]
        self.t = np.array([0.5, -2.0, 11.0])
        self.Q = self.s * self.P @ self.R.T + self.t

    def test_sim3_umeyama_returns_s_R_t(self):
        out = he.sim3_umeyama(self.P, self.Q)
        self.assertEqual(len(out), 3)
        s, R, t = out                                   # explicit order: scale, rotation, translation
        self.assertTrue(np.isscalar(s) or np.ndim(s) == 0)
        self.assertEqual(np.shape(R), (3, 3))
        self.assertEqual(np.shape(t), (3,))
        self.assertLess(abs(float(s) - self.s), 1e-9)
        self.assertLess(rotation_error(R, self.R), 1e-9)
        self.assertLess(np.abs(t - self.t).max(), 1e-9)
        s1, R1, t1 = he.sim3_umeyama(self.P, self.P @ self.R.T + self.t, with_scale=False)
        self.assertEqual(s1, 1.0)
        self.assertLess(rotation_error(R1, self.R), 1e-9)
        self.assertLess(np.abs(t1 - self.t).max(), 1e-9)

    def test_official_umeyama_returns_c_R_t_column_convention(self):
        c, R, t = he.official_umeyama(self.P.T, self.Q.T)   # (3, N) inputs
        self.assertTrue(np.isscalar(c) or np.ndim(c) == 0)
        self.assertEqual(np.shape(R), (3, 3))
        self.assertEqual(np.shape(t), (3, 1))
        self.assertLess(abs(float(c) - self.s), 1e-9)
        self.assertLess(rotation_error(R, self.R), 1e-9)
        self.assertLess(np.abs(t.ravel() - self.t).max(), 1e-9)

    def test_umeyama_with_reflection_guard(self):
        """Degenerate planar data must still yield a proper rotation (det +1)."""
        P = self.P.copy(); P[:, 2] = 0.0
        Q = self.s * P @ self.R.T + self.t
        s, R, t = he.sim3_umeyama(P, Q)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)
        self.assertLess(np.abs(s * P @ R.T + t - Q).max(), 1e-9)


# ----------------------------------------------------------------------------- #
#  (4) end-to-end hand–eye on a synthetic rig
# ----------------------------------------------------------------------------- #
class TestHandEyeEndToEnd(unittest.TestCase):
    def _check_solution(self, rig, R_c2g, t_c2g, s):
        self.assertLess(rotation_error(R_c2g, rig["T_c2g"][:3, :3]), TOL)
        self.assertLess(np.abs(np.asarray(t_c2g).reshape(3) - rig["T_c2g"][:3, 3]).max(), TOL)
        self.assertLess(abs(s * rig["scale_world"] - 1.0), TOL)      # log units per COLMAP unit

    def test_solve_hand_eye_arrays(self):
        for scale_world, t_c2g in [(1.0, (36.7, -24.6, 27.6)), (1.0 / 237.0, (36.7, -24.6, 27.6)),
                                   (1.0 / 237.0, (0.0, 0.0, 0.0)), (5.0, (-80.0, 12.0, 3.0))]:
            with self.subTest(scale_world=scale_world, t_c2g=t_c2g):
                rig = synthetic_rig(n=25, seed=61, scale_world=scale_world, t_c2g=t_c2g)
                res = he.solve_hand_eye(rig["robot_T"], rig["cam_c2w"])
                self._check_solution(rig, res["R_c2g"], res["t_c2g"], res["scale"])
                self.assertTrue(res["scale_observable"])
                self.assertLess(res["rms_residual"], TOL)
                self.assertEqual(res["n_poses"], rig["n"])
                if np.linalg.norm(t_c2g) > 0:
                    # documents why the joint solve exists: the centre-vs-gripper Umeyama
                    # scale is only approximate when the camera is offset from the gripper
                    self.assertGreater(abs(res["scale_umeyama"] * scale_world - 1.0), 1e-4)

    def test_tsai_rotation_is_scale_independent_but_translation_is_not(self):
        rig = synthetic_rig(n=25, seed=62)
        w2c_colmap = he.build_scaled_w2c_using_scale_only(rig["cam_c2w"], 1.0)
        R_c2g, t_bad = he.hand_eye_calibration(rig["robot_T"], w2c_colmap)
        self.assertLess(rotation_error(R_c2g, rig["T_c2g"][:3, :3]), TOL)
        self.assertGreater(np.abs(t_bad - rig["T_c2g"][:3, 3]).max(), 1.0)   # wrong at COLMAP scale
        t_c2g, s, info = he.solve_translation_and_scale(rig["robot_T"], w2c_colmap, R_c2g)
        self.assertTrue(info["observable"])
        self.assertEqual(info["mode_used"], "joint")
        self.assertGreater(info["fixed_point_rel_residual"], 0.1)     # random poses share no fixed point
        self._check_solution(rig, R_c2g, t_c2g, s)
        w2c_metric = he.build_scaled_w2c_using_scale_only(rig["cam_c2w"], 1.0 / rig["scale_world"])
        _, t_good = he.hand_eye_calibration(rig["robot_T"], w2c_metric)
        self.assertLess(np.abs(t_good - rig["T_c2g"][:3, 3]).max(), TOL)   # Tsai is exact at true scale

    def test_end_to_end_from_files(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=30, seed=63)
        scan_log, images_txt, transforms = write_fixtures(rig, tmp)
        for poses in (images_txt, transforms):
            with self.subTest(poses=os.path.basename(poses)):
                R_c2g, t_c2g, s = he.estimate_camera2gripper(scan_log, poses, phi_theta_tol=1e-3)
                self._check_solution(rig, R_c2g, t_c2g, s)
                T_BW = he.estimate_world2base(scan_log, poses, R_c2g, t_c2g, phi_theta_tol=1e-3)
                self.assertLess(np.abs(T_BW - rig["T_BW"]).max(), TOL)
                self.assertLess(np.abs(T_BW[3] - [0, 0, 0, 1]).max(), 1e-15)
                # world→base maps every COLMAP camera centre onto T_g2b @ t_c2g
                for Tg, C2W in zip(rig["robot_T"], rig["cam_c2w"]):
                    C_base = (T_BW @ np.r_[C2W[:3, 3], 1.0])[:3]
                    self.assertLess(np.abs(C_base - (Tg @ rig["T_c2g"])[:3, 3]).max(), TOL)

    def test_cli_entry_point(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=20, seed=64)
        scan_log, images_txt, _ = write_fixtures(rig, tmp)
        out = Path(tmp) / "hand_eye.json"
        cmd = [sys.executable, str(CALIB_DIR / "hand_eye.py"), "--scan_log_path", scan_log,
               "--poses_path", images_txt, "--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"Matched {rig['n']}/{rig['n']} frames", proc.stdout)
        res = load_json(out)
        self._check_solution(rig, np.array(res["R_c2g"]), np.array(res["t_c2g"]), res["scale_world_to_base"])
        self.assertLess(np.abs(np.array(res["T_BW"]) - rig["T_BW"]).max(), TOL)
        self.assertEqual(res["n_poses"], rig["n"])
        # the JSON declares its camera-axis convention and carries the OpenGL-frame matrix alongside
        self.assertEqual(res["camera_convention"], "opencv")
        self.assertLess(rotation_error(np.array(res["R_c2g_opengl"]),
                                       np.array(res["R_c2g"]) @ np.diag([1, -1, -1])), 1e-15)
        self.assertTrue(res["scale_observable"])
        self.assertEqual(res["scale_mode_used"], "joint")
        self.assertLess(res["scale_attenuation_est"], 1e-12)                  # noise-free fixtures
        self.assertEqual(res["scale_joint"], res["scale_world_to_base"])
        # second run re-using the solved transform via --c2g_json reproduces T_BW
        out2 = Path(tmp) / "hand_eye_prior.json"
        proc2 = subprocess.run(cmd[:-2] + ["--c2g_json", str(out), "--out", str(out2)],
                               capture_output=True, text=True, cwd=tmp)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        self.assertLess(np.abs(np.array(load_json(out2)["T_BW"]) - rig["T_BW"]).max(), TOL)
        self.assertFalse(list(Path(tmp).glob("*.ply")))

    @staticmethod
    def _spherical_rig(n=25, seed=66, scale_world=1.0 / 237.0, p_f=(40.0, 600.0, 0.0)):
        """Rig whose gripper poses all keep the gripper-frame point p_f on one base point
        (the rig's φ/θ pattern: position = focus − R·p_f, exactly)."""
        rig = synthetic_rig(n=n, seed=seed, scale_world=scale_world)
        p_f = np.asarray(p_f, float)
        focus = np.array([600.0, 0.0, 300.0])
        robot_T = [he.build_4x4(T[:3, :3], focus - T[:3, :3] @ p_f) for T in rig["robot_T"]]
        R_wb = rig["T_WB"][:3, :3] / scale_world
        t_wb = rig["T_WB"][:3, 3]
        cam_c2w = [he.build_4x4(R_wb @ (Tg @ rig["T_c2g"])[:3, :3],
                                scale_world * (R_wb @ (Tg @ rig["T_c2g"])[:3, 3]) + t_wb) for Tg in robot_T]
        rig.update(robot_T=robot_T, cam_c2w=cam_c2w, p_f=p_f, focus=focus)
        return rig

    def test_spherical_scan_scale_unobservable_falls_back_with_warning(self):
        """
        The rig's φ/θ pattern orbits one focus point: every gripper pose maps the
        gripper-frame point p_f onto the same base point, so (t_X, s) = (p_f + λ(t* − p_f), λ s*)
        satisfies AX = XB for every λ. The solver must flag this instead of returning the
        minimum-norm solution, keep Tsai's exact rotation, and fall back to the Umeyama
        centre-fit scale — landing on the ambiguity line.
        """
        import warnings
        rig = self._spherical_rig()
        p_f, scale_world = rig["p_f"], rig["scale_world"]
        for Tg in rig["robot_T"]:                                     # sanity: focus point is fixed
            self.assertLess(np.abs((Tg @ np.r_[p_f, 1.0])[:3] - rig["focus"]).max(), 1e-9)

        w2c = he.build_scaled_w2c_using_scale_only(rig["cam_c2w"], 1.0)
        R_c2g, _ = he.hand_eye_calibration(rig["robot_T"], w2c)
        self.assertLess(rotation_error(R_c2g, rig["T_c2g"][:3, :3]), TOL)   # rotation still exact
        with self.assertRaises(ValueError):
            he.solve_translation_and_scale(rig["robot_T"], w2c, R_c2g)         # no fallback given
        with self.assertRaises(ValueError):
            he.solve_translation_and_scale(rig["robot_T"], w2c, R_c2g, s_fallback=1.0, mode="bogus")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = he.solve_hand_eye(rig["robot_T"], rig["cam_c2w"])
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) and "unobservable" in str(w.message)
                            for w in caught))
        self.assertFalse(res["scale_observable"])
        self.assertEqual(res["scale_mode_used"], "umeyama")
        self.assertLess(res["fixed_point_rel_residual"], 1e-9)
        self.assertEqual(res["scale"], res["scale_umeyama"])
        self.assertGreater(res["scale"], 0.0)
        self.assertLess(res["rms_residual"], TOL)                           # still an exact AX=XB fit
        # fallback solution lies on the ambiguity line with λ = s_used / s_true
        lam = res["scale"] * scale_world
        t_star = rig["T_c2g"][:3, 3]
        self.assertLess(np.abs(res["t_c2g"] - (p_f + lam * (t_star - p_f))).max(), TOL)
        self.assertGreater(abs(lam - 1.0), 1e-3)                            # ...and is not the truth
        # forcing the joint branch is allowed but still reports the degeneracy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forced = he.solve_hand_eye(rig["robot_T"], rig["cam_c2w"], scale_mode="joint")
        self.assertFalse(forced["scale_observable"])
        self.assertEqual(forced["scale_mode_used"], "joint")

    def test_exact_spherical_robot_poses_with_noisy_cameras_are_detected(self):
        """
        Regression for the real session: the log positions are analytically spherical while
        COLMAP poses are noisy, so (t_X = p_f, s = 0) fits AX = XB with zero residual and the
        joint least squares collapses onto it even though the singular values look healthy.
        Detection must come from the gripper poses alone.
        """
        import warnings
        rig = self._spherical_rig(seed=67)
        rng = np.random.default_rng(68)
        noisy_c2w = []
        for C2W in rig["cam_c2w"]:
            dR = Rot.from_rotvec(rng.normal(scale=2e-3, size=3)).as_matrix()
            noisy_c2w.append(he.build_4x4(dR @ C2W[:3, :3], C2W[:3, 3] + rng.normal(scale=2e-3, size=3)))
        w2c = he.build_scaled_w2c_using_scale_only(noisy_c2w, 1.0)
        R_c2g, _ = he.hand_eye_calibration(rig["robot_T"], w2c)
        self.assertLess(rotation_error(R_c2g, rig["T_c2g"][:3, :3]), 5e-2)

        # the unguarded joint solve collapses to the degenerate (p_f, s≈0) solution
        t_joint, s_joint, info = he.solve_translation_and_scale(rig["robot_T"], w2c, R_c2g, mode="joint")
        self.assertFalse(info["observable"])
        self.assertLess(info["fixed_point_rel_residual"], 1e-9)
        self.assertGreater(info["sv_ratio"], he.SCALE_OBSERVABILITY_TOL)     # σ check alone is fooled
        self.assertLess(abs(s_joint * rig["scale_world"]), 0.05)
        self.assertLess(np.abs(t_joint - rig["p_f"]).max(), 1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = he.solve_hand_eye(rig["robot_T"], noisy_c2w)
        self.assertTrue(any("unobservable" in str(w.message) for w in caught))
        self.assertFalse(res["scale_observable"])
        self.assertEqual(res["scale_mode_used"], "umeyama")
        self.assertEqual(res["scale"], res["scale_umeyama"])
        # fallback stays near the ambiguity line (noise-limited), not at the collapsed point
        lam = res["scale"] * rig["scale_world"]
        expected = rig["p_f"] + lam * (rig["T_c2g"][:3, 3] - rig["p_f"])
        self.assertLess(np.abs(res["t_c2g"] - expected).max(), 5.0)
        self.assertGreater(np.abs(res["t_c2g"] - rig["p_f"]).max(), 100.0)

    @classmethod
    def _near_spherical_rig(cls, n, eps, seed, cam_rot_noise=1e-3, cam_t_noise=1e-3):
        """Spherical rig whose gripper positions are perturbed by eps·radius; camera poses are
        generated from the perturbed poses and then corrupted by cam_rot_noise [rad] /
        cam_t_noise [COLMAP units] (the verifier's probe)."""
        rig = cls._spherical_rig(n=n, seed=seed)
        rng = np.random.default_rng(seed + 1000)
        radius = float(np.linalg.norm(rig["p_f"]))
        robot_T = [he.build_4x4(T[:3, :3], T[:3, 3] + eps * radius * rng.normal(size=3)) for T in rig["robot_T"]]
        R_wb = rig["T_WB"][:3, :3] / rig["scale_world"]
        t_wb = rig["T_WB"][:3, 3]
        cam_c2w = []
        for Tg in robot_T:
            Tc2b = Tg @ rig["T_c2g"]
            C2W = he.build_4x4(R_wb @ Tc2b[:3, :3], rig["scale_world"] * (R_wb @ Tc2b[:3, 3]) + t_wb)
            dR = Rot.from_rotvec(rng.normal(scale=cam_rot_noise, size=3)).as_matrix()
            cam_c2w.append(he.build_4x4(dR @ C2W[:3, :3], C2W[:3, 3] + rng.normal(scale=cam_t_noise, size=3)))
        rig.update(robot_T=robot_T, cam_c2w=cam_c2w)
        return rig

    def test_near_spherical_scan_auto_is_never_worse_than_umeyama(self):
        """
        A scan that is *almost* spherical passes the fixed-point test, yet the joint (t_c2g, s)
        least squares is an errors-in-variables problem whose scale is attenuated towards zero
        (eps = 1e-3: about −50 %, 300 mm in t_c2g, versus −4 % / 25 mm for the Umeyama centre
        fit). 'auto' must estimate that attenuation from the fit and fall back to the Umeyama
        scale whenever it exceeds SCALE_ATTENUATION_TOL, and must switch to the joint solve once
        the scan is clearly non-spherical — so it is never worse than the original algorithm.
        """
        def solve(rig, mode):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = he.solve_hand_eye(rig["robot_T"], rig["cam_c2w"], scale_mode=mode)
            ds = abs(res["scale"] * rig["scale_world"] - 1.0)
            dt = float(np.linalg.norm(res["t_c2g"] - rig["T_c2g"][:3, 3]))
            return res, ds, dt, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

        for eps in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            for seed in (66, 67, 68):
                with self.subTest(eps=eps, seed=seed):
                    rig = self._near_spherical_rig(n=40, eps=eps, seed=seed)
                    auto, ds_auto, dt_auto, warns = solve(rig, "auto")
                    umey, ds_umey, dt_umey, _ = solve(rig, "umeyama")
                    joint, ds_joint, dt_joint, _ = solve(rig, "joint")
                    self.assertLess(rotation_error(auto["R_c2g"], rig["T_c2g"][:3, :3]), 1e-2)
                    # the fixed-point test alone does not catch this regime ...
                    self.assertGreater(auto["fixed_point_rel_residual"], he.FIXED_POINT_REL_TOL)
                    self.assertGreater(auto["sv_ratio"], he.SCALE_OBSERVABILITY_TOL)
                    # ... but auto is never worse than the Umeyama fallback (original algorithm)
                    self.assertLessEqual(ds_auto, ds_umey + 1e-9)
                    self.assertLessEqual(dt_auto, dt_umey + 1e-9)
                    self.assertLess(ds_umey, 0.05)
                    self.assertEqual(auto["scale_umeyama"], umey["scale"])
                    self.assertEqual(auto["scale_joint"], joint["scale"])
                    rho = auto["scale_attenuation_est"]
                    self.assertEqual(auto["scale_observable"], rho <= he.SCALE_ATTENUATION_TOL)
                    self.assertEqual(auto["scale_mode_used"], "joint" if auto["scale_observable"] else "umeyama")
                    if auto["scale_mode_used"] == "joint":
                        self.assertEqual(auto["scale"], joint["scale"])
                        self.assertLess(ds_auto, 2.5e-2)
                        self.assertFalse(warns)
                    else:
                        self.assertEqual(auto["scale"], umey["scale"])
                        self.assertTrue(any("unobservable" in w and "attenuated" in w for w in warns))
                    if eps <= 3e-3:          # joint scale badly biased here → must fall back
                        self.assertGreater(ds_joint, 0.05)
                        self.assertEqual(auto["scale_mode_used"], "umeyama")
                    if eps >= 3e-2:          # clearly non-spherical → joint solve, far better than Umeyama
                        self.assertEqual(auto["scale_mode_used"], "joint")
                        self.assertLess(ds_auto, 1e-2)
                        self.assertLess(dt_auto, 0.5 * dt_umey)
                    # first-order prediction of the joint bias, ρ/(1+ρ), is right to within a factor 2
                    # wherever that bias dominates the random error
                    if ds_joint > 0.05:
                        self.assertLess(abs(np.log((rho / (1 + rho)) / ds_joint)), np.log(2.0))

    def test_estimate_camera2gripper_exposes_scale_mode(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=20, seed=69)
        scan_log, images_txt, _ = write_fixtures(rig, tmp)
        R_auto, t_auto, s_auto = he.estimate_camera2gripper(scan_log, images_txt, scale_mode="auto")
        self._check_solution(rig, R_auto, t_auto, s_auto)
        R_um, t_um, s_um = he.estimate_camera2gripper(scan_log, images_txt, scale_mode="umeyama")
        self.assertLess(rotation_error(R_um, rig["T_c2g"][:3, :3]), TOL)        # Tsai rotation unaffected
        self.assertGreater(abs(s_um * rig["scale_world"] - 1.0), 1e-4)          # the (biased) centre-fit scale
        self.assertNotEqual(s_um, s_auto)
        with self.assertRaises(ValueError):
            he.estimate_camera2gripper(scan_log, images_txt, scale_mode="bogus")

    def test_too_few_matched_frames_is_a_clear_error(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=8, seed=70)
        _, images_txt, _ = write_fixtures(rig, tmp)
        shifted = Path(tmp) / "scan_log_shifted.json"          # φ offset → nothing matches within --tol
        with open(shifted, "w") as f:
            json.dump([dict(e, phi=e["phi"] + 0.5) for e in rig["scan_log"]], f)
        prior = Path(tmp) / "prior.json"
        with open(prior, "w") as f:
            json.dump(dict(R_c2g=rig["T_c2g"][:3, :3].tolist(), t_c2g=rig["T_c2g"][:3, 3].tolist()), f)
        cmd = [sys.executable, str(CALIB_DIR / "hand_eye.py"), "--scan_log_path", str(shifted),
               "--poses_path", images_txt, "--c2g_json", str(prior), "--tol", "1e-3"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("matched", proc.stderr)
        self.assertIn("--tol", proc.stderr)
        self.assertIn("phi", proc.stderr)
        # library level: the similarity needs ≥3 matched poses too
        with self.assertRaises(ValueError):
            he.world_to_base_sim3(rig["robot_T"][:2], rig["cam_c2w"][:2], rig["T_c2g"][:3, :3], rig["T_c2g"][:3, 3])

    def test_rejects_unknown_pose_file_and_too_few_poses(self):
        with self.assertRaises(ValueError):
            he.load_matched_poses("scan_log.json", "poses.bin")
        rig = synthetic_rig(n=2, seed=65)
        with self.assertRaises(ValueError):
            he.solve_hand_eye(rig["robot_T"], rig["cam_c2w"])


# ----------------------------------------------------------------------------- #
#  camera-axis convention of R_c2g (hand_eye.py output ↔ charuco_calibration.py)
# ----------------------------------------------------------------------------- #
class TestCameraConvention(unittest.TestCase):
    def test_flip_is_an_involution_consistent_with_the_gl_cv_helpers(self):
        for R_cv in random_rotations(20, seed=81):
            R_gl = he.flip_camera_convention_c2g(R_cv)
            self.assertLess(rotation_error(R_gl, R_cv @ np.diag([1, -1, -1])), 1e-15)
            self.assertLess(rotation_error(he.flip_camera_convention_c2g(R_gl), R_cv), 1e-15)
            self.assertAlmostEqual(np.linalg.det(R_gl), 1.0, places=12)
            T = he.build_4x4(R_cv, [1.0, 2.0, 3.0])
            self.assertLess(np.abs(he._cv_to_gl(T) - he.build_4x4(R_gl, [1.0, 2.0, 3.0])).max(), 1e-15)

    def test_load_c2g_json_honours_camera_convention(self):
        tmp = make_tmpdir(self)
        R_cv = random_rotations(1, seed=82)[0]
        t = np.array([10.0, -20.0, 30.0])
        for conv, R_file in (("opencv", R_cv), ("opengl", he.flip_camera_convention_c2g(R_cv)), (None, R_cv)):
            p = Path(tmp) / f"c2g_{conv}.json"
            d = dict(R_c2g=R_file.tolist(), t_c2g=t.tolist())
            if conv:
                d["camera_convention"] = conv
            with open(p, "w") as f:
                json.dump(d, f)
            R_load, t_load = he.load_c2g_json(str(p))
            self.assertLess(rotation_error(R_load, R_cv), 1e-15, conv)
            self.assertLess(np.abs(t_load - t).max(), 1e-15)
        bad = Path(tmp) / "bad.json"
        with open(bad, "w") as f:
            json.dump(dict(R_c2g=R_cv.tolist(), t_c2g=t.tolist(), camera_convention="blender"), f)
        with self.assertRaises(ValueError):
            he.load_c2g_json(str(bad))


class TestCharucoCameraToGripper(unittest.TestCase):
    """
    charuco_calibration.py composes T_c2g with solvePnP board poses, which live in the
    OpenCV camera frame. A hand_eye.py --out JSON (OpenCV convention) must therefore be
    used as is, while the built-in constants (OpenGL convention, like rig_constants.yaml)
    must be flipped by 180° about the camera x axis.
    """

    def test_hand_eye_out_json_is_used_without_extra_flip(self):
        tmp = make_tmpdir(self)
        rig = synthetic_rig(n=20, seed=91)
        scan_log, images_txt, _ = write_fixtures(rig, tmp)
        out = Path(tmp) / "hand_eye.json"
        proc = subprocess.run([sys.executable, str(CALIB_DIR / "hand_eye.py"), "--scan_log_path", scan_log,
                               "--poses_path", images_txt, "--out", str(out)],
                              capture_output=True, text=True, cwd=tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        d = load_json(out)
        T_loaded = cc.load_camera_to_gripper(str(out))
        self.assertLess(np.abs(T_loaded - he.build_4x4(np.array(d["R_c2g"]), np.array(d["t_c2g"]))).max(), 1e-15)
        # ... which is the rig's true OpenCV-frame camera→gripper: a board pose seen by the camera
        # lands on the same base pose whether composed with the truth or with the loaded transform
        self.assertLess(np.abs(T_loaded - rig["T_c2g"]).max(), TOL)
        T_board2cam = he.build_4x4(random_rotations(1, seed=92)[0], [5.0, -7.0, 400.0])
        for Tg in rig["robot_T"][:3]:
            self.assertLess(np.abs(Tg @ T_loaded @ T_board2cam - Tg @ rig["T_c2g"] @ T_board2cam).max(), TOL)
        # a file without the field defaults to opencv; --c2g_convention overrides
        d.pop("camera_convention")
        bare = Path(tmp) / "bare.json"
        with open(bare, "w") as f:
            json.dump(d, f)
        self.assertLess(np.abs(cc.load_camera_to_gripper(str(bare)) - T_loaded).max(), 1e-15)
        self.assertLess(np.abs(cc.load_camera_to_gripper(str(bare), "opencv") - T_loaded).max(), 1e-15)
        self.assertLess(np.abs(cc.load_camera_to_gripper(str(bare), "opengl") - T_loaded @ cc.T_CV2GL).max(), 1e-15)
        # the R_c2g_opengl that hand_eye.py writes, declared as opengl, is the same effective transform
        gl_json = Path(tmp) / "gl.json"
        with open(gl_json, "w") as f:
            json.dump(dict(R_c2g=d["R_c2g_opengl"], t_c2g=d["t_c2g"], camera_convention="opengl"), f)
        self.assertLess(np.abs(cc.load_camera_to_gripper(str(gl_json)) - T_loaded).max(), 1e-15)
        with self.assertRaises(ValueError):
            cc.load_camera_to_gripper(str(bare), "blender")

    def test_builtin_constants_are_opengl_and_get_flipped(self):
        T_builtin = cc.load_camera_to_gripper()
        self.assertLess(np.abs(T_builtin - cc.make_T(cc.R_c2g, cc.t_c2g) @ cc.T_CV2GL).max(), 1e-15)
        self.assertLess(np.abs(T_builtin[:3, 3] - cc.t_c2g).max(), 1e-15)                      # centre unchanged
        self.assertLess(np.abs(T_builtin[:3, :3] - cc.R_c2g @ np.diag([1, -1, -1])).max(), 1e-15)
        self.assertTrue(np.array_equal(cc.T_CV2GL @ cc.T_CV2GL, np.eye(4)))                     # involution
        self.assertLess(np.abs(cc.load_camera_to_gripper(None, "opencv") - cc.make_T(cc.R_c2g, cc.t_c2g)).max(), 1e-15)

    def test_real_session_output_agrees_with_builtin_once_conventions_match(self):
        """R_c2g solved by hand_eye.py on a real capture session (OpenCV convention) versus the
        built-in constant (OpenGL, a different session): ~2.7° apart when both are brought into
        the OpenCV frame, ~179° apart if the built-in flip were applied to the hand_eye output."""
        tmp = make_tmpdir(self)
        R_real = [[0.03420015903202844, -0.9987669531439001, -0.03598503063545694],
                  [0.012269239602441362, -0.03558379741257567, 0.999291378488416],
                  [-0.9993396894358604, -0.03461743302660636, 0.011037139524649353]]
        p = Path(tmp) / "real.json"
        with open(p, "w") as f:
            json.dump(dict(R_c2g=R_real, t_c2g=[28.80, -0.83, 25.57], camera_convention="opencv"), f)

        def angle_deg(Ra, Rb):
            return float(np.degrees(np.linalg.norm(Rot.from_matrix(Ra.T @ Rb).as_rotvec())))

        R_builtin_cv = cc.load_camera_to_gripper()[:3, :3]
        self.assertLess(angle_deg(cc.load_camera_to_gripper(str(p))[:3, :3], R_builtin_cv), 5.0)
        self.assertGreater(angle_deg(cc.load_camera_to_gripper(str(p), "opengl")[:3, :3], R_builtin_cv), 170.0)


# ----------------------------------------------------------------------------- #
#  release hygiene
# ----------------------------------------------------------------------------- #
class TestNoInternalPaths(unittest.TestCase):
    def test_calibration_sources_have_no_hardcoded_internal_paths(self):
        for name in ("hand_eye.py", "charuco_calibration.py"):
            src = (CALIB_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("/media/raid", src, name)
            self.assertNotIn("Pipeline_test_images", src, name)

    def test_charuco_entry_point_requires_cli_paths(self):
        proc = subprocess.run([sys.executable, str(CALIB_DIR / "charuco_calibration.py"), "--help"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ("--calib_glob", "--top_glob", "--tilt_glob", "--scan_log_top", "--scan_log_tilt",
                     "--c2g_json", "--c2g_convention"):
            self.assertIn(flag, proc.stdout)
        proc = subprocess.run([sys.executable, str(CALIB_DIR / "charuco_calibration.py")],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)          # --calib_glob is required
        self.assertIn("--calib_glob", proc.stderr)


if __name__ == "__main__":
    unittest.main()
