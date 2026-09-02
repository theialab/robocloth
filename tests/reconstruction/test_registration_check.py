"""
B7 — staged-model validation in reconstruction/registration_check.py
(the gate colmap.sh / colmap_exhaustive.sh run before publishing).

Run:  <python> -m unittest discover -s tests/reconstruction -v
"""
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import fixtures
import numpy as np

sys.path.insert(0, str(fixtures.RECON_DIR))
import registration_check as rc  # noqa: E402

RC_SCRIPT = fixtures.RECON_DIR / "registration_check.py"


class ValidateSparseModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = fixtures.make_tmpdir("regcheck")
        self.project = fixtures.make_project(self.tmp, n_frames=10, existing_sparse=False)
        self.scan_log = self.project / "scan_log.json"
        self.staged = self.tmp / "staged_sparse"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_model_passes(self):
        fixtures.write_model_dir(self.staged, 10)
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_OK, msg)
        self.assertIn("10/10", msg)

    def test_low_registration_is_code_3_and_threshold_is_respected(self):
        fixtures.write_model_dir(self.staged, 5)  # 50 % of the 10 scans
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_LOW_REGISTRATION, msg)
        self.assertIn("below threshold", msg)
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log, threshold=0.4)
        self.assertEqual(code, rc.VALIDATE_OK, msg)
        # boundary: ratio == threshold passes (>=)
        code, _ = rc.validate_sparse_model(self.staged, self.scan_log, threshold=0.5)
        self.assertEqual(code, rc.VALIDATE_OK)

    def test_empty_model_is_code_2(self):
        fixtures.write_model_dir(self.staged, 0)
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("images.bin", msg)

    def test_missing_files_is_code_2(self):
        self.staged.mkdir()
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("missing", msg)
        # flattened copy present but submodel 0 missing
        fixtures.write_model_dir(self.staged, 10)
        shutil.rmtree(self.staged / "0")
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("0/", msg.replace(os.sep, "/"))

    def test_truncated_images_bin_is_code_2(self):
        fixtures.write_model_dir(self.staged, 10)
        (self.staged / "images.bin").write_bytes(b"\x00\x01\x02\x03")
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("does not load", msg)

    def test_submodel_flattened_mismatch_is_code_2(self):
        fixtures.write_model_dir(self.staged, 10)
        cams, imgs, pts = fixtures.build_model(5)
        fixtures.rwm.write_images_binary(imgs, str(self.staged / "0" / "images.bin"))
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("registers 5", msg)

    def test_unknown_camera_id_is_code_2(self):
        cams, imgs, pts = fixtures.build_model(10)
        imgs[3] = imgs[3]._replace(camera_id=42)
        sub = self.staged / "0"
        sub.mkdir(parents=True)
        fixtures.rwm.write_model(cams, imgs, pts, str(sub), ext=".bin")
        fixtures.rwm.write_model(cams, imgs, pts, str(self.staged), ext=".bin")
        fixtures.write_ply(self.staged / "points3D.ply", pts)
        code, msg = rc.validate_sparse_model(self.staged, self.scan_log)
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("unknown camera ids [42]", msg)

    def test_missing_scan_log_is_code_2(self):
        fixtures.write_model_dir(self.staged, 10)
        code, msg = rc.validate_sparse_model(self.staged, self.tmp / "nope.json")
        self.assertEqual(code, rc.VALIDATE_INVALID_MODEL, msg)
        self.assertIn("scan_log", msg)

    # ---- CLI + environment threshold (what the shell scripts use) ----

    def _cli(self, env_threshold=None, extra_args=()):
        env = {k: v for k, v in os.environ.items() if k != rc.REGISTRATION_THRESHOLD_ENV}
        if env_threshold is not None:
            env[rc.REGISTRATION_THRESHOLD_ENV] = env_threshold
        return subprocess.run(
            [sys.executable, str(RC_SCRIPT), "--sparse-dir", str(self.staged),
             "--scan-log", str(self.scan_log), *extra_args],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)

    def test_cli_exit_codes_and_env_threshold(self):
        fixtures.write_model_dir(self.staged, 5)  # 50 %
        res = self._cli()
        self.assertEqual(res.returncode, rc.VALIDATE_LOW_REGISTRATION, res.stdout)
        self.assertIn("[validate] FAIL (3)", res.stdout)
        self.assertIn("threshold 90%", res.stdout)  # default 0.90
        res = self._cli(env_threshold="0.4")
        self.assertEqual(res.returncode, rc.VALIDATE_OK, res.stdout)
        self.assertIn("[validate] OK (0)", res.stdout)
        # explicit --threshold wins over the environment
        res = self._cli(env_threshold="0.4", extra_args=("--threshold", "0.6"))
        self.assertEqual(res.returncode, rc.VALIDATE_LOW_REGISTRATION, res.stdout)

    def test_cli_rejects_malformed_env_threshold(self):
        fixtures.write_model_dir(self.staged, 10)
        for bad in ("abc", "1.5", "0"):
            res = self._cli(env_threshold=bad)
            self.assertNotEqual(res.returncode, 0, bad)
            self.assertIn(rc.REGISTRATION_THRESHOLD_ENV, res.stdout)

    # ---- material-dir API used by the scheduler is unchanged ----

    def test_material_dir_api(self):
        self.assertEqual(rc.registration_ratio(self.project), (-1, 10, -1.0))
        self.assertFalse(rc.is_well_registered(self.project))
        fixtures.write_model_dir(self.project / "sparse", 9)
        self.assertEqual(rc.registration_ratio(self.project), (9, 10, 0.9))
        self.assertTrue(rc.is_well_registered(self.project))
        self.assertFalse(rc.is_well_registered(self.project, threshold=0.95))
        self.assertEqual(rc.DEFAULT_REGISTRATION_THRESHOLD, 0.90)
        self.assertTrue(np.isclose(rc.REGISTRATION_THRESHOLD, 0.90)
                        or rc.REGISTRATION_THRESHOLD_ENV in os.environ)


if __name__ == "__main__":
    unittest.main()
