"""
B7 — reconstruction/colmap.sh and colmap_exhaustive.sh fail closed.

Runs the real shell scripts against a fake `colmap` executable (fake_colmap.py,
installed first on PATH) — no real COLMAP, no GPU. Fixture material dirs are
created under ROBOCLOTH_CI_ROOT (default: the system temp dir) and removed
afterwards; nothing is written inside the repository.

Asserted contract:
  * a valid run replaces sparse/ and keeps the previous one as sparse.prev-<ts>
    (sentinel file intact), staging dir removed;
  * a mapper crash / empty model / unreadable model / low registration exit
    nonzero, leave the existing sparse/ byte-identical, create no sparse.prev-*,
    keep the staged output as sparse.failed-<ts>, remove the staging dir;
  * COLMAP_REGISTRATION_THRESHOLD overrides the 0.90 gate;
  * SIGINT / SIGTERM (to the process group, or children-then-parent the way
    scheduler.terminate_process() kills) exit 130 / 143 through the same trap:
    marker line, existing sparse/ untouched, staging removed, partial output
    kept as sparse.failed-<ts>.

Run:  <python> -m unittest discover -s tests/reconstruction -v
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

import fixtures

sys.path.insert(0, str(fixtures.RECON_DIR))
import registration_check as rc  # noqa: E402

FAKE_COLMAP = Path(__file__).resolve().parent / "fake_colmap.py"
N_FRAMES = 10


class _WrapperCases:
    """Test bodies shared by the sequential and exhaustive variants."""
    SCRIPT = "colmap.sh"
    MATCHER = "sequential_matcher"

    def setUp(self):
        self.tmp = fixtures.make_tmpdir(self.SCRIPT.replace(".sh", ""))
        self.tmp_base = self.tmp / "colmap_tmp"       # COLMAP_TMP (staging base)
        self.bin_dir = self.tmp / "bin"
        self.bin_dir.mkdir()
        shim = self.bin_dir / "colmap"
        shim.write_text('#!/usr/bin/env bash\nexec "$FAKE_COLMAP_PYTHON" '
                        f'"{FAKE_COLMAP}" "$@"\n')
        shim.chmod(0o755)
        self.fake_log = self.tmp / "fake_colmap_calls.log"
        self.project = fixtures.make_project(self.tmp, n_frames=N_FRAMES, existing_sparse=True)
        self.before = fixtures.snapshot(self.project / "sparse")
        self.assertIn("SENTINEL.txt", self.before)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers ----

    def script_env(self, mode, extra_env=None):
        env = {k: v for k, v in os.environ.items()
               if k not in ("COLMAP_REGISTRATION_THRESHOLD", "CUDA_VISIBLE_DEVICES", "COLMAP_TMP", "PYTHON")}
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        env.update(FAKE_COLMAP_PYTHON=sys.executable, FAKE_COLMAP_LOG=str(self.fake_log),
                   FAKE_COLMAP_MODE=mode, COLMAP_TMP=str(self.tmp_base), PYTHON=sys.executable)
        env.update(extra_env or {})
        return env

    def run_script(self, mode, extra_env=None, gpu="1", project=None):
        project = project or self.project
        res = subprocess.run(["bash", str(fixtures.RECON_DIR / self.SCRIPT), str(project), gpu],
                             env=self.script_env(mode, extra_env), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=180)
        # mimic the scheduler, which redirects the script's output to colmap.log
        (project / "colmap.log").write_text(res.stdout)
        return res

    def start_script(self, mode):
        """Background variant of run_script for the signal tests: own session and
        output straight into colmap.log, exactly like scheduler.spawn_logged_job()."""
        with open(self.project / "colmap.log", "w") as log:
            proc = subprocess.Popen(["bash", str(fixtures.RECON_DIR / self.SCRIPT), str(self.project), "1"],
                                    env=self.script_env(mode), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        self.addCleanup(self._reap, proc)
        return proc

    @staticmethod
    def _reap(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    def wait_for_hung_mapper(self, proc, timeout=30):
        """Block until the `hang` mapper has written its partial output and is sleeping."""
        partial = self.tmp_base / f"{self.project.name}_{proc.pid}" / "sparse" / "0" / "cameras.bin"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if partial.is_file() and "mapper" in self.fake_calls():
                time.sleep(0.2)  # let the fake enter its sleep loop
                return
            if proc.poll() is not None:
                self.fail("script exited before the mapper hung:\n" + (self.project / "colmap.log").read_text())
            time.sleep(0.05)
        self.fail("mapper never reached the hang point")

    def fake_calls(self):
        return [line.split()[0] for line in self.fake_log.read_text().splitlines()] \
            if self.fake_log.exists() else []

    def prev_dirs(self, project=None):
        return sorted((project or self.project).glob("sparse.prev-*"))

    def failed_dirs(self, project=None):
        return sorted((project or self.project).glob("sparse.failed-*"))

    def assert_staging_removed(self):
        if self.tmp_base.exists():
            self.assertEqual(list(self.tmp_base.iterdir()), [], "staging dir left behind")

    def assert_sparse_untouched(self, res):
        output = res if isinstance(res, str) else res.stdout
        self.assertEqual(fixtures.snapshot(self.project / "sparse"), self.before,
                         "existing sparse/ was modified by a failed run")
        self.assertEqual(self.prev_dirs(), [], "no sparse.prev-* must be created on failure")
        self.assertEqual(list(self.project.glob(".sparse.incoming-*")), [])
        self.assertNotIn("== Finished COLMAP reconstruction ==", output)

    # ---- (a) success: replace + keep previous ----

    def test_valid_run_publishes_and_keeps_previous(self):
        res = self.run_script("valid")
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertIn("== Finished COLMAP reconstruction ==", res.stdout)
        self.assertNotIn("== FAILED:", res.stdout)

        sparse = self.project / "sparse"
        for rel in ("0/cameras.bin", "0/images.bin", "0/points3D.bin", "cameras.bin", "images.bin",
                    "points3D.bin", "cameras.txt", "images.txt", "points3D.txt", "points3D.ply"):
            self.assertTrue((sparse / rel).is_file(), rel)
        self.assertFalse((sparse / "SENTINEL.txt").exists(), "new sparse/ still holds the old sentinel")
        self.assertEqual(rc.registration_ratio(self.project), (N_FRAMES, N_FRAMES, 1.0))

        prev = self.prev_dirs()
        self.assertEqual(len(prev), 1, prev)
        self.assertRegex(prev[0].name, r"^sparse\.prev-\d{8}-\d{6}$")
        self.assertEqual(fixtures.snapshot(prev[0]), self.before, "previous sparse/ not preserved intact")
        self.assertEqual(self.failed_dirs(), [])
        self.assertEqual(list(self.project.glob(".sparse.incoming-*")), [])
        self.assert_staging_removed()

        calls = self.fake_calls()
        self.assertEqual(calls, ["feature_extractor", self.MATCHER, "mapper", "model_converter", "model_converter"])
        gpu_lines = [l for l in self.fake_log.read_text().splitlines() if "CUDA_VISIBLE_DEVICES=1" in l]
        self.assertGreaterEqual(len(gpu_lines), 4, self.fake_log.read_text())
        self.assertIn("[validate] OK (0)", res.stdout)

    def test_second_success_keeps_both_previous_results(self):
        self.assertEqual(self.run_script("valid").returncode, 0)
        time.sleep(1.1)  # distinct timestamp
        self.assertEqual(self.run_script("valid").returncode, 0)
        prev = self.prev_dirs()
        self.assertEqual(len(prev), 2, prev)
        self.assertIn("SENTINEL.txt", fixtures.snapshot(prev[0]))

    def test_no_previous_sparse(self):
        project = fixtures.make_project(self.tmp, name="mat_fresh", n_frames=N_FRAMES, existing_sparse=False)
        res = self.run_script("valid", project=project)
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertTrue((project / "sparse" / "points3D.ply").is_file())
        self.assertEqual(self.prev_dirs(project), [])
        self.assert_staging_removed()

    # ---- (b) mapper fails: nonzero, sparse untouched ----

    def test_mapper_failure_leaves_sparse_untouched(self):
        res = self.run_script("mapper_fail")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("== FAILED: stage=mapper exit=1 ==", res.stdout)
        self.assert_sparse_untouched(res)
        self.assert_staging_removed()
        failed = self.failed_dirs()
        self.assertEqual(len(failed), 1, failed)
        self.assertRegex(failed[0].name, r"^sparse\.failed-\d{8}-\d{6}$")
        self.assertTrue((failed[0] / "0" / "cameras.bin").is_file(), "partial staged output not kept")
        self.assertIn(f"Staged sparse kept at {failed[0]}", res.stdout)
        # nothing after the mapper ran
        self.assertEqual(self.fake_calls(), ["feature_extractor", self.MATCHER, "mapper"])

    # ---- (c) empty / invalid model: rejected by validation ----

    def test_empty_model_rejected(self):
        res = self.run_script("empty")
        self.assertEqual(res.returncode, rc.VALIDATE_INVALID_MODEL, res.stdout)
        self.assertIn("== FAILED: stage=validate exit=2 ==", res.stdout)
        self.assertIn("[validate] FAIL (2)", res.stdout)
        self.assert_sparse_untouched(res)
        self.assert_staging_removed()
        failed = self.failed_dirs()
        self.assertEqual(len(failed), 1, failed)
        self.assertTrue((failed[0] / "images.bin").is_file())

    def test_truncated_model_rejected(self):
        res = self.run_script("truncated")
        self.assertEqual(res.returncode, rc.VALIDATE_INVALID_MODEL, res.stdout)
        self.assertIn("== FAILED: stage=validate exit=2 ==", res.stdout)
        self.assertIn("does not load", res.stdout)
        self.assert_sparse_untouched(res)
        self.assert_staging_removed()

    def test_low_registration_rejected_unless_threshold_lowered(self):
        res = self.run_script("low_reg")  # registers 5/10
        self.assertEqual(res.returncode, rc.VALIDATE_LOW_REGISTRATION, res.stdout)
        self.assertIn("== FAILED: stage=validate exit=3 ==", res.stdout)
        self.assertIn("registered 5/10 (50.0%), threshold 90%", res.stdout)
        self.assert_sparse_untouched(res)
        self.assert_staging_removed()
        self.assertEqual(len(self.failed_dirs()), 1)

        res = self.run_script("low_reg", extra_env={"COLMAP_REGISTRATION_THRESHOLD": "0.4"})
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertIn("threshold 40%", res.stdout)
        self.assertEqual(rc.registration_ratio(self.project), (5, N_FRAMES, 0.5))
        self.assertEqual(len(self.prev_dirs()), 1)
        self.assertIn("SENTINEL.txt", fixtures.snapshot(self.prev_dirs()[0]))

    # ---- signals: an interrupted run takes the same fail-closed path ----

    def _assert_interrupted(self, proc, expected_rc, signame):
        out = (self.project / "colmap.log").read_text()
        self.assertEqual(proc.returncode, expected_rc, out)
        self.assertIn(f"== FAILED: stage=mapper exit={expected_rc} ==", out)
        self.assert_sparse_untouched(out)
        self.assert_staging_removed()
        failed = self.failed_dirs()
        self.assertEqual(len(failed), 1, failed)
        self.assertTrue((failed[0] / "0" / "cameras.bin").is_file(), f"partial output not kept after {signame}")
        self.assertIn(f"Staged sparse kept at {failed[0]}", out)
        self.assertEqual(self.fake_calls(), ["feature_extractor", self.MATCHER, "mapper"])

    def test_sigint_to_process_group_cleans_up(self):
        proc = self.start_script("hang")
        self.wait_for_hung_mapper(proc)
        os.killpg(proc.pid, signal.SIGINT)  # what Ctrl-C delivers
        proc.wait(timeout=60)
        self._assert_interrupted(proc, 130, "SIGINT")

    def test_sigterm_to_process_group_cleans_up(self):
        proc = self.start_script("hang")
        self.wait_for_hung_mapper(proc)
        os.killpg(proc.pid, signal.SIGTERM)  # kill -- -<pgid>
        proc.wait(timeout=60)
        self._assert_interrupted(proc, 143, "SIGTERM")

    def test_sigterm_children_first_then_parent_like_the_scheduler(self):
        """The order scheduler.terminate_process() uses (psutil: children, then parent)."""
        try:
            import psutil
        except ImportError:  # pragma: no cover
            self.skipTest("psutil not installed")
        proc = self.start_script("hang")
        self.wait_for_hung_mapper(proc)
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        self.assertTrue(children, "no colmap child found under the script")
        for child in children:
            child.terminate()
        parent.terminate()
        proc.wait(timeout=60)
        self._assert_interrupted(proc, 143, "SIGTERM")

    # ---- preflight ----

    def test_missing_ldr_fails_before_touching_anything(self):
        shutil.rmtree(self.project / "ldr")
        res = self.run_script("valid")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("== FAILED: stage=preflight exit=1 ==", res.stdout)
        self.assert_sparse_untouched(res)
        self.assertEqual(self.failed_dirs(), [])
        self.assertEqual(self.fake_calls(), [])
        self.assert_staging_removed()

    def test_missing_scan_log_fails_preflight(self):
        (self.project / "scan_log.json").unlink()
        res = self.run_script("valid")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("stage=preflight", res.stdout)
        self.assertIn("scan_log.json", res.stdout)
        self.assert_sparse_untouched(res)


class SequentialWrapperTest(_WrapperCases, unittest.TestCase):
    SCRIPT = "colmap.sh"
    MATCHER = "sequential_matcher"


class ExhaustiveWrapperTest(_WrapperCases, unittest.TestCase):
    SCRIPT = "colmap_exhaustive.sh"
    MATCHER = "exhaustive_matcher"


if __name__ == "__main__":
    unittest.main()
