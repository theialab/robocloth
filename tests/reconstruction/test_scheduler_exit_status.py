"""
B7 — reconstruction/scheduler.py checks child exit statuses and fails closed.

Exercises spawn_logged_job / child_exit_status / classify_colmap_exit and the
COLMAP + shape-matching branches of update_job_status with tiny bash children
standing in for colmap.sh / reconstruct.py, plus archive_rejected_sparse (the
scheduler-side sparse_seq_failed-<ts> archive never deletes an earlier one).
No COLMAP, no GPU, no NFS.

Run:  <python> -m unittest discover -s tests/reconstruction -v
(needs psutil + pynvml for the scheduler import)
"""
import os
import shutil
import sys
import time
import unittest
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import fixtures

sys.path.insert(0, str(fixtures.RECON_DIR))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # pynvml deprecation notice
    import scheduler as S  # noqa: E402

FINISHED = "== Finished COLMAP reconstruction =="
FAILED_MARKER = "== FAILED: stage=mapper exit=1 =="


def _wait(proc, timeout=30):
    proc.wait(timeout=timeout)
    return proc


class SpawnAndClassifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = fixtures.make_tmpdir("sched")
        self.folder = self.tmp / "mat_001"
        self.folder.mkdir()
        self.log = self.folder / "colmap.log"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spawn_records_exit_status(self):
        p = _wait(S.spawn_logged_job(["bash", "-c", "echo hello; exit 7"], str(self.log)))
        self.assertEqual(S.child_exit_status(p.pid), 7)
        self.assertEqual(S.child_exit_status(p.pid), 7, "status must stay readable after the first poll")
        self.assertEqual(self.log.read_text().strip(), "hello")
        self.assertFalse(S.is_process_alive(p.pid))
        self.assertIsNone(S.child_exit_status(None))
        self.assertIsNone(S.child_exit_status(2 ** 22 + 12345), "unknown pid (previous scheduler) -> None")

    def test_spawn_reports_signal_death_as_negative(self):
        p = _wait(S.spawn_logged_job(["bash", "-c", "kill -9 $$"], str(self.log)))
        self.assertEqual(S.child_exit_status(p.pid), -9)

    def test_spawn_rejects_shell_strings(self):
        with self.assertRaises(TypeError):
            S.spawn_logged_job("echo hi", str(self.log))
        with self.assertRaises(TypeError):
            S.spawn_logged_job(["bash", 3], str(self.log))

    def test_classify(self):
        f = str(self.folder)
        # nonzero status wins, marker text is surfaced when present
        self.assertEqual(S.classify_colmap_exit(f, 1)[0], "failed")
        self.log.write_text("stuff\n" + FAILED_MARKER + "\nTemporary directory removed: x\n")
        self.assertEqual(S.classify_colmap_exit(f, 1), ("failed", FAILED_MARKER))
        # marker alone (status unknown after a scheduler restart) is enough
        self.assertEqual(S.classify_colmap_exit(f, None), ("failed", FAILED_MARKER))
        # signal death is "vanished", not a script verdict
        self.assertEqual(S.classify_colmap_exit(f, -9), ("vanished", "killed by signal 9"))
        # success signals
        self.log.write_text("...\n" + FINISHED + "\nTemporary directory removed: x\n")
        self.assertEqual(S.classify_colmap_exit(f, 0)[0], "ok")
        self.assertEqual(S.classify_colmap_exit(f, None)[0], "ok")
        self.log.write_text("no markers at all\n")
        self.assertEqual(S.classify_colmap_exit(f, 0), ("ok", "exit status 0"))
        # unknown status + no marker: only a FRESH points3D.ply counts
        self.assertEqual(S.classify_colmap_exit(f, None)[0], "vanished")
        ply = self.folder / "sparse" / "points3D.ply"
        ply.parent.mkdir()
        ply.write_text("ply\n")
        now = datetime.now()
        old = (now - timedelta(hours=1)).timestamp()
        os.utime(ply, (old, old))
        self.assertEqual(S.classify_colmap_exit(f, None, now.isoformat())[0], "vanished",
                         "a pre-existing sparse/ from an older run must not count as success")
        self.assertEqual(S.classify_colmap_exit(f, None, (now - timedelta(hours=2)).isoformat())[0], "ok")
        self.assertEqual(S.classify_colmap_exit(f, None, None)[0], "ok")  # legacy state without start time

    def test_classify_signal_status_reported_by_trap(self):
        """colmap.sh's INT/TERM trap exits 130/143 with a FAILED marker: that is
        "vanished" (interrupted), not a failed reconstruction to retry exhaustively."""
        f = str(self.folder)
        self.assertEqual(S.classify_colmap_exit(f, 143), ("vanished", "killed by signal 15 (exit status 143)"))
        self.assertEqual(S.classify_colmap_exit(f, 130)[0], "vanished")
        self.assertEqual(S.classify_colmap_exit(f, 137)[0], "vanished")  # OOM-killed COLMAP stage via set -e
        marker = "== FAILED: stage=mapper exit=143 =="
        self.log.write_text("stuff\n" + marker + "\nTemporary directory removed: x\n")
        outcome, reason = S.classify_colmap_exit(f, 143)
        self.assertEqual(outcome, "vanished")
        self.assertIn("killed by signal 15", reason)
        self.assertIn(marker, reason)
        # status unknown after a scheduler restart: the marker's exit code decides
        self.assertEqual(S.classify_colmap_exit(f, None)[0], "vanished")
        # a plain failure status in the marker is still a verdict
        self.log.write_text("== FAILED: stage=validate exit=3 ==\n")
        self.assertEqual(S.classify_colmap_exit(f, None)[0], "failed")
        self.assertEqual(S.classify_colmap_exit(f, 3)[0], "failed")
        self.assertEqual(S.classify_colmap_exit(f, 128)[0], "failed")  # boundary: 128 is not a signal
        # helpers
        self.assertEqual(S._marker_exit_code(marker), 143)
        self.assertIsNone(S._marker_exit_code(None))
        self.assertIsNone(S._marker_exit_code("== FAILED: stage=mapper exit=abc =="))
        self.assertEqual([S._signal_of(s) for s in (None, 0, 1, 128, 129, 143, -9)], [None, None, None, None, 1, 15, 9])


class UpdateJobStatusTest(unittest.TestCase):
    """update_job_status() with real (tiny) children so pid/exit handling is exercised end to end."""

    def setUp(self):
        self.tmp = fixtures.make_tmpdir("sched-status")
        self.project = fixtures.make_project(self.tmp, n_frames=10, existing_sparse=True)
        self.config = S.Config(dataset_root=str(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _state(self, status, pid, variant="sequential"):
        return {"materials": {"mat_001": {
            "status": status, "folder_path": str(self.project), "pid": pid, "gpu": 1,
            "workers": 4, "colmap_variant": variant, "colmap_attempts": 1,
            "colmap_start_time": datetime.now().isoformat(), "ready": True, "error": None,
        }}}

    def _run_colmap_child(self, script):
        p = _wait(S.spawn_logged_job(["bash", "-c", script], str(self.project / "colmap.log")))
        return p.pid

    def test_sequential_script_failure_retries_with_exhaustive(self):
        pid = self._run_colmap_child(f"echo '{FAILED_MARKER}'; exit 1")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid)
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.NOT_STARTED)
        self.assertEqual(info["colmap_variant"], "exhaustive")
        self.assertIsNone(info["pid"])
        self.assertIsNone(info["gpu"])

    def test_exhaustive_script_failure_marks_failed(self):
        pid = self._run_colmap_child("echo '== FAILED: stage=validate exit=3 =='; exit 3")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid, variant="exhaustive")
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.FAILED)
        self.assertIn("stage=validate exit=3", info["error"])

    def test_nonzero_exit_without_marker_is_still_a_failure(self):
        # even if the log claims success, a nonzero status is trusted (fail closed)
        pid = self._run_colmap_child(f"echo '{FINISHED}'; exit 5")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid, variant="exhaustive")
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.FAILED)
        self.assertIn("exited with status 5", info["error"])

    def test_signal_death_marks_failed_not_retried(self):
        pid = self._run_colmap_child("kill -9 $$")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid)
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.FAILED)
        self.assertEqual(info["colmap_variant"], "sequential")
        self.assertIn("killed by signal 9", info["error"])

    def test_trapped_signal_exit_marks_failed_not_retried(self):
        # what colmap.sh's TERM trap produces: marker + exit 143 (sequential variant)
        pid = self._run_colmap_child("echo '== FAILED: stage=mapper exit=143 =='; exit 143")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid)
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.FAILED)
        self.assertEqual(info["colmap_variant"], "sequential", "an interrupted run must not trigger the exhaustive retry")
        self.assertIn("killed by signal 15", info["error"])
        self.assertIn("stage=mapper", info["error"])

    def test_clean_exit_passes_registration_gate(self):
        pid = self._run_colmap_child(f"echo '{FINISHED}'; exit 0")
        state = self._state(S.JobStatus.COLMAP_RUNNING, pid)
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.COLMAP_DONE, info.get("error"))
        self.assertIsNotNone(info["colmap_end_time"])

    def test_shape_matching_nonzero_exit_fails_even_with_marker(self):
        log = str(self.project / "shape_matching.log")
        p = _wait(S.spawn_logged_job(
            ["bash", "-c", f"echo 'Finished shape matching for {self.project}'; exit 1"], log))
        state = self._state(S.JobStatus.SHAPE_MATCHING_RUNNING, p.pid)
        S.update_job_status(state, self.config)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.FAILED)
        self.assertIn("exited with status 1", info["error"])
        self.assertIsNone(info["workers"])

    def test_shape_matching_clean_exit_completes(self):
        log = str(self.project / "shape_matching.log")
        p = _wait(S.spawn_logged_job(
            ["bash", "-c", f"echo 'Finished shape matching for {self.project}'; exit 0"], log))
        state = self._state(S.JobStatus.SHAPE_MATCHING_RUNNING, p.pid)
        S.update_job_status(state, self.config)
        self.assertEqual(state["materials"]["mat_001"]["status"], S.JobStatus.COMPLETED)

    def test_launch_colmap_uses_wrapper_and_exports_colmap_tmp(self):
        script = self.tmp / "fake_colmap_script.sh"
        script.write_text('#!/usr/bin/env bash\necho "project=$1 gpu=$2 COLMAP_TMP=$COLMAP_TMP"\n'
                          f'echo "{FINISHED}"\n')
        self.config.COLMAP_SCRIPT = str(script)
        self.config.COLMAP_TMP_BASE = str(self.tmp / "ctmp")
        state = self._state(S.JobStatus.NOT_STARTED, None)
        state["materials"]["mat_001"]["colmap_attempts"] = 0
        pid = S.launch_colmap("mat_001", str(self.project), 1, state, self.config)
        self.assertIsNotNone(pid)
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.COLMAP_RUNNING)
        self.assertEqual(info["pid"], pid)
        self.assertEqual(info["colmap_attempts"], 1)
        _wait(S._CHILD_PROCESSES[pid])
        self.assertEqual(S.child_exit_status(pid), 0)
        log = (self.project / "colmap.log").read_text()
        self.assertIn(f"project={self.project} gpu=1 COLMAP_TMP={self.tmp / 'ctmp'}", log)
        # and the poll loop then promotes it (valid fixture sparse/ passes the gate)
        S.update_job_status(state, self.config)
        self.assertEqual(info["status"], S.JobStatus.COLMAP_DONE, info.get("error"))


class ArchiveRejectedSparseTest(unittest.TestCase):
    """The scheduler-side archive before an exhaustive retry is timestamped and never deletes."""

    def setUp(self):
        self.tmp = fixtures.make_tmpdir("sched-archive")
        self.project = fixtures.make_project(self.tmp, n_frames=10, existing_sparse=False)
        # a published sparse/ that registers only 5 of the 10 frames (below the 0.90 gate)
        fixtures.write_model_dir(self.project / "sparse", 5)
        (self.project / "sparse" / "SENTINEL.txt").write_text("rejected reconstruction\n")
        # earlier archives that must survive untouched: a timestamped one and the
        # legacy unsuffixed name the old code used to rmtree
        self.old = self.project / "sparse_seq_failed-20200101-000000"
        self.old.mkdir()
        (self.old / "OLD.txt").write_text("older archive\n")
        self.legacy = self.project / "sparse_seq_failed"
        self.legacy.mkdir()
        (self.legacy / "LEGACY.txt").write_text("legacy archive\n")
        self.before_old = fixtures.snapshot(self.old)
        self.before_legacy = fixtures.snapshot(self.legacy)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def archives(self):
        return sorted(p.name for p in self.project.glob("sparse_seq_failed*"))

    def assert_earlier_archives_intact(self):
        self.assertEqual(fixtures.snapshot(self.old), self.before_old)
        self.assertEqual(fixtures.snapshot(self.legacy), self.before_legacy)

    def test_archive_is_timestamped_and_never_deletes(self):
        new = S.archive_rejected_sparse(str(self.project))
        self.assertIsNotNone(new)
        self.assertRegex(Path(new).name, r"^sparse_seq_failed-\d{8}-\d{6}$")
        self.assertFalse((self.project / "sparse").exists())
        self.assertTrue((Path(new) / "SENTINEL.txt").is_file())
        self.assert_earlier_archives_intact()
        self.assertEqual(len(self.archives()), 3)
        # nothing left to archive
        self.assertIsNone(S.archive_rejected_sparse(str(self.project)))
        # a same-second clash gets a suffix instead of overwriting
        fixtures.write_model_dir(self.project / "sparse", 5)
        new2 = S.archive_rejected_sparse(str(self.project))
        self.assertNotEqual(new, new2)
        self.assertTrue(Path(new).is_dir() and Path(new2).is_dir())
        self.assertEqual(len(self.archives()), 4)

    def test_evaluate_colmap_registration_archives_without_deleting(self):
        info = {"folder_path": str(self.project), "colmap_variant": "sequential"}
        ok, msg = S.evaluate_colmap_registration("mat_001", info)
        self.assertFalse(ok)
        self.assertIn("will retry with exhaustive matcher", msg)
        self.assertFalse((self.project / "sparse").exists())
        self.assertEqual(len(self.archives()), 3, self.archives())
        self.assert_earlier_archives_intact()
        # the exhaustive variant gives up without touching sparse/
        fixtures.write_model_dir(self.project / "sparse", 5)
        (self.project / "sparse" / "SENTINEL.txt").write_text("x\n")
        ok, msg = S.evaluate_colmap_registration("mat_001", {**info, "colmap_variant": "exhaustive"})
        self.assertFalse(ok)
        self.assertIn("exhaustive retry also failed", msg)
        self.assertTrue((self.project / "sparse" / "SENTINEL.txt").is_file())
        self.assertEqual(len(self.archives()), 3)

    def test_verify_completed_materials_archives_without_deleting(self):
        state = {"materials": {"mat_001": {"status": S.JobStatus.COMPLETED, "folder_path": str(self.project)}}}
        self.assertEqual(S.verify_completed_materials(state, verbose=False), (1, 0))
        info = state["materials"]["mat_001"]
        self.assertEqual(info["status"], S.JobStatus.NOT_STARTED)
        self.assertEqual(info["colmap_variant"], "exhaustive")
        self.assertFalse((self.project / "sparse").exists())
        self.assertEqual(len(self.archives()), 3, self.archives())
        self.assert_earlier_archives_intact()


if __name__ == "__main__":
    unittest.main()
