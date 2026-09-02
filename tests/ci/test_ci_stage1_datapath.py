"""Dry-run harness for scripts/ci_stage1_datapath.sh — no network, no GPU, no training env.

A fake `hf` (tests/ci/fake_hf.py) is put first on PATH so the REAL
scripts/download_dataset_stage1.sh runs unmodified and assembles the layout the
CI script then validates.  Two groups of cases:

  * CiStage1DataPathDryRun     download -> validate with CI_SKIP_TRAIN=1, plus the
                               empty-root / CI_FRESH gate (including symlinked roots)
  * CiStage1TrainStepPlumbing  step 3's wiring — env vars, Hydra overrides, the
                               job-local `python` wrapper, metrics.csv / checkpoint
                               assertions, exit code 3 — against a scratch copy of
                               scripts/ whose train_stage1.sh is a FAKE.  The real
                               epoch is covered by the real run (see the CI script).

    <python with numpy> -B -m unittest discover -s tests/ci -v

Scratch: each case gets its own mkdtemp() under $ROBOCLOTH_TEST_TMP (created on
demand; unset -> the system temp dir) and removes it afterwards (keep with
ROBOCLOTH_CI_KEEP=1).  Nothing is left inside the repository:
without -B the import system caches this module under tests/ci/__pycache__ before
its body runs, so the module removes that cache again on exit.
"""
import sys

sys.dont_write_bytecode = True   # nothing imported from here on is cached ...

import atexit  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402


def _forget_own_bytecode():
    # ... but this module's own .pyc was written by the import system before the
    # flag above ran (compile -> cache -> exec).  A bytecode cache is always safe
    # to drop, so remove it on exit — and the __pycache__ directory once that
    # empties it — whenever it sits inside the repository next to this file
    # (PYTHONPYCACHEPREFIX would put it elsewhere; then it is not our residue).
    pyc = globals().get("__cached__")
    here = os.path.dirname(os.path.abspath(__file__))
    if not pyc or os.path.dirname(os.path.dirname(os.path.abspath(pyc))) != here:
        return

    def _remove():
        for op, path in ((os.remove, pyc), (os.rmdir, os.path.dirname(pyc))):
            try:
                op(path)
            except OSError:   # already gone, or the directory still holds other caches
                pass

    atexit.register(_remove)


_forget_own_bytecode()

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
SCRIPT = SCRIPTS / "ci_stage1_datapath.sh"
FAKE_HF = Path(__file__).resolve().with_name("fake_hf.py")
MATERIAL_FILES = ["observations_structured.npz", "scan_log.json", "rotated_camera.json", "point_metadata.json"]
# knobs that must never leak in from the caller's environment
KNOBS = ("CI_ROOT", "CI_FRESH", "CI_SKIP_TRAIN", "CI_TRAIN_BATCHES", "CI_TRAIN_ARGS", "ROBOCLOTH_PYTHON",
         "FAKE_HF_OMIT", "FAKE_HF_NPZ_DROP_KEY", "FAKE_HF_META_POINTS_DELTA",
         "FAKE_TRAIN_EXIT", "FAKE_TRAIN_LOSS0", "FAKE_TRAIN_NO_CKPT", "FAKE_TRAIN_STALE_CKPT")

SCRATCH_PARENT = os.environ.get("ROBOCLOTH_TEST_TMP") or None   # None -> tempfile's default location
KEEP = os.environ.get("ROBOCLOTH_CI_KEEP") == "1"

# Stand-in for scripts/train_stage1.sh (same contract: DATA_ROOT / OUTPUT_ROOT /
# TRAINING_LIST / EXP_NAME + Hydra overrides as "$@", calls a bare `python`).
# Records how it was called, then writes the metrics.csv / checkpoint the CI
# script asserts on.  Knobs: FAKE_TRAIN_EXIT, FAKE_TRAIN_LOSS0, FAKE_TRAIN_NO_CKPT,
# FAKE_TRAIN_STALE_CKPT (the checkpoint's mtime predates this run, like a leftover).
FAKE_TRAIN = r"""#!/bin/bash
set -euo pipefail
python - "$OUTPUT_ROOT/fake_train.json" "$@" <<'PY'
import json, os, shutil, sys
with open(sys.argv[1], "w") as f:
    json.dump({
        "argv": sys.argv[2:],
        "env": {k: os.environ.get(k) for k in ("DATA_ROOT", "OUTPUT_ROOT", "TRAINING_LIST", "EXP_NAME")},
        "python_on_path": shutil.which("python"),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "in_venv": sys.prefix != sys.base_prefix,
    }, f, indent=1)
PY
if [[ ${FAKE_TRAIN_EXIT:-0} != 0 ]]; then
    echo "fake train_stage1.sh: exiting $FAKE_TRAIN_EXIT"
    exit "$FAKE_TRAIN_EXIT"
fi
d=$OUTPUT_ROOT/$EXP_NAME/csv/version_0
mkdir -p "$d"
printf 'epoch,step,train/total_loss\n0,0,%s\n0,9,0.51\n' "${FAKE_TRAIN_LOSS0:-0.73}" > "$d/metrics.csv"
if [[ ${FAKE_TRAIN_NO_CKPT:-0} != 1 ]]; then
    mkdir -p "$OUTPUT_ROOT/$EXP_NAME/checkpoints"
    echo fake > "$OUTPUT_ROOT/$EXP_NAME/checkpoints/epoch=0-step=50.ckpt"
    if [[ ${FAKE_TRAIN_STALE_CKPT:-0} == 1 ]]; then
        touch -d '1 hour ago' "$OUTPUT_ROOT/$EXP_NAME/checkpoints/epoch=0-step=50.ckpt"
    fi
fi
"""


class _CiCase(unittest.TestCase):
    skip_train = True

    def setUp(self):
        if SCRATCH_PARENT:
            os.makedirs(SCRATCH_PARENT, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="ci_stage1_", dir=SCRATCH_PARENT))
        self.fakebin = self.tmp / "fakebin"
        self.fakebin.mkdir()
        shim = self.fakebin / "hf"
        shim.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{FAKE_HF}" "$@"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.data = self.tmp / "data"
        self.out = self.tmp / "out"
        self.hf_log = self.tmp / "hf_calls.jsonl"

    def tearDown(self):
        if not KEEP:
            shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers -------------------------------------------------------------
    def run_ci(self, script=SCRIPT, **overrides):
        env = os.environ.copy()
        for k in KNOBS:
            env.pop(k, None)
        env.update({
            "PATH": f"{self.fakebin}:{env.get('PATH', '')}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CI_DATA_ROOT": str(self.data),
            "CI_OUTPUT_ROOT": str(self.out),
            "CI_SKIP_TRAIN": "1" if self.skip_train else "0",
            "CI_WIPE_PREFIX": str(self.tmp),      # CI_FRESH may clear anything under this case's scratch
            "ROBOCLOTH_PYTHON": sys.executable,
            "MATERIALS": "145 226",
            "FAKE_HF_LOG": str(self.hf_log),
        })
        env.update({k: v for k, v in overrides.items() if v is not None})
        for k, v in overrides.items():
            if v is None:
                env.pop(k, None)
        proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=600)
        proc.combined = proc.stdout + proc.stderr
        return proc

    def hf_calls(self):
        if not self.hf_log.exists():
            return []
        return [json.loads(line) for line in self.hf_log.read_text().splitlines() if line.strip()]

    def assert_material_assembled(self, mid, root=None):
        for name in MATERIAL_FILES:
            p = (root or self.data) / str(mid) / name
            self.assertTrue(p.is_file() and p.stat().st_size > 0, f"{p} missing/empty")

    def populated_dir(self, name, sentinel="do not touch\n"):
        d = self.tmp / name
        d.mkdir()
        (d / "leftover.txt").write_text(sentinel)
        return d, d / "leftover.txt"

    def scratch_repo(self):
        """Copy of scripts/ with a FAKE train_stage1.sh.  The CI script locates its
        siblings through its own path, so the copy runs the real downloader and the
        real step-3 logic, only the epoch itself is faked."""
        scripts = self.tmp / "repo" / "scripts"
        scripts.mkdir(parents=True)
        for name in ("ci_stage1_datapath.sh", "download_dataset_stage1.sh"):
            shutil.copy(SCRIPTS / name, scripts / name)
        fake = scripts / "train_stage1.sh"
        fake.write_text(FAKE_TRAIN)
        fake.chmod(0o755)
        return scripts / "ci_stage1_datapath.sh"


class CiStage1DataPathDryRun(_CiCase):
    # -- (i) happy path ------------------------------------------------------
    def test_happy_path_downloads_two_materials_and_validates(self):
        proc = self.run_ci()
        self.assertEqual(proc.returncode, 0, proc.combined)
        for mid in (145, 226):
            self.assert_material_assembled(mid)
        self.assertFalse((self.data / "materials").exists(), "downloader left materials/ behind")
        self.assertTrue((self.data / "globals").is_dir())
        for name in ("emitter_calibration.json", "camera_factor.json", "sample_size.json", "training_list_500.txt"):
            self.assertEqual((self.data / name).read_bytes(), (self.data / "globals" / name).read_bytes(), name)
        self.assertIn("validation OK: 2 material(s)", proc.stdout)
        self.assertIn("PASS (download + validation)", proc.stdout)
        self.assertIn("skipping the training step", proc.stdout)
        self.assertTrue((self.out / "download.log").is_file() and (self.out / "validate.log").is_file())
        self.assertFalse((self.out / "ci_stage1").exists(), "training ran despite CI_SKIP_TRAIN=1")
        self.assertFalse((self.out / "pybin").exists(), "python wrapper belongs to the training step only")
        # exactly one hf call, with exactly the include list the downloader documents
        calls = self.hf_calls()
        self.assertEqual(len(calls), 1, calls)
        argv = calls[0]
        self.assertEqual(argv[:2], ["download", "koalapenguin/RoboCloth"])
        self.assertEqual(argv[argv.index("--repo-type") + 1], "dataset")
        self.assertEqual(argv[argv.index("--local-dir") + 1], str(self.data))
        includes = [argv[i + 1] for i, tok in enumerate(argv) if tok == "--include"]
        expected = ["globals/*"] + [f"materials/{mid}/{f}" for mid in (145, 226) for f in MATERIAL_FILES]
        self.assertEqual(sorted(includes), sorted(expected))

    # -- (ii) a required file missing upstream -------------------------------
    def test_missing_required_file_exits_2_and_names_it(self):
        proc = self.run_ci(FAKE_HF_OMIT="materials/226/rotated_camera.json")
        self.assertEqual(proc.returncode, 2, proc.combined)
        self.assertIn("VALIDATION FAILED", proc.stdout)
        self.assertRegex(proc.stdout, r"MISSING\s+\S+/226/rotated_camera\.json")
        self.assertNotRegex(proc.stdout, r"MISSING\s+\S+/145/")
        self.assertIn("validation failed", proc.stderr)

    def test_npz_missing_array_exits_2_and_names_it(self):
        proc = self.run_ci(FAKE_HF_NPZ_DROP_KEY="light_pos")
        self.assertEqual(proc.returncode, 2, proc.combined)
        self.assertRegex(proc.stdout, r"INVALID\s+\S+/145/observations_structured\.npz\s+\(missing arrays \['light_pos'\]")
        self.assertRegex(proc.stdout, r"INVALID\s+\S+/226/observations_structured\.npz")

    def test_point_metadata_mismatch_exits_2(self):
        proc = self.run_ci(FAKE_HF_META_POINTS_DELTA="3")
        self.assertEqual(proc.returncode, 2, proc.combined)
        self.assertRegex(proc.stdout, r"INVALID\s+\S+/145/point_metadata\.json\s+\(num_points=11 but observations_structured\.npz has V=8")

    # -- (iii) refuse a non-empty data root ----------------------------------
    def test_refuses_nonempty_data_root_without_ci_fresh(self):
        _, sentinel = self.populated_dir("data")
        proc = self.run_ci()
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("CI_DATA_ROOT is non-empty", proc.stderr)
        self.assertIn("CI_FRESH=1", proc.stderr)
        self.assertEqual(sentinel.read_text(), "do not touch\n")
        self.assertEqual(self.hf_calls(), [], "downloader must not run on a non-empty root")

    def test_ci_fresh_refused_outside_wipe_prefix(self):
        _, sentinel = self.populated_dir("data", "still here\n")
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        proc = self.run_ci(CI_FRESH="1", CI_WIPE_PREFIX=str(elsewhere))
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("CI_FRESH=1 refused", proc.stderr)
        self.assertTrue(sentinel.exists() and sentinel.read_text() == "still here\n")
        self.assertEqual(self.hf_calls(), [])

    def test_ci_fresh_wipes_data_root_under_prefix_then_passes(self):
        _, sentinel = self.populated_dir("data", "stale\n")
        proc = self.run_ci(CI_FRESH="1")   # CI_WIPE_PREFIX defaults to self.tmp in run_ci
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertIn(f"CI_FRESH=1: wiping {self.data.resolve()}", proc.stdout)
        self.assertFalse(sentinel.exists())
        self.assert_material_assembled(145)

    # -- (iii, symlinked roots) `-d` follows a link, plain `find` does not ---
    def test_symlink_to_populated_dir_is_refused_like_a_nonempty_root(self):
        outside, sentinel = self.populated_dir("outside")
        os.symlink(outside, self.data)
        proc = self.run_ci()
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("CI_DATA_ROOT is non-empty", proc.stderr)
        self.assertIn(f"{self.data} -> {outside.resolve()}", proc.stderr)   # both paths named
        self.assertEqual(sentinel.read_text(), "do not touch\n")
        self.assertTrue(self.data.is_symlink())
        self.assertEqual(self.hf_calls(), [], "downloader must not run into a populated link target")

    def test_ci_fresh_gate_applies_to_the_link_target_not_the_link(self):
        # link inside the allowed prefix, target outside it -> must refuse
        prefix = self.tmp / "prefix"
        prefix.mkdir()
        link = prefix / "data"
        outside, sentinel = self.populated_dir("outside", "still here\n")
        os.symlink(outside, link)
        proc = self.run_ci(CI_DATA_ROOT=str(link), CI_FRESH="1", CI_WIPE_PREFIX=str(prefix))
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("CI_FRESH=1 refused", proc.stderr)
        self.assertIn(str(outside.resolve()), proc.stderr)
        self.assertEqual(sentinel.read_text(), "still here\n")
        self.assertTrue(link.is_symlink() and outside.is_dir())
        self.assertEqual(self.hf_calls(), [])

    def test_ci_fresh_on_symlinked_root_empties_target_and_keeps_link(self):
        outside, sentinel = self.populated_dir("outside", "stale\n")
        os.symlink(outside, self.data)
        proc = self.run_ci(CI_FRESH="1")   # the self.tmp prefix covers the target
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertIn(f"CI_FRESH=1: wiping {outside.resolve()}", proc.stdout)
        self.assertFalse(sentinel.exists())
        self.assertTrue(self.data.is_symlink() and Path(os.readlink(self.data)) == outside)
        self.assert_material_assembled(145)            # through the link ...
        self.assert_material_assembled(226, outside)   # ... and in the target

    def test_symlink_to_empty_dir_is_accepted(self):
        target = self.tmp / "empty_target"
        target.mkdir()
        os.symlink(target, self.data)
        proc = self.run_ci()
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertTrue(self.data.is_symlink())
        self.assert_material_assembled(226, target)
        self.assertEqual(len(self.hf_calls()), 1)

    def test_dangling_symlink_root_is_rejected_with_a_clear_message(self):
        os.symlink(self.tmp / "nowhere", self.data)
        proc = self.run_ci()
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("dangling symlink", proc.stderr)
        self.assertIn(str(self.tmp / "nowhere"), proc.stderr)
        self.assertEqual(self.hf_calls(), [])

    # -- downloader / input failures propagate as exit 1 ---------------------
    def test_material_absent_upstream_fails_in_downloader(self):
        proc = self.run_ci(MATERIALS="145 9999")
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("requested material 9999 was not downloaded", proc.combined)
        self.assertIn("download_dataset_stage1.sh failed", proc.stderr)

    def test_non_numeric_material_id_rejected(self):
        proc = self.run_ci(MATERIALS="145 abc")
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("numeric material IDs", proc.stderr)
        self.assertEqual(self.hf_calls(), [])

    # -- defaults: everything hangs off CI_ROOT, which hangs off TMPDIR ------
    def test_default_locations_derive_from_ci_root(self):
        unset = dict(CI_DATA_ROOT=None, CI_OUTPUT_ROOT=None, CI_WIPE_PREFIX=None)
        root = self.tmp / "ci_root"
        proc = self.run_ci(CI_ROOT=str(root), **unset)
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assert_material_assembled(145, root / "data")
        self.assertTrue((root / "out" / "validate.log").is_file())
        self.assertIn(f"data={root / 'data'} ", proc.stdout)
        # CI_WIPE_PREFIX defaults to CI_ROOT, so CI_FRESH=1 may clear $CI_ROOT/data on a rerun
        proc = self.run_ci(CI_ROOT=str(root), CI_FRESH="1", **unset)
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertIn(f"CI_FRESH=1: wiping {(root / 'data').resolve()}", proc.stdout)
        # without CI_ROOT: $TMPDIR/robocloth-ci
        tmpdir = self.tmp / "tmpdir"
        tmpdir.mkdir()
        proc = self.run_ci(CI_ROOT=None, TMPDIR=str(tmpdir), **unset)
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assert_material_assembled(226, tmpdir / "robocloth-ci" / "data")
        self.assertTrue((tmpdir / "robocloth-ci" / "out" / "download.log").is_file())

    # -- static checks -------------------------------------------------------
    def test_bash_syntax(self):
        proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_shellcheck(self):
        sc = os.environ.get("SHELLCHECK") or shutil.which("shellcheck")
        if not sc:
            self.skipTest("shellcheck not available (set SHELLCHECK=/path/to/shellcheck)")
        proc = subprocess.run([sc, "-S", "warning", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class CiStage1TrainStepPlumbing(_CiCase):
    """Step 3 against the FAKE train_stage1.sh (download + validation stay real)."""
    skip_train = False

    def record(self):
        return json.loads((self.out / "fake_train.json").read_text())

    def test_train_step_wiring_and_pass_summary(self):
        proc = self.run_ci(self.scratch_repo(), CI_TRAIN_BATCHES="7", CI_TRAIN_ARGS="data.rays_num=1000 model.foo=bar")
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertIn("PASS: stage-1 data path OK", proc.stdout)
        self.assertIn("train/total_loss: 2 values, first 0.7300 -> last 0.5100 (all finite)", proc.stdout)
        self.assertIn("epoch=0-step=50.ckpt", proc.stdout)
        self.assertEqual((self.out / "ci_training_list.txt").read_text(), "145\n226\n")
        self.assertTrue((self.out / "train.log").is_file())
        rec = self.record()
        self.assertEqual(rec["env"], {"DATA_ROOT": str(self.data), "OUTPUT_ROOT": str(self.out),
                                      "TRAINING_LIST": str(self.out / "ci_training_list.txt"), "EXP_NAME": "ci_stage1"})
        for arg in ("model.trainer.max_epochs=1", "model.trainer.check_val_every_n_epoch=1",
                    "model.trainer.limit_train_batches=7",
                    "model.logger._target_=pytorch_lightning.loggers.CSVLogger", "~model.logger.project"):
            self.assertIn(arg, rec["argv"])
        self.assertEqual(rec["argv"][-2:], ["data.rays_num=1000", "model.foo=bar"])
        # the bare `python` train_stage1.sh calls is the job-local wrapper around ROBOCLOTH_PYTHON
        self.assertEqual(rec["python_on_path"], str(self.out / "pybin" / "python"))
        self.assertEqual(Path(rec["sys_executable"]).resolve(), Path(sys.executable).resolve())

    def test_python_wrapper_keeps_a_venv_interpreter_in_its_venv(self):
        # a `python` SYMLINK to venv/bin/python would start the base interpreter
        # (no pyvenv.cfg next to the link); the wrapper must not have that flaw
        venv = self.tmp / "venv"
        r = subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not (venv / "bin" / "python").exists():
            self.skipTest(f"cannot create a venv here: {r.stderr.strip()}")
        proc = self.run_ci(self.scratch_repo(), ROBOCLOTH_PYTHON=str(venv / "bin" / "python"))
        self.assertEqual(proc.returncode, 0, proc.combined)
        rec = self.record()
        self.assertEqual(Path(rec["sys_executable"]).parent.resolve(), (venv / "bin").resolve())
        self.assertEqual(Path(rec["sys_prefix"]).resolve(), venv.resolve())
        self.assertTrue(rec["in_venv"], rec)

    def test_non_finite_loss_exits_3(self):
        proc = self.run_ci(self.scratch_repo(), FAKE_TRAIN_LOSS0="nan")
        self.assertEqual(proc.returncode, 3, proc.combined)
        self.assertIn("non-finite train/total_loss", proc.stdout)
        self.assertIn("post-training assertions failed", proc.stderr)

    def test_missing_checkpoint_exits_3(self):
        proc = self.run_ci(self.scratch_repo(), FAKE_TRAIN_NO_CKPT="1")
        self.assertEqual(proc.returncode, 3, proc.combined)
        self.assertIn("no checkpoint (*.ckpt)", proc.stdout)

    def test_training_crash_exits_3(self):
        proc = self.run_ci(self.scratch_repo(), FAKE_TRAIN_EXIT="7")
        self.assertEqual(proc.returncode, 3, proc.combined)
        self.assertIn("stage-1 training exited with 7", proc.stderr)
        self.assertFalse((self.out / "ci_stage1").exists())

    # -- a rerun into the same CI_OUTPUT_ROOT must be judged on its own files -
    def test_rerun_without_checkpoint_into_same_output_root_exits_3(self):
        script = self.scratch_repo()
        proc = self.run_ci(script)                                   # run 1: passes
        self.assertEqual(proc.returncode, 0, proc.combined)
        old_ckpt = self.out / "ci_stage1" / "checkpoints" / "epoch=0-step=50.ckpt"
        self.assertTrue(old_ckpt.is_file())
        # run 2, same CI_OUTPUT_ROOT (CI_FRESH=1: the data root is populated from run 1); its
        # training writes a metrics.csv but NO checkpoint -> must fail, not pass on run 1's ckpt
        proc = self.run_ci(script, CI_FRESH="1", FAKE_TRAIN_NO_CKPT="1")
        self.assertEqual(proc.returncode, 3, proc.combined)
        self.assertIn(f"removing previous run's {self.out / 'ci_stage1'}", proc.stdout)
        self.assertIn("no checkpoint (*.ckpt)", proc.stdout)
        self.assertIn("post-training assertions failed", proc.stderr)
        self.assertNotIn("PASS: stage-1 data path OK", proc.stdout)
        self.assertFalse(old_ckpt.exists(), "run 1's checkpoint must not survive into run 2")
        self.assertEqual(self.record()["python_on_path"], str(self.out / "pybin" / "python"))  # wrapper recreated

    def test_checkpoint_predating_this_run_is_not_accepted(self):
        # second line of defence behind the wipe: a *.ckpt older than the start marker
        # (a leftover the wipe could not see) does not satisfy the assertions
        proc = self.run_ci(self.scratch_repo(), FAKE_TRAIN_STALE_CKPT="1")
        self.assertEqual(proc.returncode, 3, proc.combined)
        self.assertIn("no checkpoint (*.ckpt) written by this run", proc.stdout)
        self.assertIn("1 older file(s) from an earlier run ignored", proc.stdout)

    def test_robocloth_python_accepts_a_bare_command_name(self):
        wrapper = self.fakebin / "robocloth-python"                  # on PATH via fakebin
        wrapper.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
        wrapper.chmod(0o755)
        proc = self.run_ci(self.scratch_repo(), ROBOCLOTH_PYTHON="robocloth-python")
        self.assertEqual(proc.returncode, 0, proc.combined)
        self.assertEqual(Path(self.record()["sys_executable"]).resolve(), Path(sys.executable).resolve())
        proc = self.run_ci(ROBOCLOTH_PYTHON="no-such-python-xyz")
        self.assertEqual(proc.returncode, 1, proc.combined)
        self.assertIn("ROBOCLOTH_PYTHON='no-such-python-xyz' is not an executable on PATH", proc.stderr)
        self.assertEqual(len(self.hf_calls()), 1, "only the first run may have reached the downloader")


if __name__ == "__main__":
    unittest.main()
