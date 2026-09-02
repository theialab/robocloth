"""Fail-closed behaviour of the paper-table reproduction drivers.

Covers scripts/run_table_ours.sh + scripts/comparisons/run_table_ubo.sh (drivers),
scripts/eval_stage2.sh + scripts/comparisons/eval_stage2_ubo.sh (per-cell evaluators)
and scripts/collect_eval_results{,_ubo}.py (collectors / result-JSON schema).

No GPU and no training environment are needed: the drivers are pointed at
stub_eval.sh through ROBOCLOTH_EVAL_SCRIPT, and the evaluators at
fake_train_python.sh (a stand-in for the ROBOCLOTH_PYTHON interpreter that
emulates train.py's CSV logger).  Scratch files go into a fresh temporary
directory per test (under $ROBOCLOTH_TEST_TMP if that is set, else the system
temp dir) and are removed after each test.  Nothing is written inside
the repository: the in-process imports below set sys.dont_write_bytecode and
every subprocess gets PYTHONDONTWRITEBYTECODE=1 (clean_env()).

Run:  <python> -B -m unittest discover -s tests/repro_drivers -v
      (-B keeps this module's own __pycache__ out of the repository)
"""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SCRIPTS = os.path.join(REPO, "scripts")
STUB = os.path.join(HERE, "stub_eval.sh")
FAKE_PYTHON = os.path.join(HERE, "fake_train_python.sh")
CI_TMP = os.environ.get("ROBOCLOTH_TEST_TMP") or None       # scratch parent; None -> tempfile's default

sys.dont_write_bytecode = True              # no scripts/__pycache__ from the imports below
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "comparisons"))
import collect_eval_results as cer          # noqa: E402
import collect_eval_results_ubo as cer_ubo  # noqa: E402

MODELS = ["Ours", "Bonn", "MERL", "PBR"]
KINDS = {
    "ours": dict(driver=os.path.join(SCRIPTS, "run_table_ours.sh"),
                 collector=os.path.join(SCRIPTS, "collect_eval_results.py"),
                 results="eval_results", log="[run_table]", paper=cer.PAPER, mats=list(cer.PAPER)),
    "ubo": dict(driver=os.path.join(SCRIPTS, "comparisons", "run_table_ubo.sh"),
                collector=os.path.join(SCRIPTS, "comparisons", "collect_eval_results_ubo.py"),
                results="eval_results_ubo", log="[run_table_ubo]", paper=cer_ubo.PAPER, mats=list(cer_ubo.PAPER)),
}
# Environment knobs the scripts read; scrubbed from the inherited environment.
KNOBS = ["DATA_ROOT", "CKPT_ROOT", "OUTPUT_ROOT", "MATERIALS", "MODELS", "TOLERANCE_DB", "FORCE",
         "ALLOW_MISSING", "ROBOCLOTH_PYTHON", "PYTHON", "REAL_PYTHON", "ROBOCLOTH_EVAL_SCRIPT", "EXP_NAME",
         "SAVE_ALL_VIEWS", "EMITTER_CALIB", "CKPT_SHA256", "PRIOR_METRICS"]


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if k not in KNOBS and not k.startswith("STUB_") and not k.startswith("FAKE_TRAIN_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"    # subprocess imports of scripts/*.py leave no __pycache__
    return env


def scratch_dir():
    if CI_TMP:
        os.makedirs(CI_TMP, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="repro_drivers_", dir=CI_TMP)


def write_ckpt(path, payload=b"fake checkpoint\n" * 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return path


def table_row(stdout, name):
    """The printed table row for material `name` (or 'average')."""
    for line in stdout.splitlines():
        if line.split("|")[0].strip() == name:
            return line
    raise AssertionError(f"no table row for {name!r} in:\n{stdout}")


class DriverCase(unittest.TestCase):
    """Runs a driver against a fake CKPT_ROOT and the stub evaluator."""

    def setUp(self):
        self._tmp = scratch_dir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.ckpt_root = {}

    # -- fixtures -------------------------------------------------------------
    def make_ckpts(self, kind, omit=(), extra=()):
        root = os.path.join(self.tmp, kind, "ckpts")
        for mat in KINDS[kind]["mats"]:
            for model in MODELS:
                if (mat, model) not in omit:
                    write_ckpt(os.path.join(root, mat, f"{model}_epoch{len(mat) * 10 + 2}.ckpt"))
        for mat, name in extra:
            write_ckpt(os.path.join(root, mat, name))
        self.ckpt_root[kind] = root
        return root

    def ckpt(self, kind, mat, model):
        (path,) = [os.path.join(self.ckpt_root[kind], mat, n) for n in os.listdir(os.path.join(self.ckpt_root[kind], mat))
                   if n.startswith(f"{model}_epoch")]
        return path

    def results_dir(self, kind):
        return os.path.join(self.tmp, kind, "out", KINDS[kind]["results"])

    def result(self, kind, mat, model):
        with open(os.path.join(self.results_dir(kind), f"{mat}_{model}.json")) as f:
            return json.load(f)

    def calls(self, kind):
        path = os.path.join(self.tmp, kind, "calls.log")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return f.read().split()

    def reset_calls(self, kind):
        open(os.path.join(self.tmp, kind, "calls.log"), "w").close()

    # -- running --------------------------------------------------------------
    def run_driver(self, kind, mats=None, models=None, **over):
        env = clean_env()
        os.makedirs(os.path.join(self.tmp, "data"), exist_ok=True)     # DATA_ROOT must be a directory
        env.update(CKPT_ROOT=self.ckpt_root[kind], DATA_ROOT=os.path.join(self.tmp, "data"),
                   OUTPUT_ROOT=os.path.join(self.tmp, kind, "out"), ROBOCLOTH_EVAL_SCRIPT=STUB,
                   ROBOCLOTH_PYTHON=sys.executable, STUB_TABLE=kind,
                   STUB_CALL_LOG=os.path.join(self.tmp, kind, "calls.log"))
        if mats is not None:
            env["MATERIALS"] = " ".join(mats)
        if models is not None:
            env["MODELS"] = " ".join(models)
        env.update({k: str(v) for k, v in over.items()})
        return subprocess.run(["bash", KINDS[kind]["driver"]], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=900)

    def assertRc(self, proc, rc):
        self.assertEqual(proc.returncode, rc, f"exit {proc.returncode}, expected {rc}\n--- stdout ---\n"
                                              f"{proc.stdout}\n--- stderr ---\n{proc.stderr}")


class TestDriverFailsClosed(DriverCase):

    def test_missing_checkpoint_aborts_before_any_evaluation(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1, m2 = k["mats"][:3]
                self.make_ckpts(kind, omit={(m1, "Bonn")})
                p = self.run_driver(kind, mats=[m0, m1, m2], models=["Ours", "Bonn"])
                self.assertRc(p, 4)
                self.assertIn(f"MISSING checkpoint for", p.stderr)
                self.assertIn(f"{m1}/Bonn", p.stderr)
                self.assertIn("aborting before any evaluation", p.stderr)
                self.assertEqual(self.calls(kind), [], "evaluator must not run when a checkpoint is missing")
                self.assertFalse(os.path.isdir(self.results_dir(kind)))

    def test_all_missing_checkpoints_are_listed_up_front(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind, omit={(m0, "MERL"), (m1, "Ours"), (m1, "PBR")})
                p = self.run_driver(kind, mats=[m0, m1])
                self.assertRc(p, 4)
                for cell in (f"{m0}/MERL", f"{m1}/Ours", f"{m1}/PBR"):
                    self.assertIn(cell, p.stderr)
                self.assertIn("3 cell(s) without a usable checkpoint", p.stderr)
                self.assertEqual(self.calls(kind), [])

    def test_ambiguous_checkpoint_is_an_error(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind, extra=[(m0, "Bonn_epoch99.ckpt")])
                p = self.run_driver(kind, mats=[m0], models=["Ours", "Bonn"])
                self.assertRc(p, 4)
                self.assertIn("AMBIGUOUS checkpoint", p.stderr)
                self.assertIn(f"{m0}/Bonn", p.stderr)
                self.assertEqual(self.calls(kind), [])

    def test_allow_missing_evaluates_the_rest_and_still_exits_nonzero(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind, omit={(m1, "Bonn")})
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"], ALLOW_MISSING=1)
                self.assertRc(p, 6)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours"])
                self.assertIn("MISSING", table_row(p.stdout, m1))
                self.assertIn("PASS", table_row(p.stdout, m0))
                self.assertIn(f"MISSING cells (1): {m1}/Bonn", p.stdout)
                self.assertIn("RESULT: FAIL", p.stdout)

    def test_all_within_tolerance_passes(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours", f"{m1}/Bonn"])
                for mat in (m0, m1):
                    row = table_row(p.stdout, mat)
                    self.assertEqual(row.count("PASS"), 2, row)
                    self.assertEqual(row.count("n/a"), 2, row)      # MERL / PBR not requested
                    self.assertNotIn("FAIL", row)
                self.assertIn("RESULT: PASS", p.stdout)
                self.assertIn(f"{k['log']} PASS: all 4 cells", p.stdout)
                # result JSONs carry provenance and no temp files are left behind
                r = self.result(kind, m0, "Ours")
                for key in ("ckpt", "ckpt_abs", "ckpt_size", "ckpt_mtime_ns", "git_head", "timestamp",
                            "hostname", "material", "model", "val_psnr"):
                    self.assertIn(key, r)
                self.assertEqual(r["ckpt_abs"], os.path.realpath(self.ckpt(kind, m0, "Ours")))
                self.assertFalse([n for n in os.listdir(self.results_dir(kind)) if n.endswith(".tmp")])

    def test_full_default_table_passes(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                self.make_ckpts(kind)
                p = self.run_driver(kind)                       # default MATERIALS x MODELS
                self.assertRc(p, 0)
                self.assertEqual(len(self.calls(kind)), len(k["mats"]) * len(MODELS))
                for mat in k["mats"] + ["average"]:
                    row = table_row(p.stdout, mat)
                    self.assertEqual(row.count("PASS"), 4, row)
                    for bad in ("FAIL", "MISSING", "n/a"):
                        self.assertNotIn(bad, row)
                self.assertIn(f"cells: {len(k['mats']) * 4} expected, {len(k['mats']) * 4} present, 0 missing; "
                              f"{len(k['mats']) * 4} PASS, 0 FAIL", p.stdout)
                self.assertIn("RESULT: PASS", p.stdout)

    def test_evaluator_failure_stops_the_run_with_the_command(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1, m2 = k["mats"][:3]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0, m1, m2], models=["Ours"], STUB_FAIL_CELLS=f"{m1}/Ours")
                self.assertRc(p, 5)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m1}/Ours"], "must stop at the failure")
                self.assertIn("failed (exit 17)", p.stderr)
                self.assertIn(f"bash {STUB} {m1} {self.ckpt(kind, m1, 'Ours')} Ours", p.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.results_dir(kind), f"{m1}_Ours.json")))

    def test_evaluator_that_writes_no_result_is_an_error(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], STUB_NO_JSON_CELLS=f"{m0}/Ours")
                self.assertRc(p, 5)
                self.assertIn("exited 0 but left no valid result", p.stderr)
                self.assertIn("no previous result", p.stderr)

    def test_evaluator_recording_another_checkpoint_is_an_error(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], STUB_WRONG_CKPT_CELLS=f"{m0}/Ours")
                self.assertRc(p, 5)
                self.assertIn("checkpoint path changed", p.stderr)

    def test_evaluator_recording_another_configuration_is_an_error(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], STUB_WRONG_CONFIG_CELLS=f"{m0}/Ours")
                self.assertRc(p, 5)
                self.assertIn("experiment differs from this run: recorded 'stage2_debug'", p.stderr)

    def test_force_with_evaluator_writing_nothing_is_an_error(self):
        """FORCE=1 must not let the previous (fresh, matching) JSON pass the post-evaluation check."""
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                self.assertRc(self.run_driver(kind, mats=[m0], models=["Ours"]), 0)
                json_path = os.path.join(self.results_dir(kind), f"{m0}_Ours.json")
                with open(json_path) as f:
                    old = json.load(f)
                self.reset_calls(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], FORCE=1, STUB_NO_JSON_CELLS=f"{m0}/Ours")
                self.assertRc(p, 5)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours"])
                self.assertIn("exited 0 but left no valid result", p.stderr)
                self.assertIn(f"previous result kept as {json_path}.prev", p.stdout)
                self.assertFalse(os.path.exists(json_path), "the superseded JSON must not stay in place")
                with open(json_path + ".prev") as f:
                    self.assertEqual(json.load(f), old)
                # nothing to reuse afterwards either: the next run evaluates the cell again
                self.reset_calls(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours"])
                self.assertIn("no previous result", p.stdout)

    def test_stale_result_is_set_aside_before_reevaluation(self):
        """A stale JSON never survives in place, even when its re-evaluation fails."""
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                self.assertRc(self.run_driver(kind, mats=[m0], models=["Ours"]), 0)
                ck = self.ckpt(kind, m0, "Ours")
                st = os.stat(ck)
                os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
                p = self.run_driver(kind, mats=[m0], models=["Ours"], STUB_FAIL_CELLS=f"{m0}/Ours")
                self.assertRc(p, 5)
                json_path = os.path.join(self.results_dir(kind), f"{m0}_Ours.json")
                self.assertFalse(os.path.exists(json_path))
                self.assertTrue(os.path.exists(json_path + ".prev"))
                self.assertEqual(cer.load_results(self.results_dir(kind)), {}, ".prev files are invisible to the collector")

    def test_result_for_another_configuration_is_reevaluated(self):
        """Same checkpoint, but the JSON came from another experiment / dataset / overrides / cell."""
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                self.assertRc(self.run_driver(kind, mats=[m0, m1], models=["Ours", "PBR"]), 0)
                rd = self.results_dir(kind)

                def rewrite(mat, model, **changes):
                    path = os.path.join(rd, f"{mat}_{model}.json")
                    with open(path) as f:
                        r = json.load(f)
                    r.update(changes)
                    with open(path, "w") as f:
                        json.dump(r, f)

                rewrite(m0, "Ours", overrides=["data.debug_num=10"])                     # smoke-style subset run
                rewrite(m0, "PBR", experiment="eval_ubo" if kind == "ubo" else "eval_stage2")  # PBR cell, neural config
                rewrite(m1, "Ours", dataset_folder=os.path.join(self.tmp, "other_data", m1))  # another DATA_ROOT
                rewrite(m1, "PBR", material=m0)                                          # file name vs content
                self.reset_calls(kind)
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "PBR"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m0}/PBR", f"{m1}/Ours", f"{m1}/PBR"])
                for needle in ("overrides differs from this run: recorded 'data.debug_num=10', now ''",
                               "experiment differs from this run", "dataset_folder differs from this run",
                               f"material differs from this run: recorded '{m0}', now '{m1}'"):
                    self.assertIn(needle, p.stdout)
                for mat, model in ((m0, "Ours"), (m0, "PBR"), (m1, "Ours"), (m1, "PBR")):
                    self.assertTrue(os.path.exists(os.path.join(rd, f"{mat}_{model}.json.prev")))
                    self.assertEqual(self.result(kind, mat, model)["overrides"], [])
                self.assertIn("RESULT: PASS", p.stdout)

    def test_bad_environment_is_rejected_before_anything_runs(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], ROBOCLOTH_PYTHON="/nonexistent/python")
                self.assertRc(p, 2)
                self.assertIn("interpreter not found or not executable: /nonexistent/python", p.stderr)
                p = self.run_driver(kind, mats=[m0], models=["Ours"], DATA_ROOT=os.path.join(self.tmp, "no_such_data"))
                self.assertRc(p, 2)
                self.assertIn("DATA_ROOT is not a directory", p.stderr)
                self.assertEqual(self.calls(kind), [])
                self.assertFalse(os.path.isdir(self.results_dir(kind)))
                # the generic PYTHON variable (autoconf, node-gyp, ...) does not select the interpreter
                p = self.run_driver(kind, mats=[m0], models=["Ours"], PYTHON="/nonexistent/python")
                self.assertRc(p, 0)


class TestStaleResults(DriverCase):
    """Reuse of an existing result JSON is tied to the checkpoint's path/size/mtime."""

    def first_run(self, kind, mats, models=("Ours", "Bonn")):
        p = self.run_driver(kind, mats=mats, models=list(models))
        self.assertRc(p, 0)
        self.reset_calls(kind)

    def test_fresh_matching_result_is_reused(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                self.first_run(kind, [m0, m1])
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [], "stub must not be called for fresh results")
                self.assertEqual(p.stdout.count("checkpoint unchanged"), 4)
                self.assertIn("RESULT: PASS", p.stdout)

    def test_changed_mtime_triggers_reevaluation_of_that_cell_only(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                self.first_run(kind, [m0, m1])
                ck = self.ckpt(kind, m1, "Ours")
                st = os.stat(ck)
                os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 * 10**9))
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m1}/Ours"])
                self.assertIn("checkpoint mtime changed", p.stdout)
                self.assertEqual(self.result(kind, m1, "Ours")["ckpt_mtime_ns"], os.stat(ck).st_mtime_ns)
                prev = [n for n in os.listdir(self.results_dir(kind)) if n.endswith(".prev")]
                self.assertEqual(prev, [f"{m1}_Ours.json.prev"], "only the stale cell's JSON is set aside")

    def test_changed_size_triggers_reevaluation(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                self.first_run(kind, [m0])
                ck = self.ckpt(kind, m0, "Bonn")
                st = os.stat(ck)
                with open(ck, "ab") as f:
                    f.write(b"more weights\n")
                os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns))    # keep mtime: size alone must trigger
                p = self.run_driver(kind, mats=[m0], models=["Ours", "Bonn"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Bonn"])
                self.assertIn("checkpoint size changed", p.stdout)

    def test_moved_checkpoint_root_triggers_reevaluation(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                root = self.make_ckpts(kind)
                self.first_run(kind, [m0])
                moved = root + "_moved"
                os.rename(root, moved)
                self.ckpt_root[kind] = moved
                p = self.run_driver(kind, mats=[m0], models=["Ours", "Bonn"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m0}/Bonn"])
                self.assertIn("checkpoint path changed", p.stdout)

    def test_legacy_result_without_provenance_is_reevaluated(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                os.makedirs(self.results_dir(kind))
                legacy = {"material": m0, "model": "Ours", "ckpt": self.ckpt(kind, m0, "Ours"),
                          "val_psnr": k["paper"][m0]["Ours"], "val_loss": 0.01}   # pre-B3 schema
                with open(os.path.join(self.results_dir(kind), f"{m0}_Ours.json"), "w") as f:
                    json.dump(legacy, f)
                p = self.run_driver(kind, mats=[m0], models=["Ours"])
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours"])
                self.assertIn("records no checkpoint provenance", p.stdout)
                self.assertIn("ckpt_mtime_ns", self.result(kind, m0, "Ours"))

    def test_force_reevaluates_everything(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                self.first_run(kind, [m0, m1])
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"], FORCE=1)
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours", f"{m1}/Bonn"])
                self.assertEqual(p.stdout.count("FORCE=1: re-evaluating"), 4)


class TestTolerance(DriverCase):

    def test_breach_fails_with_fail_cell_and_tolerance_env_is_honored(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0, m1 = k["mats"][:2]
                self.make_ckpts(kind)
                offsets = json.dumps({f"{m1}/Bonn": 0.2})
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"], STUB_PSNR_OFFSETS=offsets)
                self.assertRc(p, 7)
                row = table_row(p.stdout, m1)
                self.assertIn("FAIL", row)
                self.assertIn("+0.20", row)
                self.assertNotIn("FAIL", table_row(p.stdout, m0))
                self.assertRegex(p.stdout, rf"FAIL {m1}/Bonn: repro .* \(\+0\.2000 dB, \|diff\| > 0\.05 dB\)")
                self.assertIn("RESULT: FAIL", p.stdout)
                self.assertIn(f"{k['log']} FAILED (exit 7)", p.stderr)
                # same (reused) results pass under a looser explicit tolerance
                self.reset_calls(kind)
                p = self.run_driver(kind, mats=[m0, m1], models=["Ours", "Bonn"], TOLERANCE_DB=0.5)
                self.assertRc(p, 0)
                self.assertEqual(self.calls(kind), [])
                self.assertIn("tolerance: |repro - paper| <= 0.50 dB", p.stdout)
                self.assertEqual(table_row(p.stdout, m1).count("PASS"), 2)

    def test_edge_of_tolerance(self):
        for kind, k in KINDS.items():
            with self.subTest(kind=kind):
                m0 = k["mats"][0]
                self.make_ckpts(kind)
                p = self.run_driver(kind, mats=[m0], models=["Ours", "Bonn"],
                                    STUB_PSNR_OFFSETS=json.dumps({f"{m0}/Ours": -0.04, f"{m0}/Bonn": 0.05}))
                self.assertRc(p, 0)
                p = self.run_driver(kind, mats=[m0], models=["MERL"], STUB_PSNR_OFFSETS=json.dumps({f"{m0}/MERL": -0.051}))
                self.assertRc(p, 7)


class TestCollectorAndSchema(unittest.TestCase):
    """collect_eval_results.py used directly: gaps, tolerance, arguments, --reuse-check, write_result."""

    def setUp(self):
        self._tmp = scratch_dir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.ckpt = write_ckpt(os.path.join(self.tmp, "ckpts", "226", "Ours_epoch1.ckpt"))
        self.results = os.path.join(self.tmp, "eval_results")
        os.makedirs(self.results)

    def write(self, mat, model, psnr, kind="ours", ckpt=None):
        ckpt = ckpt or self.ckpt
        return cer.write_result(os.path.join(self.results, f"{mat}_{model}.json"), material=mat, model=model,
                                ckpt=ckpt, val_psnr=psnr, val_loss=0.01, experiment="test")

    def collect(self, *args, kind="ours", **env):
        e = clean_env()
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run([sys.executable, KINDS[kind]["collector"], self.results, *args],
                              env=e, capture_output=True, text=True)

    def test_missing_cell_is_listed_and_exits_6(self):
        for mat, model in (("226", "Ours"), ("226", "Bonn"), ("314", "Ours")):
            self.write(mat, model, cer.PAPER[mat][model])
        p = self.collect("--materials", "226 314", "--models", "Ours Bonn")
        self.assertEqual(p.returncode, 6, p.stdout + p.stderr)
        self.assertIn("MISSING cells (1): 314/Bonn", p.stdout)
        self.assertIn("MISSING", table_row(p.stdout, "314"))
        self.assertIn("cells: 4 expected, 3 present, 1 missing; 3 PASS, 0 FAIL", p.stdout)

    def test_tolerance_beats_missing_in_exit_code(self):
        self.write("226", "Ours", cer.PAPER["226"]["Ours"] + 1.0)
        p = self.collect("--materials", "226", "--models", "Ours Bonn")
        self.assertEqual(p.returncode, 7, p.stdout)
        self.assertIn("MISSING cells (1): 226/Bonn", p.stdout)
        self.assertIn("FAIL 226/Ours", p.stdout)

    def test_null_psnr_and_unreadable_json_count_as_missing(self):
        with open(os.path.join(self.results, "226_Ours.json"), "w") as f:
            json.dump({"material": "226", "model": "Ours", "val_psnr": None}, f)
        with open(os.path.join(self.results, "226_Bonn.json"), "w") as f:
            f.write("{not json")
        p = self.collect("--materials", "226", "--models", "Ours Bonn")
        self.assertEqual(p.returncode, 6, p.stdout + p.stderr)
        self.assertIn("MISSING cells (2): 226/Ours 226/Bonn", p.stdout)
        self.assertIn("ignoring unreadable result", p.stderr)

    def test_unknown_material_or_model_is_bad_args(self):
        self.assertEqual(self.collect("--materials", "999").returncode, 2)
        self.assertEqual(self.collect("--models", "Nope").returncode, 2)
        self.assertEqual(self.collect("--tolerance-db", "-1").returncode, 2)
        self.assertEqual(self.collect(TOLERANCE_DB="abc").returncode, 2)

    def test_forced_missing_overrides_leftover_result(self):
        self.write("226", "Ours", cer.PAPER["226"]["Ours"])
        p = self.collect("--materials", "226", "--models", "Ours", "--missing", "226/Ours")
        self.assertEqual(p.returncode, 6, p.stdout)
        self.assertIn("leftover result(s) ignored", p.stdout)
        self.assertIn("MISSING", table_row(p.stdout, "226"))

    def test_results_outside_requested_subset_are_reported_not_checked(self):
        self.write("226", "Ours", cer.PAPER["226"]["Ours"])
        self.write("314", "PBR", cer.PAPER["314"]["PBR"] + 5.0)           # way off, but not requested
        p = self.collect("--materials", "226", "--models", "Ours")
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("note: 1 result(s) outside the requested MATERIALS x MODELS were not checked: 314/PBR", p.stdout)
        self.assertIn("n/a", table_row(p.stdout, "314"))

    def test_env_tolerance_default_and_full_table_average_row(self):
        for mat in cer.PAPER:
            for model in MODELS:
                self.write(mat, model, cer.PAPER[mat][model] + 0.05)     # at the edge in every cell
        p = self.collect()                                             # positional dir only, defaults
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual(table_row(p.stdout, "average").count("PASS"), 4)
        self.assertIn("average row: <= 0.06 dB", p.stdout)
        p = self.collect(TOLERANCE_DB=0.04)
        self.assertEqual(p.returncode, 7, p.stdout)
        self.assertIn("cells: 20 expected, 20 present, 0 missing; 0 PASS, 20 FAIL", p.stdout)

    def test_ubo_collector_shares_the_contract(self):
        ck = write_ckpt(os.path.join(self.tmp, "ckpts", "felt01", "Bonn_epoch60.ckpt"))
        with contextlib.redirect_stdout(io.StringIO()) as buf:          # importable, same collect()
            rc = cer.collect(self.results, cer_ubo.UBO, ["felt01"], ["Bonn"])
        self.assertEqual(rc, 6)
        self.assertIn("MISSING cells (1): felt01/Bonn", buf.getvalue())
        self.write("felt01", "Bonn", 33.596622467041016, ckpt=ck)      # the beta-test number
        p = self.collect("--materials", "felt01", "--models", "Bonn", kind="ubo")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("Cross-dataset transfer to UBO2014", p.stdout)
        self.assertEqual(table_row(p.stdout, "felt01").count("PASS"), 1)
        p = self.collect("--materials", "felt01", "--models", "Bonn Ours", kind="ubo")
        self.assertEqual(p.returncode, 6)
        p = self.collect("--reuse-check", os.path.join(self.results, "felt01_Bonn.json"), ck, kind="ubo")
        self.assertEqual(p.returncode, 0, p.stdout)

    def test_write_result_is_atomic_and_records_provenance(self):
        out = os.path.join(self.results, "226_Ours.json")
        r = self.write("226", "Ours", 29.24)
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".tmp"))
        with open(out) as f:
            self.assertEqual(json.load(f), r)
        st = os.stat(self.ckpt)
        self.assertEqual(r["ckpt_abs"], os.path.realpath(self.ckpt))
        self.assertEqual((r["ckpt_size"], r["ckpt_mtime_ns"]), (st.st_size, st.st_mtime_ns))
        self.assertTrue(r["git_head"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", r["git_head"]), r["git_head"])
        self.assertIn("git_dirty", r)
        self.assertRegex(r["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")
        self.assertEqual((r["material"], r["model"], r["val_psnr"], r["experiment"]), ("226", "Ours", 29.24, "test"))
        self.assertEqual(r["hostname"], os.uname().nodename)
        # optional sha256
        os.environ["CKPT_SHA256"] = "1"
        try:
            r2 = self.write("226", "Ours", 29.24)
        finally:
            del os.environ["CKPT_SHA256"]
        self.assertRegex(r2["ckpt_sha256"], r"^[0-9a-f]{64}$")

    def test_reuse_check_reasons(self):
        out = os.path.join(self.results, "226_Ours.json")
        self.assertEqual(cer.check_reuse(out, self.ckpt)[0], False)
        self.assertIn("no previous result", cer.check_reuse(out, self.ckpt)[1])
        self.write("226", "Ours", 29.24)
        ok, reason = cer.check_reuse(out, self.ckpt)
        self.assertTrue(ok, reason)
        self.assertIn("checkpoint unchanged", reason)
        # another path to the same bytes is a different checkpoint
        other = write_ckpt(os.path.join(self.tmp, "elsewhere", "Ours_epoch1.ckpt"))
        self.assertIn("checkpoint path changed", cer.check_reuse(out, other)[1])
        # size
        st = os.stat(self.ckpt)
        with open(self.ckpt, "ab") as f:
            f.write(b"x")
        os.utime(self.ckpt, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertIn("checkpoint size changed", cer.check_reuse(out, self.ckpt)[1])
        # mtime
        self.write("226", "Ours", 29.24)
        os.utime(self.ckpt, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        self.assertIn("checkpoint mtime changed", cer.check_reuse(out, self.ckpt)[1])
        # missing value / provenance / checkpoint
        with open(out, "w") as f:
            json.dump({"material": "226", "model": "Ours", "ckpt": self.ckpt, "val_psnr": None}, f)
        self.assertIn("no val_psnr", cer.check_reuse(out, self.ckpt)[1])
        with open(out, "w") as f:
            json.dump({"material": "226", "model": "Ours", "ckpt": self.ckpt, "val_psnr": 29.24}, f)
        self.assertIn("no checkpoint provenance", cer.check_reuse(out, self.ckpt)[1])
        self.write("226", "Ours", 29.24)
        self.assertIn("checkpoint not readable", cer.check_reuse(out, self.ckpt + ".gone")[1])
        with open(out, "w") as f:
            f.write("{broken")
        self.assertIn("unreadable", cer.check_reuse(out, self.ckpt)[1])

    def test_reuse_check_expectations(self):
        out = os.path.join(self.results, "226_Ours.json")
        self.write("226", "Ours", 29.24)                          # experiment="test"; no dataset_folder / overrides
        want = dict(material="226", model="Ours", experiment="test")
        self.assertTrue(cer.check_reuse(out, self.ckpt, want)[0])
        self.assertIn("experiment differs from this run: recorded 'test', now 'eval_stage2'",
                      cer.check_reuse(out, self.ckpt, dict(want, experiment="eval_stage2"))[1])
        self.assertIn("does not record 'overrides'", cer.check_reuse(out, self.ckpt, dict(want, overrides=""))[1])
        # lists are compared space-joined, PATH_KEYS by realpath (symlink + trailing slash)
        data = os.path.join(self.tmp, "data")
        os.makedirs(os.path.join(data, "226"))
        link = os.path.join(self.tmp, "data_link")
        os.symlink(data, link)
        cer.write_result(out, material="226", model="Ours", ckpt=self.ckpt, val_psnr=29.24, experiment="eval_stage2",
                         dataset_folder=os.path.join(data, "226"), overrides=["a=1", "b=2"])
        ok, reason = cer.check_reuse(out, self.ckpt, dict(dataset_folder=os.path.join(link, "226") + "/",
                                                          overrides="a=1 b=2"))
        self.assertTrue(ok, reason)
        self.assertIn("overrides differs from this run: recorded 'a=1 b=2', now ''",
                      cer.check_reuse(out, self.ckpt, dict(overrides=""))[1])
        self.assertIn("dataset_folder differs from this run",
                      cer.check_reuse(out, self.ckpt, dict(dataset_folder=os.path.join(data, "314")))[1])
        # CLI form used by the drivers
        p = self.collect("--reuse-check", out, self.ckpt, "--expect", "material=226", "overrides=a=1 b=2")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = self.collect("--reuse-check", out, self.ckpt, "--expect", "model=PBR")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("model differs from this run: recorded 'Ours', now 'PBR'", p.stdout)
        self.assertEqual(self.collect("--reuse-check", out, self.ckpt, "--expect", "novalue").returncode, 2)
        self.assertEqual(self.collect("--expect", "material=226").returncode, 2)

    def test_metrics_snapshot_and_fresh_selection(self):
        """metrics_files()/fresh_metrics_csv(): only a metrics.csv that appeared or changed since the snapshot counts."""
        log = os.path.join(self.tmp, "Eval_Ours145_Ours", "Eval_Ours145_Ours")   # CSVLogger save_dir/name

        def put(version, text):
            d = os.path.join(log, version)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "metrics.csv"), "w") as f:
                f.write(text)
            return os.path.join(d, "metrics.csv")

        self.assertEqual(cer.metrics_files(log), {})                     # tree does not exist yet
        self.assertIsNone(cer.fresh_metrics_csv(log, {}))
        v0 = put("version_0", "epoch,val/psnr\n0,11.11\n")
        p = self.collect("--metrics-snapshot", log)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        snap = json.loads(p.stdout)
        st = os.stat(v0)
        self.assertEqual(snap, {v0: [st.st_ino, st.st_mtime_ns, st.st_size]})
        self.assertIsNone(cer.fresh_metrics_csv(log, snap), "nothing new since the snapshot -> None")
        t0 = st.st_mtime_ns
        v1 = put("version_1", "epoch,val/psnr\n0,22.22\n")
        os.utime(v1, ns=(t0 + 10**9, t0 + 10**9))
        self.assertEqual(cer.fresh_metrics_csv(log, snap), v1)
        snap2 = cer.metrics_files(log)
        put("version_0", "epoch,val/psnr\n0,11.11\n1,33.33\n")           # rewritten in place ...
        os.utime(v0, ns=(t0, t0))                                          # ... with its old mtime restored
        self.assertEqual(cer.fresh_metrics_csv(log, snap2), v0, "size change alone makes it fresh")
        self.assertEqual(cer.fresh_metrics_csv(log, snap), v1, "both fresh vs. the first snapshot: newest wins")

    def test_subprocesses_write_no_bytecode_into_the_repo(self):
        """clean_env() sets PYTHONDONTWRITEBYTECODE=1: collector / stub imports leave scripts/ untouched."""
        dirs = [os.path.join(SCRIPTS, "__pycache__"), os.path.join(SCRIPTS, "comparisons", "__pycache__"),
                os.path.join(HERE, "__pycache__")]

        def listing():
            return {d: sorted((n, os.stat(os.path.join(d, n)).st_mtime_ns) for n in os.listdir(d))
                    if os.path.isdir(d) else None for d in dirs}

        before = listing()
        self.assertEqual(clean_env().get("PYTHONDONTWRITEBYTECODE"), "1")
        self.write("226", "Ours", 29.24)
        self.assertEqual(self.collect("--materials", "226", "--models", "Ours").returncode, 0)
        self.assertEqual(self.collect("--materials", "felt01", "--models", "Bonn", kind="ubo").returncode, 6)
        e = clean_env()
        e.update(STUB_CALL_LOG=os.path.join(self.tmp, "calls.log"), OUTPUT_ROOT=self.tmp, DATA_ROOT=self.tmp,
                 ROBOCLOTH_PYTHON=sys.executable)
        self.assertEqual(subprocess.run(["bash", STUB, "226", self.ckpt, "Ours"], env=e, capture_output=True).returncode, 0)
        self.assertEqual(listing(), before)


class TestEvaluatorScripts(unittest.TestCase):
    """eval_stage2.sh / eval_stage2_ubo.sh with a fake `python` standing in for train.py."""

    EVAL_OURS = os.path.join(SCRIPTS, "eval_stage2.sh")
    EVAL_UBO = os.path.join(SCRIPTS, "comparisons", "eval_stage2_ubo.sh")

    def setUp(self):
        self._tmp = scratch_dir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.fake = shutil.copy(FAKE_PYTHON, os.path.join(self.tmp, "python"))
        os.chmod(self.fake, 0o755)
        self.ckpt = write_ckpt(os.path.join(self.tmp, "ckpts", "145", "Ours_epoch112.ckpt"))
        self.out = os.path.join(self.tmp, "out")
        self.log = os.path.join(self.tmp, "train_calls.log")
        os.makedirs(os.path.join(self.tmp, "data", "145"))

    def run_eval(self, script, *args, **env):
        e = clean_env()
        e.update(ROBOCLOTH_PYTHON=self.fake, REAL_PYTHON=sys.executable, FAKE_TRAIN_LOG=self.log,
                 DATA_ROOT=os.path.join(self.tmp, "data"), OUTPUT_ROOT=self.out)
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", script, *args], env=e, cwd=self.tmp, capture_output=True, text=True)

    def train_argv(self):
        with open(self.log) as f:
            return f.read().splitlines()

    def load(self, rel):
        with open(os.path.join(self.out, rel)) as f:
            return json.load(f)

    def test_eval_stage2_writes_provenance_json(self):
        p = self.run_eval(self.EVAL_OURS, "145", self.ckpt, "Ours", "model.foo=1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        argv = self.train_argv()
        self.assertEqual(len(argv), 1)
        for token in ("+experiment=eval_stage2", f"model.ckpt_path={self.ckpt}", "data.valid_num=20", "model.foo=1",
                      "model.logger._target_=pytorch_lightning.loggers.CSVLogger"):
            self.assertIn(token, argv[0])
        r = self.load("eval_results/145_Ours.json")
        self.assertAlmostEqual(r["val_psnr"], 28.442256927490234)
        self.assertAlmostEqual(r["val_loss"], 0.0123)
        self.assertEqual((r["material"], r["model"], r["ckpt"], r["experiment"]), ("145", "Ours", self.ckpt, "eval_stage2"))
        self.assertEqual((r["valid_num"], r["overrides"]), (20, ["model.foo=1"]))
        self.assertEqual(r["dataset_folder"], os.path.join(self.tmp, "data", "145"))
        self.assertTrue(os.path.exists(r["metrics_csv"]))
        self.assertTrue(r["metrics_csv"].endswith("/Eval_Ours145_Ours/Eval_Ours145_Ours/version_0/metrics.csv"), r["metrics_csv"])
        st = os.stat(self.ckpt)
        self.assertEqual((r["ckpt_abs"], r["ckpt_size"], r["ckpt_mtime_ns"]),
                         (os.path.realpath(self.ckpt), st.st_size, st.st_mtime_ns))
        self.assertIn("git_head", r)
        self.assertFalse(os.path.exists(os.path.join(self.out, "eval_results", "145_Ours.json.tmp")))
        self.assertIn("val/psnr = 28.44 dB", p.stdout)
        # the produced JSON satisfies the driver's reuse check
        self.assertTrue(cer.check_reuse(os.path.join(self.out, "eval_results", "145_Ours.json"), self.ckpt)[0])

    def test_eval_stage2_pbr_tag_and_save_all_views(self):
        p = self.run_eval(self.EVAL_OURS, "145", self.ckpt, "PBR", SAVE_ALL_VIEWS=1)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("+experiment=eval_stage2_pbr", self.train_argv()[0])
        self.assertIn("data.valid_num=-1", self.train_argv()[0])
        r = self.load("eval_results/145_PBR.json")
        self.assertEqual((r["experiment"], r["valid_num"], r["model"]), ("eval_stage2_pbr", -1, "PBR"))

    def test_eval_stage2_ubo_writes_provenance_json(self):
        ck = write_ckpt(os.path.join(self.tmp, "ckpts", "felt01", "Bonn_epoch60.ckpt"))
        p = self.run_eval(self.EVAL_UBO, "felt01", ck, "Bonn", FAKE_TRAIN_PSNR="33.596622467041016")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        argv = self.train_argv()[0]
        for token in ("+experiment=eval_ubo", "data.btf_filename=felt01_W400xH400_L151xV151.btf", f"model.ckpt_path={ck}"):
            self.assertIn(token, argv)
        r = self.load("eval_results_ubo/felt01_Bonn.json")
        self.assertAlmostEqual(r["val_psnr"], 33.596622467041016)
        self.assertEqual((r["material"], r["model"], r["experiment"]), ("felt01", "Bonn", "eval_ubo"))
        self.assertEqual(r["dataset_folder"], os.path.join(self.tmp, "data"))
        self.assertEqual(r["btf_path"], os.path.join(self.tmp, "data", "felt01_W400xH400_L151xV151.btf"))
        self.assertEqual(r["overrides"], [])
        self.assertEqual(r["ckpt_abs"], os.path.realpath(ck))
        self.assertTrue(cer.check_reuse(os.path.join(self.out, "eval_results_ubo", "felt01_Bonn.json"), ck)[0])
        # PBR tag picks the PBR experiment
        p = self.run_eval(self.EVAL_UBO, "felt01", ck, "PBR")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("+experiment=eval_ubo_pbr", self.train_argv()[1])

    def test_train_failure_propagates_and_writes_nothing(self):
        for script, mat in ((self.EVAL_OURS, "145"), (self.EVAL_UBO, "felt01")):
            with self.subTest(script=os.path.basename(script)):
                p = self.run_eval(script, mat, self.ckpt, "Ours", FAKE_TRAIN_RC=3)
                self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.out, "eval_results", f"{mat}_Ours.json")))
                self.assertFalse(os.path.exists(os.path.join(self.out, "eval_results_ubo", f"{mat}_Ours.json")))

    def test_missing_metrics_csv_fails(self):
        for script, mat in ((self.EVAL_OURS, "145"), (self.EVAL_UBO, "felt01")):
            with self.subTest(script=os.path.basename(script)):
                p = self.run_eval(script, mat, self.ckpt, "Ours", FAKE_TRAIN_NO_CSV=1)
                self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
                self.assertIn("wrote no new metrics.csv", p.stderr)
                self.assertIn("0 metrics.csv from earlier runs", p.stderr)

    def test_stale_metrics_csv_from_an_earlier_run_is_never_reused(self):
        """EXP_NAME is reused across runs: a metrics.csv left by an earlier checkpoint must not be
        laundered into a JSON carrying the new checkpoint's provenance."""
        for script, mat, sub in ((self.EVAL_OURS, "145", "eval_results"), (self.EVAL_UBO, "felt01", "eval_results_ubo")):
            with self.subTest(script=os.path.basename(script)):
                ck = write_ckpt(os.path.join(self.tmp, "ckpts", mat, "Ours_epoch7.ckpt"))
                out_json = os.path.join(self.out, sub, f"{mat}_Ours.json")
                p = self.run_eval(script, mat, ck, "Ours", FAKE_TRAIN_PSNR="11.11")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertTrue(self.load(f"{sub}/{mat}_Ours.json")["metrics_csv"].endswith("/version_0/metrics.csv"))
                with open(out_json, "rb") as f:
                    before = f.read()
                # the checkpoint is replaced (new size + mtime); train.py "succeeds" but logs nothing
                st = os.stat(ck)
                with open(ck, "ab") as f:
                    f.write(b"new weights\n")
                os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
                p = self.run_eval(script, mat, ck, "Ours", FAKE_TRAIN_NO_CSV=1)
                self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
                self.assertIn("wrote no new metrics.csv", p.stderr)
                self.assertIn("1 metrics.csv from earlier runs", p.stderr)
                with open(out_json, "rb") as f:
                    self.assertEqual(f.read(), before, "the earlier JSON must be left byte-for-byte untouched")
                self.assertFalse(cer.check_reuse(out_json, ck)[0], "and it now reads as stale for the new checkpoint")
                # a run that does log lands in version_1, which is picked over the stale version_0
                p = self.run_eval(script, mat, ck, "Ours", FAKE_TRAIN_PSNR="22.22")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                r = self.load(f"{sub}/{mat}_Ours.json")
                self.assertAlmostEqual(r["val_psnr"], 22.22)
                self.assertTrue(r["metrics_csv"].endswith("/version_1/metrics.csv"), r["metrics_csv"])
                self.assertTrue(cer.check_reuse(out_json, ck)[0])

    def test_eval_scripts_reject_a_bad_interpreter_up_front(self):
        def train_calls():
            return len(self.train_argv()) if os.path.exists(self.log) else 0

        for script, mat in ((self.EVAL_OURS, "145"), (self.EVAL_UBO, "felt01")):
            with self.subTest(script=os.path.basename(script)):
                n = train_calls()
                p = self.run_eval(script, mat, self.ckpt, "Ours", ROBOCLOTH_PYTHON="/nonexistent/python")
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("interpreter not found or not executable: /nonexistent/python", p.stderr)
                self.assertEqual(train_calls(), n, "train.py must not have been attempted")
                # the generic PYTHON variable is ignored: train.py still runs under ROBOCLOTH_PYTHON
                p = self.run_eval(script, mat, self.ckpt, "Ours", PYTHON="/nonexistent/python")
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertEqual(train_calls(), n + 1)

    def test_missing_psnr_value_fails(self):
        for script, mat in ((self.EVAL_OURS, "145"), (self.EVAL_UBO, "felt01")):
            with self.subTest(script=os.path.basename(script)):
                p = self.run_eval(script, mat, self.ckpt, "Ours", FAKE_TRAIN_PSNR="")
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertIn("no val/psnr logged", p.stderr)
                self.assertFalse(os.path.exists(os.path.join(self.out, "eval_results", f"{mat}_Ours.json")))
                self.assertFalse(os.path.exists(os.path.join(self.out, "eval_results_ubo", f"{mat}_Ours.json")))

    def test_missing_checkpoint_fails_before_train(self):
        for script, mat in ((self.EVAL_OURS, "145"), (self.EVAL_UBO, "felt01")):
            with self.subTest(script=os.path.basename(script)):
                p = self.run_eval(script, mat, self.ckpt + ".gone", "Ours")
                self.assertEqual(p.returncode, 4, p.stdout + p.stderr)
                self.assertIn("checkpoint not found", p.stderr)
                self.assertFalse(os.path.exists(self.log))

    def test_drivers_end_to_end_with_real_evaluators(self):
        """Driver -> real eval script -> fake train.py -> real collector; only train.py is faked."""
        ck_ubo = write_ckpt(os.path.join(self.tmp, "ckpts", "felt01", "Bonn_epoch60.ckpt"))
        for kind, mat, model, psnr in (("ours", "145", "Ours", "28.442256927490234"),
                                       ("ubo", "felt01", "Bonn", "33.596622467041016")):
            with self.subTest(kind=kind):
                e = clean_env()
                e.update(ROBOCLOTH_PYTHON=self.fake, REAL_PYTHON=sys.executable, FAKE_TRAIN_LOG=self.log,
                         FAKE_TRAIN_PSNR=psnr, DATA_ROOT=os.path.join(self.tmp, "data"), OUTPUT_ROOT=self.out,
                         CKPT_ROOT=os.path.join(self.tmp, "ckpts"), MATERIALS=mat, MODELS=model)
                p = subprocess.run(["bash", KINDS[kind]["driver"]], env=e, cwd=self.tmp, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertEqual(table_row(p.stdout, mat).count("PASS"), 1)
                self.assertIn("RESULT: PASS", p.stdout)
                self.assertIn(f"{KINDS[kind]['log']} PASS: all 1 cells", p.stdout)
                # second run reuses the JSON: train.py is not invoked again
                n = len(self.train_argv())
                p = subprocess.run(["bash", KINDS[kind]["driver"]], env=e, cwd=self.tmp, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertEqual(len(self.train_argv()), n)
                self.assertIn("checkpoint unchanged", p.stdout)
                # the verifier's laundering scenario, end to end: checkpoint replaced, FORCE=1, train.py
                # exits 0 without logging -> the stale metrics.csv is not reused, the run fails (exit 5)
                # and the superseded JSON is kept aside instead of standing in for the new checkpoint
                ck = os.path.join(self.tmp, "ckpts", mat, os.listdir(os.path.join(self.tmp, "ckpts", mat))[0])
                with open(ck, "ab") as f:
                    f.write(b"new weights\n")
                p = subprocess.run(["bash", KINDS[kind]["driver"]], env=dict(e, FORCE="1", FAKE_TRAIN_NO_CSV="1"),
                                   cwd=self.tmp, capture_output=True, text=True)
                self.assertEqual(p.returncode, 5, p.stdout + p.stderr)
                self.assertEqual(len(self.train_argv()), n + 1)
                self.assertIn("wrote no new metrics.csv", p.stderr)
                json_path = os.path.join(self.out, KINDS[kind]["results"], f"{mat}_{model}.json")
                self.assertFalse(os.path.exists(json_path))
                self.assertTrue(os.path.exists(json_path + ".prev"))
                self.assertNotIn("PASS", p.stdout)
                # a run whose train.py does log recovers the cell (fresh version_1/metrics.csv)
                p = subprocess.run(["bash", KINDS[kind]["driver"]], env=e, cwd=self.tmp, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertIn("no previous result", p.stdout)
                with open(json_path) as f:
                    self.assertTrue(json.load(f)["metrics_csv"].endswith("/version_1/metrics.csv"))
        self.assertTrue(cer.check_reuse(os.path.join(self.out, "eval_results_ubo", "felt01_Bonn.json"), ck_ubo)[0])


if __name__ == "__main__":
    unittest.main()
