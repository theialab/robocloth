"""Fail-closed behaviour of the Bonn (UBOFAB19) table driver.

Covers scripts/comparisons/run_table_bonn.sh (driver + its inline per-cell
evaluation, evaluate_cell) and scripts/comparisons/collect_eval_results_bonn.py
(collector, data check, held-out split counts, the weighted paper metric).
Same test pattern as tests/repro_drivers.

No GPU, no Bonn data and no checkpoints are needed: the driver is pointed at
stub_eval.sh through ROBOCLOTH_EVAL_SCRIPT, or — for the inline evaluation —
at fake_train_python.sh, a stand-in for the ROBOCLOTH_PYTHON interpreter that
emulates train.py's CSV logger.  The split-count and inline-evaluation tests
write tiny EXR files with the real Bonn channel naming and are skipped when
pyexr (a training-environment dependency) is not importable.

Scratch files go to ROBOCLOTH_CI_TMP if set, otherwise the system temp dir
(TMPDIR), and are removed after each test.  Nothing is written inside the
repository: the in-process imports below set sys.dont_write_bytecode and
every subprocess gets PYTHONDONTWRITEBYTECODE=1 (clean_env()).

Run:  <python> -B -m unittest discover -s tests/bonn_driver -v
      (-B keeps this module's own __pycache__ out of the repository)
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SCRIPTS = os.path.join(REPO, "scripts")
COMPARISONS = os.path.join(SCRIPTS, "comparisons")
DRIVER = os.path.join(COMPARISONS, "run_table_bonn.sh")
COLLECTOR = os.path.join(COMPARISONS, "collect_eval_results_bonn.py")
STUB = os.path.join(HERE, "stub_eval.sh")
FAKE_PYTHON = os.path.join(HERE, "fake_train_python.sh")
CI_TMP = os.environ.get("ROBOCLOTH_CI_TMP") or None
LOG = "[run_table_bonn]"

sys.dont_write_bytecode = True              # no scripts/__pycache__ from the imports below
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, COMPARISONS)
import collect_eval_results as cer           # noqa: E402
import collect_eval_results_bonn as cerb     # noqa: E402

try:
    import numpy as np
    import pyexr
    HAVE_PYEXR = True
except ImportError:                          # the training environment has both
    HAVE_PYEXR = False

MODELS = ["Ours", "Bonn", "MERL", "PBR"]
MATS = list(cerb.PAPER)                      # 318 377 32 226 37
N_POLY, N_GRAY = 27, 106                     # held-out split of every table material (668 images)
# Environment knobs the scripts read; scrubbed from the inherited environment.
KNOBS = ["DATA_ROOT", "CKPT_ROOT", "OUTPUT_ROOT", "MATERIALS", "MODELS", "TOLERANCE_DB", "FORCE",
         "ALLOW_MISSING", "ROBOCLOTH_PYTHON", "PYTHON", "REAL_PYTHON", "ROBOCLOTH_EVAL_SCRIPT", "EXP_NAME",
         "SAVE_ALL_VIEWS", "LLS_SPP", "CKPT_SHA256", "PRIOR_METRICS"]


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if k not in KNOBS and not k.startswith("STUB_") and not k.startswith("FAKE_TRAIN_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"    # subprocess imports of scripts/*.py leave no __pycache__
    return env


def scratch_dir():
    if CI_TMP:
        os.makedirs(CI_TMP, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="bonn_driver_", dir=CI_TMP)


def write_ckpt(path, payload=b"fake checkpoint\n" * 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return path


def gray_for(target, poly=40.0):
    """val/gray_psnr that makes the 27/106-weighted mean with `poly` equal `target`."""
    return (target * (N_POLY + N_GRAY) - N_POLY * poly) / N_GRAY


def channel_inventory():
    """Channel names with the inventory of every table material: 100 poly images (3 channels
    each), 388 pan images of which 288 use LEDs il001-il024, 280 LLS images -> 668 images."""
    poly = [f"poly_cv01_il{il:03d}_rot000_{c}" for il in range(1, 101) for c in "BGR"]
    pan = [f"pan_cv{cv:02d}_il{il:03d}_rot000" for cv in range(1, 13) for il in range(1, 25)]
    pan += [f"pan_cv13_il{il:03d}_rot000" for il in range(25, 125)]
    lls = [f"lls_cv{cv:02d}_lls{j:02d}_la22.00_rot045" for cv in range(1, 5) for j in range(1, 71)]
    return {"_poly.exr": poly, "_pan.exr": pan, "_lls.exr": lls,
            "_xyz_rot000.exr": ["B", "G", "R"]}


def write_exr(path, names):
    pyexr.write(path, np.zeros((2, 2, len(names)), dtype=np.float16), channel_names=names, precision=pyexr.HALF)


def make_data(root, materials=MATS, real_exr=False, omit_files=(), omit_meta=()):
    """A Bonn_val-like folder: bonn_point_metadata.json + the per-material files the loader opens."""
    os.makedirs(root, exist_ok=True)
    meta = {str(int(m)): {"H": 2, "W": 2, "num_points": 4} for m in materials if m not in omit_meta}
    with open(os.path.join(root, cerb.METADATA_JSON), "w") as f:
        json.dump(meta, f)
    inventory = channel_inventory() if real_exr else {}
    for m in materials:
        prefix = cerb.material_prefix(root, m)
        for suffix in cerb.REQUIRED_SUFFIXES:
            if (m, suffix) in omit_files:
                continue
            if suffix in inventory:
                write_exr(prefix + suffix, inventory[suffix])
            else:
                open(prefix + suffix, "wb").close()
    return root


def table_row(stdout, name):
    """The printed table row for material `name` (or 'average')."""
    for line in stdout.splitlines():
        if line.split("|")[0].strip() == name:
            return line
    raise AssertionError(f"no table row for {name!r} in:\n{stdout}")


class DriverCase(unittest.TestCase):
    """Runs the driver against a fake CKPT_ROOT, a fake Bonn folder and the stub evaluator."""

    def setUp(self):
        self._tmp = scratch_dir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.ckpt_root = None
        self.data_root = make_data(os.path.join(self.tmp, "Bonn_val"))

    # -- fixtures -------------------------------------------------------------
    def make_ckpts(self, omit=(), extra=()):
        root = os.path.join(self.tmp, "ckpts")
        for mat in MATS:
            for model in MODELS:
                if (mat, model) not in omit:
                    write_ckpt(os.path.join(root, mat, f"{model}_epoch{100 + len(mat) * 10}.ckpt"))
        for mat, name in extra:
            write_ckpt(os.path.join(root, mat, name))
        self.ckpt_root = root
        return root

    def ckpt(self, mat, model):
        (path,) = [os.path.join(self.ckpt_root, mat, n) for n in os.listdir(os.path.join(self.ckpt_root, mat))
                   if n.startswith(f"{model}_epoch")]
        return path

    def results_dir(self):
        return os.path.join(self.tmp, "out", "eval_results_bonn")

    def result(self, mat, model):
        with open(os.path.join(self.results_dir(), f"{mat}_{model}.json")) as f:
            return json.load(f)

    def calls(self):
        path = os.path.join(self.tmp, "calls.log")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return f.read().split()

    def reset_calls(self):
        open(os.path.join(self.tmp, "calls.log"), "w").close()

    # -- running --------------------------------------------------------------
    def run_driver(self, mats=None, models=None, **over):
        env = clean_env()
        env.update(CKPT_ROOT=self.ckpt_root, DATA_ROOT=self.data_root, OUTPUT_ROOT=os.path.join(self.tmp, "out"),
                   ROBOCLOTH_EVAL_SCRIPT=STUB, ROBOCLOTH_PYTHON=sys.executable,
                   STUB_CALL_LOG=os.path.join(self.tmp, "calls.log"))
        if mats is not None:
            env["MATERIALS"] = " ".join(mats)
        if models is not None:
            env["MODELS"] = " ".join(models)
        env.update({k: str(v) for k, v in over.items()})
        return subprocess.run(["bash", DRIVER], env=env, cwd=self.tmp, capture_output=True, text=True, timeout=900)

    def assertRc(self, proc, rc):
        self.assertEqual(proc.returncode, rc, f"exit {proc.returncode}, expected {rc}\n--- stdout ---\n"
                                              f"{proc.stdout}\n--- stderr ---\n{proc.stderr}")


class TestDriverFailsClosed(DriverCase):

    def test_missing_checkpoint_aborts_before_any_evaluation(self):
        m0, m1, m2 = MATS[:3]
        self.make_ckpts(omit={(m1, "Bonn")})
        p = self.run_driver(mats=[m0, m1, m2], models=["Ours", "Bonn"])
        self.assertRc(p, 4)
        self.assertIn("MISSING checkpoint for", p.stderr)
        self.assertIn(f"{m1}/Bonn", p.stderr)
        self.assertIn("aborting before any evaluation", p.stderr)
        self.assertEqual(self.calls(), [], "evaluator must not run when a checkpoint is missing")
        self.assertFalse(os.path.isdir(self.results_dir()))

    def test_all_missing_checkpoints_are_listed_up_front(self):
        m0, m1 = MATS[:2]
        self.make_ckpts(omit={(m0, "MERL"), (m1, "Ours"), (m1, "PBR")})
        p = self.run_driver(mats=[m0, m1])
        self.assertRc(p, 4)
        for cell in (f"{m0}/MERL", f"{m1}/Ours", f"{m1}/PBR"):
            self.assertIn(cell, p.stderr)
        self.assertIn("3 cell(s) without a usable checkpoint", p.stderr)
        self.assertEqual(self.calls(), [])

    def test_ambiguous_checkpoint_is_an_error(self):
        m0 = MATS[0]
        self.make_ckpts(extra=[(m0, "Bonn_epoch99.ckpt")])
        p = self.run_driver(mats=[m0], models=["Ours", "Bonn"])
        self.assertRc(p, 4)
        self.assertIn("AMBIGUOUS checkpoint", p.stderr)
        self.assertIn(f"{m0}/Bonn", p.stderr)
        self.assertEqual(self.calls(), [])

    def test_allow_missing_evaluates_the_rest_and_still_exits_nonzero(self):
        m0, m1 = MATS[:2]
        self.make_ckpts(omit={(m1, "Bonn")})
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"], ALLOW_MISSING=1)
        self.assertRc(p, 6)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours"])
        self.assertIn("MISSING", table_row(p.stdout, m1))
        self.assertIn("PASS", table_row(p.stdout, m0))
        self.assertIn(f"MISSING cells (1): {m1}/Bonn", p.stdout)
        self.assertIn("RESULT: FAIL", p.stdout)

    def test_all_within_tolerance_passes(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours", f"{m1}/Bonn"])
        for mat in (m0, m1):
            row = table_row(p.stdout, mat)
            self.assertEqual(row.count("PASS"), 2, row)
            self.assertEqual(row.count("n/a"), 2, row)      # MERL / PBR not requested
            self.assertNotIn("FAIL", row)
        self.assertIn("RESULT: PASS", p.stdout)
        self.assertIn(f"{LOG} PASS: all 4 cells", p.stdout)
        self.assertIn("held-out Bonn (UBOFAB19) test set", p.stdout)
        # result JSONs carry provenance, the Bonn components, and no temp files are left behind
        r = self.result(m0, "Ours")
        for key in ("ckpt", "ckpt_abs", "ckpt_size", "ckpt_mtime_ns", "git_head", "timestamp", "hostname",
                    "material", "model", "val_psnr", "val_poly_psnr", "val_gray_psnr", "n_val_poly", "n_val_gray"):
            self.assertIn(key, r)
        self.assertEqual((r["n_val_poly"], r["n_val_gray"], r["experiment"]), (N_POLY, N_GRAY, "stage2_bonn"))
        self.assertAlmostEqual(r["val_psnr"], cerb.bonn_val_psnr(r["val_poly_psnr"], r["val_gray_psnr"], N_POLY, N_GRAY))
        self.assertEqual(r["ckpt_abs"], os.path.realpath(self.ckpt(m0, "Ours")))
        self.assertFalse([n for n in os.listdir(self.results_dir()) if n.endswith(".tmp")])

    def test_full_default_table_passes(self):
        self.make_ckpts()
        p = self.run_driver()                       # default MATERIALS x MODELS
        self.assertRc(p, 0)
        self.assertEqual(len(self.calls()), len(MATS) * len(MODELS))
        for mat in MATS + ["average"]:
            row = table_row(p.stdout, mat)
            self.assertEqual(row.count("PASS"), 4, row)
            for bad in ("FAIL", "MISSING", "n/a"):
                self.assertNotIn(bad, row)
        self.assertIn(f"cells: {len(MATS) * 4} expected, {len(MATS) * 4} present, 0 missing; "
                      f"{len(MATS) * 4} PASS, 0 FAIL", p.stdout)
        self.assertIn("RESULT: PASS", p.stdout)
        self.assertEqual(self.result("377", "PBR")["experiment"], "stage2_bonn_pbr")

    def test_evaluator_failure_stops_the_run_with_the_command(self):
        m0, m1, m2 = MATS[:3]
        self.make_ckpts()
        p = self.run_driver(mats=[m0, m1, m2], models=["Ours"], STUB_FAIL_CELLS=f"{m1}/Ours")
        self.assertRc(p, 5)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m1}/Ours"], "must stop at the failure")
        self.assertIn("failed (exit 17)", p.stderr)
        self.assertIn(f"bash {STUB} {m1} {self.ckpt(m1, 'Ours')} Ours", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.results_dir(), f"{m1}_Ours.json")))

    def test_evaluator_that_writes_no_result_is_an_error(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours"], STUB_NO_JSON_CELLS=f"{m0}/Ours")
        self.assertRc(p, 5)
        self.assertIn("exited 0 but left no valid result", p.stderr)
        self.assertIn("no previous result", p.stderr)

    def test_evaluator_recording_another_checkpoint_is_an_error(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours"], STUB_WRONG_CKPT_CELLS=f"{m0}/Ours")
        self.assertRc(p, 5)
        self.assertIn("checkpoint path changed", p.stderr)

    def test_evaluator_recording_another_configuration_is_an_error(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours"], STUB_WRONG_CONFIG_CELLS=f"{m0}/Ours")
        self.assertRc(p, 5)
        self.assertIn("experiment differs from this run: recorded 'stage2_debug'", p.stderr)

    def test_force_with_evaluator_writing_nothing_is_an_error(self):
        """FORCE=1 must not let the previous (fresh, matching) JSON pass the post-evaluation check."""
        m0 = MATS[0]
        self.make_ckpts()
        self.assertRc(self.run_driver(mats=[m0], models=["Ours"]), 0)
        json_path = os.path.join(self.results_dir(), f"{m0}_Ours.json")
        with open(json_path) as f:
            old = json.load(f)
        self.reset_calls()
        p = self.run_driver(mats=[m0], models=["Ours"], FORCE=1, STUB_NO_JSON_CELLS=f"{m0}/Ours")
        self.assertRc(p, 5)
        self.assertEqual(self.calls(), [f"{m0}/Ours"])
        self.assertIn("exited 0 but left no valid result", p.stderr)
        self.assertIn(f"previous result kept as {json_path}.prev", p.stdout)
        self.assertFalse(os.path.exists(json_path), "the superseded JSON must not stay in place")
        with open(json_path + ".prev") as f:
            self.assertEqual(json.load(f), old)
        # nothing to reuse afterwards either: the next run evaluates the cell again
        self.reset_calls()
        p = self.run_driver(mats=[m0], models=["Ours"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours"])
        self.assertIn("no previous result", p.stdout)

    def test_stale_result_is_set_aside_before_reevaluation(self):
        """A stale JSON never survives in place, even when its re-evaluation fails."""
        m0 = MATS[0]
        self.make_ckpts()
        self.assertRc(self.run_driver(mats=[m0], models=["Ours"]), 0)
        ck = self.ckpt(m0, "Ours")
        st = os.stat(ck)
        os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        p = self.run_driver(mats=[m0], models=["Ours"], STUB_FAIL_CELLS=f"{m0}/Ours")
        self.assertRc(p, 5)
        json_path = os.path.join(self.results_dir(), f"{m0}_Ours.json")
        self.assertFalse(os.path.exists(json_path))
        self.assertTrue(os.path.exists(json_path + ".prev"))
        self.assertEqual(cer.load_results(self.results_dir()), {}, ".prev files are invisible to the collector")

    def test_result_for_another_configuration_is_reevaluated(self):
        """Same checkpoint, but the JSON came from another experiment / dataset / overrides / cell."""
        m0, m1 = MATS[:2]
        self.make_ckpts()
        self.assertRc(self.run_driver(mats=[m0, m1], models=["Ours", "PBR"]), 0)
        rd = self.results_dir()

        def rewrite(mat, model, **changes):
            path = os.path.join(rd, f"{mat}_{model}.json")
            with open(path) as f:
                r = json.load(f)
            r.update(changes)
            with open(path, "w") as f:
                json.dump(r, f)

        rewrite(m0, "Ours", overrides=["data.debug_num=10"])                    # smoke-style subset run
        rewrite(m0, "PBR", experiment="stage2_bonn")                            # PBR cell, neural config
        rewrite(m1, "Ours", dataset_folder=os.path.join(self.tmp, "other_Bonn_val"))  # another DATA_ROOT
        rewrite(m1, "PBR", material=m0)                                         # file name vs content
        self.reset_calls()
        p = self.run_driver(mats=[m0, m1], models=["Ours", "PBR"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m0}/PBR", f"{m1}/Ours", f"{m1}/PBR"])
        for needle in ("overrides differs from this run: recorded 'data.debug_num=10', now ''",
                       "experiment differs from this run: recorded 'stage2_bonn', now 'stage2_bonn_pbr'",
                       "dataset_folder differs from this run",
                       f"material differs from this run: recorded '{m0}', now '{m1}'"):
            self.assertIn(needle, p.stdout)
        for mat, model in ((m0, "Ours"), (m0, "PBR"), (m1, "Ours"), (m1, "PBR")):
            self.assertTrue(os.path.exists(os.path.join(rd, f"{mat}_{model}.json.prev")))
            self.assertEqual(self.result(mat, model)["overrides"], [])
        self.assertIn("RESULT: PASS", p.stdout)

    def test_lls_spp_is_recorded_as_an_override_and_separates_results(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours"], LLS_SPP=4, STUB_OVERRIDES="model.lls_spp=4")
        self.assertRc(p, 0)
        self.assertEqual(self.result(m0, "Ours")["overrides"], ["model.lls_spp=4"])
        # the same JSON is reused under the same LLS_SPP ...
        self.reset_calls()
        p = self.run_driver(mats=[m0], models=["Ours"], LLS_SPP=4)
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [])
        # ... but not for a run with the config default
        p = self.run_driver(mats=[m0], models=["Ours"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours"])
        self.assertIn("overrides differs from this run: recorded 'model.lls_spp=4', now ''", p.stdout)
        # a stub that records no override cannot stand in for an LLS_SPP run
        p = self.run_driver(mats=[m0], models=["Ours"], LLS_SPP=4, FORCE=1)
        self.assertRc(p, 5)
        self.assertIn("overrides differs from this run: recorded '', now 'model.lls_spp=4'", p.stderr)

    def test_bad_environment_is_rejected_before_anything_runs(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours"], ROBOCLOTH_PYTHON="/nonexistent/python")
        self.assertRc(p, 2)
        self.assertIn("interpreter not found or not executable: /nonexistent/python", p.stderr)
        p = self.run_driver(mats=[m0], models=["Ours"], DATA_ROOT=os.path.join(self.tmp, "no_such_data"))
        self.assertRc(p, 2)
        self.assertIn("DATA_ROOT is not a directory", p.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(os.path.isdir(self.results_dir()))
        # the generic PYTHON variable (autoconf, node-gyp, ...) does not select the interpreter
        p = self.run_driver(mats=[m0], models=["Ours"], PYTHON="/nonexistent/python")
        self.assertRc(p, 0)

    def test_incomplete_bonn_folder_is_rejected_before_anything_runs(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        # no bonn_point_metadata.json
        root = make_data(os.path.join(self.tmp, "no_meta"), [m0, m1])
        os.remove(os.path.join(root, cerb.METADATA_JSON))
        p = self.run_driver(mats=[m0, m1], models=["Ours"], DATA_ROOT=root)
        self.assertRc(p, 2)
        self.assertIn("bonn_point_metadata.json is missing", p.stderr)
        self.assertIn("generate_bonn_metadata.py", p.stderr)
        self.assertIn("not a usable Bonn folder", p.stderr)
        # a material's EXR missing, another material absent from the metadata: every problem is listed
        root = make_data(os.path.join(self.tmp, "partial"), [m0, m1], omit_files={(m1, "_lls.exr")}, omit_meta={m0})
        p = self.run_driver(mats=[m0, m1], models=["Ours"], DATA_ROOT=root)
        self.assertRc(p, 2)
        self.assertIn(f"material {m1}: missing {cerb.material_prefix(root, m1)}_lls.exr", p.stderr)
        self.assertIn(f"material {m0}: no entry in", p.stderr)
        self.assertIn("2 problem(s)", p.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(os.path.isdir(self.results_dir()))
        # materials outside the request are not required
        root = make_data(os.path.join(self.tmp, "subset"), [m0])
        p = self.run_driver(mats=[m0], models=["Ours"], DATA_ROOT=root)
        self.assertRc(p, 0)


class TestStaleResults(DriverCase):
    """Reuse of an existing result JSON is tied to the checkpoint's path/size/mtime."""

    def first_run(self, mats, models=("Ours", "Bonn")):
        p = self.run_driver(mats=mats, models=list(models))
        self.assertRc(p, 0)
        self.reset_calls()

    def test_fresh_matching_result_is_reused(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        self.first_run([m0, m1])
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [], "stub must not be called for fresh results")
        self.assertEqual(p.stdout.count("checkpoint unchanged"), 4)
        self.assertIn("RESULT: PASS", p.stdout)

    def test_changed_mtime_triggers_reevaluation_of_that_cell_only(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        self.first_run([m0, m1])
        ck = self.ckpt(m1, "Ours")
        st = os.stat(ck)
        os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 * 10**9))
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m1}/Ours"])
        self.assertIn("checkpoint mtime changed", p.stdout)
        self.assertEqual(self.result(m1, "Ours")["ckpt_mtime_ns"], os.stat(ck).st_mtime_ns)
        prev = [n for n in os.listdir(self.results_dir()) if n.endswith(".prev")]
        self.assertEqual(prev, [f"{m1}_Ours.json.prev"], "only the stale cell's JSON is set aside")

    def test_changed_size_triggers_reevaluation(self):
        m0 = MATS[0]
        self.make_ckpts()
        self.first_run([m0])
        ck = self.ckpt(m0, "Bonn")
        st = os.stat(ck)
        with open(ck, "ab") as f:
            f.write(b"more weights\n")
        os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns))    # keep mtime: size alone must trigger
        p = self.run_driver(mats=[m0], models=["Ours", "Bonn"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Bonn"])
        self.assertIn("checkpoint size changed", p.stdout)

    def test_moved_checkpoint_root_triggers_reevaluation(self):
        m0 = MATS[0]
        root = self.make_ckpts()
        self.first_run([m0])
        moved = root + "_moved"
        os.rename(root, moved)
        self.ckpt_root = moved
        p = self.run_driver(mats=[m0], models=["Ours", "Bonn"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m0}/Bonn"])
        self.assertIn("checkpoint path changed", p.stdout)

    def test_legacy_result_without_provenance_is_reevaluated(self):
        m0 = MATS[0]
        self.make_ckpts()
        os.makedirs(self.results_dir())
        legacy = {"material": m0, "model": "Ours", "ckpt": self.ckpt(m0, "Ours"),
                  "val_psnr": cerb.PAPER[m0]["Ours"], "val_loss": 0.01}      # pre-provenance schema
        with open(os.path.join(self.results_dir(), f"{m0}_Ours.json"), "w") as f:
            json.dump(legacy, f)
        p = self.run_driver(mats=[m0], models=["Ours"])
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours"])
        self.assertIn("records no checkpoint provenance", p.stdout)
        self.assertIn("ckpt_mtime_ns", self.result(m0, "Ours"))

    def test_force_reevaluates_everything(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        self.first_run([m0, m1])
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"], FORCE=1)
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [f"{m0}/Ours", f"{m0}/Bonn", f"{m1}/Ours", f"{m1}/Bonn"])
        self.assertEqual(p.stdout.count("FORCE=1: re-evaluating"), 4)


class TestTolerance(DriverCase):

    def test_breach_fails_with_fail_cell_and_tolerance_env_is_honored(self):
        m0, m1 = MATS[:2]
        self.make_ckpts()
        offsets = json.dumps({f"{m1}/Bonn": 0.2})
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"], STUB_PSNR_OFFSETS=offsets)
        self.assertRc(p, 7)
        row = table_row(p.stdout, m1)
        self.assertIn("FAIL", row)
        self.assertIn("+0.20", row)
        self.assertNotIn("FAIL", table_row(p.stdout, m0))
        self.assertRegex(p.stdout, rf"FAIL {m1}/Bonn: repro .* \(\+0\.2000 dB, \|diff\| > 0\.05 dB\)")
        self.assertIn("RESULT: FAIL", p.stdout)
        self.assertIn(f"{LOG} FAILED (exit 7)", p.stderr)
        # same (reused) results pass under a looser explicit tolerance
        self.reset_calls()
        p = self.run_driver(mats=[m0, m1], models=["Ours", "Bonn"], TOLERANCE_DB=0.5)
        self.assertRc(p, 0)
        self.assertEqual(self.calls(), [])
        self.assertIn("tolerance: |repro - paper| <= 0.50 dB", p.stdout)
        self.assertEqual(table_row(p.stdout, m1).count("PASS"), 2)

    def test_edge_of_tolerance(self):
        m0 = MATS[0]
        self.make_ckpts()
        p = self.run_driver(mats=[m0], models=["Ours", "Bonn"],
                            STUB_PSNR_OFFSETS=json.dumps({f"{m0}/Ours": -0.04, f"{m0}/Bonn": 0.05}))
        self.assertRc(p, 0)
        p = self.run_driver(mats=[m0], models=["MERL"], STUB_PSNR_OFFSETS=json.dumps({f"{m0}/MERL": -0.051}))
        self.assertRc(p, 7)


class TestCollectorAndMetric(unittest.TestCase):
    """collect_eval_results_bonn.py used directly: table, data check, split counts, weighted metric."""

    def setUp(self):
        self._tmp = scratch_dir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.ckpt = write_ckpt(os.path.join(self.tmp, "ckpts", "318", "Ours_epoch120.ckpt"))
        self.results = os.path.join(self.tmp, "eval_results_bonn")
        os.makedirs(self.results)

    def write(self, mat, model, psnr, ckpt=None):
        return cer.write_result(os.path.join(self.results, f"{mat}_{model}.json"), material=mat, model=model,
                                ckpt=ckpt or self.ckpt, val_psnr=psnr, val_loss=None, experiment="stage2_bonn")

    def collect(self, *args, **env):
        e = clean_env()
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run([sys.executable, COLLECTOR, *args], env=e, capture_output=True, text=True)

    def test_paper_values_are_the_bonn_block_of_table_3(self):
        self.assertEqual(MATS, ["318", "377", "32", "226", "37"])
        self.assertEqual(cerb.PAPER["318"], {"Ours": 44.38, "Bonn": 45.83, "MERL": 40.26, "PBR": 43.16})
        self.assertEqual(cerb.PAPER["37"], {"Ours": 17.03, "Bonn": 18.22, "MERL": 16.36, "PBR": 16.75})
        self.assertEqual(cerb.PAPER_AVG, {"Ours": 23.43, "Bonn": 24.47, "MERL": 21.77, "PBR": 22.99})
        for model in MODELS:   # the printed averages are the means of the printed cells, to 2 decimals
            mean = sum(cerb.PAPER[m][model] for m in MATS) / len(MATS)
            self.assertLessEqual(abs(mean - cerb.PAPER_AVG[model]), 0.005 + 1e-9, model)
        self.assertEqual(cerb.BONN.default_dir, "eval_results_bonn")
        self.assertEqual(cerb.BONN.materials, MATS)

    def test_missing_cell_is_listed_and_exits_6(self):
        for mat, model in (("318", "Ours"), ("318", "Bonn"), ("377", "Ours")):
            self.write(mat, model, cerb.PAPER[mat][model])
        p = self.collect(self.results, "--materials", "318 377", "--models", "Ours Bonn")
        self.assertEqual(p.returncode, 6, p.stdout + p.stderr)
        self.assertIn("MISSING cells (1): 377/Bonn", p.stdout)
        self.assertIn("MISSING", table_row(p.stdout, "377"))
        self.assertIn("cells: 4 expected, 3 present, 1 missing; 3 PASS, 0 FAIL", p.stdout)
        self.assertIn("held-out Bonn (UBOFAB19) test set", p.stdout)

    def test_tolerance_and_bad_args(self):
        self.write("318", "Ours", cerb.PAPER["318"]["Ours"] + 1.0)
        p = self.collect(self.results, "--materials", "318", "--models", "Ours Bonn")
        self.assertEqual(p.returncode, 7, p.stdout)
        self.assertIn("FAIL 318/Ours", p.stdout)
        self.assertEqual(self.collect(self.results, "--materials", "145").returncode, 2)     # a RoboCloth material
        self.assertEqual(self.collect(self.results, "--models", "Nope").returncode, 2)
        self.assertEqual(self.collect(self.results, "--tolerance-db", "-1").returncode, 2)

    def test_full_table_in_process_and_reuse_check_cli(self):
        for mat in MATS:
            for model in MODELS:
                self.write(mat, model, cerb.PAPER[mat][model] + 0.05)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = cer.collect(self.results, cerb.BONN)
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertEqual(table_row(buf.getvalue(), "average").count("PASS"), 4)
        out = os.path.join(self.results, "318_Ours.json")
        p = self.collect("--reuse-check", out, self.ckpt, "--expect", "material=318", "experiment=stage2_bonn")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = self.collect("--reuse-check", out, self.ckpt, "--expect", "experiment=stage2_bonn_pbr")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("experiment differs from this run: recorded 'stage2_bonn', now 'stage2_bonn_pbr'", p.stdout)

    def test_check_data(self):
        root = make_data(os.path.join(self.tmp, "Bonn_val"), ["318", "377"])
        self.assertEqual(cerb.check_data(root, ["318", "377"]), [])
        self.assertEqual(cerb.check_data(root, ["318", "0318"]), [], "ids are compared numerically")
        p = self.collect("--check-data", root, "--materials", "318 377")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("all per-material files present", p.stdout)
        # missing file / missing metadata entry / not a directory / bad usage
        os.remove(cerb.material_prefix(root, "377") + "_pan.exr")
        problems = cerb.check_data(root, ["318", "377", "32"])
        self.assertEqual(len(problems), 1 + len(cerb.REQUIRED_SUFFIXES) + 1, problems)
        self.assertTrue(any("_pan.exr" in q and "377" in q for q in problems))
        self.assertTrue(any(q.startswith("material 32: no entry in") for q in problems))
        p = self.collect("--check-data", root, "--materials", "377")
        self.assertEqual(p.returncode, 2)
        self.assertIn("1 problem(s)", p.stderr)
        self.assertEqual(cerb.check_data(os.path.join(self.tmp, "nope"), ["318"]), [f"{self.tmp}/nope is not a directory"])
        os.remove(os.path.join(root, cerb.METADATA_JSON))
        self.assertTrue(any("bonn_point_metadata.json is missing" in q for q in cerb.check_data(root, ["318"])))
        self.assertEqual(self.collect("--check-data", root).returncode, 2)

    def test_bonn_val_psnr_is_the_image_count_weighted_mean(self):
        self.assertAlmostEqual(cerb.bonn_val_psnr(40.0, gray_for(44.38), N_POLY, N_GRAY), 44.38)
        self.assertAlmostEqual(cerb.bonn_val_psnr(10.0, 20.0, 1, 3), 17.5)
        self.assertEqual(cerb.bonn_val_psnr(12.5, None, 4, 0), 12.5)     # no gray views: poly only
        self.assertEqual(cerb.bonn_val_psnr(None, 12.5, 0, 4), 12.5)
        with self.assertRaisesRegex(ValueError, "no val/gray_psnr"):
            cerb.bonn_val_psnr(40.0, None, N_POLY, N_GRAY)
        with self.assertRaisesRegex(ValueError, "no val/poly_psnr"):
            cerb.bonn_val_psnr(None, 40.0, N_POLY, N_GRAY)
        with self.assertRaisesRegex(ValueError, "invalid split counts"):
            cerb.bonn_val_psnr(40.0, 40.0, 0, 0)

    @unittest.skipUnless(HAVE_PYEXR, "pyexr not importable (training environment dependency)")
    def test_val_split_counts_from_exr_headers(self):
        root = make_data(os.path.join(self.tmp, "Bonn_val"), ["318"], real_exr=True)
        self.assertEqual(cerb.val_split_counts(root, "318"), (N_POLY, N_GRAY))
        self.assertEqual(cerb.val_split_counts(root, 318), (N_POLY, N_GRAY))
        # without pan images the pool is 100 poly + 280 LLS = 380 -> 76 held out; recompute independently
        perm = np.random.RandomState(42).permutation(380)[:76]
        n_poly = int((perm < 100).sum())
        self.assertEqual(cerb.val_split_counts(root, "318", use_pan=False), (n_poly, 76 - n_poly))
        self.assertEqual(sum(cerb.val_split_counts(root, "318", use_lls=False)), int(388 * 0.2))
        p = self.collect("--val-split-counts", root, "318")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(json.loads(p.stdout), {"material": "318", "n_val_poly": N_POLY, "n_val_gray": N_GRAY,
                                                "n_val_views": N_POLY + N_GRAY})
        self.assertEqual(self.collect("--val-split-counts", root).returncode, 2)

    def test_subprocesses_write_no_bytecode_into_the_repo(self):
        dirs = [os.path.join(SCRIPTS, "__pycache__"), os.path.join(COMPARISONS, "__pycache__"),
                os.path.join(HERE, "__pycache__")]

        def listing():
            return {d: sorted((n, os.stat(os.path.join(d, n)).st_mtime_ns) for n in os.listdir(d))
                    if os.path.isdir(d) else None for d in dirs}

        before = listing()
        self.assertEqual(clean_env().get("PYTHONDONTWRITEBYTECODE"), "1")
        self.write("318", "Ours", cerb.PAPER["318"]["Ours"])
        self.assertEqual(self.collect(self.results, "--materials", "318", "--models", "Ours").returncode, 0)
        e = clean_env()
        e.update(STUB_CALL_LOG=os.path.join(self.tmp, "calls.log"), OUTPUT_ROOT=self.tmp, DATA_ROOT=self.tmp,
                 ROBOCLOTH_PYTHON=sys.executable)
        self.assertEqual(subprocess.run(["bash", STUB, "318", self.ckpt, "Ours"], env=e, capture_output=True).returncode, 0)
        self.assertEqual(listing(), before)


@unittest.skipUnless(HAVE_PYEXR, "pyexr not importable (training environment dependency)")
class TestInlineEvaluation(DriverCase):
    """run_table_bonn.sh's own evaluate_cell with a fake `python` standing in for train.py."""

    def setUp(self):
        super().setUp()
        self.data_root = make_data(os.path.join(self.tmp, "Bonn_val_exr"), real_exr=True)
        self.fake = shutil.copy(FAKE_PYTHON, os.path.join(self.tmp, "python"))
        os.chmod(self.fake, 0o755)
        self.log = os.path.join(self.tmp, "train_calls.log")
        self.make_ckpts()

    def run_inline(self, mats, models, **over):
        env = clean_env()
        env.update(CKPT_ROOT=self.ckpt_root, DATA_ROOT=self.data_root, OUTPUT_ROOT=os.path.join(self.tmp, "out"),
                   ROBOCLOTH_PYTHON=self.fake, REAL_PYTHON=sys.executable, FAKE_TRAIN_LOG=self.log,
                   MATERIALS=" ".join(mats), MODELS=" ".join(models))
        env.update({k: str(v) for k, v in over.items()})
        return subprocess.run(["bash", DRIVER], env=env, cwd=self.tmp, capture_output=True, text=True, timeout=900)

    def train_argv(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return f.read().splitlines()

    def test_single_cell_end_to_end(self):
        gray = gray_for(44.38)
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_POLY_PSNR="40.0", FAKE_TRAIN_GRAY_PSNR=repr(gray))
        self.assertRc(p, 0)
        argv = self.train_argv()
        self.assertEqual(len(argv), 1)
        for token in ("train.py +experiment=stage2_bonn ", f"dataset_folder={os.path.realpath(self.data_root)}",
                      "data.overfit_mat_id=318", "data.valid_num=20", "model.test=true",
                      "model.trainer.enable_checkpointing=false", f"model.ckpt_path={self.ckpt('318', 'Ours')}",
                      "experiment_name=Eval_Bonn_318_Ours", "model.logger._target_=pytorch_lightning.loggers.CSVLogger",
                      "~model.logger.project"):
            self.assertIn(token, argv[0])
        self.assertNotIn("lls_spp", argv[0])
        r = self.result("318", "Ours")
        self.assertAlmostEqual(r["val_psnr"], 44.38)
        self.assertAlmostEqual(r["val_poly_psnr"], 40.0)
        self.assertAlmostEqual(r["val_gray_psnr"], gray)
        self.assertEqual((r["n_val_poly"], r["n_val_gray"], r["n_val_views"]), (N_POLY, N_GRAY, N_POLY + N_GRAY))
        self.assertEqual((r["material"], r["model"], r["experiment"], r["valid_num"], r["overrides"], r["val_loss"]),
                         ("318", "Ours", "stage2_bonn", 20, [], None))
        self.assertEqual(r["dataset_folder"], os.path.realpath(self.data_root))
        self.assertAlmostEqual(r["val_all_psnr"], 44.0)
        self.assertTrue(r["metrics_csv"].endswith("/Eval_Bonn_318_Ours/Eval_Bonn_318_Ours/version_0/metrics.csv"), r["metrics_csv"])
        st = os.stat(self.ckpt("318", "Ours"))
        self.assertEqual((r["ckpt_abs"], r["ckpt_size"], r["ckpt_mtime_ns"]),
                         (os.path.realpath(self.ckpt("318", "Ours")), st.st_size, st.st_mtime_ns))
        self.assertIn("val psnr = 44.38 dB (poly 27 views: 40.00, gray 106 views:", p.stdout)
        self.assertEqual(table_row(p.stdout, "318").count("PASS"), 1)
        self.assertIn(f"{LOG} PASS: all 1 cells", p.stdout)
        # a second run reuses the JSON: train.py is not invoked again
        p = self.run_inline(["318"], ["Ours"])
        self.assertRc(p, 0)
        self.assertEqual(len(self.train_argv()), 1)
        self.assertIn("checkpoint unchanged", p.stdout)

    def test_pbr_save_all_views_and_lls_spp(self):
        p = self.run_inline(["318"], ["PBR"], SAVE_ALL_VIEWS=1, LLS_SPP=4,
                            FAKE_TRAIN_GRAY_PSNR=repr(gray_for(43.16)))
        self.assertRc(p, 0)
        argv = self.train_argv()[0]
        for token in ("+experiment=stage2_bonn_pbr", "data.valid_num=-1", "model.lls_spp=4"):
            self.assertIn(token, argv)
        r = self.result("318", "PBR")
        self.assertEqual((r["experiment"], r["valid_num"], r["overrides"]), ("stage2_bonn_pbr", -1, ["model.lls_spp=4"]))
        # the config-default run does not reuse the LLS_SPP=4 result
        p = self.run_inline(["318"], ["PBR"], FAKE_TRAIN_GRAY_PSNR=repr(gray_for(43.16)))
        self.assertRc(p, 0)
        self.assertEqual(len(self.train_argv()), 2)
        self.assertIn("overrides differs from this run: recorded 'model.lls_spp=4', now ''", p.stdout)
        self.assertEqual(self.result("318", "PBR")["overrides"], [])

    def test_train_failure_is_exit_5_with_the_rerun_command(self):
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_RC=3)
        self.assertRc(p, 5)
        self.assertIn("failed (exit 3)", p.stderr)
        self.assertIn(f"MATERIALS=318 MODELS=Ours FORCE=1 DATA_ROOT={os.path.realpath(self.data_root)}", p.stderr)
        self.assertIn("bash " + DRIVER, p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.results_dir(), "318_Ours.json")))
        # LLS_SPP is part of the printed command
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_RC=3, LLS_SPP=4)
        self.assertRc(p, 5)
        self.assertIn(" LLS_SPP=4 bash ", p.stderr)

    def test_no_metrics_csv_is_exit_5(self):
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_NO_CSV=1)
        self.assertRc(p, 5)
        self.assertIn("wrote no new metrics.csv", p.stderr)
        self.assertIn("0 metrics.csv from earlier runs", p.stderr)
        self.assertIn("failed (exit 3)", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.results_dir(), "318_Ours.json")))

    def test_missing_gray_metric_is_exit_5(self):
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_GRAY_PSNR="")
        self.assertRc(p, 5)
        self.assertIn("106 gray views in the split but no val/gray_psnr was logged", p.stderr)
        self.assertIn("failed (exit 1)", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.results_dir(), "318_Ours.json")))

    def test_stale_metrics_csv_from_an_earlier_run_is_never_reused(self):
        """The experiment name is reused across runs: a metrics.csv left by an earlier checkpoint must
        not be laundered into a JSON carrying the new checkpoint's provenance."""
        gray = repr(gray_for(44.38))
        json_path = os.path.join(self.results_dir(), "318_Ours.json")
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_GRAY_PSNR=gray)
        self.assertRc(p, 0)
        self.assertTrue(self.result("318", "Ours")["metrics_csv"].endswith("/version_0/metrics.csv"))
        # the checkpoint is replaced; FORCE=1; train.py "succeeds" but logs nothing -> exit 5, JSON set aside
        ck = self.ckpt("318", "Ours")
        with open(ck, "ab") as f:
            f.write(b"new weights\n")
        p = self.run_inline(["318"], ["Ours"], FORCE=1, FAKE_TRAIN_NO_CSV=1)
        self.assertRc(p, 5)
        self.assertEqual(len(self.train_argv()), 2)
        self.assertIn("wrote no new metrics.csv", p.stderr)
        self.assertIn("1 metrics.csv from earlier runs", p.stderr)
        self.assertFalse(os.path.exists(json_path))
        self.assertTrue(os.path.exists(json_path + ".prev"))
        self.assertNotIn("PASS", p.stdout)
        # a run whose train.py does log recovers the cell (fresh version_1/metrics.csv)
        p = self.run_inline(["318"], ["Ours"], FAKE_TRAIN_GRAY_PSNR=gray)
        self.assertRc(p, 0)
        self.assertIn("no previous result", p.stdout)
        r = self.result("318", "Ours")
        self.assertTrue(r["metrics_csv"].endswith("/version_1/metrics.csv"), r["metrics_csv"])
        self.assertEqual(r["ckpt_size"], os.stat(ck).st_size)


if __name__ == "__main__":
    unittest.main()
