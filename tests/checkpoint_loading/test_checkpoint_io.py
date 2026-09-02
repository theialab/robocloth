"""Unit tests for the fail-closed checkpoint policy in training/utils/checkpoint_io.py.

Run (CPU only, torch is the only dependency):
    <python with torch> -m unittest discover -s tests/checkpoint_loading -v

Synthetic checkpoints are written to a temporary directory (under
$ROBOCLOTH_TEST_TMP if that is set, else the system temp dir) and deleted
when each test finishes — nothing is left in the repository.
"""
import importlib.util
import os
import tempfile
import unittest

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TMP_PARENT = os.environ.get("ROBOCLOTH_TEST_TMP") or None   # None -> tempfile's default location
if TMP_PARENT:
    os.makedirs(TMP_PARENT, exist_ok=True)


def _load_checkpoint_io():
    """Import training/utils/checkpoint_io.py by path (no sys.path games)."""
    path = os.path.join(REPO, "training", "utils", "checkpoint_io.py")
    spec = importlib.util.spec_from_file_location("checkpoint_io_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cio = _load_checkpoint_io()


class Material(nn.Module):
    """Mimics material.{decoder,latent_texture,factor} of the real models."""

    def __init__(self, hidden=8, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.decoder = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.latent_texture = nn.Module()
        self.latent_texture.params = nn.Parameter(torch.randn(1, 5, 4, 4))
        self.factor = nn.Parameter(torch.rand(3))


class Trainer(nn.Module):
    """Mimics the LightningModule layout: material.* + gt_material.* + emitter.*."""

    def __init__(self, hidden=8, seed=0, n_lights=7):
        super().__init__()
        self.material = Material(hidden, seed)
        self.gt_material = nn.Module()
        self.gt_material.albedo = nn.Parameter(torch.rand(3))
        self.emitter = nn.Module()
        self.emitter.register_buffer("light_positions", torch.rand(n_lights, 3))


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_equal_state(test, a, b, keys=None):
    for k in (keys or a.keys()):
        test.assertTrue(torch.equal(a[k], b[k]), f"tensor differs: {k}")


class CheckpointIOTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_PARENT, prefix="ckpt_io_test_")
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, obj, name="ckpt.ckpt"):
        path = os.path.join(self.tmp, name)
        torch.save(obj, path)
        return path

    # ---- file level --------------------------------------------------------
    def test_absent_path_raises_file_not_found_with_path(self):
        missing = os.path.join(self.tmp, "does", "not", "exist.ckpt")
        with self.assertRaises(FileNotFoundError) as cm:
            cio.load_checkpoint_file(missing)
        self.assertIn(missing, str(cm.exception))
        with self.assertRaises(FileNotFoundError):
            cio.load_checkpoint_file(None)
        with self.assertRaises(FileNotFoundError):
            cio.load_checkpoint_file("")

    def test_unreadable_file_raises_runtime_error(self):
        path = os.path.join(self.tmp, "garbage.ckpt")
        with open(path, "wb") as f:
            f.write(b"this is not a torch file")
        with self.assertRaises(RuntimeError) as cm:
            cio.load_checkpoint_file(path)
        self.assertIn(path, str(cm.exception))

    # ---- state-dict extraction --------------------------------------------
    def test_lightning_wrapper_and_other_formats(self):
        model = Trainer(seed=1)
        sd = model.state_dict()
        lightning = {"state_dict": sd, "hyper_parameters": {"material": {"texture_resolution": 4}},
                     "epoch": 3, "optimizer_states": [{}]}
        for obj in (lightning, {"model_state_dict": sd}, sd):
            loaded = cio.load_checkpoint_file(self._save(obj))
            got = cio.extract_state_dict(loaded, source="x")
            self.assertEqual(list(got.keys()), list(sd.keys()))
            assert_equal_state(self, got, sd)
        # Lightning wrapper end-to-end into a fresh model: identical parameters.
        target = Trainer(seed=2)
        loaded = cio.load_checkpoint_file(self._save(lightning))
        cio.load_state_dict_strict(target, cio.extract_state_dict(loaded), source="x")
        assert_equal_state(self, snapshot(target), sd)

    def test_non_dict_checkpoint_is_an_error_not_random_init(self):
        with self.assertRaises(RuntimeError):
            cio.extract_state_dict(torch.zeros(3), source="x")
        with self.assertRaises(RuntimeError):
            cio.extract_state_dict({"state_dict": [1, 2, 3]}, source="x")
        with self.assertRaises(RuntimeError) as cm:
            cio.extract_state_dict({"a": torch.zeros(1), "b": "not a tensor"}, source="x")
        self.assertIn("b", str(cm.exception))

    def test_split_namespace(self):
        sd = {"material.a": 1, "material.b.c": 2, "emitter.x": 3}
        inside, outside = cio.split_namespace(sd, "material.")
        self.assertEqual(dict(inside), {"a": 1, "b.c": 2})
        self.assertEqual(dict(outside), {"emitter.x": 3})

    # ---- strict load -------------------------------------------------------
    def test_full_match_loads_identical_parameters(self):
        src, dst = Trainer(seed=3), Trainer(seed=4)
        before = snapshot(dst)
        self.assertFalse(torch.equal(before["material.factor"], src.state_dict()["material.factor"]))
        report = cio.load_state_dict_strict(dst, src.state_dict(), source="x")
        assert_equal_state(self, snapshot(dst), snapshot(src))
        self.assertEqual(report.skipped, [])
        self.assertEqual(report.untouched, [])
        self.assertEqual(len(report.loaded), len(src.state_dict()))
        self.assertIn("scope: all model keys", report.summary())

    def test_missing_key_raises_and_names_it(self):
        src, dst = Trainer(seed=3), Trainer(seed=4)
        sd = dict(src.state_dict())
        del sd["material.decoder.2.bias"]
        before = snapshot(dst)
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, sd, source="my.ckpt")
        msg = str(cm.exception)
        self.assertIn("missing from checkpoint (1)", msg)
        self.assertIn("material.decoder.2.bias", msg)
        self.assertIn("my.ckpt", msg)
        self.assertIsInstance(cm.exception, cio.IncompatibleCheckpointError)
        assert_equal_state(self, snapshot(dst), before)  # nothing was loaded

    def test_unexpected_key_raises_and_names_it(self):
        src, dst = Trainer(seed=3), Trainer(seed=4)
        sd = dict(src.state_dict())
        sd["material.decoder.film_mappers.0.weight"] = torch.zeros(2, 2)
        before = snapshot(dst)
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, sd, source="x")
        msg = str(cm.exception)
        self.assertIn("unexpected in checkpoint (1)", msg)
        self.assertIn("material.decoder.film_mappers.0.weight", msg)
        assert_equal_state(self, snapshot(dst), before)

    def test_shape_mismatch_raises_with_both_shapes(self):
        src, dst = Trainer(hidden=6, seed=3), Trainer(hidden=8, seed=4)
        before = snapshot(dst)
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, src.state_dict(), source="x")
        msg = str(cm.exception)
        self.assertIn("shape mismatch (", msg)
        self.assertIn("material.decoder.0.weight: checkpoint (6, 4) vs model (8, 4)", msg)
        self.assertNotIn("strict=False mode", msg)
        assert_equal_state(self, snapshot(dst), before)

    def test_error_groups_are_truncated_with_a_count(self):
        class Wide(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(15)])  # 30 tensors
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(Wide(), {}, source="x")
        msg = str(cm.exception)
        self.assertIn("missing from checkpoint (30)", msg)
        self.assertIn(f"... and {30 - cio.MAX_LISTED_KEYS} more", msg)

    # ---- explicit partial loading (allowlist) ------------------------------
    def test_decoder_only_allowlist_loads_decoder_and_reports_skips(self):
        src, dst = Trainer(seed=5), Trainer(seed=6)
        before = snapshot(dst)
        report = cio.load_state_dict_strict(
            dst, src.state_dict(), allow_prefixes=("material.decoder.",), source="stage1.ckpt")
        after = snapshot(dst)
        decoder_keys = [k for k in after if k.startswith("material.decoder.")]
        other_keys = [k for k in after if not k.startswith("material.decoder.")]
        self.assertEqual(len(decoder_keys), 4)
        assert_equal_state(self, after, snapshot(src), decoder_keys)   # decoder copied
        assert_equal_state(self, after, before, other_keys)            # rest untouched
        self.assertEqual(report.loaded, decoder_keys)
        self.assertEqual(sorted(report.skipped), sorted(other_keys))
        self.assertEqual(sorted(report.untouched), sorted(other_keys))
        summary = report.summary()
        self.assertIn("scope: material.decoder.*", summary)
        self.assertIn("skipped 4 checkpoint keys", summary)
        self.assertIn("material.latent_texture.params", summary)

    def test_allowlist_still_requires_every_scoped_key(self):
        src, dst = Trainer(seed=5), Trainer(seed=6)
        sd = dict(src.state_dict())
        del sd["material.decoder.2.weight"]           # in-scope key missing
        del sd["material.latent_texture.params"]      # out-of-scope key missing is fine
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, sd, allow_prefixes=("material.decoder.",), source="x")
        self.assertIn("material.decoder.2.weight", str(cm.exception))
        self.assertNotIn("material.latent_texture.params", str(cm.exception))

    def test_allowlist_rejects_unexpected_or_mismatched_in_scope(self):
        src, dst = Trainer(hidden=6, seed=5), Trainer(hidden=8, seed=6)
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, src.state_dict(), allow_prefixes=("material.decoder.",))
        self.assertIn("material.decoder.0.weight: checkpoint (6, 4) vs model (8, 4)", str(cm.exception))
        sd = dict(Trainer(seed=5).state_dict())
        sd["material.decoder.9.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, sd, allow_prefixes=("material.decoder.",))
        self.assertIn("unexpected in checkpoint (1)", str(cm.exception))

    def test_allowlist_matching_nothing_is_an_error(self):
        with self.assertRaises(RuntimeError):
            cio.load_state_dict_strict(Trainer(), Trainer().state_dict(), allow_prefixes=("nonexistent.",))
        with self.assertRaises(ValueError):
            cio.load_state_dict_strict(Trainer(), Trainer().state_dict(), allow_prefixes=())

    def test_ignore_prefixes_skip_rebuilt_state_on_both_sides(self):
        src, dst = Trainer(seed=7, n_lights=7), Trainer(seed=8, n_lights=9)  # emitter shapes differ
        with self.assertRaises(RuntimeError) as cm:
            cio.load_state_dict_strict(dst, src.state_dict(), source="x")
        self.assertIn("emitter.light_positions", str(cm.exception))
        before = snapshot(dst)
        report = cio.load_state_dict_strict(
            dst, src.state_dict(), ignore_prefixes=("emitter.",), source="x")
        after = snapshot(dst)
        self.assertEqual(report.skipped, ["emitter.light_positions"])
        self.assertEqual(report.untouched, ["emitter.light_positions"])
        self.assertTrue(torch.equal(after["emitter.light_positions"], before["emitter.light_positions"]))
        loaded_keys = [k for k in after if not k.startswith("emitter.")]
        assert_equal_state(self, after, snapshot(src), loaded_keys)


if __name__ == "__main__":
    unittest.main()
