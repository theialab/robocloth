"""The Mitsuba BRDF plugin must fail closed on checkpoint problems.

Imports rendering/brdf_plugin/mlp.py under the CPU ``llvm_ad_rgb`` variant
(the plugin subclasses mi.BSDF, so a variant must be active) — no GPU and no
real checkpoint are needed: the absent-file check runs before any CUDA
allocation, and the strict-load policy itself is unit-tested in
test_checkpoint_io.py.
"""
import os
import sys
import unittest

import mitsuba as mi

if mi.variant() is None:
    mi.set_variant("llvm_ad_rgb")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RENDER_DIR = os.path.join(REPO, "rendering")
if RENDER_DIR not in sys.path:
    sys.path.insert(0, RENDER_DIR)

from brdf_plugin import mlp  # noqa: E402


class MlpFailClosedTest(unittest.TestCase):
    def test_no_strict_false_fallback_in_source(self):
        with open(os.path.join(RENDER_DIR, "brdf_plugin", "mlp.py")) as f:
            src = f.read()
        self.assertNotIn("strict=False", src)
        self.assertNotIn("Using randomly initialized model", src)
        self.assertNotIn("does not exist, using randomly initialized", src)

    def test_shares_the_training_policy_module(self):
        expected = os.path.join(REPO, "training", "utils", "checkpoint_io.py")
        self.assertEqual(os.path.realpath(mlp.checkpoint_io.__file__), os.path.realpath(expected))
        for name in ("load_checkpoint_file", "extract_state_dict", "split_namespace",
                     "load_state_dict_strict", "IncompatibleCheckpointError"):
            self.assertTrue(hasattr(mlp.checkpoint_io, name), name)
        self.assertIsInstance(mlp.IGNORED_MATERIAL_KEY_PREFIXES, tuple)

    def test_absent_checkpoint_raises_before_model_construction(self):
        missing = os.path.join(REPO, "no", "such", "Ours_epoch1.ckpt")
        with self.assertRaises(FileNotFoundError) as cm:
            mlp.create_anisotropic_model(missing, "AnisotropicLatentTexturedModel")
        self.assertIn(missing, str(cm.exception))

    def test_use_btf_without_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            mlp.create_anisotropic_model("", "UBOLatentBRDF", use_btf=True, btf_path="/no/such.btf")


if __name__ == "__main__":
    unittest.main()
