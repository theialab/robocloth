"""render.py checkpoint resolution must fail closed with actionable errors.

Needs the rendering environment (hydra, drjit, mitsuba, numpy, torch) because
render.py imports them at module level; no variant is set and no GPU is used.
Temporary checkpoint roots live under $ROBOCLOTH_TEST_TMP (if set, else the
system temp dir) and are removed after each test.
"""
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RENDER_DIR = os.path.join(REPO, "rendering")
if RENDER_DIR not in sys.path:
    sys.path.insert(0, RENDER_DIR)

import render  # noqa: E402

TMP_PARENT = os.environ.get("ROBOCLOTH_TEST_TMP") or None   # None -> tempfile's default location
if TMP_PARENT:
    os.makedirs(TMP_PARENT, exist_ok=True)


class ResolveCheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_PARENT, prefix="render_ckpt_test_")
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, *parts):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00")
        return path

    def test_nonexistent_root_is_a_clear_error(self):
        root = os.path.join(self.tmp, "no_such_root")
        with self.assertRaises(FileNotFoundError) as cm:
            render.resolve_checkpoint("cloth", {"material": "145"}, root)
        msg = str(cm.exception)
        self.assertIn("[cloth]", msg)
        self.assertIn("checkpoint_root is not a directory", msg)
        self.assertIn(root, msg)
        self.assertIn("BRDF_CKPT_ROOT", msg)

    def test_empty_root_is_a_clear_error(self):
        root = os.path.join(self.tmp, "empty_root")
        os.makedirs(root)
        with self.assertRaises(FileNotFoundError) as cm:
            render.resolve_checkpoint("cloth", {"material": "145"}, root)
        msg = str(cm.exception)
        self.assertIn("no folder for material '145'", msg)
        self.assertIn("empty directory", msg)
        self.assertIn("download_material.sh 145", msg)

    def test_material_folder_without_checkpoint_is_a_clear_error(self):
        root = os.path.join(self.tmp, "root")
        os.makedirs(os.path.join(root, "145"))
        self._touch("root", "145", "README.txt")
        with self.assertRaises(FileNotFoundError) as cm:
            render.resolve_checkpoint("cloth", {"material": "145"}, root)
        msg = str(cm.exception)
        self.assertIn("no checkpoint matches", msg)
        self.assertIn("Ours_epoch*.ckpt", msg)
        self.assertIn("README.txt", msg)

    def test_ambiguous_glob_keeps_existing_error(self):
        root = os.path.join(self.tmp, "root")
        self._touch("root", "145", "Ours_epoch100.ckpt")
        self._touch("root", "145", "Ours_epoch112.ckpt")
        with self.assertRaises(RuntimeError) as cm:
            render.resolve_checkpoint("cloth", {"material": "145"}, root)
        self.assertIn("ambiguous checkpoint glob", str(cm.exception))

    def test_single_match_resolves(self):
        root = os.path.join(self.tmp, "root")
        ckpt = self._touch("root", "145", "Ours_epoch112.ckpt")
        self._touch("root", "145", "Bonn_epoch112.ckpt")  # other model tags are not globbed
        self.assertEqual(render.resolve_checkpoint("cloth", {"material": 145}, root), ckpt)

    def test_missing_root_value(self):
        with self.assertRaises(ValueError) as cm:
            render.resolve_checkpoint("cloth", {"material": "145"}, "")
        self.assertIn("checkpoint_root", str(cm.exception))
        with self.assertRaises(ValueError):
            render.resolve_checkpoint("cloth", {"ckpt": "145/Ours_epoch112.ckpt"}, "")

    def test_explicit_ckpt_paths(self):
        root = os.path.join(self.tmp, "root")
        ckpt = self._touch("root", "145", "Ours_epoch112.ckpt")
        self.assertEqual(render.resolve_checkpoint("c", {"ckpt": ckpt}, ""), ckpt)
        self.assertEqual(render.resolve_checkpoint("c", {"ckpt": "145/Ours_epoch112.ckpt"}, root), ckpt)
        with self.assertRaises(FileNotFoundError):
            render.resolve_checkpoint("c", {"ckpt": os.path.join(root, "145", "missing.ckpt")}, "")
        with self.assertRaises(FileNotFoundError) as cm:
            render.resolve_checkpoint("c", {"ckpt": "x.ckpt"}, os.path.join(self.tmp, "nope"))
        self.assertIn("checkpoint_root is not a directory", str(cm.exception))

    def test_load_materials_end_to_end(self):
        scene_dir = os.path.join(self.tmp, "scene")
        os.makedirs(scene_dir)
        root = os.path.join(self.tmp, "root")
        ckpt = self._touch("root", "145", "Ours_epoch112.ckpt")
        spec = {"checkpoint_root": "${ROBOCLOTH_TEST_CKPT_ROOT:-/absolute/path/to/ckpts}",
                "assignments": {"cloth": {"material": "145", "uv_tiling": 5.0}}}
        with open(os.path.join(scene_dir, "materials.json"), "w") as f:
            json.dump(spec, f)
        os.environ.pop("ROBOCLOTH_TEST_CKPT_ROOT", None)
        with self.assertRaises(FileNotFoundError) as cm:   # stale placeholder root
            render.load_materials(scene_dir, default_two_sided=True)
        self.assertIn("/absolute/path/to/ckpts", str(cm.exception))
        os.environ["ROBOCLOTH_TEST_CKPT_ROOT"] = root
        try:
            overrides, radiance = render.load_materials(scene_dir, default_two_sided=True)
        finally:
            del os.environ["ROBOCLOTH_TEST_CKPT_ROOT"]
        self.assertEqual(overrides["cloth"]["model_path"], ckpt)
        self.assertIsNone(radiance)


if __name__ == "__main__":
    unittest.main()
