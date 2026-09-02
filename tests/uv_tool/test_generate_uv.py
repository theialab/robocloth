"""Tests for rendering/tools/generate_uv.py.

Two layers:

* helper tests that need no Blender (argument splitting, output-path rules,
  PLY header + triangle validation, the pre-Blender failure paths of main());
* end-to-end tests that drive the real tool on tiny programmatically written
  box meshes (ASCII OBJ, ASCII PLY without UVs, GLB, glTF) through both
  documented entry points:
    - ``blender --background --python generate_uv.py -- ...`` via ``$BLENDER``
      (default /usr/bin/blender; the class is skipped when no binary exists);
    - ``python generate_uv.py ...`` via ``$ROBOCLOTH_UV_PYTHON`` (an interpreter
      with the bpy wheel; skipped unless set).

For every format the tests check that the input's sha256 is unchanged, that the
output appears at the distinct default path and validates (ply magic, UV and
normal properties, faces > 0, every face a triangle), that the vertex positions
were not rotated, that a rerun without --force is refused, and that no temp
files are left behind. Every produced PLY is additionally loaded with Mitsuba's
own ``ply`` shape plugin -- the consumer render.py uses -- when ``mitsuba`` is
importable by the harness Python or ``$ROBOCLOTH_MITSUBA_PYTHON`` names an
interpreter that has it (otherwise that assertion is skipped).

Fixtures and outputs go to a fresh directory under ``$ROBOCLOTH_TEST_TMP``
(default: the system temp dir) and are removed afterwards unless
``ROBOCLOTH_TEST_KEEP=1``; nothing is written inside the repository (the
bytecode cache this module's own import creates is removed in tearDownModule).

Run:  python -m unittest discover -s tests/uv_tool -v
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Covers the modules loaded below (_fixtures, the tool). This module's own .pyc
# is written by `unittest discover` before this line runs; tearDownModule()
# removes it again, and PYTHONDONTWRITEBYTECODE=1 avoids it up front.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOL = REPO / "rendering" / "tools" / "generate_uv.py"
FIXTURE_SCRIPT = HERE / "_fixtures.py"

sys.path.insert(0, str(HERE))
import _fixtures  # noqa: E402


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("generate_uv", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_uv = _load_tool_module()


def tearDownModule():
    """Leave no files in the repo: drop the bytecode cache written for these two modules."""
    for source in (Path(__file__).resolve(), FIXTURE_SCRIPT):
        cached = Path(importlib.util.cache_from_source(str(source)))
        if cached.parent == HERE / "__pycache__" and cached.is_file():
            cached.unlink()
    pycache = HERE / "__pycache__"
    if pycache.is_dir() and not any(pycache.iterdir()):
        pycache.rmdir()


def _find_blender():
    candidates = [os.environ.get("BLENDER"), "/usr/bin/blender", shutil.which("blender")]
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


BLENDER = _find_blender()
UV_PYTHON = os.environ.get("ROBOCLOTH_UV_PYTHON") or None
MITSUBA_PYTHON = os.environ.get("ROBOCLOTH_MITSUBA_PYTHON") or None
TEST_TMP = os.environ.get("ROBOCLOTH_TEST_TMP") or None
KEEP = os.environ.get("ROBOCLOTH_TEST_KEEP") == "1"
CORNERS = set(_fixtures.CORNERS)
# The box's 6 quads are triangulated on export (Mitsuba's PLY loader needs
# triangles); glTF stores triangles to begin with.
EXPECTED_FACES = {"obj": 12, "ply": 12, "glb": 12, "gltf": 12}

_PLY_STRUCT = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
               "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
               "float": "f", "float32": "f", "double": "d", "float64": "d"}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_ply(path):
    """Minimal PLY reader (ascii / binary); returns {element name: [row dict, ...]}."""
    header = generate_uv.parse_ply_header(path)
    with open(path, "rb") as f:
        f.seek(header["header_bytes"])
        body = f.read()
    elements = {}
    if header["format"] == "ascii":
        tokens = body.decode("ascii").split()
        pos = 0
        for element in header["elements"]:
            rows = []
            for _ in range(element["count"]):
                row = {}
                for prop in element["properties"]:
                    if prop[0] == "list":
                        n = int(tokens[pos])
                        pos += 1
                        row[prop[-1]] = [float(t) for t in tokens[pos:pos + n]]
                        pos += n
                    else:
                        row[prop[-1]] = float(tokens[pos])
                        pos += 1
                rows.append(row)
            elements[element["name"]] = rows
    else:
        endian = "<" if header["format"] == "binary_little_endian" else ">"
        pos = 0
        for element in header["elements"]:
            rows = []
            for _ in range(element["count"]):
                row = {}
                for prop in element["properties"]:
                    if prop[0] == "list":
                        count_fmt = endian + _PLY_STRUCT[prop[1]]
                        item_fmt = endian + _PLY_STRUCT[prop[2]]
                        n = struct.unpack_from(count_fmt, body, pos)[0]
                        pos += struct.calcsize(count_fmt)
                        size = struct.calcsize(item_fmt)
                        row[prop[-1]] = [struct.unpack_from(item_fmt, body, pos + i * size)[0] for i in range(n)]
                        pos += n * size
                    else:
                        item_fmt = endian + _PLY_STRUCT[prop[0]]
                        row[prop[-1]] = struct.unpack_from(item_fmt, body, pos)[0]
                        pos += struct.calcsize(item_fmt)
                rows.append(row)
            elements[element["name"]] = rows
    return elements


def _header_text(path):
    with open(path, "rb") as f:
        raw = f.read(1 << 16)
    return raw.split(b"end_header", 1)[0].decode("ascii", errors="replace")


class ToolRunner:
    """Runs a bpy script either through a Blender binary or a Python with the bpy wheel."""

    def __init__(self, kind, executable):
        self.kind = kind
        self.executable = executable

    def command(self, script, args):
        if self.kind == "blender":
            return [self.executable, "--background", "--python", str(script), "--", *map(str, args)]
        return [self.executable, str(script), *map(str, args)]

    def run(self, args, script=TOOL, timeout=600):
        return subprocess.run(self.command(script, args), capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------ Mitsuba (consumer)

_MITSUBA_SNIPPET = """\
import json, sys
import mitsuba as mi
mi.set_variant("scalar_rgb")
report = {}
for path in sys.argv[1:]:
    try:
        mesh = mi.load_dict({"type": "ply", "filename": path})
        report[path] = {"faces": mesh.face_count(), "vertices": mesh.vertex_count(),
                        "uv": bool(mesh.has_vertex_texcoords()), "normals": bool(mesh.has_vertex_normals())}
    except Exception as exc:
        report[path] = {"error": str(exc).splitlines()[0] if str(exc) else repr(exc)}
print("MITSUBA_JSON " + json.dumps(report))
"""


class MitsubaProbe:
    """Loads a PLY with Mitsuba's 'ply' shape plugin, the way render.py's scenes do.

    Uses mitsuba in-process when the harness Python can import it, otherwise
    the interpreter named by $ROBOCLOTH_MITSUBA_PYTHON. `available` is False
    when neither exists; callers then skip the Mitsuba assertions.
    """

    def __init__(self):
        self.in_process = importlib.util.find_spec("mitsuba") is not None
        self.python = MITSUBA_PYTHON
        self.available = self.in_process or bool(self.python)

    def load(self, path):
        """Return {'faces', 'vertices', 'uv', 'normals'} or {'error': message}."""
        if self.in_process:
            return self._load_in_process(path)
        proc = subprocess.run([self.python, "-c", _MITSUBA_SNIPPET, path],
                              capture_output=True, text=True, timeout=600)
        for line in proc.stdout.splitlines():
            if line.startswith("MITSUBA_JSON "):
                return json.loads(line[len("MITSUBA_JSON "):])[path]
        raise RuntimeError(f"Mitsuba probe did not run (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    @staticmethod
    def _load_in_process(path):
        import mitsuba as mi
        if not mi.variant():
            mi.set_variant("scalar_rgb")
        try:
            mesh = mi.load_dict({"type": "ply", "filename": path})
        except Exception as exc:
            return {"error": str(exc).splitlines()[0] if str(exc) else repr(exc)}
        return {"faces": mesh.face_count(), "vertices": mesh.vertex_count(),
                "uv": bool(mesh.has_vertex_texcoords()), "normals": bool(mesh.has_vertex_normals())}


MITSUBA = MitsubaProbe()


# ----------------------------------------------------------- no Blender needed

class TestHelpersWithoutBlender(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="uv_tool_unit_", dir=TEST_TMP)
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name, content=b"x"):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _synthetic_ply(self, name, vertex_props, faces=1, vertices=3, body=b"\x00" * 64):
        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {vertices}"]
        lines += [f"property float {p}" for p in vertex_props]
        lines += [f"element face {faces}", "property list uchar uint vertex_indices", "end_header"]
        return self._touch(name, ("\n".join(lines) + "\n").encode("ascii") + body)

    def _ply_with_faces(self, name, faces, ascii_format=False):
        """A complete PLY: 4 vertices (x y z nx ny nz s t) and the given faces (index tuples)."""
        vertices = [(0, 0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 1, 1, 0),
                    (0, 1, 0, 0, 0, 1, 0, 1), (1, 1, 0, 0, 0, 1, 1, 1)]
        lines = ["ply", f"format {'ascii' if ascii_format else 'binary_little_endian'} 1.0",
                 f"element vertex {len(vertices)}"]
        lines += [f"property float {p}" for p in ("x", "y", "z", "nx", "ny", "nz", "s", "t")]
        lines += [f"element face {len(faces)}", "property list uchar uint vertex_indices", "end_header"]
        head = ("\n".join(lines) + "\n").encode("ascii")
        if ascii_format:
            body = "".join(" ".join(f"{v:g}" for v in vert) + "\n" for vert in vertices)
            body += "".join(f"{len(face)} " + " ".join(map(str, face)) + "\n" for face in faces)
            body = body.encode("ascii")
        else:
            body = b"".join(struct.pack("<8f", *vert) for vert in vertices)
            body += b"".join(struct.pack(f"<B{len(face)}I", len(face), *face) for face in faces)
        return self._touch(name, head + body)

    def test_split_cli_args_blender_style(self):
        argv = ["/usr/bin/blender", "-b", "--python", "generate_uv.py", "--", "m.obj", "--output", "o.ply"]
        self.assertEqual(generate_uv.split_cli_args(argv), ["m.obj", "--output", "o.ply"])

    def test_split_cli_args_plain_python(self):
        self.assertEqual(generate_uv.split_cli_args(["generate_uv.py", "m.obj", "--force"]), ["m.obj", "--force"])

    def test_split_cli_args_missing_separator_under_blender(self):
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.split_cli_args(["/usr/bin/blender", "--background", "--python", "generate_uv.py", "m.obj"])
        self.assertIn("'--'", str(ctx.exception))

    def test_default_output_path(self):
        self.assertEqual(generate_uv.default_output_path("/a/b/mesh.obj"), "/a/b/mesh_uv.ply")
        self.assertEqual(generate_uv.default_output_path("/a/b/mesh.PLY"), "/a/b/mesh_uv.ply")

    def test_resolve_paths_defaults_and_case_insensitive_extension(self):
        for name in ("m.obj", "M.OBJ", "m.ply", "m.glb", "m.GLTF"):
            src = self._touch(name)
            inp, out = generate_uv.resolve_paths(src)
            self.assertEqual(inp, src)
            self.assertEqual(out, os.path.splitext(src)[0] + "_uv.ply")

    def test_resolve_paths_rejects_unsupported_and_missing_input(self):
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.resolve_paths(self._touch("m.stl"))
        self.assertIn("unsupported", str(ctx.exception))
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.resolve_paths(os.path.join(self.dir, "missing.obj"))
        self.assertIn("not found", str(ctx.exception))

    def test_resolve_paths_rejects_output_equal_to_input(self):
        src = self._touch("m.ply")
        for spelling in (src, os.path.join(self.dir, ".", "m.ply"), os.path.relpath(src)):
            with self.assertRaises(generate_uv.UVToolError) as ctx:
                generate_uv.resolve_paths(src, output=spelling, force=True)
            self.assertIn("input file", str(ctx.exception))

    def test_resolve_paths_rejects_non_ply_output(self):
        src = self._touch("m.obj")
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.resolve_paths(src, output=os.path.join(self.dir, "out.obj"))
        self.assertIn(".ply", str(ctx.exception))

    def test_resolve_paths_existing_output_needs_force(self):
        src = self._touch("m.obj")
        existing = self._touch("m_uv.ply")
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.resolve_paths(src)
        self.assertIn("--force", str(ctx.exception))
        self.assertEqual(generate_uv.resolve_paths(src, force=True)[1], existing)
        self.assertEqual(generate_uv.resolve_paths(src, output=os.path.join(self.dir, "new.PLY"))[1],
                         os.path.join(self.dir, "new.PLY"))

    def test_validate_ply_accepts_uv_and_normals(self):
        good = self._synthetic_ply("good.ply", ["x", "y", "z", "nx", "ny", "nz", "s", "t"], faces=2)
        summary = generate_uv.validate_ply(good)
        self.assertEqual(summary["faces"], 2)
        self.assertEqual(summary["uv_properties"], ("s", "t"))
        self.assertIsNone(summary["triangles"])  # face data is not read without require_triangles
        uv_variant = self._synthetic_ply("uv.ply", ["x", "y", "z", "nx", "ny", "nz", "u", "v"])
        self.assertEqual(generate_uv.validate_ply(uv_variant)["uv_properties"], ("u", "v"))

    def test_validate_ply_rejects_bad_files(self):
        cases = {
            "no_uv.ply": dict(vertex_props=["x", "y", "z", "nx", "ny", "nz"]),
            "no_normals.ply": dict(vertex_props=["x", "y", "z", "s", "t"]),
            "no_faces.ply": dict(vertex_props=["x", "y", "z", "nx", "ny", "nz", "s", "t"], faces=0),
            "no_body.ply": dict(vertex_props=["x", "y", "z", "nx", "ny", "nz", "s", "t"], body=b""),
        }
        for name, kwargs in cases.items():
            with self.assertRaises(generate_uv.UVToolError, msg=name):
                generate_uv.validate_ply(self._synthetic_ply(name, **kwargs))
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.validate_ply(self._touch("not.ply", b"solid cube\n"))
        self.assertIn("magic", str(ctx.exception))
        with self.assertRaises(generate_uv.UVToolError):
            generate_uv.validate_ply(os.path.join(self.dir, "absent.ply"))

    def test_validate_ply_triangle_check(self):
        for ascii_format in (False, True):
            tag = "ascii" if ascii_format else "bin"
            tri = self._ply_with_faces(f"tri_{tag}.ply", [(0, 1, 2), (1, 3, 2)], ascii_format)
            summary = generate_uv.validate_ply(tri, require_triangles=True)
            self.assertEqual((summary["faces"], summary["triangles"]), (2, True))
            quad = self._ply_with_faces(f"quad_{tag}.ply", [(0, 1, 3, 2)], ascii_format)
            generate_uv.validate_ply(quad)  # header-only checks still pass ...
            with self.assertRaises(generate_uv.UVToolError) as ctx:
                generate_uv.validate_ply(quad, require_triangles=True)  # ... the triangle check does not
            self.assertIn("1 of 1 faces are not triangles", str(ctx.exception))
            self.assertIn("Mitsuba", str(ctx.exception))
            mixed = self._ply_with_faces(f"mixed_{tag}.ply", [(0, 1, 2), (0, 1, 3, 2), (1, 3, 2)], ascii_format)
            with self.assertRaises(generate_uv.UVToolError) as ctx:
                generate_uv.validate_ply(mixed, require_triangles=True)
            self.assertIn("1 of 3 faces are not triangles", str(ctx.exception))
        # a binary body cut short inside the last face
        with open(self._ply_with_faces("full.ply", [(0, 1, 2), (1, 3, 2)]), "rb") as f:
            data = f.read()
        with self.assertRaises(generate_uv.UVToolError) as ctx:
            generate_uv.validate_ply(self._touch("truncated.ply", data[:-5]), require_triangles=True)
        self.assertIn("truncated", str(ctx.exception))

    def test_main_fails_before_blender_on_bad_paths(self):
        src = self._touch("m.obj")
        for args in (["missing.obj"], [src, "--output", "x.obj"], [src, "--output", src],
                     [self._touch("m.stl")]):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = generate_uv.main(["generate_uv.py", *args])
            self.assertEqual(code, 1, msg=args)
            self.assertIn("ERROR", stderr.getvalue())
        # a nonexistent output dir is caught as well (no bpy needed)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(generate_uv.main(["generate_uv.py", src, "--output",
                                               os.path.join(self.dir, "nope", "o.ply")]), 1)
        self.assertIn("directory", stderr.getvalue())

    def test_main_rejects_triangulate_with_keep_polygons(self):
        src = self._touch("m.obj")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            generate_uv.main(["generate_uv.py", src, "--triangulate", "--keep-polygons"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("mutually exclusive", stderr.getvalue())

    def test_main_reports_missing_bpy(self):
        if importlib.util.find_spec("bpy") is not None:
            self.skipTest("bpy is importable here; the no-bpy message cannot be exercised")
        src = self._touch("m.obj")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(generate_uv.main(["generate_uv.py", src]), 1)
        self.assertIn("bpy", stderr.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.dir, "m_uv.ply")))


# ------------------------------------------------------------- end-to-end mixin

class _EndToEndCases:
    """End-to-end cases; concrete subclasses bind `runner` (Blender binary or bpy Python)."""
    runner = None
    skip_reason = "no runner"

    @classmethod
    def setUpClass(cls):
        if cls.runner is None:
            raise unittest.SkipTest(cls.skip_reason)
        cls.root = tempfile.mkdtemp(prefix="robocloth_uv_tool_", dir=TEST_TMP)
        fixture_dir = os.path.join(cls.root, "fixtures")
        proc = cls.runner.run([fixture_dir], script=FIXTURE_SCRIPT)
        if proc.returncode != 0 or "FIXTURES_OK" not in proc.stdout:
            raise RuntimeError(f"fixture generation failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        cls.fixtures = {"obj": os.path.join(fixture_dir, "box.obj"),
                        "ply": os.path.join(fixture_dir, "box.ply"),
                        "glb": os.path.join(fixture_dir, "box.glb"),
                        "gltf": os.path.join(fixture_dir, "box_separate.gltf")}
        for path in list(cls.fixtures.values()) + [os.path.join(fixture_dir, "box_separate.bin")]:
            if not os.path.isfile(path):
                raise RuntimeError(f"fixture missing: {path}")

    @classmethod
    def tearDownClass(cls):
        if not KEEP and getattr(cls, "root", None):
            shutil.rmtree(cls.root, ignore_errors=True)

    # -- helpers ------------------------------------------------------------

    def _case_dir(self):
        return tempfile.mkdtemp(prefix=self._testMethodName + "_", dir=self.root)

    def _stage(self, key, name=None):
        """Copy a fixture (plus the glTF .bin sidecar) into a fresh case dir; returns the input path."""
        src = self.fixtures[key]
        case = self._case_dir()
        dst = os.path.join(case, name or os.path.basename(src))
        shutil.copyfile(src, dst)
        if key == "gltf":
            shutil.copyfile(src[:-len(".gltf")] + ".bin", os.path.join(case, "box_separate.bin"))
        return dst

    def _write(self, case, name, text):
        path = os.path.join(case, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def _run(self, *args):
        return self.runner.run(list(args))

    def _describe(self, proc):
        return f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"

    def _assert_ok(self, proc):
        self.assertEqual(proc.returncode, 0, msg=self._describe(proc))

    def _assert_fails(self, proc, *needles):
        self.assertNotEqual(proc.returncode, 0, msg="expected failure but got\n" + self._describe(proc))
        for needle in needles:
            self.assertIn(needle, proc.stderr, msg=self._describe(proc))
        self.assertNotIn("Traceback", proc.stderr, msg=self._describe(proc))

    def _assert_mitsuba_loads(self, path, expect_faces):
        """render.py's consumer must accept the file: Mitsuba's PLY plugin, triangles only."""
        info = MITSUBA.load(path)
        self.assertNotIn("error", info, msg=f"Mitsuba rejected {path}: {info.get('error')}")
        self.assertEqual(info["faces"], expect_faces, info)
        self.assertTrue(info["uv"] and info["normals"], info)
        return info

    def _check_output_ply(self, path, expect_faces):
        with open(path, "rb") as f:
            self.assertEqual(f.read(3), b"ply")
        summary = generate_uv.validate_ply(path, require_triangles=True)  # the tool's own validator
        header = _header_text(path)                                       # plus independent header checks
        has_st = "property float s" in header and "property float t" in header
        has_uv = "property float u" in header and "property float v" in header
        self.assertTrue(has_st or has_uv, msg=header)
        for name in ("nx", "ny", "nz"):
            self.assertIn(f"property float {name}", header)
        self.assertGreater(summary["faces"], 0)
        self.assertEqual(summary["faces"], expect_faces)
        elements = read_ply(path)
        self.assertEqual(len(elements["face"]), expect_faces)
        self.assertTrue(all(len(face["vertex_indices"]) == 3 for face in elements["face"]), "non-triangle face")
        positions = {tuple(round(row[k], 4) for k in "xyz") for row in elements["vertex"]}
        self.assertEqual(positions, CORNERS, "vertex positions changed (axis conversion?)")
        u_name, v_name = summary["uv_properties"]
        uvs = {(round(row[u_name], 4), round(row[v_name], 4)) for row in elements["vertex"]}
        self.assertGreater(len(uvs), 1, "all UVs identical")
        for u, v in uvs:
            self.assertTrue(-1e-4 <= u <= 1 + 1e-4 and -1e-4 <= v <= 1 + 1e-4, (u, v))
        if MITSUBA.available:
            self._assert_mitsuba_loads(path, expect_faces)
        return elements, summary

    def _no_temp_files(self, directory):
        leftovers = [n for n in os.listdir(directory) if ".tmp" in n]
        self.assertEqual(leftovers, [])

    def _roundtrip(self, key, name=None):
        src = self._stage(key, name)
        before = sha256(src)
        expected_out = os.path.splitext(src)[0] + "_uv.ply"
        self.assertNotEqual(expected_out, src)
        self.assertFalse(os.path.exists(expected_out))

        proc = self._run(src)
        self._assert_ok(proc)
        self.assertIn(f"{EXPECTED_FACES[key]} triangles", proc.stdout)
        self.assertEqual(sha256(src), before, "input file was modified")
        self.assertTrue(os.path.isfile(expected_out), "output not at the default path")
        self._check_output_ply(expected_out, EXPECTED_FACES[key])
        out_sha = sha256(expected_out)

        # rerun without --force: refused, both files untouched
        self._assert_fails(self._run(src), "already exists", "--force")
        self.assertEqual(sha256(src), before)
        self.assertEqual(sha256(expected_out), out_sha)

        # --force replaces the output; input still untouched
        self._assert_ok(self._run(src, "--force"))
        self.assertEqual(sha256(src), before)
        self._check_output_ply(expected_out, EXPECTED_FACES[key])
        self._no_temp_files(os.path.dirname(src))

    # -- per-format ----------------------------------------------------------

    def test_obj(self):
        self._roundtrip("obj")

    def test_ply_without_uvs(self):
        self._roundtrip("ply")

    def test_glb(self):
        self._roundtrip("glb")

    def test_gltf_separate(self):
        self._roundtrip("gltf")

    def test_uppercase_extension(self):
        self._roundtrip("obj", name="BOX.OBJ")

    def test_ngon_obj_is_triangulated(self):
        # one hexagon face -> 4 triangles; Smart UV on a single planar face
        case = self._case_dir()
        lines = ["# hexagon fixture"]
        for k in range(6):
            angle = math.tau * k / 6
            lines.append(f"v {math.cos(angle):.6f} {math.sin(angle):.6f} 0")
        lines.append("f 1 2 3 4 5 6")
        src = self._write(case, "hexagon.obj", "\n".join(lines) + "\n")
        before = sha256(src)
        proc = self._run(src)
        self._assert_ok(proc)
        self.assertIn("4 triangles", proc.stdout)
        self.assertEqual(sha256(src), before)
        out = os.path.join(case, "hexagon_uv.ply")
        summary = generate_uv.validate_ply(out, require_triangles=True)
        self.assertEqual((summary["faces"], summary["triangles"]), (4, True))
        faces = read_ply(out)["face"]
        self.assertTrue(all(len(face["vertex_indices"]) == 3 for face in faces))
        if MITSUBA.available:
            self._assert_mitsuba_loads(out, 4)

    # -- refusals ------------------------------------------------------------

    def test_output_equal_to_input_is_refused(self):
        src = self._stage("ply")
        before = sha256(src)
        for spelling in (src, os.path.join(os.path.dirname(src), ".", os.path.basename(src))):
            self._assert_fails(self._run(src, "--output", spelling, "--force"), "input file")
            self.assertEqual(sha256(src), before, "input file was modified")
        self._no_temp_files(os.path.dirname(src))

    def test_non_ply_output_is_refused(self):
        src = self._stage("obj")
        before = sha256(src)
        out = os.path.join(os.path.dirname(src), "unwrapped.obj")
        self._assert_fails(self._run(src, "--output", out), ".ply")
        self.assertFalse(os.path.exists(out))
        self.assertEqual(sha256(src), before)

    def test_unsupported_extension_is_refused(self):
        src = self._stage("obj", name="box.stl")
        before = sha256(src)
        self._assert_fails(self._run(src), "unsupported")
        self.assertEqual(sha256(src), before)
        self.assertFalse(os.path.exists(os.path.splitext(src)[0] + "_uv.ply"))

    def test_missing_input_is_refused(self):
        self._assert_fails(self._run(os.path.join(self._case_dir(), "absent.obj")), "not found")

    def test_mesh_without_faces_is_refused(self):
        # a vertex-only PLY imports as a mesh with 0 polygons: refuse with a reason, not a traceback
        case = self._case_dir()
        src = self._write(case, "points.ply",
                          "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                          "property float z\nelement face 0\nproperty list uchar int vertex_indices\n"
                          "end_header\n0 0 0\n1 0 0\n0 1 0\n")
        before = sha256(src)
        self._assert_fails(self._run(src), "no faces")
        self.assertEqual(sha256(src), before)
        self.assertFalse(os.path.exists(os.path.join(case, "points_uv.ply")))
        self._no_temp_files(case)

    def test_unreadable_glb_is_refused(self):
        case = self._case_dir()
        src = self._write(case, "empty.glb", "")
        self._assert_fails(self._run(src), "failed to import")
        self.assertFalse(os.path.exists(os.path.join(case, "empty_uv.ply")))
        self._no_temp_files(case)

    # -- options -------------------------------------------------------------

    def test_explicit_output_triangulate_ascii(self):
        # --triangulate is the default and stays accepted; --ascii must also load in Mitsuba
        src = self._stage("obj")
        before = sha256(src)
        out = os.path.join(os.path.dirname(src), "custom.ply")
        self._assert_ok(self._run(src, "--output", out, "--triangulate", "--ascii",
                                  "--angle-limit", "66", "--island-margin", "0.01"))
        self.assertEqual(sha256(src), before)
        elements, summary = self._check_output_ply(out, expect_faces=12)
        self.assertEqual(summary["format"], "ascii")
        self.assertFalse(os.path.exists(os.path.splitext(src)[0] + "_uv.ply"))

    def test_keep_polygons(self):
        src = self._stage("obj")
        before = sha256(src)
        out = os.path.join(os.path.dirname(src), "quads.ply")
        proc = self._run(src, "--output", out, "--keep-polygons")
        self._assert_ok(proc)
        self.assertIn("WARNING", proc.stdout)
        self.assertIn("6 faces (polygons kept)", proc.stdout)
        self.assertEqual(sha256(src), before)
        summary = generate_uv.validate_ply(out)  # header contract still holds ...
        self.assertEqual((summary["faces"], summary["triangles"]), (6, None))
        with self.assertRaises(generate_uv.UVToolError):  # ... but it is not a triangle mesh
            generate_uv.validate_ply(out, require_triangles=True)
        faces = read_ply(out)["face"]
        self.assertEqual(sorted(len(face["vertex_indices"]) for face in faces), [4] * 6)

    def test_mitsuba_loads_outputs(self):
        if not MITSUBA.available:
            self.skipTest("mitsuba is not importable and $ROBOCLOTH_MITSUBA_PYTHON is unset")
        src = self._stage("obj")
        self._assert_ok(self._run(src))
        out_ascii = os.path.join(os.path.dirname(src), "ascii.ply")
        self._assert_ok(self._run(src, "--output", out_ascii, "--ascii"))
        for path in (os.path.splitext(src)[0] + "_uv.ply", out_ascii):
            info = self._assert_mitsuba_loads(path, 12)
            self.assertEqual(info["vertices"], len(read_ply(path)["vertex"]))

    def test_existing_uv_map_is_kept(self):
        # run once, then feed the UV-mapped output back in: the tool must keep its UV map
        src = self._stage("obj")
        self._assert_ok(self._run(src))
        first = os.path.splitext(src)[0] + "_uv.ply"
        proc = self._run(first)
        self._assert_ok(proc)
        self.assertIn("already exists, skipping unwrapping", proc.stdout)
        second = os.path.splitext(first)[0] + "_uv.ply"
        self.assertTrue(os.path.isfile(second))
        first_uv = read_ply(first)
        second_uv = read_ply(second)
        self.assertEqual(sorted((round(r["s"], 4), round(r["t"], 4)) for r in first_uv["vertex"]),
                         sorted((round(r["s"], 4), round(r["t"], 4)) for r in second_uv["vertex"]))


class TestWithBlenderBinary(_EndToEndCases, unittest.TestCase):
    runner = ToolRunner("blender", BLENDER) if BLENDER else None
    skip_reason = "no Blender binary found (set $BLENDER)"

    def test_missing_separator_gives_hint(self):
        src = self._stage("obj")
        proc = subprocess.run([BLENDER, "--background", "--python", str(TOOL), src],
                              capture_output=True, text=True, timeout=600)
        self.assertNotEqual(proc.returncode, 0, msg=self._describe(proc))
        self.assertIn("'--'", proc.stderr, msg=self._describe(proc))
        self.assertFalse(os.path.exists(os.path.splitext(src)[0] + "_uv.ply"))


class TestWithBpyPython(_EndToEndCases, unittest.TestCase):
    runner = ToolRunner("python", UV_PYTHON) if UV_PYTHON else None
    skip_reason = "set $ROBOCLOTH_UV_PYTHON to a Python interpreter with the bpy wheel installed"


if __name__ == "__main__":
    unittest.main()
