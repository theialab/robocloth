"""UV toolkit: unwrap a mesh that has no UV coordinates.

The neural material is a latent *texture*, so every shape it is applied to
needs a UV parametrization. This tool Smart-UV-unwraps an OBJ / PLY / GLB /
glTF mesh in headless Blender and exports a triangulated, UV-mapped PLY
(positions, normals, UVs) that Mitsuba's PLY loader -- and therefore
render.py -- accepts; combine with the per-object "uv_tiling" knob in
materials.json to control the texture repeat density.

Two supported ways to run it (details: docs/uv_tool.md, envs/uv.txt):

    # (a) a regular Blender installation (>= 3.6; 4.0.2 verified) -- note the '--'
    blender --background --python rendering/tools/generate_uv.py -- input_mesh.obj

    # (b) the bpy wheel in its own Python 3.11 env (bpy==4.5.13 verified)
    python rendering/tools/generate_uv.py input_mesh.obj

Options:
    --output <file>.ply    default: <input_stem>_uv.ply next to the input
    --force                replace an existing output file
    --angle-limit 89       Smart-UV angle limit in degrees
    --island-margin 0.02   Smart-UV island margin
    --keep-polygons        export the source quads/n-gons instead of triangles
                           (such a PLY is rejected by Mitsuba, i.e. render.py)
    --ascii                write an ASCII PLY (default: binary little-endian)
    --triangulate          accepted for compatibility; triangles are the default

Safety contract (an earlier version overwrote .obj inputs with a PLY):
  * the input file is never written to, whatever its format;
  * the output must be a .ply path that is not the input file; an existing
    output is only replaced with --force;
  * the PLY is written under a temporary name and only moved into place after
    it was re-read and checked (ply magic, UV + normal vertex properties, at
    least one face and -- unless --keep-polygons -- every face a triangle);
  * the exported geometry keeps the input file's coordinate frame;
  * every failure exits non-zero with a message (an uncaught exception under
    'blender --python' would otherwise exit 0).

The module imports without Blender: bpy is only imported inside generate_uv().
"""
import argparse
import math
import os
import sys
import traceback

SUPPORTED_INPUT_EXTENSIONS = (".obj", ".ply", ".glb", ".gltf")
OUTPUT_EXTENSION = ".ply"
# Accepted names of the per-vertex UV pair in the exported header (Blender's
# exporter writes s/t; Mitsuba's PLY loader also understands the others).
UV_PROPERTY_PAIRS = (("s", "t"), ("u", "v"), ("texture_u", "texture_v"), ("texture_s", "texture_t"))
NORMAL_PROPERTIES = ("nx", "ny", "nz")
FACE_INDEX_PROPERTIES = ("vertex_indices", "vertex_index")
PLY_FORMATS = ("ascii", "binary_little_endian", "binary_big_endian")
_PLY_SCALAR_BYTES = {"char": 1, "int8": 1, "uchar": 1, "uint8": 1, "short": 2, "int16": 2,
                     "ushort": 2, "uint16": 2, "int": 4, "int32": 4, "uint": 4, "uint32": 4,
                     "float": 4, "float32": 4, "double": 8, "float64": 8}

# Exporter axis settings, as (forward, up) in the C++ operators' spelling.
# Blender's own frame is Z-up / Y-forward, so ("Y", "Z") is the identity.
IDENTITY_AXES = ("Y", "Z")
# The glTF importer always converts Y-up glTF data to Blender's Z-up; exporting
# with these axes applies the inverse rotation, restoring the file's frame.
GLTF_AXES = ("NEGATIVE_Z", "Y")
# Same axes in the legacy python add-on exporter's spelling (Blender < 3.6).
_LEGACY_AXIS_NAME = {"Y": "Y", "Z": "Z", "NEGATIVE_Z": "-Z"}

LOG_PREFIX = "[generate_uv]"


class UVToolError(Exception):
    """A user-facing failure; main() prints it and exits non-zero."""


def log(message):
    print(f"{LOG_PREFIX} {message}", flush=True)


# --------------------------------------------------------------------------- CLI

def split_cli_args(argv):
    """Return the tool's own arguments from a raw sys.argv.

    Under 'blender --background --python generate_uv.py -- <args>' Blender keeps
    its own options in sys.argv and everything after the first '--' is ours.
    Under a plain interpreter there is no '--' and argv[1:] is ours.
    """
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    if any(arg in ("--python", "-P", "--background", "-b") for arg in argv[1:]):
        raise UVToolError(
            "no '--' separator found: under Blender the tool arguments must follow '--', e.g. "
            "blender --background --python rendering/tools/generate_uv.py -- input_mesh.obj")
    return argv[1:]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="generate_uv.py",
        description="Smart-UV-unwrap an OBJ/PLY/GLB/glTF mesh in headless Blender and write a "
                    "triangulated, UV-mapped PLY (positions, normals, UVs) for render.py.",
        epilog="Under Blender: blender --background --python rendering/tools/generate_uv.py -- <args>")
    parser.add_argument("input_path", help="input mesh: .obj, .ply, .glb or .gltf (never modified)")
    parser.add_argument("--output", default=None,
                        help="output .ply path (default: <input_stem>_uv.ply next to the input)")
    parser.add_argument("--force", action="store_true", help="replace the output file if it already exists")
    parser.add_argument("--angle-limit", type=float, default=89.0,
                        help="Smart UV Project angle limit in degrees, (0, 90] (default: 89)")
    parser.add_argument("--island-margin", type=float, default=0.02,
                        help="Smart UV Project island margin, >= 0 (default: 0.02)")
    parser.add_argument("--keep-polygons", action="store_true",
                        help="export the source quads/n-gons instead of triangles; Mitsuba's PLY loader "
                             "rejects such a file, so render.py cannot use it")
    parser.add_argument("--triangulate", action="store_true",
                        help="triangulate faces on export (the default; accepted for compatibility)")
    parser.add_argument("--ascii", action="store_true", help="write an ASCII PLY instead of binary little-endian")
    return parser


# ------------------------------------------------------------------------- paths

def default_output_path(input_path):
    stem, _ = os.path.splitext(input_path)
    return stem + "_uv" + OUTPUT_EXTENSION


def _same_file(path_a, path_b):
    if os.path.realpath(path_a) == os.path.realpath(path_b):
        return True
    try:
        return os.path.samefile(path_a, path_b)
    except OSError:
        return False


def resolve_paths(input_path, output=None, force=False):
    """Validate the input and choose/validate the output; returns absolute (input, output)."""
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise UVToolError(f"input mesh not found: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise UVToolError(f"unsupported input format '{ext or '(no extension)'}': {input_path} "
                          f"(expected one of {', '.join(SUPPORTED_INPUT_EXTENSIONS)})")
    output_path = os.path.abspath(output) if output else default_output_path(input_path)
    if os.path.splitext(output_path)[1].lower() != OUTPUT_EXTENSION:
        raise UVToolError(f"output must be a {OUTPUT_EXTENSION} file (this tool only writes PLY), "
                          f"got: {output_path}")
    if _same_file(input_path, output_path):
        raise UVToolError(f"output resolves to the input file, refusing to overwrite it: {output_path}")
    if os.path.exists(output_path) and not force:
        raise UVToolError(f"output already exists: {output_path} (pass --force to replace it)")
    out_dir = os.path.dirname(output_path)
    if not os.path.isdir(out_dir):
        raise UVToolError(f"output directory does not exist: {out_dir}")
    return input_path, output_path


def _temporary_output_path(output_path):
    out_dir, base = os.path.split(output_path)
    # Same directory as the final file (so the final move is a rename) and still
    # a .ply name (the exporter derives nothing from it, but keep it honest).
    return os.path.join(out_dir, f".{base}.{os.getpid()}.tmp{OUTPUT_EXTENSION}")


# ---------------------------------------------------------------- PLY validation

def parse_ply_header(path, max_header_bytes=1 << 20):
    """Parse a PLY header; returns {'format', 'elements': [{name, count, properties}], 'header_bytes'}."""
    elements = []
    current = None
    fmt = None
    with open(path, "rb") as f:
        magic = f.readline()
        if magic.rstrip(b"\r\n") != b"ply":
            raise UVToolError(f"{path} does not start with the 'ply' magic")
        consumed = len(magic)
        while True:
            line = f.readline()
            if not line:
                raise UVToolError(f"{path}: PLY header has no end_header line")
            consumed += len(line)
            if consumed > max_header_bytes:
                raise UVToolError(f"{path}: PLY header longer than {max_header_bytes} bytes")
            tokens = line.decode("ascii", errors="replace").split()
            if not tokens:
                continue
            key = tokens[0]
            if key == "format" and len(tokens) >= 2:
                fmt = tokens[1]
            elif key == "element" and len(tokens) >= 3:
                try:
                    count = int(tokens[2])
                except ValueError:
                    raise UVToolError(f"{path}: bad element count in header line {line!r}") from None
                current = {"name": tokens[1], "count": count, "properties": []}
                elements.append(current)
            elif key == "property" and len(tokens) >= 3 and current is not None:
                # ['float', 'x'] or ['list', 'uchar', 'uint', 'vertex_indices']; name is last
                current["properties"].append(tokens[1:])
            elif key == "end_header":
                break
    return {"format": fmt, "elements": elements, "header_bytes": consumed}


def _face_index_property(element):
    """The list property holding a face's vertex indices, or None for non-face elements."""
    if element["name"] != "face":
        return None
    lists = [prop for prop in element["properties"] if prop[0] == "list"]
    named = [prop for prop in lists if prop[-1] in FACE_INDEX_PROPERTIES]
    return (named or lists or [None])[0]


def face_arities(path, header):
    """Return {vertices per face: number of such faces}, read from the PLY body.

    Elements are consumed in header order: scalar-only elements by stride, list
    elements row by row. A binary face block that is all triangles has a fixed
    stride, so the common case is checked with one strided byte slice (no
    Python-level loop); anything else falls back to the row walk.
    """
    with open(path, "rb") as f:
        f.seek(header["header_bytes"])
        body = f.read()
    arities = {}
    malformed = UVToolError(f"{path}: PLY data ends before the declared elements (truncated or malformed file)")
    try:
        if header["format"] == "ascii":
            tokens = body.split()
            pos = 0
            for element in header["elements"]:
                index_prop = _face_index_property(element)
                for _ in range(element["count"]):
                    for prop in element["properties"]:
                        if prop[0] == "list":
                            n = int(tokens[pos])
                            pos += 1 + n
                            if prop is index_prop:
                                arities[n] = arities.get(n, 0) + 1
                        else:
                            pos += 1
            if pos > len(tokens):
                raise malformed
        else:
            byteorder = "little" if header["format"] == "binary_little_endian" else "big"
            pos = 0
            for element in header["elements"]:
                props = element["properties"]
                index_prop = _face_index_property(element)
                if not any(prop[0] == "list" for prop in props):
                    pos += element["count"] * sum(_PLY_SCALAR_BYTES[prop[0]] for prop in props)
                    continue
                if index_prop is not None and len(props) == 1 and _PLY_SCALAR_BYTES[index_prop[1]] == 1:
                    # Fast path: with a one-byte count, an all-triangle block puts
                    # a count byte of 3 at every multiple of the triangle stride.
                    stride = 1 + 3 * _PLY_SCALAR_BYTES[index_prop[2]]
                    block = body[pos:pos + element["count"] * stride]
                    if len(block) == element["count"] * stride and set(block[::stride]) == {3}:
                        arities[3] = arities.get(3, 0) + element["count"]
                        pos += len(block)
                        continue
                for _ in range(element["count"]):
                    for prop in props:
                        if prop[0] == "list":
                            count_bytes = _PLY_SCALAR_BYTES[prop[1]]
                            if pos + count_bytes > len(body):
                                raise malformed
                            n = int.from_bytes(body[pos:pos + count_bytes], byteorder)
                            pos += count_bytes + n * _PLY_SCALAR_BYTES[prop[2]]
                            if prop is index_prop:
                                arities[n] = arities.get(n, 0) + 1
                        else:
                            pos += _PLY_SCALAR_BYTES[prop[0]]
            if pos > len(body):
                raise malformed
    except (IndexError, ValueError):
        raise malformed from None
    except KeyError as exc:
        raise UVToolError(f"{path}: unsupported PLY property type {exc}") from None
    return arities


def validate_ply(path, require_triangles=False):
    """Check that `path` is a PLY with x/y/z + normals + UVs per vertex and > 0 faces.

    With require_triangles the face data is read as well and every face must
    have exactly three vertices (Mitsuba's PLY loader accepts nothing else).
    Returns a summary dict ({'format', 'vertices', 'faces', 'uv_properties',
    'triangles'}); raises UVToolError describing the first problem found.
    """
    if not os.path.isfile(path):
        raise UVToolError(f"expected PLY file was not written: {path}")
    header = parse_ply_header(path)
    if header["format"] not in PLY_FORMATS:
        raise UVToolError(f"{path}: unknown PLY format {header['format']!r}")
    by_name = {element["name"]: element for element in header["elements"]}
    vertex = by_name.get("vertex")
    if vertex is None or vertex["count"] <= 0:
        raise UVToolError(f"{path}: PLY header declares no vertices")
    names = [prop[-1] for prop in vertex["properties"]]
    if not all(name in names for name in ("x", "y", "z")):
        raise UVToolError(f"{path}: vertex element lacks x/y/z properties (found {names})")
    uv_pair = next((pair for pair in UV_PROPERTY_PAIRS if all(name in names for name in pair)), None)
    if uv_pair is None:
        expected = " / ".join("+".join(pair) for pair in UV_PROPERTY_PAIRS)
        raise UVToolError(f"{path}: vertex element declares no UV properties (expected {expected}; "
                          f"found {names})")
    if not all(name in names for name in NORMAL_PROPERTIES):
        raise UVToolError(f"{path}: vertex element lacks normal properties nx/ny/nz (found {names})")
    face = by_name.get("face")
    if face is None or face["count"] <= 0:
        raise UVToolError(f"{path}: PLY header declares no faces")
    if os.path.getsize(path) <= header["header_bytes"]:
        raise UVToolError(f"{path}: PLY has a header but no data")
    summary = {"format": header["format"], "vertices": vertex["count"], "faces": face["count"],
               "uv_properties": uv_pair, "triangles": None}
    if require_triangles:
        arities = face_arities(path, header)
        found = sum(arities.values())
        if found != face["count"]:
            raise UVToolError(f"{path}: PLY header declares {face['count']} faces but {found} were read")
        non_triangles = sum(count for arity, count in arities.items() if arity != 3)
        if non_triangles:
            raise UVToolError(
                f"{path}: {non_triangles} of {found} faces are not triangles (vertices per face: "
                f"{sorted(arities)}); Mitsuba's PLY loader needs a triangle mesh, render.py could not "
                f"load this file")
        summary["triangles"] = True
    return summary


# ------------------------------------------------------------------ Blender side

def _require_bpy():
    try:
        import bpy
    except ImportError as exc:
        raise UVToolError(
            "Blender's Python API (bpy) is not importable. Run the tool as "
            "'blender --background --python rendering/tools/generate_uv.py -- <args>' "
            "or install the bpy wheel into its own env (see envs/uv.txt)") from exc
    return bpy


def _operator_exists(bpy, module, name):
    """True if bpy.ops.<module>.<name> is registered (plain hasattr is always True)."""
    try:
        getattr(getattr(bpy.ops, module), name).get_rna_type()
    except Exception:
        return False
    return True


def _operator_has_property(bpy, module, name, prop):
    """True if the registered operator bpy.ops.<module>.<name> takes the option `prop`."""
    try:
        rna = getattr(getattr(bpy.ops, module), name).get_rna_type()
    except Exception:
        return False
    return any(p.identifier == prop for p in rna.properties)


def _select_only(bpy, objects):
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    for ob in objects:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def _first_line(exc):
    text = (str(exc).strip() or exc.__class__.__name__).splitlines()[0]
    return text[len("Error: "):] if text.startswith("Error: ") else text


def _run_importer(operator, input_path, **options):
    """Call an import operator; an unreadable file becomes a UVToolError, not a traceback."""
    try:
        result = operator(**options)
    except RuntimeError as exc:  # Blender raises this when an operator reports an error
        raise UVToolError(f"failed to import {input_path}: {_first_line(exc)}") from None
    if "FINISHED" not in result:
        raise UVToolError(f"failed to import {input_path} (importer returned {set(result)})")


def import_mesh(bpy, input_path):
    """Import `input_path` with the importer matching its extension.

    Returns (imported mesh objects, exporter axes that keep the file's frame).
    OBJ and PLY are imported with an identity axis mapping; glTF is always
    rotated to Z-up by Blender's importer, so its export axes undo that.
    """
    ext = os.path.splitext(input_path)[1].lower()
    before = set(bpy.context.scene.objects)
    export_axes = IDENTITY_AXES
    if ext == ".obj":
        if _operator_exists(bpy, "wm", "obj_import"):          # Blender >= 3.3 (C++ importer)
            _run_importer(bpy.ops.wm.obj_import, input_path, filepath=input_path, forward_axis="Y", up_axis="Z")
        elif _operator_exists(bpy, "import_scene", "obj"):     # legacy python add-on
            _run_importer(bpy.ops.import_scene.obj, input_path, filepath=input_path, axis_forward="Y", axis_up="Z")
        else:
            raise UVToolError("this Blender has no OBJ importer (wm.obj_import / import_scene.obj)")
    elif ext == ".ply":
        if _operator_exists(bpy, "wm", "ply_import"):          # Blender >= 3.6 (C++ importer)
            _run_importer(bpy.ops.wm.ply_import, input_path, filepath=input_path, forward_axis="Y", up_axis="Z")
        elif _operator_exists(bpy, "import_mesh", "ply"):      # legacy python add-on
            _run_importer(bpy.ops.import_mesh.ply, input_path, filepath=input_path)
        else:
            raise UVToolError("this Blender has no PLY importer (wm.ply_import / import_mesh.ply)")
    elif ext in (".glb", ".gltf"):
        if not _operator_exists(bpy, "import_scene", "gltf"):
            raise UVToolError("this Blender has no glTF importer (import_scene.gltf; "
                              "is the glTF 2.0 add-on disabled?)")
        _run_importer(bpy.ops.import_scene.gltf, input_path, filepath=input_path)
        export_axes = GLTF_AXES
    else:  # resolve_paths() already rejected this
        raise UVToolError(f"unsupported input format '{ext}'")
    imported = [ob for ob in bpy.context.scene.objects if ob not in before]
    meshes = [ob for ob in imported if ob.type == "MESH"]
    if not meshes:
        raise UVToolError(f"no mesh objects were imported from {input_path}")
    return meshes, export_axes


def merge_objects(bpy, meshes):
    """Join several imported mesh objects into one so the PLY holds the whole file."""
    if len(meshes) == 1:
        return meshes[0]
    log(f"{len(meshes)} mesh objects imported, joining them into one")
    _select_only(bpy, meshes)
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def check_mesh_has_faces(obj, input_path):
    """A point cloud or a wireframe cannot be unwrapped (or rendered); say so plainly."""
    mesh = obj.data
    if len(mesh.vertices) == 0:
        raise UVToolError(f"{input_path}: the imported mesh has no vertices")
    if len(mesh.polygons) == 0:
        raise UVToolError(f"{input_path}: the imported mesh has no faces ({len(mesh.vertices)} vertices, "
                          f"{len(mesh.edges)} edges); a point cloud or wireframe cannot be UV-unwrapped")


def ensure_uv_map(bpy, obj, angle_limit=89.0, island_margin=0.02):
    """Smart-UV-project `obj` unless it already carries a UV map. Returns True if unwrapped."""
    if obj.data.uv_layers.active is not None:
        log(f"UV map '{obj.data.uv_layers.active.name}' already exists, skipping unwrapping")
        return False
    log("No UV map found, running Smart UV Project ...")
    _select_only(bpy, [obj])
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        # angle_limit is a radian property since Blender 2.91 (degrees before).
        angle = math.radians(angle_limit) if bpy.app.version >= (2, 91, 0) else angle_limit
        result = bpy.ops.uv.smart_project(angle_limit=angle, island_margin=island_margin)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    if "FINISHED" not in result or obj.data.uv_layers.active is None:
        raise UVToolError(f"Smart UV Project did not produce a UV map (operator returned {set(result)})")
    return True


def _add_triangulate_modifier(obj):
    # Applied by the exporter (apply_modifiers), so the scene mesh stays untouched.
    obj.modifiers.new(name="__generate_uv_triangulate__", type="TRIANGULATE")


def export_ply(bpy, obj, path, export_axes=IDENTITY_AXES, triangulate=True, ascii_format=False):
    """Export the selected `obj` (modifiers applied) as a PLY with normals and UVs.

    Mitsuba's PLY loader -- what render.py uses -- accepts triangle meshes only,
    so faces are triangulated unless the caller asked to keep the polygons.
    """
    _select_only(bpy, [obj])
    forward, up = export_axes
    if _operator_exists(bpy, "wm", "ply_export"):              # Blender >= 3.6 (C++ exporter)
        options = dict(filepath=path, check_existing=False, export_selected_objects=True, apply_modifiers=True,
                       export_uv=True, export_normals=True, ascii_format=ascii_format,
                       forward_axis=forward, up_axis=up)
        if triangulate:
            if _operator_has_property(bpy, "wm", "ply_export", "export_triangulated_mesh"):
                options["export_triangulated_mesh"] = True
            else:
                _add_triangulate_modifier(obj)
        result = bpy.ops.wm.ply_export(**options)
    elif _operator_exists(bpy, "export_mesh", "ply"):          # legacy python add-on (writes polygons as-is)
        if triangulate:
            _add_triangulate_modifier(obj)
        result = bpy.ops.export_mesh.ply(
            filepath=path, check_existing=False, use_selection=True, use_mesh_modifiers=True,
            use_normals=True, use_uv_coords=True, use_ascii=ascii_format,
            axis_forward=_LEGACY_AXIS_NAME[forward], axis_up=_LEGACY_AXIS_NAME[up])
    else:
        raise UVToolError("this Blender has no PLY exporter (wm.ply_export / export_mesh.ply)")
    if "FINISHED" not in result:
        raise UVToolError(f"PLY export failed (operator returned {set(result)})")


# ---------------------------------------------------------------------- pipeline

def generate_uv(input_path, output=None, force=False, angle_limit=89.0, island_margin=0.02,
                triangulate=True, ascii_format=False):
    """Import -> (Smart UV if needed) -> export PLY -> validate -> move into place.

    Returns the absolute output path; raises UVToolError on any failure. The
    input file is only read. With triangulate=False the source polygons are
    kept, which Mitsuba's PLY loader (render.py) does not accept.
    """
    input_path, output_path = resolve_paths(input_path, output, force)
    bpy = _require_bpy()
    log(f"Blender {bpy.app.version_string} (Python {sys.version.split()[0]})")
    log(f"input:  {input_path}")
    # Start from an empty scene whatever the startup file holds (the default
    # cube would otherwise be a candidate for export).
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meshes, export_axes = import_mesh(bpy, input_path)
    obj = merge_objects(bpy, meshes)
    check_mesh_has_faces(obj, input_path)
    ensure_uv_map(bpy, obj, angle_limit=angle_limit, island_margin=island_margin)
    if not triangulate:
        log("WARNING: --keep-polygons: quads/n-gons are written as-is; Mitsuba's PLY loader (render.py) "
            "needs triangles, so this file is for other consumers only")

    tmp_path = _temporary_output_path(output_path)
    try:
        export_ply(bpy, obj, tmp_path, export_axes=export_axes, triangulate=triangulate,
                   ascii_format=ascii_format)
        summary = validate_ply(tmp_path, require_triangles=triangulate)
        # Re-check just before the move: the export took time and the checks are cheap.
        if _same_file(input_path, output_path):
            raise UVToolError(f"output resolves to the input file, refusing to overwrite it: {output_path}")
        if os.path.exists(output_path) and not force:
            raise UVToolError(f"output already exists: {output_path} (pass --force to replace it)")
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    faces = f"{summary['faces']} triangles" if summary["triangles"] else f"{summary['faces']} faces (polygons kept)"
    log(f"output: {output_path} ({summary['format']}, {summary['vertices']} vertices, {faces}, "
        f"UV properties {'/'.join(summary['uv_properties'])})")
    return output_path


def main(argv=None):
    """CLI entry; returns the process exit code (0 = success)."""
    parser = build_parser()
    try:
        args = parser.parse_args(split_cli_args(sys.argv if argv is None else argv))
        if not 0.0 < args.angle_limit <= 90.0:
            parser.error("--angle-limit must be in (0, 90] degrees")
        if args.island_margin < 0.0:
            parser.error("--island-margin must be >= 0")
        if args.triangulate and args.keep_polygons:
            parser.error("--triangulate and --keep-polygons are mutually exclusive")
        generate_uv(args.input_path, output=args.output, force=args.force,
                    angle_limit=args.angle_limit, island_margin=args.island_margin,
                    triangulate=not args.keep_polygons, ascii_format=args.ascii)
    except UVToolError as exc:
        print(f"{LOG_PREFIX} ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    except SystemExit:
        raise
    except Exception as exc:  # Blender would exit 0 on an uncaught exception
        traceback.print_exc()
        print(f"{LOG_PREFIX} ERROR: unexpected failure: {exc!r}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
