# UV generation for meshes without texture coordinates

Our materials are latent *textures*: every shape a RoboCloth checkpoint is
assigned to needs a UV map, and `render.py` reads it from the mesh file. Meshes
that come without one (most scanned geometry, many downloaded assets) can be
Smart-UV-unwrapped with `rendering/tools/generate_uv.py`, which runs Blender
headlessly and writes a triangulated, UV-mapped PLY next to the input -- a file
Mitsuba's PLY loader, and therefore `render.py`, accepts as is.

## Supported environments

The tool needs Blender's Python API (`bpy`). Two ways to get it:

### A. A Blender installation (preferred)

Any Blender >= 3.6 works; 4.0.2 is what the tests run against. No Python
environment is required. Under `blender` the tool's own arguments must follow
a `--` (everything before it is parsed by Blender itself):

```bash
blender --background --python rendering/tools/generate_uv.py -- /absolute/path/to/mesh.obj
```

### B. The `bpy` wheel in its own environment

PyPI publishes `bpy` wheels only for Python 3.11 (`bpy` 4.2.0 ... 5.0.1) and,
for Blender 5.1+, Python 3.13 -- nothing for Python 3.10 or 3.12 (checked
against PyPI on 2026-09-01). The `bpy==3.6.0` / Python 3.10 combination from
earlier docs is therefore no longer installable. The pinned, verified
combination is `bpy==4.5.13` (Blender 4.5 LTS) on Python 3.11; the wheel pins
`numpy<2`, so keep it out of the training and rendering environments:

```bash
conda create -n robocloth-uv python=3.11 -y && conda activate robocloth-uv
pip install -r envs/uv.txt          # bpy==4.5.13 (+ cython, numpy<2, requests, zstandard)
python rendering/tools/generate_uv.py /absolute/path/to/mesh.obj
```

Verified: `bpy-4.5.13-cp311-cp311-manylinux_2_28_x86_64.whl` (373 MB) with
Python 3.11.15 on Linux x86_64 passes the same test suite as the Blender
binary.

## Usage

```
generate_uv.py INPUT [--output OUT.ply] [--force] [--angle-limit 89]
               [--island-margin 0.02] [--keep-polygons] [--ascii]
```

| Option | Meaning |
|---|---|
| `INPUT` | `.obj`, `.ply`, `.glb` or `.gltf` (extension matched case-insensitively; anything else is rejected). Never modified. |
| `--output` | Output path; must end in `.ply`. Default: `<input_stem>_uv.ply` next to the input. |
| `--force` | Replace an existing output. Without it an existing output is an error. |
| `--angle-limit` | Smart UV Project angle limit in degrees, (0, 90]. Default 89. |
| `--island-margin` | Smart UV Project island margin. Default 0.02. |
| `--keep-polygons` | Write the source quads/n-gons instead of triangles. **Not loadable by `render.py`** (see below); the tool prints a warning. |
| `--ascii` | Write an ASCII PLY (default: binary little-endian). |
| `--triangulate` | Accepted for compatibility with earlier versions; triangles are now the default. |

What happens:

1. The input is imported with the importer matching its extension (OBJ, PLY
   or glTF; the operator names differ between Blender 3.x and 4.x and are
   feature-detected). If the file holds several mesh objects they are joined
   into one. A file Blender cannot read, or one without faces (a point cloud
   or wireframe), is reported and nothing is written.
2. If the mesh already has a UV map it is kept (`UV map '...' already exists,
   skipping unwrapping`); otherwise Smart UV Project runs on all faces.
3. The mesh is exported as PLY with positions, normals and UVs (`s`/`t` vertex
   properties), triangulated on export, under a temporary name. The file is
   then re-read and checked: `ply` magic, UV and normal vertex properties
   present, at least one face, and every face a triangle (the face data is
   actually walked, not just the header). Only then is it moved to the
   output path.
4. Exit code 0 on success; 1 with `[generate_uv] ERROR: ...` on stderr for any
   failure (2 for command-line errors).

Guarantees:

- The input file is never written to, whatever its format. (An earlier version
  always used the PLY importer and, for a `.obj` input, overwrote the input
  with a binary PLY still named `.obj`.)
- The output is always a distinct `.ply`: an `--output` that resolves to the
  input file, or one with another extension, is refused before Blender starts.
- The exported vertex positions stay in the input file's coordinate frame: OBJ
  and PLY are imported and exported without axis conversion, and the Y-up to
  Z-up rotation Blender applies to glTF on import is undone on export.
  Swapping `mesh.obj` for `mesh_uv.ply` in `scene.xml` therefore does not move
  the object.
- The default output is a triangle mesh and loads in Mitsuba. Mitsuba's PLY
  plugin rejects quads and n-gons (`[PLYMesh] ... incompatible contents -- is
  this a triangle mesh?!` for binary files, `trailing tokens after end of PLY
  file!` for ASCII ones), so faces are triangulated on export (Blender's
  `export_triangulated_mesh`, or a Triangulate modifier on Blender versions
  whose exporter lacks that option) and the post-export check proves it.
  `--keep-polygons` opts out for other consumers; do not point `render.py`
  at such a file.

## Using the result

Reference the PLY from `scene.xml` and control the repeat density with
`uv_tiling` in `materials.json` (see [rendering/README.md](../rendering/README.md)):

```xml
<shape type="ply" id="curtain">
    <string name="filename" value="meshes/curtain_uv.ply"/>
</shape>
```

```json
{"assignments": {"curtain": {"material": "145", "uv_tiling": 12.0}}}
```

Smart UV Project packs all islands into the unit square, so `uv_tiling` is the
number of times the material texture repeats across that square; larger
objects want larger values.

## Troubleshooting

- `no '--' separator found`: under `blender` the tool's arguments must come
  after `--`, otherwise Blender tries to open your mesh as a `.blend` file.
- `output already exists`: pass `--force` or choose another `--output`.
- `the imported mesh has no faces`: the input is a point cloud or a wireframe;
  there is nothing to unwrap (or render).
- `failed to import ...`: Blender could not read the file (the importer's own
  message follows, e.g. `Bad glTF: json error ...`).
- `render.py` fails with `is this a triangle mesh?!` or `trailing tokens after
  end of PLY file!`: the PLY holds quads/n-gons -- it was written with
  `--keep-polygons` or by the earlier version of this tool. Re-run without
  `--keep-polygons` (add `--force` to replace it).
- `this Blender has no glTF importer`: the glTF 2.0 add-on is disabled in your
  Blender preferences (it is on by default and built into the `bpy` wheel).
- `Blender's Python API (bpy) is not importable`: the script was started with a
  Python that has no `bpy`; use method A or B above.
- Blender prints its own banner and importer/exporter timings around the
  `[generate_uv]` lines; the exit code and the `[generate_uv]` lines are what
  matter.

## Tests

`tests/uv_tool/` drives the real tool on tiny box meshes (ASCII OBJ, ASCII PLY
without UVs, GLB, glTF) plus a single-hexagon OBJ and checks: input sha256
unchanged, output at the default path, header validation, every face a
triangle, vertex positions preserved, `--force` semantics, `--keep-polygons`,
and the refusals (output equal to the input, non-`.ply` output, unsupported
extension, faceless or unreadable input). When the harness Python can `import
mitsuba`, or `$ROBOCLOTH_MITSUBA_PYTHON` names an interpreter that can (e.g.
the rendering env), every produced PLY is also loaded with Mitsuba's `ply`
plugin and must report UVs, normals and the expected triangle count; otherwise
that assertion is skipped. Fixtures live in a temp dir (`$ROBOCLOTH_TEST_TMP`,
default: the system temp dir), never in the repo, and the tests remove the
bytecode cache their own import creates. The harness Python needs no `bpy`.

```bash
# Blender binary (auto-detected: $BLENDER, /usr/bin/blender, then $PATH)
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/uv_tool -v
# additionally through a Python that has the bpy wheel, with Mitsuba loading every output
ROBOCLOTH_UV_PYTHON=/path/to/envs/robocloth-uv/bin/python \
ROBOCLOTH_MITSUBA_PYTHON=/path/to/envs/robocloth-render/bin/python \
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/uv_tool -v
```
