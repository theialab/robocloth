# Neural-BRDF rendering

Path-trace any Mitsuba 3 scene with our Stage-2 neural BRDF checkpoints
attached to selected shapes.

## Scene folder

A renderable scene is a folder containing:

```
my_scene/
├── scene.xml        # Mitsuba 3 scene; shapes carry ids
├── materials.json   # shape id -> checkpoint assignment
└── meshes/ ...      # anything scene.xml references relatively (optional)
```

`materials.json`:

Use an absolute `checkpoint_root`. The renderer may be launched after changing
directories, while an absolute root always resolves to the intended release
bundle.

```json
{
  "checkpoint_root": "/path/to/checkpoints/stage2/RoboCloth",
  "two_sided": false,
  "assignments": {
    "shape_id_in_xml": {"material": "145"},
    "another_shape":   {"material": "370", "uv_tiling": 12.0},
    "custom":          {"ckpt": "/absolute/path/to/some.ckpt"}
  }
}
```

- `"material": "<id>"` resolves to `<checkpoint_root>/<id>/Ours_epoch*.ckpt`
  (the glob must match exactly one file).  `"ckpt"` takes a path instead —
  absolute, or relative to `checkpoint_root`.
- Shapes not listed keep the BSDF from `scene.xml`.
- Optional per-assignment keys: `material_type` (defaults to
  `AnisotropicLatentTexturedModel`, the wrapper for our released Stage-2
  checkpoints; UBO/Bonn/PBR wrapper classes are also available), `uv_tiling`
  (texture repeats across the mesh UV, default 5.0), `two_sided`, `uv_inset`,
  `grazing_mask_deg`.
- Optional top-level keys: `two_sided` (default from `configs/render.yaml`:
  true — back-face hits render fabric instead of black), `uv_inset`,
  `grazing_mask_deg`, and `radiance` (`{shape_id: intensity}` area-emitter
  override).

Environment variables: `${VAR:-default}` and `$VAR` are expanded inside
`checkpoint_root` / `ckpt` values and inside `<default value="...">` elements
of `scene.xml` (the bundled examples use `BRDF_CKPT_ROOT`,
`TEASER_SCENE_ROOT` and `MESH_DIR` to relocate the checkpoint / asset
downloads).

## Rendering

```bash
conda activate robocloth-render
BRDF_CKPT_ROOT=/absolute/path/to/ckpts/checkpoints/stage2/RoboCloth \
MESH_DIR=/absolute/path/to/assets/render_assets/cloth_on_bar \
    python render.py scene=examples/cloth_on_bar render.spp=64 \
    output_base=/absolute/path/to/renders output_name=cloth_on_bar
```

Quality/output knobs live in `configs/render.yaml` (`render.spp`,
`render.batch_spp`, `render.width/height/fov/max_depth` — null keeps the
scene XML's own values — `tonemap`, `variant`, `output_base`,
`output_name`).  Output is `<output_base>/<output_name>.png` plus a linear
`.exr`.

## Bundled examples

- `examples/cloth_on_bar` — cloth draped over a metal bar, HDRI lighting
  (meshes via `$MESH_DIR`).
- `examples/teaser` — the paper-teaser room: 17 shapes (two sofas, eight
  pillows, the carpet, six curtain panels) mapped to 13 Stage-2 checkpoints,
  materials 8, 9, 26, 50, 145, 190, 226, 311, 314, 367, 370, 452 and 453.
  Everything it needs is published: the checkpoints under
  `checkpoints/stage2/RoboCloth/<id>/Ours_epoch*.ckpt` (via
  `$BRDF_CKPT_ROOT`) and the 37-file room download
  `render_assets/teaser_room/` (via `$TEASER_SCENE_ROOT`). Its
  `materials.json` deliberately sets two legacy flags the room was authored
  against — `"two_sided": false` (back-face hits render black instead of
  mirrored fabric, the one-sided path of the original teaser render) and
  `"uv_inset": true` (after tiling, `uv = (uv*0.8 + 0.1) mod 1`, so each tile
  samples only the inner 80 % of the latent texture and never its border).
  Leave both at their defaults in new scenes. Commands, download sizes and
  runtimes: [examples/teaser/README.md](examples/teaser/README.md).

## Meshes without UVs

The material is a latent texture, so every shape it is assigned to needs a UV
map. `tools/generate_uv.py` Smart-UV-unwraps an OBJ / PLY / GLB / glTF mesh in
headless Blender and writes `<input_stem>_uv.ply` (positions, normals, UVs)
next to the input, in the input's coordinate frame. Faces are triangulated on
export because Mitsuba's PLY loader — what `render.py` uses — rejects quads
and n-gons. The input file is never modified, the output must be a distinct
`.ply`, and an existing output is only replaced with `--force`. Reference the
`_uv.ply` from `scene.xml` and set `uv_tiling` in `materials.json` to control
the repeat density.

```bash
# any Blender >= 3.6 (4.0.2 verified); the tool's arguments follow '--'
blender --background --python tools/generate_uv.py -- /absolute/path/to/mesh.obj

# or the bpy wheel in its own Python 3.11 env (pip install -r ../envs/uv.txt)
python tools/generate_uv.py /absolute/path/to/mesh.obj
```

Options: `--output <file>.ply`, `--force`, `--angle-limit 89`,
`--island-margin 0.02`, `--ascii`, `--keep-polygons` (source quads/n-gons
instead of triangles — not loadable by `render.py`). Environment notes,
guarantees and tests: [docs/uv_tool.md](../docs/uv_tool.md).
