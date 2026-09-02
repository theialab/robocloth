# Teaser scene

The living room from the paper teaser. Two sofas, eight pillows, the carpet
and six curtain panels are shaded with 13 of our Stage-2 neural-BRDF
checkpoints; walls, floor, dresser and lamps keep the ordinary Mitsuba
materials declared in `scene.xml`. `scene.xml` is a thin copy of the room
scene (camera, lights, materials, shape list): the meshes and textures it
references are not in this repository but in the released
`render_assets/teaser_room/` download, located through `$TEASER_SCENE_ROOT`;
the checkpoints are located through `$BRDF_CKPT_ROOT` (`materials.json`).

## Shape -> material

All 17 assignments use `{"material": "<id>"}` and resolve to
`$BRDF_CKPT_ROOT/<id>/Ours_epoch*.ckpt`.

| shape id | mesh (`meshes/`) | material | `uv_tiling` |
|---|---|---|---|
| `elm__52` | `Cloud_Sofa-Fabric.ply` | 370 | 3 |
| `elm__54` | `Cloud_Sofa001-Fabric.001.ply` | 453 | 12 |
| `elm__43` | `carpet-carpet.ply` | 190 | 15 |
| `elm__2` | `polstar_3.ply` | 8 | 3 |
| `elm__3` | `Pillow_008.ply` | 9 | 3 |
| `elm__5` | `Pillow_007.ply` | 452 | 3 |
| `elm__55` | `Pillow.ply` | 226 | 3 |
| `elm__56` | `Round_Pillow.ply` | 367 | 3 |
| `elm__57` | `Pillow_002.ply` | 145 | 3 |
| `elm__58` | `LovePillow_002.ply` | 311 | 3 |
| `elm__59` | `Pillow_001.ply` | 314 | 3 |
| `elm__29` `elm__301` `elm__303` `elm__304` | `curtains_001.ply` `curtains_uv2.ply` `curtains_003-curtain.ply` `curtains_003-curtain.green.ply` | 50 | 16 |
| `elm__302` `elm__30` | `curtains_002.ply` `curtains_004.ply` | 26 | 12 |

Released checkpoint files (one per material folder, 1.16 GB each):
`8/Ours_epoch92`, `9/Ours_epoch92`, `26/Ours_epoch92`, `50/Ours_epoch92`,
`145/Ours_epoch112`, `190/Ours_epoch80`, `226/Ours_epoch100`,
`311/Ours_epoch94`, `314/Ours_epoch80`, `367/Ours_epoch96`,
`370/Ours_epoch80`, `452/Ours_epoch80`, `453/Ours_epoch100` (`.ckpt`).

## Assets

Two absolute directories, laid out exactly as released:

```
$ROBOCLOTH_CKPTS/checkpoints/stage2/RoboCloth/<id>/Ours_epoch<N>.ckpt      # the 13 ids above, 15.1 GB
$ROBOCLOTH_ASSETS/render_assets/teaser_room/{room.xml,meshes/,textures/}    # 37 files, 122 MB
```

```bash
export ROBOCLOTH_REPO=/absolute/path/to/RoboCloth
export ROBOCLOTH_CKPTS=/absolute/path/to/robocloth-checkpoints
export ROBOCLOTH_ASSETS=/absolute/path/to/robocloth-assets
```

<!-- PUBLIC-ONLY:BEGIN -->
```bash
# the 13 teaser checkpoints only (Ours_epoch*.ckpt, 15.1 GB) ...
hf download koalapenguin/RoboCloth-assets --repo-type dataset \
    --include "checkpoints/stage2/RoboCloth/*/Ours_epoch*.ckpt" --local-dir "$ROBOCLOTH_CKPTS"
# ... and the room scene, meshes and texture (37 files, 122 MB)
hf download koalapenguin/RoboCloth-assets --repo-type dataset \
    --include "render_assets/teaser_room/*" --local-dir "$ROBOCLOTH_ASSETS"
```
<!-- PUBLIC-ONLY:END -->

The broader `--include "checkpoints/stage2/RoboCloth/*"` download from the
top-level README (28 files, 29 GB) works too: `"material"` lookups glob only
`Ours_epoch*.ckpt`, so the `Bonn_`/`MERL_`/`PBR_` variants of the five paper
materials are ignored. `TEASER_SCENE_ROOT` must be the folder that contains
`meshes/` and `textures/`.

## Render

```bash
conda activate robocloth-render
cd "$ROBOCLOTH_REPO/rendering"
TEASER_SCENE_ROOT="$ROBOCLOTH_ASSETS/render_assets/teaser_room" \
BRDF_CKPT_ROOT="$ROBOCLOTH_CKPTS/checkpoints/stage2/RoboCloth" \
    python render.py scene=examples/teaser \
    output_base=/absolute/path/to/render-output output_name=teaser \
    render.spp=16 render.width=960 render.height=540
```

The last line is the draft setting; drop it for final quality and pass
`render.spp=1024` instead. `render.py` takes the sample count from
`render.spp` (default 16 in `configs/render.yaml`), not from the `spp`
default declared in `scene.xml` (1024), so it has to be given explicitly;
resolution (1920x1080), FOV (60 deg, x axis) and path depth (8) come from
`scene.xml` unless overridden with `render.width/height/fov/max_depth`.
Output: `<output_base>/teaser.png` (linear radiance, sRGB, clipped) and
`teaser.exr` (linear, unclipped).

Runtime and memory on one 48 GB GPU (the 13 resident checkpoints alone take
9.4 GB: 2048x2048, 46-channel latent textures):

| setting | flags | render time | GPU memory |
|---|---|---|---|
| draft (measured) | `render.spp=16 render.width=960 render.height=540` | 114 s, 141 s wall incl. loading the 13 checkpoints | 11.5 GB |
| 1920x1080, 8 spp (measured) | `render.spp=8 render.batch_spp=2` | 136 s, 181 s wall | 14.4 GB |
| 1920x1080, 1024 spp (final) | `render.spp=1024 render.batch_spp=2` | ~4.8 h — 128x the 8-spp run; every 2-spp batch is identical work, so time is linear in `render.spp` (not run here) | 14.4 GB |

Memory grows with `render.batch_spp` x resolution, not with `render.spp`:
the default `render.batch_spp=4` needs ~19 GB at 1920x1080 (it ran out of
memory here on a GPU shared with other jobs), `2` needs 14.4 GB. On a
smaller GPU lower `render.batch_spp` and keep `render.spp`.

Expected console output: `[scene_loader] default scene_root = <your
TEASER_SCENE_ROOT>`, 17 `Shape -> checkpoint` lines each ending in
`.../<id>/Ours_epoch<N>.ckpt`, `render time: ...s`, `Saved .../teaser.png`
and finite `raw min/max/mean` (draft: `min=0.0000 max=7.1991 mean=0.4435`;
1920x1080 at 8 spp: `mean=0.4436` — the mean is a quick sanity check, it
barely depends on resolution or spp).
The image is the paper teaser: a beige and a red sofa with cushions, four
floor pillows on a carpet, patterned curtains left and right, a neon ring
lamp on the back wall. At 16 spp it is visibly grainy; that is Monte-Carlo
noise, not a material problem.

## `two_sided: false`, `uv_inset: true`

`materials.json` sets both scene-wide because the room was authored against
the legacy plugin behaviour and is kept that way for fidelity to the paper
figure:

- `two_sided: false` — back-face hits render black instead of mirrored
  fabric (the one-sided path of the original teaser render). The default,
  `true` (from `configs/render.yaml`), is what new scenes should use: an open
  drape then shows fabric on its inside as well.
- `uv_inset: true` — after tiling, `uv = (uv*0.8 + 0.1) mod 1`, so each tile
  samples only the inner 80 % of the latent texture and never its border.
  Default `false`; not needed for new scenes.

Both can also be set per assignment; see the `materials.json` reference in
[../../README.md](../../README.md).

## If it fails

Every lookup fails closed with the offending shape id in the message:

- `checkpoint_root is not a directory` — `BRDF_CKPT_ROOT` unset or wrong; it
  must be the directory that contains the numbered material folders.
- `no folder for material '<id>'` / `no checkpoint matches` — incomplete
  checkpoint download; the message lists what the root contains.
- `ambiguous checkpoint glob` — more than one `Ours_epoch*.ckpt` in a folder.
- Mitsuba cannot open a `.ply` or `.jpg` under `$scene_root` —
  `TEASER_SCENE_ROOT` must point at the `render_assets/teaser_room` folder
  itself (the one containing `meshes/` and `textures/`), not at a parent or a
  subfolder of it.
