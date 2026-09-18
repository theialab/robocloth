# RoboCloth project page

Static HTML/CSS/JS site (served by GitHub Pages from this `docs/` folder at
<https://theialab.github.io/robocloth/>) that previews the
[koalapenguin/RoboCloth](https://huggingface.co/datasets/koalapenguin/RoboCloth)
dataset directly from HuggingFace — no server-side code, no rebuilt assets.

## Quick start

```bash
cd docs
python3 -m http.server 8765
# open http://localhost:8765/
```

The first page load fetches `data/manifest.json` (shipped in this folder)
to populate the preview grid. Clicking any card opens
`obj.html?id=N&s=1` which streams `scan_log.json` + `rotated_camera.json`
straight from HuggingFace and pulls individual HDR PNGs out of the
remote `hdr.tar` via HTTP Range requests.

## Folder layout

```
docs/
├── index.html              Landing page (hero + dataset preview grid)
├── obj.html                Per-material viewer (sliders + 3D primitives)
├── css/style.css           Custom styles (Bootstrap loaded via CDN)
├── js/
│   ├── config.js           HF URLs + calibration constants
│   ├── transforms.js       Port of utils/transform.py + utils/io.py
│   ├── tar_reader.js       HTTP-Range tar walker (no full download)
│   ├── hdr_display.js      Tone-mapping for 16-bit HDR PNGs
│   ├── viz.js              Three.js scenes (turntable, arc, hemisphere ball)
│   ├── main.js             Landing page logic
│   └── obj.js              Material viewer logic
├── data/
│   ├── manifest.json       Preview-material list (20 by default)
│   └── tar_index/{id}.json (optional) precomputed offset table per hdr.tar
└── tools/
    └── build_manifest.py   Rebuild data/manifest.json + tar indexes
```

## Updating the preview list

```bash
# Default: 20 materials (5 from sample/, 15 from full materials/)
python3 docs/tools/build_manifest.py --no-index

# Custom set
python3 docs/tools/build_manifest.py --ids 156 249 309 2 50 350

# Include tar indexes (slow — issues ~500 range requests per material to
# scan ustar headers. Indexes make the obj.html page snappier because we
# don't have to walk the tar in the browser).
python3 docs/tools/build_manifest.py --ids 156 249 309 428 483
```

Tar indexes are not required — if `data/tar_index/{id}.json` is missing,
`tar_reader.js` falls back to walking the headers in-browser via Range
requests. That works but is slower on first open of each material. To keep
this Pages branch small we ship indexes **only for the 25 preview materials**
in `data/manifest.json`; every other material uses the in-browser fallback.
Regenerate more with `build_manifest.py --ids …` if you want them.

## Large media

Videos are **not** committed here (GitHub rejects >100 MB and Pages does not
serve Git LFS). Both page videos stream from Hugging Face
[`koalapenguin/RoboCloth-assets`](https://huggingface.co/datasets/koalapenguin/RoboCloth-assets):
`media/capture_314_5x.mp4` (capture system) and
`media/web/cloth_with_sphere_final.mp4` (Stage-2 relighting demo).

## How it works

### Coordinate transforms (`js/transforms.js`)

The dataset stores raw gripper-base poses for the camera in
`scan_log.json` (millimetres, robot base frame). The training pipeline
applies a chain of transforms to recover the camera and light positions
in a turntable-aligned "zero-angle world" frame. This module ports the
same chain — `cameraC2W0FromScan` and `lightL2W0FromScan` mirror the
Python functions in `utils/io.py` and `recon/calibration/shape_matching.py`.

| Step | Python source | JS port |
|------|---------------|---------|
| `rodrigues_axis_angle` | `utils/transform.py:42` | `rodriguesAxisAngle` |
| `build_rot_about_point` | `utils/transform.py:59` | `buildRotAboutPoint` |
| `rotated_c2w` | `recon/calibration/shape_matching.py:128` | `cameraC2W0FromScan` |
| `read_light_transforms` | `utils/io.py:164` | `lightL2W0FromScan` |

### Tar HTTP-Range extraction (`js/tar_reader.js`)

Each material's `hdr.tar` is 5 GB (full) or 85 MB (sample subset). We
avoid downloading either by:

1. Reading the 512-byte ustar header for each entry to record
   `{filename → [offset, size]}`.
2. Issuing one HTTP Range request per requested PNG.

The HuggingFace CDN supports `Accept-Ranges: bytes` and emits a CORS
header echoing the request origin, so this works straight from a static
page.

### 3D scenes (`js/viz.js`)

Three primitives, all rendered with three.js:

| Primitive | Element | Encoding |
|-----------|---------|----------|
| Turntable | grey disc + arrow | scale matches `bbox.json[bbox_size]` |
| Camera trajectory | blue points + faint arc | union of all `cam_pos_w0` per camera_id |
| Light positions | orange points | union of all `light_pos_w0` per light_id |
| Selected camera | dark blue dot + dashed ray to centre | the slider's chosen frame |
| Selected light | bright orange dot + dashed ray | same |
| Hemisphere ball | unit sphere with both scatters | direction = `(pos − turntable_centre).normalize()` |

## Notes / caveats

- **HDR display**: PNG files are 16-bit linear, but `<img>` decodes them
  to 8-bit before drawing to canvas. Tone-mapping is applied in JS
  (`hdr_display.js`) — exposure + gamma. Use `+` / `−` buttons to bias.
- **Slider interaction**: the four sliders are coupled — moving any one
  picks the nearest scan_log entry matching that axis. Not every
  `(camera_id, light_id, turn_angle)` triplet exists (the capture is
  ~588 configurations, not the full Cartesian product), so secondary
  sliders may move on their own to stay consistent.
- **Sample folder vs. full**: materials `{156, 249, 309, 428, 483}` are
  in `sample/material_<id>/` with only ~12 HDR PNGs each. Other
  materials use `materials/<id>/` with full 500+ captures. The manifest
  records this per-entry.
