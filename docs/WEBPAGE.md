# RoboCloth project page

Static HTML/CSS/JS site (served by GitHub Pages from this `docs/` folder at
<https://theialab.github.io/robocloth/>) that previews the
[koalapenguin/RoboCloth](https://huggingface.co/datasets/koalapenguin/RoboCloth)
dataset directly from Hugging Face — no server-side code, no rebuilt assets.

## Quick start

```bash
python3 docs/tools/preview_server.py 8765 docs
# open http://localhost:8765/
```

(`python3 -m http.server` works for a quick look but has no HTTP Range support, so
`<video>` seeking and the material viewer's range reads misbehave on it.)

The first page load fetches `data/manifest.json` (shipped in this folder)
to populate the preview grid, whose thumbnails are pre-rendered JPGs in
`images/thumbs/`. Clicking any card opens
`obj.html?id=N&s=1` which streams `scan_log.json` + `rotated_camera.json`
straight from Hugging Face and pulls individual HDR PNGs out of the
remote `hdr.tar` via HTTP Range requests.

## Folder layout

```
docs/
├── index.html              Landing page (hero + dataset preview grid)
├── obj.html                Per-material viewer (sliders + 3D primitives)
├── css/
│   ├── style.css           Page styles (Bootstrap loaded via CDN)
│   ├── title_anim.css      Animated fabric title (hero)
│   ├── sectnav.css         Right-edge section navigator + back-to-top
│   ├── room.css            "Apply a trained material" panel
│   ├── handoff.css         Method video -> room panel hand-off
│   ├── results.css         Results tables, chart and comparison grid
│   ├── viz3d.css           3-D capture viewer on obj.html
│   └── hero_curtain.css    Optional dark backdrop behind the title (off; data-curtain)
├── js/
│   ├── config.js           HF URLs + calibration constants
│   ├── transforms.js       Port of utils/transform.py + utils/io.py
│   ├── tar_reader.js       HTTP-Range tar walker (no full download)
│   ├── hdr_display.js      Tone-mapping for 16-bit HDR PNGs
│   ├── hdr_worker.js       Off-main-thread decode + tone map (used by hdr_display.js)
│   ├── viz.js              Three.js scenes (turntable, arc, hemisphere ball)
│   ├── main.js             Landing page logic (preview grid)
│   ├── media.js            Landing page video playback (sound button, play-in-view)
│   ├── title_anim.js       Hero title: assembly + light sweep, pointer scrubs the lamp
│   ├── sectnav.js          Section navigator (labels read from each section's <h3>)
│   ├── room_core.js        Room panel compositor (WebGL); room_a.js = the interaction
│   ├── handoff.js          Flies the six room materials out of the method video's last frame
│   ├── swatch_tex.js       Lit swatch texture for the 3-D viewer
│   └── obj.js              Material viewer logic
├── media/                  Web-encoded videos + posters (tracked, each < 10 MB)
│   ├── title/              Hero title assets (lit/assembled WebP, intro.webm, 49 sweep frames)
│   ├── room/               Room panel layers (WebP) + meta.json
│   └── robocloth_method_tiles.json   tile rectangles of the method video's last frame
├── images/                 Method figures, og_card.jpg, favicons,
│   ├── thumbs/{id}.jpg     pre-rendered preview thumbnails (tools/export_thumbs.py)
│   ├── results/            PSNR chart (SVG/PNG) + qualitative comparison tiles
│   └── viz/                Camera / LED cut-outs for the 3-D viewer
├── data/
│   ├── manifest.json       Preview-material list (25: 20 train + 5 test)
│   └── tar_index/{id}.json precomputed offset table per hdr.tar (all 500 shipped)
└── tools/
    ├── preview_server.py   Local preview with HTTP Range support
    ├── build_manifest.py   Rebuild data/manifest.json + tar indexes
    ├── build_tar_indexes_local.py  Tar indexes from the local Dataset_submission tree
    ├── export_thumbs.py    Re-render images/thumbs/{id}.jpg (headless Chrome)
    ├── results/            build_results.py: tex tables -> HTML/chart (+ --check)
    └── room/               Room panel render passes + asset build (see below)
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

Tar indexes are shipped for **all 500 materials** (`data/tar_index/{id}.json`,
~33 KB each, 19 MB in total), so `obj.html` never has to walk a remote tar. If
an index is missing, `tar_reader.js` falls back to walking the headers
in-browser via Range requests (one CDN round trip per capture, ~580 in
sequence: the first image shows after a few seconds, the full index takes
about 3 min).

The cheap way to make more is `tools/build_tar_indexes_local.py`: the
published tars were written from the sorted files of
`Dataset_submission/<id>/hdr/`, so every offset follows from the local file
sizes (reproduces the 25 shipped indexes exactly; `--verify` checks each
predicted archive size against Hugging Face in one batched API call per 100
materials). All 500 take about a minute and ~33 KB each:

```bash
python3 docs/tools/build_tar_indexes_local.py --all --verify --write
```

If you change the preview list, also run `tools/export_thumbs.py` (needs
Playwright + Chrome) to pre-render the new thumbnails; a card whose JPG is
missing falls back to range-fetching and tone-mapping its HDR PNG in the
browser (slow: ~10 MB per card).

## Media

All page videos are small web encodes committed under `media/` (H.264 High,
yuv420p, `+faststart`; every file < 10 MB). Only the launch video loads up
front; the others use `preload="metadata"` and play while in view
(`js/media.js`). The full-resolution originals stay on Hugging Face
[`koalapenguin/RoboCloth-assets`](https://huggingface.co/datasets/koalapenguin/RoboCloth-assets)
and are linked under each video.

| file | source | encode |
|------|--------|--------|
| `media/robocloth_teaser.mp4` (+ `_poster.jpg`, last frame) | launch video (`robocloth_teaser_v9.mp4`, 1920x1080, 19.8 s) | 1920x1080, 2-pass 2950k, AAC 128k |
| `media/robocloth_teaser_720.mp4` | same | 1280x720, crf 27 capped 1700k, AAC 96k — chosen by an inline script for viewports ≤ 700 px or data-saver |
| `media/capture_314_web.mp4` | HF `media/capture_314_5x.mp4` (3034x1080, 60 s) | 1920x684, 2-pass 1300k, no audio |
| `media/cloth_with_sphere_web.mp4` | HF `media/web/cloth_with_sphere_final.mp4` (2048x2048, 4:4:4) | 1080x1080, 2-pass 4000k |
| `media/robocloth_method.mp4` (+ `_poster.jpg`) | `thread/method/light_v3/method_light_v3.mp4` via `method_video.py --web-cut` (ends on the flat 13-tile grid, 16.0 s) | 1920x1080, crf 24, no audio; tile rects in `robocloth_method_tiles.json` |

To replace the launch video with the final cut (same file name, keep it < 8 MB):

```bash
SRC=robocloth_teaser_final.mp4
V="-vf scale=1920:1080:flags=lanczos -c:v libx264 -preset veryslow -profile:v high -level:v 4.1 -pix_fmt yuv420p -b:v 2950k -maxrate 6000k -bufsize 6000k -g 60"
ffmpeg -y -i $SRC $V -pass 1 -an -f mp4 /dev/null
ffmpeg -y -i $SRC $V -pass 2 -c:a aac -b:a 128k -ac 2 -movflags +faststart docs/media/robocloth_teaser.mp4
ffmpeg -y -sseof -0.05 -i $SRC -update 1 -frames:v 1 -q:v 3 docs/media/robocloth_teaser_poster.jpg
```

`images/og_card.jpg` (1200x630 social card) and the favicons are cut from the
same end-title frame. The `og:`/`twitter:` tags in `index.html` point at the
absolute URL `https://theialab.github.io/robocloth/images/og_card.jpg`.

## Hero title, section navigator, hand-off, results

- **Title** (`js/title_anim.js`, `media/title/`): the word "RoboCloth" assembled from real fabric
  patches, rendered for the light header as thin fabric letters floating in front of a white
  backdrop (the darkness is their real cast shadow). Plays once (assembly, light sweep, evenly lit),
  then the pointer's x position scrubs the 49 real sweep renders; phones loop and drag-scrub;
  `prefers-reduced-motion` shows the lit still. Built by `~/projects/robocloth-teaser/tools/elements/title/web_title.py build-light`
  (renders: `title_light.py`). `data-curtain` on `#hero` (`css/hero_curtain.css`) can put a dark
  band behind it; it is off.
- **Section navigator** (`js/sectnav.js`): right-edge dots with the active section's label (labels
  from each section's `<h3>`), hidden below 1200 px; back-to-top button on all widths.
- **Hand-off** (`js/handoff.js`): when the method video ends on screen with the room tile row in
  view, six clones of the room materials fly out of the video's last frame into the tile row
  (rectangles from `media/robocloth_method_tiles.json`); runs once, never under reduced motion.
- **Results** (`css/results.css`, `images/results/`): numbers come from the paper's tex tables via
  `tools/results/build_results.py --tex-dir <paper>/tables --check` (writes the chart SVG/PNG and
  fills the marked HTML blocks).

## "Apply a trained material" room panel

Below the method video. Ten trained materials (tiles) and eleven pieces of furniture (each
sofa body, each of the four throw pillows, the four floor cushions, the rug); every piece takes
its own material, and a drop on a sofa body also dresses its pillows while they are still
grey. `js/room_a.js` is the interaction (drag a tile: it decodes into the material in your
hand; phones: tap a tile, tap a piece, or drag). While the panel is in view and untouched, a
teaching loop plays: a ghost pointer dresses 3-4 pieces one after another (pick up, decode in
hand, carry, highlight, drop, wipe, hold), the room cross-fades back to grey, a hint pulses,
and the next cycle dresses it differently; the visitor's first press, touch, key or mouse
movement in the panel stops it for good (`prefers-reduced-motion`: the hint only). If the
method video's hand-off (`js/handoff.js`) runs, the loop waits for its tiles to land.

`js/room_core.js` composites the pre-rendered room in WebGL (fixed camera):

    page = base + sum_r cov_r * (C_r - G_r)                 per piece r: its colour in its current
                                                             material / in grey, antialiased edges
           * prod_r [1 + (ratio_r - 1) * (1 - cov_r)]       first-order crosstalk: how piece r in
                                                             its material changes the light that
                                                             reaches the rest of the room

The composite is cached and only redrawn while a wipe runs. Two resolution sets from one set of
4K renders (`meta.json` -> `sets`): `media/room/4k/` (3840x1800) when devicePixelRatio x panel
width > 2000 (retina desktops), `media/room/` (1920x900, the 4K renders downsampled) otherwise.
Layout, hit testing, halos, positions and crosstalk ratios are shared (1920 px space). Each
material's images (`mat_<id>_<class>.webp`, `ratio_<id>.webp`) load the first time it is needed;
#453 (the loop's first) is preloaded. The latent tile of each material comes from one asset map,
`meta.json` -> `tiles`, so other tile images can be swapped in by editing paths.

Rebuilding (render env `fipt-mitsuba`, compositing env `teaser`, both with `PYTHONNOUSERSITE=1`;
passes go to `/media/raid/cloth/output/teaser_video/room_demo/passes_r5/`, per-pass timings in
`times.json` and `times.d/`):

```bash
T=$PWD/docs/tools/room                   # from the repo root
$T/run_batch.sh 1 "masks:::256"                                    # 11 region masks + positions (4K)
BATCH=0 $T/run_queue2.sh 1 queue.txt & BATCH=0 $T/run_queue2.sh 2 queue.txt   # job list, 2 GPUs:
                                         # neural:<piece>:<mat>:1024 (4K crop), lo:<piece>:<mat>: (480 px)
$T/check_passes.sh                                                 # lists any missing pass
/data/colin/envs/teaser/bin/python $T/build_assets.py              # -> docs/media/room/ (+ 4k/)
/data/colin/envs/teaser/bin/python $T/record.py a all              # scripted recordings
/data/colin/envs/teaser/bin/python $T/measure_page.py              # page vs a full render
```
(`passes.py` must run from `~/projects/SGHyperMaterials`: `mlpbrdf` reads
`config/material/*.yaml` relative to the cwd; `run_batch.sh` does that itself.)
`xtalk.py` measures what per-piece compositing loses against a full render of the dressed room.

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

The Hugging Face CDN supports `Accept-Ranges: bytes` and emits a CORS
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
  ~580 configurations, not the full Cartesian product), so secondary
  sliders may move on their own to stay consistent.
- **Sample folder vs. full**: materials `{156, 249, 309, 428, 483}` are
  in `sample/material_<id>/` with only ~12 HDR PNGs each. Other
  materials use `materials/<id>/` with ≈580 captures. The manifest
  records this per-entry.
