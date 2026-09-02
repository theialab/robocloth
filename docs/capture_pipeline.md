# Capture pipeline: rig, calibration & reconstruction

How the RoboCloth dataset was produced, written as a walkthrough you can
actually run: the robotic capture session (needs the physical rig — the video
below shows it), the offline rig calibration (`calibration/`, needs the
dedicated calibration sessions) and the per-material reconstruction
(`reconstruction/`, needs a raw session). Every command below is the current
CLI of the script it names; paths are placeholders. All solved constants live
in [`configs/renderer/rig_constants.yaml`](../configs/renderer/rig_constants.yaml),
file schemas in [data_formats.md](data_formats.md).

## What the release contains, and what you can run

| Step | Needs | From the released data |
|---|---|---|
| Capture session | the physical rig | no — the [video](#the-capture-session) shows it |
| Calibration boards | nothing | yes |
| Intrinsics, hand–eye, turntable axis, colour matrix | the calibration sessions (not released) | the CLIs run; the solved values are in `rig_constants.yaml` |
| Reconstruction (COLMAP → alignment → tensors) | a raw session: `ldr/`, `hdr_raw/`, `scan_log.json` | not from a released material (raw capture sessions are not part of the release) |

A released material folder (`bash scripts/download_material.sh <id>`) holds
the *products* of this pipeline — `scan_log.json`, the debayered `hdr/` views
(the per-view crop applied at packaging time is recorded in
`hdr_crop_bboxes.json`), `rotated_camera.json`, `bbox.json`,
`point_positions.npz`, `point_metadata.json`, `observations_structured.npz`,
`unmatched_scan_ids.json` — plus the dataset-level `globals/`. The raw inputs
(`ldr/`, `hdr_raw/`) and the COLMAP `sparse/` model are not part of the
release.

## Environment

```bash
conda create -p /absolute/path/to/envs/robocloth-calib python=3.10 -y
conda activate /absolute/path/to/envs/robocloth-calib
pip install -r envs/calibration.txt
colmap -h | head -1        # "COLMAP 3.8" — reconstruction only; install options in envs/calibration.txt
```

Run every command from the repository root. COLMAP must be a CUDA build:
the wrappers run SIFT extraction and matching on the GPU. A CPU-only build
needs `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0` (and a
display for its Qt front end), which the wrappers do not pass.

## The capture session

[docs/media/capture_314_5x.mp4](media/capture_314_5x.mp4) (7.6 MB, 1280 × 456, 60 s)

<video src="media/capture_314_5x.mp4" controls muted loop playsinline width="100%">
  Your viewer does not render HTML video — open
  <a href="media/capture_314_5x.mp4">docs/media/capture_314_5x.mp4</a>.
</video>

Material 314 being captured: the first five minutes of the session at 5×
speed. **Left**: the rig — the camera arm (left) steps the camera along an
elevation arc around the sample while the light arm (right) holds the LED
still; after each arc the LED moves to a new position and the turntable
rotates the sample. **Right**: the frames as they are captured (LED strobed
on for the exposure; one fixed 16-bit exposure per pose — there is no
exposure bracketing), with the running scan / light counters (`158/590` at
the end). The full-resolution version is on the project page.

<!-- PUBLIC-ONLY:BEGIN -->
Project page (full-resolution capture video and demos):
<https://colinzhenli.github.io/BRDF-Fipt/>
<!-- PUBLIC-ONLY:END -->

### Hardware

* Two 6-axis robot arms (UFACTORY xArm 6) inside a matte-black enclosure: one
  carries the camera, the other the LED. The two robot bases are registered by
  touching a set of common reference points with both end-effectors and
  fitting a rigid transform (`base2_to_base1` in `rig_constants.yaml`).
* Camera: a 3072 × 2048 colour machine-vision camera driven through the
  Spinnaker SDK — raw `BayerRG16` frames, software trigger, fixed exposure
  (20 ms; 8 ms for a subset of materials, see `globals/camera_factor.json`),
  gain 0 dB, auto exposure / gain / white balance and every in-camera gamma,
  LUT, sharpening, saturation and colour transform switched off. Intrinsics in
  `rig_constants.yaml` (SIMPLE_RADIAL, focal 6721 px).
* Light: a single 4000 K COB LED on the light arm's tool flange (radius 7 mm,
  115° FWHM from the spec sheet), switched through the arm's tool digital
  outputs; its mount transform is `R_l2g / t_l2g`.
* Turntable: a networked stepper turntable, ferromagnetic so the cloth is held
  flat by corner magnets. The sample sits in the 250 × 170 mm cut-out of an
  AprilTag ring board (`calibration/boards/`) that surrounds it; rotating the
  table reuses each arm pose at several sample-relative azimuths. Its axis and
  centre are `emitter.turntable` in `rig_constants.yaml`.

### Procedure (one material, ~20 min, ~590 frames)

The operator lays the cloth flat inside the board's cut-out, starts the
capture program and enters the material id. Everything else is automatic:

1. **Set-up.** Both arms move to safe start joints, the LED is warmed up, and
   the random streams are seeded from the material id (light positions and
   the stage-2 pattern are shared within groups of ten ids; turntable angles
   and camera jitter are unique per material), so a session is reproducible.
   The turntable starts at a random angle.
2. **Stage 1 — 50 lights × 10 views = 500 frames.** For each light position
   (sampled on a vertical plane beside the sample, ±0.35 m wide and
   0.06–0.45 m high, always aimed at the sample centre) the camera visits 10
   poses on one half (0–36° or 36–72° from the surface normal) of an
   elevation arc of radius 0.51 m around the sample centre, at azimuth 0° or
   180°, each pose jittered by up to half a step, always looking at the
   centre. After every arc the turntable turns by 29° ± 19°.
3. **Stage 2 — 8 "special-angle" lights, up to 16 views each (~90 frames).**
   The arms move to mid poses, then the LED visits 8 positions 0.3 m from a
   reference point near the sample, at polar angles stepping from grazing
   (~84°) to near-normal (~6°). For each, the camera visits up to 16 poses
   spanning both azimuths and elevations from ~4° up to ~77°; poses within
   30° of the light's elevation on the same side are skipped (occlusion /
   collision avoidance), and the turntable turns 29° ± 19° every three
   frames. This is the stage where the two arms come closest — the operator
   checks the start poses before trying a new random seed.
4. **Per frame.** The arm settles, the flange poses of both arms and the joint
   angles are read back, the LED is switched on, the camera is software-
   triggered 0.3 s later and the LED is switched off again (a 5 s watchdog
   turns it off if anything stalls). A background thread subtracts the
   per-Bayer-plane black level, zeroes every pixel outside a 0.27 × 0.27 m
   rectangle around the sample (projected with the robot pose, the hand–eye
   transform and the turntable angle) and writes `hdr_raw/<name>.png` (the
   masked 16-bit Bayer mosaic) and `ldr/<name>.png` (an 8-bit demosaiced,
   roughly white-balanced copy for COLMAP only). Frames are named
   `scan-<id>_light-<light>_camera-<id>.png`.
5. **End of session.** `scan_log.json` is written (per frame: id, gripper
   `position` (mm) and `rotation_matrix`, `position_light`,
   `rotation_matrix_light`, servo angles, `turn_angle`, `filename` — schema in
   [data_formats.md](data_formats.md)) and the turntable returns to 0°.

The calibration sessions of the next section are recorded by variants of the
same program: a fixed light, a few camera poses each repeated over a sweep of
turntable steps, and — for the hand–eye and axis sessions — the camera's
spherical coordinates encoded in the filename
(`scan-<id>_light-<l>_camera-<c>_phi<φ>_theta<θ>.png`) and as `phi` / `theta`
fields in the log, which is how `hand_eye.py` and `turntable_axis.py` match
frames to poses.

## Offline calibration (once per rig)

Performed in this order. Steps 2–5 need their calibration session; the values
they produced for our rig are the ones in `rig_constants.yaml`.

### 1. Calibration boards (runs without the rig)

```bash
mkdir -p /absolute/path/to/boards && cd /absolute/path/to/boards
python /absolute/path/to/RoboCloth/calibration/boards/generate_AprilTag_board.py        # double ring + inner grid
python /absolute/path/to/RoboCloth/calibration/boards/generate_AprilTag_board_inner.py  # single ring, grid fills the hole
```

Each writes a 300 dpi A4 PNG and PDF into the current directory
(`apriltag_board_double_ring_innergrid_rot180.{png,pdf}`,
`apriltag_board_inner_ring_filled_grid_rot180.{png,pdf}`): AprilTag 36h11
tags, 16 mm, around a 250 × 170 mm cut-out — print at 100 % and cut the hole.
The ChArUco target used for intrinsics is OpenCV's standard board with the
geometry hard-coded in `charuco_calibration.py` (11 × 8 squares, 15 mm
squares, 11 mm markers, `DICT_4X4_50`, legacy pattern):

```bash
python -c "
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
b = cv2.aruco.CharucoBoard((11, 8), 15.0, 11.0, d); b.setLegacyPattern(True)
cv2.imwrite('charuco_11x8_15mm.png', b.generateImage((11 * 300, 8 * 300), marginSize=150))"
```

### 2. Camera intrinsics (+ board-based axis sanity check)

`charuco_calibration.py` runs Zhang's calibration on ChArUco images and then,
optionally, estimates the turntable axis from board poses seen from one or two
fixed gripper poses ("top" and "tilt") while the table steps around. The
shipped intrinsics were cross-checked against COLMAP self-calibration and
stored as a SIMPLE_RADIAL model (focal, principal point, one radial
coefficient).

```bash
python calibration/charuco_calibration.py \
    --calib_glob '/absolute/path/to/charuco_session/ldr/*.png' \
    --top_glob   '/absolute/path/to/axis_top/ldr/*.png'  --scan_log_top  /absolute/path/to/axis_top/scan_log.json \
    --tilt_glob  '/absolute/path/to/axis_tilt/ldr/*.png' --scan_log_tilt /absolute/path/to/axis_tilt/scan_log.json
# optional: --c2g_json /absolute/path/to/hand_eye.json [--c2g_convention opencv|opengl]
```

`--calib_glob` is required (≥ 8 views with valid ChArUco corners); the axis
part needs at least one `--top_glob` / `--tilt_glob` with its `scan_log_*`
(first entry = the fixed gripper pose). Without `--c2g_json` the built-in
hand–eye constants are used; a JSON written by `hand_eye.py --out` is read in
the right camera convention automatically. Prints `K`, the distortion vector,
the reprojection RMS, and the axis direction / point / fit residual.

### 3. Hand–eye (camera → gripper)

Reconstruct the hand–eye session with COLMAP first (`bash
reconstruction/colmap.sh /absolute/path/to/handeye_session <gpu_id>`; the
wrapper also writes `sparse/images.txt`), then:

```bash
python calibration/hand_eye.py \
    --scan_log_path /absolute/path/to/handeye_session/scan_log.json \
    --poses_path    /absolute/path/to/handeye_session/sparse/images.txt \
    --out           /absolute/path/to/hand_eye.json
# options: --tol 1e-3 (φ/θ match tolerance, rad)  --scale_mode auto|joint|umeyama
#          --c2g_json prior.json (skip the solve, only fit world→base)
#          --mesh_path in.ply --mesh_out out.ply (transform a mesh to the base frame)
```

Frames are matched to log entries through the φ/θ in the filename (within
`--tol`; fewer than 3 matches stops the run with a message). COLMAP's
`images.txt` is read as **world→camera** with the scalar-first quaternion
`[qw qx qy qz]` and inverted to camera→world in the OpenCV camera
convention (a NeRF-style `transforms.json`, camera→world in OpenGL
convention, is accepted too). The solve then proceeds in three steps: a
Umeyama fit of the camera centres to the gripper positions *initialises* the
metric scale, Tsai's method (`cv2.calibrateHandEye`) recovers `R_c2g`, and
`(t_c2g, scale)` are solved jointly by linear least squares on `AX = XB`. On a
purely or nearly spherical scan — every pose looking at one point from one
distance, the rig's φ/θ pattern — the scale is not observable from `AX = XB`;
the solver detects the shared fixed point (or a noise-to-signal estimate of
the joint scale above 1e-2), keeps the Umeyama scale and emits a
`RuntimeWarning`. `--scale_mode joint|umeyama` forces either branch; poses at
other focus distances make the scale observable. `--c2g_json` skips the solve
and only fits the COLMAP-world → base similarity with the given transform (a
`camera_convention: opengl` field is converted). `--out` records `R_c2g` in
the **OpenCV** camera convention plus `R_c2g_opengl`, `t_c2g` (log units,
mm), the scale diagnostics (`scale_world_to_base`, `scale_umeyama_init`,
`scale_joint`, `scale_attenuation_est`, `scale_observable`,
`scale_mode_used`) and `T_BW`. `rig_constants.yaml` stores the OpenGL form
(`camera.R_c2g / t_c2g`, metres): `R_c2g_opengl = R_c2g · diag(1, −1, −1)`,
same `t_c2g`. Unit tests for these conventions: `tests/calibration/`.

### 4. Turntable axis

A fixed-light session at several table angles (`turn_angle` in the log,
`theta` in the filenames), reconstructed with COLMAP:

```bash
python calibration/turntable_axis.py \
    --scan_log_path /absolute/path/to/axis_session/scan_log.json \
    --model_path    /absolute/path/to/axis_session/sparse
# --flip_cw if the logged angles are clockwise
```

Alternating refinement — Umeyama alignment ↔ least-squares centre ↔ a small
Levenberg–Marquardt step on the axis, Huber-robust, then an LM polish with the
Sim(3) frozen — prints axis `n`, centre `p` (metres, base frame), scale and
residual statistics. The hand–eye transform it uses is the pair of constants
at the top of the file (`R_CAMERA2GRIPPER`, `t_CAMERA2GRIPPER`, metres):
update them after re-running step 3. The result is `emitter.turntable` in
`rig_constants.yaml` and is what un-rotates every frame during reconstruction.

### 5. Colour matrix

A ColorChecker captured like a material (several distances and light
directions). The script is interactive: it opens an OpenCV window per image
and asks for the four board corners (TL → TR → BR → BL), so it needs a display.

```bash
python calibration/color_matrix.py \
    --lab_file      calibration/data/ColorChecker24_After_Nov2014.txt \
    --images_folder /absolute/path/to/colorchecker_session/hdr \
    --illuminant_xy 0.3818 0.3797        # the 4000 K LED; omit to adapt to D65
```

Fits one 3 × 3 matrix (camera RGB → linear sRGB; Lab-D50 reference,
Bradford-adapted to the LED white point), prints ΔE00 statistics and writes
`*_ccorr_u16.png` / `*_ccorr_preview.png` next to the inputs. The released
images are white-balanced sensor RGB *without* this matrix applied, so users
keep colorimetric control.

### 6. Radiometry (camera scale + LED profile)

A grey patch of known reflectance (0.9/π) is captured like a material and
*rendered* with the differentiable renderer under a constant emitter; the
captured/rendered ratio per emitter angle, binned at 1°, gives the LED angular
falloff Λ(θ) (`globals/emitter_calibration.json`) and its maximum the camera
counts-per-radiance scale (`camera.linear_factor*`). Materials were captured
under a few exposure regimes; `globals/camera_factor.json` maps material ids
to the matching factor. This step runs through the training stack
(`envs/training.txt`), not through this environment.

Fixed constants measured externally: the LED mount transform (`R_l2g/t_l2g`),
the dual-robot base registration (`base2_to_base1`), and the LED's physical
radius and beam FWHM (spec sheet).

## Per-session reconstruction

Input: a material folder `DATA_ROOT/<id>/` with `scan_log.json`, `ldr/`
(8-bit, COLMAP input) and `hdr_raw/` (16-bit Bayer), where `<id>` is an
integer and `DATA_ROOT/sample_size.json` lists the sample footprint for that
id.

### One material

```bash
export DATA_ROOT=/absolute/path/to/DATA_ROOT      # contains <id>/ and sample_size.json
export COLMAP_TMP=/absolute/path/to/fast-local-scratch   # COLMAP staging (default /tmp/robocloth_colmap)
bash scripts/reconstruct_material.sh "$DATA_ROOT/<id>" <gpu_id>
# extra arguments are Hydra overrides for reconstruct.py, e.g.
#   hydra.run.dir=/absolute/path/to/logs/<id>      (default: reconstruction/outputs/<date>/<time>/)
#   +shape_matching.error_threshold_mm=20          (alignment gate, default 16 mm)
# NUM_WORKERS=<n> sets the debayer / reprojection worker count (default 8).
```

The script runs `reconstruction/colmap.sh` when `<id>/sparse/` has no model
yet, then `reconstruction/reconstruct.py` (Hydra; config
`configs/renderer/reconstruction.yaml` on top of `rig_constants.yaml`).
What happens, in order:

1. **Sparse SfM** — COLMAP (feature extraction → sequential matching →
   mapping) on the 8-bit LDR copies. A registration gate requires ≥ 90 % of
   frames to register (`COLMAP_REGISTRATION_THRESHOLD`, default 0.90); a
   material below it is retried once with exhaustive matching — automatically
   by the batch scheduler, or by hand with
   `bash reconstruction/colmap_exhaustive.sh "$DATA_ROOT/<id>" <gpu_id>`
   before re-running the script above. `colmap.sh` /
   `colmap_exhaustive.sh` fail closed: the model is built in a local staging
   dir (`COLMAP_TMP`, removed on every exit path), validated there (loads
   with `read_write_model`, non-empty, passes the gate) and only then
   published — an existing `sparse/` is renamed to `sparse.prev-<timestamp>`
   (never deleted) and the new one renamed in. If any stage or the validation
   fails, the script exits nonzero, `sparse/` is left untouched and the staged
   output is kept as `<material>/sparse.failed-<timestamp>` for inspection.
   Published layout: `sparse/0/*.bin` plus flattened `*.bin`, `*.txt` and
   `points3D.ply`.
2. **Debayering** — the 16-bit Bayer mosaics (`hdr_raw/`) are demosaiced with
   Menon 2007 and white-balance gains into the linear 16-bit `hdr/` views
   (always regenerated from `hdr_raw/`; `RECON_TMP` names an optional local
   staging dir). No gamma or tone mapping is ever applied.
3. **Robot-frame alignment** — each frame's robot-logged pose is lifted to a
   common 0°-turntable world (undoing the table angle about the calibrated
   axis); one Umeyama Sim(3) then aligns the COLMAP reconstruction to this
   metric robot frame. Scans COLMAP never registered go to
   `unmatched_scan_ids.json`; frames with > 16 mm residual are discarded
   (`excluded_high_error_scan_ids.json`); the refined per-frame cameras are
   written to `rotated_camera.json`.
4. **Sample cropping** — the aligned sparse cloud is cropped to the sample's
   recorded physical footprint (`sample_size.json`) and depth-filtered by the
   IQR rule (`shape_matching.z_outlier_percentile`, 5.0), producing
   `bbox.json` and `sparse/points3D_transformed_filtered.ply`.
5. **Observation tensor** — every kept 3D point is reprojected into every
   HDR view (SIMPLE_RADIAL projection) and its RGB read from the **nearest
   pixel**: the sub-pixel projection is rounded with `np.round` (IEEE
   round-half-to-even — 1.49 → 1, 1.5 → 2, 2.5 → 2) and clipped to the image
   bounds; there is no bilinear interpolation. The released
   `observations_structured.npz` files were produced with exactly this lookup
   (`sample_pixels_nearest` in `reconstruction/reconstruct.py`), so it is kept
   as-is for parity with the released data. The result is assembled into
   `observations_structured.npz`: `rgbs (K frames × V points × 3)`
   uint16 with zeros marking missing observations, plus point positions and
   per-frame camera/light positions (`point_positions.npz` and
   `point_metadata.json` alongside). This single tensor — about 1 GB per
   material, 1.3 GB for material 145 (582 × 371 844 × 3) — is what stage-1
   training consumes: no per-frame image decoding at train time.

### Checks

```bash
# the gate colmap.sh applies before publishing (exit 0 ok · 2 unreadable/empty · 3 below threshold)
python reconstruction/registration_check.py --sparse-dir "$DATA_ROOT/<id>/sparse" --scan-log "$DATA_ROOT/<id>/scan_log.json"
# post-run warning flags (CATASTROPHIC_TRANS, MANY_EXCLUDED, MANY_UNMATCHED) from <id>/shape_matching.log + the json outputs
python -c "import sys; sys.path.insert(0, 'reconstruction'); from quality_check import detect_quality_warnings as q; print(q('$DATA_ROOT/<id>'))"
```

`shape_matching.log` is the captured stdout of `reconstruct.py` (the batch
scheduler writes it; for a manual run `tee` the output into the material
folder). `python reconstruction/read_write_model.py --input_model <sparse>
--input_format .bin --output_model <dir> --output_format .txt` converts a
model if you need the text form of a `.bin`-only reconstruction.

### Whole capture batches

`reconstruction/scheduler.py` drives COLMAP + reconstruction for every
integer-named material folder under a dataset root (minus the ids listed in
`<dataset>/skip.txt`) across several GPUs, keeps its state in
`<dataset>/scheduler_state.json`, dispatches the exhaustive-matcher retry for
materials below the registration gate and records the quality flags above:

```bash
python reconstruction/scheduler.py --dataset "$DATA_ROOT" --mode streaming --auto_detect
python reconstruction/scheduler.py --dataset "$DATA_ROOT" --status
# --only 13 42 · --force_redo 42 105 · --max_colmap_per_gpu N · --cpu_per_colmap N
```

GPU ids and per-job resource limits are class constants at the top of the
file (`Config.GPU_IDS`, ...); the flags above override the per-job limits.

`reconstruction/preprocess/` holds standalone helpers from capture time:
`debayering_multi_thread.py --input <dir> --output <dir> --pattern RGGB
[--format png|exr] [--jobs N]` (OpenCV demosaicing; the pipeline itself uses
the Menon-2007 path inside `reconstruct.py`), and two legacy scripts
(`purple_filter.py`, `background_mask_multi_thread.py`) that refer to modules
and configs of the original capture repository and are not wired into this
pipeline.
