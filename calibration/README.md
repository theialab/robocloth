# Calibration code

Offline rig-calibration solvers. The full explanation of the procedure — what
each script calibrates, from which captures, and where the results live — is
in [docs/capture_pipeline.md](../docs/capture_pipeline.md). All resulting
constants are recorded in
[configs/renderer/rig_constants.yaml](../configs/renderer/rig_constants.yaml).

Environment: `pip install -r envs/calibration.txt` (Python 3.10–3.12; OpenCV
4.12 with `cv2.aruco`, SciPy, colour-science, open3d — the recipe also
documents the COLMAP 3.8 install used by the reconstruction side).

| Script | Solves |
|---|---|
| `boards/generate_AprilTag_board*.py` | printable AprilTag 36h11 ring boards around the 250 × 170 mm sample cut-out (A4, 300 dpi PNG + PDF written to the current directory); the ChArUco intrinsics target is OpenCV's standard board — one-liner in docs/capture_pipeline.md §"Calibration boards" |
| `charuco_calibration.py` | camera intrinsics (Zhang) + board-based axis estimate — `--calib_glob`, `--top_glob/--scan_log_top`, `--tilt_glob/--scan_log_tilt`, optional `--c2g_json [--c2g_convention opencv\|opengl]` |
| `hand_eye.py` | camera→gripper transform (Tsai + metric upgrade) and COLMAP-world→base Sim(3) — `--scan_log_path`, `--poses_path` (`images.txt` or `transforms.json`), `--scale_mode auto\|joint\|umeyama`, `--out` (R_c2g in the **OpenCV** camera convention; `R_c2g_opengl` alongside for comparison with `rig_constants.yaml`) |
| `turntable_axis.py` | turntable rotation axis + center (robust alternating fit) |
| `color_matrix.py` | ColorChecker CCM + 4000 K white balance (`data/` holds the reference chart) |

The radiometric grey-patch flow (camera scale + LED angular profile) runs
through the training stack — see docs/capture_pipeline.md §"Radiometry".

## Order of operations

Run from the repository root with the calibration environment active; paths
are placeholders. Step 1 runs anywhere; steps 2–5 need their calibration
session (what each session records is described in docs/capture_pipeline.md).

1. **Boards** — `python calibration/boards/generate_AprilTag_board.py` and
   `python calibration/boards/generate_AprilTag_board_inner.py`, run inside the
   directory that should receive the PNG/PDF.
2. **Intrinsics (+ board-based axis check)** —
   `python calibration/charuco_calibration.py --calib_glob '/absolute/path/to/charuco/ldr/*.png' --top_glob '/absolute/path/to/axis_top/ldr/*.png' --scan_log_top /absolute/path/to/axis_top/scan_log.json`
   (`--tilt_glob` / `--scan_log_tilt` for a second fixed pose; `--c2g_json`
   to use a `hand_eye.py --out` result instead of the built-in constants).
3. **Hand–eye** — reconstruct the session first
   (`bash reconstruction/colmap.sh /absolute/path/to/handeye_session <gpu_id>`;
   its frames must carry `phi<φ>_theta<θ>` in the filename and `phi`/`theta`
   in the log), then
   `python calibration/hand_eye.py --scan_log_path /absolute/path/to/handeye_session/scan_log.json --poses_path /absolute/path/to/handeye_session/sparse/images.txt --out /absolute/path/to/hand_eye.json`.
4. **Turntable axis** — reconstruct the fixed-light turntable session the same
   way, copy `R_c2g`/`t_c2g` from step 3 into `R_CAMERA2GRIPPER` /
   `t_CAMERA2GRIPPER` at the top of `turntable_axis.py` (metres), then
   `python calibration/turntable_axis.py --scan_log_path /absolute/path/to/axis_session/scan_log.json --model_path /absolute/path/to/axis_session/sparse`.
5. **Colour matrix** —
   `python calibration/color_matrix.py --lab_file calibration/data/ColorChecker24_After_Nov2014.txt --images_folder /absolute/path/to/colorchecker/hdr --illuminant_xy 0.3818 0.3797`
   (interactive: click the four board corners in the OpenCV window for each image).
6. **Record** the results in `configs/renderer/rig_constants.yaml`
   (`camera.intrinsics`, `camera.R_c2g`/`t_c2g` in the OpenGL camera
   convention and metres, `emitter.turntable.center`/`axis`).

Unit tests for the hand-eye transforms (quaternion order, w2c↔c2w, Umeyama
return order, synthetic end-to-end solve) live in `tests/calibration/`:
`python -m unittest discover -s tests/calibration -v` (NumPy + SciPy + OpenCV only).

## Correctness fixes (2026-09-01)

| Where | What changed |
|---|---|
| `hand_eye.py` COLMAP `images.txt` parser | quaternion `[qw qx qy qz]` was passed straight to `scipy` `Rotation.from_quat` (which expects `[x y z w]`) — now converted by the named helper `colmap_qvec_to_rotmat`; the pose was also used as camera→world although COLMAP stores world→camera — now inverted (camera centre `= -Rᵀt`), matching `reconstruction/reconstruct.py` |
| `hand_eye.py` `estimate_camera2gripper` | `sim3_umeyama` returns `(s, R, t)` but was unpacked as `R, t, s`; the scaled **w2c** tuple was then passed where a list of c2w matrices was expected. Both fixed; `hand_eye_calibration` now takes world→camera matrices explicitly |
| `hand_eye.py` metric scale | the Umeyama fit of camera centres against *gripper* positions only approximates the scale (centres are offset by `t_c2g`); it now seeds Tsai's rotation, and `(t_c2g, scale)` are solved exactly by a joint linear least squares on `AX = XB` (`solve_translation_and_scale`) whenever the pose set makes the scale observable. On a purely spherical scan (camera orbiting one focus point — the rig's φ/θ pattern) the scale is provably unobservable from `AX = XB` (and with analytically exact log positions the joint solve collapses onto `t = p_f, s = 0`); the solver detects that the gripper motions share a fixed point, keeps the Umeyama scale (= the original behaviour, Tsai's translation at that scale) and emits a `RuntimeWarning`; `--scale_mode {auto,joint,umeyama}` overrides, `--out` records `scale_observable` and the diagnostics |
| `hand_eye.py` near-spherical gate (repair round) | a scan that is *almost* spherical passed the fixed-point test, but the joint solve is an errors-in-variables problem (only the camera column carries noise) whose scale is attenuated towards zero — on the test rig with 1e-3 rad camera noise, −50 % at 1e-3·radius perturbation, −12 % at 3e-3, versus the Umeyama fit's constant −4 %. `auto` now estimates the attenuation from the fit itself (`ρ ≈ ‖r‖² / (s²‖m_⊥‖²)`, first-order exact; reported as `scale_attenuation_est`, together with the raw `scale_joint`) and only accepts the joint solution when `ρ ≤ SCALE_ATTENUATION_TOL = 1e-2` (bias ≲ 1 %); otherwise it falls back to the Umeyama scale with a `RuntimeWarning` naming the reason. `scale_mode` is also exposed on `estimate_camera2gripper`. Regression test: `auto` is never worse than `umeyama` for perturbations 1e-3 … 1e-1 and switches to the joint solve (≤ 1 % scale error) once the scan is clearly non-spherical |
| `hand_eye.py` `estimate_world2base` | read the module-level `images_txt_path` instead of its argument; now every entry point takes its inputs from `argparse` (`--scan_log_path`, `--poses_path`) — no built-in paths |
| `hand_eye.py` matching failure (repair round) | with `--c2g_json` and < 3 matched frames the run died inside `np.vstack`; `main` now exits with a message naming the filename convention (`scan-<id>-phi<φ>_theta<θ>`), the log fields and `--tol`; `world_to_base_sim3` checks `≥ 3` poses like `solve_hand_eye` |
| `charuco_calibration.py` | machine-local absolute-path module globals (`CALIB_GLOB`, `SCANS1/2_GLOB`, `SCAN_LOG_TOP/TILT`) became CLI arguments; rig constants (board geometry, hand-eye result) unchanged |
| `charuco_calibration.py` `--c2g_json` convention (repair round) | the built-in `R_c2g` is expressed for the **OpenGL** camera frame (as in `rig_constants.yaml` / `reconstruct.py`) and is right-multiplied by `diag(1,−1,−1)` before being composed with the **OpenCV**-frame `solvePnP` board poses; the same flip was applied to a `--c2g_json` from `hand_eye.py --out`, which is already OpenCV-convention — every board pose ended up 180° about the camera x axis off in the base frame. `load_camera_to_gripper` now flips only OpenGL-convention inputs: the JSON's `camera_convention` field (written by `hand_eye.py`) decides, `--c2g_convention opencv\|opengl` overrides for files without it. `hand_eye.py --out` records `camera_convention: "opencv"` and `R_c2g_opengl`, and `hand_eye.py --c2g_json` honours the field too. Tests feed a `hand_eye.py --out` JSON through charuco's loader (effective `T_c2g` = `[R_json t_json]`, no extra flip) and check that a real-session `hand_eye.py` result lands 2.7° (not 179°) from the built-in constant |
