import torch
import numpy as np
import os
import re
import math
import scipy.io as spio
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import time
from dataclasses import dataclass
import random
import pyexr
import imageio
from concurrent.futures import ProcessPoolExecutor, as_completed

DTYPE_POLY = 0
DTYPE_PAN = 1
DTYPE_LLS = 2


def _pool_load_worker(args):
    """ProcessPoolExecutor worker: load one material in a subprocess.

    Builds a stub BonnDataset (bypassing __init__) with only the attributes
    _load_single_material needs, then delegates to it. Keeps the existing
    loader logic as the single source of truth — no duplicated code.
    """
    (mat_id, root_folder_str, calib, svfresnel_dir_str,
     use_pan, use_lls, point_subsample_ratio) = args
    stub = BonnDataset.__new__(BonnDataset)
    stub.root_folder = Path(root_folder_str)
    stub.calibrations = {mat_id: calib}
    stub.svfresnel_dir = Path(svfresnel_dir_str)
    stub.use_pan = use_pan
    stub.use_lls = use_lls
    stub.point_subsample_ratio = point_subsample_ratio
    return stub._load_single_material(mat_id)


# ---------------------------------------------------------------------------
# Channel-name parsers
# ---------------------------------------------------------------------------

def _parse_poly_channels(channel_names):
    """Parse poly channel names into per-image descriptors.

    pyexr returns channels sorted alphabetically.  Every 3 consecutive
    channels form one RGB image: (_B, _G, _R) which correspond to
    visual (R, G, B) — matching the official Bonn reference code that
    passes ``poly[:, :, :3]`` directly to ``plt.imshow``.

    Returns: list of dicts with keys camera, led, rotation, ch_start
             (ch_start = starting index of the 3-channel group).
    """
    pattern = re.compile(r'poly_(cv\d+)_(il\d+)_(rot\d+)_[BGR]')
    images = []
    for i in range(0, len(channel_names), 3):
        if i + 2 >= len(channel_names):
            break
        m = pattern.match(channel_names[i])
        if m:
            cam, led, rot = m.groups()
            images.append(dict(camera=cam, led=led, rotation=rot, ch_start=i))
    return images


def _parse_pan_channels(channel_names):
    """Parse pan channel names.

    Channel format: 'pan_cv01_il001_rot000'
    Returns: list of dicts with keys camera, led, rotation, channel
    """
    pattern = re.compile(r'pan_(cv\d+)_(il\d+)_(rot\d+)')
    images = []
    for name in channel_names:
        m = pattern.match(name)
        if m:
            cam, led, rot = m.groups()
            images.append(dict(camera=cam, led=led, rotation=rot, channel=name))
    return images


def _parse_lls_channels(channel_names):
    """Parse LLS channel names.

    Channel format: 'lls_cv01_lls01_la22.00_rot045'
    Returns: list of dicts with keys camera, angle, rotation, channel
    """
    pattern = re.compile(r'lls_(cv\d+)_lls\d+_la([+-]?\d+\.?\d*)_(rot\d+)')
    images = []
    for name in channel_names:
        m = pattern.match(name)
        if m:
            cam, angle, rot = m.groups()
            images.append(dict(camera=cam, angle=float(angle), rotation=rot, channel=name))
    return images


# ---------------------------------------------------------------------------
# Calibration parsing helpers
# ---------------------------------------------------------------------------

# Fallback weights when per-image calibration is unavailable (taken from the
# global mean across the per-camera poly→pan coefficients, ~[0.35, 0.36, 0.28]).
_DEFAULT_PAN_WEIGHTS = np.array([0.34, 0.36, 0.28], dtype=np.float32)

# Note: a single-scalar LLS empirical scale was previously applied here
# (0.75, fit from one pixel of mat0001). A population sweep across 25
# materials × 200 pixels (≈1.3M KNN matches) showed the true median
# `ref/lls` ratio is ~0.98 — i.e. essentially 1.0 — and varies per LLS
# la_angle (0.75–1.05). The single constant was wrong and is removed.
# See scripts/tests/compute_lls_factors_dataset.py for the diagnostic.


def _parse_poly2pan_weights(raw_mat):
    """Extract poly→pan RGB weights from a Bonn calibration .mat file.

    The .mat file's ``poly2panWeights`` struct holds 20 entries keyed by
    ``cv{01-04}_il{026,027,028,031,032}`` — the RGB-channel weights that
    convert a polychromatic RGB capture under poly LED ``il`` seen by
    camera ``cv`` to the corresponding panchromatic scalar response.

    For non-poly captures (pan LEDs il01-il24 and LLS strips) the .mat
    file provides no dedicated weights, so we fall back to the *global*
    mean across all 20 calibrated poly LED × camera entries (the per-cv
    mean differs from the global mean by <1 % — see
    ``scripts/tests/inspect_calibration_per_cv.py``).

    Returns a dict with three sub-dicts / arrays:
        per_il:     {(cv, il) -> (3,) float32}   — poly LEDs only
        per_cv:     {cv -> (3,) float32}         — mean across poly LEDs
        global_avg: (3,) float32                 — mean across all 20
    """
    if 'poly2panWeights' not in raw_mat:
        # Legacy / partial calibration files — fall back to global default.
        per_cv = {f'cv{i:02d}': _DEFAULT_PAN_WEIGHTS.copy() for i in range(1, 5)}
        return {'per_il': {}, 'per_cv': per_cv,
                'global_avg': _DEFAULT_PAN_WEIGHTS.copy()}

    wstruct = raw_mat['poly2panWeights'][0, 0]
    per_il = {}
    per_cv_accum = {f'cv{i:02d}': [] for i in range(1, 5)}
    for field in wstruct.dtype.names:
        # field example: 'cv01_il026'
        m = re.match(r'(cv\d{2})_(il\d{3})', field)
        if not m:
            continue
        cv, il = m.groups()
        vec = np.asarray(wstruct[field], dtype=np.float32).flatten()
        per_il[(cv, il)] = vec
        per_cv_accum[cv].append(vec)

    per_cv = {}
    for cv, vecs in per_cv_accum.items():
        if vecs:
            per_cv[cv] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        else:
            per_cv[cv] = _DEFAULT_PAN_WEIGHTS.copy()

    if per_il:
        all_vecs = np.stack(list(per_il.values()), axis=0)
        global_avg = all_vecs.mean(axis=0).astype(np.float32)
    else:
        global_avg = _DEFAULT_PAN_WEIGHTS.copy()
    return {'per_il': per_il, 'per_cv': per_cv, 'global_avg': global_avg}


def _pan_weights_for_image(calib, rot, camera, led=None,
                           is_poly=False, is_lls=False):
    """Look up RGB→pan weights for a single image.

    Args:
        calib:   dict produced by the dataset's ``_load_calibration`` — must
                 contain ``poly2pan_per_il`` and ``poly2pan_global_avg``.
        rot:     rotation key (unused today — weights are instrument-only —
                 kept to make future extensions obvious).
        camera:  cv key e.g. ``'cv01'``.
        led:     il key e.g. ``'il026'`` (poly LED) or ``None`` for pan/lls.
        is_poly: True iff this is a poly-LED capture.  If the (cv, il)
                 pair is one of the 5 calibrated filter LEDs
                 (il026/027/028/031/032), the exact per-(cv, il) weight
                 is returned; otherwise we fall back to the global mean
                 across all 20 calibrated entries.
        is_lls:  True iff this is a linear-light-source capture.  LLS has
                 no dedicated radiometric calibration in the released
                 dataset; the global pan weights are used as-is (no
                 empirical scaling — see the comment above).

    Returns: (3,) float32 vector.
    """
    per_il     = calib.get('poly2pan_per_il', {})
    global_avg = calib.get('poly2pan_global_avg', _DEFAULT_PAN_WEIGHTS)
    if is_poly and led is not None:
        key = (camera, led)
        if key in per_il:
            return per_il[key].copy()
        # poly under il01–il24 (no per-(cv, il) calibration).
        return global_avg.copy()
    if is_lls:
        return global_avg.copy()
    # Pan: use global average of the 5 calibrated filter LEDs × 4 cameras.
    return global_avg.copy()


# ---------------------------------------------------------------------------
# EXR reading helpers  (uses pyexr, matching the official Bonn reference code)
# ---------------------------------------------------------------------------

def _read_exr(filepath):
    """Read all channels from an EXR using pyexr.

    Returns (data (H,W,C) float16, channel_names list, H, W).
    The Bonn EXR files store data natively as half precision,
    so we keep float16 to save memory.
    """
    exr = pyexr.open(str(filepath))
    ch_names = exr.channel_map['all']
    data = exr.get(group='all', precision=pyexr.HALF)  # float16
    H, W = data.shape[:2]
    return data, ch_names, H, W


def _read_xyz_map(filepath):
    """Read xyz_rot000.exr → (xyz (H,W,3) float32, H, W)."""
    xyz = pyexr.read(str(filepath))
    H, W = xyz.shape[:2]
    return xyz, H, W


def _read_gt_normal_map(svfresnel_dir, mat_id, H, W):
    """Read the AxF-decoded tangent-space normal map and convert to world space.

    The AxF SDK decodes normal maps with ORIGIN_TOPLEFT, so V increases
    downward = decreasing Y in tangent space.  The Bonn xyz maps have rows
    increasing in +Y world.  A vertical flip aligns the two grids.
    Because the samples lie on a nearly-flat surface, the TBN is
    approximately axis-aligned, so the flipped tangent-space normal
    already serves as a world-space normal.

    Returns (normals_flat (H*W, 3) float32) or None if the file is missing.
    """
    import cv2
    svfresnel_dir = Path(svfresnel_dir)
    exr_path = svfresnel_dir / f'mat{mat_id:04d}_svfresnel' / f'mat{mat_id:04d}_svfresnel_Normal.exr'
    if not exr_path.exists():
        return None

    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    bgr = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (H_tex, W_tex, 3) float32
    H_tex, W_tex = rgb.shape[:2]

    if (H_tex, W_tex) != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    normals = np.flipud(rgb).copy()  # vertical flip: TOPLEFT V-axis → +Y world
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    return normals.reshape(-1, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# BonnDataset
# ---------------------------------------------------------------------------

class BonnDataset(IterableDataset):
    """Bonn SVBRDF database (UBOFAB19) dataset.

    Loads calibrated HDR measurements from multi-channel EXR files.
    All images are reprojected onto the top camera's pixel grid so
    pixel (i, j) across ALL channels refers to the same surface point.

    Data type indices:
        0 = polychromatic  (RGB, color-filtered LEDs)
        1 = panchromatic   (grayscale, unfiltered LEDs)
        2 = LLS            (grayscale, linear light source)
    """

    DTYPE_POLY = DTYPE_POLY

    # ------------------------------------------------------------------
    @dataclass
    class ChunkData:
        """All materials loaded into memory.

        Each entry in *materials* is a dict from ``_load_single_material``:
        mat_id, xyz (V,3), point_ids (V,), gt_normals (V,3) or None,
        rgbs (K,V,3) float16, light_pos (K,3), cam_pos (K,3).
        """
        materials: list            # list of per-material dicts
        total_obs: int             # sum(n_pixels * n_images) across materials

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, cfg, root_folder, split='train'):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.split = split
        self.rays_num = cfg.data.rays_num

        self.debug = getattr(cfg.data, 'debug', False)
        self.debug_num = getattr(cfg.data, 'debug_num', 1)
        self.points_per_material = getattr(cfg.data, 'points_per_material', 2000)
        self.random_observations = getattr(cfg.data, 'random_observations', False)
        self.subsample_ratio = getattr(cfg.data, 'subsample_ratio', 1.0)
        self.val_materials = getattr(cfg.data, 'val_materials', 2)
        self.val_points = getattr(cfg.data, 'val_points', 500)
        self.random_sample_material_number = getattr(cfg.data, 'random_sample_material_number', None)
        self.debug_rotate = getattr(cfg.data, 'debug_rotate', False)
        self.debug_swap_channels = getattr(cfg.data, 'debug_swap_channels', False)
        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)
        self.svfresnel_dir = self.root_folder / 'Bonn_svfresnel'
        self.num_load_workers = int(getattr(cfg.data, 'num_load_workers', 0))
        self.step = 0

        # Per-material point subsampling ratio. Mirrors MultiMaterialDenseDataset
        # in points.py: each material keeps a deterministic random subset of its
        # H*W pixels (seeded by mat_id) and emits dense point_ids 0..N_sub-1 so
        # the BonnLatentBRDF latent bank stays aligned across train/val/decoder.
        self.point_subsample_ratio = float(
            getattr(cfg.data, 'point_subsample_ratio', 1.0))

        # Approximate luminance weights for pan→scalar projection
        self.pan_weights = np.array([0.34, 0.36, 0.28], dtype=np.float32)

        # Discover materials
        self.mat_ids = self._discover_materials()
        if self.debug:
            self.mat_ids = self.mat_ids[:self.debug_num]
            print(f"[DEBUG] Using {self.debug_num} material(s): {self.mat_ids}")

        print(f"\n{'='*60}")
        print(f"BonnDataset ({split})  |  materials={len(self.mat_ids)}  "
              f"use_pan={self.use_pan}  use_lls={self.use_lls}")
        print(f"{'='*60}")

        # Load lightweight calibration for every material
        self.calibrations = {}
        for mid in tqdm(self.mat_ids, desc="Loading calibrations"):
            self.calibrations[mid] = self._load_calibration(mid)

        # Load ALL materials into memory (no chunk switching)
        if split == 'train':
            print(f"Loading ALL {len(self.mat_ids)} materials into memory ...")
            self._all_data = self._load_all_materials()
            print(f"All data loaded: {self._all_data.total_obs:,} total observations "
                  f"from {len(self._all_data.materials)} materials")
        else:
            print("Loading validation subset …")
            self._val_data = self._load_all_materials(val_mode=True)
            print(f"Validation: {self._val_data.total_obs:,} observations")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Material discovery
    # ------------------------------------------------------------------
    def _discover_materials(self):
        training_list_path = getattr(self.cfg.data, 'training_list_path', '')
        if training_list_path and os.path.isfile(training_list_path):
            mat_ids = []
            with open(training_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        mat_ids.append(int(line))
            print(f"Loaded {len(mat_ids)} material IDs from {training_list_path}")
            return sorted(mat_ids)

        poly_files = sorted(self.root_folder.glob('mat*_poly.exr'))
        mat_ids = []
        for p in poly_files:
            mat_str = p.stem.split('_')[0]       # 'mat0001'
            mat_ids.append(int(mat_str[3:]))      # 1
        print(f"Auto-discovered {len(mat_ids)} materials in {self.root_folder}")
        return sorted(mat_ids)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _mat_prefix(self, mat_id):
        return self.root_folder / f'mat{mat_id:04d}'

    def _load_calibration(self, mat_id):
        path = f'{self._mat_prefix(mat_id)}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)   # (3,4,14)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()  # (3,)
                # Calibration uses 2-digit names (il01, cv01) but EXR
                # channels use 3-digit (il001, cv01).  Store both keys
                # so lookups from either convention work.
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
        w = _parse_poly2pan_weights(raw)
        calib['poly2pan_per_il']     = w['per_il']
        calib['poly2pan_per_cv']     = w['per_cv']
        calib['poly2pan_global_avg'] = w['global_avg']
        return calib

    # ------------------------------------------------------------------
    # Single-material loader
    # ------------------------------------------------------------------
    def _load_single_material(self, mat_id):
        """Load one Bonn material with ALL valid pixels.

        Uses pyexr (matching the official Bonn reference code) to read
        all EXR data.  Poly channels come in alphabetically sorted groups
        of 3 (_B, _G, _R) which map to visual (R, G, B).

        Returns compact dict or None on failure.
        """
        try:
            prefix = self._mat_prefix(mat_id)
            calib = self.calibrations[mat_id]

            # ---- xyz map ------------------------------------------------
            xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
            n_pixels = H * W
            xyz_pts = xyz_map.reshape(n_pixels, 3).astype(np.float32)
            pids = np.arange(n_pixels, dtype=np.int64)

            # ---- ground-truth normal map (from decoded AxF) -------------
            gt_normals = _read_gt_normal_map(self.svfresnel_dir, mat_id, H, W)

            # ============================================================
            # Read all channels from each EXR using pyexr
            # ============================================================
            poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
            assert (pH, pW) == (H, W)

            pan_data, pan_ch_names = None, None
            if self.use_pan:
                pan_data, pan_ch_names, panH, panW = _read_exr(f'{prefix}_pan.exr')
                assert (panH, panW) == (H, W)

            lls_data, lls_ch_names = None, None
            lls_angle_to_idx = {}
            if self.use_lls:
                lls_angles = calib['llsAnglesDegrees']
                lls_angle_to_idx = {float(a): i for i, a in enumerate(lls_angles)}
                lls_data, lls_ch_names, lH, lW = _read_exr(f'{prefix}_lls.exr')
                assert (lH, lW) == (H, W)

            # ============================================================
            # Assembly
            # ============================================================
            rgbs_parts        = []
            gray_parts        = []
            light_parts       = []
            cam_parts         = []
            dtype_parts       = []
            eid_parts         = []
            lls_corner_parts  = []
            pan_weight_parts  = []
            eid = 0

            # --- Poly (RGB) ---------------------------------------------
            poly_images = _parse_poly_channels(poly_ch_names)
            n_poly = len(poly_images)
            if n_poly > 0:
                poly_rgbs = poly_data.reshape(n_pixels, n_poly, 3)  # (V, K, 3)
                poly_rgbs = poly_rgbs.transpose(1, 0, 2)            # (K, V, 3)

                poly_light = np.array([
                    calib[im['rotation']][im['led']] for im in poly_images])
                poly_cam = np.array([
                    calib[im['rotation']][im['camera']] for im in poly_images])
                poly_pan_w = np.stack([
                    _pan_weights_for_image(calib, im['rotation'],
                                           im['camera'], im['led'], is_poly=True)
                    for im in poly_images], axis=0).astype(np.float32)

                rgbs_parts.append(poly_rgbs)
                light_parts.append(poly_light)
                cam_parts.append(poly_cam)
                dtype_parts.append(np.full(n_poly, DTYPE_POLY, dtype=np.int64))
                eid_parts.append(np.arange(eid, eid + n_poly, dtype=np.int64))
                lls_corner_parts.append(np.zeros((n_poly, 4, 3), dtype=np.float32))
                pan_weight_parts.append(poly_pan_w)
                eid += n_poly
            del poly_data

            # --- Pan (grayscale, il001-il024 only) -----------------------
            if self.use_pan and pan_data is not None:
                pan_name_to_idx = {n: i for i, n in enumerate(pan_ch_names)}
                pan_images = [im for im in _parse_pan_channels(pan_ch_names)
                              if int(im['led'][2:]) <= 24]
                n_pan = len(pan_images)
                if n_pan > 0:
                    pan_flat = pan_data.reshape(n_pixels, -1)       # (V, C)
                    ch_ci = np.array([pan_name_to_idx[im['channel']]
                                      for im in pan_images])
                    pan_gray = pan_flat[:, ch_ci].T                  # (K, V)

                    pan_light = np.array([
                        calib[im['rotation']][im['led']] for im in pan_images])
                    pan_cam = np.array([
                        calib[im['rotation']][im['camera']] for im in pan_images])
                    pan_pan_w = np.stack([
                        _pan_weights_for_image(calib, im['rotation'],
                                               im['camera'], led=None, is_poly=False)
                        for im in pan_images], axis=0).astype(np.float32)
                    gray_parts.append(pan_gray)
                    light_parts.append(pan_light)
                    cam_parts.append(pan_cam)
                    dtype_parts.append(np.full(n_pan, DTYPE_PAN, dtype=np.int64))
                    eid_parts.append(np.arange(eid, eid + n_pan, dtype=np.int64))
                    lls_corner_parts.append(np.zeros((n_pan, 4, 3), dtype=np.float32))
                    pan_weight_parts.append(pan_pan_w)
                    eid += n_pan
                del pan_data

            # --- LLS (grayscale) -----------------------------------------
            if self.use_lls and lls_data is not None:
                lls_name_to_idx = {n: i for i, n in enumerate(lls_ch_names)}
                lls_images = _parse_lls_channels(lls_ch_names)
                n_lls = len(lls_images)
                if n_lls > 0:
                    lls_flat = lls_data.reshape(n_pixels, -1)       # (V, C)
                    ch_ci = np.array([lls_name_to_idx[im['channel']]
                                      for im in lls_images])
                    lls_gray = lls_flat[:, ch_ci].T                  # (K, V)

                    corners_list = []
                    center_list = []
                    for im in lls_images:
                        rd = calib[im['rotation']]
                        ai = lls_angle_to_idx[im['angle']]
                        c = rd['llsCorners'][:, :, ai].T             # (4, 3)
                        corners_list.append(c)
                        center_list.append(c.mean(axis=0))

                    lls_light = np.array(center_list, dtype=np.float32)
                    lls_cam = np.array([
                        calib[im['rotation']][im['camera']] for im in lls_images])
                    lls_corners_arr = np.array(corners_list, dtype=np.float32)
                    lls_pan_w = np.stack([
                        _pan_weights_for_image(calib, im['rotation'],
                                               im['camera'], led=None,
                                               is_poly=False, is_lls=True)
                        for im in lls_images], axis=0).astype(np.float32)

                    gray_parts.append(lls_gray)
                    light_parts.append(lls_light)
                    cam_parts.append(lls_cam)
                    dtype_parts.append(np.full(n_lls, DTYPE_LLS, dtype=np.int64))
                    eid_parts.append(np.arange(eid, eid + n_lls, dtype=np.int64))
                    lls_corner_parts.append(lls_corners_arr)
                    pan_weight_parts.append(lls_pan_w)
                    eid += n_lls
                del lls_data

            # ---- concatenate across source types ------------------------
            if not rgbs_parts and not gray_parts:
                return None

            all_rgbs       = np.concatenate(rgbs_parts, axis=0) if rgbs_parts else np.empty((0, n_pixels, 3), dtype=np.float16)
            gray_vals      = np.concatenate(gray_parts, axis=0) if gray_parts else None
            all_light_pos  = np.concatenate(light_parts, axis=0)
            all_cam_pos    = np.concatenate(cam_parts, axis=0)
            all_data_type  = np.concatenate(dtype_parts, axis=0)
            all_emitter_id = np.concatenate(eid_parts, axis=0)
            all_lls_corner = np.concatenate(lls_corner_parts, axis=0)
            all_pan_weight = np.concatenate(pan_weight_parts, axis=0) \
                if pan_weight_parts else np.empty((0, 3), dtype=np.float32)

            np.clip(all_rgbs, 0, None, out=all_rgbs)
            if gray_vals is not None:
                np.clip(gray_vals, 0, None, out=gray_vals)

            Visualize = False
            if Visualize:
                # --- debug: save all images ordered by camera then light ------
                _dbg_dir = os.path.join(os.environ.get("ROBOCLOTH_DEBUG_DIR", "./visualizations_debug"), f"mat{mat_id:04d}")
                os.makedirs(_dbg_dir, exist_ok=True)

                n_images = all_rgbs.shape[0]

                def _order_positions(positions):
                    """Assign ordered indices to unique 3D positions.

                    Order: elevation low→high, then azimuth 0→180°.
                    Elevation = arctan2(z, sqrt(x²+y²)), azimuth = arctan2(y, x).
                    Returns per-image ordered index array (int).
                    """
                    # round to avoid floating-point duplicates
                    rounded = np.round(positions, decimals=4)
                    unique_pos, inverse = np.unique(
                        rounded, axis=0, return_inverse=True)
                    # compute spherical coords for each unique position
                    x, y, z = unique_pos[:, 0], unique_pos[:, 1], unique_pos[:, 2]
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
                    azimuth   = np.arctan2(y, x)
                    # sort unique positions: primary=elevation, secondary=azimuth
                    order = np.lexsort((azimuth, elevation))
                    # rank[old_unique_idx] = new_ordered_idx
                    rank = np.empty_like(order)
                    rank[order] = np.arange(len(order))
                    # map back to per-image ordered indices
                    return rank[inverse]

                cam_idx   = _order_positions(all_cam_pos)
                light_idx = _order_positions(all_light_pos)

                # sort images: camera first, then light
                sort_order = np.lexsort((light_idx, cam_idx))

                # assign local light index within each camera group
                local_light = np.empty_like(light_idx)
                for ci in np.unique(cam_idx):
                    mask = cam_idx == ci
                    # get globally-sorted light indices for this camera
                    li_vals = light_idx[mask]
                    # assign dense local ranks 0,1,2,... preserving global order
                    _, inv = np.unique(li_vals, return_inverse=True)
                    local_light[mask] = inv

                for seq, img_i in enumerate(sort_order):
                    # all_rgbs shape is (K, V, 3) where V = H*W
                    _rgb = all_rgbs[img_i].reshape(H, W, 3).astype(np.float32)
                    _rgb_uint8 = (np.clip(_rgb, 0, 1) * 255).astype(np.uint8)
                    _ci = cam_idx[img_i]
                    _li = local_light[img_i]
                    imageio.imwrite(
                        os.path.join(_dbg_dir,
                                    f"img_{seq:03d}_cam{_ci:02d}_light{_li:02d}.png"),
                        _rgb_uint8)
                print(f"  [Debug] Saved {n_images} images to {_dbg_dir}")

                # --- save coloured point clouds of camera & light positions ---
                def _save_colored_ply(filepath, positions, order_indices):
                    """Save a PLY point cloud coloured red→green by order index."""
                    n = len(positions)
                    max_idx = order_indices.max() if n > 0 else 1
                    t = order_indices.astype(np.float64) / max(max_idx, 1)  # 0→1
                    r = ((1 - t) * 255).astype(np.uint8)   # red channel
                    g = (t * 255).astype(np.uint8)          # green channel
                    b = np.zeros(n, dtype=np.uint8)
                    with open(filepath, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {n}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        f.write("property uchar red\n")
                        f.write("property uchar green\n")
                        f.write("property uchar blue\n")
                        f.write("end_header\n")
                        for j in range(n):
                            f.write(f"{positions[j,0]:.6f} {positions[j,1]:.6f} "
                                    f"{positions[j,2]:.6f} {r[j]} {g[j]} {b[j]}\n")

                _save_colored_ply(
                    os.path.join(_dbg_dir, "pointcloud_cameras.ply"),
                    all_cam_pos, cam_idx)
                _save_colored_ply(
                    os.path.join(_dbg_dir, "pointcloud_lights.ply"),
                    all_light_pos, light_idx)
                print(f"  [Debug] Saved camera & light point clouds to {_dbg_dir}")
                # --- end debug ------------------------------------------------

            # Per-material point subsampling. Compacts every V-indexed array
            # (xyz, all_rgbs, gray_vals, gt_normals) onto a deterministic random
            # subset of pixels, seeded by mat_id so train/val/BRDF agree on the
            # same subset. point_ids are reset to dense 0..N_sub-1 so the latent
            # bank in BonnLatentBRDF (sized num_points*ratio per material) lines
            # up with the dataloader's emitted indices.
            ratio = float(getattr(self, 'point_subsample_ratio', 1.0))
            if ratio < 1.0:
                rng = np.random.default_rng(mat_id)
                N_sub = max(1, int(n_pixels * ratio))
                sub_idx = np.sort(rng.choice(n_pixels, size=N_sub, replace=False))

                xyz_pts = xyz_pts[sub_idx]
                pids = np.arange(N_sub, dtype=np.int64)
                if gt_normals is not None:
                    gt_normals = gt_normals[sub_idx]
                if all_rgbs.shape[1] > 0:
                    all_rgbs = all_rgbs[:, sub_idx, :]
                if gray_vals is not None:
                    gray_vals = gray_vals[:, sub_idx]
                n_pixels = N_sub

            n_poly_img = all_rgbs.shape[0]
            n_gray_img = gray_vals.shape[0] if gray_vals is not None else 0
            n_images = n_poly_img + n_gray_img
            mem_mb = all_rgbs.nbytes / 1e6
            if gray_vals is not None:
                mem_mb += gray_vals.nbytes / 1e6
            n_poly_loaded = (all_data_type == DTYPE_POLY).sum()
            n_pan_loaded  = (all_data_type == DTYPE_PAN).sum()
            n_lls_loaded  = (all_data_type == DTYPE_LLS).sum()
            print(f"  mat{mat_id:04d}: {n_pixels:,} px × {n_images} img "
                  f"(poly={n_poly_loaded}, pan={n_pan_loaded}, lls={n_lls_loaded})  "
                  f"pixel data {mem_mb:.0f} MB")

            return {
                'mat_id':      mat_id,
                'xyz':         xyz_pts,         # (V, 3)   float32
                'point_ids':   pids,            # (V,)     int64
                'rgbs':        all_rgbs,        # (K_poly, V, 3) float16
                'gray_vals':   gray_vals,       # (K_gray, V) float16 or None
                'light_pos':   all_light_pos,   # (K, 3)   float32
                'cam_pos':     all_cam_pos,     # (K, 3)   float32
                'data_type':   all_data_type,   # (K,)     int64
                'emitter_ids': all_emitter_id,  # (K,)     int64
                'lls_corners': all_lls_corner,  # (K, 4, 3) float32
                'pan_weights': all_pan_weight,  # (K, 3)   float32
                'gt_normals':  gt_normals,      # (V, 3)   float32 or None
            }

        except Exception as exc:
            print(f"  [Warning] Failed to load mat{mat_id:04d}: {exc}")
            import traceback; traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _make_debug_pair(self, mat):
        """Create a modified copy of *mat* as a synthetic second material.

        - debug_rotate:        rotate light_pos and cam_pos 180° around Z.
        - debug_swap_channels: swap R and G channels in rgbs.
        """
        import copy
        mat2 = copy.deepcopy(mat)
        mat2['mat_id'] = mat['mat_id'] + 1  # fake second ID

        if self.debug_rotate:
            # 180° rotation around Z: (x, y, z) -> (-x, -y, z)
            mat2['light_pos'] = mat2['light_pos'].copy()
            mat2['light_pos'][:, 0] *= -1
            mat2['light_pos'][:, 1] *= -1
            mat2['cam_pos'] = mat2['cam_pos'].copy()
            mat2['cam_pos'][:, 0] *= -1
            mat2['cam_pos'][:, 1] *= -1
            print(f"[DEBUG] Created mat {mat2['mat_id']} by rotating light/cam 180° around Z")

        if self.debug_swap_channels:
            # Swap R (idx 0) and G (idx 1)
            rgbs = mat2['rgbs'].copy()          # (K, V, 3) float16
            rgbs[:, :, 0], rgbs[:, :, 1] = mat['rgbs'][:, :, 1].copy(), mat['rgbs'][:, :, 0].copy()
            mat2['rgbs'] = rgbs
            print(f"[DEBUG] Created mat {mat2['mat_id']} by swapping R/G channels")

        return mat2

    # ------------------------------------------------------------------
    # Load all materials
    # ------------------------------------------------------------------
    def _load_all_materials(self, val_mode=False):
        """Load all materials into memory at once (no chunking)."""
        if val_mode:
            selected = self.mat_ids[:min(self.val_materials, len(self.mat_ids))]
        else:
            selected = self.mat_ids  # load ALL materials

        tag = 'val' if val_mode else 'train'

        if self.num_load_workers > 0 and len(selected) > 1:
            materials = self._load_materials_parallel(selected, tag)
        else:
            materials = []
            for mid in tqdm(selected, desc=f"Loading {tag} (all materials)"):
                result = self._load_single_material(mid)
                if result is not None:
                    materials.append(result)

        # Debug pair: duplicate first material with modification
        if (self.debug_rotate or self.debug_swap_channels) and materials:
            mat2 = self._make_debug_pair(materials[0])
            materials.append(mat2)

        if not materials:
            return BonnDataset.ChunkData(materials=[], total_obs=0)

        total_obs = sum(
            (m['rgbs'].shape[0] + (m['gray_vals'].shape[0] if m['gray_vals'] is not None else 0))
            * m['rgbs'].shape[1] for m in materials)
        total_pixel_mb = sum(
            m['rgbs'].nbytes + (m['gray_vals'].nbytes if m['gray_vals'] is not None else 0)
            for m in materials) / 1e6

        print(f"[{tag}] All data loaded: {total_obs:,} observations "
              f"from {len(materials)} materials  "
              f"(pixel data {total_pixel_mb:.0f} MB)")

        return BonnDataset.ChunkData(materials=materials, total_obs=total_obs)

    # ------------------------------------------------------------------
    # Parallel material loader (ProcessPoolExecutor)
    # ------------------------------------------------------------------
    def _load_materials_parallel(self, mat_ids, tag):
        """Load materials in parallel using ProcessPoolExecutor.

        Returns materials in the same order as mat_ids (sorted by mat_id),
        matching the serial loader. Failed loads are dropped.
        """
        n_workers = min(self.num_load_workers, len(mat_ids))
        print(f"[{tag}] Parallel load: {len(mat_ids)} materials, "
              f"{n_workers} workers (ProcessPoolExecutor)")

        tasks = [
            (mid, str(self.root_folder), self.calibrations[mid],
             str(self.svfresnel_dir), self.use_pan, self.use_lls,
             self.point_subsample_ratio)
            for mid in mat_ids
        ]

        results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_mid = {ex.submit(_pool_load_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(future_to_mid),
                            total=len(future_to_mid),
                            desc=f"Loading {tag} (parallel)"):
                mid = future_to_mid[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [Warning] mat{mid:04d} worker raised: {exc}")
                    res = None
                if res is not None:
                    results[mid] = res

        return [results[mid] for mid in mat_ids if mid in results]

    # ------------------------------------------------------------------
    # Training-loop interface
    # ------------------------------------------------------------------
    def set_step(self, step: int):
        """No-op: chunk switching is disabled, all data is in memory."""
        self.step = step

    def __len__(self):
        if hasattr(self, '_val_data'):
            return max(1, math.ceil(self._val_data.total_obs / self.rays_num))
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, chunk, n_rays):
        """Randomly sample n_rays from chunk.

        Rays are distributed across materials proportionally to their
        observation count (n_images * n_pixels), then random (img, pixel)
        pairs are drawn within each material.
        """
        materials = chunk.materials
        k = self.random_sample_material_number
        if k is not None and 0 < k < len(materials):
            materials = [materials[i] for i in np.random.choice(len(materials), k, replace=False)]
        obs_counts = np.array(
            [(m['rgbs'].shape[0] + (m['gray_vals'].shape[0] if m['gray_vals'] is not None else 0))
             * m['rgbs'].shape[1] for m in materials],
            dtype=np.float64)
        weights = obs_counts / obs_counts.sum()
        rays_per_mat = np.round(weights * n_rays).astype(int)
        rays_per_mat[-1] = n_rays - rays_per_mat[:-1].sum()

        parts_xyz     = []
        parts_rgbs    = []
        parts_pids    = []
        parts_mids    = []
        parts_dtype   = []
        parts_eids    = []
        parts_lls     = []
        parts_light   = []
        parts_cam     = []
        parts_normals = []
        parts_panw    = []

        for mi, mat in enumerate(materials):
            n = int(rays_per_mat[mi])
            if n <= 0:
                continue

            n_poly = mat['rgbs'].shape[0]
            n_gray = mat['gray_vals'].shape[0] if mat['gray_vals'] is not None else 0
            n_images = n_poly + n_gray
            n_pixels = mat['rgbs'].shape[1]

            img_i = np.random.randint(0, n_images, n)
            pix_i = np.random.randint(0, n_pixels, n)

            parts_xyz.append(mat['xyz'][pix_i])
            # Fetch pixel values: poly → 3-ch RGB, pan/lls → 1-ch grayscale in ch0
            rgbs_sampled = np.zeros((n, 3), dtype=np.float32)
            poly_m = img_i < n_poly
            if poly_m.any():
                rgbs_sampled[poly_m] = mat['rgbs'][img_i[poly_m], pix_i[poly_m]].astype(np.float32)
            if n_gray > 0:
                gray_m = ~poly_m
                if gray_m.any():
                    rgbs_sampled[gray_m, 0] = mat['gray_vals'][img_i[gray_m] - n_poly, pix_i[gray_m]].astype(np.float32)
            parts_rgbs.append(rgbs_sampled)
            parts_pids.append(mat['point_ids'][pix_i])
            parts_mids.append(np.full(n, mat['mat_id'], dtype=np.int64))
            parts_dtype.append(mat['data_type'][img_i])
            parts_eids.append(mat['emitter_ids'][img_i])
            parts_lls.append(mat['lls_corners'][img_i])
            parts_light.append(mat['light_pos'][img_i])
            parts_cam.append(mat['cam_pos'][img_i])
            parts_panw.append(mat['pan_weights'][img_i])
            if mat['gt_normals'] is not None:
                parts_normals.append(mat['gt_normals'][pix_i])

        xyz     = np.concatenate(parts_xyz)
        rgbs    = np.concatenate(parts_rgbs)
        light   = np.concatenate(parts_light)
        cam     = np.concatenate(parts_cam)
        has_normals = len(parts_normals) > 0
        normals = np.concatenate(parts_normals) if has_normals else None

        wi = light - xyz
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam - xyz
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        # Shuffle so rays from different materials are interleaved
        pids    = np.concatenate(parts_pids)
        mids    = np.concatenate(parts_mids)
        dtypes  = np.concatenate(parts_dtype)
        eids    = np.concatenate(parts_eids)
        lls_c   = np.concatenate(parts_lls)
        panw    = np.concatenate(parts_panw)
        perm = np.random.permutation(n_rays)
        xyz, wi, wo, rgbs, pids, mids = \
            xyz[perm], wi[perm], wo[perm], rgbs[perm], pids[perm], mids[perm]
        dtypes, eids, lls_c, panw = dtypes[perm], eids[perm], lls_c[perm], panw[perm]
        if normals is not None:
            normals = normals[perm]

        # confidence = 0 only when all rgb channels are 0 (occluded pixels)
        confidence = (rgbs.sum(axis=-1) > 0).astype(np.float32)

        result = {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(pids).long(),
            'material_ids': torch.from_numpy(mids).long(),
            'data_type':    torch.from_numpy(dtypes).long(),
            'emitter_ids':  torch.from_numpy(eids).long(),
            'lls_corners':  torch.from_numpy(lls_c).float(),
            'pan_weights':  torch.from_numpy(panw).float(),
            'confidence':   torch.from_numpy(confidence),
            'gt_params':    torch.zeros(1),
        }
        if has_normals:
            result['gt_normals'] = torch.from_numpy(normals).float()
        return result

    # ------------------------------------------------------------------
    def __iter__(self):
        # Per-(rank, worker) RNG seed so DDP ranks and DataLoader workers
        # don't all draw the same indices. Sampling is with replacement, so
        # different streams across ranks just contribute fresh batches to the
        # effective per-step batch (gradients all-reduce afterwards).
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            rank = 0
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = (np.random.SeedSequence(entropy=[rank, worker_id, self.step])
                .generate_state(1)[0])
        np.random.seed(int(seed))

        # ----- training (infinite, all data in memory) -----
        if hasattr(self, '_all_data'):
            while True:
                chunk = self._all_data
                if chunk is None or chunk.total_obs == 0:
                    time.sleep(0.01)
                    continue

                yield self._sample_batch(chunk, self.rays_num)

        # ----- validation (finite, sequential) -----
        else:
            chunk = self._val_data
            total = chunk.total_obs
            n_batches = max(1, math.ceil(total / self.rays_num))

            for i in range(n_batches):
                n = min(self.rays_num, total - i * self.rays_num)
                yield self._sample_batch(chunk, n)


# ---------------------------------------------------------------------------
# BonnValDataset  (standard Dataset, one full image per __getitem__)
# ---------------------------------------------------------------------------

class BonnValDataset(Dataset):
    """Validation dataset for Bonn SVBRDF.

    Randomly picks one material, selects ``valid_num`` poly images.
    Each ``__getitem__`` returns **all pixels** (H*W) for one image,
    together with ``img_hw`` so the trainer can reconstruct the 2-D
    image for side-by-side visualisation.

    Return dict keys match the training batch (xyz, wi, wo, rgbs, …)
    so the same forward-model code can be reused.
    """

    def __init__(self, cfg, root_folder):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.valid_num = getattr(cfg.data, 'valid_num', 5)
        self.debug = getattr(cfg.data, 'debug', False)
        self.debug_rotate = getattr(cfg.data, 'debug_rotate', False)
        self.debug_swap_channels = getattr(cfg.data, 'debug_swap_channels', False)
        # Per-material subsampling — must match BonnDataset (and the latent bank
        # in BonnLatentBRDF). Seed is the EXR mat_id so the subset is identical
        # to training, even when validate_id differs (debug_rotate / swap mode).
        self.point_subsample_ratio = float(
            getattr(cfg.data, 'point_subsample_ratio', 1.0))
        self.svfresnel_dir = Path(root_folder) / 'Bonn_svfresnel'

        # ---- discover materials & pick one ---------------------
        mat_ids = self._discover_materials()
        if self.debug:
            self.mat_id = 1
            if self.debug_rotate or self.debug_swap_channels:
                self.validate_id = 2
            else:
                self.validate_id = self.mat_id
        else:
            self.mat_id = 1
            self.validate_id = 1
        prefix = self.root_folder / f'mat{self.mat_id:04d}'

        print(f"\n{'='*60}")
        print(f"BonnValDataset  |  material=mat{self.mat_id:04d}  "
              f"valid_num={self.valid_num}")
        print(f"{'='*60}")

        # ---- load xyz map & calibration ---------------------------------
        self.xyz_map, self.H, self.W = \
            _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        self.n_pixels = self.H * self.W

        # ---- ground-truth normal map ------------------------------------
        self.gt_normals = _read_gt_normal_map(
            self.svfresnel_dir, self.mat_id, self.H, self.W)

        calib = self._load_calibration(self.mat_id)

        xyz_flat = self.xyz_map.reshape(self.n_pixels, 3).astype(np.float32)
        pids = np.arange(self.n_pixels, dtype=np.int64)

        # Per-material point subsampling (same recipe as BonnDataset).
        # Seed by self.mat_id (the EXR loaded), not validate_id, so the subset
        # matches training — in debug_rotate/swap mode the synthetic mat 2
        # in BonnDataset just deepcopies mat 1's already-subsampled points.
        self._sub_indices = None
        if self.point_subsample_ratio < 1.0:
            rng = np.random.default_rng(self.mat_id)
            N_sub = max(1, int(self.n_pixels * self.point_subsample_ratio))
            self._sub_indices = np.sort(
                rng.choice(self.n_pixels, size=N_sub, replace=False))
            xyz_flat = xyz_flat[self._sub_indices]
            pids = np.arange(N_sub, dtype=np.int64)
            if self.gt_normals is not None:
                self.gt_normals = self.gt_normals[self._sub_indices]
            self.n_pixels = N_sub

        # ---- read poly data using pyexr (same as official Bonn code) -----
        poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
        assert (pH, pW) == (self.H, self.W)
        poly_images = _parse_poly_channels(poly_ch_names)

        n_select = min(self.valid_num, len(poly_images))
        if n_select == len(poly_images):
            selected = poly_images
        else:
            selected = random.sample(poly_images, n_select)
        print(f"Selected {n_select} poly images for validation "
              f"({self.n_pixels} pixels each)")

        # ---- preload every selected image --------------------------------
        self._items = []
        for eid, img in enumerate(selected):
            rot_data = calib[img['rotation']]

            cam_pos = rot_data[img['camera']].copy()
            led_pos = rot_data[img['led']].copy()

            idx = img['ch_start']
            rgbs = poly_data[:, :, idx:idx+3].reshape(-1, 3)    # (H*W, 3)
            np.clip(rgbs, 0, None, out=rgbs)
            if self._sub_indices is not None:
                rgbs = rgbs[self._sub_indices]

            # Apply same debug transforms as training _make_debug_pair
            if self.debug and self.debug_rotate:
                # 180° rotation around Z: (x, y, z) -> (-x, -y, z)
                cam_pos[0] *= -1; cam_pos[1] *= -1
                led_pos[0] *= -1; led_pos[1] *= -1

            if self.debug and self.debug_swap_channels:
                # Swap R (idx 0) and G (idx 1)
                rgbs = rgbs.copy()
                rgbs[:, 0], rgbs[:, 1] = rgbs[:, 1].copy(), rgbs[:, 0].copy()

            wo = cam_pos[None, :] - xyz_flat
            wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

            wi = led_pos[None, :] - xyz_flat
            wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)

            label = (f"mat{self.mat_id:04d}_{img['camera']}_"
                     f"{img['led']}_{img['rotation']}")

            # confidence = 0 only when all rgb channels are 0 (occluded pixels)
            confidence = (rgbs.sum(axis=-1) > 0).astype(np.float32)

            pan_w = _pan_weights_for_image(
                calib, img['rotation'], img['camera'], img['led'], is_poly=True)

            # sub_indices maps each of the N_sub flat entries back to its
            # ORIGINAL position in the H*W image grid. The trainer scatters
            # per-pixel predictions onto a zero canvas at these positions
            # (non-supervised pixels stay black) so 2-D visualisations remain
            # geometrically correct even when point_subsample_ratio < 1.0.
            # When ratio == 1.0 we still emit arange(H*W) so the trainer can
            # always rely on the same scatter path.
            if self._sub_indices is not None:
                sub_idx_t = torch.from_numpy(self._sub_indices.astype(np.int64))
            else:
                sub_idx_t = torch.arange(self.n_pixels, dtype=torch.long)

            item = {
                'xyz':          torch.from_numpy(xyz_flat.copy()).float(),
                'wi':           torch.from_numpy(wi).float(),
                'wo':           torch.from_numpy(wo).float(),
                'rgbs':         torch.from_numpy(rgbs).float(),
                'point_ids':    torch.from_numpy(pids.copy()).long(),
                'material_ids': torch.full((self.n_pixels,), self.validate_id,
                                           dtype=torch.long),
                'emitter_ids':  torch.full((self.n_pixels,), eid,
                                           dtype=torch.long),
                'data_type':    torch.full((self.n_pixels,), DTYPE_POLY,
                                           dtype=torch.long),
                'lls_corners':  torch.zeros(self.n_pixels, 4, 3),
                'pan_weights':  torch.from_numpy(
                    np.broadcast_to(pan_w, (self.n_pixels, 3)).copy()).float(),
                'confidence':   torch.from_numpy(confidence),
                'img_hw':       torch.tensor([self.H, self.W]),
                'sub_indices':  sub_idx_t,
                'gt_params':    torch.zeros(1),
                'label':        label,
            }
            if self.gt_normals is not None:
                item['gt_normals'] = torch.from_numpy(self.gt_normals.copy()).float()
            self._items.append(item)
        del poly_data

        print(f"BonnValDataset ready  ({len(self._items)} images)\n"
              f"{'='*60}\n")

    # ------------------------------------------------------------------
    def _discover_materials(self):
        training_list_path = getattr(self.cfg.data, 'training_list_path', '')
        if training_list_path and os.path.isfile(training_list_path):
            ids = []
            with open(training_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.append(int(line))
            return sorted(ids)
        poly_files = sorted(self.root_folder.glob('mat*_poly.exr'))
        return sorted(int(p.stem.split('_')[0][3:]) for p in poly_files)

    def _load_calibration(self, mat_id):
        path = f'{self.root_folder / f"mat{mat_id:04d}"}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten() \
                                         .astype(np.float64)
        w = _parse_poly2pan_weights(raw)
        calib['poly2pan_per_il']     = w['per_il']
        calib['poly2pan_per_cv']     = w['per_cv']
        calib['poly2pan_global_avg'] = w['global_avg']
        return calib

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


# ---------------------------------------------------------------------------
# Single-material full loader (no pixel subsampling)
# ---------------------------------------------------------------------------

def _load_single_material_full(root_folder, mat_id, use_pan=False, use_lls=False):
    """Load one Bonn material with ALL pixels (no subsampling).

    Same assembly logic as BonnDataset._load_single_material but keeps
    all H*W pixels and also returns H, W for image reconstruction.

    Returns dict or None on failure.
    """
    try:
        root_folder = Path(root_folder)
        prefix = root_folder / f'mat{mat_id:04d}'
        svfresnel_dir = root_folder / 'Bonn_svfresnel'

        # ---- calibration ------------------------------------------------
        path = f'{prefix}_calibration.mat'
        raw = spio.loadmat(path)
        calib = {}
        for rot_key in ['rot000', 'rot045', 'rot090', 'rot135', 'rot180']:
            rd = raw[rot_key][0, 0]
            rot_dict = {}
            for field in rd.dtype.names:
                val = rd[field]
                if field == 'llsCorners':
                    rot_dict[field] = np.array(val, dtype=np.float32)
                else:
                    rot_dict[field] = np.array(val, dtype=np.float32).flatten()
                m = re.match(r'(il|cv)(\d+)', field)
                if m and len(m.group(2)) < 3:
                    padded = f'{m.group(1)}{int(m.group(2)):03d}'
                    rot_dict[padded] = rot_dict[field]
            calib[rot_key] = rot_dict
        calib['llsAnglesDegrees'] = raw['llsAnglesDegrees'].flatten().astype(np.float64)
        w = _parse_poly2pan_weights(raw)
        calib['poly2pan_per_il']     = w['per_il']
        calib['poly2pan_per_cv']     = w['per_cv']
        calib['poly2pan_global_avg'] = w['global_avg']

        # ---- xyz map ----------------------------------------------------
        xyz_map, H, W = _read_xyz_map(f'{prefix}_xyz_rot000.exr')
        n_pixels = H * W
        xyz_pts = xyz_map.reshape(n_pixels, 3).astype(np.float32)
        pids = np.arange(n_pixels, dtype=np.int64)

        # ---- ground-truth normal map (from decoded AxF) -----------------
        gt_normals = _read_gt_normal_map(svfresnel_dir, mat_id, H, W)

        # ---- read EXR data ----------------------------------------------
        poly_data, poly_ch_names, pH, pW = _read_exr(f'{prefix}_poly.exr')
        assert (pH, pW) == (H, W)

        pan_data, pan_ch_names = None, None
        if use_pan:
            pan_data, pan_ch_names, panH, panW = _read_exr(f'{prefix}_pan.exr')
            assert (panH, panW) == (H, W)

        lls_data, lls_ch_names = None, None
        lls_angle_to_idx = {}
        if use_lls:
            lls_angles = calib['llsAnglesDegrees']
            lls_angle_to_idx = {float(a): i for i, a in enumerate(lls_angles)}
            lls_data, lls_ch_names, lH, lW = _read_exr(f'{prefix}_lls.exr')
            assert (lH, lW) == (H, W)

        # ---- assembly (no pixel subsampling) ----------------------------
        rgbs_parts       = []
        gray_parts       = []
        light_parts      = []
        cam_parts        = []
        dtype_parts      = []
        eid_parts        = []
        lls_corner_parts = []
        pan_weight_parts = []
        eid = 0

        # Poly (RGB)
        poly_images = _parse_poly_channels(poly_ch_names)
        n_poly = len(poly_images)
        if n_poly > 0:
            poly_flat = poly_data.reshape(n_pixels, -1)
            poly_rgbs = poly_flat.reshape(n_pixels, n_poly, 3).transpose(1, 0, 2)
            poly_light = np.array([
                calib[im['rotation']][im['led']] for im in poly_images])
            poly_cam = np.array([
                calib[im['rotation']][im['camera']] for im in poly_images])
            poly_pan_w = np.stack([
                _pan_weights_for_image(calib, im['rotation'],
                                       im['camera'], im['led'], is_poly=True)
                for im in poly_images], axis=0).astype(np.float32)

            rgbs_parts.append(poly_rgbs)
            light_parts.append(poly_light)
            cam_parts.append(poly_cam)
            dtype_parts.append(np.full(n_poly, DTYPE_POLY, dtype=np.int64))
            eid_parts.append(np.arange(eid, eid + n_poly, dtype=np.int64))
            lls_corner_parts.append(np.zeros((n_poly, 4, 3), dtype=np.float32))
            pan_weight_parts.append(poly_pan_w)
            eid += n_poly
        del poly_data

        # Pan (grayscale, il001-il024 only)
        if use_pan and pan_data is not None:
            pan_name_to_idx = {n: i for i, n in enumerate(pan_ch_names)}
            pan_images = [im for im in _parse_pan_channels(pan_ch_names)
                          if int(im['led'][2:]) <= 24]
            n_pan = len(pan_images)
            if n_pan > 0:
                pan_flat = pan_data.reshape(n_pixels, -1)
                ch_ci = np.array([pan_name_to_idx[im['channel']]
                                  for im in pan_images])
                pan_gray = pan_flat[:, ch_ci].T                  # (K, V)

                pan_light = np.array([
                    calib[im['rotation']][im['led']] for im in pan_images])
                pan_cam = np.array([
                    calib[im['rotation']][im['camera']] for im in pan_images])
                pan_pan_w = np.stack([
                    _pan_weights_for_image(calib, im['rotation'],
                                           im['camera'], led=None, is_poly=False)
                    for im in pan_images], axis=0).astype(np.float32)
                gray_parts.append(pan_gray)
                light_parts.append(pan_light)
                cam_parts.append(pan_cam)
                dtype_parts.append(np.full(n_pan, DTYPE_PAN, dtype=np.int64))
                eid_parts.append(np.arange(eid, eid + n_pan, dtype=np.int64))
                lls_corner_parts.append(np.zeros((n_pan, 4, 3), dtype=np.float32))
                pan_weight_parts.append(pan_pan_w)
                eid += n_pan
            del pan_data

        # LLS (grayscale)
        if use_lls and lls_data is not None:
            lls_name_to_idx = {n: i for i, n in enumerate(lls_ch_names)}
            lls_images = _parse_lls_channels(lls_ch_names)
            n_lls = len(lls_images)
            if n_lls > 0:
                lls_flat = lls_data.reshape(n_pixels, -1)
                ch_ci = np.array([lls_name_to_idx[im['channel']]
                                  for im in lls_images])
                lls_gray = lls_flat[:, ch_ci].T                  # (K, V)

                corners_list = []
                center_list = []
                for im in lls_images:
                    rd = calib[im['rotation']]
                    ai = lls_angle_to_idx[im['angle']]
                    c = rd['llsCorners'][:, :, ai].T             # (4, 3)
                    corners_list.append(c)
                    center_list.append(c.mean(axis=0))

                lls_light = np.array(center_list, dtype=np.float32)
                lls_cam = np.array([
                    calib[im['rotation']][im['camera']] for im in lls_images])
                lls_corners_arr = np.array(corners_list, dtype=np.float32)
                lls_pan_w = np.stack([
                    _pan_weights_for_image(calib, im['rotation'],
                                           im['camera'], led=None,
                                           is_poly=False, is_lls=True)
                    for im in lls_images], axis=0).astype(np.float32)

                gray_parts.append(lls_gray)
                light_parts.append(lls_light)
                cam_parts.append(lls_cam)
                dtype_parts.append(np.full(n_lls, DTYPE_LLS, dtype=np.int64))
                eid_parts.append(np.arange(eid, eid + n_lls, dtype=np.int64))
                lls_corner_parts.append(lls_corners_arr)
                pan_weight_parts.append(lls_pan_w)
                eid += n_lls
            del lls_data

        if not rgbs_parts and not gray_parts:
            return None

        all_rgbs       = np.concatenate(rgbs_parts, axis=0) if rgbs_parts else np.empty((0, n_pixels, 3), dtype=np.float16)
        gray_vals      = np.concatenate(gray_parts, axis=0) if gray_parts else None
        all_light_pos  = np.concatenate(light_parts, axis=0)      # (K, 3)
        all_cam_pos    = np.concatenate(cam_parts, axis=0)        # (K, 3)
        all_data_type  = np.concatenate(dtype_parts, axis=0)      # (K,)
        all_emitter_id = np.concatenate(eid_parts, axis=0)        # (K,)
        all_lls_corner = np.concatenate(lls_corner_parts, axis=0) # (K, 4, 3)
        all_pan_weight = np.concatenate(pan_weight_parts, axis=0) \
            if pan_weight_parts else np.empty((0, 3), dtype=np.float32)

        np.clip(all_rgbs, 0, None, out=all_rgbs)
        if gray_vals is not None:
            np.clip(gray_vals, 0, None, out=gray_vals)

        n_poly_img = all_rgbs.shape[0]
        n_gray_img = gray_vals.shape[0] if gray_vals is not None else 0
        n_images = n_poly_img + n_gray_img

        n_poly_loaded = (all_data_type == DTYPE_POLY).sum()
        n_pan_loaded  = (all_data_type == DTYPE_PAN).sum()
        n_lls_loaded  = (all_data_type == DTYPE_LLS).sum()
        mem_mb = all_rgbs.nbytes / 1e6
        if gray_vals is not None:
            mem_mb += gray_vals.nbytes / 1e6
        print(f"  mat{mat_id:04d}: {n_pixels:,} pixels × {n_images} images "
              f"(poly={n_poly_loaded}, pan={n_pan_loaded}, lls={n_lls_loaded})  "
              f"pixel data {mem_mb:.0f} MB")

        return {
            'mat_id':      mat_id,
            'n_pixels':    n_pixels,
            'n_images':    n_images,
            'H': H, 'W': W,
            'xyz':         xyz_pts,           # (V, 3)
            'point_ids':   pids,              # (V,)
            'rgbs':        all_rgbs,          # (K_poly, V, 3)
            'gray_vals':   gray_vals,         # (K_gray, V) or None
            'light_pos':   all_light_pos,     # (K, 3)
            'cam_pos':     all_cam_pos,       # (K, 3)
            'data_type':   all_data_type,     # (K,)
            'emitter_ids': all_emitter_id,    # (K,)
            'lls_corners': all_lls_corner,    # (K, 4, 3)
            'pan_weights': all_pan_weight,    # (K, 3)
            'gt_normals':  gt_normals,        # (V, 3)  float32 or None
        }

    except Exception as exc:
        print(f"  [Warning] Failed to load mat{mat_id:04d}: {exc}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# BonnSingleMaterialDataset  (IterableDataset, single-material stage-2)
# ---------------------------------------------------------------------------

class BonnSingleMaterialDataset(IterableDataset):
    """Single-material Bonn dataset for stage 2 overfitting.

    Loads ALL pixels for one material into memory (no subsampling,
    no double-buffer chunk switching).  Images are split 80/20 into
    train/val with a fixed seed.  Training yields infinite random
    (image, pixel) pair batches.

    Return dict keys match the stage-1 training batch exactly.
    """

    def __init__(self, cfg, root_folder, split='train'):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.split = split
        self.rays_num = cfg.data.rays_num
        self.mat_id = cfg.data.overfit_mat_id
        self.val_view_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)
        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)

        print(f"\n{'='*60}")
        print(f"BonnSingleMaterialDataset ({split})  |  mat{self.mat_id:04d}  "
              f"use_pan={self.use_pan}  use_lls={self.use_lls}")
        print(f"{'='*60}")

        # Load full material (all pixels, all images)
        mat_data = _load_single_material_full(root_folder, self.mat_id,
                                               use_pan=self.use_pan,
                                               use_lls=self.use_lls)
        if mat_data is None:
            raise RuntimeError(f"Failed to load mat{self.mat_id:04d}")

        # Split images 80/20 with fixed seed (no sort: keeps random order so the
        # first valid_num val items used for image saving are representative)
        n_images = mat_data['n_images']
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_images)
        n_val = max(1, int(n_images * self.val_view_ratio))
        val_indices   = perm[:n_val]
        train_indices = perm[n_val:]
        if split == 'train':
            indices = train_indices
        else:
            indices = val_indices

        # Store the split's subset
        self.n_pixels = mat_data['n_pixels']
        self.n_images = len(indices)
        self.H = mat_data['H']
        self.W = mat_data['W']
        self.xyz        = mat_data['xyz']                     # (V, 3)
        self.point_ids  = mat_data['point_ids']               # (V,)
        self.gt_normals = mat_data['gt_normals']              # (V, 3) or None
        self.light_pos  = mat_data['light_pos'][indices]      # (K', 3)
        self.cam_pos    = mat_data['cam_pos'][indices]        # (K', 3)
        self.data_type  = mat_data['data_type'][indices]      # (K',)
        self.emitter_ids = mat_data['emitter_ids'][indices]   # (K',)
        self.lls_corners = mat_data['lls_corners'][indices]   # (K', 4, 3)
        self.pan_weights = mat_data['pan_weights'][indices]   # (K', 3)

        # Split pixel data: poly (3-ch) vs gray (1-ch) for memory efficiency
        n_poly_total = mat_data['rgbs'].shape[0]
        is_poly = indices < n_poly_total
        self.rgbs = mat_data['rgbs'][indices[is_poly]] if is_poly.any() else np.empty((0, self.n_pixels, 3), dtype=np.float16)
        if (~is_poly).any() and mat_data['gray_vals'] is not None:
            self.gray_vals = mat_data['gray_vals'][indices[~is_poly] - n_poly_total]
        else:
            self.gray_vals = None
        # local_idx: maps subset position → per-type array index
        self.local_idx = np.empty(self.n_images, dtype=np.int64)
        self.local_idx[is_poly] = np.arange(is_poly.sum())
        if (~is_poly).any():
            self.local_idx[~is_poly] = np.arange((~is_poly).sum())

        total_obs = self.n_pixels * self.n_images
        mem_mb = self.rgbs.nbytes / 1e6
        if self.gray_vals is not None:
            mem_mb += self.gray_vals.nbytes / 1e6
        print(f"  {split}: {self.n_pixels:,} pixels × {self.n_images} images "
              f"= {total_obs:,} obs  (pixel data {mem_mb:.0f} MB)")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    def set_step(self, step: int):
        pass

    def __len__(self):
        return 1_000_000

    # ------------------------------------------------------------------
    def _sample_batch(self, n_rays):
        img_i = np.random.randint(0, self.n_images, n_rays)
        pix_i = np.random.randint(0, self.n_pixels, n_rays)

        xyz      = self.xyz[pix_i]
        # Fetch pixel values: poly → 3-ch RGB, pan/lls → 1-ch grayscale in ch0
        dt = self.data_type[img_i]
        li = self.local_idx[img_i]
        rgbs = np.zeros((n_rays, 3), dtype=np.float32)
        poly_m = dt == DTYPE_POLY
        if poly_m.any():
            rgbs[poly_m] = self.rgbs[li[poly_m], pix_i[poly_m]].astype(np.float32)
        gray_m = ~poly_m
        if gray_m.any() and self.gray_vals is not None:
            rgbs[gray_m, 0] = self.gray_vals[li[gray_m], pix_i[gray_m]].astype(np.float32)
        light    = self.light_pos[img_i]
        cam      = self.cam_pos[img_i]

        wi = light - xyz
        wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
        wo = cam - xyz
        wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

        # confidence = 0 only when all rgb channels are 0 (occluded pixels)
        confidence = (rgbs.sum(axis=-1) > 0).astype(np.float32)

        result = {
            'xyz':          torch.from_numpy(xyz).float(),
            'wi':           torch.from_numpy(wi).float(),
            'wo':           torch.from_numpy(wo).float(),
            'rgbs':         torch.from_numpy(rgbs).float(),
            'point_ids':    torch.from_numpy(self.point_ids[pix_i].copy()).long(),
            'material_ids': torch.full((n_rays,), self.mat_id, dtype=torch.long),
            'emitter_ids':  torch.from_numpy(self.emitter_ids[img_i].copy()).long(),
            'data_type':    torch.from_numpy(self.data_type[img_i].copy()).long(),
            'lls_corners':  torch.from_numpy(self.lls_corners[img_i].copy()).float(),
            'pan_weights':  torch.from_numpy(self.pan_weights[img_i].copy()).float(),
            'confidence':   torch.from_numpy(confidence),
            'gt_params':    torch.zeros(1),
        }
        if self.gt_normals is not None:
            result['gt_normals'] = torch.from_numpy(self.gt_normals[pix_i]).float()
        return result

    # ------------------------------------------------------------------
    def __iter__(self):
        while True:
            yield self._sample_batch(self.rays_num)


# ---------------------------------------------------------------------------
# BonnSingleMaterialValDataset  (Dataset, single-material stage-2 val)
# ---------------------------------------------------------------------------

class BonnSingleMaterialValDataset(Dataset):
    """Validation dataset for single-material Bonn stage-2 overfitting.

    Uses the SAME material and the SAME fixed-seed split as
    ``BonnSingleMaterialDataset`` but takes the held-out 20 % of images.
    Each ``__getitem__`` returns **all pixels** (H*W) for one image
    together with ``img_hw`` so the trainer can reconstruct 2-D images.

    Return dict keys match the stage-1 validation batch exactly.
    """

    def __init__(self, cfg, root_folder):
        self.cfg = cfg
        self.root_folder = Path(root_folder)
        self.mat_id = cfg.data.overfit_mat_id
        self.val_view_ratio = getattr(cfg.data, 'val_view_ratio', 0.2)
        self.val_seed = getattr(cfg.data, 'val_seed', 42)
        self.valid_num = getattr(cfg.data, 'valid_num', -1)
        self.use_pan = getattr(cfg.data, 'use_pan', False)
        self.use_lls = getattr(cfg.data, 'use_lls', False)

        print(f"\n{'='*60}")
        print(f"BonnSingleMaterialValDataset  |  mat{self.mat_id:04d}")
        print(f"{'='*60}")

        # Load full material
        mat_data = _load_single_material_full(root_folder, self.mat_id,
                                               use_pan=self.use_pan,
                                               use_lls=self.use_lls)
        if mat_data is None:
            raise RuntimeError(f"Failed to load mat{self.mat_id:04d}")

        self.H = mat_data['H']
        self.W = mat_data['W']
        self.n_pixels = mat_data['n_pixels']
        self.gt_normals = mat_data['gt_normals']  # (V, 3) or None

        # Reproduce the same split as training. Keep ALL val images so val
        # metrics are computed on the full 20% held out; ``valid_num`` is
        # exposed for the trainer to gate per-view image saving (the first
        # ``valid_num`` items, which are random thanks to the unsorted perm).
        n_images = mat_data['n_images']
        rng = np.random.RandomState(self.val_seed)
        perm = rng.permutation(n_images)
        n_val = max(1, int(n_images * self.val_view_ratio))
        val_indices = perm[:n_val]

        n_save = min(self.valid_num, len(val_indices)) if self.valid_num > 0 else len(val_indices)
        print(f"  {len(val_indices)} val images for metrics; "
              f"saving images for first {n_save}  ({self.n_pixels:,} pixels each)")

        # Precompute one item per val image (all pixels)
        xyz_flat = mat_data['xyz']      # (V, 3)
        pids     = mat_data['point_ids']  # (V,)

        n_poly_total = mat_data['rgbs'].shape[0]
        self._items = []
        for eid_local, k in enumerate(val_indices):
            light = mat_data['light_pos'][k]          # (3,)
            cam   = mat_data['cam_pos'][k]            # (3,)

            wi = light[None, :] - xyz_flat
            wi /= np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-8)
            wo = cam[None, :] - xyz_flat
            wo /= np.maximum(np.linalg.norm(wo, axis=1, keepdims=True), 1e-8)

            # Fetch pixel data: poly → 3-ch, gray → 1-ch in ch0
            if k < n_poly_total:
                rgbs = mat_data['rgbs'][k]            # (V, 3) float16
            else:
                v = mat_data['gray_vals'][k - n_poly_total]  # (V,) float16
                rgbs = np.zeros((self.n_pixels, 3), dtype=np.float16)
                rgbs[:, 0] = v

            dtype_val = int(mat_data['data_type'][k])
            eid_val   = int(mat_data['emitter_ids'][k])

            label = f"mat{self.mat_id:04d}_img{k:03d}"

            # confidence = 0 only when all rgb channels are 0 (occluded pixels)
            confidence = (rgbs.sum(axis=-1) > 0).astype(np.float32)

            pan_w = mat_data['pan_weights'][k]  # (3,)

            item = {
                'xyz':          torch.from_numpy(xyz_flat.copy()).float(),
                'wi':           torch.from_numpy(wi).float(),
                'wo':           torch.from_numpy(wo).float(),
                'rgbs':         torch.from_numpy(rgbs.copy()).float(),
                'point_ids':    torch.from_numpy(pids.copy()).long(),
                'material_ids': torch.full((self.n_pixels,), self.mat_id,
                                           dtype=torch.long),
                'emitter_ids':  torch.full((self.n_pixels,), eid_val,
                                           dtype=torch.long),
                'data_type':    torch.full((self.n_pixels,), dtype_val,
                                           dtype=torch.long),
                'lls_corners':  torch.from_numpy(
                    np.broadcast_to(mat_data['lls_corners'][k],
                                    (self.n_pixels, 4, 3)).copy()).float(),
                'pan_weights':  torch.from_numpy(
                    np.broadcast_to(pan_w, (self.n_pixels, 3)).copy()).float(),
                'confidence':   torch.from_numpy(confidence),
                'img_hw':       torch.tensor([self.H, self.W]),
                'sub_indices':  torch.arange(self.n_pixels, dtype=torch.long),
                'gt_params':    torch.zeros(1),
                'label':        label,
            }
            if self.gt_normals is not None:
                item['gt_normals'] = torch.from_numpy(self.gt_normals.copy()).float()
            self._items.append(item)

        print(f"BonnSingleMaterialValDataset ready  ({len(self._items)} images)\n"
              f"{'='*60}\n")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
