#!/usr/bin/env python3
import os
import re
import json
import shutil
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

# COLMAP helpers (same ones you already use)
from read_write_model import read_model, qvec2rotmat, read_points3D_binary, read_points3D_text, detect_model_format

# -------------------- small linear-algebra helpers --------------------

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3]  = np.asarray(t, float).reshape(3)
    return T

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot (base) camera centres (or targets)
    Returns: s, R, t  such that  Q ≈ s R P + t
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q
    Sigma = (Q0.T @ P0) / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT
    s = (np.trace(np.diag(D) @ S) / (P0**2).sum() * len(P)) if with_scale else 1.0
    t = mu_Q - s * (R @ mu_P)
    return s, R, t

def rodrigues_axis_angle(n: np.ndarray, degrees: float | np.ndarray) -> np.ndarray:
    """
    Right-handed CCW rotation about axis n by angle_rad (rad).
    Supports scalar or vector of angles → returns (..,3,3).
    """
    th = np.deg2rad(degrees)
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=float)
    I = np.eye(3)
    angle = np.asarray(th, float)[..., None, None]
    Sa = np.sin(angle); Ca = np.cos(angle)
    return I + Sa * K + (1.0 - Ca) * (K @ K)

def T_about_point(R, p):
    """Homogeneous transform that rotates by R about world point p (3,)."""
    p = np.asarray(p, float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = (np.eye(3) - R) @ p
    return T

def sample_pixels_nearest(img, px, py):
    """
    Nearest-pixel RGB lookup — the dataset's observation-sampling convention.

    Sub-pixel projections (px, py) are rounded with np.round (IEEE
    round-half-to-even: 1.49 -> 1, 1.5 -> 2, 2.5 -> 2), cast to int32 and
    clipped to the image bounds; there is NO bilinear interpolation. Returns
    img[y, x] with shape (M, C). Every released observations_structured.npz
    was produced with exactly this lookup, so it must not be changed (a
    bilinear variant would silently break parity with the released data).
    See docs/capture_pipeline.md, step 5.
    """
    x_coords = np.round(px).astype(np.int32)
    y_coords = np.round(py).astype(np.int32)
    x_coords = np.clip(x_coords, 0, img.shape[1] - 1)
    y_coords = np.clip(y_coords, 0, img.shape[0] - 1)
    return img[y_coords, x_coords]

# -------------------- your existing helpers (kept) --------------------

def parse_colmap_images_txt(images):
    """
    images (read_model output) → dict[name] = (colmap_image_id, 4x4 c2w)
    """
    cam_c2w_dict = {}
    for img_id, img in images.items():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        opencv_to_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)
        w2c = opencv_to_opengl @ w2c
        c2w = np.linalg.inv(w2c)
        cam_c2w_dict[img.name] = (img_id, c2w)
    return cam_c2w_dict

def find_matching_entry(fname, scan_log):
    """
    Expect filenames like ...scan-<id>....*  → match scan_log entry with e['id']==<id>
    """
    base = fname.replace(".png", "").replace(".jpg", "")
    m = re.search(r'scan-(\d+)', base)
    if not m:
        raise ValueError(f"No 'scan-<id>' in filename: {fname}")
    scan_id = int(m.group(1))
    for i, e in enumerate(scan_log):
        if int(e["scan_id"]) == scan_id:
            return i
    raise ValueError(f"No match for scan_id={scan_id}")

def save_unmatched_scan_ids(colmap_c2w_dict, scan_log, unmatched_scan_ids_path):
    """
    Save scan_ids that are in scan_log but not found in any COLMAP filename to a JSON file.
    """
    # Extract all scan_ids from filenames
    matched_scan_ids = set()
    for fname in colmap_c2w_dict.keys():
        base = fname.replace(".png", "").replace(".jpg", "")
        m = re.search(r'scan-(\d+)', base)
        if m:
            matched_scan_ids.add(int(m.group(1)))
    
    # Find scan_ids in scan_log that weren't matched
    scan_log_ids = set(int(e["scan_id"]) for e in scan_log)
    unmatched_ids = sorted(list(scan_log_ids - matched_scan_ids))
    
    # Save to JSON file (just the list of IDs)
    with open(unmatched_scan_ids_path, 'w') as f:
        json.dump(unmatched_ids, f, indent=2)
    
    if unmatched_ids:
        print(f"Scan IDs in scan_log but not in COLMAP filenames: {unmatched_ids}")
    else:
        print("All scan_log IDs were matched in COLMAP filenames.")
    print(f"Saved unmatched scan IDs to: {unmatched_scan_ids_path}")

# -------------------- rotation undo and pose building --------------------

def rotated_c2w(json_entry, R_c2g, t_c2g, rotation_center, rotation_axis):
    """
    Compute T_{c->w0}: camera pose in the 0-angle world by undoing table rotation.
    """
    # gripper->base from robot
    R_g2b = np.asarray(json_entry["rotation_matrix"], dtype=float)
    t_g2b = np.asarray(json_entry["position"], dtype=float) / 1000.0  # mm -> m
    T_g2b = build_4x4(R_g2b, t_g2b)

    # camera->gripper (given)
    T_c2g = build_4x4(R_c2g, t_c2g)

    # camera->base at this frame
    T_c2b = T_g2b @ T_c2g

    theta = float(json_entry.get("turn_angle", 0.0))

    R_undo = rodrigues_axis_angle(rotation_axis, -theta)
    T_b2w0 = T_about_point(R_undo, rotation_center)

    # camera->0-angle world
    return T_b2w0 @ T_c2b

# -------------------- your matching function (unchanged) --------------------

def load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis, unmatched_scan_ids_path):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    (Matching uses your scan-(light)-(camera) scheme; not θ.)
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images)

    robot_poses, cam_c2w, scan_id, image_ids = [], [], [], []
    for fname, (img_id, c2w) in colmap_c2w_dict.items():

        idx = find_matching_entry(fname, scan_log)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot camera pose in 0-angle world (using current TURNTABLE_CENTER)
        entry = scan_log[idx]
        T = rotated_c2w(entry, R_c2g, t_c2g, rotation_center, rotation_axis)
        robot_poses.append(T)
        scan_id.append(idx)
        image_ids.append(img_id)

    save_unmatched_scan_ids(colmap_c2w_dict, scan_log, unmatched_scan_ids_path)

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, scan_id, image_ids


# -------------------- world->base estimation (your pipeline kept) --------------------

def estimate_world2base(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis, unmatched_scan_ids_path, solve_center_xy=True):
    """
    Returns T_BW: world->base Sim3 mapping COLMAP world to robot base.
    If solve_center_xy=True, first refines ROTATION_CENTER[0:2] from data.
    """
    # Now build your matched robot & colmap poses using the (possibly) updated center
    robot_T, cam_c2w, scan_id, image_ids = load_robot_poses_c2w0(scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis, unmatched_scan_ids_path)

    # collect centres
    cam_centres_base, cam_centres_world = [], []
    for T_c2b, c2w in zip(robot_T, cam_c2w):
        cam_centres_base.append(T_c2b[:3, 3])
        cam_centres_world.append(c2w[:3, 3])
    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # single Sim3 world->base
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    T_BW = build_4x4(s * R, t)

    # debug
    print(f"[umeyama] scale: {s:.6f}")
    print(f"[umeyama] rotation matrix:\n{R}")
    print(f"[umeyama] translation:\n{t}")
    W_h = np.hstack([cam_centres_world, np.ones((len(cam_centres_world), 1))])
    W2B = (T_BW @ W_h.T).T[:, :3]
    mean_err = np.mean(np.linalg.norm(W2B - cam_centres_base, axis=1))
    print(f"[umeyama] mean error: {mean_err:.6f} m  (N={len(cam_centres_world)})")

    return T_BW, cam_c2w, robot_T, s, scan_id, image_ids


def filter_high_error_images(cam_c2w, robot_T, scan_id, image_ids, T_BW, threshold_mm=16.0):
    """Remove images whose per-image translation error exceeds threshold_mm.

    These images are treated exactly like never-registered scans in downstream
    outputs (rotated_camera.json, observations_structured.npz). Their excluded
    records are returned so callers can persist them to JSON.
    """
    threshold_m = threshold_mm / 1000.0
    keep_cam, keep_robot, keep_scan, keep_ids = [], [], [], []
    excluded = []

    for C2W, R_pose, sid, iid in zip(cam_c2w, robot_T, scan_id, image_ids):
        C2B = T_BW @ np.asarray(C2W)
        err_m = float(np.linalg.norm(C2B[:3, 3] - np.asarray(R_pose)[:3, 3]))
        if err_m <= threshold_m:
            keep_cam.append(C2W)
            keep_robot.append(R_pose)
            keep_scan.append(sid)
            keep_ids.append(iid)
        else:
            excluded.append({
                'colmap_image_id': int(iid),
                'scan_log_index': int(sid),
                'translation_error_m': err_m,
                'translation_error_mm': err_m * 1000.0,
            })

    n_excl = len(excluded)
    n_total = len(cam_c2w)
    print(f"\n[filter] Removing {n_excl}/{n_total} images with per-image "
          f"translation error > {threshold_mm:.1f} mm")
    for ex in excluded[:50]:
        print(f"  excluded: scan_log_index={ex['scan_log_index']} "
              f"(colmap img_id={ex['colmap_image_id']}) "
              f"error={ex['translation_error_mm']:.1f} mm")
    if n_excl > 50:
        print(f"  ... and {n_excl - 50} more")

    return keep_cam, keep_robot, keep_scan, keep_ids, excluded

# -------------------- mesh transform & camera log save (kept) --------------------

def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    mesh.transform(T_BW)
    if output_path is not None:
        ok = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not ok:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")
    return mesh

def process_pointcloud_to_base(pointcloud_path, T_BW, cfg, output_path=None, z_outlier_percentile=5.0, material_id=None):
    """
    Transform Colmap sparse pointcloud to base frame, crop by XY rectangle, remove Z outliers,
    and compute axis-aligned bounding box.
    
    Args:
        pointcloud_path: Path to point3d.ply file from Colmap
        T_BW: Transformation matrix from world to base frame
        cfg: Config dict containing mesh.rectangle parameters
        output_path: Path to save filtered pointcloud (optional)
        z_outlier_percentile: Percentage of points to remove as outliers from top and bottom (default 5%)
        material_id: Material ID to look up width/length from sample_size_json (optional)
    
    Returns:
        tuple: (filtered_pcd, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices, points_world_filtered)
            filtered_indices: numpy array of indices of filtered points in the original pointcloud
            points_world_filtered: (N, 3) array of filtered points in COLMAP world frame
    """
    import open3d as o3d
    
    # Load pointcloud
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load pointcloud from {pointcloud_path}")
    
    print(f"Loaded {len(pcd.points)} points from {pointcloud_path}")
    
    # SAVE WORLD COORDINATES BEFORE TRANSFORMATION
    points_world_original = np.asarray(pcd.points).copy()  # (N, 3) in COLMAP world frame
    
    # Transform to base frame
    pcd.transform(T_BW)
    print(f"Transformed pointcloud to base frame")
    
    # Get rectangle parameters from config
    center = np.array(cfg.mesh.rectangle.center)
    
    # Get width and length from sample_size_json if material_id is provided
    width = cfg.mesh.rectangle.width  # x direction (fallback)
    length = cfg.mesh.rectangle.length  # y direction (fallback)
    
    if material_id is not None and hasattr(cfg.mesh.rectangle, 'sample_size_json'):
        import json
        sample_size_json_path = cfg.mesh.rectangle.sample_size_json
        with open(sample_size_json_path, 'r') as f:
            sample_sizes_data = json.load(f)
        
        for entry in sample_sizes_data["sample_sizes"]:
            if entry["id_start"] <= material_id <= entry["id_end"]:
                width = entry["width"]
                length = entry["length"]
                print(f"Using sample size for material_id {material_id}: width={width}, length={length}")
                break
        else:
            print(f"Warning: material_id {material_id} not found in sample_size_json, using default width={width}, length={length}")
    
    # Calculate XY bounds
    x_min = center[0] - width / 2.0
    x_max = center[0] + width / 2.0
    y_min = center[1] - length / 2.0
    y_max = center[1] + length / 2.0
    
    print(f"Cropping rectangle: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    
    # Crop by XY coordinates
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # Track indices through the filtering process
    original_indices = np.arange(len(points))
    
    xy_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
              (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    
    points_cropped = points[xy_mask]
    colors_cropped = colors[xy_mask] if colors is not None else None
    indices_after_xy = original_indices[xy_mask]
    
    print(f"After XY cropping: {len(points_cropped)} points ({100*len(points_cropped)/len(points):.1f}%)")
    
    if len(points_cropped) == 0:
        raise ValueError("No points remain after XY cropping. Check rectangle parameters.")
    
    use_iqr = True
    z_values = points_cropped[:, 2]
    # Remove Z outliers
    if use_iqr:
        # IQR (Interquartile Range) method - adaptive outlier detection
        Q1 = np.percentile(z_values, 25)
        Q3 = np.percentile(z_values, 75)
        IQR = Q3 - Q1
        z_lower = Q1 - 1.5 * IQR
        z_upper = Q3 + 1.5 * IQR
    else:
        # Fixed percentile method
        z_lower = np.percentile(z_values, z_outlier_percentile)
        z_upper = np.percentile(z_values, 100 - z_outlier_percentile)
    
    z_mask = (z_values >= z_lower) & (z_values <= z_upper)
    filtered_percentage = 100 * np.sum(z_mask) / len(z_mask)
    print(f"Z filtering: keeping {np.sum(z_mask)} / {len(z_mask)} points ({filtered_percentage:.1f}%)")
    points_filtered = points_cropped[z_mask]
    colors_filtered = colors_cropped[z_mask] if colors_cropped is not None else None
    filtered_indices = indices_after_xy[z_mask]
    
    # APPLY SAME FILTERING TO WORLD COORDINATES - ensures matching indices
    points_world_filtered = points_world_original[filtered_indices]
    
    print(f"After Z outlier removal ({z_outlier_percentile}% top/bottom): {len(points_filtered)} points "
          f"({100*len(points_filtered)/len(points_cropped):.1f}% of cropped)")
    print(f"Z range: [{z_lower:.6f}, {z_upper:.6f}]")
    
    # Create filtered pointcloud
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)
    if colors_filtered is not None:
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    # Compute axis-aligned bounding box
    bbox_min = points_filtered.min(axis=0)
    bbox_max = points_filtered.max(axis=0)
    # Use config center for x,y and mean of points for z
    bbox_center = np.array([center[0], center[1], points_filtered[:, 2].mean()])
    bbox_size = bbox_max - bbox_min
    
    print(f"\n{'='*60}")
    print(f"Axis-Aligned Bounding Box (Base Frame):")
    print(f"{'='*60}")
    print(f"Min:    [{bbox_min[0]:9.6f}, {bbox_min[1]:9.6f}, {bbox_min[2]:9.6f}]")
    print(f"Max:    [{bbox_max[0]:9.6f}, {bbox_max[1]:9.6f}, {bbox_max[2]:9.6f}]")
    print(f"Center: [{bbox_center[0]:9.6f}, {bbox_center[1]:9.6f}, {bbox_center[2]:9.6f}]")
    print(f"Size:   [{bbox_size[0]:9.6f}, {bbox_size[1]:9.6f}, {bbox_size[2]:9.6f}]")
    print(f"{'='*60}\n")
    
    # Save filtered pointcloud if output path specified
    if output_path is not None:
        ok = o3d.io.write_point_cloud(str(output_path), pcd_filtered)
        if not ok:
            raise RuntimeError(f"Failed to save pointcloud to {output_path}")
        print(f"Filtered pointcloud saved to: {output_path}")
    
    return pcd_filtered, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices, points_world_filtered

def save_camera_log_from_colmap(camera_c2w, robot_T, s, T_BW, scan_id, output_path):
    """
    Save camera log and compute alignment errors between COLMAP and robot poses.
    
    Args:
        camera_c2w: List of COLMAP camera-to-world poses
        robot_T: List of robot camera poses in base frame
        T_BW: Transformation matrix from COLMAP world to robot base
        scan_id: List of scan IDs
        output_path: Path to save camera log JSON
    """
    camera_log = []
    translation_errors = []
    angular_errors = []
    
    
    for i, C2W in enumerate(camera_c2w):
        # Transform COLMAP pose to base frame
        C2B = T_BW @ np.asarray(C2W)
        position = C2B[:3, 3] * 1000.0  # m->mm to match scan_log
        rotation_matrix = C2B[:3, :3]
        
        camera_entry = {
            "overall_id": scan_id[i],
            "camera_id": scan_id[i],
            "position": position.tolist(),
            "rotation_matrix": rotation_matrix.tolist()
        }
        camera_log.append(camera_entry)
        
        # Compute errors between C2B and robot_T[i]
        robot_pose = np.asarray(robot_T[i])
        
        # Translation error (in meters)
        trans_error = np.linalg.norm(C2B[:3, 3] - robot_pose[:3, 3])
        translation_errors.append(trans_error)
        
        # Angular error using rotation matrix difference
        # Method: compute the relative rotation R_rel = R_colmap^T @ R_robot
        # Then extract the angle from the trace of R_rel
        R_colmap = C2B[:3, :3]
        R_robot = robot_pose[:3, :3]
        R_colmap_rot = R_colmap / s  # now ~ proper rotation

        # Now compute relative rotation and angle
        R_rel = R_colmap_rot.T @ R_robot

        trace_val = np.trace(R_rel)
        cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        angular_error = np.rad2deg(np.arccos(cos_angle))
        angular_errors.append(angular_error)
    
    # Convert to numpy arrays
    translation_errors = np.array(translation_errors)
    angular_errors = np.array(angular_errors)
    
    # Compute statistics
    avg_trans_error = np.mean(translation_errors)
    avg_angular_error = np.mean(angular_errors)
    
    # Get top 5 maximum errors
    top10_trans_idx = np.argsort(translation_errors)[-10:][::-1]
    top10_angular_idx = np.argsort(angular_errors)[-10:][::-1]
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Camera Pose Alignment Analysis")
    print(f"{'='*60}")
    print(f"Number of camera poses: {len(camera_c2w)}")
    print(f"\nAverage translation error: {avg_trans_error:.6f} m ({avg_trans_error*1000:.3f} mm)")
    print(f"Average angular error: {avg_angular_error:.4f} degrees")
    
    print(f"\n{'='*60}")
    print(f"Top 10 Maximum Translation Errors:")
    print(f"{'='*60}")
    for rank, idx in enumerate(top10_trans_idx, 1):
        print(f"  {rank}. Index {idx:4d} (scan_id={scan_id[idx]:4d}): "
              f"{translation_errors[idx]:.6f} m ({translation_errors[idx]*1000:.3f} mm)")
    
    print(f"\n{'='*60}")
    print(f"Top 10 Maximum Angular Errors:")
    print(f"{'='*60}")
    for rank, idx in enumerate(top10_angular_idx, 1):
        print(f"  {rank}. Index {idx:4d} (scan_id={scan_id[idx]:4d}): "
              f"{angular_errors[idx]:.4f} degrees")
    print(f"{'='*60}\n")
    
    # Save camera log
    with open(output_path, 'w') as f:
        json.dump(camera_log, f, indent=2)
    print(f"Camera log saved to: {output_path}")
    print(f"Saved {len(camera_log)} camera poses")
    
    return camera_log

def save_points_pixel_data(pcd_filtered, filtered_indices, images, points3D, hdr_path, output_path):
    """
    Save point-ray observation data from COLMAP sparse reconstruction with HDR RGB values.
    
    Args:
        pcd_filtered: Open3D pointcloud (filtered and transformed to base frame)
        filtered_indices: numpy array of indices mapping filtered points to original COLMAP points
        images: dict of COLMAP Image objects (from read_images_binary/text)
        points3D: dict of COLMAP Point3D objects (from read_points3D_binary/text)
        hdr_path: Path to folder containing HDR images (filenames match COLMAP image names)
        output_path: Path to save the compressed .npz file
    
    Saves:
        observations.npz containing:
            - observations: (N, 10) array with [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
            - filtered_indices: array mapping to original COLMAP point IDs
        
        point_metadata.json in the same folder containing:
            - num_points: total unique 3D points
            - num_observations: total observations (points × cameras)
    """
    import cv2
    import os
    import json
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"Extracting Point-Ray Observations with HDR RGB")
    print(f"{'='*60}")
    
    # 1. Create mapping from PLY index to COLMAP point ID
    colmap_point_ids = list(points3D.keys())  # ordered list
    print(f"Total COLMAP points: {len(colmap_point_ids)}")
    print(f"Filtered points: {len(filtered_indices)}")
    
    # 2. Count total observations for filtered points
    total_observations = 0
    for filt_idx in filtered_indices:
        colmap_point_id = colmap_point_ids[filt_idx]
        point = points3D[colmap_point_id]
        total_observations += len(point.image_ids)
    
    print(f"Total observations to extract: {total_observations:,}")
    
    # 3. Allocate observations array
    observations = np.zeros((total_observations, 9), dtype=np.float32)
    
    # 4. Pre-load ALL HDR images into memory
    print("\nPre-loading HDR images...")
    image_cache = {}
    # Cache directory listing once — hdr_path is read-only during this loop
    # (was being re-scanned per image, O(images × listdir) NFS calls).
    all_hdr_files = os.listdir(hdr_path)
    for img_id, image in tqdm(images.items(), desc="Loading HDR images"):
        hdr_image_path = os.path.join(hdr_path, image.name)
        # Remove '_max' and everything after it, but keep .png extension
        # Find image file starting with 'scan-{img_id}'
        prefix = f'scan-{img_id}'
        matching_files = [f for f in all_hdr_files if f.startswith(prefix)]
        if matching_files:
            hdr_image_path = os.path.join(hdr_path, matching_files[0])
        hdr_img = cv2.imread(hdr_image_path, cv2.IMREAD_UNCHANGED)
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        image_cache[img_id] = hdr_img
    
    print(f"Loaded {len(image_cache)} HDR images")
    
    # 5. Build observation data using vectorized operations
    print("\nBuilding observation arrays...")
    
    # Pre-allocate lists for vectorized assembly
    xyz_list = []
    img_id_list = []
    pixel_xy_list = []
    point_indices_list = []  # Track which filtered point each observation belongs to
    
    # First pass: collect all metadata (fast, no RGB sampling yet)
    for i, filt_idx in enumerate(filtered_indices):
        colmap_point_id = colmap_point_ids[filt_idx]
        point = points3D[colmap_point_id]
        xyz = np.asarray(pcd_filtered.points[i])
        
        for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            if img_id not in images:
                continue
            
            image = images[img_id]
            pixel_xy = image.xys[point2D_idx]
            
            xyz_list.append(xyz)
            img_id_list.append(img_id)
            pixel_xy_list.append(pixel_xy)
            point_indices_list.append(i)
    
    # Convert to numpy arrays
    xyz_array = np.array(xyz_list, dtype=np.float32)  # (N, 3)
    img_id_array = np.array(img_id_list, dtype=np.int32)  # (N,)
    pixel_xy_array = np.array(pixel_xy_list, dtype=np.float32)  # (N, 2)
    point_id_array = np.array(point_indices_list, dtype=np.int32)  # (N,) - local point ID
    
    actual_observations = len(xyz_array)
    print(f"Collected {actual_observations:,} valid observations")
    
    # 6. Vectorized RGB sampling
    print("Sampling RGB values from HDR images...")
    rgb_array = np.zeros((actual_observations, 3), dtype=np.float32)
    
    # Group observations by image for efficient sampling
    from collections import defaultdict
    obs_by_image = defaultdict(list)
    for obs_idx, img_id in enumerate(img_id_array):
        obs_by_image[img_id].append(obs_idx)
    
    # Sample RGB for each image's observations in batch
    for img_id, obs_indices in tqdm(obs_by_image.items(), desc="Sampling RGB"):
        hdr_img = image_cache[img_id]
        obs_indices = np.array(obs_indices)
        
        # Get pixel coordinates for this image's observations
        pixels = pixel_xy_array[obs_indices]  # (M, 2)
        
        # Vectorized RGB sampling — NEAREST pixel (np.round + clip), the
        # convention the released dataset was built with; see sample_pixels_nearest.
        rgb_array[obs_indices] = sample_pixels_nearest(hdr_img, pixels[:, 0], pixels[:, 1])
    
    # 7. Assemble final observations array
    observations = np.column_stack([
        xyz_array,           # (N, 3) - x, y, z
        img_id_array.reshape(-1, 1).astype(np.float32),  # (N, 1) - image_id
        pixel_xy_array,      # (N, 2) - pixel_x, pixel_y
        rgb_array,           # (N, 3) - r, g, b
        point_id_array.reshape(-1, 1).astype(np.float32),  # (N, 1) - point_id
    ])  # Final shape: (N, 10)
    
    skipped_count = total_observations - actual_observations
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} observations (image not in images dict)")
    
    print(f"\nFinal observations extracted: {actual_observations:,}")
    print(f"Unique images loaded: {len(image_cache)}")
    
    # 8. Calculate storage size
    storage_size_mb = (observations.nbytes + filtered_indices.nbytes) / (1024 * 1024)
    print(f"Uncompressed size: {storage_size_mb:.2f} MB")
    
    # 9. Save point metadata JSON
    num_points = len(filtered_indices)  # Total unique 3D points
    num_observations = actual_observations  # Total observations (point-camera pairs)
    
    # Get the material folder (parent of output_path)
    material_folder = os.path.dirname(output_path)
    metadata_path = os.path.join(material_folder, 'point_metadata.json')
    
    point_metadata = {
        'num_points': int(num_points),
        'num_observations': int(num_observations),
        'observations_file': os.path.basename(output_path)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(point_metadata, f, indent=2)
    
    print(f"\nSaved point metadata to: {metadata_path}")
    print(f"  - num_points: {num_points:,} (unique 3D points)")
    print(f"  - num_observations: {num_observations:,} (point-camera pairs)")
    
    # 10. Save compressed observations
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(output_path,
                       observations=observations,
                       filtered_indices=filtered_indices)
    
    # Check actual file size
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Compressed file size: {actual_size_mb:.2f} MB")
    print(f"Compression ratio: {storage_size_mb/actual_size_mb:.2f}x")
    print(f"{'='*60}\n")

def _process_camera_batch(args):
    """
    Worker function to process a batch of cameras.
    Returns list of observation arrays for the batch.
    """
    import cv2
    img_ids, images, cameras, points_world, points_base, hdr_path = args
    batch_observations = []

    # hdr_path is populated once by debayer_mosaic_hdr before this worker runs,
    # then read-only for the rest of shape_matching. Cache once to avoid
    # O(batch × N) redundant NFS readdir calls (was the top throughput tax).
    all_hdr_files = os.listdir(hdr_path)

    for img_id in img_ids:
        image = images[img_id]
        camera = cameras[image.camera_id]

        # Load HDR image
        hdr_image_path = os.path.join(hdr_path, image.name)
        prefix = f'scan-{(img_id-1):04d}' # -1 because colmap indices start from 1
        matching_files = [f for f in all_hdr_files if f.startswith(prefix)]
        if matching_files:
            hdr_image_path = os.path.join(hdr_path, matching_files[0])
        else:
            raise ValueError(f"No matching file found for {prefix}")
        hdr_img = cv2.imread(hdr_image_path, cv2.IMREAD_UNCHANGED)
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        
        # Get camera extrinsics (world to camera)
        R_cam = qvec2rotmat(image.qvec)  # 3x3
        t_cam = image.tvec  # 3,
        
        # Transform all points to camera frame
        points_cam = (R_cam @ points_world.T).T + t_cam  # (N, 3)
        
        # Filter points in front of camera
        valid_mask = points_cam[:, 2] > 0
        
        if not valid_mask.any():
            continue
        
        # Get camera intrinsics for SIMPLE_RADIAL
        f = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
        k = camera.params[3] if len(camera.params) > 3 else 0.0
        
        # Project to image plane with distortion (vectorized)
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]
        
        # Normalized coordinates
        x_norm = X / Z
        y_norm = Y / Z
        
        # Apply radial distortion
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k * r2
        
        # Distorted pixel coordinates
        pixels_x = f * distortion * x_norm + cx
        pixels_y = f * distortion * y_norm + cy
        
        # Check image bounds
        valid_mask &= (pixels_x >= 0) & (pixels_x < camera.width)
        valid_mask &= (pixels_y >= 0) & (pixels_y < camera.height)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Sample RGB for valid pixels (vectorized) — NEAREST pixel (np.round +
        # clip), the convention the released dataset was built with; see
        # sample_pixels_nearest. Not bilinear.
        rgb_values = sample_pixels_nearest(hdr_img, pixels_x[valid_indices], pixels_y[valid_indices])  # (M, 3)
        
        # Build observations for this camera (vectorized - NO FOR LOOP!)
        camera_observations = np.column_stack([
            points_base[valid_indices],           # xyz in base frame (M, 3)
            np.full(len(valid_indices), float(img_id)),  # image_id (M,)
            pixels_x[valid_indices],              # pixel x (M,)
            pixels_y[valid_indices],              # pixel y (M,)
            rgb_values,                           # rgb (M, 3)
            valid_indices.astype(np.float32)      # point_id (M,) - local point index
        ])  # Shape: (M, 10)
        batch_observations.append(camera_observations)
        
        # Debug visualization
        visualize = False
        if visualize:
            import open3d as o3d
            first_img_id = img_id
            xyz = points_base[valid_indices]
            rgbs = np.clip(rgb_values.astype(np.float32), 0, 65535) / 65535.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
            _dbg = os.environ.get("ROBOCLOTH_DEBUG_DIR")  # opt-in debug dump
            if _dbg:
                os.makedirs(_dbg, exist_ok=True)
                o3d.io.write_point_cloud(f"{_dbg}/debug_points_base_img{first_img_id}.ply", pcd)
                print(f"Saved debug point cloud to {_dbg}/debug_points_base_img{first_img_id}.ply ({len(xyz)} points)")
            
            xyz = points_world[valid_indices]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgbs)
            if _dbg:
                o3d.io.write_point_cloud(f"{_dbg}/debug_points_world_img{first_img_id}.ply", pcd)
                print(f"Saved debug point cloud to {_dbg}/debug_points_world_img{first_img_id}.ply ({len(xyz)} points)")
    
    
    return batch_observations


def build_and_save_structured_observations(
    observations,
    unique_pids,
    unique_xyz,
    scan_log_path,
    output_path,
    verbose=True,
):
    """
    Build the dense (K, V, 3) RGB matrix from an in-memory observations array
    and save it as observations_structured.npz.

    This mirrors scripts/reformat_data/convert_single.py exactly so the inline
    output is bit-for-bit identical to running convert_single.py on the chunks.

    Args:
        observations: (M, 10) float64 array
            columns [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
            image_id is 1-based COLMAP ID; point_id is 0-based local index.
        unique_pids: (V,) int array of unique point ids that were observed
            (the same array stored in point_positions.npz['point_ids']).
        unique_xyz: (V, 3) float32 array of point positions, same order as unique_pids
            (the same array stored in point_positions.npz['positions']).
        scan_log_path: path to scan_log.json (provides K, cam_pos, light_pos).
        output_path: where to write observations_structured.npz.
        verbose: if True, print stats.
    """
    import time

    t0 = time.time()
    V = len(unique_pids)

    # Vectorised point_id -> dense-row lookup (matches convert_single.py:55-58)
    point_ids_arr = np.asarray(unique_pids)
    max_pid = int(point_ids_arr.max())
    pid_lookup = np.full(max_pid + 1, -1, dtype=np.int32)
    pid_lookup[point_ids_arr.astype(np.int64)] = np.arange(V, dtype=np.int32)

    # Load scan_log for cam / light positions and K (matches convert_single.py:60-66)
    with open(scan_log_path) as f:
        scan_log = json.load(f)
    K = len(scan_log)
    cam_pos = np.array([e['position'] for e in scan_log], dtype=np.float32)
    light_pos = np.array([e['position_light'] for e in scan_log], dtype=np.float32)

    if verbose:
        mem_gb = K * V * 3 * 2 / 1e9
        print(f"\n{'='*60}")
        print(f"Building structured observations: V={V:,} points, K={K} images")
        print(f"  Dense rgbs will be ({K}, {V}, 3) uint16 = {mem_gb:.2f} GB")
        print(f"{'='*60}")

    # Allocate dense RGB array
    rgbs = np.zeros((K, V, 3), dtype=np.uint16)

    # Fill from in-memory observations (matches convert_single.py chunk loop:88-113)
    img_ids = observations[:, 3].astype(np.int64) - 1          # 1-based -> 0-based
    pt_ids_raw = observations[:, 9].astype(np.int64)
    rgb_vals = np.clip(observations[:, 6:9], 0, 65535).astype(np.uint16)

    safe_pt = np.clip(pt_ids_raw, 0, max_pid)
    pt_indices = pid_lookup[safe_pt]
    pt_indices[pt_ids_raw < 0] = -1
    pt_indices[pt_ids_raw > max_pid] = -1

    valid = (img_ids >= 0) & (img_ids < K) & (pt_indices >= 0)
    vi = img_ids[valid]
    vp = pt_indices[valid]
    vr = rgb_vals[valid]

    # Count duplicates (same image+point written twice -> should be 0)
    already_set = rgbs[vi, vp].sum(axis=1) > 0
    total_dupes = int(already_set.sum())

    rgbs[vi, vp] = vr
    total_filled = int(valid.sum())

    # Stats
    non_zero = (rgbs.sum(axis=2) > 0).sum()
    density = non_zero / (K * V) * 100
    if verbose:
        print(f"  Total obs:        {len(observations):,}")
        print(f"  Filled cells:     {total_filled:,}")
        print(f"  Duplicate writes: {total_dupes:,}")
        print(f"  Non-zero cells:   {non_zero:,} / {K * V:,}  ({density:.1f}%)")

    # Save (mirrors convert_single.py:136-143)
    np.savez(
        output_path,
        xyz=unique_xyz.astype(np.float32),
        point_ids=point_ids_arr.astype(np.int32),
        rgbs=rgbs,
        cam_pos=cam_pos,
        light_pos=light_pos,
    )
    file_size_gb = os.path.getsize(output_path) / 1e9
    if verbose:
        print(f"  Saved to: {output_path} ({file_size_gb:.2f} GB)")
        print(f"  Total time: {time.time() - t0:.1f}s")
        print(f"{'='*60}\n")

    return {
        'V': V, 'K': K,
        'total_obs': int(len(observations)),
        'filled': total_filled,
        'dupes': total_dupes,
        'density_pct': float(density),
        'file_size_gb': float(file_size_gb),
    }


def save_points_pixel_data_reprojection(pcd_filtered, points_world_filtered, cameras, images, hdr_path, material_folder, num_workers=32):
    """
    Reproject filtered 3D points to all camera views with SIMPLE_RADIAL distortion.
    Uses multiprocessing to speed up processing.

    Args:
        pcd_filtered: Open3D pointcloud (filtered and transformed to base frame) - used for getting base frame coordinates
        points_world_filtered: (N, 3) array of filtered points in COLMAP world frame (same order as pcd_filtered)
        cameras: dict of COLMAP Camera objects (from read_cameras_binary/text)
        images: dict of COLMAP Image objects (from read_images_binary/text)
        hdr_path: Path to folder containing HDR images
        material_folder: Path to the material's folder (where outputs are written)
        num_workers: Number of parallel workers

    Saves:
        - observations_structured.npz in material_folder (dense (K, V, 3) RGB format)
        - point_positions.npz in material_folder (unique point_id -> xyz)
        - point_metadata.json in material_folder (num_points and num_observations)
    """
    import cv2
    import os
    from tqdm import tqdm
    from multiprocessing import Pool

    print(f"\n{'='*60}")
    print(f"Reprojecting Points to All Camera Views (Multiprocessing)")
    print(f"{'='*60}")

    # Get points_base and points_world with guaranteed matching indices
    points_base = np.asarray(pcd_filtered.points).astype(np.float32)  # (N, 3) in base frame
    points_world = points_world_filtered.astype(np.float32)  # (N, 3) in COLMAP world frame - SAME ORDER!

    num_points = len(points_world)

    print(f"Filtered points to reproject: {num_points:,}")
    print(f"Total cameras: {len(images)}")
    print(f"Using {num_workers} parallel workers")

    # Split camera IDs into batches for better progress tracking
    img_ids = list(images.keys())
    camera_batch_size = max(1, len(img_ids) // num_workers)  # Process cameras in batches based on workers
    batches = [img_ids[i:i+camera_batch_size] for i in range(0, len(img_ids), camera_batch_size)]

    print(f"Split into {len(batches)} batches (~{camera_batch_size} cameras each)")

    # Prepare arguments for each batch
    batch_args = [
        (batch, images, cameras, points_world, points_base, hdr_path)
        for batch in batches
    ]

    # Process batches in parallel
    print("\nProcessing cameras in parallel...")
    all_observations = []

    with Pool(processes=num_workers) as pool:
         results = list(tqdm(
            pool.imap(_process_camera_batch, batch_args),
            total=len(batch_args),
            desc="Processing batches"
        ))

    # Combine results from all batches (concatenate numpy arrays)
    for batch_result in results:
        all_observations.extend(batch_result)

    # Convert to numpy array by stacking all observation arrays
    if all_observations:
        observations = np.vstack(all_observations)  # (M, 10)

    total_obs = len(observations)

    print(f"\nTotal valid observations: {total_obs:,}")
    print(f"Average observations per point: {total_obs / num_points:.2f}")
    print(f"Average observations per camera: {total_obs / len(images):.2f}")

    # Calculate storage size
    storage_size_mb = observations.nbytes / (1024 * 1024)
    print(f"Uncompressed size: {storage_size_mb:.2f} MB")

    # Extract unique point_id -> xyz mapping from observations
    import json
    point_ids = observations[:, 9].astype(np.int64)
    unique_pids, first_idx = np.unique(point_ids, return_index=True)
    unique_xyz = observations[first_idx, :3].astype(np.float32)

    # Save as (num_unique, 4): [point_id, x, y, z]
    positions_path = os.path.join(material_folder, 'point_positions.npz')
    np.savez(positions_path, point_ids=unique_pids, positions=unique_xyz)
    print(f"\nSaved point positions to: {positions_path}")
    print(f"  Unique points: {len(unique_pids)} / {num_points}")

    # Save point metadata JSON
    metadata_path = os.path.join(material_folder, 'point_metadata.json')

    point_metadata = {
        'num_points': int(num_points),
        'num_observations': int(total_obs),
    }

    with open(metadata_path, 'w') as f:
        json.dump(point_metadata, f, indent=2)

    print(f"\nSaved point metadata to: {metadata_path}")
    print(f"  num_points: {num_points}")
    print(f"  num_observations: {total_obs}")

    # Build observations_structured.npz inline from the in-memory observations array.
    # This is the dense (K, V, 3) format used downstream by training.
    structured_path = os.path.join(material_folder, 'observations_structured.npz')
    scan_log_path = os.path.join(material_folder, 'scan_log.json')
    if os.path.exists(scan_log_path):
        build_and_save_structured_observations(
            observations=observations,
            unique_pids=unique_pids,
            unique_xyz=unique_xyz,
            scan_log_path=scan_log_path,
            output_path=structured_path,
            verbose=True,
        )
    else:
        print(f"WARNING: scan_log.json not found at {scan_log_path}; "
              f"skipping observations_structured.npz")


def _debayer_single_image(args):
    """Worker function to debayer a single mosaic HDR image."""
    import cv2
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    
    # White balance parameters (same as capture pipeline)
    QE_R, QE_G, QE_B = 1.58056, 1, 1.06588
    
    src_path, dst_path = args
    
    # Read 16-bit PNG mosaic image
    raw16 = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if raw16 is None:
        return None
    
    # Debayer using Menon2007 method with RGGB pattern
    rgb = np.clip(demosaicing_CFA_Bayer_Menon2007(raw16.astype(np.float32), "RGGB"), 0, 65535).astype(np.uint16)
    
    # Convert RGB to BGR for OpenCV
    img16 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Apply white balance correction (same as capture pipeline)
    imgf = img16.astype(np.float32)
    imgf[..., 0] *= QE_R
    imgf[..., 1] *= QE_G
    imgf[..., 2] *= QE_B
    np.clip(imgf, 0, 65535, out=imgf)
    img16 = imgf.astype(np.uint16)
    
    # Save the debayered image. cv2.imwrite returns False on disk-full or
    # other I/O failure; treat that as a failed write so the caller can detect
    # partial output instead of silently producing an incomplete hdr/ folder.
    ok = cv2.imwrite(str(dst_path), img16)
    if not ok:
        try:
            Path(dst_path).unlink(missing_ok=True)
        except Exception:
            pass
        return None

    return dst_path

def debayer_mosaic_hdr(mosaic_hdr_path, hdr_path, num_workers=32):
    """
    Debayer all mosaic HDR images from mosaic_hdr_path folder and save to hdr_path folder.

    Workers stage their debayered PNGs to /mnt/data/shape_matching_tmp first
    (separate filesystem from the small /mnt/colmap_tmp loop device used by
    COLMAP — avoids cross-pipeline contention), then a single process moves
    them to hdr_path. Writing ~500 parallel O_CREATs into an NFS directory was
    previously serialized by the directory's i_rwsem, causing every worker to
    pile up on rwsem_down_write_slowpath; staging locally avoids that.

    Args:
        mosaic_hdr_path: Path to folder containing original mosaic HDR images (16-bit PNG)
        hdr_path: Path to output folder for debayered images
        num_workers: Number of parallel workers for multiprocessing
    """
    import cv2
    import shutil
    from multiprocessing import Pool
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"Debayering Mosaic HDR Images")
    print(f"{'='*60}")

    mosaic_folder = Path(mosaic_hdr_path)
    output_folder = Path(hdr_path)

    # Get all PNG files from mosaic folder
    png_files = sorted(mosaic_folder.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {mosaic_hdr_path}")
        return

    print(f"Found {len(png_files)} mosaic images to debayer")
    print(f"Output folder: {hdr_path}")
    print(f"Using {num_workers} parallel workers")

    # Stage outputs on local scratch if available — avoids NFS directory
    # rwsem contention from num_workers parallel file creates. Use a
    # dedicated dir on /mnt/data (~3 TB) instead of /mnt/colmap_tmp (93 GB
    # loop device shared with COLMAP) so debayer can't be starved when
    # COLMAP is running.
    local_scratch_root = Path(os.environ.get("RECON_TMP", "/tmp/robocloth_debayer_tmp"))
    use_local = local_scratch_root.is_dir()
    if use_local:
        staging_folder = local_scratch_root / f"debayer_{output_folder.parent.name}_{os.getpid()}"
        if staging_folder.exists():
            shutil.rmtree(staging_folder, ignore_errors=True)
        staging_folder.mkdir(parents=True, exist_ok=True)
        write_folder = staging_folder
        print(f"Staging debayered outputs in local: {write_folder}")
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        write_folder = output_folder

    # Prepare arguments for each image (source path, destination path with same filename)
    args_list = [
        (src_path, write_folder / src_path.name)
        for src_path in png_files
    ]

    # Process images in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_debayer_single_image, args_list),
            total=len(args_list),
            desc="Debayering images"
        ))

    # Count successful conversions
    successful = sum(1 for r in results if r is not None)
    print(f"\nSuccessfully debayered {successful}/{len(png_files)} images")

    # Hard-fail on any partial debayer. Previously the moved-but-incomplete
    # hdr/ folder would satisfy the existence check on retry and produce
    # nondeterministic "No matching file found for scan-XXXX" errors
    # downstream. Better to abort here and let the caller retry from a clean
    # state.
    if successful != len(png_files):
        if use_local:
            shutil.rmtree(write_folder, ignore_errors=True)
        raise RuntimeError(
            f"Debayer wrote only {successful}/{len(png_files)} images "
            f"(staging dir: {write_folder}). Disk-full or worker crash; "
            f"refusing to leave a partial hdr/ folder."
        )

    # If we staged locally, move results to the NFS output folder in a single
    # process (sequential creates → no rwsem pile-up), then clean up.
    if use_local:
        output_folder.mkdir(parents=True, exist_ok=True)
        staged_files = sorted(write_folder.glob("*.png"))
        print(f"Moving {len(staged_files)} files from local staging → {output_folder}")
        for f in tqdm(staged_files, desc="Moving to NFS"):
            shutil.move(str(f), str(output_folder / f.name))
        shutil.rmtree(write_folder, ignore_errors=True)

    print(f"{'='*60}\n")

# -------------------- main --------------------

def main_process(cfg, cameras, images, points3D=None, scan_log_path=None, mesh_path=None, camera_log_path=None, hdr_path=None, mosaic_hdr_path=None, pointcloud_path=None, bbox_json_path=None, unmatched_scan_ids_path=None, excluded_scan_ids_path=None, error_threshold_mm=16.0, z_outlier_percentile=5.0, num_workers=32):
    # Extract parameters from config
    R_c2g = np.array(cfg.camera.R_c2g)
    t_c2g = np.array(cfg.camera.t_c2g)
    rotation_center = np.array(cfg.emitter.turntable.center)
    rotation_axis = np.array(cfg.emitter.turntable.axis)

    T_BW, cam_c2w, robot_T, s, scan_id, image_ids = estimate_world2base(
        scan_log_path, images, R_c2g, t_c2g, rotation_center, rotation_axis,
        unmatched_scan_ids_path)

    # Filter out high-error images. They are treated identically to never-registered
    # scans: dropped from rotated_camera.json and reprojection (so their rows in
    # observations_structured.npz stay zero).
    cam_c2w, robot_T, scan_id, image_ids, excluded = filter_high_error_images(
        cam_c2w, robot_T, scan_id, image_ids, T_BW, threshold_mm=error_threshold_mm)

    if excluded_scan_ids_path is not None:
        with open(excluded_scan_ids_path, 'w') as f:
            json.dump(excluded, f, indent=2)
        print(f"Saved {len(excluded)} excluded (high-error) records to: {excluded_scan_ids_path}")

    # Drop excluded images from COLMAP dict before reprojection.
    excluded_img_ids = {ex['colmap_image_id'] for ex in excluded}
    images = {k: v for k, v in images.items() if k not in excluded_img_ids}

    material_id = int(Path(camera_log_path).parent.name)
    material_folder = str(Path(camera_log_path).parent)

    if mesh_path is not None:
        transform_mesh_to_base(mesh_path, T_BW, output_path=str(Path(mesh_path).with_name(Path(mesh_path).stem + "_transformed.ply")))

    # Always re-debayer from hdr_raw/. The previous count-based skip let a
    # stale hdr/ from a pre-recapture run survive when hdr_raw/ was
    # re-captured, silently feeding wrong RGBs into observations_structured.npz
    # (see Dataset_Nov11 mats 243-246, 336-339, 343-346, 349-351 corruption,
    # 2026-05-22). Force-remove any existing hdr/ first so the debayer step
    # always runs from the current hdr_raw/.
    if mosaic_hdr_path is not None and Path(mosaic_hdr_path).is_dir():
        if Path(hdr_path).is_dir():
            shutil.rmtree(hdr_path)
        debayer_mosaic_hdr(mosaic_hdr_path, hdr_path, num_workers=num_workers)
    # Process pointcloud if path provided
    if pointcloud_path is not None:
        output_pcd_path = str(Path(pointcloud_path).with_name(Path(pointcloud_path).stem + "_transformed_filtered.ply"))
        pcd_filtered, bbox_min, bbox_max, bbox_center, bbox_size, filtered_indices, points_world_filtered = process_pointcloud_to_base(
            pointcloud_path, T_BW, cfg, output_path=output_pcd_path, z_outlier_percentile=z_outlier_percentile,
            material_id=material_id,
        )

        save_points_pixel_data_reprojection(pcd_filtered, points_world_filtered, cameras, images, hdr_path, material_folder, num_workers=num_workers)

        # Save bounding box info to a JSON file
        bbox_info = {
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_center": bbox_center.tolist(),
            "bbox_size": bbox_size.tolist(),
            "num_points": len(pcd_filtered.points)
        }
        with open(bbox_json_path, 'w') as f:
            json.dump(bbox_info, f, indent=2)
        print(f"Bounding box info saved to: {bbox_json_path}")

    save_camera_log_from_colmap(cam_c2w, robot_T, s, T_BW, scan_id, camera_log_path)

@hydra.main(version_base=None, config_path="../configs/renderer", config_name="reconstruction")
def main(cfg: DictConfig) -> None:
    # Get parameters from Hydra config (can be overridden via command line)
    folder_path = cfg.shape_matching.folder_path
    num_workers = cfg.shape_matching.num_workers
    z_outlier_percentile = cfg.shape_matching.z_outlier_percentile
    # Threshold on per-image translation error; images above this are dropped from
    # rotated_camera.json and reprojection (treated like never-registered scans).
    error_threshold_mm = getattr(cfg.shape_matching, 'error_threshold_mm', 16.0)

    # Build paths from folder_path
    scan_log_path = os.path.join(folder_path, "scan_log.json")
    mesh_path = None  # Optional mesh transformation
    model_path = os.path.join(folder_path, "sparse")
    pointcloud_path = os.path.join(model_path, "points3D.ply")
    points3D_path = os.path.join(model_path, "points3D.bin")
    mosaic_hdr_path = os.path.join(folder_path, "hdr_raw")
    hdr_path = os.path.join(folder_path, "hdr")
    camera_log_path = os.path.join(folder_path, "rotated_camera.json")
    bbox_json_path = os.path.join(folder_path, "bbox.json")
    unmatched_scan_ids_path = os.path.join(folder_path, "unmatched_scan_ids.json")
    excluded_scan_ids_path = os.path.join(folder_path, "excluded_high_error_scan_ids.json")
    print(f"Unmatched scan ids path: {unmatched_scan_ids_path}")
    print(f"Excluded (high-error) scan ids path: {excluded_scan_ids_path}")
    print(f"Processing folder: {folder_path}")
    print(f"Number of workers: {num_workers}")
    print(f"Z outlier percentile: {z_outlier_percentile}")
    print(f"Per-image error threshold: {error_threshold_mm:.1f} mm")

    # Load COLMAP data (cameras, images, points3D)
    cameras, images = read_model(model_path, ext=".bin")

    # Load points3D with format detection
    if detect_model_format(model_path, ".bin"):
        points3D = read_points3D_binary(points3D_path)
    elif detect_model_format(model_path, ".txt"):
        points3D = read_points3D_text(points3D_path)
    else:
        raise ValueError(f"Could not detect COLMAP model format in {model_path}")

    print(f"Loaded {len(points3D)} 3D points from COLMAP")

    main_process(cfg, cameras, images, points3D=points3D, scan_log_path=scan_log_path, mesh_path=mesh_path,
                 camera_log_path=camera_log_path, hdr_path=hdr_path, mosaic_hdr_path=mosaic_hdr_path,
                 pointcloud_path=pointcloud_path, bbox_json_path=bbox_json_path,
                 unmatched_scan_ids_path=unmatched_scan_ids_path,
                 excluded_scan_ids_path=excluded_scan_ids_path,
                 error_threshold_mm=error_threshold_mm,
                 z_outlier_percentile=z_outlier_percentile, num_workers=num_workers)
    print(f"Finished shape matching for {folder_path}")

if __name__ == "__main__":
    main()
