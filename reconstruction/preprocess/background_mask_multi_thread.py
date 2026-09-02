import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
NUM_WORKERS = 16
INITIAL_ANGLE_DEG = 0.0

# Project rotation center to plane z = -0.055 along the rotation axis
def project_center_to_plane(center, axis, plane_z):
    """
    Project a point along a direction vector to a plane z = plane_z.
    
    Args:
        center: (3,) array, the point to project
        axis: (3,) array, the direction vector
        plane_z: float, the z-coordinate of the target plane
    
    Returns:
        (3,) array, the projected point on the plane
    """
    center = np.array(center)
    axis = np.array(axis)
    
    # Normalize the axis vector
    axis_norm = axis / np.linalg.norm(axis)
    
    # Calculate parameter t for intersection with plane z = plane_z
    # Point on line: center + t * axis_norm
    # For intersection: center[2] + t * axis_norm[2] = plane_z
    t = (plane_z - center[2]) / axis_norm[2]
    
    # Calculate the projected point
    projected_center = center + t * axis_norm
    
    return projected_center

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io import load_camera_turntable_light_metadata

# ----------------------- math & geometry -----------------------

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(np.asarray(camera_info["rotation_matrix"], float),
                    np.asarray(camera_info["position"], float))
    c2g = build_4x4(np.asarray(R_c2g, float), np.asarray(t_c2g, float))
    c2w = g2w @ c2g
    return c2w

def _H_from_extrinsics(K, T_g2c, center_w):
    """Homography from plane z=z0 (world normal [0,0,1]) to image."""
    R = T_g2c[:3, :3]
    t = T_g2c[:3, 3]
    r1 = R[:, 0]
    r2 = R[:, 1]
    p0_c = R @ center_w + t  # plane origin in camera coords
    H = K @ np.column_stack([r1, r2, p0_c])  # 3x3
    return H

def _conic_from_H(H, radius):
    """C_img = H^{-T} diag(1,1,-r^2) H^{-1}; return A,B,C,D,E,F of Ax^2 + Bxy + Cy^2 + Dx + Ey + F."""
    Q_plane = np.diag([1.0, 1.0, -float(radius) ** 2])
    H_inv = np.linalg.inv(H)
    C_img = H_inv.T @ Q_plane @ H_inv
    A  = C_img[0, 0]
    B  = C_img[0, 1] + C_img[1, 0]
    Cc = C_img[1, 1]
    D  = C_img[0, 2] + C_img[2, 0]
    E  = C_img[1, 2] + C_img[2, 1]
    F  = C_img[2, 2]
    return A, B, Cc, D, E, F

def create_circular_mask_from_turntable(width, height, K, T_g2c, 
                                        turntable_center, turntable_radius):
    """
    Create a mask of the projected turntable (a circle in 3D) as an ellipse in image space.

    Args:
        width (int): Image width (pixels)
        height (int): Image height (pixels)
        K (np.ndarray): Intrinsic matrix (3x3)
        T_g2c (np.ndarray): World/Global to camera transform (4x4) with [R|t] in top-left 3x4
        turntable_center (tuple or array): (x, y, z) center in world coords
        turntable_radius (float): Circle radius in world units

    Returns:
        np.ndarray: Binary mask (height x width), dtype=uint8 (1 inside, 0 outside)
    """
    center_w = np.asarray(turntable_center, dtype=np.float64).reshape(3)
    H = _H_from_extrinsics(np.asarray(K, dtype=np.float64),
                           np.asarray(T_g2c, dtype=np.float64),
                           center_w)
    A, B, Cc, D, E, F = _conic_from_H(H, turntable_radius)

    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    X, Y = np.meshgrid(xs, ys)  # (H,W)

    val = (A * X * X) + (B * X * Y) + (Cc * Y * Y) + (D * X) + (E * Y) + F
    mask = (val <= 0).astype(np.uint8)
    return mask

def create_rotated_rectangle_mask_closedform(
    width, height,
    K, T_g2c,
    turntable_center,              # (x, y, z) world coords
    rect_center,                   # (x, y, z) world coords (on XY plane)
    rect_size,                     # (w, h) in world units
    angle_deg                      # CCW rotation about +Z through turntable_center
):
    """
    Create a binary mask (H, W) for a rectangle on the XY plane using a closed-form
    projective test (no polygon filling). The rectangle is first defined axis-aligned
    by rect_center_w & rect_size, then rigidly rotated CCW by `angle_deg` about +Z
    through the turntable_center. The mask is the intersection of 4 half-planes
    obtained by mapping the rectangle edges with the dual homography H^{-T}.

    Args:
        width, height: output mask size in pixels
        K: (3,3) intrinsics
        T_g2c: (4,4) world->camera transform
        turntable_center: (3,) world coords
        rect_center: (3,) world coords (pre-rotation)
        rect_size: (w, h) in world units
        angle_deg: float, CCW

    Returns:
        np.uint8 mask of shape (height, width), values {0,1}
    """
    K = np.asarray(K, dtype=np.float64)
    T_g2c = np.asarray(T_g2c, dtype=np.float64)
    tt = np.asarray(turntable_center, dtype=np.float64).reshape(3)
    rc = np.asarray(rect_center, dtype=np.float64).reshape(3)
    rw, rh = float(rect_size[0]), float(rect_size[1])

    # ----- 1) Build homography H for plane Z = Zp (the XY plane at Zp)
    Zp = rc[2]  # plane height (the rectangle lies on this plane)
    R = T_g2c[:3, :3]
    t = T_g2c[:3, 3]
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    p0 = r3 * Zp + t
    H = K @ np.column_stack((r1, r2, p0))             # (3x3) plane->image
    H_invT = np.linalg.inv(H).T                       # dual homography for lines

    # ----- 2) Axis-aligned rectangle corners in the plane, then rotate about tt
    dx, dy = rw * 0.5, rh * 0.5
    # CCW order in plane coordinates
    base_corners = np.array([
        [rc[0] - dx, rc[1] - dy, 1.0],
        [rc[0] + dx, rc[1] - dy, 1.0],
        [rc[0] + dx, rc[1] + dy, 1.0],
        [rc[0] - dx, rc[1] + dy, 1.0],
    ], dtype=np.float64)

    # Rotation about any axis through ROTATION_CENTER
    th = np.deg2rad(angle_deg)
    # Default to Z-axis rotation if no axis is specified
    rotation_axis = np.array(ROTATION_AXIS, dtype=np.float64)
    
    # Use Rodrigues' rotation formula for arbitrary axis
    n = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=np.float64)
    I = np.eye(3)
    c, s = np.cos(th), np.sin(th)
    Rz2 = I + s * K + (1.0 - c) * (K @ K)

    # Apply rotation in the plane (homogeneous points with last coord 1)
    tt_xy1 = np.array([tt[0], tt[1], 1.0], dtype=np.float64)
    corners_plane = ((Rz2 @ (base_corners.T - tt_xy1[:, None])) + tt_xy1[:, None]).T  # (4,3)

    # ----- 3) Build the 4 plane lines (each edge) and map them to image lines
    # Line from homogeneous points p and q: l = p x q
    def cross2(p, q):
        return np.array([
            p[1]*q[2] - p[2]*q[1],
            p[2]*q[0] - p[0]*q[2],
            p[0]*q[1] - p[1]*q[0],
        ], dtype=np.float64)

    lines_plane = []
    for i in range(4):
        p = corners_plane[i]
        q = corners_plane[(i + 1) % 4]
        l = cross2(p, q)                 # plane line (a x + b y + c = 0)
        lines_plane.append(l / (np.linalg.norm(l[:2]) + 1e-15))
    lines_plane = np.stack(lines_plane, axis=0)  # (4,3)

    # Map plane lines to image lines: l' ~ H^{-T} l
    lines_img = (H_invT @ lines_plane.T).T
    # Normalize for numerical stability
    lines_img = lines_img / (np.linalg.norm(lines_img[:, :2], axis=1, keepdims=True) + 1e-15)  # (4,3)

    # Determine consistent inequality direction.
    # Project plane centroid, then enforce "inside" as having non-negative dot.
    centroid_plane = np.mean(corners_plane, axis=0)  # (3,)
    centroid_img_h = H @ centroid_plane
    centroid_img = centroid_img_h[:2] / (centroid_img_h[2] + 1e-15)
    centroid_h = np.array([centroid_img[0], centroid_img[1], 1.0], dtype=np.float64)
    sgn = np.sign(lines_img @ centroid_h)  # (4,)
    sgn[sgn == 0] = 1.0
    lines_img *= sgn[:, None]  # flip if needed so centroid is inside (>=0)

    # ----- 4) Evaluate line inequalities at pixel centers (vectorized)
    xs = (np.arange(width, dtype=np.float64) + 0.5)
    ys = (np.arange(height, dtype=np.float64) + 0.5)
    X, Y = np.meshgrid(xs, ys)   # (H,W)
    ones = np.ones_like(X)

    # For each line a*u + b*v + c >= 0
    a = lines_img[:, 0][:, None, None]   # (4,1,1)
    b = lines_img[:, 1][:, None, None]   # (4,1,1)
    cst = lines_img[:, 2][:, None, None] # (4,1,1)

    vals = a * X[None, ...] + b * Y[None, ...] + cst   # (4,H,W)
    inside = np.all(vals >= 0.0, axis=0)               # (H,W)

    return inside.astype(np.uint8)

def _process_one_image_task(task):
    (
        filename, width, height, K, R_c2g, t_c2g,
        camera_info, turn_angle,
        image_folder, mask_dir, masked_images_dir
    ) = task

    try:
        # Build per-frame extrinsics
        c2w = get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g)
        # Debug: Use hardcoded c2w matrix
        # c2w = np.array([
        #     [-0.02085261,  0.36318968,  0.93148184,  0.80234801660],
        #     [-0.99967522,  0.00607778, -0.02474899, -0.10792496141],
        #     [-0.01464992, -0.93169540,  0.36294499,  0.19784899404],
        #     [ 0.00000000,  0.00000000,  0.00000000,    1.00000000]
        # ], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        # Convert from OpenGL to OpenCV coordinate system
        # OpenGL: +Y up, -Z forward, +X right
        # OpenCV: -Y up, +Z forward, +X right
        # Transformation matrix: flip Y and Z axes
        opengl_to_opencv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)

        w2c = opengl_to_opencv @ w2c
        

        # Paths & ext
        image_path = os.path.join(image_folder, filename)

        # ---- Read image (use OpenCV for EXR as requested) ----
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return {'original': filename, 'error': 'Image read failed'}

        # ---- Create binary mask (0/1) ----
        mask = create_rotated_rectangle_mask_closedform(
            width, height,
            K, w2c,
            ROTATION_CENTER,
            RECT_CENTER,
            (RECT_SIZE[0], RECT_SIZE[1]),
            turn_angle
        ).astype(np.uint8)

        masked = np.where(mask[..., None] == 1, image, 0)

        # ---- Project rectangle center and rotation center to image coordinates ----
        # Rectangle center (rotated)
        # th = np.deg2rad(turn_angle)
        # c, s = np.cos(th), np.sin(th)
        # Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        
        # # Rotate rectangle center about rotation center
        # rect_center_vec = np.array(RECT_CENTER) - np.array(ROTATION_CENTER)
        # rotated_rect_center_vec = Rz @ rect_center_vec
        # rotated_rect_center = np.array(ROTATION_CENTER) + rotated_rect_center_vec
        
        # # Project to image coordinates
        # rect_center_h = np.array([rotated_rect_center[0], rotated_rect_center[1], rotated_rect_center[2], 1.0])
        # rotation_center_h = np.array([ROTATION_CENTER[0], ROTATION_CENTER[1], ROTATION_CENTER[2], 1.0])
        
        # rect_center_cam = w2c @ rect_center_h
        # rotation_center_cam = w2c @ rotation_center_h
        
        # rect_center_img = K @ rect_center_cam[:3]
        # rotation_center_img = K @ rotation_center_cam[:3]
        
        # rect_center_px = (int(rect_center_img[0] / rect_center_img[2]), int(rect_center_img[1] / rect_center_img[2]))
        # rotation_center_px = (int(rotation_center_img[0] / rotation_center_img[2]), int(rotation_center_img[1] / rotation_center_img[2]))

        # ---- Save mask PNG (0/255) ----
        mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
        mask_out_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_out_path, (mask * 255).astype(np.uint8))

        # ---- Save masked image per simple policy ----
        masked_filename = f"masked_{filename}"
        masked_out_path = os.path.join(masked_images_dir, masked_filename)
        ok = cv2.imwrite(masked_out_path, masked)
        if not ok:
            return {'original': filename, 'error': 'Write failed for masked image'}

        return {'original': filename, 'mask': mask_filename, 'masked': masked_filename}

    except Exception as e:
        return {'original': filename, 'error': str(e)}
    
# ----------------------- public API (threaded) -----------------------

def create_turntable_mask(image_folder, output_folder, json_path, config, 
                         turntable_radius=0.1, turntable_center=(0.0, 0.0, 0.0),
                         num_workers=None):
    """
    Create masks for camera images based on turntable projection using multithreading.

    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to output folder for masks and masked images
        json_path (str): Path to JSON file containing camera metadata
        config (dict): Configuration containing camera intrinsics (cfg.renderer.camera)
        turntable_radius (float): Radius of the turntable in meters
        turntable_center (tuple): Center position of turntable (x, y, z) in world/robot base space
        num_workers (int or None): Number of threads (default: min(8, os.cpu_count()))

    Returns:
        list: List of processed file info dicts
    """
    # Output dirs
    mask_dir = os.path.join(output_folder, 'masks')
    masked_images_dir = os.path.join(output_folder, 'masked_images')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)
    
    # Load camera metadata
    metadata, camera_metadata, emitter_metadata = load_camera_turntable_light_metadata(json_path)
    
    # Intrinsics
    intrinsics = config['intrinsics']
    width  = int(intrinsics['width'])
    height = int(intrinsics['height'])
    focal_length = float(intrinsics['focal_length'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    # Camera-to-global from config
    R_c2g = np.array(config['R_c2g'], dtype=np.float64)
    t_c2g = np.array(config['t_c2g'], dtype=np.float64)

    # Build tasks
    tasks = []
    for img_data in metadata:
        overall_id = img_data["overall_id"]
        camera_id = img_data["camera_id"]
        camera_info = camera_metadata[str(camera_id)]
        camera_dict = {
            "position": camera_info["position"],
            "rotation_matrix": camera_info["rotation_matrix"],
            "euler": camera_info["euler"]
        }
        turn_angle = img_data['turn_angle'] + INITIAL_ANGLE_DEG
        tasks.append((
            img_data['filename'], width, height, K, R_c2g, t_c2g,
            camera_dict, turn_angle, image_folder, mask_dir, masked_images_dir
        ))

    processed_files = []
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 8)

    # Threaded execution
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_image_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing image masks"):
            res = fut.result()
            if res is not None:
                processed_files.append(res)

    return processed_files

def process_images_with_turntable_mask(image_folder, output_folder, json_path, 
                                     config,
                                     num_workers=None):
    """
    Process all images in a folder with turntable masking using multiple threads.
    """
    processed_files = create_turntable_mask(
        image_folder, output_folder, json_path, config,
        num_workers=num_workers
    )
    
    summary_path = os.path.join(output_folder, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'processed_files': processed_files,
            'total_processed': len(processed_files)
        }, f, indent=2)
    
    print(f"Processed {len(processed_files)} images")
    print(f"Results saved to: {output_folder}")
    return processed_files

# ----------------------- hydra entry -----------------------

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    global ROTATION_CENTER, ROTATION_AXIS, RECT_CENTER, RECT_SIZE, INITIAL_ANGLE_DEG, NUM_WORKERS
    
    # Initialize global variables from config
    if hasattr(cfg.renderer, 'emitter') and hasattr(cfg.renderer.emitter, 'turntable'):
        turntable_config = cfg.renderer.emitter.turntable
        ROTATION_CENTER = tuple(turntable_config.center)
        ROTATION_AXIS = tuple(turntable_config.axis)
    
    if hasattr(cfg.renderer, 'mesh') and hasattr(cfg.renderer.mesh, 'rectangle'):
        rect_config = cfg.renderer.mesh.rectangle
        # RECT_CENTER = tuple(rect_config.center)
        RECT_CENTER = tuple(rect_config.center)
        RECT_SIZE = (rect_config.width, rect_config.length)
    
    # Project the rotation center to the plane z = RECT_CENTER[2]
    ROTATION_CENTER = project_center_to_plane(ROTATION_CENTER, ROTATION_AXIS, RECT_CENTER[2])
    print(f"Projected rotation center: {ROTATION_CENTER}")
    
    # image_folder = os.path.join(cfg.exp_folder, "ldr")
    image_folder = "/absolute/path/to/material/ldr"   # example input folder
    output_folder = os.path.join(cfg.exp_folder, "masks")
    json_path = os.path.join(cfg.exp_folder, "scan_log.json")

    # threads only
    num_workers = min(NUM_WORKERS, os.cpu_count() or NUM_WORKERS)

    process_images_with_turntable_mask(
        image_folder, output_folder, json_path, cfg.renderer.camera,
        num_workers=num_workers
    )

if __name__ == "__main__":
    main()
