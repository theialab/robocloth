/**
 * Coordinate transforms used by the dataset.
 *
 * Ports the relevant numpy logic from:
 *   - utils/transform.py             (rotation primitives)
 *   - utils/io.py                    (scan_log parsing, light & camera transforms)
 *   - recon/calibration/shape_matching.py (rotated_c2w())
 *
 * Conventions
 * -----------
 *   scan_log.json     stores positions in millimetres in robot-base frame
 *   rotated_camera.json stores positions in millimetres in the COLMAP-world
 *                       frame (after sim3 Umeyama alignment to base).
 *   We work in metres everywhere in this module; positions returned from this
 *   module are in metres.
 *
 *   Cameras are stored as gripper poses in scan_log; we apply the hand-eye
 *   calibration `c2g` to get the actual camera pose. The turntable rotation
 *   is then "undone" so that all cameras are expressed in a single "0-angle
 *   world" frame.
 */

(() => {
    const C = window.CONFIG.CALIB;

    // -------------------------------------------------------------------------
    // Tiny vector / 3x3 / 4x4 helpers (no external dependency)
    // -------------------------------------------------------------------------

    /** Identity 4x4. */
    function eye4() {
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ];
    }

    /** Build a 4x4 from a 3x3 rotation and a 3-vector translation. */
    function build4x4(R, t) {
        return [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0, 0, 0, 1],
        ];
    }

    /** 4x4 @ 4x4. */
    function mat4Mul(A, B) {
        const C = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++) {
                let s = 0;
                for (let k = 0; k < 4; k++) s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        return C;
    }

    /** 4x4 @ 4-vector (homogeneous coords). */
    function mat4MulVec4(A, v) {
        const o = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++)
            o[i] = A[i][0]*v[0] + A[i][1]*v[1] + A[i][2]*v[2] + A[i][3]*v[3];
        return o;
    }

    function v3Sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
    function v3Scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
    function v3Norm(v) { return Math.hypot(v[0], v[1], v[2]); }
    function v3Normalize(v) {
        const n = v3Norm(v);
        return n > 1e-12 ? [v[0]/n, v[1]/n, v[2]/n] : [0,0,0];
    }

    // -------------------------------------------------------------------------
    // Rotation primitives (port of utils/transform.py)
    // -------------------------------------------------------------------------

    /**
     * Right-handed CCW rotation about unit axis n by `degrees` degrees.
     * Returns a 3x3 matrix.
     *
     * Mirrors rodrigues_axis_angle() in utils/transform.py.
     */
    function rodriguesAxisAngle(n, degrees) {
        const th = degrees * Math.PI / 180.0;
        const u = v3Normalize(n);
        const [nx, ny, nz] = u;
        const sa = Math.sin(th);
        const ca = Math.cos(th);
        const one_ca = 1 - ca;

        // K = skew(n); R = I + sin*K + (1-cos)*K^2
        // Direct expansion (Rodrigues' formula).
        return [
            [
                ca + nx*nx*one_ca,
                nx*ny*one_ca - nz*sa,
                nx*nz*one_ca + ny*sa,
            ],
            [
                ny*nx*one_ca + nz*sa,
                ca + ny*ny*one_ca,
                ny*nz*one_ca - nx*sa,
            ],
            [
                nz*nx*one_ca - ny*sa,
                nz*ny*one_ca + nx*sa,
                ca + nz*nz*one_ca,
            ],
        ];
    }

    /**
     * 4x4 transform that applies rotation R about world point p.
     *
     *   T = T(p) · R(0) · T(-p)
     *
     * Mirrors build_rot_about_point() in utils/transform.py.
     */
    function buildRotAboutPoint(R, p) {
        const T = build4x4(R, [0, 0, 0]);
        // Place translation = (I - R) @ p
        const tx = p[0] - (R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2]);
        const ty = p[1] - (R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2]);
        const tz = p[2] - (R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2]);
        T[0][3] = tx;
        T[1][3] = ty;
        T[2][3] = tz;
        return T;
    }

    // -------------------------------------------------------------------------
    // Per-frame transforms.
    //
    // The training pipeline applies these for each scan_log entry; the
    // viewer's two viz panels show two snapshots of this chain:
    //
    //   1. "Combined base1"  — camera + light expressed in the camera-arm
    //                          base (b1). The light is brought into b1 via
    //                          the calibrated base2_to_base1 transform.
    //                          No turntable rotation is undone, so the user
    //                          sees the *physical* layout at each frame.
    //
    //   2. "Post-calibration / world0" — after the turntable rotation is
    //                          undone, all frames share a single world0
    //                          coordinate system anchored at the turntable.
    //                          This is what the renderer / emitter use.
    //
    // Notation:
    //   • g2b1   = gripper-of-camera-arm pose, expressed in b1
    //   • g2b2   = gripper-of-light-arm  pose, expressed in b2
    //   • c2g, l2g = camera-to-gripper, light-to-gripper (hand-eye)
    //   • T_b1_w0 = transform from b1 to world0 (turntable-aligned)
    //
    //   camera_in_b1   = g2b1 · c2g
    //   light_in_b1    = base2_to_base1 · g2b2 · l2g
    //   *_in_world0    = T_b1_w0(turn_angle) · *_in_b1
    // -------------------------------------------------------------------------

    function cameraPoseInBase1(entry) {
        const g2b1 = build4x4(
            entry.rotation_matrix,
            entry.position.map((v) => v / 1000.0),
        );
        const c2g = build4x4(C.R_C2G, C.t_C2G);
        return mat4Mul(g2b1, c2g);
    }

    function lightPoseInBase1(entry) {
        const g2b2 = build4x4(
            entry.rotation_matrix_light,
            entry.position_light.map((v) => v / 1000.0),
        );
        const l2g = build4x4(C.R_L2G, C.t_L2G);
        const g2b1 = mat4Mul(C.BASE2_TO_BASE1, g2b2);
        return mat4Mul(g2b1, l2g);
    }

    /** Build the 4x4 b1->world0 transform for a given turn_angle (degrees). */
    function transformBase1ToWorld0(turnAngleDeg) {
        const R_undo = rodriguesAxisAngle(C.TURNTABLE_AXIS, -turnAngleDeg);
        return buildRotAboutPoint(R_undo, C.TURNTABLE_CENTER);
    }

    function poseTranslation(M) { return [M[0][3], M[1][3], M[2][3]]; }

    function cameraPosInBase1(entry)  { return poseTranslation(cameraPoseInBase1(entry)); }
    function lightPosInBase1(entry)   { return poseTranslation(lightPoseInBase1(entry)); }

    function cameraPosInWorld0(entry) {
        const T = transformBase1ToWorld0(entry.turn_angle || 0);
        return poseTranslation(mat4Mul(T, cameraPoseInBase1(entry)));
    }
    function lightPosInWorld0(entry) {
        const T = transformBase1ToWorld0(entry.turn_angle || 0);
        return poseTranslation(mat4Mul(T, lightPoseInBase1(entry)));
    }

    /**
     * Process the full scan_log. Returns per-entry records (each containing
     * the four positions we plot) plus aggregate slider axes.
     */
    function processScanLog(scanLog) {
        const entries = scanLog.map((e) => ({
            id: e.id,
            scan_id: e.scan_id,
            camera_id: e.camera_id,
            light_id: e.light_id,
            turn_angle: e.turn_angle || 0.0,
            filename: e.filename,
            // base1 (camera-arm base): both arms unified via base2_to_base1
            cam_pos_b1:   cameraPosInBase1(e),
            light_pos_b1: lightPosInBase1(e),
            // world0: turntable rotation undone, what the trained model uses
            cam_pos_w0:   cameraPosInWorld0(e),
            light_pos_w0: lightPosInWorld0(e),
        }));

        const uniqInts = (arr) => [...new Set(arr)].sort((a, b) => a - b);
        const uniqFloats = (arr) =>
            [...new Set(arr.map((v) => Number(v.toFixed(6))))].sort((a, b) => a - b);

        return {
            entries,
            camera_ids:  uniqInts(entries.map((e) => e.camera_id)),
            light_ids:   uniqInts(entries.map((e) => e.light_id)),
            turn_angles: uniqFloats(entries.map((e) => e.turn_angle)),
        };
    }

    // -------------------------------------------------------------------------
    // Rotated-camera helpers
    // -------------------------------------------------------------------------

    /**
     * Build a fast index map from rotated_camera.json so the UI can look up
     * "post-calibration" camera positions by camera_id.
     *
     * Note: rotated_camera positions are in millimetres (matching scan_log),
     * so we divide by 1000.
     */
    function indexRotatedCamera(rotatedCameraJSON) {
        const m = new Map();
        rotatedCameraJSON.forEach((c) => {
            m.set(Number(c.camera_id), {
                camera_id: c.camera_id,
                overall_id: c.overall_id,
                position_m: c.position.map((v) => v / 1000.0),
                rotation_matrix: c.rotation_matrix,
            });
        });
        return m;
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // World0 → image-pixel projection
    //
    // rotated_camera.json stores camera-to-world0 poses in OpenGL convention
    // *with the Umeyama scale `s` baked into the 3x3 rotation matrix*. We
    // extract `s` from the column norm, divide it out to recover a proper
    // unit rotation, then build w2c, convert OpenGL→OpenCV, and apply the
    // SIMPLE_RADIAL intrinsics from hdr_crop_bboxes.json.
    //
    // Pipeline mirrors recon/calibration/shape_matching.py
    // _process_camera_batch:
    //   x_norm = X/Z
    //   r2     = x_norm² + y_norm²
    //   pixel  = focal * (1 + k*r2) * (x_norm, y_norm) + (cx, cy)
    // -------------------------------------------------------------------------

    /**
     * Build a projection function from one rotated_camera entry +
     * intrinsics. The returned function takes a world0 point (metres) and
     * returns [u, v] in *image* pixels (or null if behind the camera).
     */
    function makeWorld0Projector(rotEntry, intrinsics) {
        // rotEntry.rotation_matrix has scale s baked in (the Umeyama scale
        // from sim3 align). Extract it from any column norm.
        const R = rotEntry.rotation_matrix;
        const sCol = Math.hypot(R[0][0], R[1][0], R[2][0]);
        if (!isFinite(sCol) || sCol < 1e-12) return null;
        const inv_s = 1.0 / sCol;
        const Rn = [
            [R[0][0]*inv_s, R[0][1]*inv_s, R[0][2]*inv_s],
            [R[1][0]*inv_s, R[1][1]*inv_s, R[1][2]*inv_s],
            [R[2][0]*inv_s, R[2][1]*inv_s, R[2][2]*inv_s],
        ];
        const t = rotEntry.position_m; // already in metres

        // w2c rotation = Rn^T, w2c translation = -Rn^T · t
        const wRx0 = Rn[0][0], wRx1 = Rn[1][0], wRx2 = Rn[2][0];
        const wRy0 = Rn[0][1], wRy1 = Rn[1][1], wRy2 = Rn[2][1];
        const wRz0 = Rn[0][2], wRz1 = Rn[1][2], wRz2 = Rn[2][2];
        const wtx = -(wRx0*t[0] + wRx1*t[1] + wRx2*t[2]);
        const wty = -(wRy0*t[0] + wRy1*t[1] + wRy2*t[2]);
        const wtz = -(wRz0*t[0] + wRz1*t[1] + wRz2*t[2]);

        const f = intrinsics.focal_length;
        const cx = intrinsics.cx;
        const cy = intrinsics.cy;
        const k = intrinsics.distortion ?? 0.0;

        return function project(P) {
            // OpenGL camera-space point
            const Xg = wRx0*P[0] + wRx1*P[1] + wRx2*P[2] + wtx;
            const Yg = wRy0*P[0] + wRy1*P[1] + wRy2*P[2] + wty;
            const Zg = wRz0*P[0] + wRz1*P[1] + wRz2*P[2] + wtz;
            // OpenGL→OpenCV: flip Y and Z
            const X = Xg, Y = -Yg, Z = -Zg;
            if (Z <= 1e-9) return null;       // behind camera
            const x_n = X / Z, y_n = Y / Z;
            const r2 = x_n*x_n + y_n*y_n;
            const dist = 1 + k * r2;
            return [f * dist * x_n + cx, f * dist * y_n + cy];
        };
    }

    window.Transforms = {
        eye4,
        build4x4,
        mat4Mul,
        mat4MulVec4,
        v3Sub,
        v3Scale,
        v3Norm,
        v3Normalize,
        rodriguesAxisAngle,
        buildRotAboutPoint,
        cameraPoseInBase1,
        lightPoseInBase1,
        transformBase1ToWorld0,
        cameraPosInBase1,
        lightPosInBase1,
        cameraPosInWorld0,
        lightPosInWorld0,
        processScanLog,
        indexRotatedCamera,
        makeWorld0Projector,
    };
})();
