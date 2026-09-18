/**
 * Global configuration for the RoboCloth dataset webpage.
 *
 * The webpage loads everything directly from the public HuggingFace dataset.
 * No private data is stored locally except small precomputed lookup files.
 */

window.CONFIG = (() => {
    const HF_REPO = "koalapenguin/RoboCloth";

    // resolve URL for a file in the HF dataset (Range-friendly through redirects)
    const hfResolve = (path) =>
        `https://huggingface.co/datasets/${HF_REPO}/resolve/main/${path}`;

    return {
        HF_REPO,
        HF_BASE: `https://huggingface.co/datasets/${HF_REPO}`,
        hfResolve,

        // Per-material URLs — always under materials/<id>/.
        // (We used to also support sample/material_<id>/ for a reviewer
        //  preview that contained only ~12 images per material; that path
        //  caused scan_log to reference PNGs that didn't exist in the
        //  smaller tar, so it's been removed. All data now comes from the
        //  full materials/ tree.)
        materialDir:   (id) => `materials/${id}`,
        scanLogURL:    (id) => hfResolve(`materials/${id}/scan_log.json`),
        rotatedCameraURL: (id) => hfResolve(`materials/${id}/rotated_camera.json`),
        unmatchedURL:  (id) => hfResolve(`materials/${id}/unmatched_scan_ids.json`),
        hdrTarURL:     (id) => hfResolve(`materials/${id}/hdr.tar`),
        bboxURL:       (id) => hfResolve(`materials/${id}/bbox.json`),
        cropBBoxURL:   (id) => hfResolve(`materials/${id}/hdr_crop_bboxes.json`),

        // Global metadata URLs
        cameraFactorURL: () => hfResolve("globals/camera_factor.json"),
        sampleSizeURL: () => hfResolve("globals/sample_size.json"),
        emitterCalibrationURL: () => hfResolve("globals/emitter_calibration.json"),
        testListURL: (n) => hfResolve(`globals/test_list_${n}.txt`),
        trainListURL: (n) => hfResolve(`globals/training_list_${n}.txt`),

        // Local-only resources shipped with the webpage
        // - manifest.json: preview-material list with metadata + tar indexes
        // - tar_index/{id}.json: precomputed (filename -> [offset, size]) for hdr.tar
        manifestURL: () => "data/manifest.json",
        tarIndexURL: (id) => `data/tar_index/${id}.json`,

        // Post-calibration transforms, sourced from the actual training
        // config (config/renderer/realcapture_area_emitter.yaml) so the
        // viewer's geometry matches what the trained model sees.
        //
        // Two coordinate frames matter here:
        //   • base1  — camera-arm-base (XArm @ 192.168.1.220)
        //   • base2  — light-arm-base  (XArm @ 192.168.1.240)
        //   • world0 — turntable-aligned world (turn_angle undone about the
        //              turntable centre+axis)
        //
        // The pipeline for any scan-log entry:
        //   camera_in_base1   = gripper_cam_pose_in_base1  @ R_c2g
        //   light_in_base1    = base2_to_base1 @ gripper_light_pose_in_base2 @ R_l2g
        //   *_in_world0       = T_b1_to_w0(turn_angle) @ *_in_base1
        //
        // All translations are in metres.
        CALIB: {
            // Camera-frame -> camera-arm gripper-frame (OpenGL convention).
            // Refined hand-eye solved during COLMAP calibration.
            R_C2G: [
                [-7.17667595e-04,  9.99776474e-01,  2.11302413e-02],
                [-4.42126179e-03,  2.11268680e-02, -9.99767027e-01],
                [-9.99989969e-01, -8.10922726e-04,  4.40511146e-03],
            ],
            t_C2G: [3.26272696e-02, -1.61560433e-02, 2.80920856e-02],

            // Light-frame -> light-arm gripper-frame.
            R_L2G: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            t_L2G: [0.0, -0.102, 0.02112],

            // light-arm-base (base2) -> camera-arm-base (base1).
            // Calibrated 4x4; nearly identity rotation + a few-mm offset.
            BASE2_TO_BASE1: [
                [ 9.99999959e-01,  2.33227594e-04, -1.69051819e-04,  7.22042468e-03],
                [-2.36740680e-04,  9.99777598e-01, -2.10878871e-02,  8.26339062e-03],
                [ 1.64095944e-04,  2.10879262e-02,  9.99777611e-01,  3.93500345e-03],
                [ 0.0,             0.0,             0.0,             1.0           ],
            ],

            // Turntable expressed in camera-arm-base (base1).
            TURNTABLE_CENTER: [0.16084722, -0.11011424, -0.021],
            TURNTABLE_AXIS:   [-0.00876202, -0.01346449, 0.99987096],
        },

        // Visualization defaults
        VIZ: {
            // Tone mapping for HDR 16-bit PNG
            HDR_GAMMA: 2.2,
            HDR_EXPOSURE: 1.0,
            // CCM applied when loading PNGs in the training pipeline.
            // For visualization we use identity (tone-mapping is enough).
            CCM_IDENTITY: true,
        },
    };

})();
