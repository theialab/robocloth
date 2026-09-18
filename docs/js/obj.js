/**
 * Per-material viewer logic.
 *
 * Workflow:
 *   1.  Parse `?id=N&s=1` from the URL.
 *   2.  Fetch scan_log.json + rotated_camera.json + (optional) bbox/crop.
 *   3.  Apply the JS coordinate transforms to get capture geometry.
 *   4.  Build the 3D primitives (turntable, arc, hemisphere ball).
 *   5.  Wire sliders for camera_id / light_id / turn_angle / frame index.
 *   6.  On slider change, look up matching scan_log entry, load that PNG
 *       out of hdr.tar via Range, tone-map onto the canvas.
 */

(async function () {
    // -----------------  parse query  -----------------
    const qs = new URLSearchParams(location.search);
    const id = parseInt(qs.get("id") || "0", 10);

    document.getElementById("material-id").textContent = id;
    const subtitle = document.getElementById("subtitle");

    // Point the HuggingFace button at this material's folder on the HF
    // dataset tree (the human-friendly tree page, not a raw file).
    const hfLink = document.getElementById("btn-hf");
    if (hfLink) {
        hfLink.href =
            `${CONFIG.HF_BASE}/tree/main/${CONFIG.materialDir(id)}`;
    }

    // Bubble unexpected errors to the subtitle so headless / no-JS-console
    // users still see them.  We swallow WebGL context errors here because
    // those are handled in-place by markVizUnavailable() below.
    window.addEventListener("unhandledrejection", (ev) => {
        const reason = String(ev.reason || "");
        if (reason.includes("WebGL")) return;
        console.error("[unhandled rejection]", ev.reason);
        subtitle.innerHTML =
            `<span style="color:#a33">Error: ${reason}</span>`;
    });

    // -----------------  load manifest (for prev/next nav)  -----------------
    let manifest;
    try {
        const r = await fetch(CONFIG.manifestURL());
        manifest = r.ok ? await r.json() : { preview: [] };
    } catch (e) { manifest = { preview: [] }; }
    setupNav(manifest, id);

    // -----------------  fetch material metadata  -----------------
    const scanURL = CONFIG.hfResolve(
        `${CONFIG.materialDir(id)}/scan_log.json`,
    );
    const rotURL = CONFIG.hfResolve(
        `${CONFIG.materialDir(id)}/rotated_camera.json`,
    );
    const cropBoxesURL = CONFIG.hfResolve(
        `${CONFIG.materialDir(id)}/hdr_crop_bboxes.json`,
    );
    const bboxURL = CONFIG.hfResolve(
        `${CONFIG.materialDir(id)}/bbox.json`,
    );
    const tarURL = CONFIG.hfResolve(
        `${CONFIG.materialDir(id)}/hdr.tar`,
    );

    // bbox.json stores the material's 3D bounding box in WORLD0 frame
    // (computed by recon/calibration/shape_matching.py:process_pointcloud_to_base).
    // crop (hdr_crop_bboxes.json) carries the per-image 2D pixel crops.
    let scanLog, rotatedCam, crop, materialBBox;
    try {
        [scanLog, rotatedCam, crop, materialBBox] = await Promise.all([
            fetch(scanURL).then(r => r.json()),
            fetch(rotURL).then(r => r.json()),
            fetch(cropBoxesURL).then(r => r.ok ? r.json() : null),
            fetch(bboxURL).then(r => r.ok ? r.json() : null),
        ]);
    } catch (err) {
        subtitle.innerHTML = `<span style="color:#a33">Failed to load metadata: ${err.message}</span>`;
        return;
    }

    subtitle.innerHTML = `
        <b>${scanLog.length}</b> capture configurations ·
        <b>${rotatedCam.length}</b> calibrated cameras
    `;

    // -----------------  process scan_log  -----------------
    // The raw scan-log entries are the source of truth for capture geometry.
    // rotated_camera.json is consulted only for the "calibrated cameras"
    // count in the subtitle now; it no longer participates in the 3D viz
    // or stats path (per-material BRDF stats are precomputed offline —
    // see webpage/tools/build_material_stats.py).
    const processed = Transforms.processScanLog(scanLog);
    const validEntries = processed.entries;

    // Fetch the precomputed per-material stats (mean colour + per-channel
    // range). This file is built once offline and shipped under
    // webpage/data/material_stats/<id>.json. If it's missing we'll just
    // skip the material-info numbers; nothing else depends on it.
    const statsURL = `data/material_stats/${id}.json`;
    fetch(statsURL)
        .then(r => (r.ok ? r.json() : null))
        .then(stats => {
            if (!stats) return;
            applyMaterialStats(stats);
        })
        .catch(err => console.warn("[stats]", err));

    // Determine the turntable disc radius — the sample-size metadata
    // describes the cloth swatch, which is mounted on the turntable surface.
    // Half the largest XY dim is a reasonable disc radius for the viz.
    let sampleHalf = 0.075;
    if (crop && crop.bbox_size) {
        sampleHalf = Math.max(crop.bbox_size[0], crop.bbox_size[1]) / 2;
    }

    // -----------------  build 3D scenes — two views of one chain  -----------------
    //
    // Both scenes show camera + light positions together. They differ only
    // in which step of the transform chain they freeze:
    //   • base1  — physical capture frame (light brought into base1 via
    //              base2_to_base1; turntable rotation NOT undone)
    //   • world0 — full chain (above PLUS turntable rotation undone)
    // The render code is identical; only the input positions / turntable
    // centre change.

    // De-duplicate per camera_id / light_id so the scatter shows one dot per
    // unique position (otherwise we'd plot 500+ near-coincident dots).
    const camB1   = new Map();
    const lightB1 = new Map();
    const camW0   = new Map();
    const lightW0 = new Map();
    processed.entries.forEach(e => {
        if (!camB1.has(e.camera_id))   camB1.set(e.camera_id,   e.cam_pos_b1);
        if (!lightB1.has(e.light_id))  lightB1.set(e.light_id,  e.light_pos_b1);
        if (!camW0.has(e.camera_id))   camW0.set(e.camera_id,   e.cam_pos_w0);
        if (!lightW0.has(e.light_id))  lightW0.set(e.light_id,  e.light_pos_w0);
    });
    // For world0 the light positions sweep across all turn angles (one dot per
    // (light_id, turn_angle) pair), since the turntable undo depends on
    // turn_angle. Use every entry rather than the de-duped Maps.
    const lightW0_all = processed.entries.map(e => e.light_pos_w0);

    // The DISC's visual XY is the calibrated rotation axis so the box
    // rotation stays coherent. The Z passed here is interpreted by viz.js
    // as the TOP SURFACE of the disc — viz.js positions the cylinder so
    // its top face lands at this Z and the disc extends downward by
    // turntableThickness. We line this top up with bbox_min[2] so the
    // material slab sits directly on the turntable.
    const ttc = CONFIG.CALIB.TURNTABLE_CENTER;
    const discTopZ = (materialBBox && materialBBox.bbox_min)
        ? materialBBox.bbox_min[2]
        : ttc[2];
    const discCenter = [ttc[0], ttc[1], discTopZ];

    // Physical turntable is much wider than the cloth swatch. Largest
    // sample we ship is 14 cm (so half = 7 cm); pick a disc that
    // dwarfs that with comfortable margin.
    const discRadius = Math.max(0.22, sampleHalf * 3.0);
    const discThickness = 0.025;  // 2.5 cm of visible turntable body

    const b1Canvas = document.getElementById("viz-base1");
    const w0Canvas = document.getElementById("viz-world0");

    let b1Scene = null, w0Scene = null;
    try {
        b1Scene = new Viz.CombinedCaptureScene(b1Canvas, {
            cameraPositions: [...camB1.values()],
            lightPositions:  [...lightB1.values()],
            turntableCenter: discCenter,
            turntableRadius: discRadius,
            turntableThickness: discThickness,
            drawArc:         true,
            // Material 3D bbox: stored in world0; in base1 we have to
            // rotate it back by +turn_angle around the turntable axis.
            materialBBox,
            rotateMaterialWithTurnAngle: true,
        });
    } catch (err) {
        console.warn("[viz base1]", err);
        markVizUnavailable(b1Canvas, err);
    }

    try {
        w0Scene = new Viz.CombinedCaptureScene(w0Canvas, {
            cameraPositions: [...camW0.values()],
            // Use the full sweep so the lights spread out across all turn
            // angles. With the turntable rotation undone they form a band
            // around the hemisphere; with only one entry per light_id we'd
            // just see a single arc.
            lightPositions:  lightW0_all,
            turntableCenter: discCenter,
            turntableRadius: discRadius,
            turntableThickness: discThickness,
            drawArc:         true,
            // World0: the bbox is already in this frame, draw it static.
            materialBBox,
            rotateMaterialWithTurnAngle: false,
        });
    } catch (err) {
        console.warn("[viz world0]", err);
        markVizUnavailable(w0Canvas, err);
    }

    // -----------------  selection-marker icons (procedural)  -----------------
    //
    // Two small "Camera A" + "Light A" 3D shapes built from primitives —
    // a machine-vision camera (body + lens + foot) and a flat LED panel
    // with a bright emissive face. They're swapped into the scenes after
    // construction so we don't block on anything.
    if (b1Scene && b1Scene.attachSelectionModels) {
        b1Scene.attachSelectionModels(
            buildCameraIcon(0x0a3aad),
            buildLightIcon(0xffe54a),  // bright yellow LED face
        );
    }
    if (w0Scene && w0Scene.attachSelectionModels) {
        w0Scene.attachSelectionModels(
            buildCameraIcon(0x0a3aad),
            buildLightIcon(0xffe54a),  // bright yellow LED face
        );
    }

    // NOTE on the local-frame convention:
    //
    // We're authoring these icons so that the "front" (lens / emitting
    // surface) is along the +Z axis. After Object3D.lookAt(target),
    // three.js orients a NON-camera/light Object3D so that its local +Z
    // points at the target. (Camera/Light objects use the opposite
    // convention — local -Z toward target — but Groups don't.) So +Z
    // forward here means lens + LED face point at the cloth.

    /** Small machine-vision-style camera (body + lens, +Z = lens forward). */
    function buildCameraIcon(bodyColor) {
        const g = new THREE.Group();

        const body = new THREE.Mesh(
            new THREE.BoxGeometry(0.022, 0.022, 0.026),
            new THREE.MeshBasicMaterial({ color: bodyColor }),
        );
        body.position.z = -0.005;             // BEHIND the lens
        g.add(body);

        const hood = new THREE.Mesh(
            new THREE.CylinderGeometry(0.012, 0.012, 0.006, 16),
            new THREE.MeshBasicMaterial({ color: 0x05205a }),
        );
        hood.rotation.x = Math.PI / 2;        // axis -> Z
        hood.position.z = 0.012;              // in front of body
        g.add(hood);

        const lens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.0085, 0.0085, 0.020, 16),
            new THREE.MeshBasicMaterial({ color: 0x111417 }),
        );
        lens.rotation.x = Math.PI / 2;
        lens.position.z = 0.008;
        g.add(lens);

        const glass = new THREE.Mesh(
            new THREE.CircleGeometry(0.0075, 24),
            new THREE.MeshBasicMaterial({ color: 0x4c6cb8 }),
        );
        glass.position.z = 0.0185;            // at lens front
        g.add(glass);

        // Small foot underneath suggests the tripod mount
        const foot = new THREE.Mesh(
            new THREE.BoxGeometry(0.014, 0.004, 0.018),
            new THREE.MeshBasicMaterial({ color: 0x05205a }),
        );
        foot.position.set(0, -0.013, -0.005);
        g.add(foot);

        return g;
    }

    /**
     * Flat LED panel (+Z = emitting face forward).
     *
     * Whole body is yellow so the icon reads as "a glowing LED" rather
     * than "a dark disc with one yellow side" — at marker size the
     * darker grey casing made it hard to tell what it was.
     */
    function buildLightIcon(emitColor) {
        const g = new THREE.Group();

        // Casing back — slightly darker yellow than the emitting face
        // so the body still has depth without going to grey.
        const casing = new THREE.Mesh(
            new THREE.CylinderGeometry(0.020, 0.020, 0.006, 16),
            new THREE.MeshBasicMaterial({ color: 0xeab308 }),
        );
        casing.rotation.x = Math.PI / 2;
        casing.position.z = -0.003;           // BEHIND the emitting face
        g.add(casing);

        // Bright emitting face
        const face = new THREE.Mesh(
            new THREE.CircleGeometry(0.018, 24),
            new THREE.MeshBasicMaterial({ color: emitColor }),
        );
        face.position.z = 0.001;              // just in front of casing
        g.add(face);

        // Outer rim — a thin bright-yellow ring around the casing for
        // extra visibility at small sizes.
        const rim = new THREE.Mesh(
            new THREE.RingGeometry(0.0185, 0.020, 24),
            new THREE.MeshBasicMaterial({ color: 0xfde047, side: THREE.DoubleSide }),
        );
        rim.position.z = 0.0035;
        g.add(rim);

        return g;
    }

    // -----------------  slider state machine  -----------------
    const state = {
        cropToBBox: false,
        exposure: 1.2,
        gamma: 2.2,
        currentFrame: 0,
    };

    // Monotonically-incrementing token. The most-recent applyFrame() wins;
    // older in-flight fetches don't overwrite the canvas. Declared up here
    // (before the initial applyFrame() call below) so it isn't read while
    // still in the `let` temporal dead zone.
    let loadSeq = 0;

    // Sliders
    const sliderFrame = document.getElementById("slider-frame");
    const sliderCam = document.getElementById("slider-camera");
    const sliderLight = document.getElementById("slider-light");
    const sliderTurn = document.getElementById("slider-turn");

    sliderFrame.max = Math.max(0, validEntries.length - 1);
    sliderCam.max = Math.max(0, processed.camera_ids.length - 1);
    sliderLight.max = Math.max(0, processed.light_ids.length - 1);
    sliderTurn.max = Math.max(0, processed.turn_angles.length - 1);

    sliderFrame.addEventListener("input", () => {
        state.currentFrame = parseInt(sliderFrame.value, 10);
        applyFrame();
    });

    // Camera/Light/Turn sliders pick the NEAREST entry that matches the
    // current combination. With 580 entries spread across 580 cameras,
    // 86 lights, and 84 turn angles, not every (cam,light,turn) triplet
    // exists — so we fall back to nearest match.
    sliderCam.addEventListener("input", () => {
        const camId = processed.camera_ids[parseInt(sliderCam.value, 10)];
        const idx = findEntry(validEntries, { camera_id: camId });
        if (idx >= 0) { state.currentFrame = idx; sliderFrame.value = idx; applyFrame(); }
    });
    sliderLight.addEventListener("input", () => {
        const lightId = processed.light_ids[parseInt(sliderLight.value, 10)];
        const idx = findEntry(validEntries, { light_id: lightId });
        if (idx >= 0) { state.currentFrame = idx; sliderFrame.value = idx; applyFrame(); }
    });
    sliderTurn.addEventListener("input", () => {
        const turn = processed.turn_angles[parseInt(sliderTurn.value, 10)];
        const idx = findEntry(validEntries, { turn_angle: turn });
        if (idx >= 0) { state.currentFrame = idx; sliderFrame.value = idx; applyFrame(); }
    });

    document.getElementById("btn-rand").addEventListener("click", () => {
        state.currentFrame = Math.floor(Math.random() * validEntries.length);
        sliderFrame.value = state.currentFrame;
        applyFrame();
    });
    document.getElementById("btn-crop").addEventListener("click", (e) => {
        state.cropToBBox = !state.cropToBBox;
        e.currentTarget.classList.toggle("active", state.cropToBBox);
        applyFrame();
    });
    document.getElementById("btn-expmore").addEventListener("click", () => {
        state.exposure *= 1.4; applyFrame();
    });
    document.getElementById("btn-expless").addEventListener("click", () => {
        state.exposure /= 1.4; applyFrame();
    });

    // -----------------  tar index (lazy)  -----------------
    let tarIndexPromise = null;
    let indexReady = false;
    function getTarIndex() {
        if (!tarIndexPromise) {
            // Two cost regimes here:
            //   1. We have a precomputed JSON in data/tar_index/<id>.json —
            //      that downloads in ~30 ms. NO indicator needed.
            //   2. We don't, and we have to walk the live tar's ~580
            //      headers via Range — that can take 20-40 s. THEN we
            //      want to tell the user what's happening.
            //
            // Strategy: don't show ANY indicator for the first 350 ms.
            // For (1) we'll be done well before then. For (2) we'll show
            // the "building" message only once we know we're on that
            // slower path.
            const meta = document.getElementById("meta-filename");
            let indicator = null;
            const showIndicator = (text) => {
                if (indicator) return;
                indicator = document.createElement("span");
                indicator.id = "tar-index-progress";
                indicator.innerHTML =
                    ' <span style="color:var(--muted);font-size:0.78rem">'
                    + `— ${text}</span>`;
                meta.after(indicator);
            };
            const removeIndicator = () => {
                if (indicator) { indicator.remove(); indicator = null; }
            };
            // Fallback: if neither path completes within 350 ms, show a
            // generic "loading…" until we know more.
            const fallbackTimer = setTimeout(
                () => showIndicator("loading tar index…"),
                350,
            );

            tarIndexPromise = (async () => {
                const idxURL = CONFIG.tarIndexURL(id);
                try {
                    const r = await fetch(idxURL);
                    if (r.ok) {
                        const data = await r.json();
                        clearTimeout(fallbackTimer);
                        removeIndicator();
                        indexReady = true;
                        return data;
                    }
                } catch (e) { /* fall through to build */ }

                // No precomputed index: walking the tar is the slow path.
                // Upgrade the indicator wording now that we know.
                clearTimeout(fallbackTimer);
                removeIndicator();
                showIndicator(
                    "building tar index from live tar (~30 s first load)…",
                );

                const data = await TarReader.buildIndex(tarURL);
                removeIndicator();
                indexReady = true;
                return data;
            })().catch(err => {
                clearTimeout(fallbackTimer);
                removeIndicator();
                console.error("[tar index]", err);
                throw err;
            });
        }
        return tarIndexPromise;
    }

    /**
     * Prefetch the K nearest frames (by current frame index) in the
     * background. Keeps the cache warm while the user is reading the page,
     * so dragging the slider feels instant.
     */
    function schedulePrefetch(centerIdx, k = 4) {
        if (!indexReady || validEntries.length === 0) return;
        const fns = [];
        for (let d = 1; d <= k; d++) {
            const a = centerIdx + d, b = centerIdx - d;
            if (a < validEntries.length) fns.push(validEntries[a].filename);
            if (b >= 0) fns.push(validEntries[b].filename);
        }
        // Fire-and-forget — TarReader.prefetchFiles is internally throttled.
        getTarIndex().then((idx) =>
            TarReader.prefetchFiles(tarURL, idx, fns, 2),
        );
    }

    // -----------------  initial render  -----------------
    applyFrame();


    // -------- helpers --------

    /**
     * Find the FIRST entry in `validEntries` whose fields match the partial
     * filter object. Falls back to the entry whose stored field is closest
     * to the requested value for `turn_angle` (float comparison).
     */
    function findEntry(arr, filter) {
        const exact = arr.findIndex(e =>
            Object.entries(filter).every(([k, v]) => e[k] === v),
        );
        if (exact >= 0) return exact;

        if (filter.turn_angle != null) {
            let best = -1, bestD = Infinity;
            arr.forEach((e, i) => {
                const d = Math.abs(e.turn_angle - filter.turn_angle);
                if (d < bestD) { bestD = d; best = i; }
            });
            return best;
        }
        return -1;
    }


    /** Apply the current frame index: update sliders, viz, and image. */
    function applyFrame() {
        if (validEntries.length === 0) return;
        const i = clamp(state.currentFrame, 0, validEntries.length - 1);
        state.currentFrame = i;
        const e = validEntries[i];

        // Update the secondary sliders to reflect the current entry
        sliderCam.value = processed.camera_ids.indexOf(e.camera_id);
        sliderLight.value = processed.light_ids.indexOf(e.light_id);
        const turnIdx = processed.turn_angles.findIndex(
            t => Math.abs(t - e.turn_angle) < 1e-3,
        );
        if (turnIdx >= 0) sliderTurn.value = turnIdx;

        document.getElementById("frame-tag").textContent =
            `${i + 1} / ${validEntries.length}`;
        document.getElementById("cam-tag").textContent = `id ${e.camera_id}`;
        document.getElementById("light-tag").textContent = `id ${e.light_id}`;
        document.getElementById("turn-tag").textContent =
            `${e.turn_angle.toFixed(1)}°`;
        document.getElementById("meta-filename").textContent = e.filename;
        document.getElementById("meta-camera").textContent = e.camera_id;
        document.getElementById("meta-light").textContent = e.light_id;
        document.getElementById("meta-turn").textContent = e.turn_angle.toFixed(1) + "°";

        // 3D scenes — highlight the active camera + light in both views.
        if (b1Scene) {
            b1Scene.setSelection({
                camPos:   e.cam_pos_b1,
                lightPos: e.light_pos_b1,
            });
            // In base1 the material follows the physical turntable: rotate
            // its world0 bbox forward by +turn_angle about the calibrated
            // axis / centre so the slab orientation matches the current
            // frame's turntable position.
            b1Scene.setMaterialTurnAngle(e.turn_angle);
        }
        if (w0Scene) w0Scene.setSelection({
            camPos:   e.cam_pos_w0,
            lightPos: e.light_pos_w0,
        });

        // Image (sequence number guards against stale paints when the user
        // drags the slider faster than the network)
        loadImage(e, ++loadSeq);

        // Warm the cache for nearby frames so the next slider tick is instant
        schedulePrefetch(i, 4);
    }


    /** Pull the HDR PNG for entry `e` and paint it tone-mapped. */
    async function loadImage(e, mySeq) {
        const imgCanvas = document.getElementById("image-canvas");
        const spinner = document.getElementById("img-spinner");
        spinner.style.display = "inline-block";

        try {
            const index = await getTarIndex();
            const buf = await TarReader.fetchFile(tarURL, index, e.filename);
            if (mySeq !== loadSeq) return; // user moved on; drop result

            let bbox = null;
            if (state.cropToBBox && crop && crop.bboxes) {
                bbox = crop.bboxes[String(e.camera_id)] || null;
            }
            await HDRDisplay.drawPng(buf, imgCanvas, {
                exposure: state.exposure,
                gamma: state.gamma,
                bbox,
                maxWidth: 1600,
            });
            if (mySeq !== loadSeq) return;
        } catch (err) {
            if (mySeq !== loadSeq) return;
            console.warn("[image]", err);
            const ctx = imgCanvas.getContext("2d");
            imgCanvas.width = 800; imgCanvas.height = 533;
            ctx.fillStyle = "#0d0f12";
            ctx.fillRect(0, 0, 800, 533);
            ctx.fillStyle = "#a33";
            ctx.textAlign = "center";
            ctx.font = "16px sans-serif";
            ctx.fillText(err.message, 400, 270);
        } finally {
            if (mySeq === loadSeq) spinner.style.display = "none";
        }
    }


    function setupNav(manifest, currentId) {
        // Prev/Next walks the full set of materials (train followed by
        // test, each block sorted by id). This lets the user step through
        // all 500 materials without going back to the index page.
        const train = manifest.train_ids || [];
        const test  = manifest.test_ids  || [];
        const ids = [...train, ...test];
        const i = ids.indexOf(currentId);
        const prev = i > 0 ? ids[i - 1] : null;
        const next = i >= 0 && i < ids.length - 1 ? ids[i + 1] : null;

        const goto = (mid) => {
            if (mid != null) location.href = `obj.html?id=${mid}`;
        };
        const prevBtn = document.getElementById("btn-prev");
        const nextBtn = document.getElementById("btn-next");
        prevBtn.disabled = prev == null;
        nextBtn.disabled = next == null;
        prevBtn.onclick = () => goto(prev);
        nextBtn.onclick = () => goto(next);
    }


    function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

    function markVizUnavailable(canvas, err) {
        // Replace the canvas with the *actual* failure reason so it's not
        // misdiagnosed as a missing WebGL context when something else
        // (OrbitControls, Three.js itself, etc.) is the real culprit.
        const msg = err ? String(err.message || err) : "unknown error";
        const looksLikeWebGL = /WebGL/i.test(msg);
        const label = looksLikeWebGL
            ? "WebGL unavailable"
            : "3D viz failed to initialize";

        // Show the first stack frame in tiny text so we can locate the
        // throwing line without opening DevTools.
        let where = "";
        if (err && err.stack) {
            const m = err.stack.match(/(viz\.js[^\s)]*)/);
            if (m) where = `<br><span style="font-size:0.72rem;opacity:0.7">at ${escapeHtml(m[1])}</span>`;
        }

        const note = document.createElement("div");
        note.style.cssText =
            "padding:0.75rem;color:var(--muted);font-size:0.84rem;text-align:center";
        note.innerHTML =
            `${label} — <code style="font-size:0.78rem">${escapeHtml(msg)}</code>${where}<br>`
            + "<span style='font-size:0.78rem'>"
            + "(Image preview + sliders still work.)</span>";
        canvas.replaceWith(note);
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
    }


    // ----------------- material info panel -----------------

    /** Fill the static fields (size, thickness) right after bbox.json loads. */
    function populateMaterialInfo() {
        if (!materialBBox || !materialBBox.bbox_size) return;
        const sx = materialBBox.bbox_size[0];
        const sy = materialBBox.bbox_size[1];
        const sz = materialBBox.bbox_size[2];
        document.getElementById("info-size").textContent =
            `${(sx*100).toFixed(1)} × ${(sy*100).toFixed(1)} cm`;
        document.getElementById("info-thickness").textContent =
            `${(sz*1000).toFixed(1)} mm`;
    }

    /**
     * Drive the material-info panel (mean colour swatch + slab colour +
     * per-channel range bars) from the precomputed material stats.
     *
     * The bars use the 0.5 / 99.5 percentile values (lo_rgb_16bit /
     * hi_rgb_16bit), which are robust to one-pixel outliers; the
     * absolute min/max are still in the JSON but not displayed because
     * they get pulled to 0 / 65535 by very rare dark or saturated
     * pixels and aren't representative of the material.
     */
    function applyMaterialStats(stats) {
        if (!stats) return;

        // Mean colour: 16-bit linear → 8-bit sRGB-encoded for display.
        // pow(x, 1/2.2) approximates sRGB encoding well enough for a
        // swatch / box colour. Without this the swatch reads very dark
        // because linear values in 0..255 look much darker than the
        // same numeric value treated as sRGB by the browser.
        const meanLinear = stats.mean_rgb_16bit;
        const toDisplay = (v) =>
            Math.round(255 * Math.pow(Math.max(0, Math.min(1, v / 65535)), 1 / 2.2));
        const mr = toDisplay(meanLinear[0]);
        const mg = toDisplay(meanLinear[1]);
        const mb = toDisplay(meanLinear[2]);
        const hex = "#" + [mr, mg, mb].map(c =>
            c.toString(16).padStart(2, "0")).join("");
        const swatch = document.getElementById("color-swatch");
        const swatchHex = document.getElementById("color-hex");
        if (swatch) swatch.style.background = hex;
        if (swatchHex) swatchHex.textContent = hex;

        // Material slab colour in both scenes follows the mean colour.
        if (b1Scene && b1Scene.setMaterialColor) b1Scene.setMaterialColor([mr, mg, mb]);
        if (w0Scene && w0Scene.setMaterialColor) w0Scene.setMaterialColor([mr, mg, mb]);

        // Per-channel range bars (0..65535).
        const FULL = 65535;
        const lo = stats.lo_rgb_16bit || stats.min_rgb_16bit; // back-compat
        const hi = stats.hi_rgb_16bit || stats.max_rgb_16bit;
        const setBar = (prefix, loV, hiV) => {
            const loR = Math.round(loV);
            const hiR = Math.round(hiV);
            const left  = Math.max(0,   Math.min(100, 100 * loR / FULL));
            const width = Math.max(0.6, Math.min(100, 100 * (hiR - loR) / FULL));
            const fill = document.getElementById(prefix + "fill");
            const minL = document.getElementById(prefix + "min");
            const maxL = document.getElementById(prefix + "max");
            if (fill) {
                fill.style.left = left + "%";
                fill.style.width = width + "%";
            }
            if (minL) minL.textContent = loR.toLocaleString();
            if (maxL) maxL.textContent = hiR.toLocaleString();
        };
        setBar("r", lo[0], hi[0]);
        setBar("g", lo[1], hi[1]);
        setBar("b", lo[2], hi[2]);
    }

    // Populate the static fields immediately (don't wait for the first
    // image to land).
    populateMaterialInfo();

})();
