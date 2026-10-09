/**
 * Capture-geometry viewer (three.js r140, no build step).
 *
 * One scene per panel, in the style of the launch video's trajectory
 * graphic: a dark stage with a soft ground ring, the material swatch at
 * true size (textured, lit by a spot light at the selected LED position),
 * the REAL camera and LED heads as photographic cut-outs placed at the
 * selected 3-D positions, a thin blue view frustum from the lens onto the
 * swatch, and a warm light cone + glow at the LED. Every capture position
 * stays visible as a faint dot (camera: blue, light: warm).
 *
 * Geometry conventions
 *   • Positions are metres, z up, in whatever frame the caller passes
 *     (base1 or world0); the scene itself does not transform frames.
 *   • Directions are true. Distances of the camera / LED (and of their
 *     dots) from the sample centre are multiplied by one factor
 *     (STYLE.distScale, the launch video's ratio) so the enlarged objects
 *     separate from the swatch; the swatch itself is drawn at true size.
 *
 * The cut-outs are billboards: each frame the sprite is rotated in screen
 * space so the lens / emitter points at the sample, foreshortened by the
 * out-of-plane angle, mirrored so the robot side faces outward, sized with
 * depth, and the photographed view (camera: side / above; LED: front /
 * side / rear) is chosen by the out-of-plane angle. This is a port of
 * place_view() in the teaser's tools/elements/gizmo_v2.py.
 */

(() => {
    const ASSET_DIR = "images/viz/";
    const ACCENT = 0x4f8cff;
    const WARM = [1.0, 0.94, 0.84];

    /** Look of the scene; everything in one place so it can be tuned. */
    const STYLE = {
        background: 0x000000,
        distScale: 1 / 1.5,     // camera / LED distances relative to the true-size swatch
        camScale: 3.8 / 1.5,    // camera cut-out size vs true
        ledScale: 5.7 / 1.5,    // LED cut-out size vs true
        frustumMargin: 1.12,    // frustum ends on the swatch corners x this
        ambient: 0.06,
        // spot: cone, partial inverse-square compensation (as the video), and a
        // range of `reach` x the LED-sample distance so grazing pools end
        spot: { angleDeg: 26, beamDeg: 15, intensity: 1.3, refDist: 0.30, comp: 0.5, reach: 1.9, decay: 1.2 },
        floorAlbedo: 0x0e0e10,
        ringColor: 0xc8ccd4, ringOpacity: 0.34, ringWidth: 1.3,
        dotSize: 2.6, camDotColor: 0x5b8fff, lightDotColor: 0xffd9a0,
        camDotOpacity: 0.40, lightDotOpacity: 0.36,
        beamGain: 0.55 * 0.17 * 0.4,
        glow: 0.8, glowAway: 0.5,
        autoRotateSpeed: 0.55,  // OrbitControls units (one turn per 60 s / speed at 60 updates/s)
        spinFps: 30,            // the hover spin is drawn at this rate (speed compensated)
        hoverDelayMs: 350,      // pointer over the panel this long -> spin
        idleResumeMs: 3000,     // after an orbit, spin again (if still hovered) after this
    };

    /**
     * Pixel geometry of the cut-out views (content size, before the
     * power-of-two padding of the PNGs). From the teaser assets
     * (cutouts/views.json, led_views_h3b.json).
     *   anchor : sprite px of the 3-D anchor (camera: mount = camera
     *            origin; LED: the emitter tip)
     *   axis   : sprite-space unit vector of the optical axis / beam (y down)
     *   side   : sprite-space direction of the robot side (mirrored to face
     *            away from the sample on screen)
     *   phi    : out-of-plane angle the view was photographed at (deg)
     *   sel    : angle at which this view is selected (cross-fade between
     *            neighbours, `blend` deg wide)
     */
    const CUTOUTS = {
        cam: {
            blend: 3,
            lensLen: 0.0485,              // m, mount -> lens front (real camera)
            views: [
                { src: "cam_side.png",  tex: [256, 512], size: [147, 387], anchor: [79, 167],
                  axis: [0.0046, 1.0], side: [-1, 0], phi: 5,  sel: 5,  pxPerM: 4555.56 },
                { src: "cam_above.png", tex: [256, 512], size: [163, 433], anchor: [92, 229],
                  axis: [0.0099, 1.0], side: [-1, 0], phi: 33, sel: 33, pxPerM: 4962.96 },
            ],
        },
        led: {
            blend: 8,
            views: [
                { src: "led_front.png", tex: [64, 128],  size: [61, 74],  anchor: [4, 40],
                  axis: [-0.8878, 0.4603], side: [0, -1], phi: -15, sel: -15, pxPerM: 4242.42 },
                { src: "led_side.png",  tex: [64, 128],  size: [50, 83],  anchor: [2, 44],
                  axis: [-0.9993, -0.037], side: [0, -1], phi: 8,   sel: 8,   pxPerM: 4242.42 },
                { src: "led_rear.png",  tex: [128, 128], size: [91, 103], anchor: [2, 51],
                  axis: [-0.9993, -0.037], side: [0, -1], phi: 8,   sel: 34,  pxPerM: 4242.42 },
            ],
        },
    };

    const smoothstep = (e0, e1, x) => {
        const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
        return t * t * (3 - 2 * t);
    };

    /** [[viewIndex, weight]] for an out-of-plane angle phi (deg). */
    function viewWeights(views, phi, band) {
        const V = views.map((v, i) => [i, v.sel]).sort((a, b) => a[1] - b[1]);
        if (phi <= V[0][1]) return [[V[0][0], 1]];
        if (phi >= V[V.length - 1][1]) return [[V[V.length - 1][0], 1]];
        for (let i = 0; i + 1 < V.length; i++) {
            const [ia, a] = V[i], [ib, b] = V[i + 1];
            if (a <= phi && phi <= b) {
                const mid = 0.5 * (a + b);
                const u = band > 0 ? smoothstep(mid - band / 2, mid + band / 2, phi)
                                   : (phi >= mid ? 1 : 0);
                return [[ia, 1 - u], [ib, u]].filter(q => q[1] > 1e-4);
            }
        }
        return [[V[0][0], 1]];
    }

    // ------------------------------------------------------------------
    // Shared image loading (one Image per file; one THREE.Texture per
    // renderer, since textures cannot be shared across WebGL contexts).
    // ------------------------------------------------------------------
    const _images = new Map();
    function loadImage(src) {
        if (!_images.has(src)) {
            _images.set(src, new Promise((res, rej) => {
                const im = new Image();
                im.onload = () => res(im);
                im.onerror = () => rej(new Error(`viz asset failed: ${src}`));
                im.src = ASSET_DIR + src;
            }));
        }
        return _images.get(src);
    }

    function canvasTex(THREE, canvas, srgb) {
        const t = new THREE.CanvasTexture(canvas);
        if (srgb) t.encoding = THREE.sRGBEncoding;
        t.anisotropy = 4;
        return t;
    }

    /** Placeholder swatch: a neutral woven tile in the material's mean colour. */
    function fabricCanvas(rgb255) {
        const N = 256, c = document.createElement("canvas");
        c.width = c.height = N;
        const ctx = c.getContext("2d");
        const img = ctx.createImageData(N, N);
        const base = rgb255 || [150, 150, 150];
        let seed = 12345;
        const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
            const wx = Math.sin((x / N) * Math.PI * 2 * 48), wy = Math.sin((y / N) * Math.PI * 2 * 48);
            const over = ((Math.floor(x / 8) + Math.floor(y / 8)) & 1) ? wx : wy;
            const k = 0.86 + 0.10 * over + 0.08 * (rnd() - 0.5);
            const i = 4 * (y * N + x);
            for (let ch = 0; ch < 3; ch++) img.data[i + ch] = Math.max(0, Math.min(255, base[ch] * k));
            img.data[i + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        return c;
    }

    // ------------------------------------------------------------------
    // Shaders (ShaderMaterial: colours are written as display values)
    // ------------------------------------------------------------------

    /** Screen-space thick lines (segment list -> camera-facing quads). */
    const LINE_VS = `
        attribute vec3 aOther;
        attribute float aSide;
        attribute float aEnd;
        uniform vec2 uRes;
        uniform float uWidth;
        varying float vSide;
        void main() {
            vec4 c0 = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            vec4 c1 = projectionMatrix * modelViewMatrix * vec4(aOther, 1.0);
            vec2 s0 = c0.xy / c0.w * uRes * 0.5;
            vec2 s1 = c1.xy / c1.w * uRes * 0.5;
            vec2 d = (s1 - s0) * (aEnd > 0.5 ? -1.0 : 1.0);   // segment direction, start -> end
            d = length(d) > 1e-5 ? normalize(d) : vec2(1.0, 0.0);
            vec2 n = vec2(-d.y, d.x);
            float hw = 0.5 * uWidth + 1.0;
            c0.xy += n * aSide * hw / (uRes * 0.5) * c0.w;
            vSide = aSide;
            gl_Position = c0;
        }`;
    const LINE_FS = `
        uniform vec3 uColor;
        uniform float uOpacity;
        uniform float uWidth;
        varying float vSide;
        void main() {
            float hw = 0.5 * uWidth + 1.0;
            float a = clamp(hw - abs(vSide) * hw, 0.0, 1.0);
            gl_FragColor = vec4(uColor, uOpacity * a);
        }`;

    /** Round, soft, slightly depth-attenuated dots. */
    const DOT_VS = `
        uniform float uSize;
        uniform float uDpr;
        uniform float uRefDepth;
        void main() {
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mv;
            gl_PointSize = uSize * uDpr * clamp(uRefDepth / max(-mv.z, 1e-3), 0.7, 1.5);
        }`;
    const DOT_FS = `
        uniform vec3 uColor;
        uniform float uOpacity;
        void main() {
            vec2 c = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(c, c);
            if (r2 > 1.0) discard;
            gl_FragColor = vec4(uColor, uOpacity * (1.0 - smoothstep(0.35, 1.0, r2)));
        }`;

    /**
     * Photographic cut-out as a screen-aligned billboard. The quad is
     * parameterised by q in [0,1]^2 (q.y = 0 at the image top); uL maps a
     * sprite-pixel offset from the anchor to a view-space offset (metres,
     * x right, y up) at the anchor's depth.
     */
    const CUT_VS = `
        uniform vec2 uSize;
        uniform vec2 uTex;
        uniform vec2 uAnchor;
        uniform vec4 uL;
        varying vec2 vUv;
        void main() {
            vec2 spx = position.xy * uSize;
            vUv = vec2(spx.x / uTex.x, 1.0 - spx.y / uTex.y);
            vec2 d = spx - uAnchor;
            vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
            mv.xy += vec2(uL.x * d.x + uL.y * d.y, uL.z * d.x + uL.w * d.y);
            gl_Position = projectionMatrix * mv;
        }`;
    const CUT_FS = `
        uniform sampler2D uMap;
        uniform vec2 uTex;
        uniform vec4 uL;
        uniform float uOpacity;
        uniform vec2 uKey;
        uniform float uKeyGain;
        uniform vec2 uWarmDir;
        uniform float uWarmGain;
        uniform vec3 uWarm;
        varying vec2 vUv;
        void main() {
            vec4 c = texture2D(uMap, vUv);
            if (c.a < 0.25) discard;
            // soft studio edge light: the silhouette normal from the alpha
            // gradient, taken to screen space with the sprite's own matrix
            vec2 e = 1.5 / uTex;
            float ax = texture2D(uMap, vUv + vec2(e.x, 0.0)).a - texture2D(uMap, vUv - vec2(e.x, 0.0)).a;
            float ay = texture2D(uMap, vUv + vec2(0.0, e.y)).a - texture2D(uMap, vUv - vec2(0.0, e.y)).a;
            vec2 nS = vec2(-ax, ay);
            float g = length(nS);
            vec2 nV = vec2(uL.x * nS.x + uL.y * nS.y, uL.z * nS.x + uL.w * nS.y);
            nV /= max(length(nV), 1e-6);
            float band = clamp(g * 1.2, 0.0, 1.0);
            float key = uKeyGain * pow(max(dot(nV, uKey), 0.0), 2.0);
            float warm = uWarmGain * pow(max(dot(nV, uWarmDir), 0.0), 1.5);
            vec3 rgb = c.rgb + band * (vec3(key) + warm * uWarm);
            gl_FragColor = vec4(min(rgb, vec3(1.0)), c.a * uOpacity);
        }`;

    /**
     * Additive light layers (cone, glow) are computed in linear light, cut to
     * exactly 0 below a small floor (soft knee, no faint far-field tails) and
     * written sRGB-encoded, as the video composites them over black.
     */
    const LIGHT_GLSL = `
        vec3 lightOut(vec3 lin) {
            lin = max(lin - 0.004, 0.0) / 0.996;
            lin = min(lin, vec3(1.0));
            return mix(12.92 * lin, 1.055 * pow(lin, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, lin));
        }`;

    /** LED glow on a camera-facing quad: warm halo + white-hot core (gizmo_v2.layer_glow). */
    const GLOW_VS = `
        uniform float uHalf;
        varying vec2 vP;
        void main() {
            vP = position.xy * uHalf;
            vec4 mv = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
            mv.xy += vP;
            gl_Position = projectionMatrix * mv;
        }`;
    const GLOW_FS = LIGHT_GLSL + `
        uniform float uS;
        uniform float uK;
        uniform vec3 uWarm;
        varying vec2 vP;
        void main() {
            float r2 = dot(vP, vP);
            float s1 = 1.1 * uS, s2 = 2.2 * uS, s3 = 9.0 * uS, s4 = 26.0 * uS;
            float g = exp(-r2 / (2.0 * s2 * s2)) + 0.30 * exp(-r2 / (2.0 * s3 * s3))
                    + 0.08 * exp(-r2 / (2.0 * s4 * s4));
            float core = 0.8 * exp(-r2 / (2.0 * s1 * s1));
            gl_FragColor = vec4(lightOut(uK * (g * uWarm + vec3(core))), 1.0);
        }`;

    /** Single-scattering light cone (port of gizmo_v2.layer_beam). */
    const BEAM_VS = `
        varying vec3 vWorld;
        void main() {
            vec4 w = modelMatrix * vec4(position, 1.0);
            vWorld = w.xyz;
            gl_Position = projectionMatrix * viewMatrix * w;
        }`;
    const BEAM_FS = LIGHT_GLSL + `
        uniform vec3 uL;
        uniform vec3 uN;
        uniform float uCosC;
        uniform float uCosB;
        uniform float uFloorZ;
        uniform vec3 uCen;
        uniform float uRad;
        uniform float uD;
        uniform float uGain;
        uniform float uUnit;
        uniform vec3 uWarm;
        varying vec3 vWorld;
        void main() {
            vec3 o = cameraPosition;
            vec3 d = normalize(vWorld - o);
            vec3 oc = o - uCen;
            float b = dot(d, oc);
            float disc = b * b - (dot(oc, oc) - uRad * uRad);
            if (disc <= 0.0) discard;
            float sq = sqrt(disc);
            float t0 = max(-b - sq, 0.0);
            float t1 = -b + sq;
            if (d.z < -1e-4) t1 = min(t1, (uFloorZ - o.z) / d.z);
            if (t1 <= t0) discard;
            float dt = (t1 - t0) / 28.0;
            float acc = 0.0;
            for (int i = 0; i < 28; i++) {
                vec3 X = o + (t0 + (float(i) + 0.5) * dt) * d;
                vec3 rel = X - uL;
                float r = length(rel);
                float s = dot(rel, uN) / max(r, 1e-6);
                float rv = max(r * uUnit, 0.06);
                // fade out past the sample (no hard end where the mesh stops)
                float reach = 1.0 - smoothstep(0.75 * uD, 1.15 * uD, r);
                acc += reach * smoothstep(uCosC, uCosB, s) / pow(rv, 1.6);
            }
            acc *= dt * uUnit * uGain;
            gl_FragColor = vec4(lightOut(min(acc, 0.6) * uWarm), 1.0);
        }`;

    // ------------------------------------------------------------------
    // Building blocks
    // ------------------------------------------------------------------

    class FatLines {
        constructor(THREE, nSegments, { color, opacity, width }) {
            this.n = nSegments;
            this.width = width;
            const g = new THREE.BufferGeometry();
            this.pos = new Float32Array(nSegments * 4 * 3);
            this.oth = new Float32Array(nSegments * 4 * 3);
            const side = new Float32Array(nSegments * 4);
            const end = new Float32Array(nSegments * 4);
            const idx = new Uint16Array(nSegments * 6);
            for (let s = 0; s < nSegments; s++) {
                // vertices: start-left, start-right, end-left, end-right
                side.set([-1, 1, -1, 1], s * 4);
                end.set([0, 0, 1, 1], s * 4);
                const v = s * 4;
                idx.set([v, v + 1, v + 3, v, v + 3, v + 2], s * 6);
            }
            g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
            g.setAttribute("aOther", new THREE.BufferAttribute(this.oth, 3));
            g.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
            g.setAttribute("aEnd", new THREE.BufferAttribute(end, 1));
            g.setIndex(new THREE.BufferAttribute(idx, 1));
            this.material = new THREE.ShaderMaterial({
                uniforms: {
                    uColor: { value: new THREE.Color(color) },
                    uOpacity: { value: opacity },
                    uWidth: { value: width },
                    uRes: { value: new THREE.Vector2(1, 1) },
                },
                vertexShader: LINE_VS, fragmentShader: LINE_FS,
                transparent: true, depthWrite: false, side: THREE.DoubleSide,
            });
            this.mesh = new THREE.Mesh(g, this.material);
            this.mesh.frustumCulled = false;
            this.geometry = g;
        }
        /** @param pairs array of [Vector3, Vector3] (at most n) */
        set(pairs) {
            for (let s = 0; s < this.n; s++) {
                const [a, b] = pairs[Math.min(s, pairs.length - 1)];
                const o = s * 12;
                this.pos.set([a.x, a.y, a.z, a.x, a.y, a.z, b.x, b.y, b.z, b.x, b.y, b.z], o);
                this.oth.set([b.x, b.y, b.z, b.x, b.y, b.z, a.x, a.y, a.z, a.x, a.y, a.z], o);
            }
            this.geometry.attributes.position.needsUpdate = true;
            this.geometry.attributes.aOther.needsUpdate = true;
        }
        setViewport(W, H, dpr) {
            this.material.uniforms.uRes.value.set(W, H);
            this.material.uniforms.uWidth.value = this.width * dpr;
        }
    }

    class Cutout {
        constructor(THREE, view, onReady) {
            this.view = view;
            const g = new THREE.BufferGeometry();
            g.setAttribute("position", new THREE.BufferAttribute(
                new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 3));
            g.setIndex([0, 2, 1, 0, 3, 2]);
            this.material = new THREE.ShaderMaterial({
                uniforms: {
                    uMap: { value: null },
                    uSize: { value: new THREE.Vector2(...view.size) },
                    uTex: { value: new THREE.Vector2(...view.tex) },
                    uAnchor: { value: new THREE.Vector2(...view.anchor) },
                    uL: { value: new THREE.Vector4(0, 0, 0, 0) },
                    uOpacity: { value: 1 },
                    uKey: { value: new THREE.Vector2(-0.55, 0.83) },
                    uKeyGain: { value: 0.22 },
                    uWarmDir: { value: new THREE.Vector2(1, 0) },
                    uWarmGain: { value: 0.0 },
                    uWarm: { value: new THREE.Vector3(...WARM) },
                },
                vertexShader: CUT_VS, fragmentShader: CUT_FS,
                // composited over the floor / swatch like the video's 2-D
                // layer (a billboard of a grazing head would otherwise dip
                // into the floor), but it still writes its depth so the
                // lines and dots behind it are hidden
                transparent: true, depthWrite: true, depthFunc: THREE.AlwaysDepth,
                side: THREE.DoubleSide,
            });
            this.mesh = new THREE.Mesh(g, this.material);
            this.mesh.frustumCulled = false;
            this.mesh.visible = false;
            this.ready = false;
            loadImage(view.src).then((im) => {
                const t = new THREE.Texture(im);
                t.minFilter = THREE.LinearMipmapLinearFilter;
                t.needsUpdate = true;
                this.material.uniforms.uMap.value = t;
                this.ready = true;
                if (onReady) onReady();
            }).catch((e) => console.warn("[viz]", e.message));
        }
    }

    // ------------------------------------------------------------------
    // The scene
    // ------------------------------------------------------------------

    class CombinedCaptureScene {
        /**
         * @param {HTMLCanvasElement} canvas
         * @param {object} opts
         *   - cameraPositions, lightPositions : arrays of [x,y,z] (m)
         *   - turntableCenter  : [x,y,z] (m), z = top of the disc
         *   - turntableRadius, turntableThickness
         *   - materialBBox     : bbox.json (world0) — the swatch
         *   - rotateMaterialWithTurnAngle : base1 view (the swatch follows
         *                        the turntable) vs world0 (static)
         *   - showTurntable    : draw the turntable disc (base1 view)
         *   - autoRotate       : gentle orbit while the pointer rests on the
         *                        panel (an idle page draws nothing)
         *   - viewAzimuthDeg / viewElevationDeg : initial eye direction
         */
        constructor(canvas, opts) {
            this.canvas = canvas;
            this.opts = opts;
            this.THREE = window.THREE;
            if (!this.THREE) throw new Error("WebGL viewer: three.js not loaded");
            this._dirty = true;
            this._kick = null;
            this._build();
        }

        /** Something visible changed: draw on the next animation frame. */
        _invalidate() {
            this._dirty = true;
            if (this._kick) this._kick();
        }

        _build() {
            const THREE = this.THREE;
            const o = this.opts;
            const S = STYLE;
            const bb = o.materialBBox && o.materialBBox.bbox_min ? o.materialBBox : null;
            const tc = o.turntableCenter;
            this._rotateMaterial = !!o.rotateMaterialWithTurnAngle;

            // ---- frame anchors ----
            // sample-top centre: the aim point, and (world0) the centre about
            // which camera / LED distances are scaled
            const top = bb ? [(bb.bbox_min[0] + bb.bbox_max[0]) / 2,
                              (bb.bbox_min[1] + bb.bbox_max[1]) / 2, bb.bbox_max[2]]
                           : [tc[0], tc[1], tc[2]];
            this._top0 = new THREE.Vector3(...top);
            this._aim = this._top0.clone();
            this._floorZ = bb ? bb.bbox_min[2] : tc[2];
            this._scaleCenter = this._rotateMaterial
                ? new THREE.Vector3(tc[0], tc[1], top[2]) : this._top0.clone();
            const k = S.distScale;
            this._scaled = (p) => new THREE.Vector3(...p).sub(this._scaleCenter)
                .multiplyScalar(k).add(this._scaleCenter);

            const renderer = new THREE.WebGLRenderer({
                canvas: this.canvas, antialias: true, alpha: false,
                powerPreference: "high-performance",
            });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            const w = this.canvas.clientWidth || 360;
            const h = this.canvas.clientHeight || 360;
            renderer.setSize(w, h, false);
            renderer.outputEncoding = THREE.sRGBEncoding;
            renderer.setClearColor(S.background, 1);
            this.renderer = renderer;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(S.background);
            this.scene = scene;
            this._lines = [];

            // ---- extents ----
            const camPts = (o.cameraPositions || []).map(this._scaled);
            const lightPts = (o.lightPositions || []).map(this._scaled);
            const C = this._scaleCenter;
            let rH = 0.12, rMax = 0.12;
            for (const p of [...camPts, ...lightPts]) {
                rH = Math.max(rH, Math.hypot(p.x - C.x, p.y - C.y));
                rMax = Math.max(rMax, p.distanceTo(C));
            }
            const ringR = rH * 1.06;
            const floorZ = this._floorZ - (o.showTurntable ? (o.turntableThickness || 0.02) : 0) - 0.0008;

            // ---- lights ----
            scene.add(new THREE.AmbientLight(0xffffff, S.ambient));
            const spot = new THREE.SpotLight(new THREE.Color(...WARM), 0, 0,
                S.spot.angleDeg * Math.PI / 180, 1 - S.spot.beamDeg / S.spot.angleDeg, 1);
            scene.add(spot);
            scene.add(spot.target);
            this.spot = spot;

            // ---- floor: a dark disc that fades out, so the light pools on it ----
            const floorTex = (() => {
                const N = 256, c = document.createElement("canvas");
                c.width = c.height = N;
                const ctx = c.getContext("2d");
                const gr = ctx.createRadialGradient(N / 2, N / 2, 0, N / 2, N / 2, N / 2);
                gr.addColorStop(0.0, "#ffffff");
                gr.addColorStop(0.3, "#a8a8a8");
                gr.addColorStop(0.75, "#000000");
                ctx.fillStyle = gr;
                ctx.fillRect(0, 0, N, N);
                return canvasTex(THREE, c, true);
            })();
            const floor = new THREE.Mesh(
                new THREE.CircleGeometry(ringR * 1.45, 96),
                new THREE.MeshStandardMaterial({
                    color: S.floorAlbedo, map: floorTex, roughness: 1, metalness: 0,
                }),
            );
            floor.position.set(C.x, C.y, floorZ);
            scene.add(floor);

            // ---- turntable (base1 view only) ----
            if (o.showTurntable) {
                const R = o.turntableRadius || 0.075;
                const T = o.turntableThickness || 0.02;
                const disc = new THREE.Mesh(
                    new THREE.CylinderGeometry(R, R, T, 96, 1, false),
                    new THREE.MeshStandardMaterial({ color: 0x2e3035, roughness: 0.9, metalness: 0 }),
                );
                disc.rotation.x = Math.PI / 2;
                disc.position.set(tc[0], tc[1], this._floorZ - T / 2);
                scene.add(disc);
                const rim = new FatLines(THREE, 128, { color: S.ringColor, opacity: 0.30, width: 1.0 });
                rim.set(this._circle(new THREE.Vector3(tc[0], tc[1], this._floorZ + 0.0004), R, 128));
                rim.mesh.renderOrder = 2;
                scene.add(rim.mesh);
                this._lines.push(rim);
            }

            // ---- ground ring ----
            const ring = new FatLines(THREE, 192, {
                color: S.ringColor, opacity: S.ringOpacity, width: S.ringWidth });
            ring.set(this._circle(new THREE.Vector3(C.x, C.y, floorZ + 0.0004), ringR, 192));
            ring.mesh.renderOrder = 2;
            scene.add(ring.mesh);
            this._lines.push(ring);

            // ---- swatch (true size), textured + lit ----
            if (bb) this._addSwatch(scene, bb);

            // ---- all capture positions: faint dots ----
            this._dotMats = [];
            for (const [pts, col, op] of [[camPts, S.camDotColor, S.camDotOpacity],
                                          [lightPts, S.lightDotColor, S.lightDotOpacity]]) {
                if (!pts.length) continue;
                const m = new THREE.ShaderMaterial({
                    uniforms: {
                        uColor: { value: new THREE.Color(col) },
                        uOpacity: { value: op },
                        uSize: { value: S.dotSize },
                        uDpr: { value: renderer.getPixelRatio() },
                        uRefDepth: { value: 1 },
                    },
                    vertexShader: DOT_VS, fragmentShader: DOT_FS,
                    transparent: true, depthWrite: false,
                });
                const P = new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts), m);
                P.renderOrder = 3;
                scene.add(P);
                this._dotMats.push(m);
            }

            // ---- the real camera + LED (cut-outs) ----
            const mk = (set) => set.views.map((v) => {
                const c = new Cutout(THREE, v, () => this._invalidate());
                c.mesh.renderOrder = 1;     // before lines / dots: they depth-test against it
                scene.add(c.mesh);
                return c;
            });
            this._camSprites = mk(CUTOUTS.cam);
            this._ledSprites = mk(CUTOUTS.led);

            // ---- view frustum: lens tip -> swatch corners, outline, faint fill ----
            this._frustum = new FatLines(THREE, 4, { color: ACCENT, opacity: 0.55, width: 1.1 });
            this._outline = new FatLines(THREE, 4, { color: ACCENT, opacity: 0.85, width: 1.4 });
            for (const L of [this._frustum, this._outline]) {
                L.mesh.renderOrder = 4;
                L.mesh.visible = false;
                scene.add(L.mesh);
                this._lines.push(L);
            }
            const fillGeom = new THREE.BufferGeometry();
            fillGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 9), 3));
            this._fill = new THREE.Mesh(fillGeom, new THREE.MeshBasicMaterial({
                color: ACCENT, transparent: true, opacity: 0.04, depthWrite: false,
                side: THREE.DoubleSide,
            }));
            this._fill.renderOrder = 4;
            this._fill.frustumCulled = false;
            this._fill.visible = false;
            scene.add(this._fill);

            // ---- LED light cone (volumetric) + glow ----
            const beamGeom = new THREE.ConeGeometry(1, 1, 40, 1, false);   // closed: rays leaving through the base still get a fragment
            beamGeom.translate(0, -0.5, 0);      // apex at the origin, opening along -y
            this._beam = new THREE.Mesh(beamGeom, new THREE.ShaderMaterial({
                uniforms: {
                    uL: { value: new THREE.Vector3() }, uN: { value: new THREE.Vector3(0, 0, -1) },
                    uCosC: { value: Math.cos((S.spot.angleDeg + 4) * Math.PI / 180) },
                    uCosB: { value: Math.cos((S.spot.beamDeg - 4) * Math.PI / 180) },
                    uFloorZ: { value: floorZ },
                    uCen: { value: new THREE.Vector3() }, uRad: { value: 1 }, uD: { value: 1 },
                    uGain: { value: S.beamGain }, uUnit: { value: 1 / k },
                    uWarm: { value: new THREE.Vector3(...WARM) },
                },
                vertexShader: BEAM_VS, fragmentShader: BEAM_FS,
                transparent: true, depthWrite: false, depthTest: false,
                blending: THREE.AdditiveBlending, side: THREE.BackSide,
            }));
            this._beam.renderOrder = 5;
            this._beam.frustumCulled = false;
            this._beam.visible = false;
            scene.add(this._beam);

            const glowGeom = new THREE.BufferGeometry();
            glowGeom.setAttribute("position", new THREE.BufferAttribute(
                new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
            glowGeom.setIndex([0, 1, 2, 0, 2, 3]);
            this._glowUnit = 0.00312 * k;     // the video's glow size unit (m), scaled like the distances
            this._glow = new THREE.Mesh(glowGeom, new THREE.ShaderMaterial({
                uniforms: {
                    uHalf: { value: 3 * 26 * this._glowUnit },
                    uS: { value: this._glowUnit },
                    uK: { value: 0 },
                    uWarm: { value: new THREE.Vector3(...WARM) },
                },
                vertexShader: GLOW_VS, fragmentShader: GLOW_FS,
                transparent: true, depthWrite: false, depthTest: true,
                blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
            }));
            this._glow.frustumCulled = false;
            this._glow.renderOrder = 6;
            this._glow.visible = false;
            scene.add(this._glow);

            // ---- eye + controls ----
            const camera = new THREE.PerspectiveCamera(30, w / h, 0.01, 20);
            camera.up.set(0, 0, 1);
            this.camera = camera;
            const az = (o.viewAzimuthDeg ?? 20) * Math.PI / 180;
            const el = (o.viewElevationDeg ?? 30) * Math.PI / 180;
            const dir = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
            const fit = this._fit(camera, dir, C, rMax, ringR, w / h);
            camera.position.copy(fit.target).addScaledVector(dir, fit.dist);
            camera.lookAt(fit.target);
            this._home = { position: camera.position.clone(), target: fit.target.clone() };

            let controls = null;
            if (typeof THREE.OrbitControls === "function") {
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.target.copy(fit.target);
                controls.enableDamping = true;
                controls.dampingFactor = 0.08;
                controls.rotateSpeed = 0.7;
                controls.enablePan = false;
                controls.minDistance = fit.dist * 0.25;
                controls.maxDistance = fit.dist * 2.2;
                controls.maxPolarAngle = Math.PI * 0.49;    // stay above the floor
                // one auto-rotate step per update(); the spin is drawn at spinFps
                controls.autoRotate = false;
                controls.autoRotateSpeed = S.autoRotateSpeed * 60 / S.spinFps;
                this._interacting = false;
                controls.addEventListener("start", () => { this._interacting = true; });
                controls.addEventListener("end", () => { this._interacting = false; });
                if (o.autoRotate) {
                    // Spin only while a mouse rests on the panel, so an idle
                    // page draws nothing and the spin never fights a drag.
                    let hover = false, timer = null;
                    const spin = (on) => {
                        clearTimeout(timer);
                        controls.autoRotate = on;
                        if (on) this._kick();
                    };
                    const spinLater = (ms) => {
                        clearTimeout(timer);
                        timer = setTimeout(() => spin(hover && !this._interacting), ms);
                    };
                    const el = renderer.domElement;
                    el.addEventListener("pointerenter", (ev) => {
                        if (ev.pointerType === "touch") return;
                        hover = true;
                        spinLater(S.hoverDelayMs);
                    });
                    el.addEventListener("pointerleave", () => { hover = false; spin(false); });
                    controls.addEventListener("start", () => spin(false));
                    controls.addEventListener("end", () => { if (hover) spinLater(S.idleResumeMs); });
                }
                controls.addEventListener("change", () => this._invalidate());
                controls.update();
                renderer.domElement.addEventListener("dblclick", () => this.resetView());
            }
            this.controls = controls;

            // ---- render loop: on demand only (a change, the orbit's damping,
            //      the hover spin), never while off screen or hidden ----
            this._visible = true;
            this.frameCount = 0;
            let raf = 0, lastSpin = 0;
            const tick = (now) => {
                raf = 0;
                if (!this._visible || document.hidden) return;    // kicked again when shown
                const c = this.controls;
                if (c && c.autoRotate && !this._dirty && now - lastSpin < 1000 / S.spinFps - 4) {
                    raf = requestAnimationFrame(tick);            // spinning: skip this frame
                    return;
                }
                const moved = c ? c.update() : false;             // damping / spin step
                if (c && c.autoRotate) lastSpin = now;
                if (this._dirty) {
                    this._dirty = false;
                    this._render();
                }
                if ((moved || (c && c.autoRotate)) && !raf) raf = requestAnimationFrame(tick);
            };
            this._kick = () => { if (!raf) raf = requestAnimationFrame(tick); };
            if ("IntersectionObserver" in window) {
                new IntersectionObserver((es) => {
                    this._visible = es[es.length - 1].isIntersecting;
                    if (this._visible) this._invalidate();
                }).observe(this.canvas);
            }
            document.addEventListener("visibilitychange", () => {
                if (!document.hidden) this._kick();
            });
            this._kick();

            this.resize();
            new ResizeObserver(() => this.resize()).observe(this.canvas);
        }

        /** Back to the initial eye (double-click); auto-rotate stays as it is. */
        resetView() {
            if (!this._home) return;
            this.camera.position.copy(this._home.position);
            if (this.controls) {
                this.controls.target.copy(this._home.target);
                this.controls.update();
            } else {
                this.camera.lookAt(this._home.target);
            }
            this._invalidate();
        }

        _circle(c, r, n) {
            const THREE = this.THREE;
            const out = [];
            for (let i = 0; i < n; i++) {
                const a0 = 2 * Math.PI * i / n, a1 = 2 * Math.PI * (i + 1) / n;
                out.push([new THREE.Vector3(c.x + r * Math.cos(a0), c.y + r * Math.sin(a0), c.z),
                          new THREE.Vector3(c.x + r * Math.cos(a1), c.y + r * Math.sin(a1), c.z)]);
            }
            return out;
        }

        /**
         * Frame the dome of capture positions (radius r about c) + the ground
         * ring from direction `dir`: returns {target, dist} so the projected
         * envelope is centred and fills ~90 % of the shorter side.
         */
        _fit(camera, dir, c, r, ringR, aspect) {
            const THREE = this.THREE;
            // the enlarged heads reach ~0.04 m x scale past their positions:
            // pad the dome sideways and above, not the ring in front
            const pad = 0.05 * STYLE.camScale;
            const pts = [];
            for (let i = 0; i < 48; i++) {
                const a = 2 * Math.PI * i / 48;
                pts.push(new THREE.Vector3(c.x + ringR * Math.cos(a), c.y + ringR * Math.sin(a), this._floorZ));
                for (const e of [0.15, 0.5, 0.9, 1.25]) {
                    const R = r + pad * (0.6 + 0.4 * Math.sin(e));
                    pts.push(new THREE.Vector3(c.x + R * Math.cos(e) * Math.cos(a),
                        c.y + R * Math.cos(e) * Math.sin(a), this._floorZ + R * Math.sin(e)));
                }
            }
            pts.push(new THREE.Vector3(c.x, c.y, this._floorZ + r + pad));
            const tanV = Math.tan(camera.fov * Math.PI / 360);
            const tanH = tanV * aspect;
            const cam = camera.clone();
            const target = new THREE.Vector3(c.x, c.y, this._floorZ + 0.3 * r);
            let dist = r * 3.5;
            for (let it = 0; it < 8; it++) {
                cam.position.copy(target).addScaledVector(dir, dist);
                cam.lookAt(target);
                cam.updateMatrixWorld();
                let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
                for (const p of pts) {
                    const v = p.clone().applyMatrix4(cam.matrixWorldInverse);
                    const z = Math.max(-v.z, 1e-3);
                    x0 = Math.min(x0, v.x / z); x1 = Math.max(x1, v.x / z);
                    y0 = Math.min(y0, v.y / z); y1 = Math.max(y1, v.y / z);
                }
                // recentre vertically (move the target along the eye's up)
                const up = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion);
                target.addScaledVector(up, 0.5 * (y0 + y1) * dist);
                const need = Math.max((x1 - x0) / 2 / tanH, (y1 - y0) / 2 / tanV) / 0.90;
                dist *= Math.pow(need, 0.8);
            }
            return { target, dist };
        }

        _addSwatch(scene, bb) {
            const THREE = this.THREE;
            const sx = bb.bbox_max[0] - bb.bbox_min[0];
            const sy = bb.bbox_max[1] - bb.bbox_min[1];
            const sz = Math.max(0.001, bb.bbox_max[2] - bb.bbox_min[2]);
            const cen = [(bb.bbox_min[0] + bb.bbox_max[0]) / 2,
                         (bb.bbox_min[1] + bb.bbox_max[1]) / 2,
                         (bb.bbox_min[2] + bb.bbox_max[2]) / 2];
            const topMat = new THREE.MeshStandardMaterial({
                color: 0xffffff, roughness: 0.92, metalness: 0,
                map: canvasTex(THREE, fabricCanvas(null), true),
            });
            const sideMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2c, roughness: 1 });
            const box = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz),
                [sideMat, sideMat, sideMat, sideMat, topMat, sideMat]);
            const group = new THREE.Group();
            group.add(box);
            group.position.set(...cen);
            scene.add(group);
            this.materialBox = group;
            this.materialBoxCenter = cen;
            this._swatchTop = topMat;
            this._swatchHalf = [sx / 2, sy / 2];
            this._swatchTopZ = bb.bbox_max[2] - cen[2];
            this._hasRealTexture = false;
        }

        _setSwatchMap(canvas) {
            const old = this._swatchTop.map;
            this._swatchTop.map = canvasTex(this.THREE, canvas, true);
            this._swatchTop.needsUpdate = true;
            if (old) old.dispose();
            this._invalidate();
        }

        /** Placeholder tile in the material's mean colour (until the real one arrives). */
        setMaterialColor(rgb255) {
            if (!this._swatchTop || this._hasRealTexture) return;
            this._setSwatchMap(fabricCanvas(rgb255));
        }

        /** Real top-down tile of the swatch (canvas, sRGB; +x right, +y up). */
        setSwatchTexture(canvas) {
            if (!this._swatchTop) return;
            this._setSwatchMap(canvas);
            this._hasRealTexture = true;
        }

        /** Kept for API compatibility (the cut-outs replace the old icons). */
        attachSelectionModels() {}

        /**
         * base1 view: rotate the swatch by +turnAngle about the calibrated
         * turntable axis / centre (no-op in the world0 view).
         */
        setMaterialTurnAngle(turnAngleDeg) {
            if (!this.materialBox || !this._rotateMaterial) return;
            const THREE = this.THREE;
            const Cc = window.CONFIG.CALIB;
            const axis = new THREE.Vector3(...Cc.TURNTABLE_AXIS).normalize();
            const q = new THREE.Quaternion().setFromAxisAngle(axis, turnAngleDeg * Math.PI / 180);
            const ttc = new THREE.Vector3(...Cc.TURNTABLE_CENTER);
            const cen0 = new THREE.Vector3(...this.materialBoxCenter);
            this.materialBox.position.copy(cen0.clone().sub(ttc).applyQuaternion(q).add(ttc));
            this.materialBox.quaternion.copy(q);
            this._aim = this._top0.clone().sub(ttc).applyQuaternion(q).add(ttc);
            this._updateSelectionGeometry();
        }

        /** Selected camera / light positions (true positions, scene frame). */
        setSelection({ camPos, lightPos }) {
            if (camPos) this._camPos = this._scaled(camPos);
            if (lightPos) this._lightPos = this._scaled(lightPos);
            this._updateSelectionGeometry();
        }

        /** Everything that depends on the selection but not on the eye. */
        _updateSelectionGeometry() {
            const THREE = this.THREE;
            const S = STYLE;
            const aim = this._aim;
            if (this._camPos) {
                const ax = aim.clone().sub(this._camPos).normalize();
                this._camAxis = ax;
                const tip = this._camPos.clone().addScaledVector(ax, CUTOUTS.cam.lensLen * S.camScale);
                const corners = this._swatchCorners(S.frustumMargin);
                this._frustum.set(corners.map((c) => [tip, c]));
                this._outline.set(corners.map((c, i) => [c, corners[(i + 1) % 4]]));
                const P = this._fill.geometry.attributes.position;
                for (let i = 0; i < 4; i++) {
                    const a = corners[i], b = corners[(i + 1) % 4];
                    P.array.set([tip.x, tip.y, tip.z, a.x, a.y, a.z, b.x, b.y, b.z], i * 9);
                }
                P.needsUpdate = true;
                this._frustum.mesh.visible = this._outline.mesh.visible = this._fill.visible = true;
            }
            if (this._lightPos) {
                const L = this._lightPos;
                const n = aim.clone().sub(L).normalize();
                this._ledAxis = n;
                const d = L.distanceTo(aim);
                this.spot.position.copy(L);
                this.spot.target.position.copy(aim);
                this.spot.target.updateMatrixWorld();
                // partial inverse-square compensation (as the video): far LEDs not too dark
                this.spot.distance = S.spot.reach * d;
                this.spot.decay = S.spot.decay;
                this.spot.intensity = S.spot.intensity
                    * Math.pow(S.spot.refDist / Math.max(d, 0.05), 2 * S.spot.comp)
                    / Math.pow(1 - 1 / S.spot.reach, S.spot.decay);   // = the old level at the sample
                // cone mesh: apex at the LED, axis n, reaching past the floor
                const len = d * 1.35;
                const rad = len * Math.tan((S.spot.angleDeg + 7) * Math.PI / 180);
                this._beam.position.copy(L);
                this._beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), n);
                this._beam.scale.set(rad, len, rad);
                const u = this._beam.material.uniforms;
                u.uL.value.copy(L);
                u.uN.value.copy(n);
                u.uCen.value.copy(L).addScaledVector(n, 0.55 * d);
                u.uRad.value = 0.85 * d;
                u.uD.value = d;
                this._beam.visible = true;
                // glow, just in front of the emitter
                this._glow.position.copy(L).addScaledVector(n, 0.004 * S.ledScale);
                this._glow.visible = true;
            }
            this._invalidate();
        }

        _swatchCorners(margin) {
            const THREE = this.THREE;
            const [hx, hy] = this._swatchHalf || [0.05, 0.05];
            const out = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([x, y]) =>
                new THREE.Vector3(x * margin, y * margin, (this._swatchTopZ || 0) + 0.0006));
            if (this.materialBox) {
                this.materialBox.updateMatrixWorld();
                out.forEach((v) => v.applyMatrix4(this.materialBox.matrixWorld));
            }
            return out;
        }

        /**
         * Place one family of cut-outs (camera or LED) for the current eye:
         * pick / cross-fade the photographed views by the out-of-plane angle,
         * aim, foreshorten, mirror and size them (gizmo_v2.place_view).
         * Returns the out-of-plane angle (deg, + = pointing away from us).
         */
        _placeCutouts(sprites, set, anchor, axis, scale, extra, key) {
            const cam = this.camera;
            sprites.forEach((s) => { s.mesh.visible = false; });
            if (!anchor || !axis) return 0;
            const inv = cam.matrixWorldInverse;
            const vA = anchor.clone().applyMatrix4(inv);
            if (vA.z > -1e-3) return 0;                     // behind the eye
            const vT = anchor.clone().addScaledVector(axis, 0.02).applyMatrix4(inv);
            const vC = this._aim.clone().applyMatrix4(inv);
            const img = (v) => [v.x / -v.z, -v.y / -v.z];   // image plane, y down
            const a2 = img(vA), t2 = img(vT), c2 = img(vC);
            let U = [t2[0] - a2[0], t2[1] - a2[1]];
            const nU = Math.hypot(U[0], U[1]);
            U = nU > 1e-9 ? [U[0] / nU, U[1] / nU] : [0, 1];
            const Vv = [-U[1], U[0]];
            let out = [a2[0] - c2[0], a2[1] - c2[1]];
            const nO = Math.hypot(out[0], out[1]);
            out = nO > 1e-9 ? [out[0] / nO, out[1] / nO] : [1, 0];
            const vdir = anchor.clone().sub(cam.position).normalize();
            const phi = Math.asin(Math.max(-1, Math.min(1, axis.dot(vdir)))) * 180 / Math.PI;

            // mirror so the robot side faces away from the sample on screen;
            // with hysteresis, so a sprite seen end-on does not flicker
            this._mirror = this._mirror || {};
            const dOut = out[0] * Vv[0] + out[1] * Vv[1];
            if (!(key in this._mirror) || Math.abs(dOut) > 0.15) this._mirror[key] = dOut >= 0 ? 1 : -1;
            for (const [i, wgt] of viewWeights(set.views, phi, set.blend)) {
                const s = sprites[i], V = s.view;
                if (!s.ready) continue;
                const rho = Math.max(0.55, Math.min(1.15,
                    Math.cos(phi * Math.PI / 180) / Math.cos(V.phi * Math.PI / 180)));
                const us = V.axis, vs = [-us[1], us[0]];
                const sideV = V.side[0] * vs[0] + V.side[1] * vs[1];
                const m = (sideV >= 0 ? 1 : -1) * this._mirror[key];
                // sprite px -> screen px (y down), unit scale
                const L00 = rho * U[0] * us[0] + m * Vv[0] * vs[0];
                const L01 = rho * U[0] * us[1] + m * Vv[0] * vs[1];
                const L10 = rho * U[1] * us[0] + m * Vv[1] * vs[0];
                const L11 = rho * U[1] * us[1] + m * Vv[1] * vs[1];
                const km = scale / V.pxPerM;
                const u = s.material.uniforms;
                u.uL.value.set(km * L00, km * L01, -km * L10, -km * L11);   // -> view, y up
                u.uOpacity.value = wgt;
                if (extra) extra(u, a2);
                s.mesh.position.copy(anchor);
                s.mesh.visible = true;
            }
            return phi;
        }

        _render() {
            const S = STYLE;
            const cam = this.camera;
            cam.updateMatrixWorld();
            // camera: studio key on its silhouette + the LED's warm rim facing the lamp
            let ledScr = null;
            if (this._lightPos) {
                const v = this._lightPos.clone().applyMatrix4(cam.matrixWorldInverse);
                if (v.z < -1e-3) ledScr = [v.x / -v.z, v.y / -v.z];
            }
            this._placeCutouts(this._camSprites, CUTOUTS.cam, this._camPos, this._camAxis, S.camScale,
                (u, a2) => {
                    u.uKeyGain.value = 0.20;
                    if (ledScr) {
                        const dx = ledScr[0] - a2[0], dy = ledScr[1] + a2[1];   // a2 is y-down
                        const n = Math.hypot(dx, dy) || 1;
                        u.uWarmDir.value.set(dx / n, dy / n);
                        u.uWarmGain.value = 0.35;
                    }
                }, "cam");
            const phiL = this._placeCutouts(this._ledSprites, CUTOUTS.led, this._lightPos, this._ledAxis,
                S.ledScale, (u) => { u.uKeyGain.value = 0.30; u.uWarmGain.value = 0.0; }, "led");
            // glow: stronger when the emitter faces us, weaker (never gone) from behind
            if (this._glow.visible) {
                const toward = -Math.sin(phiL * Math.PI / 180);
                let g = S.glow * (0.45 + 0.25 * Math.max(0, toward));
                if (toward < 0) g *= 1 - (1 - S.glowAway) * Math.min(1, -toward);
                this._glow.material.uniforms.uK.value = g;
            }
            const ref = cam.position.distanceTo(this.controls ? this.controls.target : this._aim);
            for (const m of this._dotMats) m.uniforms.uRefDepth.value = ref;
            this.renderer.render(this.scene, cam);
            this.frameCount++;
        }

        resize() {
            const w = this.canvas.clientWidth;
            const h = this.canvas.clientHeight;
            if (w === 0 || h === 0) return;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            if (this.renderer.getPixelRatio() !== dpr) this.renderer.setPixelRatio(dpr);
            this.renderer.setSize(w, h, false);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            const db = this.renderer.getDrawingBufferSize(new this.THREE.Vector2());
            for (const L of this._lines) L.setViewport(db.x, db.y, dpr);
            for (const m of this._dotMats) m.uniforms.uDpr.value = dpr;
            this._invalidate();
        }
    }

    /** True if a WebGL context can be created at all. */
    function webglAvailable() {
        try {
            const c = document.createElement("canvas");
            return !!(window.WebGLRenderingContext &&
                (c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl")));
        } catch (e) { return false; }
    }

    window.Viz = { CombinedCaptureScene, webglAvailable, STYLE };
})();
