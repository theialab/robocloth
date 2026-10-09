/* "Apply a trained material" room panel: shared core (both interaction designs use it).
 *
 * The camera is fixed, so the room is a 2D composite of pre-rendered layers (WebGL1,
 * media/room/, built by tools/room/build_assets.py from low-quality Mitsuba passes):
 *   base.webp        the room with every cloth in untextured grey (1920x900)
 *   masks_<a,b,c,d>  antialiased coverage of the regions, three per image: one region per
 *                    piece (each sofa body, each of the four throw pillows, the four floor
 *                    cushions, the rug, the left curtain)
 *   glow_<a,b,c,d>   soft outer halo of each region (hover highlight)
 *   pos.webp         world position of every pixel (the wipe spreads over the surface)
 *   neural_<m>.webp  every region path-traced in trained material m (the frozen decoder
 *                    evaluated on m's latent texture); loaded the first time m is needed
 *   tiles + swatch_<m>  m's latent tile (meta.tiles) and the same patch decoded flat
 *
 * Every piece is its own drop target. A drop on a sofa body also dresses its pillows while
 * they are still grey. Applying a material starts a clean wipe of neural_<m> over the piece
 * from the drop point, its front at a growing 3D distance on the surface; combinations are
 * composited live from the per-material renders, never rendered.
 *
 * Time comes from window.__demoClock when set (frame-exact scripted recordings), else
 * performance.now(). RoomCore.create() resolves to the API used by room_a.js / room_b.js.
 */
(function () {
    "use strict";

    const REDUCED = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
    const now = () => (window.__demoClock !== undefined ? window.__demoClock : performance.now() / 1000);
    const clamp01 = (x) => Math.min(1, Math.max(0, x));
    const smooth = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
    const easeOut3 = (x) => 1 - Math.pow(1 - clamp01(x), 3);
    const easeInOut = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };

    function loadImg(src) {
        return new Promise((res, rej) => {
            const i = new Image();
            i.decoding = "async";
            i.onload = () => res(i);
            i.onerror = () => rej(new Error("could not load " + src));
            i.src = src;
        });
    }

    /** Call cb once when el comes within ~1.5 screens of the viewport. */
    function whenNear(el, cb) {
        if (!("IntersectionObserver" in window)) { cb(); return; }
        const io = new IntersectionObserver((es) => {
            if (es.some((e) => e.isIntersecting)) { io.disconnect(); cb(); }
        }, { rootMargin: "1200px 0px 1200px 0px" });
        io.observe(el);
    }

    // ------------------------------------------------------------------ GL helpers
    const VS = `attribute vec2 p; varying vec2 vUv;
        void main(){ vUv = vec2(p.x*0.5+0.5, 0.5-p.y*0.5); gl_Position = vec4(p,0.0,1.0); }`;
    const NOISE = `
        float hash(vec3 p){ p = fract(p*0.3183099 + 0.1); p *= 17.0;
                            return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
        float noise(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f*f*(3.0-2.0*f);
            return mix(mix(mix(hash(i),               hash(i+vec3(1,0,0)), f.x),
                           mix(hash(i+vec3(0,1,0)),   hash(i+vec3(1,1,0)), f.x), f.y),
                       mix(mix(hash(i+vec3(0,0,1)),   hash(i+vec3(1,0,1)), f.x),
                           mix(hash(i+vec3(0,1,1)),   hash(i+vec3(1,1,1)), f.x), f.y), f.z); }`;
    /** "Living" latent: rotate PCs 1-3 a little towards PCs 4-6, slowly. */
    const LIVING = `vec3 living(vec3 a, vec3 b, vec3 th){ return 0.5 + cos(th)*(a-0.5) + sin(th)*(b-0.5); }`;

    function makeGL(canvas, opts) {
        const gl = canvas.getContext("webgl", Object.assign({ premultipliedAlpha: true, antialias: false,
            alpha: false, preserveDrawingBuffer: false }, opts || {}));
        if (!gl) throw new Error("WebGL is not available");
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        return gl;
    }
    function program(gl, fs) {
        const mk = (type, src) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
            return s;
        };
        const p = gl.createProgram();
        gl.attachShader(p, mk(gl.VERTEX_SHADER, VS));
        gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
        gl.bindAttribLocation(p, 0, "p");
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        const u = {};
        const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < n; i++) {
            const name = gl.getActiveUniform(p, i).name;
            u[name] = gl.getUniformLocation(p, name);
        }
        return { p, u };
    }
    function texture(gl, img, nearest) {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
        const f = nearest ? gl.NEAREST : gl.LINEAR;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
    }
    function bindTex(gl, prog, name, unit, tex) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(prog.u[name], unit);
    }
    function quad(gl) {
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    /** Per-material living-latent angles (shared by every view of that latent). */
    function theta(mi, t) {
        if (REDUCED) return [0, 0, 0];
        const s = 0.5 + 0.13 * mi;
        return [0, 1, 2].map((k) => 0.42 * Math.sin(s * t + 2.1 * k + 1.3 * mi));
    }

    /** A small live latent tile: WebGL canvas showing lat_<m>_a/_b, slowly drifting. */
    function latentTile(canvas, imgA, imgB, mi) {
        const gl = makeGL(canvas);
        const pr = program(gl, `precision mediump float; varying vec2 vUv;
            uniform sampler2D uA, uB; uniform vec3 uTh; uniform float uTime, uSeed;
            ${LIVING}
            void main(){
                float z = 1.08 + 0.04*sin(uTime*0.21 + uSeed);
                vec2 uv = (vUv - 0.5)/z + 0.5 + 0.025*vec2(sin(uTime*0.13 + uSeed), cos(uTime*0.11 + 2.0*uSeed));
                vec3 c = living(texture2D(uA, uv).rgb, texture2D(uB, uv).rgb, uTh);
                gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
            }`);
        const ta = texture(gl, imgA), tb = texture(gl, imgB);
        return {
            canvas,
            draw(t) {
                gl.viewport(0, 0, canvas.width, canvas.height);
                gl.useProgram(pr.p);
                bindTex(gl, pr, "uA", 0, ta);
                bindTex(gl, pr, "uB", 1, tb);
                gl.uniform3fv(pr.u.uTh, theta(mi, t));
                gl.uniform1f(pr.u.uTime, REDUCED ? 0 : t);
                gl.uniform1f(pr.u.uSeed, mi * 1.7);
                quad(gl);
            },
        };
    }

    /** The frozen-decoder glyph (a six-armed snowflake, as in the method figure / video). */
    function snowflake(ctx, x, y, r, rot, color, lw) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rot);
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.lineCap = "round";
        ctx.beginPath();
        for (let k = 0; k < 6; k++) {
            const a = (k * Math.PI) / 3;
            const c = Math.cos(a), s = Math.sin(a);
            ctx.moveTo(0, 0);
            ctx.lineTo(c * r, s * r);
            for (const f of [0.55]) {
                const bx = c * r * f, by = s * r * f, b = r * 0.32;
                for (const sg of [-1, 1]) {
                    const aa = a + sg * 0.75;
                    ctx.moveTo(bx, by);
                    ctx.lineTo(bx + Math.cos(aa) * b, by + Math.sin(aa) * b);
                }
            }
        }
        ctx.stroke();
        ctx.restore();
    }

    /**
     * A live latent tile on `canvas` for material index mi, from its asset-map spec:
     * "pca" (WebGL, PCs rotating), "frames" (a loop, cross-faded) or "image". All kinds use the
     * same slow zoom / drift (tileView), so the decoded swatch drawn over a tile stays aligned.
     */
    function makeTile(canvas, spec, imgs, id, mi) {
        if (spec.kind === "pca") return latentTile(canvas, imgs["latA_" + id], imgs["latB_" + id], mi);
        const ctx = canvas.getContext("2d");
        const frames = spec.kind === "frames" ? imgs["latF_" + id] : [imgs["latI_" + id]];
        const fps = spec.fps || 8;
        return {
            canvas,
            draw(t) {
                const w = canvas.width, h = canvas.height, v = tileView(mi, t);
                const put = (img, a) => {
                    const iw = img.naturalWidth, ih = img.naturalHeight;
                    ctx.globalAlpha = a;
                    ctx.drawImage(img, v[0] * iw, v[1] * ih, v[2] * iw, v[3] * ih, 0, 0, w, h);
                };
                if (frames.length === 1 || REDUCED) { put(frames[0], 1); return; }
                const x = t * fps, f = Math.floor(x), u = x - f;
                put(frames[((f % frames.length) + frames.length) % frames.length], 1);
                put(frames[(((f + 1) % frames.length) + frames.length) % frames.length], u);
                ctx.globalAlpha = 1;
            },
        };
    }

    /** The part of a latent tile image a live tile shows at time t (same maths as its shader). */
    function tileView(mi, t) {
        const tt = REDUCED ? 0 : t, seed = mi * 1.7;
        const z = 1.08 + 0.04 * Math.sin(tt * 0.21 + seed);
        const ox = 0.025 * Math.sin(tt * 0.13 + seed), oy = 0.025 * Math.cos(tt * 0.11 + 2 * seed);
        return [0.5 - 0.5 / z + ox, 0.5 - 0.5 / z + oy, 1 / z, 1 / z];   // x, y, w, h in 0..1
    }

    /**
     * Draw one "decode" frame on a 2D canvas: the latent, the decoded material revealed by a
     * circle growing from the centre (u in 0..1), a thin bright rim at the front, and the
     * frozen-decoder glyph flashing at the centre while it happens. view = tileView(): the
     * material is cropped like the live latent tile, so latent and material stay texel-aligned.
     */
    function drawDecode(ctx, w, h, latSrc, matImg, view, u, opts) {
        opts = opts || {};
        const R = Math.hypot(w, h) / 2;
        const e = easeInOut(u);
        const r = e * R * 1.02;
        const mw = matImg.naturalWidth || matImg.width, mh = matImg.naturalHeight || matImg.height;
        const src = [view[0] * mw, view[1] * mh, view[2] * mw, view[3] * mh];
        ctx.clearRect(0, 0, w, h);
        if (u < 1 && latSrc) ctx.drawImage(latSrc, 0, 0, w, h);
        if (r > 0.5) {
            ctx.save();
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, r, 0, 2 * Math.PI);
            ctx.clip();
            ctx.drawImage(matImg, src[0], src[1], src[2], src[3], 0, 0, w, h);
            ctx.restore();
        }
        if (u > 0 && u < 1) {
            const rimA = Math.sin(Math.PI * clamp01(u * 1.1));
            ctx.save();
            ctx.globalCompositeOperation = "lighter";
            const g = ctx.createRadialGradient(w / 2, h / 2, Math.max(0, r - w * 0.09), w / 2, h / 2, r + 1);
            g.addColorStop(0, "rgba(255,255,255,0)");
            g.addColorStop(0.8, `rgba(200,222,255,${0.3 * rimA})`);
            g.addColorStop(1, `rgba(255,255,255,${0.6 * rimA})`);
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, r + 1, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore();
        }
        // the frozen-decoder glyph: pops in, turns a little, flashes as the front leaves it
        const gu = opts.glyph === false ? 0 : Math.sin(Math.PI * clamp01(u * 1.15));
        if (gu > 0.01) {
            const gr = w * 0.15 * (0.75 + 0.25 * gu);
            ctx.save();
            ctx.globalAlpha = gu;
            ctx.fillStyle = "rgba(255,255,255,0.93)";
            ctx.shadowColor = "rgba(40,80,170,0.35)";
            ctx.shadowBlur = w * 0.08;
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, gr * 1.25, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore();
            ctx.save();
            ctx.globalAlpha = gu;
            snowflake(ctx, w / 2, h / 2, gr * 0.82, u * 1.2, "#2f6fde", Math.max(1.4, w * 0.018));
            ctx.restore();
        }
    }

    // ------------------------------------------------------------------ the room
    async function create(opts) {
        const A = opts.assets;
        const V = opts.v ? "?v=" + opts.v : "";
        const root = opts.root;
        const q = (sel) => root.querySelector(sel);
        const stage = q(".room-stage"), view = q(".room-view"), canvas = q(".room-canvas");
        const tag = q(".room-tag");

        const meta = await (await fetch(A + "meta.json" + V)).json();
        const W = meta.size[0], H = meta.size[1];
        const MATS = meta.materials;              // [{id, name}]
        const TGT = meta.targets;                 // [{id, label, region, companions, drop, bbox, small}]
        const NREG = meta.regions.length;         // one region per piece (12: furniture + the left curtain)
        const NTEX = Math.ceil(NREG / 3);         // masks_<a,b,c,d>.webp / glow_<..>.webp, 3 regions each
        const LET = "abcdefgh";
        const COARSE = !!(window.matchMedia && matchMedia("(hover: none), (pointer: coarse)").matches);

        const CLS = meta.classes;                 // {of: [class per region], boxes_uv: [[x, y, w, h] 0..1]}
        const NCLS = CLS.boxes_uv.length;
        // resolution set: the 4K renders on high-density wide screens (device px across the
        // panel above the set's min_px), the 1920 px set (the same renders downsampled)
        // otherwise. Layout, hit testing and the soft layers (halos, positions, crosstalk) stay
        // in the 1920 px space; masks, base, grey and material images come from the set.
        const SETS = meta.sets || [{ dir: "", size: [W, H], min_px: 0 }];
        const devPx = (window.devicePixelRatio || 1) * root.getBoundingClientRect().width;
        const SET = opts.set !== undefined ? SETS[opts.set] : SETS.filter((st) => devPx > st.min_px).pop() || SETS[0];
        const SD = SET.dir, SW = SET.size[0], SH = SET.size[1];
        // base first, so the panel shows the room while the rest streams in
        const imgs = {};
        imgs.base = await loadImg(A + SD + "base.webp" + V);
        stage.style.backgroundImage = `url(${A}${SD}base.webp${V})`;
        const jobs = [["pos", "pos.webp"], ["hdr", "hdr.webp"]];
        for (let i = 0; i < NTEX; i++) {
            jobs.push(["mask" + i, `masks_${LET[i]}.webp`], ["glow" + i, `glow_${LET[i]}.webp`]);
            if (SD) jobs.push(["smask" + i, `${SD}masks_${LET[i]}.webp`]);
        }
        for (let k = 0; k < NCLS; k++) jobs.push(["grey" + k, `${SD}grey_${k}.webp`]);
        // the latent tile of each material comes from one asset map (meta.tiles), so other tile
        // images can be swapped in by changing paths: {kind: "pca", a, b} (PCs 1-3 / 4-6,
        // rotated into each other live), {kind: "frames", frames: [...], fps} (a loop) or
        // {kind: "image", src}. Paths are relative to the assets folder (or absolute URLs).
        // A material's rendered room (neural_<id>, ~0.3 MB) loads the first time it is needed
        // (core.need), except those listed in opts.preload.
        const TILES = meta.tiles || {};
        const url = (f) => (/^(https?:)?\/\//.test(f) ? f : A + f) + (f.includes("?") ? "" : V);
        const tileJobs = [];
        MATS.forEach((m) => {
            jobs.push(["swatch_" + m.id, `swatch_${m.id}.webp`]);
            const spec = TILES[m.id] || { kind: "pca", a: `lat_${m.id}_a.webp`, b: `lat_${m.id}_b.webp` };
            m.tile = spec;
            if (spec.kind === "pca") jobs.push(["latA_" + m.id, spec.a], ["latB_" + m.id, spec.b]);
            else if (spec.kind === "image") jobs.push(["latI_" + m.id, spec.src]);
            else if (spec.kind === "frames") tileJobs.push(Promise.all(spec.frames.map((f) => loadImg(url(f))))
                .then((fr) => (imgs["latF_" + m.id] = fr)));
        });
        await Promise.all(jobs.map(([k, f]) => loadImg(url(f)).then((i) => (imgs[k] = i))).concat(tileJobs));

        // CPU copies of masks + positions: hit testing, drop point, wipe extent
        function pixels(img) {
            const c = document.createElement("canvas");
            c.width = W; c.height = H;
            const x = c.getContext("2d", { willReadFrequently: true });
            x.drawImage(img, 0, 0);
            return x.getImageData(0, 0, W, H).data;
        }
        const mpx = Array.from({ length: NTEX }, (_, i) => pixels(imgs["mask" + i]));
        const posPx = pixels(imgs.pos);
        const regCov = (i, r) => mpx[(r / 3) | 0][4 * i + (r % 3)];
        const pmin = meta.pos_min, pmax = meta.pos_max;
        const posAt = (i) => [0, 1, 2].map((k) => pmin[k] + (posPx[4 * i + k] / 255) * (pmax[k] - pmin[k]));
        const regPts = Array.from({ length: NREG }, () => []);
        for (let y = 0; y < H; y += 3) for (let x = 0; x < W; x += 3) {
            const i = y * W + x;
            for (let r = 0; r < NREG; r++) if (regCov(i, r) > 127) regPts[r].push(posAt(i));
        }
        const regionToTarget = new Array(NREG).fill(-1);
        TGT.forEach((t, k) => { regionToTarget[t.region] = k; });
        const GROW = TGT.map((t) => !!t.grow), YIELDS = TGT.map((t) => !!t.yields);

        function regionAt(px, py) {
            const x = Math.round(px), y = Math.round(py);
            if (x < 0 || y < 0 || x >= W || y >= H) return -1;
            let best = -1, bv = 110;
            for (let dy = -4; dy <= 4; dy += 4) for (let dx = -4; dx <= 4; dx += 4) {
                const xx = Math.min(W - 1, Math.max(0, x + dx)), yy = Math.min(H - 1, Math.max(0, y + dy));
                const i = yy * W + xx;
                for (let r = 0; r < NREG; r++) {
                    const v = regCov(i, r) - (dx || dy ? 40 : 0);
                    if (v > bv) { bv = v; best = r; }
                }
            }
            return best;
        }
        /** Touch: a floor cushion within reach of the finger wins over the rug or bare floor
         * under it (bigger hit areas for the small cushions); sofas and pillows hit exactly. */
        function regionNear(px, py, rad) {
            const exact = regionAt(px, py);
            if (exact >= 0 && !YIELDS[regionToTarget[exact]]) return exact;
            let best = -1, bd = rad * rad;
            const st = 4;
            for (let dy = -rad; dy <= rad; dy += st) for (let dx = -rad; dx <= rad; dx += st) {
                const d2 = dx * dx + dy * dy;
                if (d2 >= bd) continue;
                const x = Math.round(px + dx), y = Math.round(py + dy);
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                const i = y * W + x;
                for (let r = 0; r < NREG; r++) {
                    if (!GROW[regionToTarget[r]] || regCov(i, r) < 128) continue;
                    bd = d2; best = r;
                }
            }
            return best >= 0 ? best : exact;
        }
        function dropPoint(regs, px, py) {
            const x0 = Math.round(px), y0 = Math.round(py);
            for (let r = 0; r < 60; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
                if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
                const x = x0 + dx, y = y0 + dy;
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                const i = y * W + x;
                for (const g of regs) if (regCov(i, g) > 200) return posAt(i);
            }
            return regPts[regs[0]][0];
        }

        // ------------------------------------------------------------------ GL
        canvas.width = SW; canvas.height = SH;
        const gl = makeGL(canvas);
        const T = {};
        for (const k of ["base", "pos", "hdr"].concat(Array.from({ length: NTEX }, (_, i) => ["mask" + i, "glow" + i]).flat(),
            Array.from({ length: NCLS }, (_, k) => "grey" + k)))
            T[k] = texture(gl, imgs[k], k === "pos");
        // masks at the render resolution (4K set) for the compositing; the 1920 ones for halos
        const MT = Array.from({ length: NTEX }, (_, i) => (SD ? texture(gl, imgs["smask" + i]) : T["mask" + i]));
        if (SD) for (let i = 0; i < NTEX; i++) delete imgs["smask" + i];     // (CPU copies not needed)
        // the composite (base + every applied material + its crosstalk) is cached in a texture
        // and only redrawn while something changes
        const fbTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, fbTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, SW, SH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fbTex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        let dirty = true;

        // a material's rendered pieces (one image per class) + its crosstalk ratios: fetched and
        // uploaded the first time the material is needed (~0.3-0.5 MB)
        const neural = {};                         // id -> {p: Promise, cls: [tex], ratio: tex}
        let wake = null;                           // the render loop's kick, once it exists
        function need(mi) {
            if (mi < 0) return Promise.resolve();
            const id = MATS[mi].id;
            if (!neural[id]) {
                const n = neural[id] = { cls: null, ratio: null };
                n.p = Promise.all(Array.from({ length: NCLS }, (_, k) => loadImg(url(`${SD}mat_${id}_${k}.webp`)))
                    .concat([loadImg(url(`ratio_${id}.webp`))])).then((im) => {
                    n.cls = im.slice(0, NCLS).map((i) => texture(gl, i));
                    n.ratio = texture(gl, im[NCLS]);
                    dirty = true;
                    if (wake) wake();
                });
            }
            return neural[id].p;
        }
        const ready = (mi) => mi < 0 || !!(neural[MATS[mi].id] && neural[MATS[mi].id].cls);
        await Promise.all((opts.preload || meta.preload || []).map((id) => need(MATS.findIndex((m) => m.id === id))));

        const progBase = program(gl, `precision mediump float; varying vec2 vUv; uniform sampler2D uBase;
            void main(){ gl_FragColor = vec4(texture2D(uBase, vUv).rgb, 1.0); }`);

        const MS = Array.from({ length: NTEX }, (_, i) => "uMask" + i);
        const covExpr = (w) => MS.map((m, i) => `dot(texture2D(${m}, vUv).rgb, ${w}${i})`).join(" + ");
        const decl = (pre) => Array.from({ length: NTEX }, (_, i) => pre + i).join(", ");
        const progBlit = program(gl, `precision mediump float; varying vec2 vUv; uniform sampler2D uTex;
            void main(){ gl_FragColor = vec4(texture2D(uTex, vec2(vUv.x, 1.0 - vUv.y)).rgb, 1.0); }`);
        // one piece changing material (old -> new) behind a wipe from the drop point: adds
        // cov * inside * (new - old) to the composite, as two passes (positive part added,
        // negative part subtracted), so antialiased edges stay exact whatever the neighbours wear;
        // a slightly brighter sheen rides just behind the front and a faint light line ahead
        const progDelta = program(gl, `precision highp float; varying vec2 vUv;
            uniform sampler2D uMask, uPos, uNew, uOld;
            uniform vec3 uChan, uDrop, uPmin, uPmax;
            uniform vec4 uBox;
            uniform float uR, uFade, uSheen, uSign;
            ${NOISE}
            void main(){
                float cov = dot(texture2D(uMask, vUv).rgb, uChan);
                if (cov < 0.002) discard;
                vec2 cu = (vUv - uBox.xy) / uBox.zw;
                vec3 cn = texture2D(uNew, cu).rgb, co = texture2D(uOld, cu).rgb;
                vec3 P = uPmin + texture2D(uPos, vUv).rgb * (uPmax - uPmin);
                float d = distance(P, uDrop) + (noise(P*14.0) - 0.5)*0.025;
                float inside = 1.0 - smoothstep(uR - 0.028, uR, d);
                float ahead = exp(-pow((d - uR - 0.008) / 0.012, 2.0));
                float sheen = smoothstep(uR - 0.09, uR - 0.005, d) * inside * uSheen;
                vec3 delta = cov * inside * uFade * (cn - co);
                vec3 extra = cov * uFade * (sheen * 0.12 * (vec3(1.0) - cn) + vec3(1.0, 0.98, 0.95) * ahead * 0.10 * uSheen);
                vec3 v = uSign > 0.0 ? max(delta, 0.0) + extra : max(-delta, 0.0);
                gl_FragColor = vec4(v, 1.0);
            }`);
        // first-order crosstalk: a dressed piece changes the light reaching the rest of the room
        // by ratio_r = lo_piece / lo_grey (smooth, 480 px, cross-faded old -> new with the
        // wipe). Low-res: the current ratio of every piece into a 3x3 atlas, their product P into
        // one tile; full-res, one pass: f = P / prod_r ratio_r^cov_r (no change on the piece
        // itself). Darkening multiplies, brightening adds dst * (f - 1) (8-bit target).
        const RT = meta.ratio, RW = RT.tile[0], RH = RT.tile[1], RC = RT.cols, OFF = RT.offset.toFixed(3);
        const lowTarget = (w, h) => {
            const tx = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tx);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
            const f = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, f);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx, 0);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            return { tx, f, w, h };
        };
        const curAtlas = lowTarget(RC * RW, RC * RH), prodT = lowTarget(RW, RH);
        // (FBO-rendered textures are stored bottom-up: sample them with y flipped)
        const progRMix = program(gl, `precision mediump float; varying vec2 vUv;
            uniform sampler2D uRNew, uROld; uniform vec2 uTile, uTileSz;
            uniform float uK, uHasNew, uHasOld;
            void main(){
                vec2 tu = uTile + clamp(vUv, 0.003, 0.997) * uTileSz;
                vec3 rn = uHasNew > 0.5 ? texture2D(uRNew, tu).rgb : vec3(0.5);
                vec3 ro = uHasOld > 0.5 ? texture2D(uROld, tu).rgb : vec3(0.5);
                gl_FragColor = vec4(mix(ro, rn, uK), 1.0);       // ratio - ${OFF}
            }`);
        const tileU = (r) => [(r % RC) / RC, ((r / RC) | 0) / RC];
        const progRProd = program(gl, `precision mediump float; varying vec2 vUv;
            uniform sampler2D uAtlas;
            void main(){
                vec2 uv = clamp(vUv, 0.003, 0.997) / ${RC.toFixed(1)};   // image coords (top-down)
                vec3 p = vec3(1.0);
                ${Array.from({ length: NREG }, (_, r) => `p *= texture2D(uAtlas, vec2(uv.x + ${tileU(r)[0].toFixed(5)}, 1.0 - (uv.y + ${tileU(r)[1].toFixed(5)}))).rgb + ${OFF};`).join("\n                ")}
                gl_FragColor = vec4(clamp(p - ${OFF}, 0.0, 1.0), 1.0);
            }`);
        const progRApply = program(gl, `precision mediump float; varying vec2 vUv;
            uniform sampler2D ${MS.join(", ")}, uAtlas, uProd, uHdr;
            uniform float uFade, uMode;
            void main(){
                vec3 p = texture2D(uProd, vec2(vUv.x, 1.0 - vUv.y)).rgb + ${OFF};
                vec2 uv = clamp(vUv, 0.003, 0.997) / ${RC.toFixed(1)};
                vec3 own = vec3(1.0);
                ${Array.from({ length: NREG }, (_, r) => `{ float c = texture2D(${MS[(r / 3) | 0]}, vUv)[${r % 3}];
                  if (c > 0.002) own *= pow(texture2D(uAtlas, vec2(uv.x + ${tileU(r)[0].toFixed(5)}, 1.0 - (uv.y + ${tileU(r)[1].toFixed(5)}))).rgb + ${OFF}, vec3(c)); }`).join("\n                ")}
                vec3 f = 1.0 + (p / own - 1.0) * uFade;
                // the base's light above display white (window): a clipped pixel only darkens
                // once L * f drops below 1
                vec3 L = exp2(texture2D(uHdr, vUv).rgb * 2.0);
                f = mix(f, min(L * f, 1.0), step(1.02, L));
                f = pow(max(f, 0.0), vec3(1.0 / 2.2));          // the composite is sRGB-encoded
                gl_FragColor = vec4(uMode < 0.5 ? min(f, 1.0) : max(f - 1.0, 0.0), 1.0);
            }`);

        const GS = Array.from({ length: NTEX }, (_, i) => "uGlow" + i);
        const progHL = program(gl, `precision mediump float; varying vec2 vUv;
            uniform sampler2D ${MS.join(", ")}, ${GS.join(", ")};
            uniform vec3 ${decl("uI")}, ${decl("uG")};
            void main(){
                float inner = ${covExpr("uI")};
                float halo = ${GS.map((g, i) => `dot(texture2D(${g}, vUv).rgb, uG${i})`).join(" + ")};
                vec3 add = vec3(1.0) * inner * 0.075 + vec3(0.78, 0.87, 1.0) * halo * 0.8;
                gl_FragColor = vec4(add, 0.0);
            }`);

        // ------------------------------------------------------------------ state
        // one layer per piece and change: {r, mi, from, drop, t0, rmax, wait}; the composite is
        // base + sum over layers of cov_r * inside(t) * (C_mi - C_from), and a finished layer
        // folds the earlier ones on its piece into itself (from := grey)
        const layers = [];
        const regionMat = new Array(NREG).fill(-1);   // what each region shows once settled
        const hover = new Array(TGT.length).fill(0), hoverTo = new Array(TGT.length).fill(0);
        const focus = new Array(TGT.length).fill(0), focusTo = new Array(TGT.length).fill(0);
        let pick = 0, pickTo = 0;     // all targets pulse (a material is in hand)
        let fading = null;
        let lastT = null;
        const listeners = [];

        function wipeDur(L) { return REDUCED ? 0.35 : Math.min(1.5, Math.max(0.75, 0.6 + 0.5 * L.rmax)); }
        function radius(L, t) {
            const x = clamp01(t / wipeDur(L));
            return 0.02 + (L.rmax + 0.08) * (0.25 * x + 0.75 * easeOut3(x));
        }
        const layerDone = (L, t) => !L.wait && t >= wipeDur(L);

        /** Regions a drop on target k dresses: the piece itself, plus its companions (a sofa's
         * pillows) while they are still grey, so a first drop on a sofa dresses it whole but
         * pillows set explicitly are kept. */
        function regionsFor(k) {
            const t = TGT[k];
            return [t.region].concat((t.companions || []).filter((g) => regionMat[g] === -1));
        }
        function apply(k, mi, px, py) {
            const regs = regionsFor(k);
            const drop = dropPoint(regs, px, py);
            let rmax = 0;
            for (const g of regs) for (const p of regPts[g]) {
                const d = Math.hypot(p[0] - drop[0], p[1] - drop[1], p[2] - drop[2]);
                if (d > rmax) rmax = d;
            }
            if (fading !== null) { layers.length = 0; fading = null; }
            const t0 = now();
            const group = [];
            for (const r of regs) {
                // (the same material again: a sheen-only ripple, nothing changes)
                // a piece still being wiped: finish that change at once (rare double drop)
                for (const L of layers) if (L.r === r && !L.wait) L.t0 = Math.min(L.t0, t0 - 10);
                const L = { r, mi, from: regionMat[r], drop, t0, rmax, wait: false };
                layers.push(L);
                group.push(L);
                regionMat[r] = mi;
            }
            if (group.length && !ready(mi)) {        // starts when its renders have arrived
                group.forEach((L) => { L.wait = true; });
                need(mi).then(() => { const tn = now(); group.forEach((L) => { L.wait = false; L.t0 = tn; }); dirty = true; });
            }
            dirty = true;
            if (opts.panOnApply !== false) panTo(k);
            listeners.forEach((f) => f("apply", { k, mi }));
        }
        let fadeDur = 0.6;
        /** back to grey: a cross-fade of every applied material (dur s, default 0.6) */
        function reset(dur) {
            if (!layers.length) return;
            fading = now();
            fadeDur = REDUCED ? 0.2 : (dur || 0.6);
            regionMat.fill(-1);
            dirty = true;
            listeners.forEach((f) => f("reset", {}));
        }

        const clsTex = (mi, r) => (mi < 0 ? T["grey" + CLS.of[r]] : neural[MATS[mi].id].cls[CLS.of[r]]);
        function compose(t, fade) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
            gl.viewport(0, 0, SW, SH);
            gl.disable(gl.BLEND);
            gl.useProgram(progBase.p);
            bindTex(gl, progBase, "uBase", 0, T.base);
            quad(gl);
            const live = layers.filter((L) => !L.wait);
            gl.colorMask(true, true, true, false);      // the composite's alpha stays 1
            if (live.length) {
                gl.enable(gl.BLEND);
                gl.blendFunc(gl.ONE, gl.ONE);
                const P = progDelta;
                gl.useProgram(P.p);
                bindTex(gl, P, "uPos", 1, T.pos);
                gl.uniform3fv(P.u.uPmin, pmin);
                gl.uniform3fv(P.u.uPmax, pmax);
                gl.uniform1f(P.u.uFade, fade);
                gl.enable(gl.SCISSOR_TEST);
                for (const L of live) {
                    const lt = t - L.t0, b = CLS.boxes_uv[CLS.of[L.r]];
                    const bb = TGT[regionToTarget[L.r]].bbox, sx = SW / W, sy = SH / H;
                    gl.scissor(Math.max(0, Math.floor((bb[0] - 4) * sx)), Math.max(0, Math.floor((H - bb[3] - 5) * sy)),
                        Math.ceil((bb[2] - bb[0] + 9) * sx), Math.ceil((bb[3] - bb[1] + 9) * sy));
                    bindTex(gl, P, "uMask", 0, MT[(L.r / 3) | 0]);
                    bindTex(gl, P, "uNew", 2, clsTex(L.mi, L.r));
                    bindTex(gl, P, "uOld", 3, clsTex(L.from, L.r));
                    gl.uniform3fv(P.u.uChan, [0, 1, 2].map((c) => (c === L.r % 3 ? 1 : 0)));
                    gl.uniform4fv(P.u.uBox, b);
                    gl.uniform3fv(P.u.uDrop, L.drop);
                    gl.uniform1f(P.u.uR, radius(L, lt));
                    gl.uniform1f(P.u.uSheen, REDUCED ? 0 : 1 - smooth((lt - wipeDur(L) * 0.75) / (wipeDur(L) * 0.35)));
                    for (const sign of [1, -1]) {
                        gl.blendEquation(sign > 0 ? gl.FUNC_ADD : gl.FUNC_REVERSE_SUBTRACT);
                        gl.uniform1f(P.u.uSign, sign);
                        quad(gl);
                    }
                }
                gl.blendEquation(gl.FUNC_ADD);
                gl.disable(gl.SCISSOR_TEST);
                // crosstalk: every piece's current ratio (its latest change, old -> new) ...
                gl.disable(gl.BLEND);
                gl.bindFramebuffer(gl.FRAMEBUFFER, curAtlas.f);
                const M = progRMix;
                gl.useProgram(M.p);
                gl.uniform2fv(M.u.uTileSz, [1 / RC, 1 / RC]);
                let anyRatio = false;
                for (let r = 0; r < NREG; r++) {
                    let L = null;
                    for (const x of live) if (x.r === r) L = x;
                    const c = r % RC, row = (r / RC) | 0;
                    gl.viewport(c * RW, (RC - 1 - row) * RH, RW, RH);          // FBO rows are bottom-up
                    const hasN = !!L && L.mi >= 0, hasO = !!L && L.from >= 0;
                    anyRatio = anyRatio || hasN || hasO;
                    if (hasN) bindTex(gl, M, "uRNew", 0, neural[MATS[L.mi].id].ratio);
                    if (hasO) bindTex(gl, M, "uROld", 1, neural[MATS[L.from].id].ratio);
                    gl.uniform1f(M.u.uHasNew, hasN ? 1 : 0);
                    gl.uniform1f(M.u.uHasOld, hasO ? 1 : 0);
                    gl.uniform2fv(M.u.uTile, tileU(r));
                    gl.uniform1f(M.u.uK, L ? smooth((t - L.t0) / wipeDur(L)) : 0);
                    quad(gl);
                }
                if (anyRatio && !window.__roomNoCrosstalk) {
                    // ... their product ...
                    gl.bindFramebuffer(gl.FRAMEBUFFER, prodT.f);
                    gl.viewport(0, 0, RW, RH);
                    gl.useProgram(progRProd.p);
                    bindTex(gl, progRProd, "uAtlas", 0, curAtlas.tx);
                    quad(gl);
                    // ... applied at full resolution, minus each piece's own ratio on itself
                    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
                    gl.viewport(0, 0, SW, SH);
                    const Q = progRApply;
                    gl.useProgram(Q.p);
                    MS.forEach((m, i) => bindTex(gl, Q, m, i, MT[i]));
                    bindTex(gl, Q, "uAtlas", NTEX, curAtlas.tx);
                    bindTex(gl, Q, "uProd", NTEX + 1, prodT.tx);
                    bindTex(gl, Q, "uHdr", NTEX + 2, T.hdr);
                    gl.uniform1f(Q.u.uFade, fade);
                    gl.enable(gl.BLEND);
                    gl.uniform1f(Q.u.uMode, 0);
                    gl.blendFunc(gl.DST_COLOR, gl.ZERO);
                    quad(gl);
                    gl.uniform1f(Q.u.uMode, 1);
                    gl.blendFunc(gl.DST_COLOR, gl.ONE);
                    quad(gl);
                }
            }
            gl.disable(gl.BLEND);
            gl.colorMask(true, true, true, true);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }

        function draw(t) {
            // a finished layer folds the earlier ones on its piece into itself
            for (let i = layers.length - 1; i >= 0; i--) {
                const L = layers[i];
                if (!layerDone(L, t - L.t0)) continue;
                let folded = false;
                for (let j = i - 1; j >= 0; j--) {
                    if (layers[j].r === L.r) { layers.splice(j, 1); i--; folded = true; }
                }
                if (folded) L.from = -1;
                if (L.mi < 0 && L.from < 0) { layers.splice(i, 1); }      // back to grey = the base
            }
            let fade = 1;
            if (fading !== null) {
                fade = 1 - smooth((t - fading) / fadeDur);
                if (fade <= 0) { layers.length = 0; fading = null; fade = 1; dirty = true; }
            }
            const animating = fading !== null || layers.some((L) => !L.wait && t - L.t0 < wipeDur(L) + 0.05);
            if (dirty || animating) { compose(t, fade); dirty = animating; }
            gl.viewport(0, 0, SW, SH);
            gl.disable(gl.BLEND);
            gl.useProgram(progBlit.p);
            bindTex(gl, progBlit, "uTex", 0, fbTex);
            quad(gl);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

            // highlights
            const dt = lastT === null ? 0 : Math.min(0.1, Math.max(0, t - lastT));
            lastT = t;
            let any = false;
            for (let k = 0; k < TGT.length; k++) {
                hover[k] += (hoverTo[k] - hover[k]) * Math.min(1, dt * 14);
                focus[k] += (focusTo[k] - focus[k]) * Math.min(1, dt * 10);
                if (hover[k] > 0.004 || focus[k] > 0.004) any = true;
            }
            pick += (pickTo - pick) * Math.min(1, dt * 8);
            if (any || pick > 0.004) {
                const I = new Array(3 * NTEX).fill(0), G = new Array(3 * NTEX).fill(0);
                const pulse = 0.82 + 0.18 * Math.sin(t * 4.0);
                const pp = 0.5 + 0.5 * Math.sin(t * 3.0);
                TGT.forEach((tg, k) => {
                    const h = Math.max(hover[k], focus[k] * (0.75 + 0.25 * pp));
                    if (h < 0.002) return;
                    const inner = hover[k] + focus[k] * 0.6;
                    // what a drop would dress: the piece (+ its still-grey companions)
                    for (const g of regionsFor(k)) I[g] += inner;
                    G[tg.region] += h * pulse;
                });
                // a material in hand: every piece breathes softly
                const pk = pick * (0.16 + 0.2 * pp);
                TGT.forEach((tg) => { if (!tg.small) G[tg.region] += pk; });
                const P = progHL;
                gl.useProgram(P.p);
                MS.forEach((m, i) => bindTex(gl, P, m, i, T["mask" + i]));
                GS.forEach((g, i) => bindTex(gl, P, g, NTEX + i, T["glow" + i]));
                for (let i = 0; i < NTEX; i++) {
                    gl.uniform3fv(P.u["uI" + i], I.slice(3 * i, 3 * i + 3).map((x) => Math.min(1.2, x)));
                    gl.uniform3fv(P.u["uG" + i], G.slice(3 * i, 3 * i + 3).map((x) => Math.min(1.2, x)));
                }
                quad(gl);
            }
            stepPan(t);
        }

        // ------------------------------------------------------------------ geometry
        function roomPx(cx, cy) {
            const r = canvas.getBoundingClientRect();
            if (cx < r.left || cx > r.right || cy < r.top || cy > r.bottom) return null;
            // the visible part of the canvas (phones: the view clips the zoomed stage)
            const vr = view.getBoundingClientRect();
            if (cx < vr.left || cx > vr.right || cy < vr.top || cy > vr.bottom) return null;
            return [((cx - r.left) / r.width) * W, ((cy - r.top) / r.height) * H];
        }
        /** touch: true for a finger (bigger hit areas for small pieces, ~18 CSS px) */
        function targetAt(cx, cy, touch) {
            const p = roomPx(cx, cy);
            if (!p) return null;
            let g;
            if (touch === undefined ? COARSE : touch) {
                const rad = Math.round(18 * W / canvas.getBoundingClientRect().width);
                g = regionNear(p[0], p[1], rad);
            } else g = regionAt(p[0], p[1]);
            if (g < 0) return null;
            return { k: regionToTarget[g], px: p[0], py: p[1] };
        }
        function clientOf(px, py) {
            const r = canvas.getBoundingClientRect();
            return [r.left + (px / W) * r.width, r.top + (py / H) * r.height];
        }
        function stageOf(px, py) {             // stage-local CSS px
            return [(px / W) * stage.clientWidth, (py / H) * stage.clientHeight];
        }
        function labelOf(k) {
            const t = TGT[k];
            return regionsFor(k).length > 1 && t.labelWith ? t.labelWith : t.label;
        }

        // hover label (in stage coordinates, so it scrolls with the zoomed room on phones)
        /** strength < 1: a faint outline (nothing in hand), and the label in its small style */
        function setHover(k, px, py, text, strength) {
            const st = strength === undefined ? 1 : strength;
            for (let i = 0; i < TGT.length; i++) hoverTo[i] = i === k ? st : 0;
            if (!tag) return;
            tag.classList.toggle("soft", st < 1);
            if (k >= 0 && text !== null) {
                const [sx, sy] = stageOf(px, py);
                tag.textContent = text || labelOf(k);
                // the label is centred on the pointer; kept inside the stage (the view clips it),
                // which matters for the curtain along the left edge
                const half = tag.offsetWidth / 2 + 6;
                tag.style.left = Math.max(half, Math.min(stage.clientWidth - half, sx)) + "px";
                tag.style.top = sy + "px";
                tag.classList.add("show");
            } else tag.classList.remove("show");
        }
        function setFocus(k) { for (let i = 0; i < TGT.length; i++) focusTo[i] = i === k ? 1 : 0; }
        function setPick(on) { pickTo = on ? 1 : 0; }

        // ------------------------------------------------------------------ phones: pan the zoomed room
        let pan = null;
        const scrollable = () => view.scrollWidth - view.clientWidth > 2;
        function panTo(k, dur) {
            if (!scrollable()) return;
            const b = TGT[k].bbox;
            const cx = ((b[0] + b[2]) / 2 / W) * stage.clientWidth;
            const to = Math.max(0, Math.min(view.scrollWidth - view.clientWidth, cx - view.clientWidth / 2));
            if (Math.abs(to - view.scrollLeft) < 4) return;
            pan = { from: view.scrollLeft, to, t0: now(), dur: REDUCED ? 0.01 : (dur || 0.55) };
        }
        function stepPan(t) {
            if (!pan) return;
            const u = clamp01((t - pan.t0) / pan.dur);
            view.scrollLeft = pan.from + (pan.to - pan.from) * easeInOut(u);
            if (u >= 1) pan = null;
        }
        ["touchstart", "wheel", "pointerdown"].forEach((ev) =>
            view.addEventListener(ev, () => { pan = null; }, { passive: true }));
        function centerView() { if (scrollable()) view.scrollLeft = (view.scrollWidth - view.clientWidth) / 2; }
        const edges = () => {
            const s = view.scrollLeft, m = view.scrollWidth - view.clientWidth;
            root.classList.toggle("room-more-left", s > 4);
            root.classList.toggle("room-more-right", s < m - 4);
        };
        view.addEventListener("scroll", edges, { passive: true });
        centerView();
        edges();
        let lastW = view.clientWidth;
        window.addEventListener("resize", () => {
            if (Math.abs(view.clientWidth - lastW) > 1) { lastW = view.clientWidth; centerView(); edges(); }
        });

        // ------------------------------------------------------------------ loop
        let visible = true, running = false;
        function frame() {
            const t = now();
            draw(t);
            if (opts.onFrame) opts.onFrame(t);
            if (window.__demoClock === undefined && visible && !document.hidden) requestAnimationFrame(frame);
            else running = false;
        }
        function kick() {
            if (running || window.__demoClock !== undefined) return;
            running = true;
            requestAnimationFrame(frame);
        }
        if ("IntersectionObserver" in window) {
            new IntersectionObserver((es) => {
                visible = es.some((e) => e.isIntersecting);
                if (visible) kick();
                listeners.forEach((f) => f("visible", { visible }));
            }, { rootMargin: "100px 0px" }).observe(root);
        }
        document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });
        wake = kick;
        root.classList.add("room-ready");
        kick();

        return {
            meta, W, H, MATS, TGT, imgs, root, stage, view, canvas, set: SET,
            now, apply, reset, targetAt, clientOf, stageOf, roomPx, setHover, setFocus, setPick, panTo,
            regionMat, layers, frame, kick, need, labelOf,
            on(f) { listeners.push(f); },
            busy() { const t = now(); return layers.some((L) => L.wait || !layerDone(L, t - L.t0)) || fading !== null || !!pan; },
            /** the current composite (no highlights) as a PNG data URL (tests / measurements) */
            snapshot() {
                compose(now(), 1);
                gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
                const px = new Uint8Array(SW * SH * 4);
                gl.readPixels(0, 0, SW, SH, gl.RGBA, gl.UNSIGNED_BYTE, px);
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                const c = document.createElement("canvas");
                c.width = SW; c.height = SH;
                const x = c.getContext("2d"), id = x.createImageData(SW, SH);
                for (let y = 0; y < SH; y++) id.data.set(px.subarray((SH - 1 - y) * SW * 4, (SH - y) * SW * 4), y * SW * 4);
                for (let i = 3; i < id.data.length; i += 4) id.data[i] = 255;
                x.putImageData(id, 0, 0);
                return c.toDataURL("image/png");
            },
            ready,
            /** client coords of a good drop point on target k */
            targetPoint(k) { const d = TGT[k].drop; return clientOf(d[0], d[1]); },
            /** material shown by target k once settled (-1 grey, null = mixed) */
            targetMat(k) { return regionMat[TGT[k].region]; },
            isVisible() { return visible; },
        };
    }

    window.RoomCore = { create, whenNear, now, REDUCED, theta, latentTile, makeTile, tileView, drawDecode, snowflake, loadImg,
        easeOut3, easeInOut, smooth, clamp01 };
})();
