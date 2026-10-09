/**
 * Top-down texture of the material swatch for the 3-D viewer.
 *
 * Takes one capture (the most top-down camera of the material), projects
 * the swatch's world0 top rectangle (bbox.json) into it with the calibrated
 * camera (rotated_camera.json + the intrinsics in hdr_crop_bboxes.json, via
 * Transforms.makeWorld0Projector) and resamples it into a square tile:
 * +x of the swatch to the right, +y up. The capture's own illumination
 * gradient is divided out (a quadratic fitted to log luminance) and the
 * tile is scaled to the material's mean colour, so the viewer's spot light
 * does the shading.
 *
 * The browser decodes the 16-bit PNG to 8 bits; very dark materials are
 * therefore a little noisy, which is fine at swatch size.
 */

window.SwatchTexture = (() => {

    /** Index of the entry whose camera is closest to straight above the swatch. */
    function pickFrontal(entries, bbox) {
        const c = bbox.bbox_center || [
            (bbox.bbox_min[0] + bbox.bbox_max[0]) / 2,
            (bbox.bbox_min[1] + bbox.bbox_max[1]) / 2,
        ];
        const zTop = bbox.bbox_max[2];
        let best = -1, bestS = -2;
        entries.forEach((e, i) => {
            const d = [e.cam_pos_w0[0] - c[0], e.cam_pos_w0[1] - c[1], e.cam_pos_w0[2] - zTop];
            const s = d[2] / Math.hypot(d[0], d[1], d[2]);
            if (s > bestS) { bestS = s; best = i; }
        });
        return best;
    }

    // The resampling below is ~1 M projections + ~1 M pow(); give the main
    // thread back every ~10 ms so slider input and the 3-D views keep
    // running while the tile is built (one long task was ~200 ms on a
    // desktop CPU, several times that on a phone). A time budget rather
    // than a row count, so a slow CPU yields as often as a fast one.
    const yieldToMain = () => new Promise((r) => setTimeout(r, 0));
    const SLICE_MS = 10;
    let sliceStart = 0;
    const maybeYield = async () => {
        if (performance.now() - sliceStart < SLICE_MS) return;
        await yieldToMain();
        sliceStart = performance.now();
    };

    const srgbEncode = (x) => {
        x = Math.max(0, Math.min(1, x));
        return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
    };

    /** Solve the 6x6 normal equations (Gaussian elimination, partial pivoting). */
    function solve(A, b) {
        const n = b.length;
        const M = A.map((row, i) => [...row, b[i]]);
        for (let c = 0; c < n; c++) {
            let p = c;
            for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
            [M[c], M[p]] = [M[p], M[c]];
            if (Math.abs(M[c][c]) < 1e-12) return null;
            for (let r = c + 1; r < n; r++) {
                const f = M[r][c] / M[c][c];
                for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
            }
        }
        const x = new Array(n).fill(0);
        for (let r = n - 1; r >= 0; r--) {
            let s = M[r][n];
            for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
            x[r] = s / M[r][r];
        }
        return x;
    }

    /**
     * @param {ArrayBuffer} pngBytes  the capture (16-bit PNG from hdr.tar)
     * @param {function} project      world0 [x,y,z] -> image [u,v] (px) or null
     * @param {object} bbox           bbox.json (world0)
     * @param {object} opts           { N = 512, inset = 0.045, meanLin = [r,g,b] (0..1, linear) }
     * @returns {Promise<HTMLCanvasElement|null>} null if the projection does
     *          not land on the (masked) cloth — the caller keeps its placeholder.
     */
    async function rectify(pngBytes, project, bbox, opts = {}) {
        const N = opts.N || 512;
        const inset = opts.inset ?? 0.045;
        const blob = new Blob([pngBytes], { type: "image/png" });
        let bmp;
        try {
            bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
        } catch (e) {
            bmp = await createImageBitmap(blob);
        }

        const [x0, y0] = bbox.bbox_min, [x1, y1] = bbox.bbox_max;
        const z = bbox.bbox_max[2];
        const world = (u, v) => [
            x0 + (inset + (1 - 2 * inset) * u) * (x1 - x0),
            y0 + (inset + (1 - 2 * inset) * v) * (y1 - y0),
            z,
        ];

        // image-space bounds of the (inset) swatch -> crop
        let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
        for (let i = 0; i <= 8; i++) for (let j = 0; j <= 8; j++) {
            const p = project(world(i / 8, j / 8));
            if (!p) { bmp.close && bmp.close(); return null; }
            bx0 = Math.min(bx0, p[0]); bx1 = Math.max(bx1, p[0]);
            by0 = Math.min(by0, p[1]); by1 = Math.max(by1, p[1]);
        }
        const sx = Math.max(0, Math.floor(bx0) - 4), sy = Math.max(0, Math.floor(by0) - 4);
        const sw = Math.min(bmp.width, Math.ceil(bx1) + 4) - sx;
        const sh = Math.min(bmp.height, Math.ceil(by1) + 4) - sy;
        if (sw < 16 || sh < 16) { bmp.close && bmp.close(); return null; }

        const src = document.createElement("canvas");
        src.width = sw; src.height = sh;
        const sctx = src.getContext("2d", { willReadFrequently: true });
        sctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh);
        bmp.close && bmp.close();
        const D = sctx.getImageData(0, 0, sw, sh).data;
        src.width = src.height = 1;    // free

        const sample = (px, py, out) => {       // bilinear, linear 0..1
            const fx = Math.max(0, Math.min(sw - 1.001, px - sx - 0.5));
            const fy = Math.max(0, Math.min(sh - 1.001, py - sy - 0.5));
            const ix = fx | 0, iy = fy | 0, ax = fx - ix, ay = fy - iy;
            const i00 = 4 * (iy * sw + ix), i10 = i00 + 4, i01 = i00 + 4 * sw, i11 = i01 + 4;
            for (let c = 0; c < 3; c++) {
                out[c] += ((D[i00 + c] * (1 - ax) + D[i10 + c] * ax) * (1 - ay)
                         + (D[i01 + c] * (1 - ax) + D[i11 + c] * ax) * ay) / 255;
            }
        };

        // resample, 2x2 per texel; row 0 = +y (top of the tile)
        const T = new Float32Array(N * N * 3);
        const acc = [0, 0, 0];
        let dark = 0;
        sliceStart = performance.now();
        for (let j = 0; j < N; j++) {
            await maybeYield();
            for (let i = 0; i < N; i++) {
                acc[0] = acc[1] = acc[2] = 0;
                for (const [du, dv] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
                    const p = project(world((i + du) / N, 1 - (j + dv) / N));
                    if (p) sample(p[0], p[1], acc);
                }
                const o = 3 * (j * N + i);
                T[o] = acc[0] / 4; T[o + 1] = acc[1] / 4; T[o + 2] = acc[2] / 4;
                if (T[o] + T[o + 1] + T[o + 2] < 1.5 / 255) dark++;
            }
        }
        // the HDR captures are masked to zero outside the cloth: a tile that
        // is mostly black means the projection missed it
        if (dark > 0.25 * N * N) return null;

        // illumination gradient: quadratic fit to log luminance (on a grid)
        const G = 32, A = Array.from({ length: 6 }, () => new Array(6).fill(0)), b = new Array(6).fill(0);
        for (let gj = 0; gj < G; gj++) for (let gi = 0; gi < G; gi++) {
            const i = Math.floor((gi + 0.5) * N / G), j = Math.floor((gj + 0.5) * N / G);
            const o = 3 * (j * N + i);
            const L = 0.2126 * T[o] + 0.7152 * T[o + 1] + 0.0722 * T[o + 2];
            if (L < 2 / 255) continue;
            const u = (i / N) * 2 - 1, v = (j / N) * 2 - 1;
            const f = [1, u, v, u * u, v * v, u * v], y = Math.log(L);
            for (let r = 0; r < 6; r++) {
                b[r] += f[r] * y;
                for (let c = 0; c < 6; c++) A[r][c] += f[r] * f[c];
            }
        }
        const coef = solve(A, b) || [0, 0, 0, 0, 0, 0];
        const shade = (i, j) => {
            const u = (i / N) * 2 - 1, v = (j / N) * 2 - 1;
            return Math.exp(coef[1] * u + coef[2] * v + coef[3] * u * u + coef[4] * v * v + coef[5] * u * v);
        };

        // flatten, then scale to the target colour
        const mean = [0, 0, 0];
        for (let j = 0; j < N; j++) {
            await maybeYield();
            for (let i = 0; i < N; i++) {
                const s = 1 / shade(i, j), o = 3 * (j * N + i);
                for (let c = 0; c < 3; c++) { T[o + c] *= s; mean[c] += T[o + c]; }
            }
        }
        for (let c = 0; c < 3; c++) mean[c] = Math.max(1e-4, mean[c] / (N * N));
        let target = opts.meanLin
            ? opts.meanLin.map((m) => m * 2.0)
            : (() => { const l = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2];
                       return mean.map((m) => m / Math.max(l, 1e-4) * 0.3); })();
        const tMax = Math.max(...target);
        if (tMax > 0.85) target = target.map((t) => t * 0.85 / tMax);
        const tLum = 0.2126 * target[0] + 0.7152 * target[1] + 0.0722 * target[2];
        if (tLum < 0.04) target = target.map((t) => t * 0.04 / Math.max(tLum, 1e-4));
        const gain = target.map((t, c) => t / mean[c]);

        const out = document.createElement("canvas");
        out.width = out.height = N;
        const octx = out.getContext("2d");
        const img = octx.createImageData(N, N);
        for (let p = 0, q = 0; p < N * N; p++, q += 3) {
            if (p % N === 0) await maybeYield();
            for (let c = 0; c < 3; c++) img.data[4 * p + c] = Math.round(255 * srgbEncode(T[q + c] * gain[c]));
            img.data[4 * p + 3] = 255;
        }
        octx.putImageData(img, 0, 0);
        return out;
    }

    return { pickFrontal, rectify };
})();
