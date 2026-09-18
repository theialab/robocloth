/**
 * HDR display helper.
 *
 * The hdr.tar files contain 16-bit RGB PNGs (full sensor range).  Plain
 * `<img>` rendering of those works in modern browsers but the result is
 * usually very dark because no exposure / gamma curve has been applied.
 *
 * This module:
 *   1.  Decodes a PNG ArrayBuffer into an off-screen canvas at native res.
 *   2.  Optionally applies an exposure + gamma tone map.
 *   3.  Returns either a blob URL (suitable for `<img>` src) or paints
 *       directly onto a target canvas.
 *
 * The browser already decodes 16-bit PNGs down to 8-bit per channel when
 * we draw them to a canvas, so the tone-map operates on the 8-bit pixels
 * we get from getImageData. For the dataset preview that is fine — the
 * sensor pipeline already applied debayer + white balance + clipping
 * before upload, so the 16-bit precision matters mostly for training,
 * not visualisation.
 */

(() => {
    const VIZ = window.CONFIG.VIZ;

    /**
     * Decode a PNG byte slice and draw it to a target canvas with optional
     * tone mapping.
     *
     * @param {ArrayBuffer} pngBytes - the raw PNG file bytes.
     * @param {HTMLCanvasElement} targetCanvas - canvas to paint into.
     * @param {object} opts
     *   - exposure: linear gain (default CONFIG.VIZ.HDR_EXPOSURE)
     *   - gamma:    output gamma (default CONFIG.VIZ.HDR_GAMMA)
     *   - bbox:     optional [x0, y0, x1, y1] crop in source pixels
     *   - maxWidth: scale to fit (preserves aspect ratio)
     */
    async function drawPng(pngBytes, targetCanvas, opts = {}) {
        const exposure = opts.exposure ?? VIZ.HDR_EXPOSURE;
        const gamma = opts.gamma ?? VIZ.HDR_GAMMA;
        const maxWidth = opts.maxWidth ?? 1024;

        const blob = new Blob([pngBytes], { type: "image/png" });
        const bmp = await createImageBitmap(blob);

        // Determine source crop
        const [sx, sy, sx2, sy2] = opts.bbox ?? [0, 0, bmp.width, bmp.height];
        const sw = Math.max(1, sx2 - sx);
        const sh = Math.max(1, sy2 - sy);

        // Compute display size
        const scale = Math.min(1, maxWidth / sw);
        const dw = Math.max(1, Math.round(sw * scale));
        const dh = Math.max(1, Math.round(sh * scale));

        targetCanvas.width = dw;
        targetCanvas.height = dh;

        const ctx = targetCanvas.getContext("2d");
        ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, dw, dh);

        // ----- Stats pass (BEFORE tone-mapping) -----
        // Per-channel non-zero stats over the 8-bit canvas pixels. The
        // browser decodes 16-bit PNG to 8-bit before we see them, so to
        // express the result back in the original 0..65535 capture range
        // we scale by 257 (= 65535/255).
        //
        // `opts.maskPoly` is an optional polygon in canvas pixel coords
        // (array of [x,y]). When present, only pixels inside the polygon
        // are counted — used to restrict stats to the projected material
        // rectangle and ignore the masked-zero boundary.
        const rawImg = ctx.getImageData(0, 0, dw, dh);
        const stats = _computeStats(rawImg.data, dw, dh, opts.maskPoly || null);

        // ----- Tone mapping (mutates the same ImageData) -----
        if (exposure !== 1.0 || gamma !== 1.0) {
            const inv = 1.0 / gamma;
            const data = rawImg.data;
            for (let i = 0; i < data.length; i += 4) {
                data[i]   = Math.min(255, 255 * Math.pow(Math.max(0, data[i]/255) * exposure, inv));
                data[i+1] = Math.min(255, 255 * Math.pow(Math.max(0, data[i+1]/255) * exposure, inv));
                data[i+2] = Math.min(255, 255 * Math.pow(Math.max(0, data[i+2]/255) * exposure, inv));
                // alpha untouched
            }
            ctx.putImageData(rawImg, 0, 0);
        }

        bmp.close && bmp.close();
        return { width: dw, height: dh, stats };
    }

    /**
     * Walk an 8-bit RGBA buffer and compute PER-CHANNEL non-zero stats.
     * The pipeline saves materials with the off-cloth region forced to
     * (0,0,0) by the projected-rectangle mask. Even so, the mask isn't
     * pixel-perfect at the boundary, so the *caller* is encouraged to
     * pass `maskPoly` (the projected, slightly-shrunken material rect)
     * to count only well-inside pixels.
     *
     * Returns null if no pixels qualify.
     */
    function _computeStats(data, dw, dh, maskPoly) {
        // Bounding box of the polygon → only iterate inside it (huge
        // speed-up when the material covers ~10% of the canvas).
        let x0 = 0, y0 = 0, x1 = dw, y1 = dh;
        if (maskPoly && maskPoly.length >= 3) {
            x0 = Math.max(0, Math.floor(Math.min(...maskPoly.map(p => p[0]))));
            y0 = Math.max(0, Math.floor(Math.min(...maskPoly.map(p => p[1]))));
            x1 = Math.min(dw, Math.ceil(Math.max(...maskPoly.map(p => p[0]))));
            y1 = Math.min(dh, Math.ceil(Math.max(...maskPoly.map(p => p[1]))));
        }
        if (x1 <= x0 || y1 <= y0) return null;

        let minR = 256, minG = 256, minB = 256;
        let maxR = -1,  maxG = -1,  maxB = -1;
        let sumR = 0,   sumG = 0,   sumB = 0;
        let count = 0;

        for (let y = y0; y < y1; y++) {
            const rowOff = y * dw * 4;
            for (let x = x0; x < x1; x++) {
                if (maskPoly && !_pointInPoly(x + 0.5, y + 0.5, maskPoly)) continue;
                const i = rowOff + x * 4;
                const r = data[i], g = data[i+1], b = data[i+2];
                if (r === 0 && g === 0 && b === 0) continue;
                sumR += r; sumG += g; sumB += b;
                if (r < minR) minR = r;  if (r > maxR) maxR = r;
                if (g < minG) minG = g;  if (g > maxG) maxG = g;
                if (b < minB) minB = b;  if (b > maxB) maxB = b;
                count++;
            }
        }
        if (count === 0) return null;

        const S = 257;
        return {
            count,
            mean_rgb: [sumR / count, sumG / count, sumB / count],
            // Per-channel min/max in the original 0..65535 capture range.
            min_rgb_16bit: [minR * S, minG * S, minB * S],
            max_rgb_16bit: [maxR * S, maxG * S, maxB * S],
        };
    }

    /** Ray-casting point-in-polygon. `poly` = array of [x, y]. */
    function _pointInPoly(px, py, poly) {
        let inside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const xi = poly[i][0], yi = poly[i][1];
            const xj = poly[j][0], yj = poly[j][1];
            const cross = ((yi > py) !== (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
            if (cross) inside = !inside;
        }
        return inside;
    }

    /**
     * Convenience: turn a PNG ArrayBuffer into a blob URL with tone-mapping
     * applied via an off-screen canvas.  The caller should URL.revokeObjectURL
     * when done.
     */
    async function pngToBlobURL(pngBytes, opts = {}) {
        const off = document.createElement("canvas");
        await drawPng(pngBytes, off, opts);
        return new Promise((resolve) => {
            off.toBlob((b) => resolve(URL.createObjectURL(b)), "image/jpeg", 0.92);
        });
    }

    window.HDRDisplay = { drawPng, pngToBlobURL };
})();
