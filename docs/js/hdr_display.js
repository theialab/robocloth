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

    // The decode worker sits next to this script (same cache-busting query).
    const WORKER_URL = (() => {
        try {
            const src = document.currentScript && document.currentScript.src;
            if (src) return new URL("hdr_worker.js" + new URL(src).search, src).href;
        } catch (e) { /* fall through */ }
        return "js/hdr_worker.js";
    })();

    /**
     * 256-entry tone curve: out = 255 * (in/255 * exposure)^(1/gamma),
     * clipped. Identical to the per-pixel Math.pow it replaces (the values
     * land in a Uint8ClampedArray either way) but ~10x cheaper per frame.
     */
    function toneLUT(exposure, gamma) {
        const lut = new Uint8ClampedArray(256);
        const inv = 1.0 / gamma;
        for (let v = 0; v < 256; v++) {
            lut[v] = Math.min(255, 255 * Math.pow(Math.max(0, v / 255) * exposure, inv));
        }
        return lut;
    }

    /** Tone curve for a proxy that stores sqrt(linear) in 8 bits (build_proxies.py). */
    function sqrtLUT(exposure, gamma) {
        const lut = new Uint8ClampedArray(256);
        const inv = 1 / gamma;
        for (let v = 0; v < 256; v++) {
            lut[v] = Math.min(255, 255 * Math.pow(Math.pow(v / 255, 2) * exposure, inv));
        }
        return lut;
    }

    function applyLUT(data, lut) {
        for (let i = 0; i < data.length; i += 4) {
            data[i] = lut[data[i]];
            data[i + 1] = lut[data[i + 1]];
            data[i + 2] = lut[data[i + 2]];
            // alpha untouched
        }
    }

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
     *   - stats:    true to also return per-channel stats (an extra pass
     *               over every pixel; off by default)
     */
    async function drawPng(pngBytes, targetCanvas, opts = {}) {
        const exposure = opts.exposure ?? VIZ.HDR_EXPOSURE;
        const gamma = opts.gamma ?? VIZ.HDR_GAMMA;
        const maxWidth = opts.maxWidth ?? 1024;

        const blob = pngBytes instanceof Blob
            ? pngBytes : new Blob([pngBytes], { type: "image/png" });
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
        const toneMap = exposure !== 1.0 || gamma !== 1.0;
        let stats = null;
        if (opts.stats || toneMap) {
            const rawImg = ctx.getImageData(0, 0, dw, dh);
            if (opts.stats) stats = _computeStats(rawImg.data, dw, dh, opts.maskPoly || null);

            // ----- Tone mapping (mutates the same ImageData) -----
            if (toneMap) {
                applyLUT(rawImg.data, toneLUT(exposure, gamma));
                ctx.putImageData(rawImg, 0, 0);
            }
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

    // -----------------  decode(): off the main thread  -----------------
    //
    // Decoding a 3072x2048 16-bit PNG takes ~150-200 ms and the tone map
    // another ~60 ms (~250 ms on a phone-class CPU) on the main thread, where
    // it stalls slider input. decode() does all of it in a Web Worker
    // (OffscreenCanvas) and hands back a display-sized ImageBitmap; without
    // worker support it falls back to the main thread. Jobs run one at a
    // time; queued (not yet started) jobs are dropped when their signal
    // aborts, and foreground jobs jump ahead of background (prefetch) ones.

    let _worker = null, _workerBroken = false, _seq = 0;
    const _pending = new Map();           // id -> { resolve, reject }
    const _queue = [];
    let _busy = false;

    const _unsupported = (msg) => Object.assign(new Error(msg), { unsupported: true });

    function _getWorker() {
        if (_workerBroken) return null;
        if (_worker) return _worker;
        try {
            if (typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined") {
                throw _unsupported("no OffscreenCanvas worker support");
            }
            const w = new Worker(WORKER_URL);
            w.onmessage = (ev) => {
                const d = ev.data, p = _pending.get(d.id);
                if (!p) { if (d.bitmap) d.bitmap.close(); return; }
                _pending.delete(d.id);
                if (d.bitmap) p.resolve(d.bitmap);
                else p.reject(d.unsupported ? _unsupported(d.error) : new Error(d.error));
            };
            w.onerror = (ev) => {          // script failed to load / crashed
                if (ev && ev.preventDefault) ev.preventDefault();
                _workerBroken = true;
                const ps = [..._pending.values()];
                _pending.clear();
                ps.forEach((p) => p.reject(_unsupported("decode worker failed")));
                try { w.terminate(); } catch (e) { /* ignore */ }
                _worker = null;
            };
            _worker = w;
        } catch (e) {
            _workerBroken = true;
            return null;
        }
        return _worker;
    }

    function _decodeInWorker(blob, bbox, maxWidth, lut) {
        const w = _getWorker();
        if (!w) return Promise.reject(_unsupported("no worker"));
        const id = ++_seq;
        return new Promise((resolve, reject) => {
            _pending.set(id, { resolve, reject });
            w.postMessage({ id, blob, bbox, maxWidth, lut });
        });
    }

    async function _decodeOnMain(blob, bbox, maxWidth, lut) {
        const src = await createImageBitmap(blob);
        const [sx, sy, sx2, sy2] = bbox || [0, 0, src.width, src.height];
        const sw = Math.max(1, sx2 - sx), sh = Math.max(1, sy2 - sy);
        const scale = Math.min(1, maxWidth / sw);
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(sw * scale));
        c.height = Math.max(1, Math.round(sh * scale));
        const ctx = c.getContext("2d", { willReadFrequently: !!lut });
        ctx.drawImage(src, sx, sy, sw, sh, 0, 0, c.width, c.height);
        src.close && src.close();
        if (lut) {
            const img = ctx.getImageData(0, 0, c.width, c.height);
            applyLUT(img.data, lut);
            ctx.putImageData(img, 0, 0);
        }
        return (typeof createImageBitmap === "function") ? createImageBitmap(c) : c;
    }

    async function _run(job) {
        const { blob, opts } = job;
        const exposure = opts.exposure ?? VIZ.HDR_EXPOSURE;
        const gamma = opts.gamma ?? VIZ.HDR_GAMMA;
        const maxWidth = opts.maxWidth ?? 1024;
        const bbox = opts.bbox || null;
        // a scrubbing proxy stores sqrt(linear): its LUT undoes that first
        const lut = opts.encoding === "sqrt" ? sqrtLUT(exposure, gamma)
            : (exposure !== 1.0 || gamma !== 1.0) ? toneLUT(exposure, gamma) : null;
        try {
            return await _decodeInWorker(blob, bbox, maxWidth, lut);
        } catch (err) {
            if (!err.unsupported) throw err;
            _workerBroken = true;
        }
        return _decodeOnMain(blob, bbox, maxWidth, lut);
    }

    function _pump() {
        if (_busy || !_queue.length) return;
        const job = _queue.shift();
        _busy = true;
        _run(job).then(job.resolve, job.reject).finally(() => { _busy = false; _pump(); });
    }

    /**
     * Decode + crop + scale + tone-map a PNG (Blob or ArrayBuffer) into an
     * ImageBitmap (or, on very old browsers, a canvas) ready for drawImage.
     *
     * opts: exposure, gamma, bbox, maxWidth (as drawPng), signal (drops the
     *       job while it is still queued), background (yield to others).
     */
    function decode(png, opts = {}) {
        const blob = png instanceof Blob ? png : new Blob([png], { type: "image/png" });
        return new Promise((resolve, reject) => {
            const job = { blob, opts, resolve, reject };
            const sig = opts.signal;
            if (sig) {
                if (sig.aborted) { reject(new DOMException("Aborted", "AbortError")); return; }
                sig.addEventListener("abort", () => {
                    const k = _queue.indexOf(job);
                    if (k >= 0) { _queue.splice(k, 1); reject(new DOMException("Aborted", "AbortError")); }
                }, { once: true });
            }
            const k = opts.background ? -1 : _queue.findIndex((j) => j.opts.background);
            if (k < 0) _queue.push(job); else _queue.splice(k, 0, job);
            _pump();
        });
    }

    /**
     * Start the decode worker now (its script loads and its thread starts
     * while the first capture is still downloading) instead of on the first
     * decode().
     */
    function warm() { _getWorker(); }

    window.HDRDisplay = { drawPng, pngToBlobURL, decode, toneLUT, sqrtLUT, warm };
})();
