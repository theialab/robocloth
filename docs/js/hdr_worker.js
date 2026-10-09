/**
 * Off-main-thread decode for HDRDisplay.decode(): 16-bit PNG Blob ->
 * cropped / scaled / tone-mapped ImageBitmap (transferred back).
 *
 * Message in : { id, blob, bbox: [x0,y0,x1,y1] | null, maxWidth, lut: Uint8ClampedArray(256) | null }
 * Message out: { id, bitmap, width, height }  or  { id, error, unsupported }
 */
self.onmessage = async (ev) => {
    const { id, blob, bbox, maxWidth, lut } = ev.data;
    try {
        if (typeof OffscreenCanvas === "undefined" || typeof createImageBitmap === "undefined") {
            self.postMessage({ id, error: "OffscreenCanvas unavailable", unsupported: true });
            return;
        }
        const src = await createImageBitmap(blob);
        const [sx, sy, sx2, sy2] = bbox || [0, 0, src.width, src.height];
        const sw = Math.max(1, sx2 - sx), sh = Math.max(1, sy2 - sy);
        const scale = Math.min(1, maxWidth / sw);
        const dw = Math.max(1, Math.round(sw * scale));
        const dh = Math.max(1, Math.round(sh * scale));
        const oc = new OffscreenCanvas(dw, dh);
        const ctx = oc.getContext("2d", { willReadFrequently: !!lut });
        if (!ctx) {
            src.close();
            self.postMessage({ id, error: "no 2d context in worker", unsupported: true });
            return;
        }
        ctx.drawImage(src, sx, sy, sw, sh, 0, 0, dw, dh);
        src.close();
        if (lut) {
            const img = ctx.getImageData(0, 0, dw, dh);
            const d = img.data;
            for (let i = 0; i < d.length; i += 4) {
                d[i] = lut[d[i]];
                d[i + 1] = lut[d[i + 1]];
                d[i + 2] = lut[d[i + 2]];
            }
            ctx.putImageData(img, 0, 0);
        }
        const bitmap = oc.transferToImageBitmap();
        self.postMessage({ id, bitmap, width: dw, height: dh }, [bitmap]);
    } catch (err) {
        self.postMessage({ id, error: String((err && err.message) || err) });
    }
};
