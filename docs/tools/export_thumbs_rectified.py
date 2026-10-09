#!/usr/bin/env python
"""Landing-page thumbnails as UPRIGHT SQUARES: images/thumbs/<id>.jpg.

For each preview material the most top-down capture (SwatchTexture.pickFrontal) is
resampled onto the swatch's own top rectangle (bbox.json, world0) through the calibrated
camera (rotated_camera.json + the intrinsics of hdr_crop_bboxes.json, via
Transforms.makeWorld0Projector) -- the same geometry the viewer uses for its 3-D swatch --
and tone-mapped exactly like the old photo thumbnails (exposure 1.2, gamma 2.2). Only the
geometry changes: the capture's own lighting is kept, so the square still looks like a photo.

Drives the page's own JS in headless Chrome (the page must be served, Range-capable):
    python3 docs/tools/export_thumbs_rectified.py --url http://127.0.0.1:8765/ [--ids 0 32] [--out DIR]
"""
import argparse, base64, os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "thumbs")
JS = """async ({ids, N, inset, exposure, gamma, quality}) => {
  const m = await (await fetch(CONFIG.manifestURL())).json();
  const out = {};
  const bil = (S, W, H, x, y) => {       // bilinear RGB sample of an RGBA buffer
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
    const r = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const a = S[(y0 * W + x0) * 4 + c], b = S[(y0 * W + x1) * 4 + c];
      const d = S[(y1 * W + x0) * 4 + c], e = S[(y1 * W + x1) * 4 + c];
      r[c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
    }
    return r;
  };
  for (const e of m.preview) {
    if (ids.length && !ids.includes(e.id)) continue;
    const id = e.id;
    const [scanLog, rotCam, crop, bbox, index] = await Promise.all([
      CONFIG.scanLogURL(id), CONFIG.rotatedCameraURL(id), CONFIG.cropBBoxURL(id), CONFIG.bboxURL(id), CONFIG.tarIndexURL(id)
    ].map(u => fetch(u).then(r => r.json())));
    const processed = Transforms.processScanLog(scanLog);
    const rotIdx = Transforms.indexRotatedCamera(rotCam);
    const cands = processed.entries.filter(c => rotIdx.has(Number(c.camera_id)) && (c.filename in index));
    if (!cands.length) { out[id] = {error: 'no calibrated capture'}; continue; }
    const [x0, y0] = bbox.bbox_min, [x1, y1] = bbox.bbox_max, z = bbox.bbox_max[2];
    const wpt = (u, v, ins) => [x0 + (ins + (1 - 2 * ins) * u) * (x1 - x0), y0 + (ins + (1 - 2 * ins) * v) * (y1 - y0), z];
    // frontalness of every candidate; keep the ones whose whole (inset) rectangle projects
    // inside the frame with a margin, most frontal first
    const c0 = bbox.bbox_center || [(x0 + x1) / 2, (y0 + y1) / 2];
    const scored = cands.map(c => {
      const d = [c.cam_pos_w0[0] - c0[0], c.cam_pos_w0[1] - c0[1], c.cam_pos_w0[2] - z];
      return {c, s: d[2] / Math.hypot(d[0], d[1], d[2])};
    }).sort((a, b) => b.s - a.s).slice(0, 40);
    const W0 = crop.intrinsics.width || crop.intrinsics.w || 3072, H0 = crop.intrinsics.height || crop.intrinsics.h || 2048;
    const geomOK = (project, ins, W, H) => {
      for (let j = 0; j <= 8; j++) for (let i = 0; i <= 8; i++) {
        const p = project(wpt(i / 8, j / 8, ins));
        if (!p || p[0] < 12 || p[1] < 12 || p[0] > W - 13 || p[1] > H - 13) return false;
      }
      return true;
    };
    const lut = HDRDisplay.toneLUT(exposure, gamma);
    let best = null;                                   // {score, data, filename, inside, ins}
    let tried = 0;
    for (const {c: ent, s} of scored) {
      const project = Transforms.makeWorld0Projector(rotIdx.get(Number(ent.camera_id)), crop.intrinsics);
      if (!project) continue;
      if (!geomOK(project, inset, W0, H0)) continue;
      if (tried >= 3) break;
      tried++;
      const buf = await TarReader.fetchFile(CONFIG.hdrTarURL(id), index, ent.filename);
      const bmp = await createImageBitmap(buf instanceof Blob ? buf : new Blob([buf], {type: 'image/png'}));
      const sc = document.createElement('canvas'); sc.width = bmp.width; sc.height = bmp.height;
      const sctx = sc.getContext('2d', {willReadFrequently: true}); sctx.drawImage(bmp, 0, 0);
      const S = sctx.getImageData(0, 0, sc.width, sc.height).data, W = sc.width, H = sc.height;
      // the cloth is masked (black) outside its outline: grow the inset until the square is cloth
      for (const ins0 of [inset, 0.06, 0.09, 0.12]) {
        const ins = ins0 + 0.015;                      // safety margin inside the chosen window
        const oc = document.createElement('canvas'); oc.width = oc.height = N;
        const octx = oc.getContext('2d'); const img = octx.createImageData(N, N); const D = img.data;
        let inside = 0;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const u = (i + 0.5) / N, v = 1 - (j + 0.5) / N;          // +x right, +y up
          const p = project(wpt(u, v, ins));
          const k = (j * N + i) * 4;
          if (!p || p[0] < 0 || p[1] < 0 || p[0] > W - 1 || p[1] > H - 1) { D[k + 3] = 255; continue; }
          const rgb = bil(S, W, H, p[0], p[1]);
          if (rgb[0] + rgb[1] + rgb[2] > 2) inside++;
          D[k] = lut[Math.round(rgb[0])]; D[k + 1] = lut[Math.round(rgb[1])]; D[k + 2] = lut[Math.round(rgb[2])]; D[k + 3] = 255;
        }
        const cov = inside / (N * N);
        if (!best || cov > best.inside + 0.002) {
          octx.putImageData(img, 0, 0);
          best = {data: oc.toDataURL('image/jpeg', quality), filename: ent.filename, inside: cov, ins, frontal: s};
        }
        if (cov >= 0.995) break;
      }
      if (best && best.inside >= 0.995) break;
    }
    // Fallback (bbox.json not matching the cloth outline, or the swatch cut by the frame in
    // every frontal view): take the most frontal captures and search the largest sub-square of
    // the swatch rectangle that is all cloth.
    if (!best || best.inside < 0.99) {
      for (const {c: ent, s} of scored.slice(0, 3)) {
        const project = Transforms.makeWorld0Projector(rotIdx.get(Number(ent.camera_id)), crop.intrinsics);
        if (!project) continue;
        const buf = await TarReader.fetchFile(CONFIG.hdrTarURL(id), index, ent.filename);
        const bmp = await createImageBitmap(buf instanceof Blob ? buf : new Blob([buf], {type: 'image/png'}));
        const sc = document.createElement('canvas'); sc.width = bmp.width; sc.height = bmp.height;
        const sctx = sc.getContext('2d', {willReadFrequently: true}); sctx.drawImage(bmp, 0, 0);
        const S = sctx.getImageData(0, 0, sc.width, sc.height).data, W = sc.width, H = sc.height;
        const cover = (u0, v0, size, n) => {
          let ok = 0;
          for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
            const p = project(wpt(u0 + size * (i + 0.5) / n, v0 + size * (1 - (j + 0.5) / n), 0));
            if (!p || p[0] < 0 || p[1] < 0 || p[0] > W - 1 || p[1] > H - 1) continue;
            const rgb = bil(S, W, H, p[0], p[1]);
            if (rgb[0] + rgb[1] + rgb[2] > 2) ok++;
          }
          return ok / (n * n);
        };
        let win = null;
        for (const size of [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]) {
          let bestW = null;
          for (let a = 0; a <= 6 && !bestW; a++) for (let b = 0; b <= 6; b++) {
            const u0 = (1 - size) * a / 6, v0 = (1 - size) * b / 6;
            const cv = cover(u0, v0, size, 40);
            if (cv >= 0.995) { bestW = {u0, v0, size, cv}; break; }
          }
          if (bestW) { win = bestW; break; }
        }
        if (!win) continue;
        const oc = document.createElement('canvas'); oc.width = oc.height = N;
        const octx = oc.getContext('2d'); const img = octx.createImageData(N, N); const D = img.data;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const m = 0.015 * win.size;                       // safety margin inside the chosen window
          const p = project(wpt(win.u0 + m + (win.size - 2 * m) * (i + 0.5) / N, win.v0 + m + (win.size - 2 * m) * (1 - (j + 0.5) / N), 0));
          const k = (j * N + i) * 4;
          if (!p || p[0] < 0 || p[1] < 0 || p[0] > W - 1 || p[1] > H - 1) { D[k + 3] = 255; continue; }
          const rgb = bil(S, W, H, p[0], p[1]);
          D[k] = lut[Math.round(rgb[0])]; D[k + 1] = lut[Math.round(rgb[1])]; D[k + 2] = lut[Math.round(rgb[2])]; D[k + 3] = 255;
        }
        octx.putImageData(img, 0, 0);
        const cand = {data: oc.toDataURL('image/jpeg', quality), filename: ent.filename, inside: win.cv, ins: 1 - win.size, frontal: s, window: win};
        if (!best || cand.inside > best.inside) best = cand;
        if (best.inside >= 0.995) break;
      }
    }
    if (!best) { out[id] = {error: 'no capture covers the swatch'}; continue; }
    out[id] = best;
  }
  return out;
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--inset", type=float, default=0.03)
    ap.add_argument("--quality", type=float, default=0.88)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.goto(a.url, wait_until="load")
        pg.add_script_tag(url="js/swatch_tex.js")          # pickFrontal (not loaded by the landing page)
        res = pg.evaluate(JS, {"ids": a.ids, "N": a.size, "inset": a.inset, "exposure": 1.2, "gamma": 2.2, "quality": a.quality})
        for mid, r in sorted(res.items(), key=lambda kv: int(kv[0])):
            if "error" in r:
                print(f"{mid}: {r['error']}"); continue
            data = base64.b64decode(r["data"].split(",", 1)[1])
            with open(os.path.join(a.out, f"{mid}.jpg"), "wb") as fh:
                fh.write(data)
            print(f"{mid}: {r['filename']}  cloth {r['inside']:.3f}  inset {r['ins']:.2f}  frontal {r['frontal']:.3f}  {len(data) / 1e3:.0f} KB")
        b.close()


if __name__ == "__main__":
    main()
