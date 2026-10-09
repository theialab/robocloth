"""Pre-render the landing-page preview thumbnails to images/thumbs/<id>.jpg.

The landing page shows these JPGs instead of range-fetching and tone-mapping
a ~10 MB 16-bit PNG per card. To keep them identical to the in-browser HDR
rendering, this script drives the page's own code (TarReader + HDRDisplay,
same exposure 1.2 / gamma 2.2 as js/main.js:loadThumb) in headless Chrome and
saves each canvas with toDataURL('image/jpeg', 0.85) at 400 px width.

Usage (needs playwright; any local static server for docs/):
    cd docs && python3 -m http.server 8765 &
    python3 tools/export_thumbs.py --url http://127.0.0.1:8765/ [--ids 0 32 ...]
Without --ids it exports every entry of data/manifest.json "preview".
Cards whose JPG is missing fall back to the HDR path automatically.
"""
import argparse, base64, os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "thumbs")

JS = """async ({ids, width, quality}) => {
  const m = await (await fetch(CONFIG.manifestURL())).json();
  const out = {};
  for (const e of m.preview) {
    if (ids.length && !ids.includes(e.id)) continue;
    const tarURL = CONFIG.hfResolve(`${CONFIG.materialDir(e.id)}/hdr.tar`);
    const index = await TarReader.loadOrBuildIndex(tarURL, CONFIG.tarIndexURL(e.id));
    const name = (e.thumb && e.thumb.filename) ||
                 Object.keys(index).filter(n => n.endsWith('.png')).sort()[0];
    const buf = await TarReader.fetchFile(tarURL, index, name);
    const cv = document.createElement('canvas');
    await HDRDisplay.drawPng(buf, cv, {exposure: 1.2, gamma: 2.2, maxWidth: width,
                                       bbox: e.thumb && e.thumb.bbox});
    out[e.id] = cv.toDataURL('image/jpeg', quality);
  }
  return out;
}"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--quality", type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page()
        pg.goto(a.url)
        pg.wait_for_function("() => window.HDRDisplay && window.TarReader && window.CONFIG")
        res = pg.evaluate(JS, {"ids": a.ids, "width": a.width, "quality": a.quality})
        b.close()
    for i, url in res.items():
        path = os.path.join(OUT, f"{i}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(url.split(",", 1)[1]))
        print(path, os.path.getsize(path))

if __name__ == "__main__":
    main()
