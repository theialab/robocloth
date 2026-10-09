#!/usr/bin/env python
"""Screenshots of the Results section (#main-viz) in headless Chrome, plus page health checks.

  export PYTHONNOUSERSITE=1
  python3 docs/tools/preview_server.py 8772 docs &
  /data/colin/envs/teaser/bin/python docs/tools/results/shoot_results.py [URL] [OUT_DIR]

For a desktop (1440x900), a tablet (834x1194) and a phone (390x844 @2x) viewport: the whole section
and its first screen; reports console errors / failed requests, broken images, horizontal overflow
(the page must not scroll sideways; the related-datasets table may scroll inside its .res-scroll
wrapper), the number of lines the setup paragraph takes (two at desktop width) and the chart's
rendered scale (its smallest text is 15 px in the SVG, so the scale is the smallest text in CSS px / 15).
URL defaults to http://127.0.0.1:8772/, OUT_DIR to <repo>/.cache/shots/.
"""
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8772/"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "..", ".cache", "shots")
VIEWPORTS = (("desktop", 1440, 900, 1), ("tablet", 834, 1194, 1), ("phone", 390, 844, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                              args=["--autoplay-policy=user-gesture-required"])
        for tag, w, h, dpr in VIEWPORTS:
            ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=dpr,
                                is_mobile=(tag == "phone"), has_touch=(tag == "phone"))
            pg = ctx.new_page()
            msgs, fails = [], []
            pg.on("console", lambda m: msgs.append((m.type, m.text)) if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: msgs.append(("pageerror", str(e))))
            # (a request the browser itself cancels -- e.g. a <video preload> -- is not a failure)
            pg.on("requestfailed", lambda r: fails.append((r.url, r.failure))
                  if "ERR_ABORTED" not in str(r.failure) else None)
            pg.on("response", lambda r: fails.append((r.url, r.status)) if r.status >= 400 else None)
            pg.goto(URL, wait_until="load")
            sec = pg.locator("#main-viz")
            sec.scroll_into_view_if_needed()
            # walk through the section so every lazy image loads
            pg.evaluate("""async () => {
                const s = document.getElementById('main-viz');
                const top = s.getBoundingClientRect().top + scrollY;
                for (let y = top; y < top + s.offsetHeight; y += innerHeight / 2) {
                    scrollTo(0, y); await new Promise(r => setTimeout(r, 120));
                }
                await Promise.all([...s.querySelectorAll('img')].map(i => i.complete ? 0 :
                    new Promise(r => { i.onload = i.onerror = r; })));
            }""")
            pg.wait_for_timeout(300)
            pg.evaluate("document.getElementById('main-viz').scrollIntoView()")
            pg.wait_for_timeout(200)
            pg.screenshot(path=os.path.join(OUT, f"results_{tag}_first_screen.png"))
            sec.screenshot(path=os.path.join(OUT, f"results_{tag}.png"))
            info = pg.evaluate("""(() => {
                const de = document.documentElement;
                const wide = [...document.querySelectorAll('#main-viz *')].filter(e => {
                    const r = e.getBoundingClientRect();
                    return r.right > de.clientWidth + 1 && !e.closest('.res-scroll');
                }).map(e => e.tagName + '.' + e.className).slice(0, 5);
                const p = document.querySelector('#main-viz .res-setup');
                const lines = p.getBoundingClientRect().height / parseFloat(getComputedStyle(p).lineHeight);
                const img = document.querySelector('#main-viz .res-chart-img');
                const vb = parseFloat(img.currentSrc.includes('narrow') ? 330 : 880);
                const scale = img.getBoundingClientRect().width / vb;
                return {scrollW: de.scrollWidth, clientW: de.clientWidth, offenders: wide,
                        setupLines: Math.round(lines * 10) / 10,
                        chart: img.currentSrc.split('/').pop(), chartScale: Math.round(scale * 100) / 100,
                        broken: [...document.querySelectorAll('#main-viz img')]
                            .filter(i => !i.naturalWidth).map(i => i.currentSrc || i.src)};
            })()""")
            ctx.close()
            print(f"[{tag} {w}x{h}@{dpr}] chart: {info['chart']} at {info['chartScale']}x "
                  f"(smallest text {15 * info['chartScale']:.1f} css px); setup paragraph: {info['setupLines']} lines")
            if info["scrollW"] > info["clientW"] or info["offenders"]:
                ok = False
                print(f"  FAIL horizontal overflow: {info}")
            else:
                print(f"  ok   no horizontal overflow ({info['scrollW']} <= {info['clientW']})")
            if tag == "desktop" and info["setupLines"] > 2:
                ok = False
                print(f"  FAIL the setup paragraph takes {info['setupLines']} lines at desktop width (max 2)")
            if info["broken"]:
                ok = False
                print(f"  FAIL broken images: {info['broken']}")
            else:
                print("  ok   all section images loaded")
            for t, m in msgs:
                print(f"  console {t}: {m[:200]}")
            for u, f in fails:
                print(f"  request failed: {u[:120]} -> {f}")
            if any(t in ("error", "pageerror") for t, _ in msgs) or fails:
                ok = False
        b.close()
    print("screenshots in", os.path.abspath(OUT))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
