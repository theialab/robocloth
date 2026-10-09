#!/usr/bin/env python
"""Frame-exact scripted recordings of the room panel inside the full project page.

The page runs on a virtual clock (window.__demoClock, honoured by js/room_core.js and the
variant scripts); every 1/30 s of virtual time the scripted pointer / touch events are
dispatched, one frame is drawn (roomDemo.frame()) and the viewport is screenshotted -> ffmpeg.
(Same approach as R1's ~/projects/robocloth-room-demo/tools/record_frames.py.)

  export PYTHONNOUSERSITE=1
  /data/colin/envs/teaser/bin/python docs/tools/room/record.py a [desktop|phone|both]

Each recording starts with the panel's own teaching loop (two cycles), then a scripted visitor
takes over (desktop: the mouse enters, real drags; phone: taps).
-> OUT/room_a5_<desktop|desktop2x|phone>.mp4 (1440x900 @1x -> the 1920 set, @2x -> the 4K set,
   390x844 @2x -> the 1920 set; 30 fps), OUT/stills_a5/*.png, OUT/room_a5_<kind>_marks.txt
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright

URLS = {"a": "http://127.0.0.1:8765/"}
NAME = {"a": "a5"}       # r5 (10 materials, 11 pieces, 4K set, teaching loop); a4/a3/a = earlier
OUT = "/media/raid/cloth/output/teaser_video/room_demo/variants"
ARGS = ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--autoplay-policy=user-gesture-required"]
FPS = 30
T0 = 100.0

OVERLAY = r"""
window.addEventListener('DOMContentLoaded', () => {
  const st = document.createElement('style');
  st.textContent = `#__cur{position:fixed;left:0;top:0;width:22px;height:22px;z-index:2000;pointer-events:none;transform:translate(-3px,-2px);opacity:0}
   .__tap{position:fixed;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;background:rgba(51,103,214,.22);
     border:2px solid rgba(51,103,214,.75);z-index:2000;pointer-events:none}
   #__finger{position:fixed;width:38px;height:38px;margin:-19px 0 0 -19px;border-radius:50%;background:rgba(51,103,214,.25);
     border:2px solid rgba(51,103,214,.7);z-index:2000;pointer-events:none;opacity:0}`;
  document.head.appendChild(st);
  const taps = [];
  if (!window.__touch) {
    const c = document.createElement('div'); c.id = '__cur';
    c.innerHTML = '<svg viewBox="0 0 22 22" width="22" height="22"><path d="M2 1.5 L2 17 L6.2 13.2 L9 19.5 L11.6 18.4 L8.9 12.2 L14.5 12.2 Z" fill="#111" stroke="#fff" stroke-width="1.3" stroke-linejoin="round"/></svg>';
    document.body.appendChild(c);
    window.addEventListener('pointermove', e => { c.style.opacity = 1; c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  } else {
    const f = document.createElement('div'); f.id = '__finger'; document.body.appendChild(f);
    window.__finger = (x, y, o) => { f.style.left = x + 'px'; f.style.top = y + 'px'; f.style.opacity = o; };
    window.addEventListener('pointerdown', e => {
      const d = document.createElement('div'); d.className = '__tap';
      d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px';
      document.body.appendChild(d); taps.push({d, t: window.__demoClock}); }, true);
  }
  window.__overlayTick = () => {
    for (const k of taps) {
      const u = Math.min(1, (window.__demoClock - k.t) / 0.55);
      k.d.style.transform = `scale(${0.4 + 0.9 * u})`; k.d.style.opacity = String(1 - u);
    }
  };
});
"""


class Timeline:
    """Pointer script in virtual seconds. Positions are client px, or a JS expression (str)
    resolved in the page when the event fires (phones: the room pans)."""

    def __init__(self, start=None):
        self.t = 0.0
        self.pos = start
        self.segs = []      # (t0, t1, a, b, arc)
        self.events = []    # (t, kind, pos)
        self.marks = []
        self.stills = {}

    def wait(self, d):
        self.t += d

    def move(self, b, dur, arc=0.0):
        self.segs.append((self.t, self.t + dur, self.pos, b, arc))
        self.t += dur
        self.pos = b

    def ev(self, kind, pos=None, name=None):
        self.events.append((self.t, kind, pos if pos is not None else self.pos))
        if name:
            self.marks.append((round(self.t, 2), name))

    def still(self, name, dt=0.0):
        self.stills[round(self.t + dt, 4)] = name

    def resolve(self, pg):
        """Resolve JS-expression targets of motion segments (in order, as the script plays)."""
        self._pg = pg

    def at(self, t):
        p = None
        for i, (t0, t1, a, b, arc) in enumerate(self.segs):
            if t0 > t:
                break
            if isinstance(b, str):
                b = tuple(self._pg.evaluate(b))
                nxt = self.segs[i + 1] if i + 1 < len(self.segs) else None
                self.segs[i] = (t0, t1, a, b, arc)
                if nxt is not None and isinstance(nxt[2], str):
                    self.segs[i + 1] = (nxt[0], nxt[1], b, nxt[3], nxt[4])
            if isinstance(a, str):
                a = tuple(self._pg.evaluate(a))
                self.segs[i] = (t0, t1, a, b, arc)
            if t0 <= t:
                u = min(1.0, (t - t0) / max(1e-6, t1 - t0))
                e = u * u * (3 - 2 * u)
                p = (a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e + arc * math.sin(math.pi * e))
        return p


# ---------------------------------------------------------------- variant A scripts
def _ids(pg):
    return pg.evaluate("roomDemo.core.MATS.map(m => m.id)"), pg.evaluate("roomDemo.core.TGT.map(t => t.id)")


def script_a_desktop(pg):
    """the loop runs two cycles on its own, then the visitor's mouse enters (the loop stops
    and the room fades to grey) and drags two materials onto single pillows / a sofa"""
    ids, tg = _ids(pg)
    TL = lambda m: f"roomDemo.tileCenter({ids.index(m)})"
    TP = lambda k: f"roomDemo.targetPoint({tg.index(k)})"
    tl = Timeline(None)
    tl.marks.append((0.0, "teaching loop (cycle 1)"))
    for t_, name in ((2.4, "a1_loop_card_decodes_in_hand"), (3.6, "a2_loop_carry_sofa_highlight"),
                     (5.3, "a3_loop_wipe_on_sofa"), (13.0, "a4_loop_dressing_piece_by_piece"),
                     (21.0, "a5_loop_cycle_end_back_to_grey_hint")):
        tl.stills[t_] = name
    tl.wait(44.0)
    out = pg.evaluate("(() => { const r = document.getElementById('room-panel').getBoundingClientRect(); return [r.right + 60, r.bottom - 40]; })()")
    tl.pos = tuple(out)
    tl.move(tuple(out), 0.05)
    tl.marks.append((round(tl.t, 2), "visitor's mouse enters: the loop stops"))
    for n, (m, k, tag) in enumerate([(226, "pillow55", "#226 -> left sofa, left pillow"),
                                     (370, "sofaR", "#370 -> right sofa (+ its grey pillows)"),
                                     (367, "pillow5", "#367 -> right sofa, right pillow")]):
        tl.move(TL(m), 0.9 if n == 0 else 0.65, arc=-30)
        tl.wait(0.18)
        tl.ev("down", name="press " + tag.split()[0])
        tl.move(TL(m), 0.12)
        tl.move(TP(k), 1.0, arc=-40)
        tl.wait(0.4)
        tl.ev("up", name="drop " + tag)
        if n == 2:
            tl.still("a6_visitor_single_pillow", 0.5)
        tl.wait(1.6)
    tl.wait(1.0)
    return tl


def script_a_phone(pg):
    ids, tg = _ids(pg)
    TL = lambda m: f"roomDemo.tileCenter({ids.index(m)})"
    TP = lambda k: f"roomDemo.targetPoint({tg.index(k)})"
    tl = Timeline(None)
    tl.marks.append((0.0, "teaching loop, tap-tap (cycle 1)"))
    for t_, name in ((1.6, "p1_loop_tap_tile_decodes"), (3.6, "p2_loop_tap_piece_wipe"),
                     (12.0, "p3_loop_dressing"), (24.0, "p4_loop_cycle_end_hint")):
        tl.stills[t_] = name
    tl.wait(50.0)
    tl.ev("tap", TL(226), "visitor taps #226: the loop stops")
    tl.wait(1.0)
    tl.ev("swipe", ("target", tg.index("sofaR"), 0.6), "swipe to the right sofa")
    tl.wait(0.8)
    tl.ev("tap", TP("pillow3"), "tap the right sofa's left pillow")
    tl.still("p5_visitor_single_pillow", 0.6)
    tl.wait(2.0)
    tl.ev("tap", TL(370), "tap #370")
    tl.wait(0.9)
    tl.ev("tap", TP("sofaR"), "tap the right sofa")
    tl.wait(2.2)
    tl.still("p6_visitor_dressed")
    tl.wait(0.4)
    return tl


SCRIPTS = {("a", "desktop"): script_a_desktop, ("a", "desktop2x"): script_a_desktop,
           ("a", "phone"): script_a_phone}


def run(p, variant, kind):
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=ARGS)
    if kind.startswith("desktop"):
        ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                                  device_scale_factor=2 if kind == "desktop2x" else 1)
        ctx.add_init_script(OVERLAY)
    else:
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                  is_mobile=True, has_touch=True)
        ctx.add_init_script("window.__touch = true;" + OVERLAY)
    pg = ctx.new_page()
    logs = []
    pg.on("pageerror", lambda e: logs.append(str(e)))
    pg.goto(os.environ.get("ROOM_URL", URLS[variant]) + "?rec=1")
    # bring the panel to the middle of the screen (it lazy-loads near the viewport)
    pg.evaluate("document.getElementById('apply').scrollIntoView({block:'center'})")
    pg.wait_for_function("window.roomDemo !== undefined && document.querySelector('.room-ready')", timeout=60000)
    pg.wait_for_timeout(600)
    pg.evaluate("document.querySelectorAll('video').forEach(v => { v.pause(); })")
    pg.evaluate(f"window.__demoClock = {T0}")
    pg.wait_for_timeout(150)
    pg.evaluate("roomDemo.demo.arm()")
    cdp = ctx.new_cdp_session(pg)
    touching = False
    lastT = None
    scroll0 = pg.evaluate("window.scrollY")
    tl = SCRIPTS[(variant, kind)](pg)
    tl.resolve(pg)

    os.makedirs(OUT, exist_ok=True)
    sdir = os.path.join(OUT, f"stills_{NAME[variant]}")
    os.makedirs(sdir, exist_ok=True)
    out = os.path.join(OUT, f"room_{NAME[variant]}_{kind}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS),
                           "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-preset", "slow", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    nf = int(round(tl.t * FPS))
    ev = sorted(tl.events, key=lambda e: e[0])
    ei = 0
    last = None
    swipe = None
    for f in range(nf):
        t = f / FPS
        pg.evaluate(f"window.__demoClock = {T0 + t}")
        if kind.startswith("desktop"):
            q = tl.at(t)
            if q is not None and q != last:
                pg.mouse.move(q[0], q[1])
                last = q
        while ei < len(ev) and ev[ei][0] <= t + 1e-6:
            _, what, pos = ev[ei]
            if isinstance(pos, str):
                pos = pg.evaluate(pos)
            if what == "down":
                pg.mouse.down()
            elif what == "up":
                pg.mouse.up()
            elif what.startswith("key:"):
                pg.keyboard.press(what[4:])
            elif what == "tap":
                pg.touchscreen.tap(pos[0], pos[1])
            elif what == "tdown":
                touching = True
                lastT = pos
                cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": pos[0], "y": pos[1]}]})
            elif what == "tup":
                touching = False
                cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
                pg.evaluate("window.__finger && __finger(0, 0, 0)")
            elif what == "swipe":
                d, dur = pos[0], pos[1]
                info = pg.evaluate("(() => { const v = document.querySelector('.room-view'); const r = v.getBoundingClientRect();"
                                   " return [v.scrollLeft, v.scrollWidth - v.clientWidth, r.left, r.top, r.width, r.height]; })()")
                s0, smax, vx, vy, vw, vh = info
                if d == "target":
                    k = dur
                    dur = pos[2]
                    to = pg.evaluate(f"(() => {{ const c = roomDemo.core, b = c.TGT[{k}].bbox, v = c.view;"
                                     f" const cx = (b[0] + b[2]) / 2 / c.W * c.stage.clientWidth;"
                                     f" return Math.max(0, Math.min(v.scrollWidth - v.clientWidth, cx - v.clientWidth / 2)); }})()")
                    d = "right" if to > s0 else "left"
                else:
                    to = smax if d == "right" else 0
                swipe = dict(t0=t, dur=dur, s0=s0, s1=to, x=vx + vw * (0.75 if d == "right" else 0.25),
                             dx=-(to - s0) * 0.8, y=vy + vh * 0.62)
            ei += 1
        if touching:
            q = tl.at(t)
            if q is not None:
                if q != lastT:
                    cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": q[0], "y": q[1]}]})
                    lastT = q
                pg.evaluate(f"window.__finger && __finger({q[0]}, {q[1]}, 0.9)")
        if swipe is not None:
            u = min(1.0, (t - swipe["t0"]) / swipe["dur"])
            e = u * u * (3 - 2 * u)
            pg.evaluate(f"document.querySelector('.room-view').scrollLeft = {swipe['s0'] + (swipe['s1'] - swipe['s0']) * e};"
                        f"window.__finger && __finger({swipe['x'] + swipe['dx'] * e}, {swipe['y']}, {0.9 if u < 1 else 0})")
            if u >= 1:
                swipe = None
        pg.evaluate("roomDemo.frame(); window.__overlayTick && __overlayTick();")
        ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        for ts, name in tl.stills.items():
            if abs(ts - t) < 0.5 / FPS:
                pg.screenshot(path=os.path.join(sdir, f"{kind}_{name}.png"))
    ff.stdin.close()
    ff.wait()
    with open(os.path.join(OUT, f"room_{NAME[variant]}_{kind}_marks.txt"), "w") as fh:
        for t, n in tl.marks:
            fh.write(f"{t:6.2f}s  {n}\n")
    scroll1 = pg.evaluate("window.scrollY")
    ctx.close()
    browser.close()
    print(variant, kind, "->", out, f"{nf} frames ({tl.t:.1f} s)", "errors:", logs[:5],
          "page scroll moved:", scroll1 - scroll0)


def main():
    variant = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    with sync_playwright() as p:
        for kind in ("desktop", "desktop2x", "phone"):
            if which in (kind, "all") or (which == "both" and kind != "desktop2x"):
                run(p, variant, kind)


if __name__ == "__main__":
    main()
