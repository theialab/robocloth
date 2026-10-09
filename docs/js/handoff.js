/* Method video -> "Apply a trained material" hand-off.
 *
 * The method video (#method-video, muted, looping) ends on the flat 4x3 grid of 12 optimised
 * latent grids (method_light_v5); ten of them are the room panel's materials. The first time
 * the video reaches that last frame while it is on screen and the panel's tile row is in view
 * (or just below it), playback holds on the last frame, the other two tiles dim, and the ten
 * lift out of the picture and fly into the panel's tile row. Each lands on its slot and shrinks
 * away into the room's own tile beneath it, the latent texture you can drag (the reverse of the
 * decode the room plays when a tile is picked up). Then the video loops on.
 *
 * It runs at most once and never under prefers-reduced-motion, before the room is ready, while
 * the room's own gesture demo is playing, or after the visitor has touched the panel; it gives
 * way at once if any of that happens mid-flight. Otherwise the panel does not depend on it:
 * its tiles stay as they are unless the hand-off is actually running.
 *
 * Which tiles fly = the panel's tiles whose ids are in media/robocloth_method_tiles.json (tile
 * rectangles in video px, written with the video by the teaser repo's
 * tools/thread/method_video_v5.py --web-cut). Driven from the outside, through the DOM only
 * (room_core.js / room_a.js are not touched); styles in css/handoff.css.
 */
(function () {
    "use strict";
    const video = document.getElementById("method-video");
    const panel = document.getElementById("room-panel");
    if (!video || !panel || !window.requestAnimationFrame || !window.fetch) return;
    if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const JSON_URL = "media/robocloth_method_tiles.json?v=7";
    const CSS_URL = "css/handoff.css?v=1";
    const ARM_S = 1.2;            // loop is switched off this close to the end, if the hand-off may run
    const JUST_BELOW = 48;        // the tile row may start this far below the viewport (px)
    const LIFT = 0.32;            // the others dim, the ten lift (s)
    const STAGGER = 0.09;         // between flights
    const FLY = 0.95;             // one flight
    const SETTLE = 0.10;          // landed -> the material starts to shrink away
    const WIPE = 0.42;            // the material shrinks into the latent tile
    const TAIL = 0.30;            // last tile settled -> the video loops on
    const SHADOW_IN = "0 1px 3px rgba(20, 30, 50, 0.18)";   // = .room-tile

    let meta = null;
    let state = "idle";           // idle | armed | running | done
    let touched = false;
    let runT0 = null;

    fetch(JSON_URL).then((r) => r.json()).then((m) => {
        if (!m || !Array.isArray(m.tiles)) throw new Error("bad tiles json");
        meta = m;
        const l = document.createElement("link");
        l.rel = "stylesheet";
        l.href = CSS_URL;
        document.head.appendChild(l);
    }).catch(() => { state = "done"; });
    ["pointerdown", "keydown"].forEach((ev) => panel.addEventListener(ev, () => { touched = true; }, true));

    const clamp01 = (x) => Math.min(1, Math.max(0, x));
    const smooth = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
    const easeInOut = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
    const lerp = (a, b, k) => a + (b - a) * k;
    // window.__handoffClock (s), when set, replaces the wall clock (frame-exact scripted checks)
    const now = () => (window.__handoffClock !== undefined ? window.__handoffClock : performance.now() / 1000);

    // ------------------------------------------------------------------ DOM reading
    /** The panel's tiles, in row order: [{id, el}] (id from the tile's "#453" tag). */
    function roomTiles() {
        const out = [];
        panel.querySelectorAll(".room-side .room-tile").forEach((el) => {
            const tag = el.querySelector(".tile-tag");
            const m = /#(\d+)/.exec((tag && tag.textContent) || el.title || "");
            if (m) out.push({ id: +m[1], el });
        });
        return out;
    }
    function visibleFrac(r) {
        const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
        const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
        return r.width > 0 && r.height > 0 ? (w * h) / (r.width * r.height) : 0;
    }
    const opacityOf = (el) => (el ? parseFloat(getComputedStyle(el).opacity) || 0 : 0);
    /** The room's own gesture demo (or the visitor) has a tile in hand. */
    function roomBusy() {
        // the visitor's own drag / tap selection blocks the hand-off; the panel's teaching loop
        // does not -- it yields to the hand-off (js/room_a.js handoffBusy) and resumes after it
        const rd = window.roomDemo;
        if (rd && rd.visitorBusy) return rd.visitorBusy();
        if (panel.querySelector(".room-tile.lifted, .room-tile.selected")) return true;
        return opacityOf(document.querySelector(".room-demo-hand")) > 0.02 ||
            opacityOf(document.querySelector(".room-ghost")) > 0.02;
    }
    function flyers() {
        const ids = new Set(meta.tiles.map((t) => t.id));
        return roomTiles().filter((t) => ids.has(t.id));
    }
    function mayRun() {
        // runs every time the video ends (author), over the panel's own loop; only a visitor's
        // drag / selection or an invisible panel holds it back
        if (!meta || document.hidden || state === "running") return false;
        if (!panel.classList.contains("room-ready") || !flyers().length) return false;
        if (visibleFrac(video.getBoundingClientRect()) < 0.6) return false;
        const side = panel.querySelector(".room-side");
        if (!side) return false;
        const sr = side.getBoundingClientRect();
        if (sr.bottom <= 0 || sr.top > innerHeight + JUST_BELOW) return false;
        return !roomBusy();
    }

    /** Where the video picture is drawn (viewport px), letterboxing / object-fit included. */
    function pictureRect() {
        const r = video.getBoundingClientRect();
        const cs = getComputedStyle(video);
        const px = (k) => parseFloat(cs[k]) || 0;
        const l = px("borderLeftWidth") + px("paddingLeft"), t = px("borderTopWidth") + px("paddingTop");
        const cw = r.width - l - px("borderRightWidth") - px("paddingRight");
        const ch = r.height - t - px("borderBottomWidth") - px("paddingBottom");
        const vw = video.videoWidth || meta.width, vh = video.videoHeight || meta.height;
        let w = cw, h = ch;
        if (cs.objectFit !== "fill") {
            let s = cs.objectFit === "cover" ? Math.max(cw / vw, ch / vh) : Math.min(cw / vw, ch / vh);
            if (cs.objectFit === "none") s = 1;
            if (cs.objectFit === "scale-down") s = Math.min(1, s);
            w = vw * s;
            h = vh * s;
        }
        const pos = (cs.objectPosition || "50% 50%").split(/\s+/);
        const frac = (v, axis) => {
            if (!v || v === "center") return 0.5;
            if (v === "left" || v === "top") return 0;
            if (v === "right" || v === "bottom") return 1;
            if (/%$/.test(v)) return parseFloat(v) / 100;
            return (parseFloat(v) || 0) / Math.max(1, axis === "x" ? cw - w : ch - h);
        };
        return { x: r.left + l + (cw - w) * frac(pos[0], "x"), y: r.top + t + (ch - h) * frac(pos[1] || pos[0], "y"),
            w, h, frameX: r.left + l, frameY: r.top + t, frameW: cw, frameH: ch };
    }
    function tileRect(P, t) {
        const kx = P.w / meta.width, ky = P.h / meta.height;
        return { x: P.x + t.x * kx, y: P.y + t.y * ky, w: t.w * kx, h: t.h * ky };
    }

    // ------------------------------------------------------------------ playback
    function playIfSeen() {
        if (visibleFrac(video.getBoundingClientRect()) >= 0.25) {
            const p = video.play();
            if (p && p.catch) p.catch(() => { /* blocked; controls remain */ });
        } else {
            video._userPaused = false;      // js/media.js: start it again when it scrolls back in
        }
    }
    function restart(after) {
        video.loop = true;
        let fired = false;
        const go = () => {
            if (fired) return;
            fired = true;
            video.removeEventListener("seeked", go);
            if (after) after();
            playIfSeen();
        };
        video.addEventListener("seeked", go);
        setTimeout(go, 600);                // in case "seeked" never comes
        try { video.currentTime = 0; } catch (e) { go(); }
    }

    video.addEventListener("timeupdate", () => {
        if (state !== "idle" && state !== "armed") return;
        const d = video.duration;
        if (!isFinite(d) || d <= 0) return;
        const left = d - video.currentTime;
        if (state === "idle") {
            if (left < ARM_S && left > 0.04 && !video.paused && !video.seeking && mayRun()) {
                state = "armed";
                video.loop = false;          // so that it stops on the last frame ("ended")
            }
        } else if (left >= ARM_S + 0.3 || !mayRun()) {
            state = "idle";                  // seeked back, scrolled away, room busy...
            video.loop = true;
        }
    });
    // Paused while armed (scrolled out of view, or by the visitor): "ended" will not come, so
    // stand down -- the loop below must not wait for us. It re-arms next time the end nears.
    video.addEventListener("pause", () => {
        // (a non-looping video fires "pause" just before "ended": that one is the hand-off's cue,
        // not a stand-down)
        const atEnd = video.ended || video.currentTime >= (video.duration || Infinity) - 0.05;
        if (state === "armed" && !atEnd) { state = "idle"; video.loop = true; }
    });
    video.addEventListener("ended", () => {
        if (state !== "armed") return;
        if (mayRun()) run();
        else { state = "idle"; restart(); }
    });

    // ------------------------------------------------------------------ the hand-off
    function run() {
        state = "running";
        const tiles = flyers();
        const byId = new Map(meta.tiles.map((t) => [t.id, t]));
        const vw = video.videoWidth || meta.width, vh = video.videoHeight || meta.height;
        const sx = vw / meta.width, sy = vh / meta.height;

        // the picture's own background colour (its decoded #fafbfc), for the dim and the holes
        let bg = "rgb(250, 251, 252)";
        const nodes = [];
        const clones = [];
        try {
            const probe = document.createElement("canvas");
            probe.width = probe.height = 4;
            const pc = probe.getContext("2d");
            pc.drawImage(video, 8 * sx, (meta.height - 40) * sy, 4 * sx, 4 * sy, 0, 0, 4, 4);
            const d = pc.getImageData(1, 1, 1, 1).data;
            bg = `rgb(${d[0]}, ${d[1]}, ${d[2]})`;
            tiles.forEach((rt) => {
                const t = byId.get(rt.id);
                const cv = document.createElement("canvas");
                cv.width = Math.max(2, Math.round(t.w * sx));
                cv.height = Math.max(2, Math.round(t.h * sy));
                cv.getContext("2d").drawImage(video, t.x * sx, t.y * sy, t.w * sx, t.h * sy, 0, 0, cv.width, cv.height);
                const el = document.createElement("div");
                el.className = "handoff-clone";
                el.setAttribute("aria-hidden", "true");
                el.appendChild(cv);
                clones.push({ rt, t, el, radius: parseFloat(getComputedStyle(rt.el).borderTopLeftRadius) || 8 });
            });
        } catch (e) {                        // no frame to copy: no hand-off
            state = "done";
            restart();
            return;
        }
        const dim = document.createElement("div");
        dim.className = "handoff-dim";
        dim.style.background = bg.replace("rgb(", "rgba(").replace(")", ", 0.8)");
        document.body.appendChild(dim);
        nodes.push(dim);
        const holes = clones.map((c) => {
            const h = document.createElement("div");
            h.className = "handoff-hole";
            h.style.background = bg;
            document.body.appendChild(h);
            nodes.push(h);
            return h;
        });
        clones.forEach((c) => { document.body.appendChild(c.el); nodes.push(c.el); });
        const side = panel.querySelector(".room-side");
        side.classList.add("handoff-on");
        tiles.forEach((rt) => rt.el.classList.add("handoff-slot"));
        const hadControls = video.controls;
        video.controls = false;              // no paused-player bar over the picture meanwhile

        const t0 = now();
        runT0 = t0;
        const tEnd = LIFT + (clones.length - 1) * STAGGER + FLY + SETTLE + WIPE;
        let finished = false;

        function finish() {
            if (finished) return;
            finished = true;
            state = "idle";                  // ready for the next loop of the video
            tiles.forEach((rt) => rt.el.classList.remove("handoff-slot"));
            clones.forEach((c) => c.el.remove());
            setTimeout(() => side.classList.remove("handoff-on"), 400);
            restart(() => {
                nodes.forEach((n) => n.remove());
                video.controls = hadControls;
            });
        }

        function frame() {
            if (finished) return;
            const u = now() - t0;
            if (document.hidden || roomBusy() || visibleFrac(video.getBoundingClientRect()) < 0.1) {
                finish();
                return;
            }
            const P = pictureRect();
            const s = P.w / meta.width;                     // page px per video px
            // the picture dims (its frame, rounded like the video)
            dim.style.transform = `translate(${P.frameX}px, ${P.frameY}px)`;
            dim.style.width = P.frameW + "px";
            dim.style.height = P.frameH + "px";
            dim.style.opacity = smooth(u / LIFT).toFixed(3);
            clones.forEach((c, i) => {
                const ts = LIFT + i * STAGGER;
                const from = tileRect(P, c.t);
                // the hole the tile leaves in the picture (its baked shadow included)
                const hole = holes[i], pad = 6 * s;
                hole.style.transform = `translate(${from.x - pad}px, ${from.y - pad + 3 * s}px)`;
                hole.style.width = from.w + 2 * pad + "px";
                hole.style.height = from.h + 2 * pad + "px";
                hole.style.boxShadow = `0 0 ${18 * s}px ${12 * s}px ${bg}`;
                hole.style.opacity = smooth((u - ts) / 0.22).toFixed(3);
                if (c.done) return;
                const to = c.rt.el.getBoundingClientRect();
                const k = clamp01((u - ts) / FLY), e = easeInOut(k);
                const A = [from.x + from.w / 2, from.y + from.h / 2], B = [to.left + to.width / 2, to.top + to.height / 2];
                const D = Math.hypot(B[0] - A[0], B[1] - A[1]);
                const C1 = [A[0], A[1] - 0.16 * D], C2 = [B[0], B[1] - 0.5 * D];
                const m = 1 - e;
                const x = m * m * m * A[0] + 3 * m * m * e * C1[0] + 3 * m * e * e * C2[0] + e * e * e * B[0];
                const y = m * m * m * A[1] + 3 * m * m * e * C1[1] + 3 * m * e * e * C2[1] + e * e * e * B[1];
                const lift = smooth(u / LIFT) * (1 - smooth((k - 0.55) / 0.45));
                const sc = 1 + 0.07 * lift;
                const w = lerp(from.w, to.width, e) * sc, h = lerp(from.h, to.height, e) * sc;
                const st = c.el.style;
                st.transform = `translate(${(x - w / 2).toFixed(1)}px, ${(y - h / 2).toFixed(1)}px)`;
                st.width = w.toFixed(1) + "px";
                st.height = h.toFixed(1) + "px";
                st.borderRadius = (c.radius * e).toFixed(1) + "px";
                st.boxShadow = lift > 0.01
                    ? `0 ${(3 + 9 * lift).toFixed(1)}px ${(6 + 22 * lift).toFixed(1)}px rgba(20, 30, 50, ${(0.18 + 0.14 * lift).toFixed(3)})`
                    : (k >= 1 ? SHADOW_IN : "none");
                // landed: the latent tile appears beneath, the material shrinks away into it
                if (k >= 1 && !c.landed) {
                    c.landed = true;
                    c.rt.el.classList.remove("handoff-slot");
                }
                const wu = (u - ts - FLY - SETTLE) / WIPE;
                if (wu > 0) {
                    const r = 102 - 112 * easeInOut(wu);    // % of the half-diagonal; soft edge 8 %
                    const g = `radial-gradient(circle farthest-corner at 50% 50%, #000 ${r.toFixed(1)}%, rgba(0, 0, 0, 0) ${(r + 8).toFixed(1)}%)`;
                    st.webkitMaskImage = g;
                    st.maskImage = g;
                }
                if (wu >= 1) {
                    c.done = true;
                    c.el.remove();
                }
            });
            if (u >= tEnd + TAIL) finish();
            else requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    // hooks for scripted checks (headless recordings); no effect on the page
    window.methodHandoff = { state: () => state, mayRun: () => !!meta && mayRun(), t0: () => runT0,
        duration: () => LIFT + (meta ? Math.max(0, flyers().length - 1) : 5) * STAGGER + FLY + SETTLE + WIPE + TAIL };
})();
