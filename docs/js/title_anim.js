/* ============================================================
 *  Animated "RoboCloth" header title.
 *
 *  The word is pieced from real fabric patches (the launch video's closing title). Every
 *  image is a real Mitsuba render of the same scene (media/title/title.json says which):
 *    lit.webp        the evenly lit word (the resting state; also the no-JS image)
 *    assembled.webp  the word under the assembly light
 *    sweep/sNN.webp  the moving LED, one render per position (left -> right)
 *    intro.webm      the patches flying in (VP9 + alpha: Chrome / Firefox / Edge only)
 *
 *  On load it plays once: assembly -> the lamp sweeps left to right -> evenly lit.
 *  Mouse: moving over the title moves the lamp (scrubs the sweep renders); leaving eases
 *  back to the evenly lit word.  Touch screens: after the intro a slow back-and-forth
 *  sweep loops; dragging on the title scrubs.  prefers-reduced-motion: the lit still only.
 *  Safari / iOS (no VP9 alpha): the assembly is skipped, the word fades in, then the same.
 *
 *  Light transitions between two states (assembly light -> LED, LED -> lit) are cross-
 *  fades of two renders, i.e. one light dimmed while the other comes up.
 * ============================================================ */
(function () {
    "use strict";

    var root = document.getElementById("title-anim");
    if (!root || !window.requestAnimationFrame) return;

    var mm = function (q) { return window.matchMedia ? window.matchMedia(q).matches : false; };
    if (mm("(prefers-reduced-motion: reduce)")) {
        root.classList.remove("ta-pending");
        root.classList.add("ta-static");
        return;
    }

    // the render set: data-src (any folder with the live file names: media/title/,
    // media/title/cand_A/, ...); for previewing, ?title=<folder under media/title/> overrides it
    var BASE = root.getAttribute("data-src") || "media/title/";
    try {
        var qs = new URLSearchParams(window.location.search).get("title");
        if (qs && /^[A-Za-z0-9_-]{1,40}$/.test(qs)) BASE = "media/title/" + qs + "/";
    } catch (e) { /* old browsers: data-src only */ }
    if (BASE.charAt(BASE.length - 1) !== "/") BASE += "/";
    root.setAttribute("data-set", BASE);
    var VER = root.getAttribute("data-v") || "1";
    var url = function (p) { return BASE + p + "?v=" + VER; };
    var HOVER = mm("(hover: hover) and (pointer: fine)");

    // ---------------------------------------------------------------- tuning
    var BAND = 0.72;            // lamp band above the word (H units) - also in title_anim.css
    var PADX = 0.075;           // canvas reaches up to this far (W units) beyond the stage sides
    var PADT = 0.30;            // ... and this far (H units) above it (the lamp's glow)
    var T_FADEIN = 0.5;         // s, no-video path: the assembled word fades in
    var T_DOWN = 0.40;          // s, assembly light -> LED (the lamp comes in from the left)
    var T_SWEEP = 1.25;         // s, the sweep (the video's 0.8 s was a 1080p shot)
    var T_RISE = 0.80;          // s, LED -> evenly lit, the lamp parks top right
    var T_START_MAX = 3.0;      // s after start: no intro if the assets are not there yet
    var T_LOOP = 5.5;           // s per direction, touch auto sweep
    var T_LOOP_HOLD = 1.4;      // s the lit word is shown before the touch loop starts
    var LAMP_K = 1.7;           // lamp glyph size x the video's
    var CONE_A = 0.34, EDGE_A = 0.42, GLOW_K = 0.62;
    var PARK = { u: 0.905, v: -0.47, au: 0.80, av: 0.45, a: 0.9 };

    // ---------------------------------------------------------------- state
    var meta = null, N = 0;
    var lit = null, asm = null, frames = [];
    var lampU = [], lampV = [], aimU = [], aimV = [], v1 = [];
    var vRefEnd = -0.709, vRefTop = -1.82;
    var cv = document.createElement("canvas");
    cv.setAttribute("aria-hidden", "true");
    var ctx = cv.getContext("2d");
    var video = null;
    var geo = null;             // canvas px per W / H unit, origin
    var padx = PADX, lampLo = -0.07, lampHi = 1.07;
    var mode = "loading";       // loading | intro | seq | idle | hover | loop | drag
    var seqT0 = 0, seqFrom = "asm";
    var m = 0, mTarget = 0, pos = 0, posTarget = 0;
    var loopPhase = 0, loopDir = 1, loopHold = 0, visible = true;
    var raf = 0, lastT = 0, t0 = performance.now();
    var dbg = { frames: 0, drawMs: 0, maxDrawMs: 0 };

    root.classList.remove("ta-pending");
    root.classList.add("ta-running");
    root.appendChild(cv);

    // ---------------------------------------------------------------- helpers
    function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
    function smooth(e0, e1, x) { var u = clamp((x - e0) / (e1 - e0), 0, 1); return u * u * (3 - 2 * u); }
    function lerp(a, b, t) { return a + (b - a) * t; }
    function setMode(s) { mode = s; root.setAttribute("data-mode", s); }

    function loadImage(src) {
        return new Promise(function (res, rej) {
            var im = new Image();
            im.decoding = "async";
            im.onload = function () {
                var done = function () {
                    if (window.createImageBitmap) {
                        createImageBitmap(im).then(res, function () { res(im); });
                    } else { res(im); }
                };
                if (im.decode) im.decode().then(done, done); else done();
            };
            im.onerror = rej;
            im.src = src;
        });
    }

    // ---------------------------------------------------------------- geometry
    function layout() {
        if (!meta) return;
        var r = meta.still.rect;                        // x, y, w, h (word units)
        var stageW = r[2], stageH = BAND + r[1] + r[3];
        var box = root.getBoundingClientRect();
        if (box.width < 2) return;
        // the canvas may reach beyond the stage, but never past the viewport (no sideways scroll)
        var pwCss = box.width / stageW;
        var vw = document.documentElement.clientWidth || window.innerWidth;
        var room = Math.max(0, Math.min(box.left, vw - box.right) - 2);
        padx = Math.min(PADX, room / pwCss);
        var glow = 54 * GLOW_K * LAMP_K / 1311;         // the lamp glow radius, W units
        lampLo = r[0] - padx + glow;
        lampHi = r[0] + stageW + padx - glow;
        cv.style.left = (-padx / stageW * 100) + "%";
        cv.style.width = ((stageW + 2 * padx) / stageW * 100) + "%";
        cv.style.top = (-PADT / stageH * 100) + "%";
        cv.style.height = ((stageH + PADT) / stageH * 100) + "%";
        var dpr = Math.min(window.devicePixelRatio || 1, 2);
        var cw = Math.max(2, Math.round(box.width * (stageW + 2 * padx) / stageW * dpr));
        var ch = Math.max(2, Math.round(box.height * (stageH + PADT) / stageH * dpr));
        if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
        var pw = cw / (stageW + 2 * padx);              // px per W unit
        var ph = ch / (stageH + PADT);                  // px per H unit
        geo = { pw: pw, ph: ph, ox: (padx - r[0]) * pw, oy: (BAND + PADT) * ph, r: r };
        if (video) {
            var ir = meta.intro.rect;
            video.style.left = ((ir[0] - r[0]) / stageW * 100) + "%";
            video.style.width = (ir[2] / stageW * 100) + "%";
            video.style.top = ((ir[1] + BAND) / stageH * 100) + "%";
            video.style.height = (ir[3] / stageH * 100) + "%";
        }
        kick();
    }
    function X(u) { return geo.ox + u * geo.pw; }
    function Y(v) { return geo.oy + v * geo.ph; }

    // ---------------------------------------------------------------- sweep lookup
    function nearestLoaded(i, dir) {
        for (var j = i; j >= 0 && j < N; j += dir) if (frames[j]) return j;
        return -1;
    }
    function sweepLayers(p, w, out) {
        p = clamp(p, 0, N - 1);
        var i0 = Math.floor(p), f = p - i0;
        var a = nearestLoaded(i0, -1), b = nearestLoaded(Math.min(i0 + 1, N - 1), 1);
        if (a < 0 && b < 0) return false;
        if (a < 0) a = b;
        if (b < 0) b = a;
        if (a === b) { out.push([frames[a], w]); return true; }
        var t = (p - a) / (b - a);
        if (t < 1) out.push([frames[a], w * (1 - t)]);
        if (t > 0) out.push([frames[b], w * t]);
        return true;
    }
    function interp(arr, p) {
        p = clamp(p, 0, N - 1);
        var i = Math.min(Math.floor(p), N - 2), f = p - i;
        return arr[i] + (arr[i + 1] - arr[i]) * f;
    }
    function posFromU(u) {                              // the frame whose lamp is at x = u
        if (u <= lampU[0]) return 0;
        if (u >= lampU[N - 1]) return N - 1;
        var lo = 0, hi = N - 1;
        while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (lampU[mid] <= u) lo = mid; else hi = mid; }
        return lo + (u - lampU[lo]) / (lampU[hi] - lampU[lo]);
    }
    function posFromV1(f) {                             // continuous v1 frame -> kept-frame index
        if (f <= v1[0]) return 0;
        if (f >= v1[N - 1]) return N - 1;
        var lo = 0, hi = N - 1;
        while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (v1[mid] <= f) lo = mid; else hi = mid; }
        return lo + (f - v1[lo]) / (v1[hi] - v1[lo]);
    }
    function lampAt(p) {
        // the video's schematic lamp path (x = the LED's ground position), flattened for a page header
        var u = interp(lampU, p), vr = interp(lampV, p), kn = 0.06;
        // soft clamp into the canvas (narrow screens): identity until kn from the limit
        if (u < lampLo + kn) u = lampLo + kn * Math.exp((u - lampLo - kn) / kn);
        if (u > lampHi - kn) u = lampHi - kn * Math.exp((lampHi - kn - u) / kn);
        var h = clamp((vRefEnd - vr) / (vRefEnd - vRefTop), 0, 1);
        return { u: u, v: -0.30 - 0.28 * h, au: interp(aimU, p), av: interp(aimV, p) };
    }

    // ---------------------------------------------------------------- drawing
    function draw(layers, L) {
        var c0 = performance.now();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;
        ctx.clearRect(0, 0, cv.width, cv.height);
        var r = geo.r, x = X(r[0]), y = Y(r[1]), w = r[2] * geo.pw, h = r[3] * geo.ph;
        // premultiplied sum: weights add up to <= 1, every layer has the same alpha shape
        ctx.globalCompositeOperation = "lighter";
        for (var i = 0; i < layers.length; i++) {
            if (layers[i][1] <= 0.002 || !layers[i][0]) continue;
            ctx.globalAlpha = Math.min(1, layers[i][1]);
            ctx.drawImage(layers[i][0], x, y, w, h);
        }
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;
        if (L && L.a > 0.003) drawLamp(L);
        var dt = performance.now() - c0;
        dbg.frames++; dbg.drawMs += dt; if (dt > dbg.maxDrawMs) dbg.maxDrawMs = dt;
    }

    function rgba(c, a) { return "rgba(" + c + "," + clamp(a, 0, 1).toFixed(4) + ")"; }
    // lamp palettes: "dark" = the video's light-on-dark glyph; "light" (light header, the
    // renders' title.json says theme "light", or data-theme="light"): a dark lamp with a
    // dark outline, a warm bulb, glow and cone
    var PAL = {
        dark: { light: "255,180,60", glow: "255,159,46", hot: "255,244,220", shade: "19,20,23",
                stroke: "255,180,60", edge: "255,180,60", coneA: CONE_A, edgeA: EDGE_A, glowA: 1.0 },
        light: { light: "255,163,38", glow: "255,146,24", hot: "255,222,150", shade: "46,48,54",
                 stroke: "22,23,27", edge: "222,128,20", coneA: 0.30, edgeA: 0.34, glowA: 1.25 }
    };
    function pal() {
        var t = root.getAttribute("data-theme") || (meta && meta.theme) || "dark";
        return PAL[t] || PAL.dark;
    }

    function drawLamp(L) {
        var C = pal(), LIGHT = C.light, GLOW = C.glow, HOT = C.hot, SHADE = C.shade;
        var k = geo.ph / 196 * LAMP_K;                  // video px -> canvas px
        var x = X(L.u), y = Y(L.v), ax = X(L.au), ay = Math.min(Y(L.av), Y(-0.10)), a = L.a;
        var dx = ax - x, dy = ay - y, d = Math.hypot(dx, dy) || 1;
        var ux = dx / d, uy = dy / d, px = -uy, py = ux;
        var P = function (u, v) { return [x + u * ux + v * px, y + u * uy + v * py]; };
        var len = 27 * k, mouth = len * 0.35, back = -len * 0.65, sm = 18 * k, sb = 9 * k;
        var stroke = Math.max(1, 2.2 * k);
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        if (L.cone > 0.003 && d > mouth + 20 * k) {
            var L1 = d - mouth, w0 = sm - k, w1 = w0 + L1 * Math.tan(8 * Math.PI / 180);
            var p0 = P(mouth, 0);
            var g = ctx.createLinearGradient(p0[0], p0[1], ax, ay);
            g.addColorStop(0, rgba(LIGHT, C.coneA * a * L.cone));
            g.addColorStop(0.45, rgba(LIGHT, 0.32 * C.coneA * a * L.cone));
            g.addColorStop(1, rgba(LIGHT, 0));
            var q = [P(mouth, w0), P(mouth + L1, w1), P(mouth + L1, -w1), P(mouth, -w0)];
            ctx.beginPath();
            ctx.moveTo(q[0][0], q[0][1]);
            for (var i = 1; i < 4; i++) ctx.lineTo(q[i][0], q[i][1]);
            ctx.closePath();
            ctx.fillStyle = g;
            ctx.fill();
            [1, -1].forEach(function (s) {
                var e0 = P(mouth, s * w0), e1 = P(mouth + L1, s * w1);
                var ge = ctx.createLinearGradient(e0[0], e0[1], e1[0], e1[1]);
                ge.addColorStop(0, rgba(C.edge, C.edgeA * a * L.cone));
                ge.addColorStop(1, rgba(C.edge, 0));
                ctx.beginPath();
                ctx.moveTo(e0[0], e0[1]);
                ctx.lineTo(e1[0], e1[1]);
                ctx.strokeStyle = ge;
                ctx.lineWidth = stroke;
                ctx.stroke();
            });
        }
        var b = P(mouth, 0), gr = 54 * k * GLOW_K;
        var rg = ctx.createRadialGradient(b[0], b[1], 0, b[0], b[1], gr);
        rg.addColorStop(0, rgba(GLOW, 0.55 * C.glowA * a));
        rg.addColorStop(0.25, rgba(GLOW, 0.30 * C.glowA * a));
        rg.addColorStop(0.55, rgba(GLOW, 0.10 * C.glowA * a));
        rg.addColorStop(1, rgba(GLOW, 0));
        ctx.beginPath();
        ctx.arc(b[0], b[1], gr, 0, 2 * Math.PI);
        ctx.fillStyle = rg;
        ctx.fill();
        var sh = [P(back, -sb), P(mouth, -sm), P(mouth, sm), P(back, sb)];
        ctx.beginPath();
        ctx.moveTo(sh[0][0], sh[0][1]);
        for (var j = 1; j < 4; j++) ctx.lineTo(sh[j][0], sh[j][1]);
        ctx.closePath();
        ctx.fillStyle = rgba(SHADE, 0.92 * a);
        ctx.fill();
        ctx.strokeStyle = rgba(C.stroke, a);
        ctx.lineWidth = stroke;
        ctx.stroke();
        var c1 = P(back - 5 * k, -4 * k), c2 = P(back - 5 * k, 4 * k);
        ctx.beginPath();
        ctx.moveTo(c1[0], c1[1]);
        ctx.lineTo(c2[0], c2[1]);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(b[0], b[1], 7.2 * k, 0, 2 * Math.PI);
        ctx.fillStyle = rgba(HOT, a);
        ctx.fill();
        ctx.strokeStyle = rgba(C.stroke, a);
        ctx.stroke();
    }

    function parkedLamp(a) { return { u: PARK.u, v: PARK.v, au: PARK.au, av: PARK.av, a: a, cone: 0 }; }
    function mixLamp(A, B, t) {
        return { u: lerp(A.u, B.u, t), v: lerp(A.v, B.v, t), au: lerp(A.au, B.au, t),
                 av: lerp(A.av, B.av, t), a: lerp(A.a, B.a, t), cone: lerp(A.cone, B.cone, t) };
    }

    // ---------------------------------------------------------------- the load sequence
    function seqFrame(t) {
        // t: s since the assembly ended (or since the fade-in started on the no-video path)
        var layers = [], L = null;
        var tf = seqFrom === "fade" ? T_FADEIN : 0;
        if (seqFrom === "fade" && t < tf) {
            layers.push([asm, smooth(0, tf, t)]);
            draw(layers, null);
            return false;
        }
        t -= tf;
        var p0 = lampAt(0);
        if (t < T_DOWN) {
            var a = smooth(0, T_DOWN, t);
            layers.push([asm, 1 - a]);
            if (!sweepLayers(0, a, layers)) layers[0][1] = 1;
            var e = 1 - Math.pow(1 - a, 2);
            L = { u: p0.u - 0.08 * (1 - e), v: p0.v - 0.10 * (1 - e), au: p0.au, av: p0.av, a: a, cone: a };
        } else if (t < T_DOWN + T_SWEEP) {
            var s = (t - T_DOWN) / T_SWEEP;             // uniform in the video's frame time
            var p = posFromV1(lerp(v1[0], v1[N - 1], s));
            if (!sweepLayers(p, 1, layers)) layers.push([lit, 1]);
            L = lampAt(p); L.a = 1; L.cone = 1;
        } else if (t < T_DOWN + T_SWEEP + T_RISE) {
            var r = smooth(0, T_RISE, t - T_DOWN - T_SWEEP);
            sweepLayers(N - 1, 1 - r, layers);
            layers.push([lit, r]);
            var e2 = lampAt(N - 1); e2.a = 1; e2.cone = 1;
            L = mixLamp(e2, parkedLamp(PARK.a), smooth(0, 1, r));
        } else {
            draw([[lit, 1]], parkedLamp(PARK.a));
            return true;
        }
        draw(layers, L);
        return false;
    }

    // ---------------------------------------------------------------- interactive states
    function stateFrame() {
        var layers = [[lit, 1 - m]];
        if (m > 0.002 && !sweepLayers(pos, m, layers)) layers = [[lit, 1]];
        var tgt = lampAt(pos); tgt.a = 1; tgt.cone = 1;
        draw(layers, mixLamp(parkedLamp(PARK.a), tgt, clamp(m, 0, 1)));
    }

    function tick(now) {
        raf = 0;
        if (mode === "loop" && lastT && now - lastT < 30) { kick(); return; }   // ~30 fps is plenty
        var dt = clamp((now - (lastT || now)) / 1000, 0, 0.1);
        lastT = now;
        if (!geo) return;
        var busy = false;
        if (mode === "seq") {
            if (seqFrame((now - seqT0) / 1000)) {
                m = 0; mTarget = 0; pos = N - 1; posTarget = pos;
                if (HOVER) setMode("idle");
                else { setMode("loop"); loopHold = T_LOOP_HOLD; loopPhase = 1; loopDir = -1; }
                busy = !HOVER;
            } else busy = true;
        } else if (mode === "intro" || mode === "loading" || mode === "static") {
            return;                                     // the video plays / nothing to draw yet
        } else {
            if (mode === "loop") {
                if (visible) {
                    if (loopHold > 0) { loopHold -= dt; mTarget = 0; }
                    else {
                        mTarget = 1;
                        loopPhase += loopDir * dt / T_LOOP;
                        if (loopPhase >= 1) { loopPhase = 1; loopDir = -1; }
                        if (loopPhase <= 0) { loopPhase = 0; loopDir = 1; }
                        posTarget = (N - 1) * (0.5 - 0.5 * Math.cos(Math.PI * loopPhase));
                    }
                    busy = true;
                }
            }
            var tauM = mTarget > m ? 0.12 : 0.24;
            m += (mTarget - m) * (1 - Math.exp(-dt / tauM));
            if (m < 0.03 && mTarget > 0.5) pos = posTarget;   // entering from the lit state: jump
            pos += (posTarget - pos) * (1 - Math.exp(-dt / (mode === "loop" ? 0.02 : 0.06)));
            if (Math.abs(mTarget - m) < 0.002) m = mTarget;
            if (Math.abs(posTarget - pos) < 0.002) pos = posTarget;
            stateFrame();
            busy = busy || m !== mTarget || pos !== posTarget;
        }
        if (busy) kick();
    }
    function kick() { if (!raf) raf = requestAnimationFrame(tick); }

    // ---------------------------------------------------------------- pointer
    function posFromEvent(ev) {
        // the pointer across the title (the stage) moves the lamp over its whole path, which
        // reaches a little past the R and the h; in the middle the lamp is under the pointer
        var b = root.getBoundingClientRect(), r = geo.r;
        var t = clamp((ev.clientX - b.left) / b.width, 0, 1);
        return posFromU(lerp(lerp(r[0], lampU[0], 1 - t), lerp(r[0] + r[2], lampU[N - 1], t), t));
    }
    function interactive() { return mode === "idle" || mode === "hover" || mode === "loop" || mode === "drag"; }
    root.addEventListener("pointermove", function (ev) {
        if (!geo || !interactive() || !N) return;
        if (ev.pointerType === "mouse" || ev.pointerType === "pen") {
            if (mode === "loop") return;
            setMode("hover");
            mTarget = 1;
            posTarget = posFromEvent(ev);
            kick();
        } else if (mode === "drag") {
            posTarget = posFromEvent(ev);
            kick();
        }
    });
    root.addEventListener("pointerleave", function (ev) {
        if (mode === "hover") { setMode("idle"); mTarget = 0; kick(); }
    });
    root.addEventListener("pointerdown", function (ev) {
        if (!geo || !interactive() || !N || ev.pointerType === "mouse") return;
        setMode("drag");
        mTarget = 1; loopHold = 0;
        posTarget = posFromEvent(ev);
        kick();
    });
    function endDrag() {
        if (mode !== "drag") return;
        if (HOVER) { setMode("idle"); mTarget = 0; }
        else {
            // continue the loop from here, in the direction away from the nearer end
            var p = clamp(posTarget / (N - 1), 0, 1);
            loopPhase = Math.acos(clamp(1 - 2 * p, -1, 1)) / Math.PI;
            loopDir = p > 0.5 ? -1 : 1;
            setMode("loop");
        }
        kick();
    }
    root.addEventListener("pointerup", endDrag);
    root.addEventListener("pointercancel", endDrag);

    if (window.IntersectionObserver) {
        new IntersectionObserver(function (es) {
            visible = es[0].isIntersecting;
            if (visible) kick();
        }).observe(root);
    }
    if (window.ResizeObserver) new ResizeObserver(layout).observe(root);
    else window.addEventListener("resize", layout);

    // ---------------------------------------------------------------- start
    function startSeq(from) {
        seqFrom = from;
        seqT0 = performance.now();
        setMode("seq");
        kick();
    }
    function showLit(pLit) {
        // fallback when the intro assets are late: the page's own <img> shows the lit word now;
        // the canvas takes over (same pixels) once the renders are in, for the interaction
        if (mode !== "loading") return;
        setMode("static");
        if (video) { video.remove(); video = null; }
        root.classList.remove("ta-running");
        root.classList.add("ta-static");
        pLit.then(function () {
            draw([[lit, 1]], null);
            root.classList.remove("ta-static");
            root.classList.add("ta-running");
            m = 0; mTarget = 0; pos = N - 1; posTarget = pos;
            if (HOVER) setMode("idle");
            else { setMode("loop"); loopHold = T_LOOP_HOLD; loopPhase = 1; loopDir = -1; }
            kick();
        });
    }

    var isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
        (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    var ua = navigator.userAgent;
    var safari = /Safari\//.test(ua) && !/(Chrome|Chromium|Edg|OPR|Firefox)\//.test(ua);
    var wantVideo = !isIOS && !safari && !!document.createElement("video").canPlayType &&
        document.createElement("video").canPlayType('video/webm; codecs="vp9"') !== "";

    function probeAlpha(v) {
        // a transparent corner must stay transparent (Safari-like engines drop VP9 alpha)
        try {
            var c = document.createElement("canvas");
            c.width = 16; c.height = 8;
            var x = c.getContext("2d");
            x.drawImage(v, 0, 0, 16, 8);
            return x.getImageData(0, 0, 1, 1).data[3] < 16;
        } catch (e) { return false; }
    }

    fetch(url("title.json")).then(function (r) { return r.json(); }).then(function (j) {
        meta = j;
        N = j.sweep.length;
        // retina: the 2x set (title.json "hi") when the screen has more device px than the 1x stills
        var HI = !!(j.hi && (window.devicePixelRatio || 1) > (j.hi.min_dpr || 1.3));
        var litName = HI ? j.hi.lit : "lit.webp", asmName = HI ? j.hi.asm : "assembled.webp";
        var sweepDir = HI ? j.hi.sweep : "sweep/";
        if (j.park) PARK = j.park;
        root.setAttribute("data-res", HI ? "2x" : "1x");
        j.sweep.forEach(function (f, i) {
            lampU[i] = f.lamp_ref[0]; lampV[i] = f.lamp_ref[1];
            aimU[i] = f.aim[0]; aimV[i] = f.aim[1]; v1[i] = f.v1_frame;
        });
        vRefEnd = Math.max.apply(null, lampV);
        vRefTop = Math.min.apply(null, lampV);
        frames = new Array(N);
        layout();

        var pLit = loadImage(url(litName)).then(function (im) { lit = im; });
        var pAsm = loadImage(url(asmName)).then(function (im) { asm = im; });
        // the sweep, coarse to fine; the start waits for every 4th frame
        var order = [], seen = {};
        [8, 4, 2, 1].forEach(function (st) {
            for (var i = 0; i < N; i += st) if (!seen[i]) { seen[i] = 1; order.push(i); }
            if (!seen[N - 1]) { seen[N - 1] = 1; order.push(N - 1); }
        });
        var coarse = order.filter(function (i) { return i % 4 === 0 || i === N - 1; });
        var pending = {}, next = 0;
        function pump() {
            while (next < order.length && Object.keys(pending).length < 6) {
                (function (i) {
                    pending[i] = loadImage(url(sweepDir + "s" + (i < 10 ? "0" : "") + i + ".webp"))
                        .then(function (im) { frames[i] = im; }, function () {})
                        .then(function () { delete pending[i]; pump(); });
                })(order[next++]);
            }
        }
        pump();
        var pCoarse = new Promise(function (res) {
            (function chk() {
                if (coarse.every(function (i) { return frames[i]; })) res();
                else setTimeout(chk, 40);
            })();
        });

        var pVideo = null;
        if (wantVideo) {
            video = document.createElement("video");
            video.muted = true;
            video.defaultMuted = true;
            video.playsInline = true;
            video.setAttribute("playsinline", "");
            video.setAttribute("muted", "");
            video.setAttribute("aria-hidden", "true");
            video.preload = "auto";
            video.style.opacity = "0";
            video.src = url("intro.webm");
            root.appendChild(video);
            layout();
            pVideo = new Promise(function (res, rej) {
                video.addEventListener("canplaythrough", function () {
                    probeAlpha(video) ? res() : rej(new Error("no alpha"));
                }, { once: true });
                video.addEventListener("error", function () { rej(new Error("video error")); }, { once: true });
            });
        }
        if (pVideo) pVideo.catch(function () {});

        var deadline = new Promise(function (res) {
            setTimeout(res, Math.max(0, T_START_MAX * 1000 - (performance.now() - t0)));
        }).then(function () { return "late"; });

        var ready = Promise.all([pAsm, pLit, pCoarse]);
        var withVideo = ready.then(function () {
            return pVideo || Promise.reject(new Error("no video"));
        }).then(function () { return "video"; },
            function () { return "fade"; });
        Promise.race([withVideo, deadline]).then(function (how) {
            if (how === "late") { showLit(pLit); return; }
            if (how === "fade") {
                if (video) { video.remove(); video = null; }
                startSeq("fade");
                return;
            }
            setMode("intro");
            kick();
            var started = false;
            video.addEventListener("ended", function () {
                startSeq("asm");
                seqFrame(0);                            // canvas = the video's last frame ...
                requestAnimationFrame(function () {     // ... then the video goes
                    if (video) { video.remove(); video = null; }
                });
            }, { once: true });
            video.style.opacity = "1";
            var pp = video.play();
            if (pp && pp.catch) pp.catch(function () {
                if (video) { video.remove(); video = null; }
                startSeq("fade");
            });
        });
    }).catch(function () {
        root.classList.remove("ta-running");
        root.classList.add("ta-static");
    });

    window.__titleAnim = {
        state: function () { return { mode: mode, m: m, pos: pos, N: N, loaded: frames.filter(Boolean).length,
            res: root.getAttribute("data-res"), set: root.getAttribute("data-set"),
            video: !!video, draws: dbg.frames, avgDrawMs: dbg.frames ? dbg.drawMs / dbg.frames : 0,
            maxDrawMs: dbg.maxDrawMs }; },
        resetStats: function () { dbg.frames = 0; dbg.drawMs = 0; dbg.maxDrawMs = 0; }
    };
})();
