/* "Apply a trained material", variant A: decode in hand.
 *
 * Left (above on phones): the trained latent textures, alive. Grab one and the card under
 * the cursor decodes, latent -> material, with the frozen-decoder glyph flashing; what you
 * carry is the material. Drop it on a sofa (the sofa and its pillows), on a sofa's pillows,
 * on the floor cushions, the rug or the left curtain: the path-traced material wipes over the
 * object from the drop point. Tap / click mode: tap a latent (it decodes in place), then tap objects.
 * Uses js/room_core.js; assets in media/room/.
 */
(function () {
    "use strict";
    const C = window.RoomCore;
    const panel = document.getElementById("room-panel");
    if (!panel || !C) return;

    C.whenNear(panel, () => start().catch((e) => {
        console.error(e);
        panel.classList.add("room-error");
        const l = panel.querySelector(".room-loading");
        if (l) l.textContent = "The interactive room could not start (" + e.message + ").";
    }));

    async function start() {
        const hint = document.getElementById("room-hint");
        const tilesEl = panel.querySelector(".room-side");
        const resetBtn = panel.querySelector(".room-reset");
        const D = 0.5;                    // decode duration (s)
        let core = null;
        const tiles = [];
        let drag = null;                  // {mi, id, x0, y0, moved}
        let selected = -1;                // tap mode: material in hand
        const ghost = { el: null, cv: null, ctx: null, mi: -1, mode: "off", t0: 0, x: 0, y: 0, over: false,
            fromX: 0, fromY: 0, s: 1, o: 0 };

        core = await C.create({
            root: panel, assets: panel.dataset.assets, v: panel.dataset.v,
            onFrame: (t) => frame(t),
        });
        const MATS = core.MATS;

        // ------------------------------------------------------------------ tiles
        MATS.forEach((m, mi) => {
            const el = document.createElement("div");
            el.className = "room-tile";
            el.setAttribute("role", "button");
            el.setAttribute("tabindex", "0");
            el.setAttribute("aria-label", `Material ${m.id} (${m.name}): its trained latent texture. Drag it onto the room, or press Enter and then click an object.`);
            el.title = `#${m.id} ${m.name}`;
            const gl = document.createElement("canvas");
            gl.width = gl.height = 208;
            const dec = document.createElement("canvas");
            dec.width = dec.height = 208;
            const tagEl = document.createElement("span");
            tagEl.className = "tile-tag";
            tagEl.textContent = "#" + m.id;
            const st = document.createElement("span");
            st.className = "tile-state";
            st.textContent = "latent";
            el.append(gl, dec, tagEl, st);
            tilesEl.appendChild(el);
            const live = C.makeTile(gl, m.tile, core.imgs, m.id, mi);
            tiles.push({ el, mi, m, live, dec, dctx: dec.getContext("2d"), st, u: 0, from: 0, to: 0, t0: -1e9 });
        });
        function decodeTile(tl, to) {
            const t = core.now();
            tl.from = tileU(tl, t);
            tl.to = to;
            tl.t0 = t;
        }
        function tileU(tl, t) {
            const x = C.clamp01((t - tl.t0) / (C.REDUCED ? 0.01 : D));
            return tl.from + (tl.to - tl.from) * x;
        }

        // ------------------------------------------------------------------ ghost (the card in hand)
        ghost.el = document.createElement("div");
        ghost.el.className = "room-ghost";
        ghost.el.setAttribute("aria-hidden", "true");
        ghost.cv = document.createElement("canvas");
        ghost.el.appendChild(ghost.cv);
        document.body.appendChild(ghost.el);
        ghost.ctx = ghost.cv.getContext("2d");
        function ghostSize() { return tiles[0].el.getBoundingClientRect().width || 104; }
        function ghostSet(mode) {
            ghost.mode = mode;
            ghost.t0 = core.now();
            ghost.s0 = ghost.s;
            ghost.o0 = ghost.o;
            ghost.x0 = ghost.x;
            ghost.y0 = ghost.y;
        }

        function frame(t) {
            // tiles: live latents + in-place decode overlay
            for (const tl of tiles) {
                tl.live.draw(t);
                const u = tileU(tl, t);
                if (u > 0.001 || tl.u > 0.001) {
                    C.drawDecode(tl.dctx, tl.dec.width, tl.dec.height, null, core.imgs["swatch_" + tl.m.id],
                        C.tileView(tl.mi, t), u, { glyph: tl.to > tl.from });
                }
                tl.u = u;
                const s = u > 0.5 ? "decoded" : "latent";
                if (tl.st.textContent !== s) tl.st.textContent = s;
            }
            demoFrame(t);
            drawGhost(t);
        }

        function drawGhost(t) {
            if (ghost.mode === "off" && ghost.o <= 0.001) { ghost.el.style.opacity = "0"; return; }
            const size = ghostSize();
            const dpr = Math.min(2, window.devicePixelRatio || 1);
            const px = Math.round(size * dpr);
            if (ghost.cv.width !== px) { ghost.cv.width = ghost.cv.height = px; }
            ghost.el.style.setProperty("--ghost", size + "px");
            const lt = t - ghost.t0;
            let s = 1, o = 1, x = ghost.x, y = ghost.y, u = 1;
            const tl = tiles[ghost.mi];
            if (ghost.mode === "carry") {
                const k = C.easeOut3(lt / 0.16);
                s = (0.85 + 0.15 * k) * (ghost.over ? 0.8 : 1);
                o = k;
                u = C.clamp01((lt - 0.06) / D);
                ghost.u = u;
            } else if (ghost.mode === "drop") {
                const k = C.clamp01(lt / 0.28);
                s = ghost.s0 * (1 - 0.85 * k * k);
                o = ghost.o0 * (1 - k);
                if (k >= 1) ghost.mode = "off";
            } else if (ghost.mode === "return") {
                const k = C.easeInOut(lt / 0.36);
                const r = tl.el.getBoundingClientRect();
                x = ghost.x0 + (r.left + r.width / 2 - ghost.x0) * k;
                y = ghost.y0 + (r.top + r.height / 2 - ghost.y0) * k;
                s = ghost.s0 + (1 - ghost.s0) * k;
                o = ghost.o0 * (1 - C.smooth((lt - 0.2) / 0.16));
                u = (ghost.u || 1) * (1 - k);
                if (lt >= 0.36) { ghost.mode = "off"; o = 0; tl.el.classList.remove("lifted"); }
            } else { o = 0; }
            ghost.s = s; ghost.o = o;
            if (ghost.mode !== "drop" && tl) {
                C.drawDecode(ghost.ctx, px, px, tl.live.canvas, core.imgs["swatch_" + tl.m.id],
                    C.tileView(tl.mi, t), u);
            }
            ghost.el.classList.toggle("over", ghost.over && ghost.mode === "carry");
            ghost.el.style.opacity = o.toFixed(3);
            ghost.el.style.transform = `translate(${(x - size / 2).toFixed(1)}px, ${(y - size / 2).toFixed(1)}px) scale(${s.toFixed(3)})`;
        }

        // ------------------------------------------------------------------ teaching loop
        // While the panel is in view and untouched, a ghost pointer dresses the room piece by
        // piece: it moves to a tile, picks it up (the card decodes in its hand), carries it to
        // a piece (which highlights), drops it, and the material wipes in; after a short hold it
        // takes the next tile to another piece (a sofa, a pillow, a cushion, the rug, the
        // curtain). After 4-5 drops the room cross-fades back to grey, the hint pulses under the tiles, and the next
        // cycle dresses it differently. Phones: the same with tap-tap (tap a tile: it decodes in
        // place; the room pans to the piece; tap the piece). Every move starts where the last
        // ended, and a drop only starts once the previous wipe has finished. The visitor's first
        // press, touch, key or real mouse movement in the panel stops it for good (it never
        // fights the visitor's pointer). If the method video's hand-off (js/handoff.js) is about
        // to run or running, the loop waits for the tiles to land. Reduced motion: no loop, the
        // hint only.
        const COARSE = !!(window.matchMedia && matchMedia("(hover: none), (pointer: coarse)").matches);
        const midOf = (id) => MATS.findIndex((m) => m.id === id);
        const kOf = (id) => core.TGT.findIndex((t) => t.id === id);
        // cycles of (material, piece); every cycle mixes a sofa, a pillow, a cushion and the rug,
        // and every other cycle ends on the curtain at the left edge
        const CYCLES = [
            [[453, "sofaL"], [367, "pillow55"], [311, "rug"], [8, "cush56"]],
            [[314, "sofaR"], [9, "pillow3"], [145, "cush57"], [50, "rug"], [370, "curtainL"]],
            [[370, "sofaL"], [226, "cush58"], [453, "pillow5"], [311, "cush2"]],
            [[145, "sofaR"], [367, "cush57"], [9, "rug"], [314, "pillow59"], [226, "curtainL"]],
        ].map((c) => c.map(([m, k]) => [midOf(m), kOf(k)]).filter(([mi, k]) => mi >= 0 && k >= 0));
        const demo = { stopped: false, inView: false, cycle: -1, i: 0, step: null, pause: null,
            p: null, o: 0, s: 1, press: 0, hintT0: -1e9, hintMi: 0, layers: false, lastMouse: null };
        const hand = document.createElement("div");
        hand.className = "room-demo-hand" + (COARSE ? " finger" : "");
        hand.setAttribute("aria-hidden", "true");
        hand.innerHTML = COARSE ? "" : '<svg viewBox="0 0 24 24" width="30" height="30"><path d="M3.2 2.2 L3.2 19.6 L7.9 15.4 L11 22.2 L14 20.9 L11 14.2 L17.4 14.2 Z" ' +
            'fill="rgba(255,255,255,0.94)" stroke="#3367d6" stroke-width="1.5" stroke-linejoin="round"/></svg><span class="press"></span>';
        const tapRing = document.createElement("div");
        tapRing.className = "room-demo-tap";
        tapRing.setAttribute("aria-hidden", "true");
        document.body.append(hand, tapRing);
        const tap = { t0: -1e9, x: 0, y: 0 };
        const callout = document.createElement("div");
        callout.className = "room-callout";
        callout.setAttribute("role", "note");
        callout.innerHTML = '<span class="co-fine">Drag a material onto any sofa, pillow, cushion, the rug or the curtain</span>' +
            '<span class="co-coarse">Tap a material, then a sofa, pillow, cushion, the rug or the curtain</span>';
        panel.appendChild(callout);
        // the call to action: stays on the panel (pulsing softly) until the visitor's first
        // interaction, brightens during each cycle's "your turn" beat, then fades for good
        const cta = document.createElement("div");
        cta.className = "room-cta";
        cta.setAttribute("role", "note");
        cta.innerHTML = '<svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"><path d="M9 11.2V5.4a1.6 1.6 0 0 1 3.2 0v5.2" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>' +
            '<path d="M12.2 10.4V9.2a1.6 1.6 0 0 1 3.2 0v1.6M15.4 10.8a1.6 1.6 0 0 1 3.2 0v3.6c0 3.4-2.3 6.2-5.6 6.2h-1.2c-2 0-3.4-.9-4.6-2.6l-2.6-3.9a1.5 1.5 0 0 1 2.4-1.8L9 13.6" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>' +
            '<path d="M3.5 5.5l-1.5 1.5 1.5 1.5M2 7h4" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
            '<span class="cta-fine">Try it yourself &mdash; drag a material onto the furniture</span>' +
            '<span class="cta-coarse">Try it yourself &mdash; tap a material, then tap a piece</span>';
        panel.appendChild(cta);
        cta.hidden = true;                   // author: the pill crowded the hint above -- the hint line alone
        // a tile hops 4 px once in a while (between demo drops, and in the "your turn" beat)
        const hop = { mi: -1, t0: -1e9, n: 1 };
        function hopFrame(t) {
            if (hop.mi < 0) return;
            const u = (t - hop.t0) / 0.42;
            const el = tiles[hop.mi].el;
            if (u >= hop.n) { el.style.transform = ""; hop.mi = -1; return; }
            const k = u - Math.floor(u);
            el.style.transform = `translateY(${(-4 * Math.sin(Math.PI * k)).toFixed(2)}px)`;
        }

        if ("IntersectionObserver" in window) {
            // "in view" = a third of the panel is visible, or the panel fills more than half of
            // a short window (laptops with little vertical room never reach 50 % of the panel)
            new IntersectionObserver((es) => {
                for (const e of es) {
                    const vh = window.innerHeight || 900;
                    demo.inView = e.intersectionRatio >= 0.35 || e.intersectionRect.height >= 0.55 * vh;
                }
                if (!demo.inView && (demo.step || demo.pause)) suspend();
            }, { threshold: [0, 0.15, 0.25, 0.35, 0.5, 0.75, 1] }).observe(panel);
        } else demo.inView = true;

        const tileC = (mi) => { const r = tiles[mi].el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
        // the method video's hand-off (js/handoff.js) may be the panel's first entrance: the
        // loop waits while it runs or is armed, and before its own first start it also lets a
        // hand-off that is due within ~3 s happen first
        const mvideo = document.getElementById("method-video");
        const handoffBusy = () => {
            return false;                    // author: the hand-off overlays the loop, no yielding
            // eslint-disable-next-line no-unreachable
            const h = window.methodHandoff;
            const st = h && h.state ? h.state() : "done";
            if (st === "running") return true;
            if (st === "armed") return !!mvideo && !mvideo.paused;   // a paused video never "ends"
            if (st !== "idle" || !mvideo || mvideo.paused || !h.mayRun || !h.mayRun()) return false;
            const left = mvideo.duration - mvideo.currentTime;
            return isFinite(left) && left > 0 && left < 3;
        };

        /** a pointer move from wherever the pointer is: eased, gently curved; the end point is
         * re-read every frame (the room may pan underneath), so it never jumps */
        function mover(t, to, speed) {
            const from = demo.p.slice();
            const d0 = Math.hypot(to()[0] - from[0], to()[1] - from[1]);
            const dur = Math.min(1.5, Math.max(0.5, 0.42 + d0 / (speed || 1250)));
            return { t0: t, dur, from, to, bend: Math.min(70, d0 * 0.18) * (from[0] < to()[0] ? -1 : 1) };
        }
        function along(m, t) {
            const k = C.easeInOut((t - m.t0) / m.dur), e = m.to();
            const x = m.from[0] + (e[0] - m.from[0]) * k, y = m.from[1] + (e[1] - m.from[1]) * k;
            const dx = e[0] - m.from[0], dy = e[1] - m.from[1], L = Math.hypot(dx, dy) || 1;
            const b = Math.sin(Math.PI * k) * m.bend;          // a soft arc, off the straight line
            return [x - (dy / L) * b, y + (dx / L) * b];
        }
        const done = (m, t) => t - m.t0 >= m.dur;

        function startStep(t) {
            const c = CYCLES[demo.cycle % CYCLES.length];
            const [mi, k] = c[demo.i];
            const next = c[demo.i + 1] || CYCLES[(demo.cycle + 1) % CYCLES.length][0];
            core.need(mi);                                     // its renders, while it is picked up
            if (next) core.need(next[0]);                      // and the next one's, ahead of time
            demo.hintMi = mi;
            demo.step = { mi, k, phase: "toTile", m: mover(t, () => tileC(mi)), t0: t };
        }
        function targetPt(k) { const d = core.TGT[k].drop; return core.clientOf(d[0], d[1]); }

        function stepDesk(t) {
            const S = demo.step;
            if (S.phase === "toTile") {
                demo.p = along(S.m, t);
                if (done(S.m, t)) { S.phase = "press"; S.t1 = t; }
            } else if (S.phase === "press") {
                demo.press = C.smooth((t - S.t1) / 0.16);
                demo.s = 1 - 0.12 * demo.press;
                if (t - S.t1 >= 0.16) {
                    S.picked = true;
                    tiles[S.mi].el.classList.add("lifted");
                    ghost.mi = S.mi;
                    ghost.x = demo.p[0]; ghost.y = demo.p[1];
                    ghost.over = false;
                    ghostSet("carry");
                    core.stage.classList.add("carrying");
                    core.stage.style.setProperty("--ghost-half", ghostSize() / 2 + "px");
                    S.phase = "carry";
                    S.m = mover(t, () => targetPt(S.k), 1100);
                }
            } else if (S.phase === "carry" || S.phase === "hover") {
                if (S.phase === "carry") {
                    demo.p = along(S.m, t);
                    if (done(S.m, t)) { S.phase = "hover"; S.t1 = t; }
                } else demo.p = targetPt(S.k);
                ghost.x = demo.p[0]; ghost.y = demo.p[1];
                const hit = core.targetAt(demo.p[0], demo.p[1], false);
                core.setHover(hit ? hit.k : -1, hit ? hit.px : 0, hit ? hit.py : 0);
                ghost.over = !!hit;
                if (S.phase === "hover" && t - S.t1 >= 0.38) {
                    const d = core.TGT[S.k].drop;
                    core.apply(S.k, S.mi, d[0], d[1]);         // the drop
                    demo.layers = true;
                    resetBtn.disabled = false;
                    ghostSet("drop");
                    tiles[S.mi].el.classList.remove("lifted");
                    core.stage.classList.remove("carrying");
                    core.setHover(-1);
                    ghost.over = false;
                    S.phase = "release";
                    S.t1 = t;
                    const p0 = demo.p.slice();
                    S.m = { t0: t, dur: 0.55, from: p0, to: () => [p0[0] + 34, p0[1] + 26], bend: 0 };
                }
            } else if (S.phase === "release") {
                demo.press = 1 - C.smooth((t - S.t1) / 0.25);
                demo.s = 1 - 0.12 * demo.press;
                demo.p = along(S.m, t);
                if (done(S.m, t) && !core.busy()) { S.phase = "hold"; S.t1 = t; }
            } else if (S.phase === "hold") {
                if (t - S.t1 >= 1.0) demo.step = null;
            }
        }

        function stepPhone(t) {
            const S = demo.step;
            const tapNow = (pt) => { tap.t0 = t; tap.x = pt[0]; tap.y = pt[1]; };
            demo.s = 1 - 0.14 * Math.max(0, 1 - Math.abs(t - tap.t0) / 0.12);
            if (S.phase === "toTile") {
                demo.p = along(S.m, t);
                if (done(S.m, t)) {
                    tapNow(demo.p);
                    select(S.mi);                              // the tile decodes in place
                    S.phase = "decoded"; S.t1 = t;
                    core.panTo(S.k);                           // and the room glides to the piece
                }
            } else if (S.phase === "decoded") {
                demo.p = tileC(S.mi);
                if (t - S.t1 >= 0.6 && !core.busy()) { S.phase = "toPiece"; S.m = mover(t, () => targetPt(S.k), 900); }
            } else if (S.phase === "toPiece") {
                demo.p = along(S.m, t);
                if (done(S.m, t)) {
                    tapNow(demo.p);
                    const d = core.TGT[S.k].drop;
                    core.apply(S.k, S.mi, d[0], d[1]);
                    demo.layers = true;
                    resetBtn.disabled = false;
                    S.phase = "release"; S.t1 = t;
                    const p0 = demo.p.slice();
                    S.m = { t0: t, dur: 0.5, from: p0, to: () => [p0[0] + 18, p0[1] + 30], bend: 0 };
                }
            } else if (S.phase === "release") {
                demo.p = along(S.m, t);
                if (done(S.m, t) && !core.busy()) { select(-1); S.phase = "hold"; S.t1 = t; }
            } else if (S.phase === "hold") {
                if (t - S.t1 >= 0.9) demo.step = null;
            }
        }

        /** leave the room as the visitor will find it: whatever is in hand goes back, the room
         * cross-fades to grey, the pointer fades */
        function suspend(fadeRoom) {
            const S = demo.step;
            demo.step = null;
            demo.pause = null;
            if (S && S.picked && S.phase !== "release" && S.phase !== "hold") {
                ghostSet("return");
                core.stage.classList.remove("carrying");
                core.setHover(-1);
                ghost.over = false;
            }
            if (selected >= 0 && COARSE) select(-1);
            if (demo.layers && fadeRoom !== false) { core.reset(0.8); demo.layers = false; resetBtn.disabled = true; }
            if (demo.cycle >= 0) demo.nextCycle = demo.cycle + 1;     // back in view: a fresh cycle
            demo.cycle = -1;
        }
        function stopDemo() {
            if (demo.stopped) return;
            suspend();
            demo.stopped = true;
            demo.hintT0 = -1e9;
            cta.classList.remove("turn");
            cta.classList.add("gone");
            if (hop.mi >= 0) { tiles[hop.mi].el.style.transform = ""; hop.mi = -1; }
        }

        function demoFrame(t) {
            const can = !demo.stopped && !demo.mouseIn && !C.REDUCED && demo.inView && !drag && !handoffBusy();
            // blocked in the middle of a step (hand-off due, visitor dragging, ...): the card goes
            // back to its tile and the room to grey; a fresh cycle starts once the loop may run again
            if (!can && (demo.step || demo.pause)) suspend();
            if (can) {
                if (demo.cycle < 0) {                          // a new run: the pointer appears by the tiles
                    demo.started = true;
                    demo.cycle = demo.nextCycle || 0;
                    demo.i = 0;
                    const r = tilesEl.getBoundingClientRect();
                    demo.p = demo.p && demo.o > 0.05 ? demo.p : [r.left + r.width * 0.62, r.bottom + 40];
                    demo.pause = { t0: t, dur: 0.5, fadeIn: true };
                }
                if (demo.pause) {
                    const P = demo.pause, u = t - P.t0;
                    if (P.fadeIn) demo.o = Math.max(demo.o, C.smooth(u / 0.4));
                    if (P.turn) {                              // "your turn": park beside the tiles
                        demo.p = along(P.m, t);
                        demo.press = 0;
                        demo.s = 1;
                        const on = u >= 0.9 && u < P.dur - 0.15;
                        cta.classList.toggle("turn", on);
                        if (u >= 1.15 && !P.hopped) { P.hopped = true; Object.assign(hop, { mi: P.mi, t0: t, n: 2 }); }
                    }
                    if (u >= P.dur && !core.busy()) { cta.classList.remove("turn"); demo.pause = null; }
                } else if (!demo.step) {
                    const c = CYCLES[demo.cycle % CYCLES.length];
                    if (demo.i < c.length) startStep(t);
                    else {                                     // end of a cycle: back to grey, your turn
                        if (demo.layers) { core.reset(1.0); demo.layers = false; resetBtn.disabled = true; }
                        demo.cycle++;
                        demo.i = 0;
                        const mi = (demo.cycle * 3 + 4) % tiles.length;
                        const park = () => {                   // just below the tile, pointing up at it
                            const r = tiles[mi].el.getBoundingClientRect();
                            return COARSE ? [r.left + r.width / 2, r.bottom + 30] : [r.left + r.width * 0.55, r.bottom + 10];
                        };
                        demo.pause = { t0: t, dur: 2.5, turn: true, mi, m: mover(t, park, 1000) };
                    }
                }
                if (demo.step) {
                    (COARSE ? stepPhone : stepDesk)(t);
                    if (!demo.step) {
                        demo.i++;
                        if (demo.i === 2)                      // between drops: a tile hops once
                            Object.assign(hop, { mi: (demo.cycle * 5 + 7) % tiles.length, t0: t + 0.3, n: 1 });
                    }
                }
            }
            // ghost pointer + its tap ring
            const active = can && demo.cycle >= 0;
            if (!active) demo.o = Math.max(0, demo.o - 0.08);
            hand.style.opacity = demo.o.toFixed(3);
            if (demo.o > 0.001 && demo.p) {
                hand.style.transform = `translate(${demo.p[0].toFixed(1)}px, ${demo.p[1].toFixed(1)}px) scale(${demo.s.toFixed(3)})`;
                hand.style.setProperty("--press", (demo.press || 0).toFixed(3));
            }
            const tu = (t - tap.t0) / 0.55;
            if (tu >= 0 && tu <= 1) {
                tapRing.style.opacity = (1 - tu).toFixed(3);
                tapRing.style.transform = `translate(${tap.x}px, ${tap.y}px) scale(${(0.4 + 0.9 * tu).toFixed(3)})`;
            } else if (tapRing.style.opacity !== "0") tapRing.style.opacity = "0";
            hopFrame(t);
        }

        // the hint under the tiles: pulses for ~3.5 s at the end of each loop cycle; static (until the
        // visitor interacts) under reduced motion
        function drawCallout(t) {
            const u = t - demo.hintT0;
            let o = 0, ring = 0, s = 1;
            if (C.REDUCED) o = !demo.stopped && demo.inView ? 1 : 0;
            else if (u >= 0 && u < 3.6) {
                o = Math.min(C.smooth(u / 0.3), 1 - C.smooth((u - 3.1) / 0.5));
                if (u < 2.3) { const k = (u % 1.13) / 1.13; ring = 1 - k; s = 1 + 0.035 * Math.sin(Math.PI * k); }
            }
            callout.style.opacity = o.toFixed(3);
            callout.classList.toggle("show", o > 0.01);
            if (o <= 0.001) return;
            const pr = panel.getBoundingClientRect(), sr = tilesEl.getBoundingClientRect();
            const tr = tiles[demo.hintMi].el.getBoundingClientRect();
            const wide = sr.height > sr.width;              // tile column (desktop) or tile row (phones)
            callout.classList.toggle("below", !wide);
            if (wide) {
                callout.style.left = (sr.right - pr.left + 14) + "px";
                callout.style.top = (tr.top + tr.height / 2 - pr.top) + "px";
            } else {
                callout.style.left = (pr.width / 2) + "px";
                callout.style.top = (sr.bottom - pr.top + 12) + "px";
            }
            callout.style.setProperty("--s", s.toFixed(3));
            callout.style.boxShadow = `0 4px 16px rgba(20, 30, 50, 0.16), 0 0 0 ${(10 * (1 - ring)).toFixed(1)}px rgba(51, 103, 214, ${(0.35 * ring).toFixed(3)})`;
        }

        // the visitor takes over: a press, touch or key in the panel stops the loop for good.
        // A mouse moving over the panel only pauses it (the pointer steps aside, the room goes
        // back to grey, the "try it" pill stays); the loop resumes once the mouse has left,
        // so a stray pass of the mouse does not take the lesson away (scroll-induced move
        // events under a resting mouse do not count)
        panel.addEventListener("pointerdown", stopDemo, true);
        panel.addEventListener("keydown", stopDemo, true);
        panel.addEventListener("pointermove", (e) => {
            if (e.pointerType !== "mouse") return;
            const p = [e.clientX, e.clientY], q = demo.lastMouse;
            demo.lastMouse = p;
            if (!q || Math.hypot(p[0] - q[0], p[1] - q[1]) < 3) return;
            if (!demo.mouseIn) { demo.mouseIn = true; if (demo.step || demo.pause) suspend(); }
        });
        panel.addEventListener("pointerleave", () => { demo.lastMouse = null; demo.mouseIn = false; });

        // ------------------------------------------------------------------ selection (tap mode)
        function select(mi) {
            if (selected === mi) mi = -1;
            if (mi >= 0) core.need(mi);
            if (selected >= 0) decodeTile(tiles[selected], 0);
            selected = mi;
            if (mi >= 0) decodeTile(tiles[mi], 1);
            tiles.forEach((tl) => tl.el.classList.toggle("selected", tl.mi === mi));
            panel.classList.toggle("has-selection", mi >= 0);
            core.stage.classList.toggle("picking", mi >= 0);
            core.setPick(mi >= 0);
            if (hint) {
                hint.classList.toggle("active", mi >= 0);
                const fine = hint.querySelector(".hint-fine"), coarse = hint.querySelector(".hint-coarse");
                if (!hint._orig) hint._orig = [fine.textContent, coarse.textContent];
                const m = mi >= 0 ? MATS[mi] : null;
                fine.textContent = m ? `#${m.id} decoded. Click any sofa, pillow, cushion, the rug or the curtain.` : hint._orig[0];
                coarse.textContent = m ? `#${m.id} decoded. Tap any sofa, pillow, cushion, the rug or the curtain.` : hint._orig[1];
            }
            if (mi < 0) core.setHover(-1);
        }

        function applyAt(k, mi, px, py) {
            core.apply(k, mi, px, py);
            resetBtn.disabled = false;
        }

        // ------------------------------------------------------------------ drag
        // A drag that leaves the panel, loses its pointer, or is interrupted (scroll, tab switch,
        // window blur) is cancelled on the spot: the card vanishes and its tile is released, so
        // nothing is ever left floating over the page.
        let dragWatch = null;
        function outsidePanel(x, y, pad = 28) {
            const r = panel.getBoundingClientRect();
            return x < r.left - pad || x > r.right + pad || y < r.top - pad || y > r.bottom + pad;
        }
        function cancelDrag() {
            if (!drag) return;
            const d = drag;
            drag = null;
            clearTimeout(dragWatch);
            const el = tiles[d.mi] && tiles[d.mi].el;
            try { if (el && el.releasePointerCapture) el.releasePointerCapture(d.id); } catch (err) { /* already released */ }
            core.setHover(-1);
            ghost.over = false;
            core.stage.classList.remove("carrying");
            tiles.forEach((t) => t.el.classList.remove("lifted"));
            ghost.mode = "off";
            ghost.o = 0;
            ghost.el.style.opacity = "0";
        }
        window.addEventListener("blur", cancelDrag);
        window.addEventListener("scroll", () => { if (drag && drag.moved) cancelDrag(); }, { passive: true });
        document.addEventListener("visibilitychange", () => { if (document.hidden) cancelDrag(); });

        tiles.forEach((tl) => {
            const el = tl.el;
            el.addEventListener("pointerdown", (e) => {
                if (e.button !== undefined && e.button > 0) return;
                e.preventDefault();
                core.need(tl.mi);                   // fetch its rendered room while it decodes in hand
                try { el.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ }
                drag = { mi: tl.mi, id: e.pointerId, x0: e.clientX, y0: e.clientY, moved: false, lx: e.clientX, ly: e.clientY };
            });
            el.addEventListener("pointermove", (e) => {
                if (!drag || e.pointerId !== drag.id) return;
                drag.lx = e.clientX; drag.ly = e.clientY;
                if (drag.moved && outsidePanel(e.clientX, e.clientY)) { cancelDrag(); return; }
                // no pointer event for a while with the pointer already off the panel: give up
                clearTimeout(dragWatch);
                dragWatch = setTimeout(() => { if (drag && drag.moved && outsidePanel(drag.lx, drag.ly, 0)) cancelDrag(); }, 2500);
                if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 6) {
                    drag.moved = true;
                    if (selected >= 0) select(-1);
                    el.classList.add("lifted");
                    ghost.mi = drag.mi;
                    ghost.x = e.clientX; ghost.y = e.clientY;
                    ghost.over = false;
                    ghostSet("carry");
                    core.stage.classList.add("carrying");
                    core.stage.style.setProperty("--ghost-half", ghostSize() / 2 + "px");
                }
                if (!drag.moved) return;
                ghost.x = e.clientX; ghost.y = e.clientY;
                const hit = core.targetAt(e.clientX, e.clientY, e.pointerType !== "mouse");
                const k = hit ? hit.k : -1;
                core.setHover(k, hit ? hit.px : 0, hit ? hit.py : 0);
                ghost.over = k >= 0;
            });
            const end = (e) => {
                if (!drag || e.pointerId !== drag.id) return;
                if (e.type !== "pointerup") { cancelDrag(); return; }     // pointercancel / lost capture
                const d = drag;
                drag = null;
                clearTimeout(dragWatch);
                if (!d.moved) { select(d.mi); return; }
                const hit = core.targetAt(e.clientX, e.clientY, e.pointerType !== "mouse");
                core.setHover(-1);
                ghost.over = false;
                core.stage.classList.remove("carrying");
                if (hit && hit.k >= 0) {
                    applyAt(hit.k, d.mi, hit.px, hit.py);
                    ghostSet("drop");
                    el.classList.remove("lifted");
                } else {
                    ghostSet("return");
                }
            };
            el.addEventListener("pointerup", end);
            el.addEventListener("pointercancel", end);
            el.addEventListener("lostpointercapture", (e) => { if (drag && drag.moved && e.pointerId === drag.id) cancelDrag(); });
            el.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(tl.mi); }
            });
        });

        // tap mode: a material is in hand, tap objects
        const room = core.canvas;
        room.addEventListener("pointermove", (e) => {
            if (drag || e.pointerType !== "mouse") return;
            const hit = core.targetAt(e.clientX, e.clientY);
            if (selected >= 0) core.setHover(hit ? hit.k : -1, hit ? hit.px : 0, hit ? hit.py : 0);
            else core.setHover(hit ? hit.k : -1, hit ? hit.px : 0, hit ? hit.py : 0, hit ? core.labelOf(hit.k) + " · drop a material here" : null, 0.45);
        });
        room.addEventListener("pointerleave", () => { if (!drag) core.setHover(-1); });
        room.addEventListener("click", (e) => {
            if (drag) return;
            const hit = core.targetAt(e.clientX, e.clientY, !!e.pointerType && e.pointerType !== "mouse");
            if (hit && selected < 0 && hint) {           // nothing in hand: point at the tiles
                hint.classList.remove("nudge");
                void hint.offsetWidth;
                hint.classList.add("nudge");
                tilesEl.classList.remove("nudge");
                void tilesEl.offsetWidth;
                tilesEl.classList.add("nudge");
            }
            if (selected < 0 || !hit) return;
            applyAt(hit.k, selected, hit.px, hit.py);
            if (e.pointerType !== "mouse" && e.pointerType !== "") core.setHover(-1);
        });
        document.addEventListener("keydown", (e) => { if (e.key === "Escape" && selected >= 0) select(-1); });
        resetBtn.addEventListener("click", () => { core.reset(); resetBtn.disabled = true; });

        // ------------------------------------------------------------------ hooks for scripted recordings
        window.roomDemo = {
            visitorBusy: () => !!drag || selected >= 0,     // the hand-off waits only for the visitor
            core, frame: () => core.frame(),
            tileCenter(mi) { const r = tiles[mi].el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; },
            targetPoint: (k) => core.targetPoint(k),
            resetCenter() { const r = resetBtn.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; },
            select,
            demo: {
                arm() {
                    suspend(false);
                    Object.assign(demo, { stopped: false, inView: true, hintT0: -1e9, nextCycle: 0 });
                    cta.classList.remove("gone", "turn");
                },
                state: () => ({ stopped: demo.stopped, cycle: demo.cycle, step: demo.i,
                    running: !!demo.step || !!demo.pause, paused: !!demo.mouseIn, phase: demo.step ? demo.step.phase : null }),
                stop: stopDemo,
            },
        };
    }
})();
