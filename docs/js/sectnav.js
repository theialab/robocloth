/* ============================================================
 *  Section navigator + "back to top" (css/sectnav.css; the markup sits right
 *  after the hero in index.html). After powerfoam.github.io.
 *
 *  - Every <a href="#id"> in #sectnav is one entry. Its label is read from the
 *    target's first <h3> (kept in sync if the heading text changes), else from a
 *    data-nav-label attribute on the target, else from the link's own text
 *    (Capture and Apply have no <h3>). An id that is missing hides its entry.
 *  - The current entry is the one whose part of the page is under the middle of
 *    the window; a nested entry (Capture in Overview, Apply in Method) wins over
 *    its section. Above the first section (hero, launch video) none is current.
 *    Over the last half window of scroll the probe line slides down to the bottom
 *    edge, so the short sections at the end become current as well.
 *  - A click scrolls smoothly (instantly with prefers-reduced-motion), puts the
 *    id in the address bar without adding history entries, and keeps the clicked
 *    entry current until the reader scrolls on. From the keyboard, focus moves to
 *    the target like a native in-page link.
 *  - The column is shown only where it fits beside the content column: with labels
 *    if the widest label fits, as dots only if they fit (labels then appear as
 *    tooltips), otherwise not at all; never below 1100 px.
 *  - The back-to-top button appears after about one window of scrolling.
 * ============================================================ */
(() => {
    "use strict";

    const nav = document.getElementById("sectnav");
    const topBtn = document.getElementById("sectnav-top");
    const root = document.documentElement;
    const MIN_WIDTH = 1100;   // px: never a column below this (phones, tablets)
    const CLEARANCE = 12;     // px kept free between the column (labels included) and the content
    const reduceMotion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    const behavior = () => (reduceMotion && reduceMotion.matches ? "auto" : "smooth");
    const setUrl = (url) => { try { history.replaceState(history.state, "", url); } catch (e) { /* ignore */ } };
    const focusLanding = (el) => {        // like a native in-page link; undone on blur
        const added = el.tabIndex < 0 && !el.hasAttribute("tabindex");
        if (added) el.setAttribute("tabindex", "-1");
        el.classList.add("sectnav-focus");
        el.addEventListener("blur", () => {
            el.classList.remove("sectnav-focus");
            if (added) el.removeAttribute("tabindex");
        }, { once: true });
        el.focus({ preventScroll: true });
    };

    // ---------------- entries ----------------
    const items = [];
    if (nav) {
        nav.querySelectorAll('a[href^="#"]').forEach((a) => {
            const id = decodeURIComponent(a.getAttribute("href").slice(1));
            const target = id ? document.getElementById(id) : null;
            if (!target) { (a.closest("li") || a).hidden = true; return; }
            const fallback = a.textContent.replace(/\s+/g, " ").trim();
            const dot = document.createElement("span");
            dot.className = "sectnav-dot";
            dot.setAttribute("aria-hidden", "true");
            const label = document.createElement("span");
            label.className = "sectnav-label";
            a.textContent = "";
            a.className = "sectnav-link";
            a.append(dot, label);
            target.classList.add("sectnav-target");
            // A heading used as an anchor stands for its block (e.g. #capture -> the capture video block).
            const region = /^H[1-6]$/.test(target.tagName) ? target.parentElement : target;
            items.push({ a, id, target, region, label, fallback, parent: null, heading: null });
        });
        items.forEach((it) => {
            const outer = items.filter((o) => o !== it && o.region !== it.region && o.region.contains(it.region));
            it.parent = outer.length ? outer[outer.length - 1] : null;
        });
    }
    const tops = items.filter((it) => !it.parent);

    // ---------------- labels ----------------
    const headingOf = (it) => {
        if (it.target.tagName === "H3") return it.target;
        for (const h of it.region.querySelectorAll("h3")) {
            if (!items.some((o) => o.parent === it && o.region.contains(h))) return h;
        }
        return null;
    };
    const syncLabels = () => {
        items.forEach((it) => {
            it.heading = headingOf(it);
            const text = (it.heading && it.heading.textContent.replace(/\s+/g, " ").trim()) ||
                it.target.getAttribute("data-nav-label") || it.fallback;
            if (it.label.textContent !== text) it.label.textContent = text;
        });
    };

    // ---------------- fit beside the content column ----------------
    // Right edge of the widest block in any section below the hero (measured, so a wider
    // viewer or a new section is taken into account).
    const contentRight = () => {
        let right = 0;
        document.querySelectorAll("section:not(#hero)").forEach((s) => {
            for (const c of s.children) {
                const r = c.getBoundingClientRect();
                if (r.width && r.height && r.right > right) right = r.right;
            }
        });
        return right;
    };
    let revealAt = 0;
    const fit = () => {
        const hero = document.getElementById("hero");
        revealAt = hero ? hero.offsetHeight * 0.5 : 0;
        if (!nav || !items.length) return;
        let mode = "off";
        if (root.clientWidth >= MIN_WIDTH) {
            const limit = contentRight() + CLEARANCE;
            nav.dataset.mode = "full";
            if (nav.getBoundingClientRect().left >= limit) mode = "full";
            else {
                nav.dataset.mode = "compact";
                if (nav.getBoundingClientRect().left >= limit) mode = "compact";
            }
        }
        nav.dataset.mode = mode;
    };

    // ---------------- current entry ----------------
    let current = null;
    const setCurrent = (it) => {
        if (it === current) return;
        if (current) current.a.removeAttribute("aria-current");
        current = it;
        if (it) it.a.setAttribute("aria-current", "true");
    };
    const inView = (probe) => {
        let found = null;
        const live = [];                             // sections that are laid out, in page order
        tops.forEach((it) => {
            const r = it.region.getBoundingClientRect();
            if (r.height) live.push({ it, top: r.top });
        });
        for (let i = 0; i < live.length; i++) {      // a section reaches down to the next one
            const next = i + 1 < live.length ? live[i + 1].top : Infinity;
            if (live[i].top <= probe && probe < next) { found = live[i].it; break; }
        }
        for (let deeper = found; deeper; ) {       // the deepest nested entry under the probe
            deeper = null;
            for (const it of items) {
                if (it.parent !== found) continue;
                const r = it.region.getBoundingClientRect();
                if (r.height && r.top <= probe && probe < r.bottom) { found = deeper = it; break; }
            }
        }
        return found;
    };

    // A clicked entry stays current until the reader scrolls on: through the smooth scroll
    // (unless wheel / touch / keys take over) and after it, until the page moves again.
    let lock = null;
    let lockTimer = 0;
    const settle = () => { if (lock && lock.y === null) lock.y = window.scrollY; };
    const unlock = () => { if (lock) { lock = null; clearTimeout(lockTimer); schedule(); } };

    const update = () => {
        const y = window.scrollY;
        const vh = window.innerHeight;
        if (topBtn) topBtn.classList.toggle("is-shown", y > vh * 0.9);
        if (!items.length) return;
        nav.classList.toggle("is-shown", y > revealAt);
        if (lock) { setCurrent(lock.it); return; }
        const maxY = root.scrollHeight - vh;
        const t = Math.min(1, Math.max(0, 1 - (maxY - y) / (vh * 0.5)));
        setCurrent(inView(vh * (0.5 + 0.5 * t)));
    };

    let frame = 0;
    let needFit = true;
    const tick = () => {
        frame = 0;
        if (needFit) { needFit = false; fit(); }
        update();
    };
    function schedule(refit) {
        if (refit) needFit = true;
        if (!frame) frame = requestAnimationFrame(tick);
    }

    // ---------------- events ----------------
    window.addEventListener("scroll", () => {
        if (lock) {
            if (lock.y === null) { clearTimeout(lockTimer); lockTimer = setTimeout(settle, 180); }
            else if (Math.abs(window.scrollY - lock.y) > 2) lock = null;
        }
        schedule();
    }, { passive: true });
    window.addEventListener("scrollend", settle);
    window.addEventListener("resize", () => schedule(true));
    window.addEventListener("load", () => schedule(true));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => schedule(true));
    window.addEventListener("wheel", unlock, { passive: true });
    window.addEventListener("touchstart", unlock, { passive: true });
    const SCROLL_KEYS = ["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "];
    window.addEventListener("keydown", (e) => { if (SCROLL_KEYS.indexOf(e.key) >= 0) unlock(); });

    if (nav && items.length) {
        nav.addEventListener("click", (e) => {
            const a = e.target.closest("a.sectnav-link");
            if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
            const it = items.find((x) => x.a === a);
            if (!it) return;
            e.preventDefault();
            lock = { it, y: null };
            clearTimeout(lockTimer);
            lockTimer = setTimeout(settle, 180);
            setCurrent(it);
            it.target.scrollIntoView({ behavior: behavior(), block: "start" });
            setUrl("#" + it.id);
            if (e.detail === 0) focusLanding(it.target);    // keyboard activation
        });
        syncLabels();
        if ("MutationObserver" in window) {
            const mo = new MutationObserver(() => { syncLabels(); schedule(true); });
            items.forEach((it) => {
                if (it.heading) mo.observe(it.heading, { childList: true, characterData: true, subtree: true });
            });
        }
        nav.hidden = false;
    }

    if (topBtn) {
        topBtn.addEventListener("click", (e) => {
            lock = null;
            window.scrollTo({ top: 0, behavior: behavior() });
            if (location.hash) setUrl(location.pathname + location.search);
            if (e.detail === 0) {
                const hero = document.getElementById("hero");
                if (hero) focusLanding(hero);
            }
        });
        topBtn.hidden = false;
    }

    fit();
    needFit = false;
    update();
})();
