/**
 * Per-material viewer logic.
 *
 * Start-up. The first capture (a 4-14 MB range-read out of hdr.tar on the
 * HF CDN) is the long pole, so nothing it does not need stands in its way:
 *   1.  obj.html's <head> asks for the shipped tar index
 *       (data/tar_index/<id>.json) and, the moment it is in, for the first
 *       capture (the archive's first member = the first scan that has an
 *       HDR capture): before the stylesheets, three.js, the metadata or the
 *       3-D panels. (window.__objBoot; without it this file does the same.)
 *   2.  This file starts every other download at once: scan_log /
 *       rotated_camera / hdr_crop_bboxes / bbox from HF, the material stats,
 *       the manifest; the decode worker starts too, and the image pipeline
 *       joins the first capture's download.
 *   3.  When the metadata is in: sliders, labels, material info; then, once
 *       three.js has loaded (obj.html loads it after this script), the two
 *       3-D panels, one per task.
 *   4.  On slider change the labels and the 3-D panels follow every step;
 *       the capture is loaded where the slider settles, decoded + tone-mapped
 *       in a worker and painted (see "image pipeline" below).
 */

(async function () {
    // -----------------  parse query  -----------------
    const qs = new URLSearchParams(location.search);
    let id = parseInt(qs.get("id") || "0", 10);
    if (!(id >= 0)) id = 0;
    const tarURL = CONFIG.hdrTarURL(id);

    document.getElementById("material-id").textContent = id;
    document.title = `Material ${id} · RoboCloth`;
    const subtitle = document.getElementById("subtitle");

    // Point the Hugging Face button at this material's folder on the HF
    // dataset tree (the human-friendly tree page, not a raw file).
    const hfLink = document.getElementById("btn-hf");
    if (hfLink) hfLink.href = `${CONFIG.HF_BASE}/tree/main/${CONFIG.materialDir(id)}`;

    // Bubble unexpected errors to the subtitle so headless / no-JS-console
    // users still see them.  We swallow WebGL context errors here because
    // those are handled in-place by markVizUnavailable() below.
    window.addEventListener("unhandledrejection", (ev) => {
        const reason = String(ev.reason || "");
        if (reason.includes("WebGL")) return;
        console.error("[unhandled rejection]", ev.reason);
        subtitle.innerHTML = `<span style="color:#a33">Error: ${escapeHtml(reason)}</span>`;
    });

    // -----------------  start every download now  -----------------
    const getJSON = (url, optional) => fetch(url).then((r) => {
        if (r.ok) return r.json();
        if (optional) return null;
        throw new Error(`HTTP ${r.status} for ${url.split("/").slice(-2).join("/")}`);
    });
    const handled = (p) => { p.catch(() => {}); return p; };   // reported where awaited
    // obj.html's <head> has already asked for the tar index and started the
    // first capture's download (window.__objBoot); without it, start here.
    const pre = window.__objBoot && window.__objBoot.id === id ? window.__objBoot : null;
    const boot = {
        index: pre ? pre.index : getJSON(CONFIG.tarIndexURL(id), true).catch(() => null),
        scanLog: handled(getJSON(CONFIG.scanLogURL(id))),
        rotatedCam: handled(getJSON(CONFIG.rotatedCameraURL(id))),
        crop: getJSON(CONFIG.cropBBoxURL(id), true).catch(() => null),
        bbox: getJSON(CONFIG.bboxURL(id), true).catch(() => null),
        stats: getJSON(`data/material_stats/${id}.json`, true)
            .catch((err) => { console.warn("[stats]", err); return null; }),
        manifest: getJSON(CONFIG.manifestURL(), true).catch(() => null),
    };
    if (HDRDisplay.warm) HDRDisplay.warm();

    // Handle for tests / screenshots (read-only use). `shown` is the frame
    // currently on the image canvas (also announced as an "obj:shown" event).
    window.__objViewer = { b1Scene: null, w0Scene: null, swatch: "placeholder", shown: null };

    // Material data, filled in when the metadata arrives.
    let rotatedCam = null, crop = null, materialBBox = null, stats = null;
    let processed = null, validEntries = [];
    let b1Scene = null, w0Scene = null, scenesReady = false;

    // -----------------  tar index  -----------------
    //
    // Two regimes:
    //   1. data/tar_index/<id>.json is shipped (all 500 materials): one
    //      ~40 KB request, no indicator needed.
    //   2. It is not: walk the live tar's headers (one CDN round trip per
    //      file, ~580 sequential requests, tens of seconds). The walk fills
    //      `tarIdx.map` progressively, so a frame is shown as soon as its
    //      header has been read (the first capture is the first member).
    const tarIdx = { map: null, complete: false, error: null, waiters: new Set(), ready: null };

    function startTarIndex() {
        if (tarIdx.ready) return tarIdx.ready;
        const meta = document.getElementById("meta-filename");
        let indicator = null;
        const setIndicator = (text) => {
            if (!indicator) {
                indicator = document.createElement("span");
                indicator.id = "tar-index-progress";
                indicator.style.cssText = "color:var(--muted);font-size:0.78rem";
                meta.after(indicator);
            }
            indicator.textContent = ` — ${text}`;
        };
        const removeIndicator = () => {
            if (indicator) { indicator.remove(); indicator = null; }
        };
        const notify = () => { for (const w of [...tarIdx.waiters]) w(); };
        // Only say something if the index is not there within 350 ms.
        const fallbackTimer = setTimeout(() => setIndicator("loading tar index…"), 350);

        tarIdx.ready = (async () => {
            const shipped = await boot.index;
            clearTimeout(fallbackTimer);
            if (shipped) {
                tarIdx.map = shipped;
                tarIdx.complete = true;
                removeIndicator();
                return;
            }
            tarIdx.map = {};
            let n = 0;
            const label = () => `indexing hdr.tar (first visit): ${n} / ${validEntries.length || "?"} files`;
            setIndicator(label());
            TarReader.buildIndex(tarURL, {
                into: tarIdx.map,
                onEntry: () => {
                    n++;
                    if (n % 10 === 0) setIndicator(label());
                    notify();
                },
            }).then(() => {
                tarIdx.complete = true;
                removeIndicator();
                notify();
            }, (err) => {
                console.error("[tar index]", err);
                tarIdx.error = err;
                tarIdx.complete = true;
                removeIndicator();
                notify();
            });
        })();
        return tarIdx.ready;
    }

    /** Resolve to the index once it covers `filename` (or is complete). */
    async function indexFor(filename, signal) {
        await startTarIndex();
        const has = () => filename in tarIdx.map;
        while (!has() && !tarIdx.complete) {
            await new Promise((resolve, reject) => {
                const onAbort = () => finish(() => reject(new DOMException("Aborted", "AbortError")));
                const waiter = () => { if (has() || tarIdx.complete) finish(resolve); };
                const finish = (f) => {
                    tarIdx.waiters.delete(waiter);
                    if (signal) signal.removeEventListener("abort", onAbort);
                    f();
                };
                tarIdx.waiters.add(waiter);
                if (signal) {
                    if (signal.aborted) { onAbort(); return; }
                    signal.addEventListener("abort", onAbort);
                }
            });
        }
        if (!has() && tarIdx.error) throw tarIdx.error;
        return tarIdx.map;
    }

    /**
     * Drop the <head> boot script's hold on the first capture's download
     * once a real job has joined it (or wants another file, which then
     * cancels it). Waits for the boot's own request to have been made.
     */
    let bootReleased = false;
    function releaseBoot() {
        if (bootReleased || !pre) return;
        bootReleased = true;
        const drop = () => pre.ctl.abort();
        Promise.resolve(pre.first).then(drop, drop);
    }

    /** The archive's first member: the first scan that has a capture. */
    function firstCapture(index) {
        for (const k in index) if (k.indexOf("/") < 0) return k;
        return null;
    }

    // -----------------  swatch texture (once, when the viewer is idle)  -----------------
    //
    // The swatch in both scenes starts as a neutral woven tile in the
    // material's mean colour; once a capture has been shown and the viewer
    // has been idle for a moment we fetch the most top-down capture and
    // rectify the swatch out of it. Its download yields to the slider: an
    // image request aborts it, and it is retried after the next idle spell
    // (a longer one on a slow link, where it costs many seconds).
    const SWATCH_IDLE_MS = 1500;
    const SWATCH_IDLE_SLOW_MS = 6000;     // below ~8 Mbit/s
    const swatch = { state: "idle", ctl: null, fetched: false };   // idle | running | done
    let swatchTimer = null;

    function scheduleSwatch() {
        clearTimeout(swatchTimer);
        if (swatch.state !== "idle" || !scenesReady) return;
        const slow = lastBytesPerSec !== null && lastBytesPerSec < PREFETCH_MIN_BPS / 2.5;
        swatchTimer = setTimeout(() => {
            if (mainJob || settleTimer) { scheduleSwatch(); return; }
            startSwatch();
        }, slow ? SWATCH_IDLE_SLOW_MS : SWATCH_IDLE_MS);
    }

    function startSwatch() {
        if (swatch.state !== "idle" || !scenesReady) return;
        if (!(b1Scene || w0Scene) || !materialBBox || !crop || !crop.intrinsics
            || !window.SwatchTexture) { swatch.state = "done"; return; }
        swatch.state = "running";
        swatch.fetched = false;
        const ctl = swatch.ctl = new AbortController();
        buildSwatchTexture(ctl.signal).then(() => {
            swatch.state = "done";
            swatch.ctl = null;
        }, (err) => {
            swatch.ctl = null;
            if (TarReader.isAbort(err)) { swatch.state = "idle"; return; }
            swatch.state = "done";
            console.warn("[swatch]", err);
            window.__objViewer.swatch = `placeholder (${err.message})`;
        });
    }

    /** A user-requested download takes priority over the swatch's. */
    function pauseSwatch() {
        clearTimeout(swatchTimer);
        if (swatch.state === "running" && swatch.ctl && !swatch.fetched) swatch.ctl.abort();
    }

    async function buildSwatchTexture(signal) {
        // the most top-down capture that exists (unmatched scans have no
        // capture and no calibrated camera, e.g. material 27's best view)
        await startTarIndex();
        const rotIdx = Transforms.indexRotatedCamera(rotatedCam);
        const cands = validEntries.filter((c) => rotIdx.has(Number(c.camera_id))
            && (!tarIdx.complete || tarIdx.error || c.filename in tarIdx.map));
        if (!cands.length) throw new Error("no capture with a calibrated camera");
        const e = cands[SwatchTexture.pickFrontal(cands, materialBBox)];
        const rot = rotIdx.get(Number(e.camera_id));
        const project = Transforms.makeWorld0Projector(rot, crop.intrinsics);
        if (!project) throw new Error("bad camera pose");
        const index = await indexFor(e.filename, signal);
        const [buf, st] = await Promise.all([
            TarReader.fetchFile(tarURL, index, e.filename, { signal, priority: "low" }),
            boot.stats]);
        swatch.fetched = true;
        const meanLin = st && st.mean_rgb_16bit
            ? st.mean_rgb_16bit.map(v => v / 65535) : null;
        const tile = await SwatchTexture.rectify(buf, project, materialBBox, { meanLin });
        if (!tile) {
            console.info("[swatch] projection check failed; keeping the placeholder tile");
            window.__objViewer.swatch = "placeholder (projection check failed)";
            return;
        }
        if (b1Scene) b1Scene.setSwatchTexture(tile);
        if (w0Scene) w0Scene.setSwatchTexture(tile);
        window.__objViewer.swatch = `capture ${e.filename}`;
        window.__objViewer.swatchTile = tile;
    }

    // -----------------  image pipeline  -----------------
    //
    // Every capture is a 4-14 MB 16-bit PNG range-read out of hdr.tar, so:
    //   * a frame is fetched only once the slider settles (SETTLE_MS without
    //     a change) or is released; at most one such download runs, and it is
    //     aborted as soon as the user moves to another frame;
    //   * the download is an XHR straight into a Blob (TarReader), so a busy
    //     main thread cannot slow it; the badge shows MB received of the
    //     expected size (known from the index) and the time left;
    //   * decoding + tone mapping run in a worker (HDRDisplay.decode); the
    //     previous image stays up with the loading badge meanwhile;
    //   * decoded frames live in a small LRU, so revisits are instant;
    //   * after a keyboard step on a fast link the next frame in the same
    //     direction is prefetched (low priority, aborted when the user moves).
    const SETTLE_MS = 180;                // keys / after a click
    const SETTLE_DRAG_MS = 300;           // pointer still down: the user may move on
    const DECODED_MAX = 10;               // ~6.8 MB each at 1600x1067
    const PREFETCH_MIN_BPS = 2.5e6;       // prefetch only on links of ~20 Mbit/s and up

    const state = {
        cropToBBox: false,
        exposure: 1.2,
        gamma: 2.2,
        currentFrame: 0,
    };

    const imgCanvas = document.getElementById("image-canvas");
    const imgWrap = imgCanvas.parentElement;
    const spinner = document.getElementById("img-spinner");
    const badge = document.createElement("div");
    badge.className = "img-loading";
    badge.setAttribute("role", "status");
    badge.innerHTML = '<span class="spinner"></span><span class="img-loading-text"></span>';
    imgWrap.appendChild(badge);
    const badgeText = badge.querySelector(".img-loading-text");
    const progBar = document.createElement("div");
    progBar.className = "img-progress";
    progBar.innerHTML = "<i></i>";
    imgWrap.appendChild(progBar);
    const progFill = progBar.firstChild;

    const decoded = new Map();            // key -> ImageBitmap, most recent last
    const prefetchJobs = new Map();       // filename -> AbortController
    let want = null;                      // { i, filename, key, opts, label }: the frame to show
    let shownKey = null;
    let painted = false;                  // has any capture been painted yet?
    let settleTimer = null;
    let mainJob = null;                   // { key, filename, ctl, w, total, eta }
    let prefetchTimer = null;
    // Throughput of the last fresh capture download, remembered for the tab
    // (sessionStorage) so the next material page starts out knowing the link.
    const BPS_KEY = "robocloth.viewer.Bps";
    let lastBytesPerSec = (() => {
        try { const v = Number(sessionStorage.getItem(BPS_KEY)); return v > 0 ? v : null; }
        catch (e) { return null; }
    })();
    const nav = { by: null, dir: 1 };     // last navigation: "key" | "pointer", direction

    window.__objViewer.debug = () => ({
        mainJob: mainJob && mainJob.key, settling: !!settleTimer,
        prefetch: [...prefetchJobs.keys()], decoded: decoded.size,
        proxy: { have: !!proxy.index, shown: proxy.shownKey, job: proxy.job && proxy.job.key, cached: proxy.cache.size },
        lastBytesPerSec, swatch: swatch.state,
    });

    function displayOpts(e) {
        let bbox = null;
        if (state.cropToBBox && crop && crop.bboxes && e.camera_id != null) {
            bbox = crop.bboxes[String(e.camera_id)] || null;
        }
        return { exposure: state.exposure, gamma: state.gamma, bbox, maxWidth: 1600 };
    }
    function keyOf(filename, o) {
        return `${filename}|${o.bbox ? o.bbox.join(",") : "full"}|${o.exposure.toFixed(5)}|${o.gamma}`;
    }
    const mb = (b) => (b / 1e6).toFixed(1);

    function getDecoded(key) {
        const b = decoded.get(key);
        if (b) { decoded.delete(key); decoded.set(key, b); }
        return b;
    }
    function putDecoded(key, bmp) {
        const old = decoded.get(key);
        decoded.delete(key);
        if (old && old !== bmp && old.close) old.close();
        decoded.set(key, bmp);
        while (decoded.size > DECODED_MAX) {
            const k = decoded.keys().next().value;
            const b = decoded.get(k);
            decoded.delete(k);
            if (b.close) b.close();       // drawImage copied it; safe to free
        }
    }

    let badgeT = 0;
    /**
     * Loading badge over the capture (centred and larger while nothing has
     * been painted yet) + a progress bar along the image's bottom edge.
     * opts: { throttle, frac (0..1, shows the bar) }
     */
    function setLoading(text, opts = {}) {
        if (text == null) {
            imgWrap.classList.remove("is-loading", "has-progress");
            spinner.style.display = "none";
            return;
        }
        const now = performance.now();
        if (opts.throttle && now - badgeT < 100) return;
        badgeT = now;
        badgeText.textContent = text;
        imgWrap.classList.add("is-loading");
        imgWrap.classList.toggle("is-empty", !painted);
        if (opts.frac != null) {
            progFill.style.transform = `scaleX(${Math.max(0, Math.min(1, opts.frac))})`;
            imgWrap.classList.add("has-progress");
        } else {
            imgWrap.classList.remove("has-progress");
        }
        spinner.style.display = "inline-block";
    }

    /** Show frame `i`: labels + 3-D right away, the image once it settles. */
    function selectFrame(i, how) {
        if (validEntries.length === 0) return;
        i = clamp(i, 0, validEntries.length - 1);
        if (i !== state.currentFrame) nav.dir = i > state.currentFrame ? 1 : -1;
        state.currentFrame = i;
        updateFrameUI(i);
        requestImage(how);
    }

    /** Update everything but the image for the current entry (cheap). */
    function updateFrameUI(i) {
        const e = validEntries[i];
        sliderFrame.value = i;
        sliderCam.value = processed.camera_ids.indexOf(e.camera_id);
        sliderLight.value = processed.light_ids.indexOf(e.light_id);
        const turnIdx = processed.turn_angles.findIndex(
            t => Math.abs(t - e.turn_angle) < 1e-3,
        );
        if (turnIdx >= 0) sliderTurn.value = turnIdx;

        document.getElementById("frame-tag").textContent =
            `${i + 1} / ${validEntries.length}`;
        document.getElementById("cam-tag").textContent = `id ${e.camera_id}`;
        document.getElementById("light-tag").textContent = `id ${e.light_id}`;
        document.getElementById("turn-tag").textContent =
            `${e.turn_angle.toFixed(1)}°`;
        document.getElementById("meta-filename").textContent = e.filename;
        document.getElementById("meta-camera").textContent = e.camera_id;
        document.getElementById("meta-light").textContent = e.light_id;
        document.getElementById("meta-turn").textContent = e.turn_angle.toFixed(1) + "°";
        update3D(e);
    }

    /** Highlight the entry's camera + light in both 3-D views (marks them dirty). */
    function update3D(e) {
        if (b1Scene) {
            b1Scene.setSelection({ camPos: e.cam_pos_b1, lightPos: e.light_pos_b1 });
            // In base1 the material follows the physical turntable: rotate
            // its world0 bbox forward by +turn_angle about the calibrated
            // axis / centre so the slab orientation matches the current
            // frame's turntable position.
            b1Scene.setMaterialTurnAngle(e.turn_angle);
        }
        if (w0Scene) w0Scene.setSelection({ camPos: e.cam_pos_w0, lightPos: e.light_pos_w0 });
    }

    /**
     * Make the image canvas follow state.currentFrame. how = "now" loads at
     * once; "drag" waits until the selection has been still for SETTLE_MS.
     */
    function requestImage(how) {
        if (validEntries.length === 0) return;
        const i = state.currentFrame, e = validEntries[i];
        const opts = displayOpts(e);
        show({ i, filename: e.filename, key: keyOf(e.filename, opts), opts, label: `frame ${i + 1}` }, how);
    }

    function show(w, how) {
        want = w;
        clearTimeout(settleTimer);
        settleTimer = null;
        clearTimeout(prefetchTimer);

        // Stop work for frames the user has moved away from. (Same capture,
        // other exposure / crop: let the old job finish, the new one joins
        // its download instead of starting it again.)
        if (mainJob && mainJob.key !== w.key) {
            if (mainJob.filename !== w.filename) mainJob.ctl.abort();
            mainJob = null;
        }
        for (const [fn, ctl] of prefetchJobs) {
            if (fn !== w.filename) { ctl.abort(); prefetchJobs.delete(fn); }
        }

        if (decoded.has(w.key)) {
            if (w.key !== shownKey) paint(w);
            else setLoading(null);
            return;
        }
        showProxy(w);                                // the small version right away
        if (mainJob) {                               // already loading this frame
            mainJob.w = w;                           // (now with its frame number)
            return;
        }
        setLoading(w.label);
        // Bytes already here or on their way (a prefetched neighbour): no new
        // request is involved, so there is nothing to wait for.
        if (how === "now" || TarReader.hasFile(tarURL, w.filename)) startMain();
        else settleTimer = setTimeout(startMain, pointerDown ? SETTLE_DRAG_MS : SETTLE_MS);
    }

    async function startMain() {
        settleTimer = null;
        const w = want;
        if (!w || mainJob) return;
        if (decoded.has(w.key)) {                    // a prefetch got there first
            if (w.key !== shownKey) paint(w); else setLoading(null);
            return;
        }
        const ctl = new AbortController();
        const job = mainJob = { key: w.key, filename: w.filename, ctl, w, total: null, eta: null };
        try {
            const index = await indexFor(w.filename, ctl.signal);
            // (the first job may join the download begun in <head>: still news)
            const fresh = !TarReader.hasFile(tarURL, w.filename) || (pre && !bootReleased);
            const ent = index[w.filename] || index[w.filename.split("/").pop()];
            if (ent) job.total = ent[1];
            if (fresh) {
                pauseSwatch();
                if (ent && mainJob === job) {
                    setLoading(`${job.w.label} · ${mb(ent[1])} MB · connecting…`, { frac: 0 });
                }
            }
            let first = null;                        // first progress event [t, bytes]
            const p = TarReader.fetchFile(tarURL, index, w.filename, {
                signal: ctl.signal,
                priority: "high",
                onProgress: (got, total) => {
                    const now = performance.now();
                    if (!first) first = [now, got];
                    if (mainJob !== job) return;
                    // Time left at the mean rate since the first bytes, once
                    // they have flowed for a second (data arrives in bursts;
                    // the mean errs on the long side early on, never short).
                    let eta = "";
                    const dt = (now - first[0]) / 1000;
                    if (dt > 1 && got > first[1]) {
                        job.eta = (total - got) / ((got - first[1]) / dt);
                        const left = Math.round(job.eta);
                        if (left >= 2) eta = ` · ~${left} s left`;
                    }
                    setLoading(`${job.w.label} · ${mb(got)} / ${mb(total)} MB${eta}`,
                        { throttle: true, frac: got / total });
                },
            });
            // If this frame was being prefetched (or started by the <head>
            // boot script), the download is now ours.
            const pf = prefetchJobs.get(w.filename);
            if (pf) { prefetchJobs.delete(w.filename); pf.abort(); }
            releaseBoot();
            const blob = await p;
            job.eta = 0;
            // Link rate: bytes over time since the first bytes (the resolve +
            // CDN set-up before them is a fixed ~0.5-1 s, see quietMoment).
            const dt = first ? (performance.now() - first[0]) / 1000 : 0;
            if (fresh && first && blob.size - first[1] > 1e6) {
                lastBytesPerSec = (blob.size - first[1]) / Math.max(dt, 0.1);
                try { sessionStorage.setItem(BPS_KEY, String(Math.round(lastBytesPerSec))); }
                catch (e) { /* storage blocked: fine */ }
            }
            if (mainJob === job) setLoading(`${job.w.label} · decoding`);
            const bmp = await HDRDisplay.decode(blob, { ...w.opts, signal: ctl.signal });
            putDecoded(w.key, bmp);                  // kept even if the user moved on
            if (mainJob === job) mainJob = null;
            if (want && want.key === w.key) paint(want);
        } catch (err) {
            if (mainJob === job) mainJob = null;
            if (TarReader.isAbort(err)) return;
            if (!/^Tar index miss/.test(err.message)) console.warn("[image]", err);
            if (want && want.key === w.key) showImageError(err);
        }
    }

    function paint(w) {
        const bmp = getDecoded(w.key);
        if (!bmp) return;
        if (imgCanvas.width !== bmp.width || imgCanvas.height !== bmp.height) {
            imgCanvas.width = bmp.width;
            imgCanvas.height = bmp.height;
        }
        imgCanvas.getContext("2d").drawImage(bmp, 0, 0);
        shownKey = w.key;
        painted = true;
        proxy.shownKey = null;
        if (proxy.job && proxy.job.filename === w.filename) { proxy.job.ctl.abort(); proxy.job = null; }
        imgWrap.classList.remove("is-empty", "is-proxy");
        setLoading(null);
        const detail = { frame: w.i, filename: w.filename };
        window.__objViewer.shown = { ...detail, t: performance.now() };
        document.dispatchEvent(new CustomEvent("obj:shown", { detail }));
        // Background work only once the selection has settled (a drag over
        // cached frames paints on every step).
        clearTimeout(prefetchTimer);
        if (w.i != null) {
            prefetchTimer = setTimeout(() => {
                if (want && want.i === w.i) prefetchNext(w.i);
            }, SETTLE_DRAG_MS);
        }
        scheduleSwatch();
    }

    // -----------------  scrubbing proxies  -----------------
    //
    // Every capture also exists as a 640 px WebP (sqrt-encoded, built by
    // docs/tools/proxies/build_proxies.py, one ustar + index per material in
    // the assets repo). One such file is 15-40 KB, so the frame under the
    // slider is on screen at once, soft and marked "preview", while the full
    // capture follows the usual settle-then-download path and replaces it.
    const PROXY_MAX = 40;                 // decoded proxies kept (~1 MB each)
    const proxy = { index: null, ready: null, job: null, shownKey: null, cache: new Map() };
    proxy.ready = Promise.resolve(pre && pre.proxyIndex ? pre.proxyIndex
        : fetch(CONFIG.proxyIndexURL(id)).then((r) => (r.ok ? r.json() : null)).catch(() => null))
        .then((idx) => { proxy.index = idx && typeof idx === "object" ? idx : null; return proxy.index; });

    function proxyOpts(w, meta) {
        const long = (meta && meta.long) || 640;
        const hw = (meta && meta.source_hw) || [2048, 3072];
        const sc = long / Math.max(hw[0], hw[1]);
        const bbox = w.opts.bbox ? w.opts.bbox.map((v) => Math.round(v * sc)) : null;
        return { ...w.opts, bbox, encoding: "sqrt" };
    }

    function showProxy(w) {
        if (!w.filename || proxy.shownKey === w.key) return;
        const name = w.filename.split("/").pop().replace(/\.png$/, ".webp");
        const hit = proxy.cache.get(w.key);
        if (hit) { paintProxy(w, hit); return; }
        if (proxy.job) {
            if (proxy.job.key === w.key) return;
            proxy.job.ctl.abort();
            proxy.job = null;
        }
        const ctl = new AbortController();
        const job = proxy.job = { key: w.key, filename: w.filename, ctl };
        proxy.ready.then(async (index) => {
            if (!index || !index[name] || ctl.signal.aborted) { if (proxy.job === job) proxy.job = null; return; }
            const blob = await TarReader.fetchFile(CONFIG.proxyTarURL(id), index, name,
                { signal: ctl.signal, priority: "high" });
            const bmp = await HDRDisplay.decode(blob, { ...proxyOpts(w, index.__meta), signal: ctl.signal });
            proxy.cache.set(w.key, bmp);
            if (proxy.cache.size > PROXY_MAX) {
                const k0 = proxy.cache.keys().next().value;
                const b0 = proxy.cache.get(k0);
                proxy.cache.delete(k0);
                if (b0 && b0.close) b0.close();
            }
            if (proxy.job === job) proxy.job = null;
            if (want && want.key === w.key && shownKey !== w.key) paintProxy(want, bmp);
        }).catch((err) => {
            if (proxy.job === job) proxy.job = null;
            if (!TarReader.isAbort(err)) console.warn("[proxy]", err);
        });
    }

    function paintProxy(w, bmp) {
        const ctx = imgCanvas.getContext("2d");
        const sameShape = imgCanvas.width > 0 && imgCanvas.height > 0 &&
            Math.abs(imgCanvas.width / imgCanvas.height - bmp.width / bmp.height) < 0.02;
        if (shownKey && sameShape) {
            // a full capture is up: keep its canvas size (no layout jump), draw the proxy over it
            ctx.drawImage(bmp, 0, 0, imgCanvas.width, imgCanvas.height);
        } else {
            imgCanvas.width = bmp.width;
            imgCanvas.height = bmp.height;
            ctx.drawImage(bmp, 0, 0);
        }
        shownKey = null;
        proxy.shownKey = w.key;
        painted = true;
        imgWrap.classList.remove("is-empty");
        imgWrap.classList.add("is-proxy");
        document.dispatchEvent(new CustomEvent("obj:proxy", { detail: { frame: w.i, filename: w.filename } }));
    }

    function showImageError(err) {
        const ctx = imgCanvas.getContext("2d");
        imgCanvas.width = 800; imgCanvas.height = 533;
        ctx.fillStyle = "#0d0f12";
        ctx.fillRect(0, 0, 800, 533);
        ctx.textAlign = "center";
        ctx.font = "16px sans-serif";
        // ~1.5 % of the scan-log entries were not matched in calibration
        // (unmatched_scan_ids.json) and have no capture in hdr.tar.
        const miss = /^Tar index miss/.test(err.message);
        ctx.fillStyle = miss ? "#8a93a3" : "#a33";
        ctx.fillText(miss ? "No HDR capture stored for this configuration (unmatched scan)"
                          : err.message, 400, 270);
        shownKey = null;
        painted = true;
        imgWrap.classList.remove("is-empty");
        setLoading(null);
    }

    /**
     * After a keyboard step, warm the next frame in the same direction
     * (best-effort, fast links only: on a slider drag the next target is
     * anywhere, and a speculative 10 MB costs the user's bandwidth).
     */
    function prefetchNext(i) {
        if (nav.by !== "key" || !tarIdx.complete || tarIdx.error) return;
        const conn = navigator.connection;
        if (conn && (conn.saveData || /2g|3g/.test(conn.effectiveType || ""))) return;
        if (lastBytesPerSec === null || lastBytesPerSec < PREFETCH_MIN_BPS) return;
        const j = i + nav.dir;
        if (j < 0 || j >= validEntries.length || prefetchJobs.size >= 1) return;
        const e = validEntries[j], opts = displayOpts(e), key = keyOf(e.filename, opts);
        if (decoded.has(key) || prefetchJobs.has(e.filename) || !(e.filename in tarIdx.map)) return;
        const ctl = new AbortController();
        prefetchJobs.set(e.filename, ctl);
        TarReader.fetchFile(tarURL, tarIdx.map, e.filename, { signal: ctl.signal, priority: "low" })
            .then((blob) => HDRDisplay.decode(blob, { ...opts, signal: ctl.signal, background: true }))
            .then((bmp) => putDecoded(key, bmp))
            .catch(() => { /* best-effort */ })
            .finally(() => { if (prefetchJobs.get(e.filename) === ctl) prefetchJobs.delete(e.filename); });
    }

    // -----------------  sliders + buttons  -----------------
    const sliderFrame = document.getElementById("slider-frame");
    const sliderCam = document.getElementById("slider-camera");
    const sliderLight = document.getElementById("slider-light");
    const sliderTurn = document.getElementById("slider-turn");

    // While a slider is dragged (`input`) the labels and the 3-D panels
    // follow every step, but the capture itself is only fetched for the
    // position the user settles on. `change` fires on release (load at
    // once) and on every key press (wait for key repeat to settle).
    let lastPointerUp = -1e9;
    let pointerDown = false;
    const markRelease = () => { pointerDown = false; lastPointerUp = performance.now(); };
    window.addEventListener("pointerup", markRelease, true);
    window.addEventListener("pointercancel", markRelease, true);
    const onSliderChange = () =>
        requestImage(performance.now() - lastPointerUp < 250 ? "now" : "drag");

    sliderFrame.addEventListener("input", () => {
        selectFrame(parseInt(sliderFrame.value, 10), "drag");
    });

    // Camera/Light/Turn sliders pick the NEAREST entry that matches the
    // current combination. With 580 entries spread across 580 cameras,
    // 86 lights, and 84 turn angles, not every (cam,light,turn) triplet
    // exists — so we fall back to nearest match.
    sliderCam.addEventListener("input", () => {
        if (!processed) return;
        const camId = processed.camera_ids[parseInt(sliderCam.value, 10)];
        const idx = findEntry(validEntries, { camera_id: camId });
        if (idx >= 0) selectFrame(idx, "drag");
    });
    sliderLight.addEventListener("input", () => {
        if (!processed) return;
        const lightId = processed.light_ids[parseInt(sliderLight.value, 10)];
        const idx = findEntry(validEntries, { light_id: lightId });
        if (idx >= 0) selectFrame(idx, "drag");
    });
    sliderTurn.addEventListener("input", () => {
        if (!processed) return;
        const turn = processed.turn_angles[parseInt(sliderTurn.value, 10)];
        const idx = findEntry(validEntries, { turn_angle: turn });
        if (idx >= 0) selectFrame(idx, "drag");
    });
    for (const s of [sliderFrame, sliderCam, sliderLight, sliderTurn]) {
        s.addEventListener("change", onSliderChange);
        s.addEventListener("pointerdown", () => { pointerDown = true; nav.by = "pointer"; });
        s.addEventListener("keydown", () => { nav.by = "key"; });
    }

    document.getElementById("btn-rand").addEventListener("click", () => {
        nav.by = "pointer";
        selectFrame(Math.floor(Math.random() * validEntries.length), "now");
    });
    document.getElementById("btn-crop").addEventListener("click", (e) => {
        state.cropToBBox = !state.cropToBBox;
        e.currentTarget.classList.toggle("active", state.cropToBBox);
        requestImage("now");
    });
    document.getElementById("btn-expmore").addEventListener("click", () => {
        state.exposure *= 1.4; requestImage("now");
    });
    document.getElementById("btn-expless").addEventListener("click", () => {
        state.exposure /= 1.4; requestImage("now");
    });

    // -----------------  first capture: as soon as the tar index is in  -----------------
    boot.index.then((index) => {
        if (!index || want) return;                  // no shipped index / already chosen
        const filename = firstCapture(index);
        if (!filename) return;
        const opts = displayOpts({ filename });
        show({ i: null, filename, key: keyOf(filename, opts), opts, label: "first capture" }, "now");
    });

    // Small things that need no metadata.
    boot.stats.then((s) => { stats = s; if (s) applyMaterialStats(s); });
    boot.manifest.then((m) => setupNav(m || {}, id));

    // -----------------  metadata  -----------------
    // bbox.json stores the material's 3D bounding box in WORLD0 frame
    // (computed by recon/calibration/shape_matching.py:process_pointcloud_to_base).
    // crop (hdr_crop_bboxes.json) carries the per-image 2D pixel crops.
    let scanLog;
    try {
        [scanLog, rotatedCam, crop, materialBBox] = await Promise.all([
            boot.scanLog, boot.rotatedCam, boot.crop, boot.bbox]);
    } catch (err) {
        subtitle.innerHTML = `<span style="color:#a33">Failed to load metadata: ${escapeHtml(err.message)}</span>`;
        return;
    }

    subtitle.innerHTML = `
        <b>${scanLog.length}</b> capture configurations ·
        <b>${rotatedCam.length}</b> calibrated cameras
    `;

    // The raw scan-log entries are the source of truth for capture geometry.
    // rotated_camera.json is consulted for the "calibrated cameras" count and
    // the swatch texture's camera; per-material BRDF stats are precomputed
    // offline (see webpage/tools/build_material_stats.py).
    processed = Transforms.processScanLog(scanLog);
    validEntries = processed.entries;
    populateMaterialInfo();

    sliderFrame.max = Math.max(0, validEntries.length - 1);
    sliderCam.max = Math.max(0, processed.camera_ids.length - 1);
    sliderLight.max = Math.max(0, processed.light_ids.length - 1);
    sliderTurn.max = Math.max(0, processed.turn_angles.length - 1);

    // Initial frame: the first scan that has a capture in hdr.tar. (A few
    // scans per material were not matched in calibration and have none;
    // for materials 27, 355, 362 and 401 that includes scan 0.)
    const shippedIndex = await boot.index;
    let first = 0;
    if (shippedIndex) {
        const k = validEntries.findIndex((e) => e.filename in shippedIndex);
        if (k >= 0) first = k;
    }
    state.currentFrame = first;
    selectFrame(first, "now");     // joins the first capture's download

    // -----------------  3-D scenes — two views of one chain  -----------------
    build3D();

    /**
     * Setting up a WebGL panel (context, shaders) is the page's longest
     * main-thread work (~0.1-0.5 s each on a real GPU, more in software GL).
     * It must not land just as the first capture is ready to paint, nor
     * during a drag: start it once the capture is on screen, or as soon as
     * the capture is known to be seconds away (slow link), or after 4 s.
     */
    async function quietMoment() {
        const t0 = performance.now();
        for (;;) {
            if (!pointerDown && !settleTimer) {
                if (painted) return;
                const j = mainJob;
                if (!j) return;                               // nothing on its way
                let eta = j.eta;                              // set after 1 s of data
                if (eta == null && j.total && lastBytesPerSec) eta = 0.8 + j.total / lastBytesPerSec;
                if (eta != null && eta > 2) return;
                if (performance.now() - t0 > 4000) return;
            }
            await new Promise((r) => setTimeout(r, 100));
        }
    }

    async function build3D() {
        // three.js, OrbitControls and viz.js load after this script.
        await new Promise((resolve) => {
            if (document.readyState !== "loading") resolve();
            else document.addEventListener("DOMContentLoaded", resolve, { once: true });
        });
        await quietMoment();
        const b1Canvas = document.getElementById("viz-base1");
        const w0Canvas = document.getElementById("viz-world0");

        // Both panels use the same dark "launch video" stage (js/viz.js): the
        // swatch at true size lit by a spot light at the selected LED, the real
        // camera / LED heads as cut-outs, a frustum and a light cone. Without
        // WebGL (or three.js) each panel shows a static picture instead.
        const canWebGL = !!window.THREE && window.Viz && Viz.webglAvailable();
        if (!canWebGL) {
            markVizUnavailable(b1Canvas, null, false);
            markVizUnavailable(w0Canvas, null, true);
            for (const el of document.querySelectorAll(".viz-panel.is-pending")) el.classList.remove("is-pending");
            return;
        }

        // Both scenes show camera + light positions together. They differ only
        // in which step of the transform chain they freeze:
        //   • base1  — physical capture frame (light brought into base1 via
        //              base2_to_base1; turntable rotation NOT undone)
        //   • world0 — full chain (above PLUS turntable rotation undone)
        // De-duplicate per camera_id / light_id so the scatter shows one dot
        // per unique position (otherwise we'd plot 500+ coincident dots).
        const camB1 = new Map(), lightB1 = new Map(), camW0 = new Map();
        for (const e of validEntries) {
            if (!camB1.has(e.camera_id))  camB1.set(e.camera_id,  e.cam_pos_b1);
            if (!lightB1.has(e.light_id)) lightB1.set(e.light_id, e.light_pos_b1);
            if (!camW0.has(e.camera_id))  camW0.set(e.camera_id,  e.cam_pos_w0);
        }
        // For world0 the light positions sweep across all turn angles (one dot
        // per (light_id, turn_angle) pair), since the turntable undo depends
        // on turn_angle.
        const lightW0All = validEntries.map(e => e.light_pos_w0);

        // Turntable disc: the cloth swatch sits on it; half the largest XY
        // dim of the sample is a reasonable scale. The disc's XY is the
        // calibrated rotation axis; viz.js reads the Z as the TOP SURFACE of
        // the disc, lined up with bbox_min[2] so the slab sits on it.
        let sampleHalf = 0.075;
        if (crop && crop.bbox_size) sampleHalf = Math.max(crop.bbox_size[0], crop.bbox_size[1]) / 2;
        const ttc = CONFIG.CALIB.TURNTABLE_CENTER;
        const discTopZ = (materialBBox && materialBBox.bbox_min) ? materialBBox.bbox_min[2] : ttc[2];
        const discCenter = [ttc[0], ttc[1], discTopZ];
        const discRadius = Math.max(0.22, sampleHalf * 3.0);   // physical turntable >> swatch
        const discThickness = 0.025;                           // 2.5 cm of visible body

        // One panel per task, each first drawn in its own frame: building a
        // WebGL scene and compiling its shaders is the page's longest main-
        // thread work, and slider input keeps flowing in between. (The
        // capture keeps downloading meanwhile: TarReader uses an XHR blob.)
        const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
        await nextFrame();
        try {
            b1Scene = new Viz.CombinedCaptureScene(b1Canvas, {
                cameraPositions: [...camB1.values()],
                lightPositions:  [...lightB1.values()],
                turntableCenter: discCenter,
                turntableRadius: discRadius,
                turntableThickness: discThickness,
                showTurntable:   true,
                // Material 3D bbox: stored in world0; in base1 we have to
                // rotate it back by +turn_angle around the turntable axis.
                materialBBox,
                rotateMaterialWithTurnAngle: true,
                viewAzimuthDeg: -60, viewElevationDeg: 28,
            });
            update3D(validEntries[state.currentFrame]);
            b1Canvas.parentElement.classList.remove("is-pending");
        } catch (err) {
            console.warn("[viz base1]", err);
            markVizUnavailable(b1Canvas, err, false);
            b1Scene = null;
        }
        await nextFrame();
        await nextFrame();
        await quietMoment();
        try {
            w0Scene = new Viz.CombinedCaptureScene(w0Canvas, {
                cameraPositions: [...camW0.values()],
                // the full sweep: with the turntable rotation undone the
                // lights form a band around the hemisphere
                lightPositions:  lightW0All,
                turntableCenter: discCenter,
                turntableRadius: discRadius,
                turntableThickness: discThickness,
                // World0: the bbox is already in this frame, draw it static.
                materialBBox,
                rotateMaterialWithTurnAngle: false,
                autoRotate: true,              // while hovered (viz.js)
                viewAzimuthDeg: 20, viewElevationDeg: 30,
            });
            update3D(validEntries[state.currentFrame]);
        } catch (err) {
            console.warn("[viz world0]", err);
            markVizUnavailable(w0Canvas, err, true);
            w0Scene = null;
        }
        window.__objViewer.b1Scene = b1Scene;
        window.__objViewer.w0Scene = w0Scene;
        for (const el of document.querySelectorAll(".viz-panel.is-pending")) el.classList.remove("is-pending");
        if (stats) colourScenes(stats);
        scenesReady = true;
        if (painted) scheduleSwatch();
    }

    // -------- helpers --------

    /**
     * Find the FIRST entry in `validEntries` whose fields match the partial
     * filter object. Falls back to the entry whose stored field is closest
     * to the requested value for `turn_angle` (float comparison).
     */
    function findEntry(arr, filter) {
        const exact = arr.findIndex(e =>
            Object.entries(filter).every(([k, v]) => e[k] === v),
        );
        if (exact >= 0) return exact;

        if (filter.turn_angle != null) {
            let best = -1, bestD = Infinity;
            arr.forEach((e, i) => {
                const d = Math.abs(e.turn_angle - filter.turn_angle);
                if (d < bestD) { bestD = d; best = i; }
            });
            return best;
        }
        return -1;
    }


    function setupNav(manifest, currentId) {
        // Prev/Next walks the full set of materials (train followed by
        // test, each block sorted by id). This lets the user step through
        // all 500 materials without going back to the index page.
        const train = manifest.train_ids || [];
        const test  = manifest.test_ids  || [];
        const ids = [...train, ...test];
        const i = ids.indexOf(currentId);
        const prev = i > 0 ? ids[i - 1] : null;
        const next = i >= 0 && i < ids.length - 1 ? ids[i + 1] : null;

        const goto = (mid) => {
            if (mid != null) location.href = `obj.html?id=${mid}`;
        };
        const prevBtn = document.getElementById("btn-prev");
        const nextBtn = document.getElementById("btn-next");
        prevBtn.disabled = prev == null;
        nextBtn.disabled = next == null;
        prevBtn.onclick = () => goto(prev);
        nextBtn.onclick = () => goto(next);

        // Train / test chip next to the title (as in the main page's lists).
        const chip = document.getElementById("split-chip");
        const split = test.includes(currentId) ? "test" : train.includes(currentId) ? "train" : null;
        if (chip && split) {
            chip.textContent = split === "test" ? "Test" : "Train";
            chip.classList.add(split);
            chip.hidden = false;
        }
    }


    function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

    function markVizUnavailable(canvas, err, withStill) {
        // No WebGL (or the viewer failed to start): the material-frame
        // panel shows a still of the viewer, both show the reason.
        const msg = err ? String(err.message || err) : "WebGL is not available in this browser";
        const box = document.createElement("div");
        box.className = "viz-fallback";
        box.innerHTML = (withStill
            ? '<img src="images/viz/fallback.png" alt="Still of the 3-D capture viewer: the swatch, '
              + 'the camera with its view frustum and the lamp with its light cone">'
            : "")
            + `<div class="viz-fallback-note">Interactive 3-D view unavailable: ${escapeHtml(msg)}.`
            + (withStill ? " Showing a still." : "")
            + " Image preview and sliders still work.</div>";
        canvas.replaceWith(box);
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
    }


    // ----------------- material info panel -----------------

    /** Fill the static fields (size, thickness) right after bbox.json loads. */
    function populateMaterialInfo() {
        if (!materialBBox || !materialBBox.bbox_size) return;
        const sx = materialBBox.bbox_size[0];
        const sy = materialBBox.bbox_size[1];
        const sz = materialBBox.bbox_size[2];
        document.getElementById("info-size").textContent =
            `${(sx*100).toFixed(1)} × ${(sy*100).toFixed(1)} cm`;
        document.getElementById("info-thickness").textContent =
            `${(sz*1000).toFixed(1)} mm`;
    }

    /** Mean colour, 16-bit linear -> 8-bit sRGB-ish ([r, g, b] 0..255). */
    function meanDisplayRGB(st) {
        // pow(x, 1/2.2) approximates sRGB encoding well enough for a
        // swatch / box colour. Without this the swatch reads very dark
        // because linear values in 0..255 look much darker than the
        // same numeric value treated as sRGB by the browser.
        const toDisplay = (v) =>
            Math.round(255 * Math.pow(Math.max(0, Math.min(1, v / 65535)), 1 / 2.2));
        return st.mean_rgb_16bit.map(toDisplay);
    }

    /** Material slab colour in both scenes follows the mean colour. */
    function colourScenes(st) {
        if (!st || !st.mean_rgb_16bit) return;
        const rgb = meanDisplayRGB(st);
        if (b1Scene && b1Scene.setMaterialColor) b1Scene.setMaterialColor(rgb);
        if (w0Scene && w0Scene.setMaterialColor) w0Scene.setMaterialColor(rgb);
    }

    /**
     * Drive the material-info panel (mean colour swatch + slab colour +
     * per-channel range bars) from the precomputed material stats.
     *
     * The bars use the 0.5 / 99.5 percentile values (lo_rgb_16bit /
     * hi_rgb_16bit), which are robust to one-pixel outliers; the
     * absolute min/max are still in the JSON but not displayed because
     * they get pulled to 0 / 65535 by very rare dark or saturated
     * pixels and aren't representative of the material.
     */
    function applyMaterialStats(st) {
        if (!st) return;
        const [mr, mg, mb8] = meanDisplayRGB(st);
        const hex = "#" + [mr, mg, mb8].map(c =>
            c.toString(16).padStart(2, "0")).join("");
        const sw = document.getElementById("color-swatch");
        const swHex = document.getElementById("color-hex");
        if (sw) sw.style.background = hex;
        if (swHex) swHex.textContent = hex;
        colourScenes(st);

        // Per-channel range bars (0..65535).
        const FULL = 65535;
        const lo = st.lo_rgb_16bit || st.min_rgb_16bit; // back-compat
        const hi = st.hi_rgb_16bit || st.max_rgb_16bit;
        const setBar = (prefix, loV, hiV) => {
            const loR = Math.round(loV);
            const hiR = Math.round(hiV);
            const left  = Math.max(0,   Math.min(100, 100 * loR / FULL));
            const width = Math.max(0.6, Math.min(100, 100 * (hiR - loR) / FULL));
            const fill = document.getElementById(prefix + "fill");
            const minL = document.getElementById(prefix + "min");
            const maxL = document.getElementById(prefix + "max");
            if (fill) {
                fill.style.left = left + "%";
                fill.style.width = width + "%";
            }
            if (minL) minL.textContent = loR.toLocaleString();
            if (maxL) maxL.textContent = hiR.toLocaleString();
        };
        setBar("r", lo[0], hi[0]);
        setBar("g", lo[1], hi[1]);
        setBar("b", lo[2], hi[2]);
    }

})();
