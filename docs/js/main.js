/**
 * Landing page logic — populate the dataset preview page from
 *   webpage/data/manifest.json.
 *
 * Manifest shape:
 *   {
 *     "preview":   [ { "id", "split": "train"|"test", "thumb": {...} }, ... ],
 *     "train_ids": [ 0, 1, 3, 5, ... ],     // ALL train material IDs (400)
 *     "test_ids":  [ 2, 4, 8, 9, ... ],     // ALL test  material IDs (100)
 *     "counts":    { "train": 400, "test": 100 }
 *   }
 *
 * Layout:
 *   - Train preview grid (20 thumbnails)
 *   - Test  preview grid (5 thumbnails)
 *   - Full ID lists below each — every material is linked, so the user can
 *     browse every one of the 500 materials without leaving the page.
 */

(async function () {
    const trainGrid     = document.getElementById("train-preview-grid");
    const testGrid      = document.getElementById("test-preview-grid");
    const trainAllList  = document.getElementById("train-all-list");
    const testAllList   = document.getElementById("test-all-list");
    const trainCountEl  = document.getElementById("train-total");
    const testCountEl   = document.getElementById("test-total");

    let manifest;
    try {
        const resp = await fetch(CONFIG.manifestURL());
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        manifest = await resp.json();
    } catch (err) {
        trainGrid.innerHTML =
            `<div class="status-line" style="grid-column:1/-1;color:#a33">
                Failed to load <code>data/manifest.json</code> — ${err.message}.
                Run <code>python3 webpage/tools/build_manifest.py</code>.
             </div>`;
        return;
    }

    const preview = manifest.preview || [];
    const trainPreview = preview.filter(e => e.split === "train");
    const testPreview  = preview.filter(e => e.split === "test");

    if (trainCountEl) trainCountEl.textContent = manifest.counts?.train ?? "—";
    if (testCountEl)  testCountEl.textContent  = manifest.counts?.test  ?? "—";

    // Build the preview thumbnail grids.
    renderPreviewGrid(trainGrid, trainPreview);
    renderPreviewGrid(testGrid,  testPreview);

    // Build the full ID lists.
    renderAllList(trainAllList, manifest.train_ids || []);
    renderAllList(testAllList,  manifest.test_ids  || []);

    // Lazy-load thumbnails (concurrency-capped — too many parallel Range
    // requests overwhelm the HF CDN).
    const allCards = [...trainGrid.children, ...testGrid.children]
        .filter(el => el._cardLoader);
    const LIMIT = 4;
    let inFlight = 0;
    const queue  = allCards.slice();
    function pump() {
        while (inFlight < LIMIT && queue.length) {
            const el = queue.shift();
            inFlight++;
            el._cardLoader()
                .catch((err) => console.warn(`[card ${el._cardId}] thumb`, err))
                .finally(() => { inFlight--; pump(); });
        }
    }
    pump();
})();


function renderPreviewGrid(container, entries) {
    container.innerHTML = "";
    if (entries.length === 0) {
        container.innerHTML =
            '<div class="status-line" style="grid-column:1/-1">No preview entries.</div>';
        return;
    }
    entries.forEach(entry => container.appendChild(buildPreviewCard(entry)));
}


/** Build a single material preview card (thumbnail + id + label). */
function buildPreviewCard(entry) {
    const a = document.createElement("a");
    a.className = "material-card";
    a.href = `obj.html?id=${entry.id}`;

    const thumb = document.createElement("div");
    thumb.className = "thumb-wrap";

    const canvas = document.createElement("canvas");
    canvas.width = 200;
    canvas.height = 200;
    thumb.appendChild(canvas);

    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";
    spinner.textContent = "…";
    thumb.appendChild(spinner);

    const body = document.createElement("div");
    body.className = "card-body";
    body.innerHTML = `
        <div class="card-id">Material ${entry.id}</div>
        <div class="card-label">${entry.split === "test" ? "Test" : "Train"}</div>
    `;

    a.appendChild(thumb);
    a.appendChild(body);

    a._cardId = entry.id;
    a._cardLoader = async () => loadThumb(entry, canvas, spinner);
    return a;
}


/** Render a compact list of all material ids as clickable pills. */
function renderAllList(container, ids) {
    container.innerHTML = "";
    ids.forEach((id) => {
        const a = document.createElement("a");
        a.className = "id-pill";
        a.href = `obj.html?id=${id}`;
        a.textContent = id;
        container.appendChild(a);
    });
}


/** Lazy-load + tone-map one thumbnail from the material's hdr.tar. */
async function loadThumb(entry, canvas, spinner) {
    const ctx = canvas.getContext("2d");

    const tarURL   = CONFIG.hfResolve(`${CONFIG.materialDir(entry.id)}/hdr.tar`);
    const indexURL = CONFIG.tarIndexURL(entry.id);

    let index;
    try {
        index = await TarReader.loadOrBuildIndex(tarURL, indexURL);
    } catch (err) {
        spinner.remove();
        drawFallback(ctx, entry.id, "no index");
        return;
    }

    const filename = (entry.thumb && entry.thumb.filename) || pickAnyPNG(index);
    if (!filename) {
        spinner.remove();
        drawFallback(ctx, entry.id, "no images");
        return;
    }

    try {
        const buf = await TarReader.fetchFile(tarURL, index, filename);
        await HDRDisplay.drawPng(buf, canvas, {
            exposure: 1.2,
            gamma: 2.2,
            maxWidth: 300,
            bbox: entry.thumb && entry.thumb.bbox,
        });
        spinner.remove();
    } catch (err) {
        spinner.remove();
        drawFallback(ctx, entry.id, "load failed");
        console.warn("[thumb]", entry.id, err);
    }
}


function pickAnyPNG(index) {
    const names = Object.keys(index).filter((n) => n.endsWith(".png"));
    if (names.length === 0) return null;
    names.sort();
    return names[0];
}


function drawFallback(ctx, id, msg) {
    const c = ctx.canvas;
    ctx.fillStyle = "#0d0f12";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#465161";
    ctx.font = "bold 32px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`#${id}`, c.width / 2, c.height / 2 - 8);
    ctx.fillStyle = "#33394a";
    ctx.font = "11px sans-serif";
    ctx.fillText(msg, c.width / 2, c.height / 2 + 18);
}
