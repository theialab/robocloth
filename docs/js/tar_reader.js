/**
 * TarReader — fetch individual files out of a remote ustar archive using
 * HTTP Range requests.
 *
 * The Hugging Face CDN supports Range and CORS for our dataset, so we can
 * pull a single PNG (a few MB) out of a multi-GB tar without downloading
 * the whole thing.
 *
 * Strategy
 * --------
 *   1. Build (or load) an index mapping filename -> [offset, size].
 *      Building it scans only the tar's headers and skips the data blocks
 *      via Range.
 *   2. To fetch a file, look up [offset, size] and issue a single Range
 *      request for that byte slice.
 *
 * Index format (JSON):
 *   {
 *     "filename1.png": [offsetBytes, sizeBytes],
 *     "filename2.png": [...],
 *     ...
 *   }
 *
 * Request hygiene (the per-material viewer drags through ~580 captures of
 * 4-14 MB each, so this matters):
 *   - huggingface.co/.../resolve/... answers every request with a 302 to a
 *     signed CDN URL (valid ~1 h). Each resolve costs a round trip and counts
 *     against HF's "resolvers" quota (3000 / 5 min per IP), so the final URL
 *     (Response.url) is remembered and later Range requests go straight to
 *     the CDN until shortly before it expires (or fails, then we re-resolve).
 *   - fetchFile() takes an AbortSignal; concurrent requests for the same file
 *     share one download, which is cancelled only when every caller aborted.
 *   - Fetched files are kept as Blobs in a small LRU (bounded by bytes).
 *
 * Notes
 * -----
 *   - Numbers in JSON can fit 5GB exactly (JS Number = float64, safe up to
 *     2^53). All HDR tars are under that.
 *   - We support both UStar plain-name (100 bytes in header) and the
 *     "ustar long-name" prefix (155 bytes prefix + slash + name). pax
 *     extended headers ('x', as written by Python's tarfile for the float
 *     mtime) are skipped; their contents are not needed because the names
 *     are short (`hdr/scan-0000_light-0_camera-0.png`).
 */

(() => {
    const TAR_BLOCK = 512;

    /** Parse an octal number from a NUL-padded ASCII byte slice. */
    function parseOctal(view, start, len) {
        let s = "";
        for (let i = 0; i < len; i++) {
            const c = view[start + i];
            if (c === 0 || c === 32) break;
            s += String.fromCharCode(c);
        }
        return s === "" ? 0 : parseInt(s, 8);
    }

    /** Read a NUL-terminated string out of a byte slice. */
    function parseString(view, start, len) {
        let s = "";
        for (let i = 0; i < len; i++) {
            const c = view[start + i];
            if (c === 0) break;
            s += String.fromCharCode(c);
        }
        return s;
    }

    // -----------------  resolved (CDN) URL cache  -----------------

    const _resolved = new Map();      // resolve URL -> { url, expiresAt }

    function _expiryOf(u) {
        try {
            const e = new URL(u).searchParams.get("Expires");
            if (e && /^\d+$/.test(e)) return Number(e) * 1000;
        } catch (err) { /* not a URL we understand */ }
        return Date.now() + 20 * 60 * 1000;
    }

    /** The URL to range-read right now: the cached CDN URL while valid. */
    function _target(url) {
        const r = _resolved.get(url);
        if (r && Date.now() < r.expiresAt - 120 * 1000) return r.url;
        if (r) _resolved.delete(url);
        return url;
    }

    const _abortError = () => new DOMException("Aborted", "AbortError");
    const isAbort = (err) => !!err && err.name === "AbortError";

    /**
     * Range GET into a Blob with XMLHttpRequest. The body goes straight into
     * the browser's blob store instead of being pumped through this page's
     * main thread (as a fetch() body, or a tee of it for progress, is): a busy
     * main thread (3-D panels starting, a decode, GC) then cannot stall the
     * transfer, and progress events come for free. Measured with a 1.5 s
     * main-thread block during an 8 MB read at 50 Mbit/s: fetch + tee 2.8 s,
     * XHR 1.9 s (1.6 s unblocked).
     * Resolves { status, url, body }; a non-206 answer is aborted as soon as
     * its headers arrive (a 200 would be the whole multi-GB tar).
     */
    function _xhrRange(tgt, start, end, signal, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            let badStatus = 0;
            const onAbort = () => xhr.abort();
            const settle = (f, v) => {
                if (signal) signal.removeEventListener("abort", onAbort);
                f(v);
            };
            xhr.open("GET", tgt, true);
            xhr.responseType = "blob";
            xhr.setRequestHeader("Range", `bytes=${start}-${end - 1}`);
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 2 && xhr.status !== 206) { badStatus = xhr.status; xhr.abort(); }
            };
            if (onProgress) xhr.onprogress = (ev) => onProgress(ev.loaded, end - start);
            xhr.onload = () => settle(resolve, { status: xhr.status, url: xhr.responseURL, body: xhr.response });
            xhr.onerror = () => settle(reject, new TypeError(`Network error for ${tgt.split("?")[0]}`));
            xhr.onabort = () => settle(badStatus ? resolve : reject,
                badStatus ? { status: badStatus, url: xhr.responseURL, body: null } : _abortError());
            if (signal) {
                if (signal.aborted) { reject(_abortError()); return; }
                signal.addEventListener("abort", onAbort);
            }
            xhr.send();
        });
    }

    /**
     * Pull bytes [start, end) (end-exclusive) from a remote URL.
     *
     * opts: { signal, priority: "high"|"low"|"auto" (fetch path only),
     *         as: "arraybuffer" (default, fetch) | "blob" (XMLHttpRequest),
     *         onProgress(received, total) }
     */
    async function rangeFetch(url, start, end, opts = {}) {
        const { signal, priority, as, onProgress } = opts;
        const tgt = _target(url);
        const retry = () => { _resolved.delete(url); return rangeFetch(url, start, end, opts); };
        if (as === "blob" && typeof XMLHttpRequest !== "undefined") {
            let r;
            try {
                r = await _xhrRange(tgt, start, end, signal, onProgress);
            } catch (err) {
                if (tgt !== url && !isAbort(err) && !(signal && signal.aborted)) return retry();
                throw err;
            }
            if (r.status !== 206) {
                const ignoredRange = r.status >= 200 && r.status < 300;
                if (tgt !== url && !ignoredRange) return retry();
                throw new Error(ignoredRange
                    ? `Range not honoured (HTTP ${r.status}) for ${url}`
                    : `Range fetch failed: HTTP ${r.status} for ${url}`);
            }
            if (tgt === url && r.url && r.url !== url) {
                _resolved.set(url, { url: r.url, expiresAt: _expiryOf(r.url) });
            }
            return r.body;
        }
        let resp;
        try {
            resp = await fetch(tgt, {
                headers: { Range: `bytes=${start}-${end - 1}` },
                credentials: "omit",
                signal,
                priority: priority || "auto",
            });
        } catch (err) {
            // An expired / unreachable CDN URL: go through the resolver again.
            if (tgt !== url && !isAbort(err) && !(signal && signal.aborted)) return retry();
            throw err;
        }
        if (resp.status !== 206) {
            const ignoredRange = resp.ok;      // 200 = the whole multi-GB tar
            try { resp.body && resp.body.cancel(); } catch (err) { /* ignore */ }
            if (tgt !== url && !ignoredRange) return retry();
            throw new Error(ignoredRange
                ? `Range not honoured (HTTP ${resp.status}) for ${url}`
                : `Range fetch failed: HTTP ${resp.status} for ${url}`);
        }
        if (tgt === url && resp.url && resp.url !== url) {
            _resolved.set(url, { url: resp.url, expiresAt: _expiryOf(resp.url) });
        }
        // Count bytes on a tee of the body (cloned before it is consumed):
        // the payload itself is still collected natively (assembling ~10 MB
        // of chunks in JS cost a ~70 ms task on a phone-class CPU).
        const counter = onProgress && resp.body && typeof resp.clone === "function"
            ? resp.clone() : null;
        const body = as === "blob" ? resp.blob() : resp.arrayBuffer();
        if (counter) {
            const want = end - start;
            (async () => {
                try {
                    const reader = counter.body.getReader();
                    let got = 0;
                    for (;;) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        got += value.length;
                        onProgress(got, want);
                    }
                } catch (err) { /* aborted / failed: `body` reports it */ }
            })();
        }
        return body;
    }

    /**
     * Build a {filename: [offset, size]} index by walking the tar's headers.
     *
     * One Range request per member: a 4 KB window at each member start covers
     * its pax header + pax data + ustar header, so the walk costs ~1 round
     * trip per file (~580 per material) and goes straight to the CDN after
     * the first request. It is still sequential (each header gives the next
     * offset), i.e. tens of seconds, so callers can pass:
     *   into     - an object to fill progressively (readable while walking)
     *   onEntry  - (name, offset, size) for every regular file found
     *   onProgress(cursor, totalSize)
     *   signal   - AbortSignal
     */
    async function buildIndex(url, opts = {}) {
        const { totalSize = null, onProgress = null, onEntry = null, signal } = opts;
        const index = opts.into || {};
        const WINDOW = 4096;
        let cursor = 0;
        let zeroBlocks = 0;
        let buf = null, bufStart = 0;

        const add = (name, off, size) => {
            index[name] = [off, size];
            const base = name.split("/").pop();
            // Also store a basename-keyed alias so callers using the scan-log
            // filenames (which have no `hdr/` prefix) can look them up.
            if (base !== name && !(base in index)) index[base] = [off, size];
            if (onEntry) onEntry(base, off, size, name);
        };

        while (true) {
            if (signal && signal.aborted) throw _abortError();
            if (!buf || cursor < bufStart || cursor + TAR_BLOCK > bufStart + buf.length) {
                bufStart = cursor;
                buf = new Uint8Array(await rangeFetch(url, cursor, cursor + WINDOW, { signal }));
                if (buf.length < TAR_BLOCK) break;  // EOF
            }
            const p = cursor - bufStart;

            // Two consecutive zero blocks mark end of archive.
            let allZero = true;
            for (let i = 0; i < TAR_BLOCK; i++) {
                if (buf[p + i] !== 0) { allZero = false; break; }
            }
            if (allZero) {
                zeroBlocks += 1;
                cursor += TAR_BLOCK;
                if (zeroBlocks >= 2) break;
                continue;
            }
            zeroBlocks = 0;

            const name = parseString(buf, p, 100);
            const size = parseOctal(buf, p + 124, 12);
            const typeflag = String.fromCharCode(buf[p + 156] || 0x30);
            const prefix = parseString(buf, p + 345, 155);
            const fullName = prefix ? `${prefix}/${name}` : name;

            // Only record regular files. Pax extended headers ('x', 'g'),
            // GNU long-link ('L', 'K'), directories ('5'), and other
            // metadata blocks are skipped — but we still have to advance
            // past their data section.
            if (size > 0 && (typeflag === "0" || typeflag === "\0")) {
                add(fullName, cursor + TAR_BLOCK, size);
            }

            const padded = Math.ceil(size / TAR_BLOCK) * TAR_BLOCK;
            cursor += TAR_BLOCK + padded;

            if (onProgress) onProgress(cursor, totalSize);
            if (totalSize !== null && cursor >= totalSize) break;
        }
        return index;
    }

    // -----------------  file cache (LRU, Blobs)  -----------------

    const CACHE_MAX_BYTES = 64 * 1024 * 1024;
    const CACHE_MAX_ENTRIES = 8;
    const _fileCache = new Map();     // `tarUrl::filename` -> Blob
    let _cacheBytes = 0;

    function _cacheGet(key) {
        const v = _fileCache.get(key);
        if (v) { _fileCache.delete(key); _fileCache.set(key, v); }   // most recent last
        return v;
    }
    function _cachePut(key, blob) {
        if (_fileCache.has(key)) { _cacheBytes -= _fileCache.get(key).size; _fileCache.delete(key); }
        _fileCache.set(key, blob);
        _cacheBytes += blob.size;
        while (_fileCache.size > 1
               && (_cacheBytes > CACHE_MAX_BYTES || _fileCache.size > CACHE_MAX_ENTRIES)) {
            const k = _fileCache.keys().next().value;
            _cacheBytes -= _fileCache.get(k).size;
            _fileCache.delete(k);
        }
    }

    /** Drop everything (handy when switching materials). */
    function clearFileCache() { _fileCache.clear(); _cacheBytes = 0; }

    function _lookup(index, filename) {
        const hit = index[filename];
        if (hit) return hit;
        const base = filename.split("/").pop();
        if (index[base]) return index[base];
        const hit2 = Object.entries(index).find(([k]) => k.split("/").pop() === base);
        if (!hit2) throw new Error(`Tar index miss: ${filename}`);
        return hit2[1];
    }

    /** Downloads in flight: key -> { promise, ctl, refs, listeners }. */
    const _inflight = new Map();

    function _join(job, signal, onProgress) {
        job.refs++;
        if (onProgress) job.listeners.add(onProgress);
        return new Promise((resolve, reject) => {
            let done = false;
            const release = () => {
                if (done) return false;
                done = true;
                job.refs--;
                if (onProgress) job.listeners.delete(onProgress);
                if (signal) signal.removeEventListener("abort", onAbort);
                return true;
            };
            const onAbort = () => {
                if (!release()) return;
                if (job.refs === 0) job.ctl.abort();   // nobody wants it any more
                reject(_abortError());
            };
            if (signal) {
                if (signal.aborted) { onAbort(); return; }
                signal.addEventListener("abort", onAbort);
            }
            job.promise.then(
                (v) => { if (release()) resolve(v); },
                (e) => { if (release()) reject(e); },
            );
        });
    }

    /**
     * Get a single file's bytes (as a Blob) from a remote tar, given an index.
     *
     * opts: { signal, priority, onProgress(received, total) }
     * Re-selecting a cached file is free; a file already downloading is
     * joined instead of requested twice. Aborting `signal` rejects with an
     * AbortError and cancels the download once no other caller wants it.
     */
    function fetchFile(url, index, filename, opts = {}) {
        const key = `${url}::${filename}`;
        const cached = _cacheGet(key);
        if (cached) return Promise.resolve(cached);
        let job = _inflight.get(key);
        if (!job) {
            let off, sz;
            try { [off, sz] = _lookup(index, filename); }
            catch (err) { return Promise.reject(err); }
            const ctl = new AbortController();
            job = { ctl, refs: 0, listeners: new Set(), size: sz };
            job.promise = rangeFetch(url, off, off + sz, {
                signal: ctl.signal,
                priority: opts.priority,
                as: "blob",
                onProgress: (g, t) => job.listeners.forEach((f) => f(g, t)),
            }).then((blob) => { _cachePut(key, blob); return blob; })
              .finally(() => { if (_inflight.get(key) === job) _inflight.delete(key); });
            job.promise.catch(() => { /* reported to the joined callers */ });
            _inflight.set(key, job);
        }
        return _join(job, opts.signal, opts.onProgress);
    }

    /** True if `filename` is cached or already downloading. */
    function hasFile(url, filename) {
        const key = `${url}::${filename}`;
        return _fileCache.has(key) || _inflight.has(key);
    }

    /** In-memory cache for tar indexes by url. */
    const _indexCache = new Map();

    /**
     * Speculatively warm the cache for a list of filenames. Concurrency is
     * capped so we don't drown the CDN; callers can await this or fire it
     * and forget. Errors are swallowed because prefetch is best-effort.
     */
    async function prefetchFiles(url, index, filenames, concurrency = 2, opts = {}) {
        let i = 0;
        async function worker() {
            while (i < filenames.length) {
                if (opts.signal && opts.signal.aborted) return;
                const fn = filenames[i++];
                try { await fetchFile(url, index, fn, { signal: opts.signal, priority: "low" }); }
                catch (e) { /* best-effort */ }
            }
        }
        await Promise.all(
            Array.from({ length: concurrency }, () => worker()),
        );
    }

    /**
     * Try to load a precomputed index from a JSON URL; on 404, build it
     * from the live tar by scanning headers.
     */
    async function loadOrBuildIndex(tarUrl, jsonIndexUrl, opts = {}) {
        if (_indexCache.has(tarUrl)) return _indexCache.get(tarUrl);

        if (jsonIndexUrl) {
            try {
                const r = await fetch(jsonIndexUrl);
                if (r.ok) {
                    const data = await r.json();
                    _indexCache.set(tarUrl, data);
                    return data;
                }
            } catch (e) { /* fallthrough to build */ }
        }

        const idx = await buildIndex(tarUrl, opts);
        _indexCache.set(tarUrl, idx);
        return idx;
    }

    window.TarReader = {
        buildIndex, fetchFile, hasFile, loadOrBuildIndex, rangeFetch,
        prefetchFiles, clearFileCache, isAbort,
    };
})();
