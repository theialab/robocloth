/**
 * TarReader — fetch individual files out of a remote ustar archive using
 * HTTP Range requests.
 *
 * The HuggingFace CDN supports Range and CORS for our dataset, so we can
 * pull a single PNG (a few MB) out of a multi-GB tar without downloading
 * the whole thing.
 *
 * Strategy
 * --------
 *   1. Build (or load) an index mapping filename -> [offset, size].
 *      Building it scans only the tar's 512-byte headers and skips the
 *      data blocks via Range.
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
 * Notes
 * -----
 *   - Numbers in JSON can fit 5GB exactly (JS Number = float64, safe up to
 *     2^53). All HDR tars are under that.
 *   - We support both UStar plain-name (100 bytes in header) and the
 *     "ustar long-name" prefix (155 bytes prefix + slash + name), which
 *     is what BSD/GNU tar typically uses for paths > 100 chars.
 *   - We DO NOT support pax extended headers, since the upload pipeline
 *     uses a plain ustar archive (names are short like
 *     `scan-0000_light-0_camera-0.png`).
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

    /**
     * Pull bytes [start, end) (inclusive end-exclusive) from a remote URL.
     * Returns ArrayBuffer.
     */
    async function rangeFetch(url, start, end) {
        const resp = await fetch(url, {
            headers: { Range: `bytes=${start}-${end - 1}` },
            // Force fresh CORS preflight, allow credentials never
            credentials: "omit",
        });
        if (!resp.ok && resp.status !== 206) {
            throw new Error(`Range fetch failed: HTTP ${resp.status} for ${url}`);
        }
        return resp.arrayBuffer();
    }

    /**
     * Build a {filename: [offset, size]} index by walking the tar's headers.
     *
     * This issues O(N) small Range requests, one per file header. Browsers
     * pipeline these well; for ~500 entries it is usually a few seconds the
     * first time. After that, the index can be cached in localStorage.
     */
    async function buildIndex(url, { totalSize = null, onProgress = null } = {}) {
        const index = {};
        let cursor = 0;
        // Two-block scan window: header (512) plus a slack to amortise (8 KB).
        // We don't need data here, just the headers.
        let zeroBlocks = 0;

        while (true) {
            const buf = new Uint8Array(
                await rangeFetch(url, cursor, cursor + TAR_BLOCK),
            );
            if (buf.length < TAR_BLOCK) break; // EOF

            // Two consecutive zero blocks mark end of archive.
            let allZero = true;
            for (let i = 0; i < TAR_BLOCK; i++) {
                if (buf[i] !== 0) { allZero = false; break; }
            }
            if (allZero) {
                zeroBlocks += 1;
                cursor += TAR_BLOCK;
                if (zeroBlocks >= 2) break;
                continue;
            }
            zeroBlocks = 0;

            const name = parseString(buf, 0, 100);
            const size = parseOctal(buf, 124, 12);
            const typeflag = String.fromCharCode(buf[156] || 0x30);
            const prefix = parseString(buf, 345, 155);
            const fullName = prefix ? `${prefix}/${name}` : name;

            // Only record regular files. Pax extended headers ('x', 'g'),
            // GNU long-link ('L', 'K'), directories ('5'), and other
            // metadata blocks are skipped — but we still have to advance
            // past their data section.
            if (size > 0 && (typeflag === "0" || typeflag === "\0")) {
                // Also store a basename-keyed alias so callers using the
                // scan-log filenames (which have no `hdr/` prefix) can
                // look them up directly without a fallback scan.
                index[fullName] = [cursor + TAR_BLOCK, size];
                const base = fullName.split("/").pop();
                if (base !== fullName && !(base in index)) {
                    index[base] = [cursor + TAR_BLOCK, size];
                }
            }

            const padded = Math.ceil(size / TAR_BLOCK) * TAR_BLOCK;
            cursor += TAR_BLOCK + padded;

            if (onProgress) onProgress(cursor, totalSize);
            if (totalSize !== null && cursor >= totalSize) break;
        }
        return index;
    }

    /**
     * Get a single file's bytes from a remote tar, given a pre-built index.
     * Returns ArrayBuffer. Caches successful fetches in memory by `url + filename`
     * so reselecting the same frame doesn't re-hit the CDN.
     */
    async function fetchFile(url, index, filename) {
        const cacheKey = `${url}::${filename}`;
        if (_fileCache.has(cacheKey)) {
            // Return a structured clone (ArrayBuffer can't be reused across
            // callers in some paths, but createImageBitmap is fine with a
            // shared Blob — we keep the original and return a view).
            return _fileCache.get(cacheKey);
        }

        const hit = index[filename];
        let off, sz;
        if (hit) {
            [off, sz] = hit;
        } else {
            const base = filename.split("/").pop();
            const hit2 = Object.entries(index).find(
                ([k]) => k.split("/").pop() === base,
            );
            if (!hit2) throw new Error(`Tar index miss: ${filename}`);
            [off, sz] = hit2[1];
        }
        const buf = await rangeFetch(url, off, off + sz);

        // Bound the cache so a long viewing session doesn't grow without
        // limit. 16 entries × ~5 MB = ~80 MB cap, well within browser memory.
        if (_fileCache.size >= 16) {
            const firstKey = _fileCache.keys().next().value;
            _fileCache.delete(firstKey);
        }
        _fileCache.set(cacheKey, buf);
        return buf;
    }

    /** In-memory cache for tar indexes by url. */
    const _indexCache = new Map();

    /** In-memory cache for raw PNG bytes by `tarUrl::filename`. */
    const _fileCache = new Map();

    /** Drop everything (handy when switching materials). */
    function clearFileCache() { _fileCache.clear(); }

    /**
     * Speculatively warm the cache for a list of filenames. Concurrency is
     * capped so we don't drown the CDN; callers can await this or fire it
     * and forget. Errors are swallowed because prefetch is best-effort.
     */
    async function prefetchFiles(url, index, filenames, concurrency = 2) {
        let i = 0;
        async function worker() {
            while (i < filenames.length) {
                const fn = filenames[i++];
                try { await fetchFile(url, index, fn); }
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
        buildIndex, fetchFile, loadOrBuildIndex, rangeFetch,
        prefetchFiles, clearFileCache,
    };
})();
