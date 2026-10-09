#!/usr/bin/env python3
"""Static preview server with HTTP Range support (python -m http.server has none, so
<video> seeking and the material viewer's range reads misbehave on it).
    python3 docs/tools/preview_server.py 8765 docs/ [--mount /proxies=/media/raid/cloth/output/webpage/proxies ...]
A --mount serves a directory outside the site root under a URL prefix (local copies of
data that the published page reads from Hugging Face).
"""
import os, re, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


MOUNTS = []          # [(url prefix, directory)]


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive: Safari's media loader dislikes HTTP/1.0 + close

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, directory in MOUNTS:
            if clean == prefix.rstrip("/") or clean.startswith(prefix):
                rel = os.path.normpath(clean[len(prefix):].lstrip("/"))
                if rel.startswith(".."):
                    return directory
                return os.path.join(directory, rel)
        return super().translate_path(path)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or "Range" not in self.headers:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found"); return None
        size = os.fstat(f.fileno()).st_size
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers["Range"])
        if not m:
            f.close(); return super().send_head()
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers()
            f.close(); return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(f.fileno()).st_mtime))
        self.end_headers()
        f.seek(start); self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(1 << 16, left))
            if not chunk: break
            outputfile.write(chunk); left -= len(chunk)
        self._range_left = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--mount")]
    for i, a in enumerate(sys.argv[1:]):
        if a == "--mount":
            prefix, directory = sys.argv[i + 2].split("=", 1)
            MOUNTS.append((prefix if prefix.endswith("/") else prefix + "/", os.path.abspath(directory)))
            args.remove(sys.argv[i + 2])
        elif a.startswith("--mount="):
            prefix, directory = a[len("--mount="):].split("=", 1)
            MOUNTS.append((prefix if prefix.endswith("/") else prefix + "/", os.path.abspath(directory)))
    port = int(args[0]) if len(args) > 0 else 8765
    root = args[1] if len(args) > 1 else "."
    os.chdir(root)
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
