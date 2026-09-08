"""
Local review server for sarab_full_candidate_pool.json.

Serves the review UI (index.html/style.css/app.js), the object images
referenced by each entry's local_path, and a small JSON API the page uses to
read the pool and save review flags/edits back to disk.

Usage (run from pipeline/review_ui/):
    python3 server.py [--port 8765]
"""
import argparse
import json
import mimetypes
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def _resolve_data_root():
    """Resolve Sarab's data directory: SARAB_DATA_DIR env override first
    (point it at a local checkout while developing), then a sibling
    Sarab-Dataset-HF/ directory next to this repo, then download and cache
    the dataset from the Hugging Face Hub."""
    override = os.environ.get("SARAB_DATA_DIR")
    if override:
        return Path(override)
    sibling = REPO_ROOT.parent / "Sarab-Dataset-HF"
    if sibling.exists():
        return sibling
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id="HassanB4/sarab", repo_type="dataset"))


DATA_ROOT = _resolve_data_root()
POOL = DATA_ROOT / "pipeline" / "sarab_full_candidate_pool.json"
BACKUP = DATA_ROOT / "pipeline" / "backups" / "sarab_full_candidate_pool.before_review.json"

STATIC_FILES = {
    "/": HERE / "index.html",
    "/index.html": HERE / "index.html",
    "/app.js": HERE / "app.js",
    "/style.css": HERE / "style.css",
}

EDITABLE_FIELDS = {"caption_ar", "h_item_ar"}


def ensure_backup():
    if not BACKUP.exists():
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(POOL, BACKUP)
        print(f"Backed up original pool to {BACKUP.relative_to(DATA_ROOT)}")


def load_pool():
    return json.loads(POOL.read_text(encoding="utf-8"))


def save_pool(pool):
    tmp = POOL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(POOL)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console quiet; errors still raise

    def _send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                          "application/json; charset=utf-8", status)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path in STATIC_FILES:
            f = STATIC_FILES[path]
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send_bytes(f.read_bytes(), ctype)
            return

        if path == "/api/pool":
            self._send_json(load_pool())
            return

        if path.startswith("/data/"):
            target = (DATA_ROOT / path.lstrip("/")).resolve()
            if DATA_ROOT not in target.parents or not target.is_file():
                self._send_json({"error": "not found"}, 404)
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send_bytes(target.read_bytes(), ctype)
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/pool":
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON body"}, 400)
            return

        image_id = body.get("image_id")
        review_status = body.get("review_status")
        if not image_id or review_status not in {"correct", "rejected", "edited"}:
            self._send_json({"error": "image_id and a valid review_status are required"}, 400)
            return

        pool = load_pool()
        entry = next((e for e in pool if e.get("image_id") == image_id), None)
        if entry is None:
            self._send_json({"error": f"unknown image_id {image_id!r}"}, 404)
            return

        entry["review_status"] = review_status
        for field in EDITABLE_FIELDS:
            if field in body:
                entry[field] = body[field]

        save_pool(pool)
        self._send_json({"ok": True, "entry": entry})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not POOL.exists():
        raise SystemExit(f"pool file not found: {POOL}")
    ensure_backup()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review UI running at http://127.0.0.1:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
