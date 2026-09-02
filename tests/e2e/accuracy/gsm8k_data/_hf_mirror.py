"""Minimal local HF Hub mirror for offline gsm8k dataset loading.

Serves the gsm8k repo files and synthetic /api/datasets responses so that
``datasets.load_dataset("openai/gsm8k", "main")`` resolves against a local
endpoint instead of the real Hub. Point ``HF_ENDPOINT`` at it.
"""

from __future__ import annotations

import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SHA = "740312add88f781978c0658806c59bc2815b9866"

FILE_RELPATHS = [
    "README.md",
    "eval.yaml",
    ".gitattributes",
    "main/train-00000-of-00001.parquet",
    "main/test-00000-of-00001.parquet",
]

SIBLINGS = [{"rfilename": rel} for rel in FILE_RELPATHS]


def _tree_entries() -> list[dict]:
    entries = [{"type": "dir", "path": "main", "oid": SHA}]
    for rel in FILE_RELPATHS:
        full = os.path.join(REPO_DIR, rel)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        oid = hashlib.sha256(rel.encode()).hexdigest()
        entries.append({"type": "file", "path": rel, "size": size, "oid": oid})
    return entries


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Repo-Commit", SHA)
        self.end_headers()
        self.wfile.write(body)

    def _handle_file(self, head_only: bool) -> None:
        path = self.path.split("?", 1)[0]
        filename = path.split("/resolve/", 1)[1].split("/", 1)[1]
        filepath = os.path.join(REPO_DIR, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as fh:
                body = fh.read()
            # huggingface_hub requires an ETag on resolve responses; without
            # it hf_hub_download aborts with LocalEntryNotFoundError before
            # ever issuing the GET (observed with hub 0.36.x).
            etag = hashlib.sha256(body).hexdigest()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Repo-Commit", SHA)
            self.send_header("ETag", f'"{etag}"')
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
        else:
            self._respond(404, b"not found", "text/plain")

    def _handle_api(self) -> None:
        path = self.path.split("?", 1)[0]
        if "/tree/" in path:
            body = json.dumps(_tree_entries()).encode()
        else:
            info = {
                "id": "openai/gsm8k",
                "sha": SHA,
                "siblings": SIBLINGS,
                "private": False,
                "gated": False,
                "downloads": 0,
                "likes": 0,
            }
            body = json.dumps(info).encode()
        self._respond(200, body, "application/json")

    def do_HEAD(self) -> None:
        if "/resolve/" in self.path:
            self._handle_file(head_only=True)
        else:
            self._respond(404, b"", "text/plain")

    def do_GET(self) -> None:
        if self.path.startswith("/api/datasets/"):
            self._handle_api()
        elif "/resolve/" in self.path:
            self._handle_file(head_only=False)
        else:
            self._respond(404, b"not found", "text/plain")

    def log_message(self, *args):  # log requests for offline-flow debugging
        import sys

        print(f"[hf-mirror] {args[0] % args[1:]}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("HF_MIRROR_PORT", "8000"))
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
