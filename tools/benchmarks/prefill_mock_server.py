# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small OpenAI-completions mock for exercising the patched bench client."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18000
DEFAULT_MODEL_NAME = "prefill-mock"
UTF8_RESPONSE_TEXT = "你"


class PrefillMockRequestHandler(BaseHTTPRequestHandler):
    """Validate integer prompts and return one streamed completion token."""

    protocol_version = "HTTP/1.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded_payload = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_payload)))
        self.end_headers()
        self.wfile.write(encoded_payload)

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {
                        "id": DEFAULT_MODEL_NAME,
                        "root": DEFAULT_MODEL_NAME,
                        "object": "model",
                    }
                ],
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        if not isinstance(request, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
            return
        prompt = request.get("prompt")
        if not isinstance(prompt, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in prompt
        ):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "prompt must contain integer token IDs"},
            )
            return
        if request.get("max_tokens") != 1:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "max_tokens must equal one"},
            )
            return

        completion_chunk = (
            "data: "
            + json.dumps(
                {
                    "id": self.headers.get("x-request-id", "mock"),
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": UTF8_RESPONSE_TEXT}],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode()
        usage_chunk = (
            "data: "
            + json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": 1,
                        "total_tokens": len(prompt) + 1,
                    },
                }
            )
            + "\n\n"
        ).encode()
        done_chunk = b"data: [DONE]\n\n"
        response_text_bytes = UTF8_RESPONSE_TEXT.encode()
        split_index = completion_chunk.index(response_text_bytes) + 1

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(completion_chunk[:split_index])
        self.wfile.flush()
        self.wfile.write(completion_chunk[split_index:])
        self.wfile.write(usage_chunk)
        self.wfile.write(done_chunk)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, message_format: str, *arguments: object) -> None:
        return


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), PrefillMockRequestHandler)
    print(
        f"Prefill mock server listening on http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
