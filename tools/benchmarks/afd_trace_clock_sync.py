# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Measure cross-host monotonic-clock offsets for AFD trace correlation.

Run ``server`` on one trace host and ``client`` on every other host shortly
before or after profiling. The client stores all four timestamps for each
round trip; the merge tool selects the minimum-round-trip sample and reports
its uncertainty instead of assuming host clocks are identical.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Final, TextIO

DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 29610
DEFAULT_SAMPLES: Final[int] = 16
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
MAX_MESSAGE_BYTES: Final[int] = 64 * 1024
CLOCK_SYNC_SCHEMA_VERSION: Final[int] = 1


def monotonic_raw_ns() -> int:
    """Read the same host-local clock used by the correlation recorder."""

    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if raw_clock is None:
        return time.monotonic_ns()
    return time.clock_gettime_ns(raw_clock)


def run_server(args: argparse.Namespace) -> None:
    """Serve a bounded number of clock-sampling clients."""

    completed_clients: list[dict[str, object]] = []
    with socket.create_server((args.host, args.port)) as server:
        server.settimeout(args.timeout)
        while len(completed_clients) < args.clients:
            connection, address = server.accept()
            with connection:
                connection.settimeout(args.timeout)
                input_file = connection.makefile("r", encoding="utf-8")
                output_file = connection.makefile("w", encoding="utf-8")
                completed_clients.append(
                    _serve_client(
                        input_file,
                        output_file,
                        session_id=args.session_id,
                        address=address,
                    ),
                )

    if args.output is not None:
        _write_json(
            args.output,
            {
                "schema_version": CLOCK_SYNC_SCHEMA_VERSION,
                "record_type": "clock_sync_server",
                "session_id": args.session_id,
                "reference_host": socket.gethostname(),
                "clients": completed_clients,
            },
        )


def _serve_client(
    input_file: TextIO,
    output_file: TextIO,
    *,
    session_id: str,
    address: tuple[str, int],
) -> dict[str, object]:
    sample_count = 0
    client_host = address[0]
    while True:
        request = _read_json_line(input_file)
        request_type = request.get("type")
        if request.get("session_id") != session_id:
            raise ValueError("clock-sync client and server session IDs differ")
        client_host = str(request.get("client_host", client_host))
        if request_type == "done":
            return {"client_host": client_host, "samples": sample_count}
        if request_type != "sample":
            raise ValueError(f"unsupported clock-sync message: {request_type!r}")

        server_receive_ns = monotonic_raw_ns()
        response = {
            "type": "sample",
            "session_id": session_id,
            "reference_host": socket.gethostname(),
            "server_receive_ns": server_receive_ns,
            "server_send_ns": monotonic_raw_ns(),
        }
        _write_json_line(output_file, response)
        output_file.flush()
        sample_count += 1


def run_client(args: argparse.Namespace) -> None:
    """Collect NTP-style four-timestamp samples from the reference host."""

    client_host = socket.gethostname()
    samples: list[dict[str, int]] = []
    reference_host: str | None = None
    with socket.create_connection(
        (args.server, args.port),
        timeout=args.timeout,
    ) as connection:
        connection.settimeout(args.timeout)
        input_file = connection.makefile("r", encoding="utf-8")
        output_file = connection.makefile("w", encoding="utf-8")
        for _ in range(args.samples):
            client_send_ns = monotonic_raw_ns()
            _write_json_line(
                output_file,
                {
                    "type": "sample",
                    "session_id": args.session_id,
                    "client_host": client_host,
                },
            )
            output_file.flush()
            response = _read_json_line(input_file)
            client_receive_ns = monotonic_raw_ns()
            if response.get("session_id") != args.session_id:
                raise ValueError("clock-sync response has the wrong session ID")
            reference_host = str(response["reference_host"])
            samples.append(
                {
                    "client_send_ns": client_send_ns,
                    "server_receive_ns": int(response["server_receive_ns"]),
                    "server_send_ns": int(response["server_send_ns"]),
                    "client_receive_ns": client_receive_ns,
                },
            )
        _write_json_line(
            output_file,
            {
                "type": "done",
                "session_id": args.session_id,
                "client_host": client_host,
            },
        )
        output_file.flush()

    _write_json(
        args.output,
        {
            "schema_version": CLOCK_SYNC_SCHEMA_VERSION,
            "record_type": "clock_sync_client",
            "session_id": args.session_id,
            "client_host": client_host,
            "reference_host": reference_host,
            "samples": samples,
        },
    )


def _read_json_line(input_file: TextIO) -> dict[str, object]:
    line = input_file.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        raise ConnectionError("clock-sync peer closed the connection")
    if len(line) > MAX_MESSAGE_BYTES:
        raise ValueError("clock-sync message exceeds the size limit")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("clock-sync message must be a JSON object")
    return payload


def _write_json_line(output_file: TextIO, payload: dict[str, object]) -> None:
    output_file.write(json.dumps(payload, separators=(",", ":")))
    output_file.write("\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="run the reference endpoint")
    server.add_argument("--session-id", required=True)
    server.add_argument("--host", default=DEFAULT_HOST)
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument("--clients", type=int, required=True)
    server.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    server.add_argument("--output", type=Path)
    server.set_defaults(function=run_server)

    client = subparsers.add_parser("client", help="sample the reference endpoint")
    client.add_argument("--session-id", required=True)
    client.add_argument("--server", required=True)
    client.add_argument("--port", type=int, default=DEFAULT_PORT)
    client.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    client.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    client.add_argument("--output", type=Path, required=True)
    client.set_defaults(function=run_client)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.command == "server" and args.clients <= 0:
        raise ValueError("--clients must be positive")
    if args.command == "client" and args.samples <= 0:
        raise ValueError("--samples must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
