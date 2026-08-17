# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Send a prefix-cache precondition dataset sequentially to a warm server.

Used by the Stage-2 steady flow: POST each precondition request (a per-group
block-aligned shared prefix) and wait for its completion, so the server's
prefix cache is populated before the formal run. The requests are sent strictly
in dataset order (groups ascending) so the send order matches the manifest.

Usage:
  python3 -m tools.benchmarks.send_precondition \
    --base-url http://127.0.0.1:8000 \
    --dataset tools/datasets/prefix_precondition_prefix50.jsonl \
    --served-model-name deepseek_v3_2
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

HTTP_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5


def _completion_payload(
    prompt: list[int],
    served_model_name: str,
) -> dict[str, object]:
    return {
        "model": served_model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "extra_body": {"add_special_tokens": False},
    }


def _post_completion(base_url: str, payload: dict[str, object]) -> bool:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.monotonic() + HTTP_TIMEOUT_SECONDS
    while True:
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    json.load(resp)
                    return True
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)


def send_precondition(
    base_url: str,
    dataset_path: Path,
    served_model_name: str,
) -> int:
    sent = 0
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            payload = _completion_payload(record["prompt"], served_model_name)
            if not _post_completion(base_url, payload):
                print(
                    f"FAILED precondition request {record['request_id']} "
                    f"({len(record['prompt'])} tokens)",
                    flush=True,
                )
                return 1
            sent += 1
            print(
                f"[precondition] {sent} sent: {record['request_id']} "
                f"{len(record['prompt'])} tokens",
                flush=True,
            )
    print(f"[precondition] done: {sent} requests", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--served-model-name", default="deepseek_v3_2")
    args = parser.parse_args(argv)
    return send_precondition(args.base_url, args.dataset, args.served_model_name)


if __name__ == "__main__":
    raise SystemExit(main())
