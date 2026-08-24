# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Single-request smoke client for phase-zero acceptance (plan section 8.3).

Sends one request from the dataset (shortest/longest/selectable) and reports
latency. Runs inside the experiment container (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def pick_request(dataset_path: str, mode: str) -> tuple[list[int], str]:
    best: list[int] | None = None
    best_id = ""
    with open(dataset_path, encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            prompt = record.get("prompt_token_ids") or record["prompt"]
            if best is None:
                best, best_id = prompt, str(record.get("request_id", ""))
                continue
            if mode == "longest" and len(prompt) > len(best):
                best, best_id = prompt, str(record.get("request_id", ""))
            if mode == "shortest" and len(prompt) < len(best):
                best, best_id = prompt, str(record.get("request_id", ""))
    if best is None:
        raise ValueError(f"empty dataset: {dataset_path}")
    return best, best_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        default="tools/datasets/moonconv-wildchat-prefill/workloads/"
        "warmup_requests.jsonl",
    )
    parser.add_argument("--mode", choices=("shortest", "longest"), required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    prompt, request_id = pick_request(args.dataset, args.mode)
    body = json.dumps(
        {
            "model": "deepseek_v3_2",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "ignore_eos": True,
            "add_special_tokens": False,
        }
    ).encode()
    start = time.time()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(request, timeout=args.timeout)
    payload = json.loads(response.read())
    latency = time.time() - start
    usage = payload.get("usage", {})
    print(
        f"SMOKE_OK mode={args.mode} request_id={request_id} "
        f"prompt_len={len(prompt)} latency={latency:.2f}s "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )


if __name__ == "__main__":
    main()
