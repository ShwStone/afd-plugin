#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Send the fixed precision fixtures to a running Async CAM vLLM server.

The fixture pins every request to one Attention DP engine via the
``X-data-parallel-rank`` header so the two-request batch lands on a single
engine and produces a deterministic flattened token order for all three Async
CAM modes (no-ubatch / request split / token split).

Prompt text is deterministic (identical across runs); the client optionally
calibrates the prompt against the served tokenizer with ``/v1/tokenize`` so
the scheduled prefill token counts land near the requested targets (req0 ~768,
req1 ~256, single ~1024).

Usage::

    python -m tools.precision.precision_fixture_client \\
        --base-url http://127.0.0.1:8000 --model deepseek_v3_2 \\
        --output-dir /tmp/afd_precision_client --dp-rank 0 \\
        --two-request --single-long
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_PHRASES = [
    "The quick brown fox jumps over the lazy dog while measuring distributed inference throughput on Ascend hardware.",
    "Attention mechanisms route tokens to expert networks and the combined output is restored in global token order.",
]


def _build_prompt(seed: int) -> str:
    phrase = _PHRASES[seed % len(_PHRASES)]
    repeats = 3
    parts: list[str] = []
    for index in range(repeats):
        parts.append(phrase if (index + seed) % 2 == 0 else phrase[::-1])
    return " ".join(parts)


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def _count_tokens(base_url: str, model: str, prompt: str) -> int | None:
    url = f"{base_url}/v1/tokenize"
    try:
        result = _post_json(url, {"model": model, "prompt": prompt})
        tokens = result.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    except Exception:
        pass
    return None


def _build_target_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    seed: int,
) -> str:
    """Build a deterministic prompt tokenizing near ``target_tokens``."""
    prompt = _build_prompt(seed)
    for _ in range(8):
        count = _count_tokens(base_url, model, prompt)
        if count is None:
            break
        if count == target_tokens:
            break
        if count < target_tokens:
            prompt += " " + _build_prompt(seed + count)
        else:
            words = prompt.split()
            prompt = " ".join(words[: max(1, len(words) - (count - target_tokens))])
        if abs(count - target_tokens) < 8:
            break
    return prompt


def _request_completion(
    base_url: str,
    model: str,
    prompt: str,
    *,
    request_id: str,
    dp_rank: int,
    max_tokens: int,
    seed: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }
    headers = {"X-data-parallel-rank": str(dp_rank)}
    result = _post_json(
        f"{base_url}/v1/completions",
        payload,
        headers=headers,
    )
    choices = result.get("choices") or []
    output_text = choices[0].get("text", "") if choices else ""
    return {
        "request_id": request_id,
        "dp_rank": dp_rank,
        "prompt_chars": len(prompt),
        "output_text": output_text,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "usage": result.get("usage"),
        "error": result.get("error"),
    }


def _send_fixture(
    base_url: str,
    model: str,
    prompts: list[tuple[str, str]],
    *,
    dp_rank: int,
    max_tokens: int,
    seed: int,
) -> list[dict]:
    results: list[dict] = [None] * len(prompts)
    errors: list[Exception] = []

    def worker(index: int, request_id: str, prompt: str) -> None:
        try:
            results[index] = _request_completion(
                base_url,
                model,
                prompt,
                request_id=request_id,
                dp_rank=dp_rank,
                max_tokens=max_tokens,
                seed=seed,
            )
        except Exception as error:  # pragma: no cover - defensive
            errors.append(error)
            results[index] = {"request_id": request_id, "dp_rank": dp_rank, "error": str(error)}

    threads = [
        threading.Thread(target=worker, args=(index, request_id, prompt))
        for index, (request_id, prompt) in enumerate(prompts)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"{len(errors)} fixture request(s) failed: {errors[0]}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="vLLM OpenAI base URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dp-rank", type=int, default=0, help="attention DP engine to pin")
    parser.add_argument("--max-tokens", type=int, default=8, help="greedy decode length")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--req0-tokens", type=int, default=768)
    parser.add_argument("--req1-tokens", type=int, default=256)
    parser.add_argument("--single-tokens", type=int, default=1024)
    parser.add_argument("--two-request", action="store_true", help="send req0+req1 fixture")
    parser.add_argument("--single-long", action="store_true", help="send single long fixture")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"dp_rank": args.dp_rank, "seed": args.seed, "requests": []}

    if args.two_request:
        prompt0 = _build_target_prompt(args.base_url, args.model, args.req0_tokens, 11)
        prompt1 = _build_target_prompt(args.base_url, args.model, args.req1_tokens, 29)
        results = _send_fixture(
            args.base_url,
            args.model,
            [("fixture-req0", prompt0), ("fixture-req1", prompt1)],
            dp_rank=args.dp_rank,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        summary["requests"].extend(results)

    if args.single_long:
        prompt_single = _build_target_prompt(args.base_url, args.model, args.single_tokens, 47)
        results = _send_fixture(
            args.base_url,
            args.model,
            [("fixture-single", prompt_single)],
            dp_rank=args.dp_rank,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        summary["requests"].extend(results)

    summary_path = args.output_dir / "client_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")
    for request in summary["requests"]:
        print(f"  {request['request_id']}: prompt_chars={request['prompt_chars']} "
              f"dp={request['dp_rank']} error={request.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
