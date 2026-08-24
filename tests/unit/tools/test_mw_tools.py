# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks import (
    mw_make_acceptance_sets,
    mw_scale_arrivals,
    mw_select_mechanism_batches,
)


def _write_requests(path: Path, specs: list[tuple[str, int]]) -> None:
    lines = []
    for request_id, length in specs:
        token_ids = list(range(length))
        lines.append(
            json.dumps(
                {
                    "request_id": request_id,
                    "input_length": length,
                    "output_length": 1,
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": "x" * 64,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_arrivals(path: Path, specs: list[tuple[str, int, int]]) -> None:
    lines = []
    for index, (request_id, offset_ms, length) in enumerate(specs):
        lines.append(
            json.dumps(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "sequence_index": index,
                    "mooncake_trace_index": index,
                    "target_input_length": length,
                    "actual_input_length": length,
                    "base_arrival_offset_ms": offset_ms,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Screening fixture: 10 lengths around 1024 (1000+4k), 20 around 2048
# (2000+2k), one at 24500, nine far away (30000+100k).
SCREENING_LENGTHS = (
    [1000 + 4 * k for k in range(10)]
    + [2000 + 2 * k for k in range(20)]
    + [24500]
    + [30000 + 100 * k for k in range(9)]
)


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ds"
    workloads = dataset / "workloads"
    workloads.mkdir(parents=True)
    screening = [
        (f"screening-{i:06d}", length)
        for i, length in enumerate(SCREENING_LENGTHS)
    ]
    _write_requests(workloads / "screening_requests.jsonl", screening)
    _write_arrivals(
        workloads / "screening_arrivals.jsonl",
        [
            (rid, offset, length)
            for (rid, length), offset in zip(
                screening, [0, 0, 1000, 1000, 2000, 3000, 3000, 4000]
                + [4000] * (len(screening) - 8)
            )
        ],
    )
    _write_requests(
        workloads / "warmup_requests.jsonl",
        [(f"warmup-{i:06d}", 100 + i * 10) for i in range(32)],
    )
    _write_requests(
        workloads / "formal_1_requests.jsonl",
        [
            (f"formal-1-{i:06d}", length)
            for i, length in enumerate([5000, 63778, 51000, 50500])
        ],
    )
    _write_arrivals(
        workloads / "formal_1_arrivals.jsonl",
        [
            ("formal-1-000000", 0, 5000),
            ("formal-1-000001", 5, 63778),
            ("formal-1-000002", 5, 51000),
            ("formal-1-000003", 10, 50500),
        ],
    )
    return dataset


def test_scale_arrivals_dilation_and_integer_ns(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    out = tmp_path / "plan.json"
    total = sum(SCREENING_LENGTHS)
    # base duration 4000 ms -> base rate = total / 4 tok/s; target half of
    # that -> dilation exactly 2.
    target = str(total / 8)
    mw_scale_arrivals.main([
        "--dataset-dir", str(dataset),
        "--window", "screening",
        "--target-tokens-per-second", target,
        "--output", str(out),
    ])
    plan = json.loads(out.read_text())
    from decimal import Decimal

    assert Decimal(plan["dilation_factor"]) == 2
    offsets = {a["request_id"]: a["scaled_arrival_offset_ns"] for a in plan["arrivals"]}
    assert offsets["screening-000000"] == 0
    assert offsets["screening-000001"] == 0  # zero gaps stay zero
    assert offsets["screening-000002"] == 2_000_000_000
    assert offsets["screening-000007"] == 8_000_000_000
    assert plan["scaled_duration_s"] == "8"
    assert len(plan["base_arrivals_sha256"]) == 64
    assert len(plan["requests_sha256"]) == 64

    out2 = tmp_path / "plan2.json"
    mw_scale_arrivals.main([
        "--dataset-dir", str(dataset),
        "--window", "screening",
        "--target-tokens-per-second", target,
        "--output", str(out2),
    ])
    assert out.read_bytes() == out2.read_bytes()  # deterministic


def test_scale_arrivals_rejects_order_mismatch(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    arrivals = dataset / "workloads" / "screening_arrivals.jsonl"
    lines = arrivals.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    arrivals.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="order mismatch"):
        mw_scale_arrivals.main([
            "--dataset-dir", str(dataset),
            "--window", "screening",
            "--target-tokens-per-second", "9325",
            "--output", str(tmp_path / "plan.json"),
        ])


def test_select_mechanism_batches_no_reuse_and_order(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    out_dir = tmp_path / "out"
    mw_select_mechanism_batches.main([
        "--dataset-dir", str(dataset),
        "--output-dir", str(out_dir),
    ])
    manifest = json.loads((out_dir / "mechanism_batches.json").read_text())
    batches = {b["name"]: b for b in manifest["batches"]}
    assert set(batches) == {
        "fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short"
    }

    all_ids = [rid for b in manifest["batches"] for rid in b["request_ids"]]
    assert len(all_ids) == len(set(all_ids))  # no reuse across batches

    # 8K batch: the 8 closest to 1024 are lengths 1008..1036 (ids 2..9).
    assert batches["fixed_8k_balanced"]["input_lengths"] == [
        1024, 1020, 1028, 1016, 1032, 1012, 1036, 1008
    ]
    # 32K balanced: 8 closest to 4096 from what remains (new per DP concurrency).
    assert len(batches["fixed_32k_balanced"]["request_ids"]) == 8
    # Long-short: 24500 first (closest to 24576), then 7 closest to 1170.
    long_short = batches["fixed_32k_long_short"]
    assert long_short["input_lengths"][0] == 24500
    assert len(long_short["request_ids"]) == 8

    for batch in manifest["batches"]:
        batch_file = out_dir / batch["file"]
        records = [json.loads(l) for l in batch_file.read_text().splitlines()]
        assert [r["request_id"] for r in records] == batch["request_ids"]
        assert all("prompt" in r for r in records)
        assert sum(r["input_length"] for r in records) == batch["total_tokens"]


def test_make_acceptance_sets(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    mw_make_acceptance_sets.main(["--dataset-dir", str(dataset)])
    workloads = dataset / "workloads"

    warmup8 = [
        json.loads(l)
        for l in (workloads / "accept_8_warmup.jsonl").read_text().splitlines()
    ]
    assert len(warmup8) == 8
    lengths = [r["input_length"] for r in warmup8]
    assert lengths == sorted(lengths)  # short -> long coverage

    singles = [
        json.loads(l)
        for l in (workloads / "accept_long_singles.jsonl").read_text().splitlines()
    ]
    assert len(singles) == 3
    assert singles[-1]["input_length"] == 63778  # bundle-longest last

    four = [
        json.loads(l)
        for l in (workloads / "accept_4x52k.jsonl").read_text().splitlines()
    ]
    assert len(four) == 4
    assert [r["dp_rank"] for r in four] == [0, 1, 2, 3]
    assert four[0]["input_length"] == 51000  # closest to 52K first


def test_replay_client_helpers(tmp_path: Path) -> None:
    import sys
    import types

    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
    from tools.benchmarks import mw_replay_client

    assert mw_replay_client._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert mw_replay_client._percentile([], 99) == 0.0

    dataset = _make_dataset(tmp_path)
    requests = mw_replay_client.load_requests(
        dataset / "workloads" / "screening_requests.jsonl", None
    )
    assert len(requests) == len(SCREENING_LENGTHS)
    subset = mw_replay_client.load_requests(
        dataset / "workloads" / "screening_requests.jsonl",
        {"screening-000000", "screening-000003"},
    )
    assert [r["request_id"] for r in subset] == [
        "screening-000000",
        "screening-000003",
    ]
