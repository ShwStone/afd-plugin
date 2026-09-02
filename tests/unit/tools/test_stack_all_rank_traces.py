# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks.stack_all_rank_traces import (
    _load_correlation_events,
    stack,
)


def _write_sidecar(
    path: Path,
    *,
    role: str,
    hostname: str,
    monotonic_ns: int,
    realtime_ns: int,
    event_ns: int,
) -> None:
    common_event = {
        "record_type": "event",
        "event": "afd.cam.dispatch_send",
        "flow_id": "abcdef0123456789",
        "transaction_id": "afd-npu-0",
        "layer_idx": 1,
        "stage_idx": 0,
        "num_tokens": 8,
    }
    records = [
        {
            "record_type": "metadata",
            "schema_version": 2,
            "session_id": "session",
            "identity": {
                "role": role,
                "rank": 0,
                "role_rank": 0,
                "local_rank": 0,
                "hostname": hostname,
                "pid": 10,
            },
        },
        {
            "record_type": "clock_anchor",
            "monotonic_ns": monotonic_ns,
            "realtime_ns": realtime_ns,
        },
        {**common_event, "phase": "begin", "monotonic_ns": event_ns},
        {**common_event, "phase": "end", "monotonic_ns": event_ns + 10},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_correlation_lanes_use_explicit_cross_host_clock_sync(tmp_path: Path) -> None:
    _write_sidecar(
        tmp_path / "afd-trace-session-attention-rank0-pid10-reference.jsonl",
        role="attention",
        hostname="reference",
        monotonic_ns=1_000,
        realtime_ns=1_000_000,
        event_ns=1_100,
    )
    _write_sidecar(
        tmp_path / "afd-trace-session-ffn-rank0-pid10-client.jsonl",
        role="ffn",
        hostname="client",
        monotonic_ns=100,
        realtime_ns=500_000,
        event_ns=200,
    )
    clock_sync = tmp_path / "clock-sync.json"
    clock_sync.write_text(
        json.dumps(
            {
                "record_type": "clock_sync_client",
                "session_id": "session",
                "client_host": "client",
                "reference_host": "reference",
                "samples": [
                    {
                        "client_send_ns": 100,
                        "server_receive_ns": 1_110,
                        "server_send_ns": 1_120,
                        "client_receive_ns": 120,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    lanes, diagnostics, pid_map = _load_correlation_events(
        tmp_path,
        [clock_sync],
    )

    attention_ts = lanes[("attention", 0)]["slices"][0]["ts"]
    ffn_ts = lanes[("ffn", 0)]["slices"][0]["ts"]
    assert ffn_ts - attention_ts == pytest.approx(0.105)
    assert any(
        diagnostic["method"] == "four_timestamp_to_reference"
        for diagnostic in diagnostics
    )
    assert pid_map[("client", 10)] == ("ffn", 0)


def test_single_host_afd_stack_requires_correlation_sidecars(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()

    with pytest.raises(ValueError, match="AFD stacking requires"):
        stack(
            [("attention", "node0", trace_root, None)],
            min_dur_us=0.0,
            align_op="AllGather",
            session_ts=None,
        )


def test_afd_stack_rejects_sidecar_without_profiler_rank(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    correlation_dir = tmp_path / "correlation"
    correlation_dir.mkdir()
    _write_sidecar(
        correlation_dir / "afd-trace-session-attention-rank0-pid10.jsonl",
        role="attention",
        hostname="localhost",
        monotonic_ns=1_000,
        realtime_ns=1_000_000,
        event_ns=1_100,
    )

    with pytest.raises(ValueError, match="missing profiler traces"):
        stack(
            [("attention", "node0", trace_root, None)],
            min_dur_us=0.0,
            align_op="AllGather",
            session_ts=None,
            correlation_dir=correlation_dir,
        )


def test_cross_host_correlation_rejects_missing_clock_sync(tmp_path: Path) -> None:
    _write_sidecar(
        tmp_path / "afd-trace-session-attention-rank0-pid10-reference.jsonl",
        role="attention",
        hostname="reference",
        monotonic_ns=1_000,
        realtime_ns=1_000_000,
        event_ns=1_100,
    )
    _write_sidecar(
        tmp_path / "afd-trace-session-ffn-rank0-pid10-client.jsonl",
        role="ffn",
        hostname="client",
        monotonic_ns=100,
        realtime_ns=500_000,
        event_ns=200,
    )

    with pytest.raises(ValueError, match="one reference host"):
        _load_correlation_events(tmp_path, [])
