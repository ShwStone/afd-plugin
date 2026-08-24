# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

from tools.benchmarks.merge_afd_correlation_traces import (
    Sidecar,
    assign_clock_transforms,
    best_clock_sample,
    build_merged_trace,
)


def _sidecar(
    *,
    path: str,
    role: str,
    hostname: str,
    anchor_mono_ns: int,
    anchor_realtime_ns: int,
    events: list[dict[str, object]],
) -> Sidecar:
    return Sidecar(
        path=Path(path),
        metadata={
            "session_id": "test-session",
            "dropped_events": 0,
            "identity": {
                "role": role,
                "role_rank": 0,
                "hostname": hostname,
                "pid": 11,
            },
        },
        anchors=[
            {
                "monotonic_ns": anchor_mono_ns,
                "realtime_ns": anchor_realtime_ns,
            },
        ],
        events=events,
    )


def _range_events(
    name: str,
    start_ns: int,
    *,
    flow_id: str = "abcdef0123456789",
) -> list[dict[str, object]]:
    common = {
        "event": name,
        "flow_id": flow_id,
        "transaction_id": "afd-npu-1",
        "layer_idx": 2,
        "stage_idx": 0,
        "num_tokens": 64,
    }
    return [
        {**common, "phase": "begin", "monotonic_ns": start_ns},
        {**common, "phase": "end", "monotonic_ns": start_ns + 10},
    ]


def test_best_clock_sample_selects_minimum_round_trip() -> None:
    offset_ns, uncertainty_ns = best_clock_sample(
        {
            "samples": [
                {
                    "client_send_ns": 100,
                    "server_receive_ns": 1120,
                    "server_send_ns": 1130,
                    "client_receive_ns": 140,
                },
                {
                    "client_send_ns": 100,
                    "server_receive_ns": 1110,
                    "server_send_ns": 1120,
                    "client_receive_ns": 120,
                },
            ],
        },
    )

    assert offset_ns == 1005
    assert uncertainty_ns == 5


def test_merge_correlates_complete_cross_host_flow(tmp_path: Path) -> None:
    attention = _sidecar(
        path="attention.jsonl",
        role="attention",
        hostname="reference",
        anchor_mono_ns=1_000,
        anchor_realtime_ns=1_000_000,
        events=(
            _range_events("afd.cam.dispatch_send", 1_100)
            + _range_events("afd.cam.combine_recv", 1_500)
        ),
    )
    ffn = _sidecar(
        path="ffn.jsonl",
        role="ffn",
        hostname="client",
        anchor_mono_ns=100,
        anchor_realtime_ns=500_000,
        events=(
            _range_events("afd.cam.dispatch_recv", 200)
            + _range_events("afd.ffn.compute", 250)
            + _range_events("afd.cam.combine_send", 300)
        ),
    )
    clock_sync = {
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
    }

    diagnostics = assign_clock_transforms([attention, ffn], [clock_sync])
    profiler_path = tmp_path / "attention-profiler.json"
    profiler_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": (
                            "afd.cam.dispatch_send "
                            "flow_id=abcdef0123456789"
                        ),
                        "cat": "mstx",
                        "ph": "X",
                        "pid": 10,
                        "tid": 10,
                        "ts": 5.0,
                        "dur": 1.0,
                    },
                    {
                        "name": "CamMoeDistributeDispatchSend",
                        "cat": "npu",
                        "ph": "X",
                        "pid": 20,
                        "tid": 20,
                        "ts": 6.0,
                        "dur": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    trace, report = build_merged_trace(
        [attention, ffn],
        [(attention.path, profiler_path)],
    )

    assert diagnostics[1]["uncertainty_ns"] == 5
    assert report["flows"]["total"] == 1
    assert report["flows"]["complete"] == 1
    assert report["flows"]["incomplete"] == []
    assert report["device_flows"]["linked_device_flows"] == 1
    names = [event["name"] for event in trace["traceEvents"]]
    assert "afd.ffn.compute" in names
    assert any(event.get("ph") == "s" for event in trace["traceEvents"])
    assert any(event.get("ph") == "f" for event in trace["traceEvents"])

    flow_events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "afd.flow"
    ]
    flow_keys: dict[int, set[tuple[object, object, object]]] = {}
    for event in flow_events:
        flow_keys.setdefault(event["id"], set()).add(
            (event["cat"], event["name"], event["id"])
        )
    assert flow_keys
    assert all(len(keys) == 1 for keys in flow_keys.values())

    device_flow_events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "afd.device-flow"
    ]
    assert [event["ph"] for event in device_flow_events] == ["s", "f"]
    assert len(
        {
            (event["cat"], event["name"], event["id"])
            for event in device_flow_events
        }
    ) == 1
