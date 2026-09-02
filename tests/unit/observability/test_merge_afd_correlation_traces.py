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
    load_sidecar,
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


def test_merge_tolerates_partial_rank_participation(tmp_path: Path) -> None:
    """Flows routed to a subset of ranks are complete when every participant
    has paired phases — DP/EP topologies legitimately skip ranks."""

    def multi_pid_sidecar(path: str, role: str, pid: int) -> Sidecar:
        return Sidecar(
            path=Path(path),
            metadata={
                "session_id": "test-session",
                "dropped_events": 0,
                "identity": {
                    "role": role,
                    "role_rank": pid,
                    "hostname": "host",
                    "pid": pid,
                },
            },
            anchors=[{"monotonic_ns": 1_000, "realtime_ns": 1_000_000}],
            events=[],
        )

    # Two attention ranks, two FFN ranks; only rank 0 of each role sees flow.
    attn0 = multi_pid_sidecar("attn0.jsonl", "attention", 11)
    attn0.events = (
        _range_events("afd.cam.dispatch_send", 1_100)
        + _range_events("afd.cam.combine_recv", 1_500)
    )
    attn0.metadata["identity"]["pid"] = 11
    attn1 = multi_pid_sidecar("attn1.jsonl", "attention", 12)
    ffn0 = multi_pid_sidecar("ffn0.jsonl", "ffn", 13)
    ffn0.events = (
        _range_events("afd.cam.dispatch_recv", 1_200)
        + _range_events("afd.ffn.compute", 1_300)
        + _range_events("afd.cam.combine_send", 1_400)
    )
    ffn1 = multi_pid_sidecar("ffn1.jsonl", "ffn", 14)

    sidecars = [attn0, attn1, ffn0, ffn1]
    assign_clock_transforms(sidecars, [])
    _trace, report = build_merged_trace(sidecars, [])
    assert report["flows"]["total"] == 1
    assert report["flows"]["complete"] == 1
    assert report["flows"]["incomplete"] == []


def test_merge_flags_unpaired_phase_on_participant(tmp_path: Path) -> None:
    """A participant with a begin but no end still marks the flow incomplete."""
    attention = _sidecar(
        path="attention.jsonl",
        role="attention",
        hostname="host",
        anchor_mono_ns=1_000,
        anchor_realtime_ns=1_000_000,
        events=(
            _range_events("afd.cam.dispatch_send", 1_100)
            + [
                {  # combine_recv begin without end
                    "event": "afd.cam.combine_recv",
                    "flow_id": "abcdef0123456789",
                    "transaction_id": "afd-npu-1",
                    "layer_idx": 2,
                    "stage_idx": 0,
                    "num_tokens": 64,
                    "phase": "begin",
                    "monotonic_ns": 1_500,
                },
            ]
        ),
    )
    ffn = _sidecar(
        path="ffn.jsonl",
        role="ffn",
        hostname="host",
        anchor_mono_ns=1_000,
        anchor_realtime_ns=1_000_000,
        events=(
            _range_events("afd.cam.dispatch_recv", 1_200)
            + _range_events("afd.ffn.compute", 1_300)
            + _range_events("afd.cam.combine_send", 1_400)
        ),
    )
    sidecars = [attention, ffn]
    assign_clock_transforms(sidecars, [])
    _trace, report = build_merged_trace(sidecars, [])
    assert report["flows"]["total"] == 1
    assert report["flows"]["complete"] == 0
    incomplete = report["flows"]["incomplete"]
    assert len(incomplete) == 1
    assert incomplete[0]["missing_participants"][0]["event"] == (
        "afd.cam.combine_recv"
    )
    assert incomplete[0]["missing_participants"][0]["missing_phases"] == ["end"]


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


def test_load_unfinished_sidecar_keeps_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "afd-trace-test-attention-rank0-pid1-host.jsonl.tmp"
    records = [
        {
            "record_type": "metadata",
            "schema_version": 2,
            "session_id": "test-session",
            "identity": {
                "role": "attention",
                "role_rank": 0,
                "hostname": "host",
                "pid": 1,
            },
        },
        {
            "record_type": "clock_anchor",
            "monotonic_ns": 100,
            "realtime_ns": 1_000,
        },
        {
            "record_type": "event",
            "event": "afd.test",
            "phase": "instant",
            "monotonic_ns": 110,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + '\n{"partial":',
        encoding="utf-8",
    )

    sidecar = load_sidecar(path)

    assert len(sidecar.events) == 1
    assert sidecar.summary is None


def test_merge_aligns_deferred_markers_to_end_events(tmp_path: Path) -> None:
    """Deferred dispatch_recv markers are emitted after the receive op, so
    they correspond to the sidecar END event, not the begin. Alignment must
    pick the nearest phase and keep the clock shift constant."""
    # begin at 1.0s, end at 5.0s (a recv that blocked 4s); markers at the end.
    events = [
        {
            "event": "afd.cam.dispatch_recv",
            "flow_id": "abcdef0123456789",
            "transaction_id": "afd-npu-e0-0",
            "layer_idx": 0,
            "stage_idx": 0,
            "phase": "begin",
            "monotonic_ns": 1_000_000,
        },
        {
            "event": "afd.cam.dispatch_recv",
            "flow_id": "abcdef0123456789",
            "transaction_id": "afd-npu-e0-0",
            "layer_idx": 0,
            "stage_idx": 0,
            "phase": "end",
            "monotonic_ns": 5_000_000,
            "outcome": "ok",
        },
    ]
    sidecar = Sidecar(
        path=Path("ffn.jsonl"),
        metadata={
            "session_id": "test-session",
            "dropped_events": 0,
            "identity": {
                "role": "ffn",
                "role_rank": 0,
                "hostname": "host",
                "pid": 13,
            },
        },
        anchors=[{"monotonic_ns": 0, "realtime_ns": 1_000_000_000}],
        events=events,
    )
    assign_clock_transforms([sidecar], [])
    # Marker ts = epoch us of the END event (1_000_500 us), plus a fixed
    # 10us profiler-vs-sidecar offset; a begin-anchored match would be 4ms off.
    profiler_path = tmp_path / "ffn-profiler.json"
    profiler_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "afd.cam.dispatch_recv flow_id=abcdef0123456789",
                        "cat": "mstx",
                        "ph": "X",
                        "pid": 10,
                        "tid": 10,
                        "ts": 1_005_010.0,
                        "dur": 0.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    trace, report = build_merged_trace([sidecar], [(sidecar.path, profiler_path)])
    diag = report["profiler_traces"][0]
    assert diag["status"] == "aligned"
    assert diag["marker_delta_spread_us"] < 1.0
    shifted = [
        e
        for e in trace["traceEvents"]
        if str(e.get("name", "")).startswith("afd.") and e.get("cat") == "mstx"
    ]
    # Origin is the sidecar begin (1ms); the end sits at 4000us relative and
    # the marker must be shifted onto it (not onto the begin).
    assert abs(shifted[0]["ts"] - 4_000.0) < 1.0


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
                        "name": ("afd.cam.dispatch_send flow_id=abcdef0123456789"),
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
        event for event in trace["traceEvents"] if event.get("cat") == "afd.flow"
    ]
    flow_keys: dict[int, set[tuple[object, object, object]]] = {}
    for event in flow_events:
        flow_keys.setdefault(event["id"], set()).add(
            (event["cat"], event["name"], event["id"])
        )
    assert flow_keys
    assert all(len(keys) == 1 for keys in flow_keys.values())

    device_flow_events = [
        event for event in trace["traceEvents"] if event.get("cat") == "afd.device-flow"
    ]
    assert [event["ph"] for event in device_flow_events] == ["s", "f"]
    assert (
        len(
            {(event["cat"], event["name"], event["id"]) for event in device_flow_events}
        )
        == 1
    )


def test_merge_reports_summary_drops_and_incomplete_ranges() -> None:
    sidecar = _sidecar(
        path="unfinished-range.jsonl",
        role="attention",
        hostname="reference",
        anchor_mono_ns=1_000,
        anchor_realtime_ns=1_000_000,
        events=_range_events("afd.cam.dispatch_send", 1_100)[:1],
    )
    sidecar.summary = {"dropped_events": 3, "event_count": 1}
    assign_clock_transforms([sidecar], [])

    _, report = build_merged_trace([sidecar], [])

    assert report["dropped_events"] == 3
    assert report["incomplete_ranges"][0]["missing_phase"] == "end"


def test_device_flows_stay_with_their_profiler_sidecar(tmp_path: Path) -> None:
    first = _sidecar(
        path="first.jsonl",
        role="attention",
        hostname="reference",
        anchor_mono_ns=1_000,
        anchor_realtime_ns=1_000_000,
        events=_range_events("afd.cam.dispatch_send", 1_100),
    )
    second = _sidecar(
        path="second.jsonl",
        role="attention",
        hostname="reference",
        anchor_mono_ns=2_000,
        anchor_realtime_ns=2_000_000,
        events=_range_events("afd.cam.dispatch_send", 2_100),
    )
    assign_clock_transforms([first, second], [])

    profiler_pairs = []
    for index, (sidecar, marker_ts, op_ts) in enumerate(
        ((first, 100.0, 400.0), (second, 200.0, 300.0)),
    ):
        path = tmp_path / f"profiler-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": ("afd.cam.dispatch_send flow_id=abcdef0123456789"),
                            "ph": "X",
                            "pid": 10,
                            "tid": 10,
                            "ts": marker_ts,
                            "dur": 1.0,
                        },
                        {
                            "name": "CamMoeDistributeDispatchSend",
                            "ph": "X",
                            "pid": 20,
                            "tid": 20,
                            "ts": op_ts,
                            "dur": 1.0,
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )
        profiler_pairs.append((sidecar.path, path))

    trace, report = build_merged_trace([first, second], profiler_pairs)

    endpoints_by_flow: dict[int, set[int]] = {}
    for event in trace["traceEvents"]:
        if event.get("cat") == "afd.device-flow":
            endpoints_by_flow.setdefault(int(event["id"]), set()).add(
                int(event["pid"]),
            )
    assert set(map(frozenset, endpoints_by_flow.values())) == {
        frozenset({1, 4}),
        frozenset({2, 6}),
    }
    assert report["device_flows"]["skipped_ambiguous_correlation"] == 0
