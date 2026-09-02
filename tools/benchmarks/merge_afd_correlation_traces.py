# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Merge AFD correlation sidecars and optional profiler traces.

The output is a Chrome/Perfetto trace. A sibling validation report describes
clock-alignment quality, dropped events, incomplete logical exchanges, and
profiler traces that could not be aligned to sidecar markers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

if __package__:
    from .link_afd_device_flows import CORRELATION_PID_ARG, build_device_flows
else:
    from link_afd_device_flows import CORRELATION_PID_ARG, build_device_flows

NANOSECONDS_PER_MICROSECOND: Final[int] = 1_000
TRACE_SCHEMA_VERSION: Final[int] = 1
FLOW_MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<event>afd\.[^ ]+)(?: flow_id=(?P<flow_id>[0-9a-f]+))?$",
)
ROLE_FLOW_EVENTS: Final[dict[str, frozenset[str]]] = {
    "attention": frozenset(
        {"afd.cam.dispatch_send", "afd.cam.combine_recv"},
    ),
    "ffn": frozenset(
        {
            "afd.cam.dispatch_recv",
            "afd.ffn.compute",
            "afd.cam.combine_send",
        },
    ),
}
REQUIRED_FLOW_EVENTS: Final[frozenset[str]] = frozenset().union(
    *ROLE_FLOW_EVENTS.values(),
)


@dataclass(frozen=True, slots=True)
class ClockTransform:
    """Affine mapping from one host's monotonic nanoseconds to global time."""

    source_mono_ns: int
    target_realtime_ns: int
    method: str
    uncertainty_ns: int | None

    def to_global_ns(self, monotonic_ns: int) -> int:
        return self.target_realtime_ns + monotonic_ns - self.source_mono_ns


@dataclass(slots=True)
class Sidecar:
    """Parsed contents of one process-local JSONL sidecar."""

    path: Path
    metadata: dict[str, object]
    anchors: list[dict[str, object]]
    events: list[dict[str, object]]
    transform: ClockTransform | None = None
    pid: int = 0
    summary: dict[str, object] | None = None

    @property
    def identity(self) -> dict[str, object]:
        identity = self.metadata["identity"]
        if not isinstance(identity, dict):
            raise ValueError(f"{self.path}: metadata.identity must be an object")
        return identity

    @property
    def hostname(self) -> str:
        return str(self.identity["hostname"])


def load_sidecar(path: Path) -> Sidecar:
    metadata: dict[str, object] | None = None
    anchors: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    summary: dict[str, object] | None = None
    unfinished = path.name.endswith(".jsonl.tmp")
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if unfinished and not input_file.read().strip():
                    break
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON record",
                ) from None
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            record_type = payload.get("record_type")
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError(f"{path}: contains multiple metadata records")
                metadata = payload
            elif record_type == "clock_anchor":
                anchors.append(payload)
            elif record_type == "event":
                events.append(payload)
            elif record_type == "summary":
                if summary is not None:
                    raise ValueError(f"{path}: contains multiple summary records")
                summary = payload
            else:
                raise ValueError(
                    f"{path}:{line_number}: unknown record_type {record_type!r}",
                )
    if metadata is None:
        raise ValueError(f"{path}: missing metadata record")
    schema_version = int(metadata.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise ValueError(f"{path}: unsupported schema version {schema_version}")
    if not anchors:
        raise ValueError(f"{path}: missing clock anchors")
    return Sidecar(
        path=path,
        metadata=metadata,
        anchors=anchors,
        events=events,
        summary=summary,
    )


def load_clock_sync(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: clock-sync document must be an object")
    if payload.get("record_type") != "clock_sync_client":
        raise ValueError(f"{path}: expected a clock_sync_client document")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError(f"{path}: unsupported clock-sync schema version")
    return payload


def best_clock_sample(payload: dict[str, object]) -> tuple[int, int]:
    """Return server-minus-client offset and half-RTT uncertainty."""

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("clock-sync document has no samples")
    candidates: list[tuple[int, int]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("clock-sync sample must be an object")
        client_send_ns = int(sample["client_send_ns"])
        server_receive_ns = int(sample["server_receive_ns"])
        server_send_ns = int(sample["server_send_ns"])
        client_receive_ns = int(sample["client_receive_ns"])
        round_trip_ns = (client_receive_ns - client_send_ns) - (
            server_send_ns - server_receive_ns
        )
        if round_trip_ns < 0:
            raise ValueError("clock-sync sample has a negative network round trip")
        offset_ns = (
            (server_receive_ns - client_send_ns) + (server_send_ns - client_receive_ns)
        ) // 2
        candidates.append((round_trip_ns, offset_ns))
    round_trip_ns, offset_ns = min(candidates)
    return offset_ns, (round_trip_ns + 1) // 2


def validate_clock_sync_coverage(
    sidecars: Iterable[Sidecar],
    clock_syncs: Iterable[dict[str, object]],
) -> None:
    """Require one reference and one calibration per other sidecar host."""

    hosts = {sidecar.hostname for sidecar in sidecars}
    if len(hosts) <= 1:
        return
    clock_syncs = list(clock_syncs)
    reference_hosts = {str(clock_sync["reference_host"]) for clock_sync in clock_syncs}
    if len(reference_hosts) != 1:
        raise ValueError(
            "cross-host merging requires clock-sync files with one reference host",
        )
    reference_host = next(iter(reference_hosts))
    if reference_host not in hosts:
        raise ValueError(
            f"clock-sync reference host {reference_host!r} has no sidecar",
        )
    calibrated_hosts = {str(clock_sync["client_host"]) for clock_sync in clock_syncs}
    missing_hosts = hosts - {reference_host} - calibrated_hosts
    if missing_hosts:
        raise ValueError(
            f"missing clock-sync files for hosts: {sorted(missing_hosts)}",
        )


def assign_clock_transforms(
    sidecars: list[Sidecar],
    clock_syncs: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Assign global-time transforms and return human-readable diagnostics."""

    reference_anchors: dict[str, dict[str, object]] = {}
    for sidecar in sidecars:
        reference_anchors.setdefault(sidecar.hostname, sidecar.anchors[0])

    sync_by_client: dict[str, dict[str, object]] = {}
    reference_hosts: set[str] = set()
    for clock_sync in clock_syncs:
        client_host = str(clock_sync["client_host"])
        if client_host in sync_by_client:
            raise ValueError(f"multiple clock-sync files for host {client_host!r}")
        sync_by_client[client_host] = clock_sync
        reference_hosts.add(str(clock_sync["reference_host"]))
    if len(reference_hosts) > 1:
        raise ValueError(
            f"clock-sync files use multiple reference hosts: {reference_hosts}",
        )

    diagnostics: list[dict[str, object]] = []
    for sidecar in sidecars:
        local_anchor = sidecar.anchors[0]
        sync = sync_by_client.get(sidecar.hostname)
        if sync is None:
            is_reference = sidecar.hostname in reference_hosts
            sidecar.transform = ClockTransform(
                source_mono_ns=int(local_anchor["monotonic_ns"]),
                target_realtime_ns=int(local_anchor["realtime_ns"]),
                method=(
                    "clock_sync_reference_anchor"
                    if is_reference
                    else "local_realtime_anchor"
                ),
                uncertainty_ns=int(
                    local_anchor.get("capture_uncertainty_ns", 0),
                ),
            )
            diagnostics.append(
                _clock_diagnostic(
                    sidecar,
                    note=(
                        None
                        if is_reference or len(reference_anchors) == 1
                        else (
                            "No cross-host calibration; alignment depends on host "
                            "realtime clock synchronization"
                        )
                    ),
                ),
            )
            continue

        reference_host = str(sync["reference_host"])
        reference_anchor = reference_anchors.get(reference_host)
        if reference_anchor is None:
            raise ValueError(
                f"no sidecar from clock-sync reference host {reference_host!r}",
            )
        offset_ns, uncertainty_ns = best_clock_sample(sync)
        reference_mono_ns = int(reference_anchor["monotonic_ns"])
        reference_realtime_ns = int(reference_anchor["realtime_ns"])
        reference_uncertainty_ns = int(
            reference_anchor.get("capture_uncertainty_ns", 0),
        )
        local_source_mono_ns = int(local_anchor["monotonic_ns"])
        sidecar.transform = ClockTransform(
            source_mono_ns=local_source_mono_ns,
            target_realtime_ns=(
                reference_realtime_ns
                + local_source_mono_ns
                + offset_ns
                - reference_mono_ns
            ),
            method=f"four_timestamp_to_{reference_host}",
            uncertainty_ns=uncertainty_ns + reference_uncertainty_ns,
        )
        diagnostics.append(_clock_diagnostic(sidecar, note=None))
    return diagnostics


def _clock_diagnostic(
    sidecar: Sidecar,
    *,
    note: str | None,
) -> dict[str, object]:
    if sidecar.transform is None:
        raise RuntimeError("clock transform has not been assigned")
    return {
        "sidecar": str(sidecar.path),
        "hostname": sidecar.hostname,
        "method": sidecar.transform.method,
        "uncertainty_ns": sidecar.transform.uncertainty_ns,
        "note": note,
    }


def build_merged_trace(
    sidecars: list[Sidecar],
    profiler_trace_pairs: Iterable[tuple[Path, Path]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Create merged trace events and validation details."""

    if any(sidecar.transform is None for sidecar in sidecars):
        raise RuntimeError("clock transforms must be assigned before merging")
    all_global_times = [
        sidecar.transform.to_global_ns(int(event["monotonic_ns"]))
        for sidecar in sidecars
        for event in sidecar.events
        if sidecar.transform is not None
    ]
    if not all_global_times:
        raise ValueError("sidecars contain no events")
    origin_ns = min(all_global_times)

    trace_events: list[dict[str, object]] = []
    incomplete_ranges: list[dict[str, object]] = []
    flow_events: dict[str, list[tuple[Sidecar, dict[str, object], int]]] = defaultdict(
        list
    )
    for process_index, sidecar in enumerate(sidecars, start=1):
        sidecar.pid = process_index
        trace_events.extend(_process_metadata_events(sidecar))
        trace_events.extend(
            _sidecar_trace_events(
                sidecar,
                origin_ns,
                flow_events,
                incomplete_ranges,
            ),
        )

    flow_trace_events, flow_report = _build_flow_events(
        flow_events,
        origin_ns,
        sidecars,
    )
    trace_events.extend(flow_trace_events)

    sidecar_by_path = {
        path.resolve(): sidecar for path, sidecar in _sidecar_paths(sidecars)
    }
    profiler_report: list[dict[str, object]] = []
    next_pid = len(sidecars) + 1
    for sidecar_path, profiler_path in profiler_trace_pairs:
        sidecar = sidecar_by_path.get(sidecar_path.resolve())
        if sidecar is None:
            raise ValueError(
                f"profiler trace references an unloaded sidecar: {sidecar_path}",
            )
        profiler_events = _load_profiler_events(profiler_path)
        aligned, diagnostic, consumed_pids = _align_profiler_events(
            profiler_events,
            sidecar,
            origin_ns,
            first_pid=next_pid,
        )
        next_pid += consumed_pids
        trace_events.extend(aligned)
        profiler_report.append(
            {"profiler_trace": str(profiler_path), **diagnostic},
        )

    dropped_events = sum(_dropped_event_count(sidecar) for sidecar in sidecars)
    uncorrelated_events = sum(
        1
        for sidecar in sidecars
        for event in sidecar.events
        if event.get("flow_id") is None
    )
    error_events = [
        {
            "sidecar": str(sidecar.path),
            **_event_args(event),
            "event": event["event"],
        }
        for sidecar in sidecars
        for event in sidecar.events
        if event.get("outcome") == "error"
    ]
    device_flow_events, device_flow_report = build_device_flows(trace_events)
    trace_events.extend(device_flow_events)
    report = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "session_ids": sorted(
            {str(sidecar.metadata.get("session_id")) for sidecar in sidecars},
        ),
        "sidecars": len(sidecars),
        "sidecar_integrity": [_sidecar_integrity(item) for item in sidecars],
        "dropped_events": dropped_events,
        "uncorrelated_events": uncorrelated_events,
        "error_events": error_events,
        "incomplete_ranges": incomplete_ranges,
        "flows": flow_report,
        "device_flows": device_flow_report,
        "profiler_traces": profiler_report,
    }
    return {"traceEvents": trace_events, "displayTimeUnit": "us"}, report


def _sidecar_integrity(sidecar: Sidecar) -> dict[str, object]:
    expected_events = (
        int(sidecar.summary["event_count"]) if sidecar.summary is not None else None
    )
    return {
        "sidecar": str(sidecar.path),
        "finalized": sidecar.summary is not None,
        "loaded_events": len(sidecar.events),
        "expected_events": expected_events,
        "event_count_matches": (
            expected_events is None or expected_events == len(sidecar.events)
        ),
    }


def _dropped_event_count(sidecar: Sidecar) -> int:
    if sidecar.summary is not None:
        return int(sidecar.summary.get("dropped_events", 0))
    return int(sidecar.metadata.get("dropped_events", 0))


def _sidecar_paths(sidecars: Iterable[Sidecar]) -> Iterable[tuple[Path, Sidecar]]:
    for sidecar in sidecars:
        yield sidecar.path, sidecar


def _process_metadata_events(sidecar: Sidecar) -> list[dict[str, object]]:
    identity = sidecar.identity
    process_name = (
        f"{identity['role']} rank={identity['role_rank']} "
        f"host={identity['hostname']} pid={identity['pid']}"
    )
    return [
        {
            "name": "process_name",
            "ph": "M",
            "pid": sidecar.pid,
            "tid": 0,
            "args": {"name": process_name},
        },
        {
            "name": "thread_name",
            "ph": "M",
            "pid": sidecar.pid,
            "tid": 1,
            "args": {"name": "AFD correlation events"},
        },
    ]


def _sidecar_trace_events(
    sidecar: Sidecar,
    origin_ns: int,
    flow_events: dict[str, list[tuple[Sidecar, dict[str, object], int]]],
    incomplete_ranges: list[dict[str, object]],
) -> list[dict[str, object]]:
    if sidecar.transform is None:
        raise RuntimeError("clock transform has not been assigned")
    output: list[dict[str, object]] = []
    open_events: dict[tuple[str, str | None], list[tuple[dict[str, object], int]]] = (
        defaultdict(list)
    )
    for event in sidecar.events:
        global_ns = sidecar.transform.to_global_ns(int(event["monotonic_ns"]))
        flow_id = event.get("flow_id")
        if isinstance(flow_id, str):
            flow_events[flow_id].append((sidecar, event, global_ns))
        key = (str(event["event"]), flow_id if isinstance(flow_id, str) else None)
        if event["phase"] == "begin":
            open_events[key].append((event, global_ns))
        elif event["phase"] == "end":
            if not open_events[key]:
                incomplete_ranges.append(
                    _incomplete_range(sidecar, event, missing_phase="begin"),
                )
                continue
            begin_event, begin_ns = open_events[key].pop(0)
            output.append(
                _complete_trace_event(
                    sidecar,
                    begin_event,
                    begin_ns,
                    global_ns,
                    origin_ns,
                ),
            )
        elif event["phase"] == "instant":
            output.append(
                {
                    "name": event["event"],
                    "cat": "afd.correlation",
                    "ph": "i",
                    "s": "t",
                    "pid": sidecar.pid,
                    "tid": 1,
                    "ts": (global_ns - origin_ns) / NANOSECONDS_PER_MICROSECOND,
                    "args": _event_args(event),
                },
            )
    for unmatched in open_events.values():
        incomplete_ranges.extend(
            _incomplete_range(sidecar, event, missing_phase="end")
            for event, _ in unmatched
        )
    return output


def _incomplete_range(
    sidecar: Sidecar,
    event: dict[str, object],
    *,
    missing_phase: str,
) -> dict[str, object]:
    return {
        "sidecar": str(sidecar.path),
        "event": event["event"],
        "missing_phase": missing_phase,
        **_event_args(event),
    }


def _complete_trace_event(
    sidecar: Sidecar,
    event: dict[str, object],
    begin_ns: int,
    end_ns: int,
    origin_ns: int,
) -> dict[str, object]:
    return {
        "name": event["event"],
        "cat": "afd.correlation",
        "ph": "X",
        "pid": sidecar.pid,
        "tid": 1,
        "ts": (begin_ns - origin_ns) / NANOSECONDS_PER_MICROSECOND,
        "dur": max(0, end_ns - begin_ns) / NANOSECONDS_PER_MICROSECOND,
        "args": _event_args(event),
    }


def _event_args(event: dict[str, object]) -> dict[str, object]:
    return {
        key: event.get(key)
        for key in (
            "flow_id",
            "transaction_id",
            "layer_idx",
            "stage_idx",
            "num_tokens",
            "outcome",
        )
        if event.get(key) is not None
    }


def _build_flow_events(
    flow_events: dict[str, list[tuple[Sidecar, dict[str, object], int]]],
    origin_ns: int,
    sidecars: list[Sidecar],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    trace_events: list[dict[str, object]] = []
    incomplete: list[dict[str, object]] = []
    reversed_flows: list[dict[str, object]] = []
    for flow_index, (flow_id, entries) in enumerate(sorted(flow_events.items())):
        present_events = {str(event["event"]) for _, event, _ in entries}
        missing = sorted(REQUIRED_FLOW_EVENTS - present_events)
        present_phases = {
            (sidecar.pid, str(event["event"]), str(event["phase"]))
            for sidecar, event, _ in entries
        }
        # Expert routing and DP topology mean not every flow passes through
        # every rank; only flag ranks that participated in this flow but have
        # an unpaired begin/end phase for an event they started.
        participating_pids = {
            sidecar.pid for sidecar, _, _ in entries
        }
        missing_participants = [
            {
                "sidecar": str(sidecar.path),
                "role": sidecar.identity["role"],
                "role_rank": sidecar.identity["role_rank"],
                "event": event_name,
                "missing_phases": [
                    phase
                    for phase in ("begin", "end")
                    if (sidecar.pid, event_name, phase) not in present_phases
                ],
            }
            for sidecar in sidecars
            if sidecar.pid in participating_pids
            for event_name in present_events
            if any(
                (sidecar.pid, event_name, phase) in present_phases
                for phase in ("begin", "end")
            )
            and any(
                (sidecar.pid, event_name, phase) not in present_phases
                for phase in ("begin", "end")
            )
        ]
        if missing or missing_participants:
            incomplete.append(
                {
                    "flow_id": flow_id,
                    "missing_events": missing,
                    "missing_participants": missing_participants,
                },
            )

        trace_events.extend(
            _logical_flow_span(
                entries,
                source_event="afd.cam.dispatch_send",
                destination_event="afd.cam.dispatch_recv",
                chrome_flow_id=flow_index * 2,
                origin_ns=origin_ns,
                reversed_flows=reversed_flows,
            ),
        )
        trace_events.extend(
            _logical_flow_span(
                entries,
                source_event="afd.cam.combine_send",
                destination_event="afd.cam.combine_recv",
                chrome_flow_id=flow_index * 2 + 1,
                origin_ns=origin_ns,
                reversed_flows=reversed_flows,
            ),
        )
    return trace_events, {
        "total": len(flow_events),
        "complete": len(flow_events) - len(incomplete),
        "incomplete": incomplete,
        "reversed_clock_order": reversed_flows,
    }


def _logical_flow_span(
    entries: list[tuple[Sidecar, dict[str, object], int]],
    *,
    source_event: str,
    destination_event: str,
    chrome_flow_id: int,
    origin_ns: int,
    reversed_flows: list[dict[str, object]],
) -> list[dict[str, object]]:
    sources = [entry for entry in entries if entry[1]["event"] == source_event]
    destinations = [
        entry for entry in entries if entry[1]["event"] == destination_event
    ]
    if not sources or not destinations:
        return []
    source = min(sources, key=lambda entry: entry[2])
    destination = max(destinations, key=lambda entry: entry[2])
    flow_id = str(source[1]["flow_id"])
    if destination[2] < source[2]:
        reversed_flows.append(
            {
                "flow_id": flow_id,
                "source_event": source_event,
                "destination_event": destination_event,
            },
        )
    # Legacy Chrome/Perfetto flows are keyed by category + name + ID. The
    # endpoint slices retain their own event names; the synthetic endpoints
    # need one shared name so the importer joins them into a visible arrow.
    flow_name = f"{source_event} -> {destination_event}"
    return [
        _chrome_flow_endpoint(
            source,
            chrome_flow_id,
            "s",
            origin_ns,
            flow_name=flow_name,
        ),
        _chrome_flow_endpoint(
            destination,
            chrome_flow_id,
            "f",
            origin_ns,
            flow_name=flow_name,
        ),
    ]


def _chrome_flow_endpoint(
    entry: tuple[Sidecar, dict[str, object], int],
    chrome_flow_id: int,
    phase: str,
    origin_ns: int,
    *,
    flow_name: str,
) -> dict[str, object]:
    sidecar, event, global_ns = entry
    return {
        "name": flow_name,
        "cat": "afd.flow",
        "ph": phase,
        "id": chrome_flow_id,
        "bp": "e",
        "pid": sidecar.pid,
        "tid": 1,
        "ts": (global_ns - origin_ns) / NANOSECONDS_PER_MICROSECOND,
        "args": {"flow_id": event["flow_id"]},
    }


def _load_profiler_events(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"{path}: expected traceEvents list")
    return [event for event in events if isinstance(event, dict)]


def _align_profiler_events(
    profiler_events: list[dict[str, object]],
    sidecar: Sidecar,
    origin_ns: int,
    *,
    first_pid: int,
) -> tuple[list[dict[str, object]], dict[str, object], int]:
    marker_times: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    for event in profiler_events:
        marker = FLOW_MARKER_PATTERN.match(str(event.get("name", "")))
        if marker is not None and "ts" in event:
            # msprof exports ts AND dur in microseconds (epoch us for ts).
            marker_times[(marker["event"], marker["flow_id"])].append(
                float(event["ts"]),
            )

    for matches in marker_times.values():
        matches.sort()

    if sidecar.transform is None:
        raise RuntimeError("clock transform has not been assigned")
    deltas_us: list[float] = []
    for event in sidecar.events:
        key = (str(event["event"]), event.get("flow_id"))
        matches = marker_times.get(key)
        if event["phase"] != "begin" or not matches:
            continue
        global_us = (
            sidecar.transform.to_global_ns(int(event["monotonic_ns"])) - origin_ns
        ) / NANOSECONDS_PER_MICROSECOND
        deltas_us.append(global_us - matches.pop(0))

    if not deltas_us:
        return (
            [],
            {
                "sidecar": str(sidecar.path),
                "status": "unaligned",
                "reason": "no matching AFD MSTX markers",
            },
            0,
        )
    shift_us = statistics.median(deltas_us)
    spread_us = max(deltas_us) - min(deltas_us)

    original_pids = sorted({int(event.get("pid", 0)) for event in profiler_events})
    pid_map = {pid: first_pid + index for index, pid in enumerate(original_pids)}
    aligned: list[dict[str, object]] = []
    for original in profiler_events:
        event = dict(original)
        event["pid"] = pid_map[int(event.get("pid", 0))]
        args = event.get("args")
        event["args"] = {
            **(args if isinstance(args, dict) else {}),
            CORRELATION_PID_ARG: sidecar.pid,
        }
        if "ts" in event:
            # ts and dur are both us (epoch us for ts); shift to the merged
            # origin so device events sit on the correlation axis.
            event["ts"] = float(event["ts"]) + shift_us
        aligned.append(event)
    return (
        aligned,
        {
            "sidecar": str(sidecar.path),
            "status": "aligned",
            "matched_markers": len(deltas_us),
            "marker_delta_spread_us": spread_us,
        },
        len(original_pids),
    )


def _expand_sidecars(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("afd-trace-*.jsonl")))
            expanded.extend(sorted(path.rglob("afd-trace-*.jsonl.tmp")))
        else:
            expanded.append(path)
    return expanded


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--clock-sync", type=Path, action="append", default=[])
    parser.add_argument(
        "--profiler-trace",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("SIDECAR", "TRACE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    sidecars = [load_sidecar(path) for path in _expand_sidecars(args.sidecar)]
    if not sidecars:
        raise ValueError("no sidecar files found")
    session_ids = {sidecar.metadata.get("session_id") for sidecar in sidecars}
    if len(session_ids) != 1:
        raise ValueError(f"sidecars contain multiple session IDs: {session_ids}")
    clock_syncs = [load_clock_sync(path) for path in args.clock_sync]
    if any(
        clock_sync.get("session_id") not in session_ids for clock_sync in clock_syncs
    ):
        raise ValueError("clock-sync and sidecar session IDs differ")
    validate_clock_sync_coverage(sidecars, clock_syncs)
    clock_diagnostics = assign_clock_transforms(sidecars, clock_syncs)
    trace, report = build_merged_trace(sidecars, args.profiler_trace)
    report["clock_alignment"] = clock_diagnostics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
