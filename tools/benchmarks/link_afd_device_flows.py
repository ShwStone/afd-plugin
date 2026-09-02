# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Post-process a merged AFD trace to link correlation markers to CANN device ops.

The merge tool (``merge_afd_correlation_traces.py``) produces a Chrome trace that
contains, side by side and already time-aligned:

  * correlation markers (``afd.cam.*`` X events on the sidecar pid, e.g. pid 1/2),
  * the MSTX user-range markers from the torch_npu trace (names like
    ``afd.cam.dispatch_send flow_id=<hex>``, on the torch_npu host pid),
  * the CANN device ops (``CamMoeDistributeDispatchSend``, etc., on the device
    lane pid).

Each MSTX marker enqueues exactly one device op, so pairing the two lists by
monotonic ts (FIFO) is an exact 1:1 match.  This script turns that pairing into
Chrome flow arrows (``ph`` "s"/"f") that connect the *correlation* marker to the
*device* op, using the same ``bp: "e"`` convention as the correlation flows so
they render in Perfetto / chrome://tracing.  Both endpoints use the same name,
category, and ID because all three fields are part of the legacy flow identity.

The merge tool invokes this linker automatically when profiler traces are
provided.  The standalone command remains useful for already-merged traces::

    python3 -m tools.benchmarks.link_afd_device_flows \
        merged.json -o merged_with_device_flows.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

_MSTX_PATTERN = re.compile(r"^(?P<event>afd\.[^ ]+) flow_id=(?P<flow_id>[0-9a-f]+)$")

# mstx event name -> CANN device op name it enqueues.  The counts on each side
# must be equal for an exact FIFO pairing (they are, per rank).
EVENT_TO_DEVICE_OP: dict[str, str] = {
    "afd.cam.dispatch_send": "CamMoeDistributeDispatchSend",
    "afd.cam.combine_recv": "CamMoeDistributeCombineRecv",
    "afd.cam.dispatch_recv": "CamMoeDistributeDispatchRecv",
    "afd.cam.combine_send": "CamMoeDistributeCombineSend",
}
DEVICE_FLOW_CATEGORY = "afd.device-flow"
CORRELATION_PID_ARG = "afd_correlation_pid"

# First id reserved for device-op flows; the merge tool's correlation flows use
# small sequential ids, so start well above them to avoid collisions.
_ID_BASE = 1_000_000


def _load_events(path: Path) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as input_file:
        payload = json.load(input_file)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"{path}: expected a traceEvents list")
    return [event for event in events if isinstance(event, dict)]


def _flow_id_of_mstx(name: str) -> str | None:
    match = _MSTX_PATTERN.match(name)
    return match.group("flow_id") if match else None


def _pair_fifo(
    mstx: dict[str, list[dict[str, object]]],
    device_ops: list[dict[str, object]],
) -> list[tuple[str, str, dict[str, object], dict[str, object]]]:
    """Pair each mstx marker (by event, in ts order) to the matching device op.

    Returns ``(event_name, flow_id, mstx_marker, device_op)`` in FIFO order.
    """
    pairs: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
    for event_name, device_op_name in EVENT_TO_DEVICE_OP.items():
        markers = sorted(mstx.get(event_name, []), key=lambda e: float(e["ts"]))
        ops = sorted(
            (e for e in device_ops if e.get("name") == device_op_name),
            key=lambda e: float(e["ts"]),
        )
        if len(markers) != len(ops):
            print(
                f"WARNING: {event_name}: {len(markers)} markers vs "
                f"{len(ops)} {device_op_name} ops — pairing first "
                f"{min(len(markers), len(ops))} by FIFO",
                file=sys.stderr,
            )
        for marker, op in zip(markers, ops, strict=False):
            flow_id = _flow_id_of_mstx(str(marker.get("name", "")))
            if flow_id is None:
                continue
            pairs.append((event_name, flow_id, marker, op))
    return pairs


def _correlation_markers(
    events: list[dict[str, object]],
) -> tuple[
    dict[tuple[int, str, str], dict[str, object]],
    dict[tuple[str, str], list[dict[str, object]]],
]:
    """Index correlation slices by process and by logical marker."""

    by_process: dict[tuple[int, str, str], dict[str, object]] = {}
    by_marker: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "afd.correlation":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        flow_id = args.get("flow_id")
        if isinstance(flow_id, str):
            marker = (str(event.get("name")), flow_id)
            by_process[(int(event["pid"]), *marker)] = event
            by_marker[marker].append(event)
    return by_process, by_marker


def _profiler_group(event: dict[str, object]) -> int | None:
    args = event.get("args")
    if not isinstance(args, dict) or CORRELATION_PID_ARG not in args:
        return None
    return int(args[CORRELATION_PID_ARG])


def _flow_endpoint(
    *,
    name: str,
    flow_id: str,
    chrome_flow_id: int,
    phase: str,
    pid: object,
    tid: object,
    ts: float,
) -> dict[str, object]:
    return {
        "name": name,
        "cat": DEVICE_FLOW_CATEGORY,
        "ph": phase,
        "id": chrome_flow_id,
        "bp": "e",
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "args": {"flow_id": flow_id},
    }


def build_device_flows(
    events: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mstx: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        flow_id = _flow_id_of_mstx(str(event.get("name", "")))
        if flow_id is not None:
            event_name = str(event.get("name", "")).split(" flow_id=")[0]
            mstx[event_name].append(event)

    device_ops = [
        event for event in events if event.get("name") in EVENT_TO_DEVICE_OP.values()
    ]
    corr_by_process, corr_by_marker = _correlation_markers(events)
    profiler_groups = {
        _profiler_group(event) for markers in mstx.values() for event in markers
    }

    flows: list[dict[str, object]] = []
    linked = 0
    skipped = 0
    ambiguous = 0
    for profiler_group in sorted(
        profiler_groups,
        key=lambda group: -1 if group is None else group,
    ):
        grouped_mstx = {
            event_name: [
                event for event in markers if _profiler_group(event) == profiler_group
            ]
            for event_name, markers in mstx.items()
        }
        grouped_ops = [
            event for event in device_ops if _profiler_group(event) == profiler_group
        ]
        for event_name, flow_id, _mstx_marker, device_op in _pair_fifo(
            grouped_mstx,
            grouped_ops,
        ):
            marker = (event_name, flow_id)
            if profiler_group is None:
                candidates = corr_by_marker.get(marker, [])
                if len(candidates) > 1:
                    ambiguous += 1
                    continue
                corr_event = candidates[0] if candidates else None
            else:
                corr_event = corr_by_process.get((profiler_group, *marker))
            if corr_event is None:
                skipped += 1
                continue

            chrome_flow_id = _ID_BASE + linked
            # Legacy Chrome/Perfetto flows are keyed by category + name + ID.
            # Keep the descriptive endpoint names on their enclosing slices and
            # give both synthetic flow endpoints one shared flow name.
            flow_name = f"{event_name} -> {device_op['name']}"
            # Arrow always points forward in time: earlier endpoint is "s".
            corr_ts = float(corr_event["ts"])
            dev_ts = float(device_op["ts"])
            if corr_ts <= dev_ts:
                source, target = corr_event, device_op
            else:
                source, target = device_op, corr_event
            flows.append(
                _flow_endpoint(
                    name=flow_name,
                    flow_id=flow_id,
                    chrome_flow_id=chrome_flow_id,
                    phase="s",
                    pid=source["pid"],
                    tid=source["tid"],
                    ts=float(source["ts"]),
                ),
            )
            flows.append(
                _flow_endpoint(
                    name=flow_name,
                    flow_id=flow_id,
                    chrome_flow_id=chrome_flow_id,
                    phase="f",
                    pid=target["pid"],
                    tid=target["tid"],
                    ts=float(target["ts"]),
                ),
            )
            linked += 1

    report = {
        "linked_device_flows": linked,
        "skipped_missing_correlation": skipped,
        "skipped_ambiguous_correlation": ambiguous,
        "id_base": _ID_BASE,
    }
    return flows, report


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged", type=Path, help="merged trace JSON")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(list(argv))

    events = [
        event
        for event in _load_events(args.merged)
        if event.get("cat") != DEVICE_FLOW_CATEGORY
    ]
    flows, report = build_device_flows(events)

    payload = {"traceEvents": events + flows, "displayTimeUnit": "us"}
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
