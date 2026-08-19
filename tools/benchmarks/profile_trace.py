# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Summarize and compare Chrome/TensorBoard traces from isolated NPU replays."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CAM_WAIT_PATTERN = re.compile(r"notify[_ -]?wait", re.IGNORECASE)
CAM_SIGNAL_PATTERN = re.compile(
    r"notify[_ -]?record|inter[_ -]?rank[_ -]?record|write[_ -]?value",
    re.IGNORECASE,
)
CAM_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    # Fine-grained CAM cross-role communication (plan section 9.4). Classified
    # before generic communication so CAM stages stay distinguishable.
    "cam_dispatch_send": re.compile(
        r"async[_ -]?dispatch[_ -]?send|dispatch[_ -]?send", re.IGNORECASE
    ),
    "cam_dispatch_recv": re.compile(
        r"async[_ -]?dispatch[_ -]?recv|dispatch[_ -]?recv", re.IGNORECASE
    ),
    "cam_combine_send": re.compile(
        r"async[_ -]?combine[_ -]?send|combine[_ -]?send", re.IGNORECASE
    ),
    "cam_combine_recv": re.compile(
        r"async[_ -]?combine[_ -]?recv|combine[_ -]?recv", re.IGNORECASE
    ),
}
COMMUNICATION_PATTERN = re.compile(
    r"hccl|all[_ -]?(reduce|gather|toall)|reduce[_ -]?scatter|"
    r"dispatch|combine|send|recv|communicat",
    re.IGNORECASE,
)
MEMORY_PATTERN = re.compile(
    r"memcpy|memset|copy|\bdma\b|transpose|permute",
    re.IGNORECASE,
)
ATTENTION_PATTERN = re.compile(
    r"attention|flash[_ -]?attn|flashattention|mla|rope|rotary|kv[_ -]?cache",
    re.IGNORECASE,
)
FFN_MOE_PATTERN = re.compile(
    r"\bmoe\b|expert|grouped[_ -]?matmul|swiglu|\bffn\b|feed[_ -]?forward|"
    r"routing|topk",
    re.IGNORECASE,
)
COMPUTE_PATTERN = re.compile(
    r"matmul|gemm|cube|vector|kernel|aicore|acl",
    re.IGNORECASE,
)
HOST_PATTERN = re.compile(
    r"python|scheduler|execute[_ -]?model|model[_ -]?forward|cpu_op|enqueue",
    re.IGNORECASE,
)
TOP_EVENT_LIMIT = 30
MICROSECONDS_PER_MILLISECOND = 1_000.0
TRACE_SUMMARY_SCHEMA_VERSION = 1
HASH_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TraceEvent:
    """One complete duration event in a Chrome trace."""

    name: str
    category: str
    process_id: str
    thread_id: str
    start_us: float
    duration_us: float

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    @property
    def lane(self) -> tuple[str, str]:
        return self.process_id, self.thread_id


def _read_trace_json(trace_path: Path) -> object:
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rt", encoding="utf-8") as trace_file:
            return json.load(trace_file)
    return json.loads(trace_path.read_text(encoding="utf-8"))


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace_events(trace_path: Path) -> list[TraceEvent]:
    """Load complete-duration events from a Chrome trace JSON file."""
    trace_json = _read_trace_json(trace_path)
    if isinstance(trace_json, dict):
        raw_events = trace_json.get("traceEvents")
    else:
        raw_events = trace_json
    if not isinstance(raw_events, list):
        raise ValueError(f"{trace_path} does not contain a traceEvents list.")

    events: list[TraceEvent] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict) or raw_event.get("ph") != "X":
            continue
        name = raw_event.get("name")
        category = raw_event.get("cat", "")
        # msprof-exported timelines encode ts/dur as decimal strings.
        try:
            start_us = float(raw_event.get("ts"))
            duration_us = float(raw_event.get("dur"))
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(name, str)
            or not isinstance(category, str)
            or duration_us <= 0
        ):
            continue
        events.append(
            TraceEvent(
                name=name,
                category=category,
                process_id=str(raw_event.get("pid", "")),
                thread_id=str(raw_event.get("tid", "")),
                start_us=float(start_us),
                duration_us=float(duration_us),
            )
        )
    if not events:
        raise ValueError(f"{trace_path} contains no complete duration events.")
    return events


def classify_event(event: TraceEvent) -> str:
    """Classify an event using stable, inspectable name/category rules."""
    searchable_text = f"{event.category} {event.name}"
    if CAM_WAIT_PATTERN.search(searchable_text):
        return "cam_wait"
    if CAM_SIGNAL_PATTERN.search(searchable_text):
        return "cam_signal"
    for cam_category, cam_pattern in CAM_CATEGORY_PATTERNS.items():
        if cam_pattern.search(searchable_text):
            return cam_category
    if COMMUNICATION_PATTERN.search(searchable_text):
        return "communication"
    if MEMORY_PATTERN.search(searchable_text):
        return "memory"
    if ATTENTION_PATTERN.search(searchable_text):
        return "attention"
    if FFN_MOE_PATTERN.search(searchable_text):
        return "ffn_moe"
    if COMPUTE_PATTERN.search(searchable_text):
        return "other_compute"
    if HOST_PATTERN.search(searchable_text):
        return "host"
    return "unclassified"


OVERLAY_PROCESS_NAME = "Overlap Analysis"


def load_trace_metadata(
    trace_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read pid -> label and pid -> process name from metadata events.

    msprof timelines label processes "NPU <id>" (device) or "CPU" (host);
    torch-profiler Chrome traces may carry no labels at all.
    """
    trace_json = _read_trace_json(trace_path)
    raw_events = (
        trace_json.get("traceEvents") if isinstance(trace_json, dict) else trace_json
    )
    pid_labels: dict[str, str] = {}
    pid_names: dict[str, str] = {}
    if not isinstance(raw_events, list):
        return pid_labels, pid_names
    for raw_event in raw_events:
        if not isinstance(raw_event, dict) or raw_event.get("ph") != "M":
            continue
        meta_name = raw_event.get("name")
        args = raw_event.get("args")
        if not isinstance(args, dict):
            continue
        pid = str(raw_event.get("pid", ""))
        if meta_name == "process_labels" and isinstance(args.get("labels"), str):
            pid_labels[pid] = args["labels"]
        elif meta_name == "process_name" and isinstance(args.get("name"), str):
            pid_names[pid] = args["name"]
    return pid_labels, pid_names


DEVICE_CATEGORY_HINTS = ("kernel", "npu", "memcpy", "memset", "aicore", "aic")


def _is_device_event(
    event: TraceEvent,
    pid_labels: dict[str, str],
    pid_names: dict[str, str],
) -> bool:
    """Device-lane detection: msprof process labels, with cat-name fallback."""
    label = pid_labels.get(event.process_id, "")
    if label:
        return label.upper().startswith("NPU")
    if pid_names.get(event.process_id) == OVERLAY_PROCESS_NAME:
        return False
    category_text = event.category.lower()
    return any(hint in category_text for hint in DEVICE_CATEGORY_HINTS)


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return []
    merged_intervals = [sorted_intervals[0]]
    for start_us, end_us in sorted_intervals[1:]:
        previous_start_us, previous_end_us = merged_intervals[-1]
        if start_us <= previous_end_us:
            merged_intervals[-1] = (
                previous_start_us,
                max(previous_end_us, end_us),
            )
        else:
            merged_intervals.append((start_us, end_us))
    return merged_intervals


def _interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end_us - start_us for start_us, end_us in intervals)


def _intersection_duration(
    left_intervals: Sequence[tuple[float, float]],
    right_intervals: Sequence[tuple[float, float]],
) -> float:
    left_index = 0
    right_index = 0
    overlap_us = 0.0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        overlap_us += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap_us


def summarize_trace(trace_path: Path) -> dict[str, object]:
    """Produce overlap-aware category and top-event metrics for one trace."""
    events = load_trace_events(trace_path)
    pid_labels, pid_names = load_trace_metadata(trace_path)
    trace_start_us = min(event.start_us for event in events)
    trace_end_us = max(event.end_us for event in events)
    category_events: dict[str, list[TraceEvent]] = {}
    for event in events:
        if pid_names.get(event.process_id) == OVERLAY_PROCESS_NAME:
            # Pre-aggregated overlap lanes duplicate the raw events; keep them
            # out of every union to avoid double counting.
            continue
        category_events.setdefault(classify_event(event), []).append(event)

    category_summary: dict[str, dict[str, float | int]] = {}
    for category_name, selected_events in sorted(category_events.items()):
        global_intervals = _merge_intervals(
            (event.start_us, event.end_us) for event in selected_events
        )
        lanes: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for event in selected_events:
            lanes.setdefault(event.lane, []).append((event.start_us, event.end_us))
        category_summary[category_name] = {
            "event_count": len(selected_events),
            "event_time_ms": (
                sum(event.duration_us for event in selected_events)
                / MICROSECONDS_PER_MILLISECOND
            ),
            "global_union_ms": (
                _interval_duration(global_intervals) / MICROSECONDS_PER_MILLISECOND
            ),
            "lane_union_ms": (
                sum(
                    _interval_duration(_merge_intervals(lane_intervals))
                    for lane_intervals in lanes.values()
                )
                / MICROSECONDS_PER_MILLISECOND
            ),
        }

    communication_intervals = _merge_intervals(
        (event.start_us, event.end_us)
        for event in category_events.get("communication", [])
    )
    compute_intervals = _merge_intervals(
        (event.start_us, event.end_us)
        for category_name in ("attention", "ffn_moe", "other_compute")
        for event in category_events.get(category_name, [])
    )
    communication_union_us = _interval_duration(communication_intervals)
    communication_compute_overlap_us = _intersection_duration(
        communication_intervals,
        compute_intervals,
    )

    device_events = [
        event
        for event in events
        if pid_names.get(event.process_id) != OVERLAY_PROCESS_NAME
        and _is_device_event(event, pid_labels, pid_names)
    ]
    device_wait_events = [
        event for event in device_events if classify_event(event) == "cam_wait"
    ]
    device_busy_events = [
        event for event in device_events if classify_event(event) != "cam_wait"
    ]
    device_busy_us = _interval_duration(
        _merge_intervals((event.start_us, event.end_us) for event in device_busy_events)
    )
    device_wait_us = _interval_duration(
        _merge_intervals((event.start_us, event.end_us) for event in device_wait_events)
    )
    device_active_us = _interval_duration(
        _merge_intervals((event.start_us, event.end_us) for event in device_events)
    )
    trace_span_us = trace_end_us - trace_start_us
    busy_occupancy: dict[str, object] = {
        "device_event_count": len(device_events),
        "device_busy_ms": device_busy_us / MICROSECONDS_PER_MILLISECOND,
        "device_cam_wait_ms": device_wait_us / MICROSECONDS_PER_MILLISECOND,
        "busy_ratio": (device_busy_us / trace_span_us) if trace_span_us else None,
        "cam_wait_ratio": (device_wait_us / trace_span_us) if trace_span_us else None,
        "bubble_ratio": (
            max(0.0, 1.0 - device_active_us / trace_span_us)
            if trace_span_us
            else None
        ),
    }

    duration_by_name: Counter[str] = Counter()
    count_by_name: Counter[str] = Counter()
    for event in events:
        duration_by_name[event.name] += event.duration_us
        count_by_name[event.name] += 1
    top_event_names = [
        event_name for event_name, _ in duration_by_name.most_common(TOP_EVENT_LIMIT)
    ]

    return {
        "schema_version": TRACE_SUMMARY_SCHEMA_VERSION,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256_file(trace_path),
        "trace_span_ms": (
            (trace_end_us - trace_start_us) / MICROSECONDS_PER_MILLISECOND
        ),
        "complete_event_count": len(events),
        "lane_count": len({event.lane for event in events}),
        "categories": category_summary,
        "busy_occupancy": busy_occupancy,
        "communication_compute_overlap": {
            "overlap_ms": (
                communication_compute_overlap_us / MICROSECONDS_PER_MILLISECOND
            ),
            "communication_union_ms": (
                communication_union_us / MICROSECONDS_PER_MILLISECOND
            ),
            "overlap_ratio_of_communication": (
                communication_compute_overlap_us / communication_union_us
                if communication_union_us
                else None
            ),
        },
        "top_events": [
            {
                "name": event_name,
                "count": count_by_name[event_name],
                "event_time_ms": (
                    duration_by_name[event_name] / MICROSECONDS_PER_MILLISECOND
                ),
            }
            for event_name in top_event_names
        ],
        "interpretation_warning": (
            "Event sums can double-count nested or parallel work. Use global_union_ms, "
            "lane_union_ms, the visual timeline, and matched replay TTFT together."
        ),
    }


def _numeric_delta(
    baseline_value: float | int,
    candidate_value: float | int,
) -> dict[str, float]:
    baseline_float = float(baseline_value)
    candidate_float = float(candidate_value)
    return {
        "baseline": baseline_float,
        "candidate": candidate_float,
        "delta": candidate_float - baseline_float,
        "delta_percent": (
            (candidate_float - baseline_float) / baseline_float * 100
            if baseline_float
            else 0.0
        ),
    }


def compare_trace_summaries(
    baseline_summary: dict[str, object],
    candidate_summary: dict[str, object],
) -> dict[str, object]:
    """Compare stable numeric fields from two trace summaries."""
    baseline_categories = baseline_summary.get("categories")
    candidate_categories = candidate_summary.get("categories")
    if not isinstance(baseline_categories, dict) or not isinstance(
        candidate_categories,
        dict,
    ):
        raise ValueError("Both trace summaries must contain categories.")

    category_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for category_name in sorted(set(baseline_categories) | set(candidate_categories)):
        baseline_category = baseline_categories.get(category_name, {})
        candidate_category = candidate_categories.get(category_name, {})
        if not isinstance(baseline_category, dict) or not isinstance(
            candidate_category,
            dict,
        ):
            continue
        metric_comparison: dict[str, dict[str, float]] = {}
        for metric_name in ("event_count", "event_time_ms", "global_union_ms"):
            baseline_value = baseline_category.get(metric_name, 0)
            candidate_value = candidate_category.get(metric_name, 0)
            if isinstance(baseline_value, (int, float)) and isinstance(
                candidate_value,
                (int, float),
            ):
                metric_comparison[metric_name] = _numeric_delta(
                    baseline_value,
                    candidate_value,
                )
        category_comparison[category_name] = metric_comparison

    baseline_span = baseline_summary.get("trace_span_ms")
    candidate_span = candidate_summary.get("trace_span_ms")
    if not isinstance(baseline_span, (int, float)) or not isinstance(
        candidate_span,
        (int, float),
    ):
        raise ValueError("Both trace summaries must contain trace_span_ms.")
    return {
        "schema_version": TRACE_SUMMARY_SCHEMA_VERSION,
        "baseline_trace": baseline_summary.get("trace_path"),
        "candidate_trace": candidate_summary.get("trace_path"),
        "trace_span_ms": _numeric_delta(baseline_span, candidate_span),
        "categories": category_comparison,
        "attribution_gate": (
            "Treat deltas as causal only after a matched replay, visual timeline "
            "inspection, rollback ablation, and end-to-end confirmation."
        ),
    }


def _write_json(output_path: Path, value: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--trace", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    if args.command == "summarize":
        result = summarize_trace(args.trace)
    else:
        result = compare_trace_summaries(
            summarize_trace(args.baseline),
            summarize_trace(args.candidate),
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
