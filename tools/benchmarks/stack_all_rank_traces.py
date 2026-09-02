# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Stack per-rank msprof timeline exports into ONE all-rank Chrome trace.

Motivation: the stage-1 profile replays pinned DP rank 0, so only one DP
group had profiler steps. The all-DP collection (fp_orchestrate phase=profile
with --dp-rank -1) profiles every rank; this script merges all per-rank
exports into a single big trace for Perfetto / chrome://tracing.

Design:

* rank dirs are discovered by the ``dp{D}_..._rank{G}_{pid}_{ts}_ascend_pt``
  directory naming under the trace root; each holds
  ``mindstudio_profiler_output/msprof_*.json`` after ``msprof export``.
* Only two lanes survive per rank (device compute + communication), events
  shorter than ``--min-dur-us`` are dropped — 32 ranks of raw Level2 exports
  would otherwise be hundreds of millions of events.
* pids are remapped to ``rank_index * 1000 + lane`` (0=device, 1=comm) with
  process_name metadata ``{role} rank{g} (dp{d} tp{t})``.
* Cross-node alignment is data-driven:
  - baseline uses lockstep HCCL collectives to estimate the peer-node shift;
  - AFD requires correlation sidecars and explicit four-timestamp clock-sync
    samples. Per-rank MSTX markers map device time to the calibrated sidecar
    axis. No host-clock-equality assumption is made.
* All timestamps are then normalized to a common origin (earliest event = 0).

Usage (on a pod, where the exports live):

    python3 -m tools.benchmarks.stack_all_rank_traces \
        --trace-root <...>/02_profiles_32/traces/baseline \
        --role baseline --output baseline_allranks.json

    python3 -m tools.benchmarks.stack_all_rank_traces \
        --trace-root <...>/02_profiles_32/traces/attention --role attention \
        --peer-root <...>/02_profiles_32/traces/ffn --peer-role ffn \
        --correlation-dir <...>/traces/correlation/afd_a2/<session> \
        --clock-sync <...>/clock-sync-node1.json \
        --output afd_a2_allranks.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__:
    from .merge_afd_correlation_traces import (
        assign_clock_transforms,
        load_clock_sync,
        load_sidecar,
        validate_clock_sync_coverage,
    )
else:
    from merge_afd_correlation_traces import (
        assign_clock_transforms,
        load_clock_sync,
        load_sidecar,
        validate_clock_sync_coverage,
    )

RANK_DIR_PATTERN = re.compile(
    r"dp(?P<dp>\d+)_pp\d+_tp(?P<tp>\d+)_dcp\d+_ep(?P<ep>\d+)_rank(?P<rank>\d+)_"
)
# AFD plugin profiler dirs: <hostname>_<pid>_<ts>_ascend_pt (no rank in the
# name) — rank comes from profiler_metadata.json instead.
KEEP_PROCESSES = ("Ascend Hardware", "Communication")
LANE_ID = {"Ascend Hardware": 0, "Communication": 1}

_TRACE_DIR_PID_RE = re.compile(
    r"_(?P<pid>\d+)_(?P<ts>\d+)_ascend_pt$",
)


def _rank_identity(
    rank_dir: Path,
    role: str,
    pid_map: dict[tuple[str, int], tuple[str, int]] | None = None,
) -> tuple[int, int, int] | None:
    """Return (rank, dp, tp) for one rank dir.

    vllm profiler dirs carry dp/tp/rank in the name. AFD plugin profiler
    dirs don't — read profiler_metadata.json: the afd_async_cam group's
    group_rank is the role rank (attention: dp*8+tp; ffn: dp). If metadata
    is missing, fall back to the sidecar process identity.
    """
    m = RANK_DIR_PATTERN.match(rank_dir.name)
    if m:
        # the "rank" field is TP-local (0-7, repeated per dp group); the "ep"
        # field is the global EP rank (dp*8+tp) — use it as the global rank.
        return int(m.group("ep")), int(m.group("dp")), int(m.group("tp"))
    meta_path = rank_dir / "profiler_metadata.json"
    rank = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        for key, group in meta.get("parallel_group_info", {}).items():
            if "afd_async_cam" in key:
                rank = int(group["group_rank"])
        if rank is None:
            # fall back: tp group's global_ranks[tp_rank]
            for group in meta.get("parallel_group_info", {}).values():
                if group.get("group_name") == "tp":
                    rank = int(group["global_ranks"][int(group["group_rank"])])
    if rank is None and pid_map:
        trace_process = _TRACE_DIR_PID_RE.search(rank_dir.name)
        if trace_process:
            host_prefix = rank_dir.name[: trace_process.start()]
            host_candidates = (host_prefix, host_prefix.split(".", maxsplit=1)[0])
            worker_pid = int(trace_process.group("pid"))
            for hostname in host_candidates:
                hit = pid_map.get((hostname, worker_pid))
                if hit is not None:
                    rank = hit[1]
                    break
    if rank is None:
        return None
    if role == "attention":
        return rank, rank // 8, rank % 8
    if role == "ffn":
        return rank, rank, 0
    return rank, 0, rank


def _load_rank_trace(rank_dir: Path) -> tuple[list[dict], dict[int, str]]:
    """Load the msprof timeline export inside one rank dir.

    Exports live at ``<rank_dir>/PROF_*/mindstudio_profiler_output/msprof_*.json``
    (the PROF subdir is created by the profiler, not by us).
    """
    candidates = [
        p
        for p in sorted(
            rank_dir.glob("PROF_*/mindstudio_profiler_output/msprof_*.json")
        )
        # msprof_tx_*.json is a separate text/record export — host events
        # only, no device timeline. Never pick it.
        if "_tx_" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"{rank_dir}: no PROF_*/mindstudio_profiler_output/msprof_*.json"
        )
    with open(candidates[-1], encoding="utf-8") as f:
        payload = json.load(f)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    pid_names: dict[int, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_names[e["pid"]] = str(e.get("args", {}).get("name", ""))
    return events, pid_names


def _filter_rank_events(
    events: list[dict],
    pid_names: dict[int, str],
    min_dur_us: float,
) -> list[dict]:
    """Keep device/communication X slices above the duration floor."""
    keep_pids = {pid for pid, name in pid_names.items() if name in KEEP_PROCESSES}
    kept = []
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in keep_pids:
            continue
        if float(e.get("dur", 0)) < min_dur_us:
            continue
        kept.append(e)
    return kept


_MSTX_RE = re.compile(r"^(?P<event>afd\.[^ ]+) flow_id=(?P<flow_id>[0-9a-f]+)$")


def _mstx_marker_times(
    raw_events: list[dict],
) -> dict[tuple[str, str], list[float]]:
    """Return marker starts on one profiler export's device-clock axis."""

    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in raw_events:
        match = _MSTX_RE.match(str(event.get("name", "")))
        if match and "ts" in event:
            out[(match.group("event"), match.group("flow_id"))].append(
                float(event["ts"]),
            )
    return out


def _load_correlation_events(
    corr_dir: Path,
    clock_sync_paths: list[Path],
) -> tuple[
    dict[tuple[str, int], dict],
    list[dict[str, object]],
    dict[tuple[str, int], tuple[str, int]],
]:
    """Load sidecars onto one calibrated cross-host clock axis."""

    paths = sorted(corr_dir.rglob("afd-trace-*.jsonl"))
    paths.extend(sorted(corr_dir.rglob("afd-trace-*.jsonl.tmp")))
    sidecars = [load_sidecar(path) for path in paths]
    if not sidecars:
        return {}, [], {}
    session_ids = {sidecar.metadata.get("session_id") for sidecar in sidecars}
    if len(session_ids) != 1:
        raise ValueError(f"sidecars contain multiple session IDs: {session_ids}")

    clock_syncs = [load_clock_sync(path) for path in clock_sync_paths]
    if any(
        clock_sync.get("session_id") not in session_ids for clock_sync in clock_syncs
    ):
        raise ValueError("clock-sync and sidecar session IDs differ")
    validate_clock_sync_coverage(sidecars, clock_syncs)
    diagnostics = assign_clock_transforms(sidecars, clock_syncs)

    output: dict[tuple[str, int], dict] = {}
    pid_map: dict[tuple[str, int], tuple[str, int]] = {}
    for sidecar in sidecars:
        if sidecar.transform is None:
            raise RuntimeError("clock transform has not been assigned")
        identity = sidecar.identity
        lane_key = (str(identity["role"]), int(identity["role_rank"]))
        pid_map[(sidecar.hostname, int(identity["pid"]))] = lane_key
        if lane_key in output:
            raise ValueError(
                f"multiple correlation sidecars for {lane_key}; select one session",
            )
        begins: dict[tuple[object, ...], list[tuple[float, dict]]] = defaultdict(list)
        begin_index: dict[tuple[str, str], list[float]] = defaultdict(list)
        slices: list[dict] = []
        for event in sidecar.events:
            global_us = (
                sidecar.transform.to_global_ns(
                    int(event["monotonic_ns"]),
                )
                / 1000.0
            )
            event_key = (
                event.get("event"),
                event.get("flow_id"),
                event.get("layer_idx"),
                event.get("stage_idx"),
                event.get("transaction_id"),
            )
            marker_key = (
                str(event.get("event")),
                str(event.get("flow_id")),
            )
            if event.get("phase") == "begin":
                begins[event_key].append((global_us, event))
                begin_index[marker_key].append(global_us)
            elif event.get("phase") == "end" and begins[event_key]:
                start_us, begin_event = begins[event_key].pop(0)
                slices.append(
                    {
                        "ph": "X",
                        "name": str(event.get("event")),
                        "ts": start_us,
                        "dur": max(global_us - start_us, 0.0),
                        "args": {
                            "flow_id": event.get("flow_id"),
                            "num_tokens": begin_event.get("num_tokens"),
                            "layer_idx": begin_event.get("layer_idx"),
                            "stage_idx": begin_event.get("stage_idx"),
                            "transaction_id": begin_event.get("transaction_id"),
                        },
                    },
                )
        for unmatched in begins.values():
            for start_us, begin_event in unmatched:
                slices.append(
                    {
                        "ph": "X",
                        "name": f"{begin_event.get('event')} (unmatched-begin)",
                        "ts": start_us,
                        "dur": 0.0,
                        "args": {
                            "flow_id": begin_event.get("flow_id"),
                            "num_tokens": begin_event.get("num_tokens"),
                            "layer_idx": begin_event.get("layer_idx"),
                        },
                    },
                )
        output[lane_key] = {
            "slices": slices,
            "begin_index": begin_index,
        }
    return output, diagnostics, pid_map


def _op_starts(
    events: list[dict],
    pid_names: dict[int, str],
    name_sub: str | None,
    exact: str | None,
) -> list[float]:
    starts = []
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if exact is not None and name != exact:
            continue
        if name_sub is not None and name_sub not in name:
            continue
        starts.append(float(e["ts"]))
    return sorted(starts)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _median_pair_offset(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Median (b - a) over the FIFO pairing of two sorted ts lists; returns
    (offset, spread_p95_of_abs_deviation, n_pairs)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0
    diffs = [b[i] - a[i] for i in range(n)]
    med = _median(diffs)
    dev = sorted(abs(d - med) for d in diffs)
    p95 = dev[min(len(dev) - 1, int(0.95 * (len(dev) - 1)))]
    return med, p95, n


def stack(
    roots: list[tuple[str, str, Path, int | None]],
    min_dur_us: float,
    align_op: str,
    session_ts: str | None,
    correlation_dir: Path | None = None,
    clock_sync_paths: list[Path] | None = None,
) -> tuple[list[dict], dict]:
    """Merge all (role, node, trace_root, node_split_rank) inputs.

    ``node_split_rank`` splits a single root's ranks across two nodes:
    rank < split -> the root's node, rank >= split -> the other node
    (baseline writes all 32 ranks into one NAS dir from both nodes).
    ``session_ts`` keeps only rank dirs whose name contains the capture
    timestamp (several sessions share the attention/ffn roots).
    """
    ranks: list[dict] = []
    requested_afd_roles = {
        role for role, _node, _root, _split in roots if role in {"attention", "ffn"}
    }
    corr_lanes: dict[tuple[str, int], dict] = {}
    clock_diagnostics: list[dict[str, object]] = []
    pid_map: dict[tuple[str, int], tuple[str, int]] = {}
    if correlation_dir is not None and correlation_dir.is_dir():
        corr_lanes, clock_diagnostics, pid_map = _load_correlation_events(
            correlation_dir,
            clock_sync_paths or [],
        )
    if requested_afd_roles and not corr_lanes:
        raise ValueError(
            "AFD stacking requires --correlation-dir with per-rank sidecars",
        )
    for role, node, root, split in roots:
        for rank_dir in sorted(root.iterdir()):
            if not rank_dir.is_dir() or not rank_dir.name.endswith("_ascend_pt"):
                continue
            if session_ts and session_ts not in rank_dir.name:
                continue
            ident = _rank_identity(rank_dir, role, pid_map)
            if ident is None:
                continue
            rank_id, dp, tp = ident
            try:
                events, pid_names = _load_rank_trace(rank_dir)
            except FileNotFoundError:
                continue
            kept = _filter_rank_events(events, pid_names, min_dur_us)
            if not kept:
                continue
            rank_node = node
            if split is not None and rank_id >= split:
                rank_node = "node1" if node == "node0" else "node0"
            ranks.append(
                {
                    "role": role,
                    "node": rank_node,
                    "rank": rank_id,
                    "dp": dp,
                    "tp": tp,
                    "dir": rank_dir.name,
                    "events": kept,
                    "raw_events": events,
                    "pid_names": pid_names,
                }
            )

    report: dict = {"ranks": [], "alignment": {}}
    loaded_afd_keys = {
        (r["role"], r["rank"]) for r in ranks if r["role"] in requested_afd_roles
    }
    expected_afd_keys = {key for key in corr_lanes if key[0] in requested_afd_roles}
    missing_sidecar_roles = requested_afd_roles - {
        role for role, _rank in expected_afd_keys
    }
    if missing_sidecar_roles:
        raise ValueError(
            "missing correlation sidecars for AFD roles: "
            f"{sorted(missing_sidecar_roles)}",
        )
    if len(loaded_afd_keys) != sum(r["role"] in requested_afd_roles for r in ranks):
        raise ValueError("duplicate profiler traces for an AFD role/rank")
    missing_profiler_keys = sorted(expected_afd_keys - loaded_afd_keys)
    if missing_profiler_keys:
        raise ValueError(
            f"missing profiler traces for AFD lanes: {missing_profiler_keys}",
        )
    missing_sidecar_keys = sorted(loaded_afd_keys - expected_afd_keys)
    if missing_sidecar_keys:
        raise ValueError(
            f"missing correlation sidecars for profiler ranks: {missing_sidecar_keys}",
        )
    if not ranks:
        return [], report

    # --- cross-node alignment (only when two nodes are present) ----------
    nodes = {r["node"] for r in ranks}
    shift_by_node: dict[str, float] = {n: 0.0 for n in nodes}
    if len(nodes) == 2:
        ref_node = sorted(nodes)[0]
        peer_node = sorted(nodes)[1]
        ref_rank = next(r for r in ranks if r["node"] == ref_node)
        peer_rank = next(r for r in ranks if r["node"] == peer_node)
        is_afd = bool({r["role"] for r in ranks} & {"attention", "ffn"})
        if is_afd:
            sidecar_hosts = {
                str(corr_lanes[key]["hostname"]) for key in loaded_afd_keys
            }
            if len(sidecar_hosts) != 2:
                raise ValueError(
                    "two-node AFD stacking requires sidecars from exactly two "
                    f"hostnames, got {sorted(sidecar_hosts)}",
                )
            report["alignment"] = {
                "method": "sidecar four-timestamp clock calibration plus "
                "per-rank MSTX anchors",
                "ref_node": ref_node,
                "peer_node": peer_node,
                "clock_alignment": clock_diagnostics,
            }
        else:
            ref_ts = _op_starts(
                ref_rank["raw_events"], ref_rank["pid_names"], align_op, None
            )
            peer_ts = _op_starts(
                peer_rank["raw_events"], peer_rank["pid_names"], align_op, None
            )
            offset, spread, n_pairs = _median_pair_offset(ref_ts, peer_ts)
            shift_by_node[peer_node] = -offset
            report["alignment"] = {
                "method": f"HCCL lockstep op *{align_op}* (FIFO median)",
                "ref_node": ref_node,
                "peer_node": peer_node,
                "offset_us": offset,
                "pair_count": n_pairs,
                "abs_dev_p95_us": spread,
            }

    # --- remap + normalize ------------------------------------------------
    # Axis model:
    #   * ranks WITH a correlation lane (AFD): the mstx<->sidecar median
    #     delta maps the rank's raw device axis onto its host's epoch clock
    #     (ts_epoch = ts_raw - dev_shift). Correlation slices are already
    #     epoch. So all lanes of one host share the host epoch axis.
    #   * Cross-host offsets are already calibrated by the explicit
    #     four-timestamp clock-sync transforms loaded with the sidecars. No
    #     post-hoc causality shift is applied.
    #   * ranks without sidecars (baseline): keep raw axis + HCCL lockstep
    #     node shift (dev_shift=0).
    ranks.sort(key=lambda r: (r["role"], r["rank"]))
    out_events: list[dict] = []
    min_ts = None
    n_corr = 0
    for idx, r in enumerate(ranks):
        base_pid = idx * 1000
        node_shift = shift_by_node[r["node"]]
        lane = corr_lanes.get((r["role"], r["rank"]))
        if (
            correlation_dir is not None
            and r["role"] in {"attention", "ffn"}
            and lane is None
        ):
            raise ValueError(
                f"missing correlation sidecar for {(r['role'], r['rank'])}",
            )
        corr = lane["slices"] if lane else []
        dev_shift = 0.0
        marker_spread_us = 0.0
        n_anchors = 0
        if lane:
            markers = _mstx_marker_times(r["raw_events"])
            deltas = [
                marker_ts - begin_ts
                for marker_key, begin_times in lane["begin_index"].items()
                for marker_ts, begin_ts in zip(
                    sorted(markers.get(marker_key, [])),
                    sorted(begin_times),
                    strict=False,
                )
            ]
            n_anchors = len(deltas)
            if not deltas:
                raise ValueError(
                    f"no MSTX anchors for {(r['role'], r['rank'])}",
                )
            dev_shift = _median(deltas)
            marker_spread_us = max(deltas) - min(deltas)
        for e in r["events"]:
            lane_proc = r["pid_names"].get(e["pid"], "")
            lane_id = LANE_ID.get(lane_proc, 0)
            new = {
                "ph": "X",
                "name": e["name"],
                "pid": base_pid + lane_id,
                "tid": e.get("tid", 0),
                "ts": float(e["ts"]) - dev_shift + node_shift,
                "dur": float(e["dur"]),
            }
            if "args" in e:
                new["args"] = e["args"]
            out_events.append(new)
            ts = new["ts"]
            min_ts = ts if min_ts is None else min(min_ts, ts)
        for c in corr:
            new = dict(c)
            new["pid"] = base_pid + 2
            new["tid"] = 0
            new["ts"] = float(c["ts"]) + node_shift
            out_events.append(new)
            n_corr += 1
            min_ts = new["ts"] if min_ts is None else min(min_ts, new["ts"])
        lanes = [(0, "device"), (1, "comm")]
        if corr:
            lanes.append((2, "correlation"))
        for lane_id, lane_name in lanes:
            out_events.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": base_pid + lane_id,
                    "tid": 0,
                    "args": {
                        "name": f"{r['role']} rank{r['rank']} "
                        f"(dp{r['dp']} tp{r['tp']}) {lane_name}"
                    },
                }
            )
        report["ranks"].append(
            {
                "role": r["role"],
                "rank": r["rank"],
                "dp": r["dp"],
                "tp": r["tp"],
                "node": r["node"],
                "kept_events": len(r["events"]),
                "correlation_events": len(corr),
                "corr_mstx_anchors": n_anchors,
                "corr_mstx_spread_us": marker_spread_us,
                "dev_shift_us": dev_shift,
            }
        )
    report["correlation_event_count"] = n_corr

    # AFD ranks are already on the calibrated sidecar axis; applying an
    # additional causality-derived shift would hide real clock or data errors.
    report["causality"] = {
        "method": "no post-hoc shift; calibrated sidecar clock plus MSTX anchors",
        "ffn_shift_us": 0.0,
    }

    origin = min_ts or 0.0
    for e in out_events:
        if e.get("ph") == "X":
            e["ts"] = e["ts"] - origin
    report["rank_count"] = len(ranks)
    report["event_count"] = sum(1 for e in out_events if e["ph"] == "X")
    report["span_s"] = (
        max(e["ts"] + e["dur"] for e in out_events if e["ph"] == "X") / 1e6
        if report["event_count"]
        else 0.0
    )
    return out_events, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="dir with per-rank *_ascend_pt dirs",
    )
    parser.add_argument(
        "--role", required=True, help="lane label for --trace-root ranks"
    )
    parser.add_argument(
        "--node",
        default="node0",
        help="node id of --trace-root (ordering/alignment label)",
    )
    parser.add_argument(
        "--node-split-rank",
        type=int,
        default=None,
        help="single-root two-node split: ranks >= N are on "
        "the peer node (baseline: 32 ranks in one NAS dir, "
        "0-15 node0 / 16-31 node1 -> N=16)",
    )
    parser.add_argument(
        "--peer-root",
        type=Path,
        default=None,
        help="optional second trace root (other node)",
    )
    parser.add_argument("--peer-role", default=None)
    parser.add_argument("--peer-node", default="node1")
    parser.add_argument(
        "--align-op",
        default="alltoallv",
        help="op name substring for baseline lockstep alignment "
        "(msprof names MoE all2all kernels alltoallvAivKernel)",
    )
    parser.add_argument("--min-dur-us", type=float, default=20.0)
    parser.add_argument(
        "--session-ts",
        default=None,
        help="keep only rank dirs whose name contains this "
        "capture timestamp prefix (e.g. 202608240723)",
    )
    parser.add_argument(
        "--correlation-dir",
        type=Path,
        default=None,
        help="AFD correlation sidecar session dir "
        "(traces/correlation/<label>/<session_id>); adds a "
        "per-rank correlation lane with request metadata",
    )
    parser.add_argument(
        "--clock-sync",
        type=Path,
        action="append",
        default=[],
        help="four-timestamp clock-sync file for each non-reference AFD host",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv))

    roots = [(args.role, args.node, args.trace_root, args.node_split_rank)]
    if args.peer_root is not None:
        if not args.peer_role:
            parser.error("--peer-root requires --peer-role")
        roots.append((args.peer_role, args.peer_node, args.peer_root, None))

    events, report = stack(
        roots,
        args.min_dur_us,
        args.align_op,
        args.session_ts,
        args.correlation_dir,
        args.clock_sync,
    )
    payload = {"traceEvents": events, "displayTimeUnit": "us"}
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    report_path = args.output.with_suffix(".report.json")
    if report_path.name.endswith(".json.report.json"):
        report_path = args.output.with_suffix("").with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report.get("alignment", {}), indent=1))
    print(
        f"ranks={report['rank_count']} events={report['event_count']} "
        f"span={report['span_s']:.1f}s -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
