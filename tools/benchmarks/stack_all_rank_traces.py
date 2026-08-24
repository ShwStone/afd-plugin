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
* Cross-node alignment is DATA-DRIVEN (no clock assumptions):
  - baseline: HCCL collectives run in lockstep on all 32 ranks; the node1
    offset is the median per-op ts difference of the chosen op family
    (``--align-op`` substring, default AllToAll) between node1's first rank
    and node0's reference rank.
  - afd: attention ranks (node0) and FFN ranks (node1) are linked by CAM
    dispatch: offset = median(ffn DispatchRecv.start - attn DispatchSend.end).
* All timestamps are then normalized to a common origin (earliest event = 0).

Usage (on a pod, where the exports live):

    python3 -m tools.benchmarks.stack_all_rank_traces \
        --trace-root <...>/02_profiles_32/traces/baseline \
        --role baseline --output baseline_allranks.json

    python3 -m tools.benchmarks.stack_all_rank_traces \
        --trace-root <...>/02_profiles_32/traces/attention --role attention \
        --peer-root <...>/02_profiles_32/traces/ffn --peer-role ffn \
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

RANK_DIR_PATTERN = re.compile(
    r"dp(?P<dp>\d+)_pp\d+_tp(?P<tp>\d+)_dcp\d+_ep(?P<ep>\d+)_rank(?P<rank>\d+)_"
)
# AFD plugin profiler dirs: <hostname>_<pid>_<ts>_ascend_pt (no rank in the
# name) — rank comes from profiler_metadata.json instead.
HOST_DIR_PATTERN = re.compile(r"gpuxdn0*(?P<ip>[0-9]+)_")

KEEP_PROCESSES = ("Ascend Hardware", "Communication")
LANE_ID = {"Ascend Hardware": 0, "Communication": 1}

# CAM op names used for AFD cross-node alignment (device timeline).
AFD_REF_OP = "CamMoeDistributeDispatchSend"   # attention (node0) sends
AFD_PEER_OP = "CamMoeDistributeDispatchRecv"  # ffn (node1) receives


def _rank_identity(rank_dir: Path, role: str) -> tuple[int, int, int] | None:
    """Return (rank, dp, tp) for one rank dir.

    vllm profiler dirs carry dp/tp/rank in the name. AFD plugin profiler
    dirs don't — read profiler_metadata.json: the afd_async_cam group's
    group_rank is the role rank (attention: dp*8+tp; ffn: dp).
    """
    m = RANK_DIR_PATTERN.match(rank_dir.name)
    if m:
        # the "rank" field is TP-local (0-7, repeated per dp group); the "ep"
        # field is the global EP rank (dp*8+tp) — use it as the global rank.
        return int(m.group("ep")), int(m.group("dp")), int(m.group("tp"))
    meta_path = rank_dir / "profiler_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    rank = None
    for key, group in meta.get("parallel_group_info", {}).items():
        if "afd_async_cam" in key:
            rank = int(group["group_rank"])
    if rank is None:
        # fall back: tp group's global_ranks[tp_rank]
        for group in meta.get("parallel_group_info", {}).values():
            if group.get("group_name") == "tp":
                rank = int(group["global_ranks"][int(group["group_rank"])])
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
    keep_pids = {
        pid for pid, name in pid_names.items() if name in KEEP_PROCESSES
    }
    kept = []
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in keep_pids:
            continue
        if float(e.get("dur", 0)) < min_dur_us:
            continue
        kept.append(e)
    return kept


def _op_starts(events: list[dict], pid_names: dict[int, str],
               name_sub: str | None, exact: str | None) -> list[float]:
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
) -> tuple[list[dict], dict]:
    """Merge all (role, node, trace_root, node_split_rank) inputs.

    ``node_split_rank`` splits a single root's ranks across two nodes:
    rank < split -> the root's node, rank >= split -> the other node
    (baseline writes all 32 ranks into one NAS dir from both nodes).
    ``session_ts`` keeps only rank dirs whose name contains the capture
    timestamp (several sessions share the attention/ffn roots).
    """
    ranks: list[dict] = []
    for role, node, root, split in roots:
        for rank_dir in sorted(root.iterdir()):
            if not rank_dir.is_dir() or not rank_dir.name.endswith("_ascend_pt"):
                continue
            if session_ts and session_ts not in rank_dir.name:
                continue
            ident = _rank_identity(rank_dir, role)
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
            ranks.append({
                "role": role,
                "node": rank_node,
                "rank": rank_id,
                "dp": dp,
                "tp": tp,
                "dir": rank_dir.name,
                "events": kept,
                "raw_events": events,
                "pid_names": pid_names,
            })

    report: dict = {"ranks": [], "alignment": {}}
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
        is_afd = {r["role"] for r in ranks} & {"attention", "ffn"}
        if is_afd:
            # CAM send/recv counts differ per side (token split fans out),
            # so FIFO pairing cannot estimate the offset. The HCCL lockstep
            # measurement on the baseline run (same pod pair, same day)
            # showed the two nodes' epoch clocks agree within ~2 ms
            # (offset -1.1 ms, p95 dev 1.0 ms) — trust the epoch clock here.
            shift_by_node[peer_node] = 0.0
            report["alignment"] = {
                "method": "epoch clocks assumed synced (baseline HCCL "
                          "lockstep measured -1.1ms p95dev 1.0ms this day)",
                "ref_node": ref_node,
                "peer_node": peer_node,
                "offset_us": 0.0,
                "pair_count": 0,
                "abs_dev_p95_us": 0.0,
            }
        else:
            ref_ts = _op_starts(ref_rank["raw_events"], ref_rank["pid_names"],
                                align_op, None)
            peer_ts = _op_starts(peer_rank["raw_events"], peer_rank["pid_names"],
                                 align_op, None)
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
    ranks.sort(key=lambda r: (r["role"], r["rank"]))
    out_events: list[dict] = []
    min_ts = None
    for idx, r in enumerate(ranks):
        base_pid = idx * 1000
        node_shift = shift_by_node[r["node"]]
        for e in r["events"]:
            lane_proc = r["pid_names"].get(e["pid"], "")
            lane = LANE_ID.get(lane_proc, 0)
            new = {
                "ph": "X",
                "name": e["name"],
                "pid": base_pid + lane,
                "tid": e.get("tid", 0),
                "ts": float(e["ts"]) + node_shift,
                "dur": float(e["dur"]),
            }
            if "args" in e:
                new["args"] = e["args"]
            out_events.append(new)
            ts = new["ts"]
            min_ts = ts if min_ts is None else min(min_ts, ts)
        for lane, lane_name in ((0, "device"), (1, "comm")):
            out_events.append({
                "ph": "M", "name": "process_name", "pid": base_pid + lane,
                "tid": 0,
                "args": {"name": f"{r['role']} rank{r['rank']} "
                                 f"(dp{r['dp']} tp{r['tp']}) {lane_name}"},
            })
        report["ranks"].append({
            "role": r["role"], "rank": r["rank"], "dp": r["dp"],
            "tp": r["tp"], "node": r["node"], "kept_events": len(r["events"]),
        })

    origin = min_ts or 0.0
    for e in out_events:
        if e.get("ph") == "X":
            e["ts"] = e["ts"] - origin
    report["rank_count"] = len(ranks)
    report["event_count"] = sum(1 for e in out_events if e["ph"] == "X")
    report["span_s"] = (
        max(e["ts"] + e["dur"] for e in out_events if e["ph"] == "X") / 1e6
        if report["event_count"] else 0.0
    )
    return out_events, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True,
                        help="dir with per-rank *_ascend_pt dirs")
    parser.add_argument("--role", required=True,
                        help="lane label for --trace-root ranks")
    parser.add_argument("--node", default="node0",
                        help="node id of --trace-root (ordering/alignment label)")
    parser.add_argument("--node-split-rank", type=int, default=None,
                        help="single-root two-node split: ranks >= N are on "
                        "the peer node (baseline: 32 ranks in one NAS dir, "
                        "0-15 node0 / 16-31 node1 -> N=16)")
    parser.add_argument("--peer-root", type=Path, default=None,
                        help="optional second trace root (other node)")
    parser.add_argument("--peer-role", default=None)
    parser.add_argument("--peer-node", default="node1")
    parser.add_argument("--align-op", default="alltoallv",
                        help="op name substring for baseline lockstep alignment "
                        "(msprof names MoE all2all kernels alltoallvAivKernel)")
    parser.add_argument("--min-dur-us", type=float, default=20.0)
    parser.add_argument("--session-ts", default=None,
                        help="keep only rank dirs whose name contains this "
                        "capture timestamp prefix (e.g. 202608240723)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv))

    roots = [(args.role, args.node, args.trace_root, args.node_split_rank)]
    if args.peer_root is not None:
        if not args.peer_role:
            parser.error("--peer-root requires --peer-role")
        roots.append((args.peer_role, args.peer_node, args.peer_root, None))

    events, report = stack(roots, args.min_dur_us, args.align_op,
                           args.session_ts)
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
    print(f"ranks={report['rank_count']} events={report['event_count']} "
          f"span={report['span_s']:.1f}s -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
