#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare two Async CAM precision-capture runs and find the first divergence.

Two capture directories (one per execution mode, e.g. ``no_ubatch`` vs
``token``) are aligned by ``(role, boundary, layer_idx)`` and their saved
tensor rows are mapped into parent global-token coordinates using each
record's stage bounds. Metrics are computed per global token and aggregated,
with the first divergent ``(layer_idx, boundary, token)`` reported.

Ubatch runs store one record per stage; the CLI merges stage coverage into a
global-token map so stage boundaries never leak into the comparison.

Usage::

    python -m tools.benchmarks.compare_async_moe_captures \\
        --run-a <mode-a-dir> --run-b <mode-b-dir> \\
        [--boundaries layer_input,attn_output,gate_topk,dispatch_send] \\
        [--layers 3,4] [--threshold 0.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Tensor names that are compared per global token.
_TOKEN_ALIGNED_TENSORS = (
    "hidden_states",
    "residual",
    "positions",
    "attn_output",
    "router_logits",
    "topk_ids",
    "topk_weights",
    "logits",
)

#: Int tensors compared for exact agreement rather than numerical closeness.
_IDENTITY_TENSORS = frozenset({"topk_ids", "positions"})


@dataclass
class TensorCompare:
    tensor: str
    shape_a: list[int] | None = None
    shape_b: list[int] | None = None
    checksums_a: dict[str, Any] = field(default_factory=dict)
    checksums_b: dict[str, Any] = field(default_factory=dict)
    tokens_compared: int = 0
    max_max_abs: float = 0.0
    max_max_abs_token: int | None = None
    mean_max_abs: float = 0.0
    max_rel_l2: float = 0.0
    topk_agreement: float | None = None
    first_divergent_token: int | None = None

    @property
    def divergent(self) -> bool:
        if self.tensor in _IDENTITY_TENSORS:
            return self.topk_agreement is not None and self.topk_agreement < 1.0
        return self.max_max_abs > 0.0


@dataclass
class BoundaryCompare:
    role: str
    boundary: str
    layer_idx: int
    tensors: list[TensorCompare] = field(default_factory=list)

    @property
    def any_divergent(self) -> bool:
        return any(tc.divergent for tc in self.tensors)


def _load_records(mode_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(mode_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("schema_version") is None or record.get("boundary") is None:
            continue
        records.append((path.parent, record))
    return records


def _record_step(record: dict[str, Any]) -> int:
    step = record.get("scheduler_step")
    return int(step) if step is not None else 0


def _group_records(
    records: list[tuple[Path, dict[str, Any]]],
) -> dict[tuple[str, str, int], list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[tuple[str, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for base, record in records:
        key = (
            record.get("role", "attention"),
            record.get("boundary", "unknown"),
            int(record.get("layer_idx", -1)),
        )
        grouped.setdefault(key, []).append((base, record))
    return grouped


def _pair_runs_by_step(
    run_a: list[tuple[Path, dict[str, Any]]],
    run_b: list[tuple[Path, dict[str, Any]]],
    *,
    force_step: int | None = None,
) -> list[tuple[list, list]]:
    """Pair stage records by forward step, aligning the first prefill first.

    Each run stores one record per stage per forward; the scheduler step is a
    per-run sequential counter, so values are matched by order (first step in
    run A vs first step in run B) unless ``force_step`` is given.
    """
    by_step_a: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    by_step_b: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for base, record in run_a:
        by_step_a.setdefault(_record_step(record), []).append((base, record))
    for base, record in run_b:
        by_step_b.setdefault(_record_step(record), []).append((base, record))
    if force_step is not None:
        return [(by_step_a.get(force_step, []), by_step_b.get(force_step, []))]
    steps_a = sorted(by_step_a)
    steps_b = sorted(by_step_b)
    return [
        (by_step_a[step_a], by_step_b[step_b])
        for step_a, step_b in zip(steps_a, steps_b)
    ]


def _load_saved_rows(base: Path, meta: dict[str, Any]) -> tuple[list[int], Any]:
    """Return (stage-local row indices, saved tensor) for a tensor record."""
    fname = meta.get("file")
    if not fname:
        return [], None
    try:
        tensor = torch.load(base / fname, map_location="cpu", weights_only=True)
    except Exception:
        return [], None
    rows = meta.get("rows")
    if rows is None:
        rows = list(range(tensor.shape[0])) if tensor.dim() >= 1 else []
    return rows, tensor


def _global_token_vectors(
    base: Path,
    record: dict[str, Any],
    tensor_name: str,
) -> dict[int, Any]:
    """Map global token -> stage-local row vector for one record.

    With sequence parallelism (FlashComm1) the padded stage is split into
    contiguous per-TP-rank shards of ``n_local`` rows in TP-rank order, so a
    local row ``i`` on TP rank ``r`` maps to global token
    ``global_token_start + r * n_local + i``. Without SP the token dimension is
    TP-replicated and rows map directly from ``global_token_start``.
    """
    meta = record.get("tensors", {}).get(tensor_name)
    if not meta:
        return {}
    rows, tensor = _load_saved_rows(base, meta)
    if tensor is None or tensor.dim() < 1:
        return {}
    start = int(record.get("global_token_start") or 0)
    n_local = tensor.shape[0]
    tp_rank = int(record.get("tp_rank", 0) or 0)
    tp_size = int(record.get("tp_size", 1) or 1)
    input_tokens = record.get("input_tokens")
    sp_sharded = (
        bool(input_tokens)
        and tp_size > 1
        and n_local * tp_size == int(input_tokens)
    )
    # Cap at the explicit stage range; without one, fall back to the record's
    # real token count so TP-padding rows (actual_tokens .. input_tokens) are
    # excluded (e.g. final_hidden / dense captures that have no stage bounds).
    stop = record.get("global_token_stop")
    if stop is None:
        actual = record.get("actual_tokens")
        stop = actual if actual is not None else None
    if sp_sharded:
        offset = tp_rank * n_local
        mapping: dict[int, Any] = {}
        for position, stage_row in enumerate(rows):
            gtoken = start + offset + int(stage_row)
            if stop is not None and gtoken >= int(stop):
                continue  # padding row (beyond the real tokens)
            mapping[gtoken] = tensor[position]
        return mapping
    mapping = {}
    for position, stage_row in enumerate(rows):
        gtoken = start + int(stage_row)
        if stop is not None and gtoken >= int(stop):
            continue
        mapping[gtoken] = tensor[position]
    return mapping


def _build_global_map(
    bases_records: list[tuple[Path, dict[str, Any]]],
    tensor_name: str,
) -> dict[int, Any]:
    merged: dict[int, Any] = {}
    for base, record in bases_records:
        merged.update(_global_token_vectors(base, record, tensor_name))
    return merged


def _compare_vectors(a: Any, b: Any) -> dict[str, Any]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.shape != b.shape:
        return {"shape_mismatch": True}
    diff = (a - b).abs()
    denom = b.pow(2).sum().sqrt().item()
    return {
        "shape_mismatch": False,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel_l2": float(diff.pow(2).sum().sqrt().item() / (denom + 1e-12)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item(),
        ),
    }


def _compare_tensor(
    tensor_name: str,
    map_a: dict[int, Any],
    map_b: dict[int, Any],
    shape_a: list[int] | None,
    shape_b: list[int] | None,
    checksums_a: dict[str, Any],
    checksums_b: dict[str, Any],
) -> TensorCompare:
    compare = TensorCompare(
        tensor=tensor_name,
        shape_a=shape_a,
        shape_b=shape_b,
        checksums_a=checksums_a,
        checksums_b=checksums_b,
    )
    common = sorted(set(map_a).intersection(map_b))
    compare.tokens_compared = len(common)

    if tensor_name in _IDENTITY_TENSORS:
        agreements: list[float] = []
        first_divergent: int | None = None
        for global_token in common:
            a_ids = map_a[global_token].detach().reshape(-1)
            b_ids = map_b[global_token].detach().reshape(-1)
            if a_ids.shape != b_ids.shape:
                continue
            same = bool(torch.equal(a_ids, b_ids))
            agreements.append(1.0 if same else 0.0)
            if not same and first_divergent is None:
                first_divergent = global_token
        if agreements:
            compare.topk_agreement = sum(agreements) / len(agreements)
            compare.first_divergent_token = first_divergent
        return compare

    max_max_abs = 0.0
    mean_max_abs = 0.0
    max_rel_l2 = 0.0
    max_token: int | None = None
    for global_token in common:
        result = _compare_vectors(map_a[global_token], map_b[global_token])
        if result.get("shape_mismatch"):
            compare.max_max_abs = float("inf")
            return compare
        if result["max_abs"] > max_max_abs:
            max_max_abs = result["max_abs"]
            max_token = global_token
        mean_max_abs += result["mean_abs"]
        max_rel_l2 = max(max_rel_l2, result["rel_l2"])
    compare.max_max_abs = max_max_abs
    compare.max_max_abs_token = max_token
    compare.mean_max_abs = mean_max_abs / len(common) if common else 0.0
    compare.max_rel_l2 = max_rel_l2
    return compare


def compare_boundary(
    role: str,
    boundary: str,
    layer_idx: int,
    run_a: list[tuple[Path, dict[str, Any]]],
    run_b: list[tuple[Path, dict[str, Any]]],
) -> BoundaryCompare:
    result = BoundaryCompare(role=role, boundary=boundary, layer_idx=layer_idx)
    tensor_names = sorted(set(_TOKEN_ALIGNED_TENSORS))
    for tensor_name in tensor_names:
        maps_a = _build_global_map(run_a, tensor_name)
        maps_b = _build_global_map(run_b, tensor_name)
        if not maps_a or not maps_b:
            continue
        meta_a = run_a[0][1].get("tensors", {}).get(tensor_name, {})
        meta_b = run_b[0][1].get("tensors", {}).get(tensor_name, {})
        result.tensors.append(
            _compare_tensor(
                tensor_name,
                maps_a,
                maps_b,
                meta_a.get("shape"),
                meta_b.get("shape"),
                meta_a.get("checksums", {}),
                meta_b.get("checksums", {}),
            ),
        )
    return result


def _print_compare(boundary_results: list[BoundaryCompare]) -> None:
    print("=" * 96)
    print("Async CAM precision capture comparison")
    print("=" * 96)
    if not boundary_results:
        print("no comparable records")
        return
    print(
        f"{'layer':>5} {'boundary':<16} {'tensor':<16} {'tokens':>7} "
        f"{'max_abs':>12} {'rel_l2':>10} {'topk_agree':>10}",
    )
    print("-" * 96)
    for result in sorted(boundary_results, key=lambda r: (r.layer_idx, r.boundary)):
        for tc in result.tensors:
            topk = f"{tc.topk_agreement:.4f}" if tc.topk_agreement is not None else "-"
            print(
                f"{result.layer_idx:>5} {result.boundary:<16} {tc.tensor:<16} "
                f"{tc.tokens_compared:>7} {tc.max_max_abs:>12.6g} "
                f"{tc.max_rel_l2:>10.6g} {topk:>10}",
            )
    print("-" * 96)
    divergent = [result for result in boundary_results if result.any_divergent]
    if divergent:
        first = min(divergent, key=lambda r: (r.layer_idx, r.boundary))
        tokens = [
            tc.first_divergent_token if tc.tensor in _IDENTITY_TENSORS else tc.max_max_abs_token
            for tc in first.tensors
            if (tc.first_divergent_token if tc.tensor in _IDENTITY_TENSORS else tc.max_max_abs_token) is not None
        ]
        print(
            f"FIRST DIVERGENCE: role={first.role} layer={first.layer_idx} "
            f"boundary={first.boundary} token={min(tokens) if tokens else 'n/a'}",
        )
    else:
        print("no divergence above threshold")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path, help="mode A capture dir")
    parser.add_argument("--run-b", required=True, type=Path, help="mode B capture dir")
    parser.add_argument(
        "--boundaries",
        default="layer_input,attn_output,gate_topk,dispatch_send,restored_ffn,final_hidden",
        help="comma-separated boundaries to compare",
    )
    parser.add_argument("--layers", default=None, help="comma-separated layer filter")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="compare only this scheduler step (default: align steps by order)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write JSON report to path")
    args = parser.parse_args()

    records_a = _load_records(args.run_a)
    records_b = _load_records(args.run_b)
    if not records_a or not records_b:
        print(
            f"no records found (a={len(records_a)}, b={len(records_b)}); "
            "point --run-a/--run-b at mode capture dirs containing *_*.json",
            file=sys.stderr,
        )
        return 2

    wanted_boundaries = {part.strip() for part in args.boundaries.split(",") if part.strip()}
    wanted_layers = (
        {int(part) for part in args.layers.split(",") if part.strip()}
        if args.layers
        else None
    )
    grouped_a = _group_records(records_a)
    grouped_b = _group_records(records_b)

    boundary_results: list[BoundaryCompare] = []
    for key, run_a_records in grouped_a.items():
        role, boundary, layer_idx = key
        if boundary not in wanted_boundaries:
            continue
        if wanted_layers is not None and layer_idx not in wanted_layers:
            continue
        run_b_records = grouped_b.get(key, [])
        if not run_b_records:
            continue
        for paired_a, paired_b in _pair_runs_by_step(
            run_a_records,
            run_b_records,
            force_step=args.step,
        ):
            if not paired_a or not paired_b:
                continue
            boundary_results.append(
                compare_boundary(role, boundary, layer_idx, paired_a, paired_b),
            )

    _print_compare(boundary_results)
    if args.json is not None:
        payload = [
            {
                "role": result.role,
                "layer_idx": result.layer_idx,
                "boundary": result.boundary,
                "tensors": [
                    {
                        "tensor": tc.tensor,
                        "shape_a": tc.shape_a,
                        "shape_b": tc.shape_b,
                        "checksums_a": tc.checksums_a,
                        "checksums_b": tc.checksums_b,
                        "tokens_compared": tc.tokens_compared,
                        "max_max_abs": tc.max_max_abs,
                        "max_max_abs_token": tc.max_max_abs_token,
                        "mean_max_abs": tc.mean_max_abs,
                        "max_rel_l2": tc.max_rel_l2,
                        "topk_agreement": tc.topk_agreement,
                        "first_divergent_token": tc.first_divergent_token,
                    }
                    for tc in result.tensors
                ],
            }
            for result in boundary_results
        ]
        args.json.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
