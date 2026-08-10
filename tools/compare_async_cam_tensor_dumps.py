# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare globally indexed Attention tensors from two Async CAM dump runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

TensorKey = tuple[int, str]
TensorRows = dict[TensorKey, dict[int, torch.Tensor]]


def _load_global_token_rows(root: Path) -> TensorRows:
    rows: TensorRows = defaultdict(dict)
    for dump_path in sorted(root.glob("attention-rank-*/layer-*/stage-*.pt")):
        payload = torch.load(dump_path, map_location="cpu")
        if payload["row_coordinate"] != "global_token":
            continue
        key = (int(payload["layer_idx"]), str(payload["point"]))
        indices = payload["selected_row_indices"]
        sampled_tensor = payload["sampled_tensor"]
        if len(indices) != int(sampled_tensor.shape[0]):
            raise RuntimeError(f"invalid sampled row count in {dump_path}")
        for index, tensor_row in zip(indices, sampled_tensor, strict=True):
            token_index = int(index)
            if token_index in rows[key]:
                raise RuntimeError(
                    f"duplicate global token {token_index} for {key} in {root}"
                )
            rows[key][token_index] = tensor_row
    return dict(rows)


def _compare_dump_roots(baseline_root: Path, candidate_root: Path) -> int:
    baseline = _load_global_token_rows(baseline_root)
    candidate = _load_global_token_rows(candidate_root)
    failed_comparisons = 0
    for key in sorted(baseline.keys() | candidate.keys()):
        baseline_rows = baseline.get(key, {})
        candidate_rows = candidate.get(key, {})
        common_indices = sorted(baseline_rows.keys() & candidate_rows.keys())
        baseline_only = sorted(baseline_rows.keys() - candidate_rows.keys())
        candidate_only = sorted(candidate_rows.keys() - baseline_rows.keys())
        layer_idx, point = key
        if not common_indices:
            print(
                f"layer={layer_idx} point={point} common=0 "
                f"baseline_only={baseline_only} candidate_only={candidate_only}"
            )
            failed_comparisons += 1
            continue

        baseline_tensor = torch.stack(
            [baseline_rows[index] for index in common_indices]
        )
        candidate_tensor = torch.stack(
            [candidate_rows[index] for index in common_indices]
        )
        if baseline_tensor.shape != candidate_tensor.shape:
            print(
                f"layer={layer_idx} point={point} common={len(common_indices)} "
                f"shape_mismatch={tuple(baseline_tensor.shape)}!="
                f"{tuple(candidate_tensor.shape)}"
            )
            failed_comparisons += 1
            continue

        if baseline_tensor.is_floating_point():
            difference = (
                baseline_tensor.to(torch.float64) - candidate_tensor.to(torch.float64)
            ).abs()
            max_abs = float(difference.max().item())
            mean_abs = float(difference.mean().item())
            exact = bool(torch.equal(baseline_tensor, candidate_tensor))
            print(
                f"layer={layer_idx} point={point} common={len(common_indices)} "
                f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} exact={exact} "
                f"baseline_only={baseline_only} candidate_only={candidate_only}"
            )
        else:
            mismatches = int((baseline_tensor != candidate_tensor).sum().item())
            print(
                f"layer={layer_idx} point={point} common={len(common_indices)} "
                f"mismatches={mismatches} baseline_only={baseline_only} "
                f"candidate_only={candidate_only}"
            )
    return failed_comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    return _compare_dump_roots(args.baseline, args.candidate)


if __name__ == "__main__":
    raise SystemExit(main())
