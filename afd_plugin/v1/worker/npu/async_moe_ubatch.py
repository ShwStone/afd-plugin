# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe stage planning for AFD-managed Async CAM MoE ubatching."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from afd_plugin.async_moe import (
    ASYNC_MOE_NUM_STAGES,
    ASYNC_MOE_REQUEST_SPLIT,
    ASYNC_MOE_TOKEN_SPLIT,
    AsyncMoeStage,
)


def plan_async_moe_stages(
    num_scheduled_tokens: Sequence[int],
    *,
    num_tokens: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_stages: int,
    split: str,
    use_sequence_parallel: bool,
    tensor_parallel_size: int,
) -> tuple[tuple[AsyncMoeStage, ...], tuple[int, ...]] | None:
    """Plan exactly two ordered CAM stages without importing a device runtime.

    Token mode balances the *real* flattened token extent. Under sequence
    parallelism, each stage receives only the minimum trailing TP padding.
    Parent DP padding is removed before the split and restored after the stages,
    so it cannot distort the two stage workloads.

    Request mode keeps the established request-boundary policy used by PCP and
    adds only stage-local TP padding when FlashComm1 sequence parallelism is
    active.
    """

    scheduled_tokens = tuple(int(token_count) for token_count in num_scheduled_tokens)
    _validate_planner_inputs(
        scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_stages=num_stages,
        tensor_parallel_size=tensor_parallel_size,
    )

    if use_sequence_parallel and num_tokens_padded % tensor_parallel_size != 0:
        raise ValueError(
            "Sequence-parallel parent token extent must be TP divisible: "
            f"num_tokens_padded={num_tokens_padded}, "
            f"tensor_parallel_size={tensor_parallel_size}",
        )

    if split == ASYNC_MOE_REQUEST_SPLIT:
        input_alignment = tensor_parallel_size if use_sequence_parallel else 1
        return _plan_request_boundary_stages(
            scheduled_tokens,
            input_alignment=input_alignment,
        )

    if split != ASYNC_MOE_TOKEN_SPLIT:
        raise ValueError(f"Unsupported Async CAM MoE split policy: {split!r}")
    if tensor_parallel_size <= 1:
        return None
    split_token = _balanced_token_split(num_tokens)
    if split_token is None:
        return None

    cumulative_tokens = _cumulative_tokens(scheduled_tokens)
    input_alignment = tensor_parallel_size if use_sequence_parallel else 1
    stages = (
        _make_token_stage(
            cumulative_tokens,
            token_start=0,
            token_stop=split_token,
            input_alignment=input_alignment,
        ),
        _make_token_stage(
            cumulative_tokens,
            token_start=split_token,
            token_stop=num_tokens,
            input_alignment=input_alignment,
        ),
    )
    if any(stage.is_empty() for stage in stages):
        return None
    if use_sequence_parallel and any(
        stage.num_tokens % tensor_parallel_size != 0 for stage in stages
    ):
        raise ValueError(
            "Sequence-parallel Async CAM stage extents must be TP divisible: "
            f"stage_tokens={[stage.num_tokens for stage in stages]}, "
            f"tensor_parallel_size={tensor_parallel_size}",
        )
    actual_token_counts = tuple(stage.actual_tokens for stage in stages)
    return stages, actual_token_counts


def _validate_planner_inputs(
    scheduled_tokens: tuple[int, ...],
    *,
    num_tokens: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_stages: int,
    tensor_parallel_size: int,
) -> None:
    if num_stages != ASYNC_MOE_NUM_STAGES:
        raise ValueError(
            f"Async CAM MoE ubatching requires exactly two stages, got {num_stages}",
        )
    if any(token_count <= 0 for token_count in scheduled_tokens):
        raise ValueError(
            "Scheduled token counts must all be positive: "
            f"scheduled_tokens={scheduled_tokens}",
        )
    if sum(scheduled_tokens) != num_tokens:
        raise ValueError(
            "Scheduled token counts do not match num_tokens: "
            f"scheduled={sum(scheduled_tokens)}, num_tokens={num_tokens}",
        )
    if num_tokens_padded < num_tokens:
        raise ValueError(
            "num_tokens_padded must cover all real tokens: "
            f"num_tokens={num_tokens}, num_tokens_padded={num_tokens_padded}",
        )
    if num_reqs_padded < len(scheduled_tokens):
        raise ValueError(
            "num_reqs_padded must cover all scheduled requests: "
            f"scheduled_requests={len(scheduled_tokens)}, "
            f"num_reqs_padded={num_reqs_padded}",
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            "tensor_parallel_size must be positive: "
            f"tensor_parallel_size={tensor_parallel_size}",
        )


def _plan_request_boundary_stages(
    scheduled_tokens: tuple[int, ...],
    *,
    input_alignment: int,
) -> tuple[tuple[AsyncMoeStage, ...], tuple[int, ...]] | None:
    if len(scheduled_tokens) < ASYNC_MOE_NUM_STAGES:
        return None

    cumulative_tokens = _cumulative_tokens(scheduled_tokens)
    total_tokens = cumulative_tokens[-1]
    split_request = min(
        range(1, len(scheduled_tokens)),
        key=lambda request_index: (
            abs(cumulative_tokens[request_index] * ASYNC_MOE_NUM_STAGES - total_tokens),
            abs(request_index * ASYNC_MOE_NUM_STAGES - len(scheduled_tokens)),
        ),
    )
    split_token = cumulative_tokens[split_request]
    stages = (
        AsyncMoeStage(
            request_slice=slice(0, split_request),
            token_slice=slice(0, split_token),
            input_tokens=_align_tokens(split_token, input_alignment),
        ),
        AsyncMoeStage(
            request_slice=slice(split_request, len(scheduled_tokens)),
            token_slice=slice(split_token, total_tokens),
            input_tokens=_align_tokens(
                total_tokens - split_token,
                input_alignment,
            ),
        ),
    )
    if any(stage.is_empty() for stage in stages):
        return None
    return stages, tuple(stage.actual_tokens for stage in stages)


def _balanced_token_split(num_tokens: int) -> int | None:
    """Return the closest boundary to half of the real token extent."""

    if num_tokens < ASYNC_MOE_NUM_STAGES:
        return None
    return (num_tokens + 1) // ASYNC_MOE_NUM_STAGES


def _cumulative_tokens(scheduled_tokens: tuple[int, ...]) -> tuple[int, ...]:
    cumulative_tokens = [0]
    for token_count in scheduled_tokens:
        cumulative_tokens.append(cumulative_tokens[-1] + token_count)
    return tuple(cumulative_tokens)


def _make_token_stage(
    cumulative_tokens: tuple[int, ...],
    *,
    token_start: int,
    token_stop: int,
    input_alignment: int,
) -> AsyncMoeStage:
    request_start = bisect_right(cumulative_tokens, token_start) - 1
    request_stop = bisect_left(cumulative_tokens, token_stop)
    actual_tokens = token_stop - token_start
    input_tokens = _align_tokens(actual_tokens, input_alignment)
    return AsyncMoeStage(
        request_slice=slice(request_start, request_stop),
        token_slice=slice(token_start, token_stop),
        input_tokens=input_tokens,
    )


def _align_tokens(num_tokens: int, alignment: int) -> int:
    return ((num_tokens + alignment - 1) // alignment) * alignment


__all__ = [
    "ASYNC_MOE_NUM_STAGES",
    "ASYNC_MOE_REQUEST_SPLIT",
    "ASYNC_MOE_TOKEN_SPLIT",
    "AsyncMoeStage",
    "plan_async_moe_stages",
]
