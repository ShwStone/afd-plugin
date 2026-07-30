# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe stage planning for AFD-managed Async CAM MoE ubatching."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from itertools import accumulate

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
    """Plan two ordered CAM stages without importing a device runtime."""

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

    cumulative_tokens = tuple(accumulate(scheduled_tokens, initial=0))
    input_alignment = tensor_parallel_size if use_sequence_parallel else 1
    if split == ASYNC_MOE_REQUEST_SPLIT:
        if len(scheduled_tokens) < ASYNC_MOE_NUM_STAGES:
            return None
        split_request = min(
            range(1, len(scheduled_tokens)),
            key=lambda request_index: (
                abs(
                    cumulative_tokens[request_index] * ASYNC_MOE_NUM_STAGES
                    - num_tokens
                ),
                abs(
                    request_index * ASYNC_MOE_NUM_STAGES - len(scheduled_tokens)
                ),
            ),
        )
        split_token = cumulative_tokens[split_request]
        stage_bounds = (
            (slice(0, split_request), 0, split_token),
            (
                slice(split_request, len(scheduled_tokens)),
                split_token,
                num_tokens,
            ),
        )
    elif split == ASYNC_MOE_TOKEN_SPLIT:
        if tensor_parallel_size <= 1 or num_tokens < ASYNC_MOE_NUM_STAGES:
            return None
        split_token = (num_tokens + 1) // ASYNC_MOE_NUM_STAGES
        stage_bounds = tuple(
            (
                slice(
                    bisect_right(cumulative_tokens, token_start) - 1,
                    bisect_left(cumulative_tokens, token_stop),
                ),
                token_start,
                token_stop,
            )
            for token_start, token_stop in (
                (0, split_token),
                (split_token, num_tokens),
            )
        )
    else:
        raise ValueError(f"Unsupported Async CAM MoE split policy: {split!r}")

    stages = tuple(
        AsyncMoeStage(
            request_slice=request_slice,
            token_slice=slice(token_start, token_stop),
            input_tokens=_align_tokens(
                token_stop - token_start,
                input_alignment,
            ),
        )
        for request_slice, token_start, token_stop in stage_bounds
    )
    return stages, tuple(stage.actual_tokens for stage in stages)


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


def _align_tokens(num_tokens: int, alignment: int) -> int:
    return ((num_tokens + alignment - 1) // alignment) * alignment


__all__ = [
    "ASYNC_MOE_NUM_STAGES",
    "ASYNC_MOE_REQUEST_SPLIT",
    "ASYNC_MOE_TOKEN_SPLIT",
    "AsyncMoeStage",
    "plan_async_moe_stages",
]
