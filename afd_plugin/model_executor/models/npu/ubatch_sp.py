# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Sequence-parallel aware stage-input slicing for async MoE ubatching.

Under TP with sequence parallelism ``hidden_states``/``residual`` only hold
this rank's local token shard, so per-stage inputs must be sliced by
per-rank token ranges derived from the global ubatch split (see
``afd_plugin.v1.worker.npu.ubatch_utils``); sequence tensors that may span
the global token range (``positions``, ``llama_4_scaling``) are mapped
global to local with padding to TP-size multiples.
"""

from __future__ import annotations

import torch
from vllm.v1.worker.ubatch_utils import UBatchSlices

from afd_plugin.v1.worker.npu.ubatch_utils import (
    build_sp_local_ubatch_slices_for_current_rank,
    sp_local_token_count,
)


def build_async_moe_stage_inputs(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    ubatch_slices: UBatchSlices,
    use_sp_local_ubatch_slices: bool,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor | None],
    UBatchSlices,
]:
    """Slice stage inputs, honoring TP-SP local shards when requested."""
    sp_local_ubatch_slices = (
        build_sp_local_ubatch_slices_for_current_rank(hidden_states, ubatch_slices)
        if use_sp_local_ubatch_slices
        else ubatch_slices
    )
    if sp_local_ubatch_slices is ubatch_slices:
        return _build_async_moe_stage_inputs_with_slices(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            llama_4_scaling=llama_4_scaling,
            ubatch_slices=ubatch_slices,
            sp_local_ubatch_slices=sp_local_ubatch_slices,
            num_tokens=int(hidden_states.shape[0]),
        )

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    global_num_tokens = sum(
        int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices
    )

    stage_hidden_states: list[torch.Tensor] = []
    stage_residual: list[torch.Tensor | None] = []
    stage_positions: list[torch.Tensor] = []
    stage_llama_4_scaling: list[torch.Tensor | None] = []

    for ubatch_slice, sp_local_ubatch_slice in zip(
        ubatch_slices,
        sp_local_ubatch_slices,
        strict=True,
    ):
        local_stage_hidden = _slice_and_pad_first_dim(
            hidden_states,
            sp_local_ubatch_slice.token_slice,
        )
        local_stage_residual = _slice_and_pad_first_dim(
            residual,
            sp_local_ubatch_slice.token_slice,
        )
        local_stage_positions = _slice_sequence_tensor_for_sp_stage(
            positions,
            ubatch_slice.token_slice,
            sp_local_ubatch_slice.token_slice,
            local_num_tokens=int(hidden_states.shape[0]),
            global_num_tokens=global_num_tokens,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        local_stage_llama_4_scaling = (
            None
            if llama_4_scaling is None
            else _slice_sequence_tensor_for_sp_stage(
                llama_4_scaling,
                ubatch_slice.token_slice,
                sp_local_ubatch_slice.token_slice,
                local_num_tokens=int(hidden_states.shape[0]),
                global_num_tokens=global_num_tokens,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        )

        stage_hidden_states.append(local_stage_hidden)
        stage_residual.append(local_stage_residual)
        stage_positions.append(local_stage_positions)
        stage_llama_4_scaling.append(local_stage_llama_4_scaling)

    return (
        stage_hidden_states,
        stage_residual,
        stage_positions,
        stage_llama_4_scaling,
        sp_local_ubatch_slices,
    )


def _build_async_moe_stage_inputs_with_slices(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    ubatch_slices: UBatchSlices,
    sp_local_ubatch_slices: UBatchSlices,
    num_tokens: int,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor | None],
    UBatchSlices,
]:
    stage_hidden_states = [
        hidden_states[sp_local_ubatch_slice.token_slice]
        for sp_local_ubatch_slice in sp_local_ubatch_slices
    ]
    stage_residual = [
        _slice_and_pad_first_dim(residual, sp_local_ubatch_slice.token_slice)
        for sp_local_ubatch_slice in sp_local_ubatch_slices
    ]
    stage_positions = [
        _slice_positions(positions, sp_local_ubatch_slice.token_slice)
        for sp_local_ubatch_slice in sp_local_ubatch_slices
    ]
    stage_llama_4_scaling = [
        _slice_llama_4_scaling(
            llama_4_scaling,
            sp_local_ubatch_slice.token_slice,
            num_tokens=num_tokens,
        )
        for sp_local_ubatch_slice in sp_local_ubatch_slices
    ]
    return (
        stage_hidden_states,
        stage_residual,
        stage_positions,
        stage_llama_4_scaling,
        sp_local_ubatch_slices,
    )


def _slice_and_pad_first_dim(
    tensor: torch.Tensor | None,
    token_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    stage_tensor = tensor[token_slice]
    expected_tokens = int(token_slice.stop) - int(token_slice.start)
    missing_tokens = expected_tokens - int(stage_tensor.shape[0])
    if missing_tokens > 0:
        stage_tensor = _pad_first_dim(stage_tensor, missing_tokens)
    return stage_tensor


def _pad_first_dim(tensor: torch.Tensor, pad_tokens: int) -> torch.Tensor:
    if pad_tokens <= 0:
        return tensor
    pad_shape = (pad_tokens, *tensor.shape[1:])
    padding = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, padding], dim=0)


def _slice_sequence_tensor_for_sp_stage(
    tensor: torch.Tensor,
    global_token_slice: slice,
    tensor_token_slice: slice,
    *,
    local_num_tokens: int,
    global_num_tokens: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if _sequence_tensor_token_dim(tensor, local_num_tokens) is not None:
        return _slice_sequence_tensor(tensor, tensor_token_slice)
    if _sequence_tensor_token_dim(tensor, global_num_tokens) is None:
        return tensor

    stage_tensor = _slice_sequence_tensor(tensor, global_token_slice)
    stage_tokens = int(global_token_slice.stop) - int(global_token_slice.start)
    padded_tokens = sp_local_token_count(stage_tokens, tp_size) * tp_size
    if padded_tokens > stage_tokens:
        stage_tensor = _pad_sequence_tensor(
            stage_tensor,
            padded_tokens - stage_tokens,
        )
    tokens_per_rank = padded_tokens // tp_size
    local_start = tp_rank * tokens_per_rank
    local_stop = local_start + tokens_per_rank
    return _slice_sequence_tensor(stage_tensor, slice(local_start, local_stop))


def _sequence_tensor_token_dim(tensor: torch.Tensor, num_tokens: int) -> int | None:
    if tensor.dim() > 0 and int(tensor.shape[0]) == int(num_tokens):
        return 0
    if tensor.dim() > 1 and int(tensor.shape[1]) == int(num_tokens):
        return 1
    return None


def _slice_sequence_tensor(tensor: torch.Tensor, token_slice: slice) -> torch.Tensor:
    if tensor.dim() > 1 and int(tensor.shape[0]) <= 4:
        return tensor[:, token_slice]
    return tensor[token_slice]


def _pad_sequence_tensor(tensor: torch.Tensor, pad_tokens: int) -> torch.Tensor:
    if pad_tokens <= 0:
        return tensor
    if tensor.dim() > 1 and int(tensor.shape[0]) <= 4:
        pad_shape = (tensor.shape[0], pad_tokens, *tensor.shape[2:])
        padding = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, padding], dim=1)
    return _pad_first_dim(tensor, pad_tokens)


def _slice_positions(positions: torch.Tensor, token_slice: slice) -> torch.Tensor:
    if positions.dim() <= 1:
        return positions[token_slice]
    return positions[..., token_slice]


def _slice_llama_4_scaling(
    llama_4_scaling: torch.Tensor | None,
    token_slice: slice,
    *,
    num_tokens: int,
) -> torch.Tensor | None:
    if llama_4_scaling is None:
        return None
    if llama_4_scaling.shape[0] == num_tokens:
        return llama_4_scaling[token_slice]
    if llama_4_scaling.dim() > 1 and llama_4_scaling.shape[1] == num_tokens:
        return llama_4_scaling[:, token_slice]
    return llama_4_scaling


__all__ = ["build_async_moe_stage_inputs"]
