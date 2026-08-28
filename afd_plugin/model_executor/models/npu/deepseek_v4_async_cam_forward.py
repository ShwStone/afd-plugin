# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 model-side Async CAM ubatch orchestration."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.async_cam_layout import (
    AsyncMoeUbatchMetadata,
    CAMDispatchLayout,
    build_async_moe_stage_inputs,
    get_async_moe_ubatch_metadata_from_forward_context,
    log_async_moe_stage_attention,
    prepare_cam_dispatch_payload,
    restore_async_moe_stage_outputs,
    restore_cam_dispatch_output,
)

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v4 import (
        AFDDeepseekV4DecoderLayer,
        AFDDeepseekV4Model,
    )


_ASYNC_MOE_STAGE_COUNT = 2


@dataclass(slots=True)
class _PendingFFNState:
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    dispatch_layout: CAMDispatchLayout | None = None
    dispatch_ref: torch.Tensor | None = None


# Upstream source: vllm-ascend commit 80d8c194f,
# DeepseekV4Model.forward.
# Patch reason: native model forward serializes Attention and FFN within each
# layer, while Async CAM needs the FFN boundary exposed to a two-stage schedule.
# Patch functionality: retain the pinned native embedding, HC-head, MTP-buffer,
# PP, and normalization behavior around the AFD-owned layer pipeline.
# Signature: AFD-owned helper adds the model instance to the native arguments.
def run_model_forward(
    model: AFDDeepseekV4Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    """Run one DSV4 forward with a model-owned two-stage CAM pipeline."""

    # ### PATCH START: pinned DSV4 model lifecycle around Async CAM stages
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = model.embed_input_ids(input_ids)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]

    if get_pp_group().is_first_rank:
        hidden_states = hidden_states.unsqueeze(1).repeat(1, model.hc_mult, 1)
    if model.aux_hidden_state_layers:
        raise RuntimeError(
            "AFD DSV4 async CAM does not support aux hidden state capture",
        )

    forward_context = get_forward_context()
    afd_metadata = get_afd_metadata_from_forward_context(forward_context)
    if afd_metadata is None:
        raise RuntimeError("DSV4 async CAM ubatching requires AFD metadata")
    async_moe_metadata = get_async_moe_ubatch_metadata_from_forward_context(
        forward_context,
    )
    if async_moe_metadata is None:
        raise RuntimeError("DSV4 async CAM ubatching requires a stage plan")
    hidden_states = run_async_moe_ubatch_afd_forward(
        model,
        hidden_states,
        positions,
        input_ids,
        afd_metadata,
        async_moe_metadata,
    )

    if forward_context.flash_comm_v1_enabled:
        mtp_hidden_states = tensor_model_parallel_all_gather(
            hidden_states.flatten(1),
            dim=0,
        )
        if forward_context.pad_size > 0:
            mtp_hidden_states = mtp_hidden_states[: -forward_context.pad_size]
    else:
        mtp_hidden_states = hidden_states.flatten(1)
    num_tokens = int(mtp_hidden_states.shape[0])
    model._mtp_hidden_buffer[:num_tokens].copy_(mtp_hidden_states)
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})

    hidden_states = model.hc_head(
        hidden_states,
        model.hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
    )
    hidden_states = model.norm(hidden_states)
    # ### PATCH END: pinned DSV4 model lifecycle around Async CAM stages
    return hidden_states


def run_async_moe_ubatch_afd_forward(
    model: AFDDeepseekV4Model,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    metadata: AsyncMoeUbatchMetadata,
) -> torch.Tensor:
    """Run DSV4 layers as the same two-stage pipeline used by DeepSeek V2."""

    forward_context = get_forward_context()
    _validate_stage_plan(forward_context, metadata)
    model_layers = list(islice(model.layers, model.start_layer, model.end_layer))
    if not model_layers:
        return hidden_states

    stage_inputs = build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
        input_ids=input_ids,
    )
    stage_hidden_states = stage_inputs.hidden_states
    stage_positions = stage_inputs.positions
    stage_input_ids = stage_inputs.input_ids
    assert stage_input_ids is not None
    pending_ffn_states: list[_PendingFFNState | None] = [None for _ in metadata.stages]
    afd_connector = afd_metadata.connector

    def compute_stage_attention(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage = metadata.stages[stage_idx]
        expected_tokens = int(stage.input_tokens)
        if metadata.use_sequence_parallel:
            expected_tokens //= get_tensor_model_parallel_world_size()
        actual_tokens = int(stage_hidden_states[stage_idx].shape[0])
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                "DSV4 async CAM stage input does not match its plan: "
                f"stage={stage_idx}, expected={expected_tokens}, "
                f"actual={actual_tokens}",
            )

        stage_forward_context = _make_stage_forward_context(
            forward_context,
            metadata,
            stage_idx,
            stage_input_ids[stage_idx],
        )
        log_async_moe_stage_attention(
            stage_idx,
            stage,
            actual_tokens,
            stage_forward_context,
        )
        with override_forward_context(stage_forward_context):
            (
                stage_hidden_states[stage_idx],
                residual,
                post,
                comb,
                topk_weights,
                topk_ids,
            ) = layer.compute_async_moe_attn_output(
                stage_positions[stage_idx],
                stage_hidden_states[stage_idx],
                None,
                None,
            )
        pending_ffn_states[stage_idx] = _PendingFFNState(
            residual=residual,
            post=post,
            comb=comb,
        )
        return topk_weights, topk_ids

    def send_stage_attention(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        pending_state = pending_ffn_states[stage_idx]
        if pending_state is None:
            raise RuntimeError(
                f"DSV4 async CAM stage {stage_idx} has no pending FFN state",
            )
        dispatch_payload = prepare_cam_dispatch_payload(
            stage_hidden_states[stage_idx],
            topk_weights,
            topk_ids,
            None,
            use_sequence_parallel=metadata.use_sequence_parallel,
        )
        transfer_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        afd_connector.send_attn_output(
            dispatch_payload.hidden_states,
            AFDTransferContext(metadata=transfer_metadata),
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        pending_state.dispatch_layout = dispatch_payload.layout
        pending_state.dispatch_ref = dispatch_payload.hidden_states

    def recv_stage_ffn(
        layer: AFDDeepseekV4DecoderLayer,
        stage_idx: int,
    ) -> None:
        pending_state = pending_ffn_states[stage_idx]
        if (
            pending_state is None
            or pending_state.dispatch_layout is None
            or pending_state.dispatch_ref is None
        ):
            raise RuntimeError(
                f"DSV4 async CAM stage {stage_idx} has no pending dispatch",
            )
        local_ffn_output = afd_connector.recv_ffn_output(
            ref_tensor=pending_state.dispatch_ref,
            ubatch_idx=stage_idx,
        )
        local_ffn_output = restore_cam_dispatch_output(
            local_ffn_output,
            pending_state.dispatch_layout,
        )
        stage_hidden_states[stage_idx] = layer.apply_async_moe_ffn_output(
            local_ffn_output,
            pending_state.residual,
            pending_state.post,
            pending_state.comb,
        )
        pending_ffn_states[stage_idx] = None

    first_layer = model_layers[0]
    topk_weights, topk_ids = compute_stage_attention(first_layer, 0)
    send_stage_attention(first_layer, 0, topk_weights, topk_ids)

    # Keep exactly one CAM dispatch in flight. While FFN computes that stage,
    # Attention advances the peer stage to the same layer boundary; each recv
    # is therefore both the dependency and the credit before the next send.
    for layer_offset in range(len(model_layers) - 1):
        current_layer = model_layers[layer_offset]
        next_layer = model_layers[layer_offset + 1]

        topk_weights, topk_ids = compute_stage_attention(current_layer, 1)
        recv_stage_ffn(current_layer, 0)
        send_stage_attention(current_layer, 1, topk_weights, topk_ids)

        topk_weights, topk_ids = compute_stage_attention(next_layer, 0)
        recv_stage_ffn(current_layer, 1)
        send_stage_attention(next_layer, 0, topk_weights, topk_ids)

    last_layer = model_layers[-1]
    topk_weights, topk_ids = compute_stage_attention(last_layer, 1)
    recv_stage_ffn(last_layer, 0)
    send_stage_attention(last_layer, 1, topk_weights, topk_ids)
    recv_stage_ffn(last_layer, 1)
    return restore_async_moe_stage_outputs(stage_hidden_states, metadata)


def _validate_stage_plan(
    forward_context: ForwardContext,
    metadata: AsyncMoeUbatchMetadata,
) -> None:
    if len(metadata.stages) != _ASYNC_MOE_STAGE_COUNT:
        raise RuntimeError(
            f"DSV4 async CAM requires exactly two stages; got {len(metadata.stages)}",
        )
    runtime_sequence_parallel = bool(forward_context.flash_comm_v1_enabled)
    if runtime_sequence_parallel != metadata.use_sequence_parallel:
        raise RuntimeError(
            "DSV4 async CAM stage layout does not match the current "
            "FlashComm1 mode: "
            f"layout_sequence_parallel={metadata.use_sequence_parallel}, "
            f"flash_comm_v1_enabled={runtime_sequence_parallel}",
        )


def _make_stage_forward_context(
    parent: ForwardContext,
    metadata: AsyncMoeUbatchMetadata,
    stage_idx: int,
    input_ids: torch.Tensor,
) -> ForwardContext:
    stage = metadata.stages[stage_idx]
    stage_context = copy(parent)
    stage_context.attn_metadata = metadata.attn_metadata[stage_idx]
    stage_context.additional_kwargs = dict(parent.additional_kwargs or {})
    stage_context.ubatch_idx = stage_idx
    stage_context.num_ubatches = len(metadata.stages)
    stage_context.dbo_enabled = False
    if metadata.use_sequence_parallel:
        stage_context.num_tokens = stage.actual_tokens
        stage_context.pad_size = int(stage.input_tokens) - stage.actual_tokens
    else:
        stage_context.num_tokens = int(stage.input_tokens)
        stage_context.pad_size = 0
    stage_context.input_ids = input_ids
    return stage_context


__all__ = [
    "run_async_moe_ubatch_afd_forward",
    "run_model_forward",
]
