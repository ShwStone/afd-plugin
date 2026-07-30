# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 async CAM forward orchestration helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from itertools import islice
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import ForwardContext, get_forward_context

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import AsyncMoeUbatchMetadata
from afd_plugin.model_executor.models.npu.async_moe_sp import (
    CAMDispatchLayout,
    build_async_moe_stage_inputs,
    prepare_cam_dispatch_payload,
    restore_async_moe_stage_outputs,
    restore_cam_dispatch_output,
)

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        AFDDeepseekV2Model,
    )


def run_attention_gate_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Attention-side gate AFD path used by async CAM."""

    afd_connector = afd_metadata.connector
    forward_context = get_forward_context()
    stage_idx = int(
        getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
    )
    pending_ffn_recv = False
    pending_dispatch_layout: CAMDispatchLayout | None = None
    pending_dispatch_ref: torch.Tensor | None = None

    for layer_offset, layer in enumerate(
        islice(model.layers, model.start_layer, model.end_layer),
    ):
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        if layer_offset > 0 and pending_ffn_recv:
            if pending_dispatch_layout is None or pending_dispatch_ref is None:
                raise RuntimeError("Async CAM receive is missing its dispatch layout")
            local_ffn_output = afd_connector.recv_ffn_output(
                ref_tensor=pending_dispatch_ref,
                ubatch_idx=stage_idx,
            )
            hidden_states = restore_cam_dispatch_output(
                local_ffn_output,
                pending_dispatch_layout,
            )
            pending_ffn_recv = False
            pending_dispatch_layout = None
            pending_dispatch_ref = None

        if not layer.is_moe_layer:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            continue

        (
            hidden_states,
            residual,
            topk_weights,
            topk_ids,
            router_logits,
        ) = layer.compute_attn_output(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )

        dispatch_payload = prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=forward_context.flash_comm_v1_enabled,
        )
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        context = AFDTransferContext(metadata=metadata)
        afd_connector.send_attn_output(
            dispatch_payload.hidden_states,
            context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        pending_ffn_recv = True
        pending_dispatch_layout = dispatch_payload.layout
        pending_dispatch_ref = dispatch_payload.hidden_states

    if pending_ffn_recv:
        if pending_dispatch_layout is None or pending_dispatch_ref is None:
            raise RuntimeError("Async CAM receive is missing its dispatch layout")
        local_ffn_output = afd_connector.recv_ffn_output(
            ref_tensor=pending_dispatch_ref,
            ubatch_idx=stage_idx,
        )
        hidden_states = restore_cam_dispatch_output(
            local_ffn_output,
            pending_dispatch_layout,
        )
    return hidden_states, residual


def run_async_moe_ubatch_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    async_moe_ubatch_metadata: AsyncMoeUbatchMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the two-stage async MoE ubatch pipeline used by async CAM."""

    forward_context = get_forward_context()
    afd_connector = afd_metadata.connector
    model_layers = list(islice(model.layers, model.start_layer, model.end_layer))
    first_moe_offset = next(
        (
            layer_offset
            for layer_offset, layer in enumerate(model_layers)
            if layer.is_moe_layer
        ),
        len(model_layers),
    )
    dense_prefix_layers = model_layers[:first_moe_offset]
    moe_layers = model_layers[first_moe_offset:]
    if any(not layer.is_moe_layer for layer in moe_layers):
        raise RuntimeError(
            "async_moe_ubatching requires a dense prefix followed by "
            "contiguous MoE layers",
        )

    for layer in dense_prefix_layers:
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
    if not moe_layers:
        return hidden_states, residual

    stage_inputs = build_async_moe_stage_inputs(
        hidden_states,
        residual,
        positions,
        llama_4_scaling,
        async_moe_ubatch_metadata,
    )
    stage_hidden_states = stage_inputs.hidden_states
    stage_residual = stage_inputs.residuals
    stage_positions = stage_inputs.positions
    stage_llama_4_scaling = stage_inputs.llama_4_scaling
    stage_dispatch_layouts: list[CAMDispatchLayout | None] = [
        None for _ in stage_hidden_states
    ]
    stage_dispatch_refs: list[torch.Tensor | None] = [None for _ in stage_hidden_states]

    def compute_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with _use_async_moe_ubatch_forward_context(
            forward_context=forward_context,
            parent_afd_metadata=afd_metadata,
            async_moe_ubatch_metadata=async_moe_ubatch_metadata,
            local_input_tokens=stage_inputs.local_input_token_counts[stage_idx],
            local_actual_tokens=stage_inputs.local_actual_token_counts[stage_idx],
            stage_idx=stage_idx,
        ):
            (
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                topk_weights,
                topk_ids,
                router_logits,
            ) = layer.compute_attn_output(
                stage_positions[stage_idx],
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                stage_llama_4_scaling[stage_idx],
            )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError(
                "async_moe_ubatching requires Attention-side topk payloads",
            )
        expected_tokens = stage_inputs.local_input_token_counts[stage_idx]
        if int(stage_hidden_states[stage_idx].shape[0]) != expected_tokens:
            raise RuntimeError(
                "async_moe_ubatching stage output token count mismatch: "
                f"expected {expected_tokens}, got "
                f"{int(stage_hidden_states[stage_idx].shape[0])}",
            )
        return topk_weights, topk_ids, router_logits

    def send_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor | None,
    ) -> None:
        dispatch_payload = prepare_cam_dispatch_payload(
            stage_hidden_states[stage_idx],
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=async_moe_ubatch_metadata.use_sequence_parallel,
        )
        stage_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(dispatch_payload.hidden_states.shape[0]),
        )
        stage_context = AFDTransferContext(metadata=stage_metadata)
        afd_connector.send_attn_output(
            dispatch_payload.hidden_states,
            stage_context,
            topk_weights=dispatch_payload.topk_weights,
            topk_ids=dispatch_payload.topk_ids,
            router_logits=dispatch_payload.router_logits,
        )
        stage_dispatch_layouts[stage_idx] = dispatch_payload.layout
        stage_dispatch_refs[stage_idx] = dispatch_payload.hidden_states

    def recv_stage_ffn(stage_idx: int) -> None:
        dispatch_layout = stage_dispatch_layouts[stage_idx]
        dispatch_ref = stage_dispatch_refs[stage_idx]
        if dispatch_layout is None or dispatch_ref is None:
            raise RuntimeError(
                f"Async CAM stage {stage_idx} receive has no pending dispatch",
            )
        local_ffn_output = afd_connector.recv_ffn_output(
            ref_tensor=dispatch_ref,
            ubatch_idx=stage_idx,
        )
        stage_hidden_states[stage_idx] = restore_cam_dispatch_output(
            local_ffn_output,
            dispatch_layout,
        )
        stage_dispatch_layouts[stage_idx] = None
        stage_dispatch_refs[stage_idx] = None

    last_moe_layer_offset = len(moe_layers) - 1
    first_layer = moe_layers[0]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        first_layer,
        0,
    )
    send_stage_attention(
        first_layer,
        0,
        topk_weights,
        topk_ids,
        router_logits,
    )

    for moe_layer_offset in range(last_moe_layer_offset):
        current_layer = moe_layers[moe_layer_offset]
        next_layer = moe_layers[moe_layer_offset + 1]

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            current_layer,
            1,
        )
        recv_stage_ffn(0)
        send_stage_attention(
            current_layer,
            1,
            topk_weights,
            topk_ids,
            router_logits,
        )

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            next_layer,
            0,
        )
        recv_stage_ffn(1)
        send_stage_attention(
            next_layer,
            0,
            topk_weights,
            topk_ids,
            router_logits,
        )

    last_layer = moe_layers[last_moe_layer_offset]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        last_layer,
        1,
    )
    recv_stage_ffn(0)
    send_stage_attention(
        last_layer,
        1,
        topk_weights,
        topk_ids,
        router_logits,
    )
    recv_stage_ffn(1)
    return _restore_async_moe_stage_state(
        stage_hidden_states,
        stage_residual,
        async_moe_ubatch_metadata,
    )


def _restore_async_moe_stage_state(
    stage_hidden_states: list[torch.Tensor],
    stage_residual: list[torch.Tensor | None],
    metadata: AsyncMoeUbatchMetadata,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = restore_async_moe_stage_outputs(
        stage_hidden_states,
        metadata,
    )
    if all(stage_output is None for stage_output in stage_residual):
        return hidden_states, None
    if any(stage_output is None for stage_output in stage_residual):
        raise RuntimeError(
            "Async CAM stages returned inconsistent residual layouts",
        )
    residuals = [
        stage_output for stage_output in stage_residual if stage_output is not None
    ]
    return hidden_states, restore_async_moe_stage_outputs(residuals, metadata)


_MISSING_FORWARD_CONTEXT_ATTR = object()


@contextmanager
def _use_async_moe_ubatch_forward_context(
    *,
    forward_context: ForwardContext,
    parent_afd_metadata: AFDForwardContextMetadata,
    async_moe_ubatch_metadata: AsyncMoeUbatchMetadata,
    local_input_tokens: int,
    local_actual_tokens: int,
    stage_idx: int,
) -> Iterator[None]:
    ubatch_slices = async_moe_ubatch_metadata.ubatch_slices
    stage_afd_metadata = _build_async_moe_stage_afd_metadata(
        parent_afd_metadata,
        local_input_tokens=local_input_tokens,
        local_actual_tokens=local_actual_tokens,
        stage_idx=stage_idx,
        num_stages=len(ubatch_slices),
    )

    stage_context_attr_names = (
        "attn_metadata",
        "additional_kwargs",
        "ubatch_idx",
        "num_ubatches",
        "num_tokens",
        "pad_size",
    )
    saved_attrs = {
        name: _read_forward_context_attr(forward_context, name)
        for name in stage_context_attr_names
    }

    original_kwargs = (
        forward_context.additional_kwargs
        if saved_attrs["additional_kwargs"] is not _MISSING_FORWARD_CONTEXT_ATTR
        else None
    )
    stage_kwargs = dict(original_kwargs or {})
    stage_kwargs["afd_metadata"] = stage_afd_metadata
    stage_input_tokens = int(ubatch_slices[stage_idx].num_tokens)
    stage_actual_tokens = int(
        async_moe_ubatch_metadata.stage_actual_token_counts[stage_idx],
    )
    if not 0 < stage_actual_tokens <= stage_input_tokens:
        raise ValueError(
            "Invalid Async CAM stage token counts: "
            f"input_tokens={stage_input_tokens}, "
            f"actual_tokens={stage_actual_tokens}",
        )

    try:
        forward_context.attn_metadata = async_moe_ubatch_metadata.attn_metadata[
            stage_idx
        ]
        forward_context.additional_kwargs = stage_kwargs
        forward_context.ubatch_idx = stage_idx
        forward_context.num_ubatches = len(ubatch_slices)
        if async_moe_ubatch_metadata.use_sequence_parallel:
            # FlashComm gathers the physical TP-local stage, removes its
            # trailing pad before attention, then restores that pad before the
            # output reduce-scatter.
            forward_context.num_tokens = stage_actual_tokens
            forward_context.pad_size = stage_input_tokens - stage_actual_tokens
        else:
            forward_context.num_tokens = stage_input_tokens
            forward_context.pad_size = 0
        yield
    finally:
        for name, value in saved_attrs.items():
            _restore_forward_context_attr(forward_context, name, value)


def _build_async_moe_stage_afd_metadata(
    parent_afd_metadata: AFDForwardContextMetadata,
    *,
    local_input_tokens: int,
    local_actual_tokens: int,
    stage_idx: int,
    num_stages: int,
) -> AFDForwardContextMetadata:
    stage_metadata = parent_afd_metadata.clone()
    stage_metadata.stage_idx = stage_idx
    stage_metadata.num_stages = num_stages
    stage_metadata.tokens_start_loc = [0]
    stage_metadata.requests_start_loc = [0]
    stage_metadata.tokens_lens = [local_input_tokens]
    stage_metadata.tokens_unpadded_lens = [local_actual_tokens]
    return stage_metadata


def _read_forward_context_attr(
    forward_context: ForwardContext,
    name: str,
) -> object:
    # Ascend installs ubatch fields dynamically on vLLM's ForwardContext.
    # Preserve absence as well as values so the parent context is restored
    # exactly after each stage.
    try:
        return getattr(forward_context, name)
    except AttributeError:
        return _MISSING_FORWARD_CONTEXT_ATTR


def _restore_forward_context_attr(
    forward_context: ForwardContext,
    name: str,
    value: object,
) -> None:
    if value is _MISSING_FORWARD_CONTEXT_ATTR:
        with suppress(AttributeError):
            delattr(forward_context, name)
        return
    setattr(forward_context, name, value)


__all__ = [
    "run_async_moe_ubatch_afd_forward",
    "run_attention_gate_afd_forward",
]
