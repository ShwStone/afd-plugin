# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in tensor dumps for Async CAM accuracy investigations."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import torch

TENSOR_DUMP_ENABLE_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG"
TENSOR_DUMP_DIR_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_DIR"
TENSOR_DUMP_LAYERS_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS"
TENSOR_DUMP_POINTS_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_POINTS"
TENSOR_DUMP_TOKEN_INDICES_ENV: Final[str] = (
    "AFD_ASYNC_MOE_PRECISION_DEBUG_TOKEN_INDICES"
)
TENSOR_DUMP_EDGE_ROWS_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_EDGE_ROWS"
TENSOR_DUMP_FULL_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS"
TENSOR_DUMP_SYNC_ENV: Final[str] = "AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC"

ATTENTION_DISPATCH_HIDDEN: Final[str] = "attn_dispatch_hidden"
ATTENTION_TOPK_IDS: Final[str] = "attn_topk_ids"
ATTENTION_TOPK_WEIGHTS: Final[str] = "attn_topk_weights"
ATTENTION_ROUTER_LOGITS: Final[str] = "attn_router_logits"
ATTENTION_FFN_OUTPUT: Final[str] = "attn_ffn_output"
FFN_ROUTED_INPUT: Final[str] = "ffn_routed_input"
FFN_ROUTED_OUTPUT: Final[str] = "ffn_routed_output"
FFN_SHARED_INPUT: Final[str] = "ffn_shared_input"
FFN_SHARED_OUTPUT: Final[str] = "ffn_shared_output"
FFN_GROUP_LIST: Final[str] = "ffn_group_list"

DEFAULT_TENSOR_DUMP_POINTS: Final[frozenset[str]] = frozenset(
    {
        ATTENTION_DISPATCH_HIDDEN,
        ATTENTION_TOPK_IDS,
        ATTENTION_TOPK_WEIGHTS,
        ATTENTION_FFN_OUTPUT,
        FFN_ROUTED_INPUT,
        FFN_ROUTED_OUTPUT,
        FFN_GROUP_LIST,
    }
)
SUPPORTED_TENSOR_DUMP_POINTS: Final[frozenset[str]] = frozenset(
    {
        *DEFAULT_TENSOR_DUMP_POINTS,
        ATTENTION_ROUTER_LOGITS,
        FFN_SHARED_INPUT,
        FFN_SHARED_OUTPUT,
    }
)
TRUE_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
DEFAULT_EDGE_ROWS: Final[int] = 2
TENSOR_DUMP_SCHEMA_VERSION: Final[int] = 1

TensorRowCoordinate = Literal["global_token", "rank_local"]


@dataclass(frozen=True, slots=True)
class AsyncCamTensorDumpConfig:
    """Immutable configuration for one model-forward dump session."""

    enabled: bool = False
    output_dir: Path | None = None
    layers: frozenset[int] = frozenset()
    points: frozenset[str] = DEFAULT_TENSOR_DUMP_POINTS
    token_indices: tuple[int, ...] = ()
    edge_rows: int = DEFAULT_EDGE_ROWS
    full_tensors: bool = False
    synchronize: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> AsyncCamTensorDumpConfig:
        """Parse tensor-dump controls without retaining mutable global state."""

        values = os.environ if environ is None else environ
        if not _read_bool(values, TENSOR_DUMP_ENABLE_ENV):
            return cls()

        output_dir_value = values.get(TENSOR_DUMP_DIR_ENV, "").strip()
        if not output_dir_value:
            raise ValueError(
                f"{TENSOR_DUMP_DIR_ENV} is required when {TENSOR_DUMP_ENABLE_ENV}=1"
            )
        layers = _parse_nonnegative_ints(
            values.get(TENSOR_DUMP_LAYERS_ENV, ""),
            field_name=TENSOR_DUMP_LAYERS_ENV,
        )
        if not layers:
            raise ValueError(f"{TENSOR_DUMP_LAYERS_ENV} must select at least one layer")

        point_value = values.get(TENSOR_DUMP_POINTS_ENV, "").strip()
        points = (
            frozenset(_parse_csv(point_value))
            if point_value
            else DEFAULT_TENSOR_DUMP_POINTS
        )
        unsupported_points = points - SUPPORTED_TENSOR_DUMP_POINTS
        if unsupported_points:
            raise ValueError(
                f"{TENSOR_DUMP_POINTS_ENV} contains unsupported points: "
                f"{sorted(unsupported_points)}"
            )

        token_indices = _parse_nonnegative_ints(
            values.get(TENSOR_DUMP_TOKEN_INDICES_ENV, ""),
            field_name=TENSOR_DUMP_TOKEN_INDICES_ENV,
        )
        edge_rows = int(values.get(TENSOR_DUMP_EDGE_ROWS_ENV, str(DEFAULT_EDGE_ROWS)))
        if edge_rows < 0:
            raise ValueError(f"{TENSOR_DUMP_EDGE_ROWS_ENV} must be non-negative")

        return cls(
            enabled=True,
            output_dir=Path(output_dir_value).expanduser(),
            layers=frozenset(layers),
            points=points,
            token_indices=token_indices,
            edge_rows=edge_rows,
            full_tensors=_read_bool(values, TENSOR_DUMP_FULL_ENV),
            synchronize=_read_bool(values, TENSOR_DUMP_SYNC_ENV),
        )

    def should_dump(self, layer_idx: int, point: str) -> bool:
        """Return whether a tensor point is selected for this layer."""

        return self.enabled and layer_idx in self.layers and point in self.points


def dump_async_cam_tensor(
    tensor: torch.Tensor,
    config: AsyncCamTensorDumpConfig,
    *,
    role: Literal["attention", "ffn"],
    role_rank: int,
    layer_idx: int,
    stage_idx: int,
    point: str,
    row_coordinate: TensorRowCoordinate,
    row_start: int = 0,
    valid_rows: int | None = None,
    transaction_id: str | None = None,
) -> Path | None:
    """Dump selected first-dimension rows and their comparison metadata.

    The first observation for each role/rank/layer/stage/point wins. This keeps
    repeated decode steps bounded and gives baseline and ubatch runs stable
    filenames when they use separate output directories.
    """

    if not config.should_dump(layer_idx, point):
        return None
    if config.output_dir is None:
        raise RuntimeError("enabled Async CAM tensor dump has no output directory")
    if point not in SUPPORTED_TENSOR_DUMP_POINTS:
        raise ValueError(f"unsupported Async CAM tensor dump point: {point}")
    if tensor.ndim == 0:
        raise ValueError(f"Async CAM tensor dump requires a row dimension: {point}")

    tensor_rows = int(tensor.shape[0])
    selected_valid_rows = tensor_rows if valid_rows is None else int(valid_rows)
    if not 0 <= selected_valid_rows <= tensor_rows:
        raise ValueError(
            f"invalid valid_rows for {point}: valid_rows={selected_valid_rows}, "
            f"tensor_rows={tensor_rows}"
        )
    if row_start < 0:
        raise ValueError(f"row_start must be non-negative, got {row_start}")

    if config.synchronize and tensor.device.type == "npu":
        torch.npu.synchronize(tensor.device)

    local_indices = _select_local_rows(
        config,
        row_coordinate=row_coordinate,
        row_start=row_start,
        valid_rows=selected_valid_rows,
    )
    if local_indices:
        index_tensor = torch.tensor(local_indices, device=tensor.device)
        sampled_tensor = tensor.detach().index_select(0, index_tensor).cpu()
    else:
        sampled_tensor = tensor.detach()[:0].cpu()

    global_or_local_indices = tuple(row_start + index for index in local_indices)
    payload = {
        "schema_version": TENSOR_DUMP_SCHEMA_VERSION,
        "role": role,
        "role_rank": role_rank,
        "layer_idx": layer_idx,
        "stage_idx": stage_idx,
        "point": point,
        "transaction_id": transaction_id,
        "row_coordinate": row_coordinate,
        "row_start": row_start,
        "valid_rows": selected_valid_rows,
        "original_shape": tuple(tensor.shape),
        "original_dtype": str(tensor.dtype),
        "selected_row_indices": global_or_local_indices,
        "sampled_tensor": sampled_tensor,
        "sample_sha256": _tensor_sha256(sampled_tensor),
        "sample_summary": _tensor_summary(sampled_tensor),
    }

    output_path = (
        config.output_dir
        / f"{role}-rank-{role_rank:03d}"
        / f"layer-{layer_idx:03d}"
        / f"stage-{stage_idx:02d}-{point}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as output_file:
            torch.save(payload, output_file)
    except FileExistsError:
        return None
    return output_path


def _select_local_rows(
    config: AsyncCamTensorDumpConfig,
    *,
    row_coordinate: TensorRowCoordinate,
    row_start: int,
    valid_rows: int,
) -> tuple[int, ...]:
    if config.full_tensors:
        return tuple(range(valid_rows))
    if config.token_indices and row_coordinate == "global_token":
        row_stop = row_start + valid_rows
        return tuple(
            token_index - row_start
            for token_index in config.token_indices
            if row_start <= token_index < row_stop
        )
    if valid_rows == 0 or config.edge_rows == 0:
        return ()
    leading_stop = min(config.edge_rows, valid_rows)
    trailing_start = max(leading_stop, valid_rows - config.edge_rows)
    return tuple(range(leading_stop)) + tuple(range(trailing_start, valid_rows))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    byte_tensor = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    return hashlib.sha256(bytes(byte_tensor)).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, float | int | None]:
    if tensor.numel() == 0:
        return {"numel": 0, "min": None, "max": None, "mean": None, "l2": None}
    numeric_tensor = tensor.to(torch.float64)
    return {
        "numel": tensor.numel(),
        "min": float(numeric_tensor.min().item()),
        "max": float(numeric_tensor.max().item()),
        "mean": float(numeric_tensor.mean().item()),
        "l2": float(torch.linalg.vector_norm(numeric_tensor).item()),
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_nonnegative_ints(value: str, *, field_name: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    parsed = tuple(int(item) for item in _parse_csv(value))
    if any(item < 0 for item in parsed):
        raise ValueError(f"{field_name} must contain non-negative integers")
    return tuple(dict.fromkeys(parsed))


def _read_bool(values: Mapping[str, str], field_name: str) -> bool:
    return values.get(field_name, "").strip().lower() in TRUE_ENV_VALUES


ASYNC_CAM_TENSOR_DUMP_CONFIG: Final[AsyncCamTensorDumpConfig] = (
    AsyncCamTensorDumpConfig.from_env()
)


__all__ = [
    "ASYNC_CAM_TENSOR_DUMP_CONFIG",
    "ATTENTION_DISPATCH_HIDDEN",
    "ATTENTION_FFN_OUTPUT",
    "ATTENTION_ROUTER_LOGITS",
    "ATTENTION_TOPK_IDS",
    "ATTENTION_TOPK_WEIGHTS",
    "AsyncCamTensorDumpConfig",
    "FFN_GROUP_LIST",
    "FFN_ROUTED_INPUT",
    "FFN_ROUTED_OUTPUT",
    "FFN_SHARED_INPUT",
    "FFN_SHARED_OUTPUT",
    "dump_async_cam_tensor",
]
