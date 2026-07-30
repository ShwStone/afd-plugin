# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-independent Async CAM MoE stage contracts."""

from __future__ import annotations

from dataclasses import dataclass

ASYNC_MOE_NUM_STAGES = 2
ASYNC_MOE_REQUEST_SPLIT = "request"
ASYNC_MOE_TOKEN_SPLIT = "token"


@dataclass(frozen=True)
class AsyncMoeStage:
    """One ordered stage in the flattened Attention token layout.

    ``token_slice`` describes the stage's ordered real-token range in the
    parent batch. ``input_tokens`` includes only the minimum stage-local
    padding required by the Attention TP/SP layout.
    """

    request_slice: slice
    token_slice: slice
    input_tokens: int

    @property
    def num_tokens(self) -> int:
        return self.input_tokens

    @property
    def actual_tokens(self) -> int:
        return int(self.token_slice.stop) - int(self.token_slice.start)

    def is_empty(self) -> bool:
        return self.actual_tokens <= 0 or self.input_tokens < self.actual_tokens


__all__ = [
    "ASYNC_MOE_NUM_STAGES",
    "ASYNC_MOE_REQUEST_SPLIT",
    "ASYNC_MOE_TOKEN_SPLIT",
    "AsyncMoeStage",
]
