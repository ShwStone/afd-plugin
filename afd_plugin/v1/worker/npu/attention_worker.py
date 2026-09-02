# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD NPU Attention supports only vllm-ascend model runner v1",
            )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        self.model_runner = AFDNPUAttentionModelRunner(
            self.vllm_config,
            self.device,
        )

    # Override reason: vLLM-Ascend's worker profiler uses Level1 and does not
    # retain the Level2/MSTX settings required by AFD correlation traces.
    # Behavior: route the native vLLM profile RPC to the AFD Attention runner.
    # Signature matches the pinned NPUWorker.profile; no parameters are added.
    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
    ) -> None:
        if is_start:
            self.model_runner.start_profile(profile_prefix, self.rank)
        else:
            self.model_runner.stop_profile()


__all__ = ["AFDNPUAttentionWorker"]
