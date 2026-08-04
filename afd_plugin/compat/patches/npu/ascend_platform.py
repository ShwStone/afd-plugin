# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch vLLM-Ascend platform config normalization for AFD-owned DBO.

Upstream source: ``vllm_ascend/platform.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_optional_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ASCEND_PLATFORM_PATCH_ATTR = "_afd_plugin_ascend_platform_patch_state"
_ASCEND_FLASHCOMM1_EP_PATCH_ATTR = "_afd_plugin_ascend_flashcomm1_ep_patch_state"


def apply_afd_ascend_dbo_config_patch() -> None:
    """Preserve AFD-owned DBO settings during vLLM-Ascend config normalization.

    vLLM-Ascend's platform compatibility pass disables DBO/ubatching fields for
    ordinary NPU runs. AFD owns its NPU ubatching path, so this patch snapshots
    those fields for AFD-enabled configs, lets upstream normalization run, then
    restores the AFD DBO values. The patch is a no-op when vLLM-Ascend is not
    importable or when this process has already installed the wrapper.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return

    if hasattr(NPUPlatform, _ASCEND_PLATFORM_PATCH_ATTR):
        return

    original_fix_incompatible_config = NPUPlatform._fix_incompatible_config

    # Patch reason: vLLM-Ascend resets DBO fields inside NPUPlatform config
    # normalization, while AFD now owns the Ascend DBO/ubatching path.
    # Patch functionality: preserves upstream normalization for non-AFD configs and
    # restores AFD DBO fields after upstream normalization for AFD-enabled configs.
    # Expansion exception: upstream _fix_incompatible_config is platform-owned
    # normalization; keep narrow original-function delegation so this patch only
    # owns the AFD DBO preservation.
    # Signature: matches upstream; no added parameters.
    def _fix_incompatible_config(vllm_config: VllmConfig) -> Any:
        # ### PATCH START: AFD DBO config preservation
        saved = _snapshot_afd_dbo_config(vllm_config)
        # ### PATCH END: AFD DBO config preservation
        result = original_fix_incompatible_config(vllm_config)
        # ### PATCH START: AFD DBO config preservation
        if saved is not None:
            _restore_afd_dbo_config(vllm_config, saved)
        # ### PATCH END: AFD DBO config preservation
        return result

    NPUPlatform._fix_incompatible_config = staticmethod(_fix_incompatible_config)
    setattr(
        NPUPlatform,
        _ASCEND_PLATFORM_PATCH_ATTR,
        original_fix_incompatible_config,
    )


def apply_afd_ascend_flashcomm1_ep_config_patch() -> None:
    """Keep EP disabled for CAMAsync AFD Attention workers using FlashComm1.

    vLLM-Ascend v0.19.1rc1 classifies the process from the Hugging Face model
    config before AFD replaces the model with its role-specific implementation.
    Consequently, FlashComm1's generic MoE validation requires EP even though
    the AFD Attention model does not construct or execute routed experts.

    The wrapper presents EP as enabled only while the upstream platform method
    performs its FlashComm1 compatibility check. The final runtime config keeps
    EP disabled so vLLM does not flatten DP, PCP, and TP ranks into an expert
    group for the Attention-only process. This is config-time only and adds no
    forward-path overhead. Remove the wrapper after upstream FlashComm1
    validation distinguishes instantiated MoE layers from the model's static
    architecture metadata.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return

    if hasattr(NPUPlatform, _ASCEND_FLASHCOMM1_EP_PATCH_ATTR):
        return
    if not hasattr(NPUPlatform, "check_and_update_config"):
        return

    original_check_and_update_config = NPUPlatform.check_and_update_config

    # Patch reason: vLLM-Ascend identifies a CAMAsync AFD Attention process as
    # a MoE model from its unmodified Hugging Face config and therefore
    # requires EP whenever FlashComm1 is enabled, even though this role owns no
    # experts.
    # Patch functionality: expose EP only to the upstream FlashComm1 config
    # check, then leave EP disabled for the actual CAMAsync Attention runtime.
    # Expansion exception: upstream check_and_update_config is a large platform
    # normalization method; delegate to it so this patch only owns the AFD
    # Attention EP state around that call.
    # Signature: matches upstream; no added parameters.
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # ### PATCH START: AFD Attention FlashComm1 without expert parallelism
        if not _is_afd_async_attention_config(vllm_config):
            return original_check_and_update_config(vllm_config)

        from vllm_ascend.utils import enable_sp

        parallel_config = vllm_config.parallel_config
        parallel_config.enable_expert_parallel = bool(enable_sp(vllm_config))
        try:
            return original_check_and_update_config(vllm_config)
        finally:
            parallel_config.enable_expert_parallel = False
        # ### PATCH END: AFD Attention FlashComm1 without expert parallelism

    NPUPlatform.check_and_update_config = classmethod(check_and_update_config)
    setattr(
        NPUPlatform,
        _ASCEND_FLASHCOMM1_EP_PATCH_ATTR,
        original_check_and_update_config,
    )


def _snapshot_afd_dbo_config(vllm_config: VllmConfig) -> dict[str, bool | int] | None:
    if not _has_valid_afd_config(vllm_config):
        return None
    parallel_config = vllm_config.parallel_config
    return {
        "enable_dbo": parallel_config.enable_dbo,
        "use_ubatching": parallel_config.use_ubatching,
        "ubatch_size": parallel_config.ubatch_size,
    }


def _restore_afd_dbo_config(
    vllm_config: VllmConfig,
    saved: dict[str, bool | int],
) -> None:
    parallel_config = vllm_config.parallel_config
    if not (
        saved["enable_dbo"]
        or saved["use_ubatching"]
        or int(saved["ubatch_size"] or 0) != 0
    ):
        return
    parallel_config.enable_dbo = saved["enable_dbo"]
    parallel_config.ubatch_size = saved["ubatch_size"]


def _has_valid_afd_config(vllm_config: VllmConfig) -> bool:
    try:
        return parse_optional_afd_config(vllm_config, validate=True) is not None
    except Exception:
        return False


def _is_afd_async_attention_config(vllm_config: VllmConfig) -> bool:
    try:
        afd_config = parse_optional_afd_config(vllm_config, validate=True)
    except Exception:
        return False
    return (
        afd_config is not None
        and afd_config.role == "attention"
        and afd_config.connector == AFD_ASYNC_CONNECTOR
    )


__all__ = [
    "apply_afd_ascend_dbo_config_patch",
    "apply_afd_ascend_flashcomm1_ep_config_patch",
]
