# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request-controlled NPU profiler for AFD Ascend runners."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Literal, Protocol

from afd_plugin.compat.profiler import validate_afd_profiler_config

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

logger = logging.getLogger(__name__)

AFDNPUProfilerRole = Literal["attention", "ffn"]


class _TorchNPUProfiler(Protocol):
    def start(self) -> None: ...

    def step(self) -> None: ...

    def stop(self) -> None: ...


class AFDNPUProfiler:
    """Own one role's request-controlled torch-npu profiler lifecycle."""

    def __init__(
        self,
        role: AFDNPUProfilerRole,
        profiler_config: ProfilerConfig,
    ) -> None:
        self.role = role
        self.profiler_config = profiler_config
        self._profiler: _TorchNPUProfiler | None = None
        self._lock = threading.Lock()
        self._run_index = 0

    def start(
        self,
        *,
        profile_prefix: str | None,
        global_rank: int,
    ) -> None:
        """Start a trace; duplicate starts are idempotent."""

        with self._lock:
            if self._profiler is not None:
                logger.warning(
                    "AFD NPU %s profiler is already active; ignoring start",
                    self.role,
                )
                return

            trace_name = _trace_name(
                self.role,
                profile_prefix=profile_prefix,
                global_rank=global_rank,
                run_index=self._run_index,
            )
            profiler = self._create_profiler(trace_name)
            profiler.start()
            self._profiler = profiler
            self._run_index += 1
            logger.info(
                "AFD NPU %s profiler started. Traces will be saved to: %s",
                self.role,
                self.profiler_config.torch_profiler_dir,
            )

    def step(self) -> None:
        with self._lock:
            if self._profiler is not None:
                self._profiler.step()

    def stop(self) -> None:
        """Stop and release the active profiler; duplicate stops are idempotent."""

        with self._lock:
            profiler = self._profiler
            if profiler is None:
                return
            try:
                profiler.stop()
            finally:
                self._profiler = None
        logger.info("AFD NPU %s profiler stopped", self.role)

    def _create_profiler(self, trace_name: str) -> _TorchNPUProfiler:
        import torch_npu

        config = self.profiler_config
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            # Preserve MSTX ranges used to align sidecars and device events.
            mstx=True,
        )
        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=config.torch_profiler_with_stack,
            with_modules=config.torch_profiler_with_stack,
            record_shapes=config.torch_profiler_record_shapes,
            profile_memory=config.torch_profiler_with_memory,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )


def create_afd_npu_profiler(
    role: AFDNPUProfilerRole,
    profiler_config: ProfilerConfig | None,
) -> AFDNPUProfiler | None:
    """Create an AFD profiler when vLLM request profiling is configured."""

    if profiler_config is None or profiler_config.profiler is None:
        return None
    validate_afd_profiler_config(profiler_config)
    return AFDNPUProfiler(role, profiler_config)


def start_afd_npu_profiler(
    profiler: AFDNPUProfiler | None,
    *,
    profile_prefix: str | None,
    global_rank: int,
) -> None:
    if profiler is None:
        raise RuntimeError(
            "AFD NPU profiling is not enabled; configure --profiler-config"
        )
    profiler.start(profile_prefix=profile_prefix, global_rank=global_rank)


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.stop()


def _trace_name(
    role: AFDNPUProfilerRole,
    *,
    profile_prefix: str | None,
    global_rank: int,
    run_index: int,
) -> str:
    from vllm.distributed.utils import get_worker_rank_suffix

    rank_suffix = get_worker_rank_suffix(global_rank=global_rank)
    prefix = profile_prefix or "afd"
    return f"{prefix}_{role}_{rank_suffix}_run{run_index}"


__all__ = [
    "AFDNPUProfiler",
    "create_afd_npu_profiler",
    "start_afd_npu_profiler",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
