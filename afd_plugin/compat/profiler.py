# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request-controlled CUDA profiler for AFD GPU runners."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    from vllm.config import ProfilerConfig

logger = logging.getLogger(__name__)

AFDGPUProfilerRole = Literal["attention", "ffn"]


class AFDGPUProfiler:
    """Own one role's request-controlled torch profiler lifecycle."""

    def __init__(
        self,
        role: AFDGPUProfilerRole,
        profiler_config: ProfilerConfig,
    ) -> None:
        self.role = role
        self.profiler_config = profiler_config
        self._profiler: torch.profiler.profile | None = None
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
                    "AFD GPU %s profiler is already active; ignoring start",
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
                "AFD GPU %s profiler started. Traces will be saved to: %s",
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
        logger.info("AFD GPU %s profiler stopped", self.role)

    def _create_profiler(self, trace_name: str) -> torch.profiler.profile:
        import torch

        config = self.profiler_config
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                config.torch_profiler_dir,
                worker_name=trace_name,
                use_gzip=config.torch_profiler_use_gzip,
            ),
            record_shapes=config.torch_profiler_record_shapes,
            profile_memory=config.torch_profiler_with_memory,
            with_stack=config.torch_profiler_with_stack,
            with_flops=config.torch_profiler_with_flops,
        )


def validate_afd_profiler_config(profiler_config: ProfilerConfig) -> None:
    """Reject profiler modes that conflict with explicit HTTP windows."""

    if profiler_config.profiler != "torch":
        raise ValueError("AFD profiling requires --profiler-config.profiler=torch")
    if (
        profiler_config.delay_iterations
        or profiler_config.max_iterations
        or profiler_config.warmup_iterations
        or profiler_config.wait_iterations
    ):
        raise ValueError(
            "AFD profiling does not support iteration schedules; use "
            "/start_profile and /stop_profile",
        )


def create_afd_gpu_profiler(
    role: AFDGPUProfilerRole,
    profiler_config: ProfilerConfig | None,
) -> AFDGPUProfiler | None:
    """Create an AFD profiler when vLLM request profiling is configured."""

    if profiler_config is None or profiler_config.profiler is None:
        return None
    validate_afd_profiler_config(profiler_config)
    return AFDGPUProfiler(role, profiler_config)


def start_afd_gpu_profiler(
    profiler: AFDGPUProfiler | None,
    *,
    profile_prefix: str | None,
    global_rank: int,
) -> None:
    if profiler is None:
        raise RuntimeError(
            "AFD GPU profiling is not enabled; configure --profiler-config"
        )
    profiler.start(profile_prefix=profile_prefix, global_rank=global_rank)


def step_afd_gpu_profiler(profiler: AFDGPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_gpu_profiler(profiler: AFDGPUProfiler | None) -> None:
    if profiler is not None:
        profiler.stop()


def _trace_name(
    role: AFDGPUProfilerRole,
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
    "AFDGPUProfiler",
    "create_afd_gpu_profiler",
    "start_afd_gpu_profiler",
    "step_afd_gpu_profiler",
    "stop_afd_gpu_profiler",
    "validate_afd_profiler_config",
]
