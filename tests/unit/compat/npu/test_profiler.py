from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import profiler as profiler_module_under_test
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    start_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)


def test_npu_profiler_is_disabled_without_vllm_config():
    assert create_afd_npu_profiler("attention", None) is None
    assert (
        create_afd_npu_profiler(
            "attention",
            SimpleNamespace(profiler=None),
        )
        is None
    )


def test_npu_profiler_requires_torch_config():
    with pytest.raises(ValueError, match="profiler=torch"):
        create_afd_npu_profiler(
            "attention",
            SimpleNamespace(profiler="cuda"),
        )


def test_npu_profiler_is_lazy_repeatable_and_keeps_level2_mstx(monkeypatch):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setattr(
        profiler_module_under_test,
        "_trace_name",
        lambda role, *, profile_prefix, global_rank, run_index: (
            f"{profile_prefix or 'afd'}-{role}-rank{global_rank}-run{run_index}"
        ),
    )
    controller = create_afd_npu_profiler(
        "attention",
        _request_profiler_config(),
    )

    assert controller is not None
    assert profiler_module.created_profilers == []

    start_afd_npu_profiler(controller, profile_prefix="request", global_rank=2)
    step_afd_npu_profiler(controller)
    stop_afd_npu_profiler(controller)
    start_afd_npu_profiler(controller, profile_prefix="request", global_rank=2)

    first = profiler_module.created_profilers[0]
    assert first.started is True
    assert first.steps == 1
    assert first.stopped is True
    assert len(profiler_module.created_profilers) == 2
    assert profiler_module.worker_names == [
        "request-attention-rank2-run0",
        "request-attention-rank2-run1",
    ]
    assert profiler_module.experimental_kwargs["mstx"] is True
    assert profiler_module.experimental_kwargs["profiler_level"] == "level2"


def test_start_npu_profiler_requires_configuration():
    with pytest.raises(RuntimeError, match="not enabled"):
        start_afd_npu_profiler(None, profile_prefix=None, global_rank=0)


def _request_profiler_config():
    return SimpleNamespace(
        profiler="torch",
        torch_profiler_dir="/tmp/request-profile",
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        delay_iterations=0,
        max_iterations=0,
        warmup_iterations=0,
        wait_iterations=0,
    )


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self):
        self.steps += 1


class _FakeTorchNPUProfiler:
    class ExportType:
        Text = "text"

    class ProfilerLevel:
        Level2 = "level2"

    class AiCMetrics:
        AiCoreNone = "aicore_none"

    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    def __init__(self):
        self.created_profilers = []
        self.profile_kwargs = []
        self.worker_names = []
        self.experimental_kwargs = None
        self._ExperimentalConfig = self._experimental_config

    def _experimental_config(self, **kwargs):
        self.experimental_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir, *, worker_name=None):
        if worker_name is not None:
            self.worker_names.append(worker_name)
        return ("handler", trace_dir, worker_name)

    def profile(self, **kwargs):
        self.profile_kwargs.append(kwargs)
        profiler = _StepProfiler()
        self.created_profilers.append(profiler)
        return profiler
