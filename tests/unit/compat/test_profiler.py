from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from afd_plugin.compat import profiler as profiler_module_under_test
from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    start_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)


def test_gpu_profiler_is_disabled_without_vllm_config():
    assert create_afd_gpu_profiler("attention", None) is None
    assert (
        create_afd_gpu_profiler(
            "attention",
            SimpleNamespace(profiler=None),
        )
        is None
    )


def test_gpu_profiler_requires_torch_config():
    with pytest.raises(ValueError, match="profiler=torch"):
        create_afd_gpu_profiler(
            "attention",
            SimpleNamespace(profiler="cuda"),
        )


def test_gpu_profiler_rejects_iteration_schedules():
    config = _request_profiler_config()
    config.max_iterations = 10

    with pytest.raises(ValueError, match="does not support iteration schedules"):
        create_afd_gpu_profiler("attention", config)


def test_gpu_profiler_is_lazy_idempotent_and_repeatable(monkeypatch):
    profiler_module = _install_fake_torch(monkeypatch)
    config = _request_profiler_config()

    controller = create_afd_gpu_profiler("attention", config)

    assert controller is not None
    assert profiler_module.created_profilers == []

    start_afd_gpu_profiler(controller, profile_prefix="request", global_rank=3)
    step_afd_gpu_profiler(controller)
    start_afd_gpu_profiler(controller, profile_prefix="ignored", global_rank=3)

    first = profiler_module.created_profilers[0]
    assert first.started is True
    assert first.steps == 1
    assert len(profiler_module.created_profilers) == 1
    assert profiler_module.worker_names == ["request-attention-rank3-run0"]
    assert "schedule" not in profiler_module.profile_kwargs[0]

    stop_afd_gpu_profiler(controller)
    stop_afd_gpu_profiler(controller)
    start_afd_gpu_profiler(controller, profile_prefix="request", global_rank=3)

    assert first.stopped is True
    assert len(profiler_module.created_profilers) == 2
    assert profiler_module.worker_names[-1] == "request-attention-rank3-run1"


def test_concurrent_gpu_profiler_stops_release_once(monkeypatch):
    profiler_module = _install_fake_torch(monkeypatch)
    controller = create_afd_gpu_profiler("attention", _request_profiler_config())
    start_afd_gpu_profiler(controller, profile_prefix=None, global_rank=0)

    threads = [
        threading.Thread(target=stop_afd_gpu_profiler, args=(controller,))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert profiler_module.created_profilers[0].stop_calls == 1


def test_failed_gpu_profiler_stop_still_allows_restart(monkeypatch):
    profiler_module = _install_fake_torch(monkeypatch)
    controller = create_afd_gpu_profiler("attention", _request_profiler_config())
    start_afd_gpu_profiler(controller, profile_prefix=None, global_rank=0)
    profiler_module.created_profilers[0].stop_error = RuntimeError("flush failed")

    with pytest.raises(RuntimeError, match="flush failed"):
        stop_afd_gpu_profiler(controller)
    start_afd_gpu_profiler(controller, profile_prefix=None, global_rank=0)

    assert len(profiler_module.created_profilers) == 2


def test_start_gpu_profiler_requires_configuration():
    with pytest.raises(RuntimeError, match="not enabled"):
        start_afd_gpu_profiler(None, profile_prefix=None, global_rank=0)


def _install_fake_torch(monkeypatch) -> _FakeTorchProfiler:
    profiler_module = _FakeTorchProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setattr(
        profiler_module_under_test,
        "_trace_name",
        lambda role, *, profile_prefix, global_rank, run_index: (
            f"{profile_prefix or 'afd'}-{role}-rank{global_rank}-run{run_index}"
        ),
    )
    return profiler_module


def _request_profiler_config():
    return SimpleNamespace(
        profiler="torch",
        torch_profiler_dir="/tmp/request-profile",
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=False,
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
        self.stop_calls = 0
        self.stop_error = None

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped = True

    def step(self):
        self.steps += 1


class _FakeTorchProfiler:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    def __init__(self):
        self.created_profilers = []
        self.profile_kwargs = []
        self.worker_names = []

    def tensorboard_trace_handler(
        self,
        trace_dir,
        *,
        worker_name=None,
        use_gzip=False,
    ):
        if worker_name is not None:
            self.worker_names.append(worker_name)
        return ("handler", trace_dir, worker_name, use_gzip)

    def profile(self, **kwargs):
        self.profile_kwargs.append(kwargs)
        profiler = _StepProfiler()
        self.created_profilers.append(profiler)
        return profiler
