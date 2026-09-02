# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from tools.benchmarks import fp_orchestrate


def test_profile_envs_configure_request_controlled_output(monkeypatch):
    monkeypatch.setenv("FP_PROFILE", "1")
    monkeypatch.setenv("FP_RESULT_ROOT", "/tmp/results")

    profile_envs = fp_orchestrate._profile_envs("afd", "A2")

    assert profile_envs["attention"]["VLLM_TORCH_PROFILER_DIR"] == (
        "/tmp/results/02_profiles_32/traces/attention"
    )
    assert profile_envs["ffn"]["VLLM_TORCH_PROFILER_DIR"] == (
        "/tmp/results/02_profiles_32/traces/ffn"
    )
    assert not any(
        name.startswith(("AFD_GPU_", "AFD_NPU_"))
        for role_env in profile_envs.values()
        for name in role_env
    )


def test_correlation_can_be_enabled_without_profiler(monkeypatch):
    monkeypatch.setenv("FP_CORRELATION", "1")
    monkeypatch.setenv("FP_RESULT_ROOT", "/tmp/results")

    role_envs = fp_orchestrate._profile_envs("afd", "A2")

    assert role_envs["attention"]["AFD_TRACE_ENABLE"] == "1"
    assert role_envs["ffn"]["AFD_TRACE_ENABLE"] == "1"
    assert all(
        "VLLM_TORCH_PROFILER_DIR" not in role_env
        for role_env in role_envs.values()
    )


def test_start_afd_profile_starts_ffn_before_attention(monkeypatch):
    calls = []
    monkeypatch.setenv("FP_NODE0", "attention-pod")
    monkeypatch.setenv("FP_NODE1", "ffn-pod")
    monkeypatch.setattr(
        fp_orchestrate,
        "_set_profile_state",
        lambda pod, port, *, start: calls.append((pod, port, start)),
    )

    fp_orchestrate._start_afd_profile()

    assert calls == [
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, True),
        ("attention-pod", fp_orchestrate.ATTENTION_SERVER_PORT, True),
    ]


def test_start_afd_profile_rolls_back_ffn_on_attention_failure(monkeypatch):
    calls = []
    monkeypatch.setenv("FP_NODE0", "attention-pod")
    monkeypatch.setenv("FP_NODE1", "ffn-pod")

    def set_profile_state(pod, port, *, start):
        calls.append((pod, port, start))
        if pod == "attention-pod":
            raise RuntimeError("attention start failed")

    monkeypatch.setattr(
        fp_orchestrate,
        "_set_profile_state",
        set_profile_state,
    )

    with pytest.raises(RuntimeError, match="attention start failed"):
        fp_orchestrate._start_afd_profile()

    assert calls == [
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, True),
        ("attention-pod", fp_orchestrate.ATTENTION_SERVER_PORT, True),
        ("attention-pod", fp_orchestrate.ATTENTION_SERVER_PORT, False),
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, False),
    ]


def test_start_afd_profile_rolls_back_partial_ffn_failure(monkeypatch):
    calls = []
    monkeypatch.setenv("FP_NODE0", "attention-pod")
    monkeypatch.setenv("FP_NODE1", "ffn-pod")

    def set_profile_state(pod, port, *, start):
        calls.append((pod, port, start))
        if pod == "ffn-pod" and start:
            raise RuntimeError("ffn start failed")

    monkeypatch.setattr(
        fp_orchestrate,
        "_set_profile_state",
        set_profile_state,
    )

    with pytest.raises(RuntimeError, match="ffn start failed"):
        fp_orchestrate._start_afd_profile()

    assert calls == [
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, True),
        ("attention-pod", fp_orchestrate.ATTENTION_SERVER_PORT, False),
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, False),
    ]


def test_stop_afd_profile_attempts_both_roles(monkeypatch):
    calls = []
    monkeypatch.setenv("FP_NODE0", "attention-pod")
    monkeypatch.setenv("FP_NODE1", "ffn-pod")

    def set_profile_state(pod, port, *, start):
        calls.append((pod, port, start))
        if pod == "attention-pod":
            raise RuntimeError("attention stop failed")

    monkeypatch.setattr(
        fp_orchestrate,
        "_set_profile_state",
        set_profile_state,
    )

    with pytest.raises(RuntimeError, match="attention stop failed"):
        fp_orchestrate._stop_afd_profile()

    assert calls == [
        ("attention-pod", fp_orchestrate.ATTENTION_SERVER_PORT, False),
        ("ffn-pod", fp_orchestrate.FFN_SERVER_PORT, False),
    ]
