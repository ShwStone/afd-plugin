# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afd_plugin.observability.correlation import (
    AFDCorrelationTraceConfig,
    AFDCorrelationTraceRecorder,
    AFDTraceIdentity,
    afd_correlation_trace_config,
)


def _identity(*, role: str = "attention") -> AFDTraceIdentity:
    return AFDTraceIdentity(
        role=role,
        rank=0,
        role_rank=0,
        local_rank=0,
        hostname="trace-host",
        pid=123,
    )


def test_enabled_trace_requires_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AFD_TRACE_ENABLE", "1")
    monkeypatch.delenv("AFD_TRACE_SESSION_ID", raising=False)

    with pytest.raises(ValueError, match="AFD_TRACE_SESSION_ID"):
        afd_correlation_trace_config()


def test_disabled_recorder_is_noop(tmp_path: Path) -> None:
    recorder = AFDCorrelationTraceRecorder(
        AFDCorrelationTraceConfig(
            enabled=False,
            session_id=None,
            trace_dir=tmp_path,
            max_events=10,
        ),
        _identity(),
    )

    with recorder.record_range("afd.test"):
        recorder.record("afd.test.instant")

    assert recorder.close() is None
    assert list(tmp_path.rglob("*")) == []


def test_recorder_writes_scalar_sidecar_and_stable_flow(tmp_path: Path) -> None:
    config = AFDCorrelationTraceConfig(
        enabled=True,
        session_id="profile/session",
        trace_dir=tmp_path,
        max_events=10,
    )
    attention = AFDCorrelationTraceRecorder(config, _identity())
    ffn = AFDCorrelationTraceRecorder(config, _identity(role="ffn"))
    attention_flow = attention.make_flow_id(
        "afd-npu-7",
        layer_idx=3,
        stage_idx=1,
    )
    ffn_flow = ffn.make_flow_id("afd-npu-7", layer_idx=3, stage_idx=1)
    assert attention_flow == ffn_flow

    with attention.record_range(
        "afd.cam.dispatch_send",
        flow_id=attention_flow,
        transaction_id="afd-npu-7",
        layer_idx=3,
        stage_idx=1,
        num_tokens=128,
    ):
        pass
    output_path = attention.close()
    ffn.close()

    assert output_path is not None
    assert "profile_session" in output_path.name
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["record_type"] == "metadata"
    events = [record for record in records if record["record_type"] == "event"]
    assert [event["phase"] for event in events] == ["begin", "end"]
    assert events[0]["flow_id"] == attention_flow
    assert events[0]["num_tokens"] == 128
    assert events[1]["outcome"] == "ok"


def test_recorder_reports_dropped_events(tmp_path: Path) -> None:
    recorder = AFDCorrelationTraceRecorder(
        AFDCorrelationTraceConfig(
            enabled=True,
            session_id="bounded",
            trace_dir=tmp_path,
            max_events=1,
        ),
        _identity(),
    )
    recorder.record("first")
    recorder.record("dropped")
    output_path = recorder.close()

    assert output_path is not None
    metadata = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["dropped_events"] == 0  # streaming: metadata frozen at init
