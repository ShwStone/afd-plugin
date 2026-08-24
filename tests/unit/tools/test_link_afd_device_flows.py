# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from tools.benchmarks.link_afd_device_flows import (
    DEVICE_FLOW_CATEGORY,
    build_device_flows,
)


def test_device_flow_endpoints_share_perfetto_identity() -> None:
    flow_id = "abcdef0123456789"
    events: list[dict[str, object]] = [
        {
            "name": "afd.cam.dispatch_send",
            "cat": "afd.correlation",
            "ph": "X",
            "pid": 1,
            "tid": 1,
            "ts": 10.0,
            "dur": 1.0,
            "args": {"flow_id": flow_id},
        },
        {
            "name": f"afd.cam.dispatch_send flow_id={flow_id}",
            "cat": "mstx",
            "ph": "X",
            "pid": 2,
            "tid": 2,
            "ts": 20.0,
            "dur": 1.0,
        },
        {
            "name": "CamMoeDistributeDispatchSend",
            "cat": "npu",
            "ph": "X",
            "pid": 3,
            "tid": 3,
            "ts": 30.0,
            "dur": 1.0,
        },
    ]

    flows, report = build_device_flows(events)

    assert report["linked_device_flows"] == 1
    assert [event["ph"] for event in flows] == ["s", "f"]
    assert len({(event["cat"], event["name"], event["id"]) for event in flows}) == 1
    assert flows[0]["pid"] == 1
    assert flows[1]["pid"] == 3
    assert all(event["cat"] == DEVICE_FLOW_CATEGORY for event in flows)
