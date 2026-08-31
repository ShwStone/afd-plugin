# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.engine import EngineCoreOutputs  # noqa: E402
from vllm.v1.engine.core_client import DPLBAsyncMPClient  # noqa: E402

from afd_plugin.compat.patches import async_dp_engine as async_dp_patch  # noqa: E402

pytestmark = pytest.mark.vllm_runtime


def _make_native_dplb_client(
    request_counts: tuple[tuple[int | None, ...], ...],
    *,
    policy: str = "request_count",
    client_count: int = 1,
) -> DPLBAsyncMPClient:
    """Build the smallest client state needed by the patched DPLB method."""

    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = client_count
    client.reqs_in_flight = {}
    client.core_engines = [
        index.to_bytes(2, "little") for index in range(len(request_counts))
    ]
    client.lb_engines = [list(counts) for counts in request_counts]
    client.eng_start_index = 0
    client.vllm_config = SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async": True,
                "attention_dplb_policy": policy,
            }
        }
    )
    return client


def _request(
    request_id: str,
    *,
    prompt_tokens: int = 16,
    max_tokens: int = 1,
    data_parallel_rank: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data_parallel_rank=data_parallel_rank,
        pooling_params=None,
        sampling_params=SimpleNamespace(
            max_tokens=max_tokens,
            n=1,
            structured_outputs=None,
        ),
        prompt_token_ids=list(range(prompt_tokens)),
        prompt_embeds=None,
        mm_features=None,
        lora_request=None,
        priority=0,
        resumable=False,
        abort_immediately=False,
    )


def test_native_dplb_uses_reported_request_counts():
    client = _make_native_dplb_client(((2, 0), (0, 1)))

    chosen_engine = client.get_core_engine_for_request(_request("loaded-route"))

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines == [[2, 0], [1, 1]]


def test_native_dplb_distributes_tied_burst_with_optimistic_counts():
    client = _make_native_dplb_client(((0, 0), (0, 0)))

    chosen_engines = [
        client.get_core_engine_for_request(_request(f"burst-{index}"))
        for index in range(4)
    ]

    assert chosen_engines == [
        client.core_engines[0],
        client.core_engines[1],
        client.core_engines[0],
        client.core_engines[1],
    ]
    assert client.lb_engines == [[2, 0], [2, 0]]


def test_native_dplb_releases_completed_request():
    client = _make_native_dplb_client(((0, 0), (0, 0)))
    request = _request("finished")
    client.get_core_engine_for_request(request)

    outputs = EngineCoreOutputs(finished_requests={request.request_id})
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, outputs))

    assert request.request_id not in client.reqs_in_flight


def test_prefill_token_dplb_prefers_lower_debt_over_request_count():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((0, 0, version, 1000), (2, 0, version, 100)),
        policy="prefill_token_sum",
    )

    chosen_engine = client.get_core_engine_for_request(
        _request("token-route", prompt_tokens=64)
    )

    assert chosen_engine == client.core_engines[1]
    assert client.lb_engines == [
        [0, 0, version, 1000],
        [3, 0, version, 164],
    ]


def test_prefill_token_dplb_uses_optimistic_client_scaled_debt():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((0, 0, version, 0), (0, 0, version, 0)),
        policy="prefill_token_sum",
        client_count=2,
    )

    first = client.get_core_engine_for_request(
        _request("token-burst-0", prompt_tokens=100)
    )
    second = client.get_core_engine_for_request(
        _request("token-burst-1", prompt_tokens=10)
    )

    assert [first, second] == [client.core_engines[0], client.core_engines[1]]
    assert client.lb_engines == [
        [2, 0, version, 200],
        [2, 0, version, 20],
    ]


def test_prefill_token_dplb_falls_back_when_any_debt_is_unavailable():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((0, 1, version, 100), (1, 0, version, None)),
        policy="prefill_token_sum",
    )

    chosen_engine = client.get_core_engine_for_request(_request("fallback"))

    assert chosen_engine == client.core_engines[0]
    assert client.lb_engines[0][:2] == [1, 1]


def test_ineligible_request_invalidates_token_debt_and_uses_native_counts():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((2, 0, version, 0), (0, 1, version, 1000)),
        policy="prefill_token_sum",
    )

    chosen_engine = client.get_core_engine_for_request(_request("decode", max_tokens=2))

    assert chosen_engine == client.core_engines[1]
    assert [counts[3] for counts in client.lb_engines] == [-1, -1]


def test_explicit_rank_invalidates_local_token_snapshot():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((0, 0, version, 0), (0, 0, version, 0)),
        policy="prefill_token_sum",
    )

    chosen_engine = client.get_core_engine_for_request(
        _request("explicit", data_parallel_rank=1)
    )

    assert chosen_engine == client.core_engines[1]
    assert [counts[3] for counts in client.lb_engines] == [-1, -1]


def test_mixed_request_blocks_restored_snapshots_until_completion():
    version = async_dp_patch.AFD_DPLB_STATS_VERSION
    client = _make_native_dplb_client(
        ((0, 0, version, 0), (2, 0, version, 0)),
        policy="prefill_token_sum",
    )
    mixed_request = _request("mixed", max_tokens=2)
    client.get_core_engine_for_request(mixed_request)

    # Simulate a coordinator snapshot produced before the mixed request reached
    # the engine. Local lifecycle state must continue forcing count fallback.
    client.lb_engines = [
        [0, 0, version, 1000],
        [1, 0, version, 0],
    ]
    while_mixed = client.get_core_engine_for_request(_request("eligible-during"))
    assert while_mixed == client.core_engines[0]

    outputs = EngineCoreOutputs(finished_requests={mixed_request.request_id})
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(client, outputs))
    assert [counts[3] for counts in client.lb_engines] == [-1, -1]

    client.lb_engines = [
        [0, 0, version, 1000],
        [1, 0, version, 0],
    ]
    after_fresh_snapshot = client.get_core_engine_for_request(
        _request("eligible-after")
    )
    assert after_fresh_snapshot == client.core_engines[1]
