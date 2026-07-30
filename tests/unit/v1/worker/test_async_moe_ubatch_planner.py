# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from afd_plugin.v1.worker.npu.async_moe_ubatch import plan_async_moe_stages


@pytest.mark.parametrize(
    (
        "scheduled_tokens",
        "num_tokens_padded",
        "use_sequence_parallel",
        "tensor_parallel_size",
        "expected_actual_tokens",
        "expected_physical_tokens",
    ),
    [
        ([8, 8], 16, False, 2, (8, 8), (8, 8)),
        ([1099], 1100, True, 2, (550, 549), (550, 550)),
        ([40], 112, True, 2, (20, 20), (20, 20)),
        ([70], 112, True, 2, (35, 35), (36, 36)),
        ([5], 8, True, 2, (3, 2), (4, 2)),
        ([1, 1, 100, 2], 104, False, 2, (52, 52), (52, 52)),
    ],
)
def test_token_stage_plan_balances_real_tokens_and_preserves_coverage(
    scheduled_tokens,
    num_tokens_padded,
    use_sequence_parallel,
    tensor_parallel_size,
    expected_actual_tokens,
    expected_physical_tokens,
):
    num_tokens = sum(scheduled_tokens)

    stage_plan = plan_async_moe_stages(
        scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=len(scheduled_tokens),
        num_stages=2,
        split="token",
        use_sequence_parallel=use_sequence_parallel,
        tensor_parallel_size=tensor_parallel_size,
    )

    assert stage_plan is not None
    stages, actual_token_counts = stage_plan
    assert actual_token_counts == expected_actual_tokens
    assert tuple(stage.num_tokens for stage in stages) == expected_physical_tokens
    assert stages[0].token_slice.start == 0
    assert stages[0].token_slice.stop == stages[1].token_slice.start
    assert stages[1].token_slice.stop == num_tokens
    assert sum(stage.actual_tokens for stage in stages) == num_tokens
    assert abs(stages[0].actual_tokens - stages[1].actual_tokens) <= 1
    if use_sequence_parallel:
        assert all(stage.num_tokens % tensor_parallel_size == 0 for stage in stages)


def test_token_stage_plan_rebuilds_request_ranges_when_split_inside_request():
    stage_plan = plan_async_moe_stages(
        [1, 1, 100, 2],
        num_tokens=104,
        num_tokens_padded=104,
        num_reqs_padded=4,
        num_stages=2,
        split="token",
        use_sequence_parallel=False,
        tensor_parallel_size=2,
    )

    assert stage_plan is not None
    stages, _ = stage_plan
    assert stages[0].request_slice == slice(0, 3)
    assert stages[0].token_slice == slice(0, 52)
    assert stages[1].request_slice == slice(2, 4)
    assert stages[1].token_slice == slice(52, 104)


def test_request_stage_plan_preserves_request_boundaries_for_pcp_policy():
    stage_plan = plan_async_moe_stages(
        [824, 846, 16],
        num_tokens=1686,
        num_tokens_padded=1686,
        num_reqs_padded=3,
        num_stages=2,
        split="request",
        use_sequence_parallel=False,
        tensor_parallel_size=1,
    )

    assert stage_plan is not None
    stages, actual_token_counts = stage_plan
    assert [stage.request_slice for stage in stages] == [
        slice(0, 1),
        slice(1, 3),
    ]
    assert [stage.token_slice for stage in stages] == [
        slice(0, 824),
        slice(824, 1686),
    ]
    assert actual_token_counts == (824, 862)


def test_request_stage_plan_aligns_each_flashcomm_stage_to_tp():
    stage_plan = plan_async_moe_stages(
        [5, 6, 7],
        num_tokens=18,
        num_tokens_padded=18,
        num_reqs_padded=3,
        num_stages=2,
        split="request",
        use_sequence_parallel=True,
        tensor_parallel_size=2,
    )

    assert stage_plan is not None
    stages, actual_token_counts = stage_plan
    assert [stage.request_slice for stage in stages] == [
        slice(0, 2),
        slice(2, 3),
    ]
    assert [stage.token_slice for stage in stages] == [
        slice(0, 11),
        slice(11, 18),
    ]
    assert actual_token_counts == (11, 7)
    assert tuple(stage.num_tokens for stage in stages) == (12, 8)


def test_request_stage_plan_needs_two_scheduled_requests_with_flashcomm():
    assert (
        plan_async_moe_stages(
            [18],
            num_tokens=18,
            num_tokens_padded=18,
            num_reqs_padded=1,
            num_stages=2,
            split="request",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )
        is None
    )


def test_token_stage_plan_rejects_invalid_inputs_and_handles_minimal_batch():
    with pytest.raises(ValueError, match="exactly two"):
        plan_async_moe_stages(
            [4, 4],
            num_tokens=8,
            num_tokens_padded=8,
            num_reqs_padded=2,
            num_stages=3,
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )

    with pytest.raises(ValueError, match="do not match"):
        plan_async_moe_stages(
            [4, 3],
            num_tokens=8,
            num_tokens_padded=8,
            num_reqs_padded=2,
            num_stages=2,
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )

    assert (
        plan_async_moe_stages(
            [2],
            num_tokens=2,
            num_tokens_padded=4,
            num_reqs_padded=1,
            num_stages=2,
            split="token",
            use_sequence_parallel=True,
            tensor_parallel_size=2,
        )
        is not None
    )
