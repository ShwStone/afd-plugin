# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.model_executor.npu.async_cam_tensor_dump import (  # noqa: E402
    ATTENTION_DISPATCH_HIDDEN,
    FFN_ROUTED_INPUT,
    AsyncCamTensorDumpConfig,
    dump_async_cam_tensor,
)
from tools.compare_async_cam_tensor_dumps import (  # noqa: E402
    _compare_dump_roots,
)


def test_async_cam_tensor_dump_is_disabled_by_default():
    config = AsyncCamTensorDumpConfig.from_env({})

    assert not config.enabled
    assert config.output_dir is None


@pytest.mark.parametrize(
    ("environ", "error_match"),
    [
        (
            {"AFD_ASYNC_MOE_PRECISION_DEBUG": "1"},
            "AFD_ASYNC_MOE_PRECISION_DEBUG_DIR",
        ),
        (
            {
                "AFD_ASYNC_MOE_PRECISION_DEBUG": "1",
                "AFD_ASYNC_MOE_PRECISION_DEBUG_DIR": "/tmp/dumps",
            },
            "AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS",
        ),
        (
            {
                "AFD_ASYNC_MOE_PRECISION_DEBUG": "1",
                "AFD_ASYNC_MOE_PRECISION_DEBUG_DIR": "/tmp/dumps",
                "AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS": "3",
                "AFD_ASYNC_MOE_PRECISION_DEBUG_POINTS": "not-a-point",
            },
            "unsupported points",
        ),
    ],
)
def test_async_cam_tensor_dump_rejects_incomplete_configuration(
    environ,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        AsyncCamTensorDumpConfig.from_env(environ)


def test_async_cam_tensor_dump_selects_global_token_rows(tmp_path: Path):
    config = AsyncCamTensorDumpConfig(
        enabled=True,
        output_dir=tmp_path,
        layers=frozenset({3}),
        points=frozenset({ATTENTION_DISPATCH_HIDDEN}),
        token_indices=(99, 100, 103, 107, 108),
    )
    tensor = torch.arange(40, dtype=torch.float32).reshape(10, 4)

    output_path = dump_async_cam_tensor(
        tensor,
        config,
        role="attention",
        role_rank=1,
        layer_idx=3,
        stage_idx=0,
        point=ATTENTION_DISPATCH_HIDDEN,
        row_coordinate="global_token",
        row_start=100,
        valid_rows=8,
        transaction_id="request-0",
    )

    assert output_path == (
        tmp_path
        / "attention-rank-001"
        / "layer-003"
        / "stage-00-attn_dispatch_hidden.pt"
    )
    payload = torch.load(output_path)
    assert payload["selected_row_indices"] == (100, 103, 107)
    assert torch.equal(payload["sampled_tensor"], tensor[[0, 3, 7]])
    assert payload["row_coordinate"] == "global_token"
    assert payload["valid_rows"] == 8
    assert payload["transaction_id"] == "request-0"
    assert len(payload["sample_sha256"]) == 64


def test_async_cam_tensor_dump_uses_edges_for_rank_local_rows(tmp_path: Path):
    config = AsyncCamTensorDumpConfig(
        enabled=True,
        output_dir=tmp_path,
        layers=frozenset({4}),
        points=frozenset({FFN_ROUTED_INPUT}),
        token_indices=(100, 101),
        edge_rows=2,
    )
    tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    output_path = dump_async_cam_tensor(
        tensor,
        config,
        role="ffn",
        role_rank=0,
        layer_idx=4,
        stage_idx=1,
        point=FFN_ROUTED_INPUT,
        row_coordinate="rank_local",
        valid_rows=5,
    )

    payload = torch.load(output_path)
    assert payload["selected_row_indices"] == (0, 1, 3, 4)
    assert torch.equal(payload["sampled_tensor"], tensor[[0, 1, 3, 4]])


def test_async_cam_tensor_dump_keeps_first_observation(tmp_path: Path):
    config = AsyncCamTensorDumpConfig(
        enabled=True,
        output_dir=tmp_path,
        layers=frozenset({5}),
        points=frozenset({ATTENTION_DISPATCH_HIDDEN}),
        full_tensors=True,
    )
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    second = first + 100

    output_path = dump_async_cam_tensor(
        first,
        config,
        role="attention",
        role_rank=0,
        layer_idx=5,
        stage_idx=0,
        point=ATTENTION_DISPATCH_HIDDEN,
        row_coordinate="global_token",
        valid_rows=2,
    )
    duplicate_path = dump_async_cam_tensor(
        second,
        config,
        role="attention",
        role_rank=0,
        layer_idx=5,
        stage_idx=0,
        point=ATTENTION_DISPATCH_HIDDEN,
        row_coordinate="global_token",
        valid_rows=2,
    )

    assert duplicate_path is None
    payload = torch.load(output_path)
    assert payload["selected_row_indices"] == (0, 1)
    assert torch.equal(payload["sampled_tensor"], first[:2])


def test_async_cam_tensor_comparison_joins_tokens_across_stages_and_ranks(
    tmp_path: Path,
    capsys,
):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    tensor_rows = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    for output_dir, chunks in (
        (
            baseline_dir,
            ((0, 0, 0, 2), (1, 0, 2, 4)),
        ),
        (
            candidate_dir,
            ((0, 0, 0, 1), (1, 0, 1, 2), (0, 1, 2, 3), (1, 1, 3, 4)),
        ),
    ):
        config = AsyncCamTensorDumpConfig(
            enabled=True,
            output_dir=output_dir,
            layers=frozenset({5}),
            points=frozenset({ATTENTION_DISPATCH_HIDDEN}),
            full_tensors=True,
        )
        for role_rank, stage_idx, row_start, row_stop in chunks:
            dump_async_cam_tensor(
                tensor_rows[row_start:row_stop],
                config,
                role="attention",
                role_rank=role_rank,
                layer_idx=5,
                stage_idx=stage_idx,
                point=ATTENTION_DISPATCH_HIDDEN,
                row_coordinate="global_token",
                row_start=row_start,
            )

    assert _compare_dump_roots(baseline_dir, candidate_dir) == 0
    comparison = capsys.readouterr().out
    assert "common=4" in comparison
    assert "max_abs=0" in comparison
    assert "exact=True" in comparison
