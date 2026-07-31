# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from pathlib import Path

from tools.benchmarks.prefill_experiment import (
    build_benchmark_command,
    build_runs,
    load_experiment_config,
)

EXAMPLE_CONFIG_PATH = Path("tools/benchmarks/prefill_experiment.example.json")


def test_example_config_expands_full_primary_matrix() -> None:
    config = load_experiment_config(EXAMPLE_CONFIG_PATH)

    runs = build_runs(config)

    assert len(runs) == 2 * 5 * 5 * 3
    assert {run.system.name for run in runs} == {
        "dp4_tp8_sp",
        "afd_dp3_tp8_ep8",
    }
    assert {run.batch_tokens for run in runs} == {
        8192,
        16384,
        32768,
        49152,
        65536,
    }


def test_prefix_sensitivity_disables_overlapping_native_warmups() -> None:
    config = load_experiment_config(EXAMPLE_CONFIG_PATH)
    run = build_runs(
        config,
        system_names=["dp4_tp8_sp"],
        batch_tokens=[8192],
        prefix_ratio_key="0.25",
    )[0]

    command = build_benchmark_command(config, run)

    warmup_index = command.index("--num-warmups")
    assert command[warmup_index + 1] == "0"
    assert command[command.index("--temperature") + 1] == "0"
    assert "--skip-tokenizer-init" not in command
    assert command[-3:] == [
        "max_num_batched_tokens=8192",
        "repeat=1",
        "prefix_ratio=0.25",
    ]


def test_timeline_flag_is_opt_in() -> None:
    config = load_experiment_config(EXAMPLE_CONFIG_PATH)
    run = build_runs(
        config,
        system_names=["dp4_tp8_sp"],
        batch_tokens=[8192],
    )[0]

    assert "--plot-timeline" not in build_benchmark_command(config, run)
    assert "--plot-timeline" in build_benchmark_command(
        config,
        run,
        plot_timeline=True,
    )
