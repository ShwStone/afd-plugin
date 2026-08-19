# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run vLLM bench serve with AFD's exact-token prefill benchmark patches."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from afd_plugin.compat.patches.benchmark_serving import (
    apply_prefill_benchmark_patches,
)

REQUIRED_FLAGS = (
    ("--dataset-name", "custom"),
    ("--custom-output-len", "1"),
)
REQUIRED_SWITCHES = (
    "--skip-tokenizer-init",
    "--skip-chat-template",
    "--disable-shuffle",
    "--no-oversample",
    "--save-detailed",
)
ADD_SPECIAL_TOKENS_FIELD = "add_special_tokens"


def _option_value(arguments: Sequence[str], option_name: str) -> str | None:
    for argument_index, argument in enumerate(arguments):
        if argument == option_name:
            if argument_index + 1 >= len(arguments):
                raise ValueError(f"{option_name} requires a value.")
            return arguments[argument_index + 1]
        if argument.startswith(option_name + "="):
            return argument.partition("=")[2]
    return None


def _set_option_value(
    arguments: list[str],
    option_name: str,
    option_value: str,
) -> None:
    for argument_index, argument in enumerate(arguments):
        if argument == option_name:
            arguments[argument_index + 1] = option_value
            return
        if argument.startswith(option_name + "="):
            arguments[argument_index] = f"{option_name}={option_value}"
            return
    arguments.extend((option_name, option_value))


def build_vllm_arguments(arguments: Sequence[str]) -> list[str]:
    """Add invariant prefill-only flags and reject conflicting values."""
    vllm_arguments = list(arguments)
    for option_name, required_value in REQUIRED_FLAGS:
        configured_value = _option_value(vllm_arguments, option_name)
        if configured_value is None:
            vllm_arguments.extend((option_name, required_value))
        elif configured_value != required_value:
            raise ValueError(
                f"{option_name} must be {required_value!r}, got {configured_value!r}."
            )
    for switch_name in REQUIRED_SWITCHES:
        if switch_name not in vllm_arguments:
            vllm_arguments.append(switch_name)
    if _option_value(vllm_arguments, "--dataset-path") is None:
        raise ValueError("--dataset-path is required.")

    raw_extra_body = _option_value(vllm_arguments, "--extra-body")
    if raw_extra_body is None:
        extra_body: dict[str, object] = {}
    else:
        parsed_extra_body = json.loads(raw_extra_body)
        if not isinstance(parsed_extra_body, dict):
            raise ValueError("--extra-body must contain a JSON object.")
        extra_body = parsed_extra_body
    if extra_body.get(ADD_SPECIAL_TOKENS_FIELD) not in (None, False):
        raise ValueError("Exact token-ID prompts require add_special_tokens=false.")
    extra_body[ADD_SPECIAL_TOKENS_FIELD] = False
    _set_option_value(
        vllm_arguments,
        "--extra-body",
        json.dumps(extra_body, separators=(",", ":")),
    )
    return ["bench", "serve", *vllm_arguments]


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the patches, then delegate to the pinned vLLM CLI."""
    benchmark_arguments = sys.argv[1:] if argv is None else list(argv)
    vllm_arguments = build_vllm_arguments(benchmark_arguments)
    apply_prefill_benchmark_patches()

    from vllm.entrypoints.cli.main import main as vllm_main

    original_arguments = sys.argv
    try:
        sys.argv = [original_arguments[0], *vllm_arguments]
        vllm_main()
    finally:
        sys.argv = original_arguments
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
