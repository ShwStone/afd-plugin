# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in compatibility patches for the pinned vLLM serving benchmark.

These patches are intentionally not registered as runtime AFD patches. They are
applied only by ``tools.benchmarks.prefill_bench`` and can be removed after the
same behavior is available upstream in vLLM.

Note: the incremental UTF-8 ``StreamedResponseHandler`` fix that this module
used to carry for vLLM 0.19.1 is upstream in vLLM 0.26.0, so only the
``CustomDataset`` patches remain.
"""

from __future__ import annotations

import json
import random
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike

TARGET_VLLM_VERSION = "0.26.0"


def _is_integer_token_prompt(prompt: object) -> bool:
    return isinstance(prompt, list) and all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in prompt
    )


# Upstream source: vllm/benchmarks/datasets/datasets.py (v0.26.0).
# Patch reason: vLLM 0.26.0 loads custom JSONL through a pandas DataFrame,
# temporarily duplicating a 109 MB exact-token dataset and coercing metadata,
# and shuffles through the process-global PRNG.
# Patch functionality: retain the upstream JSONL-only contract while loading
# records directly and using a local deterministic shuffler.
# Signature: matches vLLM 0.26.0; no parameters are added.
def custom_dataset_load_data(self) -> None:
    if self.dataset_path is None:
        raise ValueError("dataset_path must be provided for loading data.")

    # self.data will be a list of dictionaries
    # e.g., [{"prompt": "What is the capital of India?"}, ...]
    # This will be the standardized format which load_data()
    # has to convert into depending on the filetype of dataset_path.
    # sample() will assume this standardized format of self.data
    self.data = []

    # Load the JSONL file
    if self.dataset_path.endswith(".jsonl"):
        # ### PATCH START: stream JSONL without a pandas DataFrame copy
        with open(self.dataset_path, encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number}."
                    ) from error
                if not isinstance(item, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must contain an object."
                    )
                if "prompt" not in item:
                    raise ValueError("JSONL file must contain a 'prompt' column.")
                self.data.append(item)
        # ### PATCH END: stream JSONL without a pandas DataFrame copy
    else:
        raise NotImplementedError("Only JSONL format is supported for CustomDataset.")

    # ### PATCH START: avoid mutating the process-global PRNG
    if not self.disable_shuffle:
        random.Random(self.random_seed).shuffle(self.data)
    # ### PATCH END: avoid mutating the process-global PRNG


# Upstream source: vllm/benchmarks/datasets/datasets.py (v0.26.0).
# Patch reason: vLLM 0.26.0 CustomDataset.sample assumes custom prompts are text
# and reports prompt_len=1 whenever --skip-tokenizer-init is used.
# Patch functionality: preserve copied upstream text behavior while accepting
# exact integer token-ID prompts and their true lengths for prefill benchmarks.
# Signature: matches vLLM 0.26.0; no parameters are added.
def custom_dataset_sample(
    self,
    tokenizer: TokenizerLike,
    num_requests: int,
    request_id_prefix: str = "",
    no_oversample: bool = False,
    lora_path: str | None = None,
    max_loras: int | None = None,
    output_len: int | None = None,
    enable_multimodal_chat: bool = False,
    skip_chat_template: bool = False,
    chat_template_kwargs: dict | None = None,
    **kwargs,
) -> list:
    # Lazy import keeps the plugin importable on development machines without
    # the optional vLLM runtime dependency.
    import vllm.benchmarks.datasets as datasets_module

    # load all data if needed
    self.num_available_samples = len(self.data)
    if num_requests <= 0:
        num_requests = self.num_available_samples
        datasets_module.logger.info(
            "num_requests is set to 0 or negative, so using all available samples: %d",
            num_requests,
        )

    sampled_requests = []
    for i, item in enumerate(self.data):
        if len(sampled_requests) >= num_requests:
            break
        prompt = item["prompt"]

        # ### PATCH START: exact token-ID custom prompts
        is_integer_token_prompt = _is_integer_token_prompt(prompt)
        if isinstance(prompt, list) and not is_integer_token_prompt:
            raise ValueError(
                "Custom token-ID prompts must contain only integer token IDs."
            )
        if is_integer_token_prompt:
            if not prompt:
                raise ValueError("Custom token-ID prompts must not be empty.")
            if not skip_chat_template:
                raise ValueError(
                    "Integer token-ID custom datasets require --skip-chat-template."
                )
            new_output_len = output_len
            if output_len is None or output_len == -1:
                if "output_tokens" not in item:
                    raise ValueError(
                        "If no output length is provided the "
                        "custom dataset must contain an 'output_tokens' field."
                    )
                try:
                    new_output_len = int(item["output_tokens"])
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid value for 'output_tokens' in custom dataset: "
                        f"'{item['output_tokens']}'. Must be an integer."
                    ) from e
            if new_output_len is None or new_output_len <= 0:
                raise ValueError("Custom dataset output length must be positive.")

            prompt_len = len(prompt)
            declared_prompt_len = item.get("prompt_len")
            if (
                declared_prompt_len is not None
                and int(declared_prompt_len) != prompt_len
            ):
                raise ValueError(
                    "Custom dataset prompt_len does not match the token-ID prompt."
                )
            dataset_request_id = str(item.get("request_id", i))
        # ### PATCH END: exact token-ID custom prompts
        elif tokenizer is None:
            new_output_len = 1
        elif output_len is None or output_len == -1:
            # check if the request has an 'output_tokens' field
            if "output_tokens" not in item:
                raise ValueError(
                    "If no output length is provided the "
                    "custom dataset must contain an 'output_tokens' field."
                )
            # Use number of output tokens from the request data
            try:
                new_output_len = int(item["output_tokens"])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid value for 'output_tokens' in custom dataset: "
                    f"'{item['output_tokens']}'. Must be an integer."
                ) from e
        else:
            new_output_len = output_len

        # ### PATCH START: token prompts already have an exact prompt length
        if is_integer_token_prompt:
            pass
        # ### PATCH END: token prompts already have an exact prompt length
        elif tokenizer is None:
            prompt_len = 1
        else:
            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    **(chat_template_kwargs or {}),
                )

            prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests.append(
            datasets_module.SampleRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=new_output_len,
                # ### PATCH START: stable dataset request IDs
                request_id=request_id_prefix
                + (dataset_request_id if is_integer_token_prompt else str(i)),
                # ### PATCH END: stable dataset request IDs
            )
        )
    self.maybe_oversample_requests(
        sampled_requests, num_requests, request_id_prefix, no_oversample
    )

    return sampled_requests


def _installed_vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "The prefill benchmark requires the optional vLLM dependency."
        ) from error


def _base_vllm_version(installed_version: str) -> str:
    """Strip the PEP 440 local suffix (for example ``0.26.0+empty``)."""
    return installed_version.split("+", 1)[0]


def apply_prefill_benchmark_patches() -> None:
    """Apply the benchmark-only patches to the pinned vLLM process."""
    installed_version = _installed_vllm_version()
    if _base_vllm_version(installed_version) != TARGET_VLLM_VERSION:
        raise RuntimeError(
            "The prefill benchmark patches target vLLM "
            f"{TARGET_VLLM_VERSION}, but {installed_version} is installed."
        )

    import vllm.benchmarks.datasets as datasets_module

    if datasets_module.CustomDataset.load_data is not custom_dataset_load_data:
        datasets_module.CustomDataset.load_data = custom_dataset_load_data
    if datasets_module.CustomDataset.sample is not custom_dataset_sample:
        datasets_module.CustomDataset.sample = custom_dataset_sample
