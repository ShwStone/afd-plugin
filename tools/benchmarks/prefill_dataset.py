# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build and validate exact token-ID datasets for prefill-only benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_RANDOM_SEED = 20260731
DEFAULT_OUTPUT_TOKENS = 1
DEFAULT_PREFIX_BLOCK_SIZE = 128
DEFAULT_PREFIX_GROUP_SIZE = 12
MINIMUM_TOKEN_ID = 0
DATASET_SCHEMA_VERSION = 1
HASH_READ_CHUNK_BYTES = 1024 * 1024
SEED_DIGEST_BYTES = 8
MANIFEST_SUFFIX = ".manifest.json"
INDEX_SUFFIX = ".index.jsonl"
TEMPORARY_SUFFIX = ".tmp"
SPECIAL_TOKEN_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "unk_token_id",
    "sep_token_id",
    "cls_token_id",
    "mask_token_id",
)
SOURCE_LENGTH_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"


@dataclass(frozen=True)
class TokenIdSpace:
    """Legal token-ID range derived from a model configuration."""

    vocab_size: int
    excluded_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SourceRequest:
    """One non-zero request length from the source CSV."""

    source_row: int
    prompt_length: int


def read_source_requests(csv_path: Path) -> list[SourceRequest]:
    """Read one positive integer input length from each non-zero CSV row."""
    source_requests: list[SourceRequest] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for source_row, row in enumerate(csv.reader(csv_file), start=1):
            nonempty_values = [value.strip() for value in row if value.strip()]
            if not nonempty_values:
                continue

            try:
                numeric_values = [int(value) for value in nonempty_values]
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}:{source_row} contains a non-integer value."
                ) from error

            nonzero_values = [value for value in numeric_values if value != 0]
            if not nonzero_values:
                continue
            if len(nonzero_values) != 1:
                raise ValueError(
                    f"{csv_path}:{source_row} must contain exactly one non-zero value."
                )
            prompt_length = nonzero_values[0]
            if prompt_length < 0:
                raise ValueError(
                    f"{csv_path}:{source_row} contains a negative input length."
                )
            source_requests.append(
                SourceRequest(
                    source_row=source_row,
                    prompt_length=prompt_length,
                )
            )

    if not source_requests:
        raise ValueError(f"{csv_path} does not contain any positive input lengths.")
    return source_requests


def _collect_token_ids(value: object) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, list):
        token_ids: set[int] = set()
        for item in value:
            token_ids.update(_collect_token_ids(item))
        return token_ids
    return set()


def load_token_id_space(
    model_config_path: Path,
    additional_excluded_token_ids: Sequence[int] = (),
) -> TokenIdSpace:
    """Load vocabulary size and declared special token IDs from config.json."""
    config_path = (
        model_config_path / "config.json"
        if model_config_path.is_dir()
        else model_config_path
    )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")

    vocab_size = raw_config.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise ValueError(f"{config_path} must define an integer vocab_size.")
    if vocab_size <= MINIMUM_TOKEN_ID:
        raise ValueError(f"{config_path} defines an invalid vocab_size.")

    tokenizer_config_path = config_path.parent / TOKENIZER_CONFIG_FILENAME
    if tokenizer_config_path.is_file():
        raw_tokenizer_config = json.loads(
            tokenizer_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(raw_tokenizer_config, dict):
            raise ValueError(f"{tokenizer_config_path} must contain a JSON object.")
    else:
        raw_tokenizer_config = {}

    excluded_token_ids = set(additional_excluded_token_ids)
    for configuration in (raw_config, raw_tokenizer_config):
        for field_name in SPECIAL_TOKEN_FIELDS:
            excluded_token_ids.update(_collect_token_ids(configuration.get(field_name)))
    added_tokens_decoder = raw_tokenizer_config.get("added_tokens_decoder")
    if isinstance(added_tokens_decoder, dict):
        for token_id, token_definition in added_tokens_decoder.items():
            if (
                isinstance(token_definition, dict)
                and token_definition.get("special") is True
            ):
                try:
                    excluded_token_ids.add(int(token_id))
                except ValueError as error:
                    raise ValueError(
                        f"{tokenizer_config_path} contains an invalid added token ID."
                    ) from error
    excluded_token_ids = {
        token_id
        for token_id in excluded_token_ids
        if MINIMUM_TOKEN_ID <= token_id < vocab_size
    }
    if len(excluded_token_ids) >= vocab_size:
        raise ValueError("All token IDs are excluded.")
    return TokenIdSpace(
        vocab_size=vocab_size,
        excluded_token_ids=tuple(sorted(excluded_token_ids)),
    )


def _derive_seed(base_seed: int, namespace: str) -> int:
    seed_material = f"{base_seed}:{namespace}".encode()
    return int.from_bytes(
        hashlib.sha256(seed_material).digest()[:SEED_DIGEST_BYTES],
        "big",
    )


def _generate_token_ids(
    token_count: int,
    token_id_space: TokenIdSpace,
    seed: int,
) -> list[int]:
    random_generator = random.Random(seed)
    excluded_token_ids = set(token_id_space.excluded_token_ids)
    token_ids: list[int] = []
    while len(token_ids) < token_count:
        token_id = random_generator.randrange(
            MINIMUM_TOKEN_ID,
            token_id_space.vocab_size,
        )
        if token_id not in excluded_token_ids:
            token_ids.append(token_id)
    return token_ids


def _aligned_prefix_length(
    prompt_length: int,
    prefix_ratio: float,
    prefix_block_size: int,
) -> int:
    unaligned_length = math.floor(prompt_length * prefix_ratio)
    maximum_prefix_length = max(prompt_length - 1, 0)
    aligned_length = unaligned_length // prefix_block_size * prefix_block_size
    return min(
        aligned_length,
        maximum_prefix_length // prefix_block_size * prefix_block_size,
    )


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: Sequence[int], percentile: int) -> float:
    rank = percentile / 100 * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _iter_dataset_records(
    source_requests: Sequence[SourceRequest],
    token_id_space: TokenIdSpace,
    random_seed: int,
    prefix_ratio: float,
    prefix_block_size: int,
    prefix_group_size: int,
) -> Iterator[dict[str, object]]:
    group_prefixes: dict[int, list[int]] = {}
    if prefix_ratio > 0:
        for group_start in range(0, len(source_requests), prefix_group_size):
            group_index = group_start // prefix_group_size
            group = source_requests[group_start : group_start + prefix_group_size]
            maximum_prefix_length = max(
                _aligned_prefix_length(
                    request.prompt_length,
                    prefix_ratio,
                    prefix_block_size,
                )
                for request in group
            )
            group_prefixes[group_index] = _generate_token_ids(
                maximum_prefix_length,
                token_id_space,
                _derive_seed(random_seed, f"prefix-group:{group_index}"),
            )

    for request_index, source_request in enumerate(source_requests):
        group_index = request_index // prefix_group_size
        prefix_length = _aligned_prefix_length(
            source_request.prompt_length,
            prefix_ratio,
            prefix_block_size,
        )
        prefix_token_ids = group_prefixes.get(group_index, [])[:prefix_length]
        suffix_length = source_request.prompt_length - prefix_length
        suffix_token_ids = _generate_token_ids(
            suffix_length,
            token_id_space,
            _derive_seed(
                random_seed,
                f"request:{request_index}:row:{source_request.source_row}",
            ),
        )
        request_id = f"cp8sp50k-{request_index + 1:06d}"
        yield {
            "request_id": request_id,
            "source_row": source_request.source_row,
            "prompt": prefix_token_ids + suffix_token_ids,
            "prompt_len": source_request.prompt_length,
            "output_tokens": DEFAULT_OUTPUT_TOKENS,
            "prefix_group": group_index if prefix_ratio > 0 else None,
            "shared_prefix_len": prefix_length,
        }


def generate_dataset(
    csv_path: Path,
    model_config_path: Path,
    output_path: Path,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    prefix_ratio: float = 0.0,
    prefix_block_size: int = DEFAULT_PREFIX_BLOCK_SIZE,
    prefix_group_size: int = DEFAULT_PREFIX_GROUP_SIZE,
    additional_excluded_token_ids: Sequence[int] = (),
) -> dict[str, object]:
    """Generate a deterministic custom JSONL dataset and its manifest."""
    if not 0.0 <= prefix_ratio < 1.0:
        raise ValueError("prefix_ratio must be in the range [0, 1).")
    if prefix_block_size <= 0:
        raise ValueError("prefix_block_size must be positive.")
    if prefix_group_size <= 0:
        raise ValueError("prefix_group_size must be positive.")

    source_requests = read_source_requests(csv_path)
    token_id_space = load_token_id_space(
        model_config_path,
        additional_excluded_token_ids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + TEMPORARY_SUFFIX)
    index_path = output_path.with_name(output_path.name + INDEX_SUFFIX)
    temporary_index_path = index_path.with_name(index_path.name + TEMPORARY_SUFFIX)
    total_prompt_tokens = 0
    total_shared_prefix_tokens = 0
    total_sequentially_reusable_prefix_tokens = 0
    maximum_seen_prefix_by_group: dict[int, int] = {}

    with (
        temporary_path.open("w", encoding="utf-8") as output_file,
        temporary_index_path.open("w", encoding="utf-8") as index_file,
    ):
        for record in _iter_dataset_records(
            source_requests,
            token_id_space,
            random_seed,
            prefix_ratio,
            prefix_block_size,
            prefix_group_size,
        ):
            total_prompt_tokens += int(record["prompt_len"])
            shared_prefix_length = int(record["shared_prefix_len"])
            total_shared_prefix_tokens += shared_prefix_length
            prefix_group = record["prefix_group"]
            if isinstance(prefix_group, int):
                maximum_seen_prefix = maximum_seen_prefix_by_group.get(prefix_group, 0)
                total_sequentially_reusable_prefix_tokens += min(
                    shared_prefix_length,
                    maximum_seen_prefix,
                )
                maximum_seen_prefix_by_group[prefix_group] = max(
                    maximum_seen_prefix,
                    shared_prefix_length,
                )
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            index_file.write(
                json.dumps(
                    {
                        "request_id": record["request_id"],
                        "source_row": record["source_row"],
                        "prompt_len": record["prompt_len"],
                        "prefix_group": prefix_group,
                        "shared_prefix_len": shared_prefix_length,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary_path, output_path)
    os.replace(temporary_index_path, index_path)
    sorted_prompt_lengths = sorted(request.prompt_length for request in source_requests)
    resolved_model_config_path = (
        model_config_path / "config.json"
        if model_config_path.is_dir()
        else model_config_path
    )
    tokenizer_config_path = (
        resolved_model_config_path.parent / TOKENIZER_CONFIG_FILENAME
    )

    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_path": str(output_path),
        "dataset_sha256": _sha256_file(output_path),
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "source_csv": str(csv_path),
        "source_csv_sha256": _sha256_file(csv_path),
        "model_config": str(resolved_model_config_path),
        "model_config_sha256": _sha256_file(resolved_model_config_path),
        "tokenizer_config": (
            str(tokenizer_config_path) if tokenizer_config_path.is_file() else None
        ),
        "tokenizer_config_sha256": (
            _sha256_file(tokenizer_config_path)
            if tokenizer_config_path.is_file()
            else None
        ),
        "request_count": len(source_requests),
        "total_prompt_tokens": total_prompt_tokens,
        "minimum_prompt_tokens": min(
            request.prompt_length for request in source_requests
        ),
        "maximum_prompt_tokens": max(
            request.prompt_length for request in source_requests
        ),
        "prompt_length_percentiles": {
            f"p{percentile}": _percentile(
                sorted_prompt_lengths,
                percentile,
            )
            for percentile in SOURCE_LENGTH_PERCENTILES
        },
        "output_tokens_per_request": DEFAULT_OUTPUT_TOKENS,
        "random_seed": random_seed,
        "token_id_space": asdict(token_id_space),
        "requested_prefix_ratio": prefix_ratio,
        "actual_prefix_token_ratio": (total_shared_prefix_tokens / total_prompt_tokens),
        "estimated_sequential_reusable_prefix_token_ratio": (
            total_sequentially_reusable_prefix_tokens / total_prompt_tokens
        ),
        "prefix_block_size": prefix_block_size,
        "prefix_group_size": prefix_group_size,
    }
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_dataset(
    dataset_path: Path,
    *,
    csv_path: Path | None = None,
    model_config_path: Path | None = None,
) -> dict[str, int | str]:
    """Validate JSONL structure, source lengths, and legal token IDs."""
    expected_requests = read_source_requests(csv_path) if csv_path else None
    token_id_space = (
        load_token_id_space(model_config_path) if model_config_path else None
    )
    excluded_token_ids = (
        set(token_id_space.excluded_token_ids) if token_id_space else set()
    )
    request_count = 0
    total_prompt_tokens = 0

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{dataset_path}:{line_number} is not valid JSON."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{dataset_path}:{line_number} must contain a JSON object."
                )
            prompt = record.get("prompt")
            if not isinstance(prompt, list) or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in prompt
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number} prompt must be a list of integers."
                )
            if not prompt:
                raise ValueError(
                    f"{dataset_path}:{line_number} prompt must not be empty."
                )
            prompt_length = record.get("prompt_len")
            if prompt_length != len(prompt):
                raise ValueError(
                    f"{dataset_path}:{line_number} prompt_len does not match prompt."
                )
            if record.get("output_tokens") != DEFAULT_OUTPUT_TOKENS:
                raise ValueError(
                    f"{dataset_path}:{line_number} must request one output token."
                )
            if expected_requests:
                if request_count >= len(expected_requests):
                    raise ValueError("Dataset has more requests than the source CSV.")
                expected_request = expected_requests[request_count]
                if (
                    record.get("source_row") != expected_request.source_row
                    or prompt_length != expected_request.prompt_length
                ):
                    raise ValueError(
                        f"{dataset_path}:{line_number} does not match the source CSV."
                    )
            if token_id_space and any(
                token_id < MINIMUM_TOKEN_ID
                or token_id >= token_id_space.vocab_size
                or token_id in excluded_token_ids
                for token_id in prompt
            ):
                raise ValueError(
                    f"{dataset_path}:{line_number} contains an illegal token ID."
                )
            request_count += 1
            total_prompt_tokens += len(prompt)

    if expected_requests and request_count != len(expected_requests):
        raise ValueError("Dataset has fewer requests than the source CSV.")
    dataset_sha256 = _sha256_file(dataset_path)
    manifest_path = dataset_path.with_name(dataset_path.name + MANIFEST_SUFFIX)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path} must contain a JSON object.")
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError("Dataset manifest schema version is unsupported.")
        if manifest.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Dataset SHA256 does not match its manifest.")
        index_path = dataset_path.with_name(dataset_path.name + INDEX_SUFFIX)
        if not index_path.is_file():
            raise ValueError("Dataset index declared by the manifest is missing.")
        if manifest.get("index_sha256") != _sha256_file(index_path):
            raise ValueError("Dataset index SHA256 does not match its manifest.")
    return {
        "dataset_sha256": dataset_sha256,
        "request_count": request_count,
        "total_prompt_tokens": total_prompt_tokens,
    }


def _parse_excluded_token_ids(raw_value: str) -> tuple[int, ...]:
    if not raw_value:
        return ()
    return tuple(int(value.strip()) for value in raw_value.split(","))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--csv", type=Path, required=True)
    generate_parser.add_argument("--model-config", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    generate_parser.add_argument("--prefix-ratio", type=float, default=0.0)
    generate_parser.add_argument(
        "--prefix-block-size",
        type=int,
        default=DEFAULT_PREFIX_BLOCK_SIZE,
    )
    generate_parser.add_argument(
        "--prefix-group-size",
        type=int,
        default=DEFAULT_PREFIX_GROUP_SIZE,
    )
    generate_parser.add_argument("--exclude-token-ids", default="")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--csv", type=Path)
    validate_parser.add_argument("--model-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_dataset(
            args.csv,
            args.model_config,
            args.output,
            random_seed=args.seed,
            prefix_ratio=args.prefix_ratio,
            prefix_block_size=args.prefix_block_size,
            prefix_group_size=args.prefix_group_size,
            additional_excluded_token_ids=_parse_excluded_token_ids(
                args.exclude_token_ids
            ),
        )
    else:
        result = validate_dataset(
            args.dataset,
            csv_path=args.csv,
            model_config_path=args.model_config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
