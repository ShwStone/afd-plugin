# DeepSeek V3.2 prefill performance experiment

This document defines the reproducible two-phase comparison of traditional
DP+TP+SP serving and AFD CAM async serving on a reduced-layer DeepSeek V3.2
model. The primary client is the pinned vLLM 0.19.1 `vllm bench serve`
implementation. No result is considered an NPU result until it was produced on
the target server and passed the checks below.

## Fixed primary comparison

Keep these choices fixed for the main matrix:

- workload: prefill only, exactly one requested output token;
- hardware budget: 32 NPUs;
- baseline: `DP4 x TP8`, SP enabled;
- AFD: Attention `DP3 x TP8`, SP enabled, plus FFN `EP8`;
- maximum batched tokens: 8K, 16K, 32K, 48K, and 64K;
- offered load: RPS 4, 6, 8, 10, and 12 with Poisson arrivals;
- repeats: three independent runs per cell;
- TTFT SLO: 10 seconds;
- primary prefix cache setting: disabled;
- request order: the 875 non-zero rows of `cp8sp50k.csv`, without shuffle or
  oversampling.
- generation: greedy (`temperature=0`), ignore EOS, exactly one output token.

This produces 150 primary result files: two systems, five batch-token limits,
five request rates, and three repeats. Server and client runs must not share a
profiler session; profiling is an independent replay.

## Phase 1: build and verify locally

### Dataset facts

`cp8sp50k.csv` currently has 1,380 rows. Its 875 non-zero rows describe
18,184,995 input tokens:

| Statistic | Tokens |
| --- | ---: |
| minimum | 71 |
| p10 | 4,783 |
| p25 | 8,606.5 |
| median | 16,936 |
| p75 | 31,586.5 |
| p90 | 45,570 |
| p95 | 49,384 |
| p99 | 50,711.12 |
| maximum | 50,773 |

Percentiles use linear interpolation over the sorted 875 lengths and are
recorded in every generated manifest.

The generated JSONL contains legal token IDs, not decoded text. This avoids
tokenizer round trips changing the intended length. Generation reads
`vocab_size` and declared special IDs from the reduced model's `config.json`
and, when present, special added-token IDs from `tokenizer_config.json`.
Every request uses a request-specific deterministic PRNG stream, retains its
one-based CSV source row, and asks for one output token.

Generate the primary dataset:

```bash
python -m tools.benchmarks.prefill_dataset generate \
  --csv cp8sp50k.csv \
  --model-config /models/DeepSeek-V3.2-reduced \
  --output tools/datasets/cp8sp50k_token_ids.jsonl
```

The command also writes
`cp8sp50k_token_ids.jsonl.manifest.json`, including input hashes, output hash,
seed, token-ID exclusions, length totals, and prefix parameters. A compact
`.index.jsonl` sidecar stores only request ID, source row, length, and prefix
metadata, so 150 result postprocessors do not repeatedly parse the 109 MB
prompt file. Validate the artifact independently:

```bash
python -m tools.benchmarks.prefill_dataset validate \
  --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
  --csv cp8sp50k.csv \
  --model-config /models/DeepSeek-V3.2-reduced
```

The existing
`tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl` may be a
Git LFS pointer in a fresh checkout. The preflight and matrix tools reject
pointer files rather than benchmarking the pointer text.

### Prefix-cache sensitivity datasets

Prefix cache is off in the main comparison. Generate separate, block-aligned
datasets for the sensitivity study:

```bash
for specification in 0.25:25 0.5:50 0.75:75; do
  ratio=${specification%%:*}
  suffix=${specification##*:}
  python -m tools.benchmarks.prefill_dataset generate \
    --csv cp8sp50k.csv \
    --model-config /models/DeepSeek-V3.2-reduced \
    --output "tools/datasets/cp8sp50k_token_ids_prefix${suffix}.jsonl" \
    --prefix-ratio "${ratio}" \
    --prefix-block-size 128 \
    --prefix-group-size 12
done
```

Requests are grouped in twelves so a group is neutral to both DP3 and DP4.
Members share a block-aligned prefix and retain request-unique suffixes. The
manifest records the actual aggregate shared-token ratio, which can differ
slightly from the requested ratio due to block alignment. It also reports an
estimated sequentially reusable ratio that excludes the first unseen prefix in
each group. Actual server hits can be lower under concurrent arrival, DP-local
cache placement, eviction, or chunk scheduling, so collect server cache-hit
counters instead of treating either manifest ratio as the observed hit rate.

Do not use vLLM's built-in warmup on these prefix datasets: it replays the same
first requests and can turn intended partial hits into full-request cache hits.
The matrix tool therefore sets built-in warmups to zero for non-zero prefix
ratios. Use a separately generated warmup dataset or restart and precondition
both systems identically when measuring cache sensitivity.

### `bench serve` capability audit and patches

The audit was performed against the exact vLLM 0.19.1 source pinned by this
repository.

| Requirement | Native vLLM 0.19.1 | Repository behavior |
| --- | --- | --- |
| finite RPS and Poisson/gamma arrivals | supported by `--request-rate` and `--burstiness` | used directly |
| warmup, no shuffle, no oversample | supported | forced or configured |
| string custom JSONL | supported | unchanged |
| large custom JSONL loading | pandas DataFrame temporarily duplicates/coerces the records | opt-in line-by-line JSON loader avoids the DataFrame copy |
| exact integer token-ID custom prompts | request payload accepts IDs, but `CustomDataset.sample` assumes text and reports length 1 with `--skip-tokenizer-init` | opt-in patch preserves IDs, validates declared length, and reports the exact length |
| exact server prompt | completions default permits special-token insertion for text | launcher merges `add_special_tokens=false` into the request body |
| one output token | supported | forced to one |
| split UTF-8 SSE chunks | each network chunk is decoded independently | opt-in incremental UTF-8 decoder |
| stable source request IDs | native custom sampling creates index IDs | opt-in patch retains dataset request IDs |
| detailed per-request arrays | supported with `--save-detailed` | forced |
| 10-second TTFT goodput | supported, but failed requests are excluded from the native good-request loop | postprocessor also reports SLO attainment over all issued requests |
| server-measured prompt-token usage | usage is received but prompt-token usage is not saved | dataset/result length equality is checked; use server/profile evidence for an independent cross-check |

The compatibility code is
`afd_plugin/compat/patches/benchmark_serving.py`. It copies the pinned upstream
methods, marks AFD-only differences, and is applied only by
`tools.benchmarks.prefill_bench`. It does not change normal plugin or server
startup. The long-term action is to upstream exact token-ID custom datasets and
incremental SSE decoding, then delete this patch when the pinned release
contains both.

### Local client smoke test

Install a platform-compatible vLLM 0.19.1 client environment. Start the local
mock endpoint:

```bash
python -m tools.benchmarks.prefill_mock_server --port 18000
```

In a second terminal, run a small generated dataset:

```bash
python -m tools.benchmarks.prefill_bench \
  --backend vllm \
  --base-url http://127.0.0.1:18000 \
  --endpoint /v1/completions \
  --model prefill-mock \
  --served-model-name prefill-mock \
  --dataset-path /tmp/prefill-smoke.jsonl \
  --num-prompts 8 \
  --request-rate 4 \
  --save-result \
  --result-filename /tmp/prefill-smoke-result.json
```

The mock rejects non-integer prompts and output lengths other than one. Its
Chinese response is deliberately written across a UTF-8 boundary. Unit tests
also exercise this boundary deterministically without depending on TCP packet
coalescing:

```bash
uv run pytest -q \
  tests/unit/tools \
  tests/unit/compat/patches/test_benchmark_serving.py
```

### Prepare and inspect the matrix

Copy and edit
`tools/benchmarks/prefill_experiment.example.json`. Set the real model,
endpoint, dataset paths, result directory, and the two operator-owned server
launch templates. Print the complete matrix without launching anything:

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/prefill_experiment.json \
  plan
```

The printed order is grouped by system and maximum batched tokens. Restart the
server for each group. The same maximum token value is required on every AFD
Attention and FFN process.

## Phase 2: run on the NPU server

### Migration preflight

Copy the repository commit, CSV, generated JSONL files, manifests, model config,
and edited experiment configuration. Collect a machine-readable report before
the first run:

```bash
python -m tools.benchmarks.prefill_preflight \
  --require-npu \
  --model-config /models/DeepSeek-V3.2-reduced \
  --experiment-config /path/to/prefill_experiment.json \
  --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
  --output bench_results/prefill/preflight.json
```

The command records Git state, Python and package versions, dataset/model
hashes, matrix dimensions, `npu-smi info`, and `msprof --version`. It fails on
the wrong vLLM version, a missing artifact/manifest, a Git LFS pointer, or
missing NPU tools.

Before each group, additionally record:

- all server command lines and environment variables;
- rank table, node IPs, physical NPU IDs, HCCL/CAM versions, and NUMA binding;
- AFD connector configuration, `dynamicQuant`, async MoE ubatching mode, and
  Attention/FFN rank mapping;
- model commit/config hash and reduced layer count;
- prefix-cache, chunked-prefill, `max_num_seqs`, block size, graph/eager, and
  scheduling settings.

The workload reaches 50,773 input tokens while the smallest batch-token limits
are 8K and 16K. Set `max_model_len` to at least 50,774 and use the same
chunked-prefill policy on both systems; otherwise those cells do not exercise
the intended workload. Verify in server logs that long prompts are admitted and
chunked rather than truncated or rejected.

### Execute one correctly configured server group

After launching exactly one system at exactly one maximum-batched-token value,
run its five RPS values and three repeats:

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/prefill_experiment.json \
  run \
  --system dp4_tp8_sp \
  --batch-tokens 32768 \
  --prefix-ratio 0 \
  --resume
```

The runner refuses to execute more than one system/batch group at once. This
prevents a 32K client label from being sent accidentally to a server still
configured for another batch-token limit. Before sending a workload, it polls
`/v1/models` and requires the
configured served-model name. This catches a dead endpoint or a still-running
server from a different deployment without warming the prefix cache with a
data request.

Each raw result is followed by a `.verified.json` result containing:

- stable request IDs and CSV source rows;
- per-request success, TTFT, derived end-to-end latency, and SLO outcome;
- failed requests counted as SLO misses;
- all-issued and successful-only SLO rates;
- TTFT and SLO summaries in `<=8K`, `8–16K`, `16–32K`, `32–48K`, and `>48K`
  input-length buckets.

Use `--resume` only after checking that the server configuration still matches
the group. It skips verified cells and postprocesses an existing raw cell.

After all repeats, aggregate per-request TTFT and all-issued SLO rates and
produce a flat CSV:

```bash
python -m tools.benchmarks.prefill_report \
  --result-dir bench_results/prefill \
  --baseline dp4_tp8_sp \
  --candidate afd_dp3_tp8_ep8 \
  --expected-repeats 3 \
  --output bench_results/prefill/report.json \
  --csv-output bench_results/prefill/report.csv
```

The report marks incomplete cells and computes candidate mean/P99 TTFT
reduction plus SLO-attainment percentage-point deltas only for paired cells.
It also retains per-repeat mean TTFT/SLO values and their standard deviations,
so a delta smaller than repeat noise is not presented as a stable gain.

### Controls and sensitivity experiments

Change one factor at a time after the primary matrix:

1. prefix cache enabled with the 25%, 50%, and 75% datasets;
2. `max_num_seqs` and chunked-prefill policy at the best and saturation cells;
3. arrival burstiness while holding mean RPS fixed;
4. dataset ordering or a recorded alternate permutation;
5. CAM `dynamicQuant` and async MoE ubatching, if both systems remain valid;
6. request-split versus token-split mode where topology constraints allow it;
7. long-only, short-only, and length-stratified subsets to separate scheduler
   balance from kernel efficiency.

Treat graph/eager mode, model shape, HCCL/CAM settings, rank placement, host
binding, block size, sampling parameters, and server log level as controls
unless they are the named sensitivity under test.

## Profile replay and performance attribution

Do not enable a profiler in any reported end-to-end cell. Select independent,
matched replays after locating:

- a low-load cell where both systems meet SLO;
- the knee where one system begins to queue;
- a high-load cell showing a repeatable gain;
- at least one 32K or 64K maximum-token setting.

Use the same request IDs, arrival schedule, model, server settings, and physical
rank placement for the baseline and AFD trace. Save client start/TTFT arrays,
server scheduler logs, Attention traces, FFN traces, and rank mapping under one
replay manifest. A wide capture window is safer than assuming one request
equals one profiler step: a complex/chunked prefill can span multiple model
steps.

Use a copied profile configuration with only the selected RPS, one repeat, and
the fixed replay request count. Add `--plot-timeline` to `plan` or `run` to
produce the client-side request/TTFT timeline HTML without enabling it for the
150 primary runs.

AFD Attention and FFN tracing is controlled independently:

```bash
export AFD_NPU_ATTENTION_PROFILER_ENABLE=true
export AFD_NPU_ATTENTION_PROFILER_WAIT=0
export AFD_NPU_ATTENTION_PROFILER_WARMUP=1
export AFD_NPU_ATTENTION_PROFILER_ACTIVE=20
export AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST=0
export AFD_NPU_ATTENTION_PROFILER_DIR=/profiles/afd/attention

export AFD_NPU_FFN_PROFILER_ENABLE=true
export AFD_NPU_FFN_PROFILER_WAIT=0
export AFD_NPU_FFN_PROFILER_WARMUP=1
export AFD_NPU_FFN_PROFILER_ACTIVE=40
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=0
export AFD_NPU_FFN_PROFILER_DIR=/profiles/afd/ffn
```

Adjust active steps after a short calibration replay. Capture all relevant
ranks; a rank-0-only trace can hide DP/SP imbalance or a straggling expert
rank.

Summarize a TensorBoard/Chrome trace:

```bash
python -m tools.benchmarks.profile_trace summarize \
  --trace /profiles/afd/attention/rank0.pt.trace.json \
  --output /profiles/afd/attention/rank0.summary.json
```

Compare matched trace files:

```bash
python -m tools.benchmarks.profile_trace compare \
  --baseline /profiles/baseline/rank0.pt.trace.json \
  --candidate /profiles/afd/attention/rank0.pt.trace.json \
  --output /profiles/comparison/rank0.json
```

The tool classifies communication, memory movement, Attention, MoE/FFN, other
compute, host work, and unclassified events. It reports event counts, event
time, time-union metrics, per-lane busy time, top event names, and
communication/compute overlap. Event sums can double-count nested events and
parallel ranks; always inspect the exported timeline and unclassified top
events before extending classification rules.

For each claimed gain, require all four forms of evidence:

1. timeline: the predicted idle gap, synchronization, transfer, or kernel
   sequence visibly changes;
2. quantity: matched windows show how many milliseconds and what fraction of
   communication are removed or overlapped;
3. ablation: disabling/rolling back the suspected async, ubatching, or
   communication behavior restores the cost;
4. end to end: unprofiled repeats move TTFT/SLO in the same direction by more
   than run-to-run noise.

DP4 versus AFD DP3+EP8 changes both execution and resource partitioning, so the
main A/B alone cannot prove that async overlap caused the gain. Use intermediate
AFD sync/async or overlap rollback replays where supported, and report any
remaining topology/resource contribution separately.

## Acceptance gate

Phase 1 is complete when the unit tests pass, dataset generation is
deterministic, manifests and validators agree, the mock smoke run succeeds in a
vLLM 0.19.1 client environment, and the edited 150-cell plan prints correctly.

Phase 2 is complete only after every primary cell has three valid unprofiled
repeats, failures are included in SLO attainment, raw artifacts and preflight
evidence are archived, profile replays are independent, and each performance
attribution satisfies the four-evidence gate above.
