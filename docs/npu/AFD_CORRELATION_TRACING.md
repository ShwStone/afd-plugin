# Correlating Attention and FFN profiler timelines

AFD correlation tracing adds a small, opt-in diagnostic layer around the
asynchronous CAM data path. It answers two questions that separate
`torch_npu` profiles cannot answer by themselves:

1. Which Attention dispatch, FFN computation, and Attention combine belong to
   the same model-execution step, layer, and microbatch?
2. Where should traces from different processes and hosts be placed on one
   timeline?

This facility is intended for diagnosis. Do not use instrumented runs as the
primary source for published throughput or latency results. The named profiler
ranges and Python event recorder have small but nonzero overhead.

## What is recorded

When enabled, every process writes a scalar-only JSONL sidecar. Each event has:

- the role, global rank, role rank, local rank, host, and process ID;
- a profile session ID;
- an execution transaction ID, model layer, microbatch stage, and token count;
- a stable logical flow ID;
- timestamps from `CLOCK_MONOTONIC_RAW`; and
- begin/end status for these ranges:
  `afd.cam.dispatch_send`, `afd.cam.dispatch_recv`, `afd.ffn.compute`,
  `afd.cam.combine_send`, and `afd.cam.combine_recv`.

The same range names and flow IDs are emitted as MSTX ranges, so the sidecar
can be aligned with the corresponding `torch_npu` trace. The NPU profiler must
enable MSTX collection; the AFD-owned profiler does this automatically. The
recorder does not inspect tensors, copy tensor values to the CPU, synchronize
the NPU, or modify CAM operator arguments. Sidecar begin/end intervals describe
CPU-side enqueue and wait boundaries; use the aligned `torch_npu` device
events, not sidecar duration, to measure kernel or communication execution
time.

For async CAM, Attention and FFN processes derive the same transaction ordinal
from the ordered receive stream. The FFN side accounts for the configured
number of MoE microbatches, so both stages of a two-stage run retain the same
transaction ID and their own stage IDs. The CAM device protocol does not have
an extra field for a Python string, so the transaction ID is not placed in the
device payload. A missing or extra transfer on either role therefore appears as
incomplete or reordered flows in the validation report; such a report must not
be used to infer cross-role durations.

## Enable tracing

Set the following variables on every Attention and FFN process before launch:

```bash
export AFD_TRACE_ENABLE=1
export AFD_TRACE_SESSION_ID=profile-20260821-01
export AFD_TRACE_DIR=/path/shared-or-collected-later/afd-sidecars
```

`AFD_TRACE_SESSION_ID` must be identical on every host and unique for one
launch. `AFD_TRACE_MAX_EVENTS` optionally limits recorded events per process;
the default is 200,000. Every event is immediately written and flushed to a
temporary sidecar, so a hard kill preserves the completed prefix. Merge tools
recover a truncated final line from `.jsonl.tmp`. Normal connector shutdown
adds the final clock anchor and an event/drop summary before atomically renaming
the file to `.jsonl`.

Correlation sidecars can be collected without an accelerator profiler. Set
`FP_CORRELATION=1` for `fp_orchestrate.py`, or `CORRELATION=1` for
`run_stage2_l0.sh`; do not set `FP_PROFILE`/`PROFILE`. This records only the
process-lifetime JSONL sidecars and does not register or call vLLM's profiler
endpoints.

To align sidecars with device events, launch both roles with vLLM
`--profiler-config`, then bound the profiler capture with `POST /start_profile`
and `POST /stop_profile`. Start FFN before Attention and stop Attention before
FFN. Sidecars still cover the process lifetime, while MSTX markers appear only
inside that explicit profiler window. AFD no longer supports an independent
fixed-step profiler schedule.

## Calibrate clocks across hosts

Host realtime clocks may look synchronized while still being inaccurate enough
to distort a sub-millisecond communication timeline. The clock helper takes
NTP-style four-timestamp samples using the same monotonic clock as the sidecar.

Choose one profiling host as the reference. On it, start a server and specify
the number of other hosts:

```bash
python tools/benchmarks/afd_trace_clock_sync.py server \
  --session-id profile-20260821-01 \
  --host 0.0.0.0 \
  --port 29610 \
  --clients 7
```

On each other host, run one client:

```bash
python tools/benchmarks/afd_trace_clock_sync.py client \
  --session-id profile-20260821-01 \
  --server REFERENCE_HOST \
  --port 29610 \
  --samples 16 \
  --output clock-sync-$(hostname).json
```

Run the calibration immediately before or after profiling. For a long capture,
run it at both ends and compare the selected offsets for clock drift. The merge
tool currently uses one file per client host and selects its minimum-round-trip
sample. Its reported uncertainty is half that sample's network round trip.

The server listens without authentication. Bind it to a trusted cluster network
and stop it after all expected clients finish.

## Merge and inspect

First merge sidecars alone:

```bash
python tools/benchmarks/merge_afd_correlation_traces.py \
  --sidecar /collected/afd-sidecars/profile-20260821-01 \
  --clock-sync /collected/clock-sync-host02.json \
  --clock-sync /collected/clock-sync-host03.json \
  --output /collected/afd-merged.json
```

Open `afd-merged.json` in Perfetto or a Chrome trace viewer. The adjacent
`afd-merged.report.json` is part of the result and must be checked first. It
lists:

- the alignment method and uncertainty for every process;
- finalized versus recovered temporary sidecars and event-count consistency;
- events dropped because the configured limit was reached;
- events without a flow ID, unmatched begin/end ranges, and error outcomes;
- incomplete flows, including missing phases on individual role ranks;
- flows whose cross-host order is reversed after clock correction; and
- profiler traces that lack matching AFD markers; and
- per-profiler correlation-to-device flow counts, including ambiguous links
  rejected during multi-rank merging.

To add raw `torch_npu` traces, map each trace to the exact sidecar produced by
the same process:

```bash
python tools/benchmarks/merge_afd_correlation_traces.py \
  --sidecar /collected/afd-sidecars/profile-20260821-01 \
  --clock-sync /collected/clock-sync-host02.json \
  --profiler-trace ATTENTION_SIDECAR.jsonl ATTENTION_PROFILER.json \
  --profiler-trace FFN_SIDECAR.jsonl FFN_PROFILER.json.gz \
  --output /collected/afd-merged-with-profiler.json
```

The tool aligns each raw trace using the median difference between its named
AFD ranges and matching sidecar begin events. The report includes the spread of
those marker differences. A large spread means the raw trace clock cannot be
modeled as a constant shift and the merged device timeline should be treated as
invalid.

When profiler traces contain both MSTX markers and CANN
`CamMoeDistribute*` operations, the merge tool also adds visible Perfetto flow
arrows from each correlation range to the device operation it enqueues. The
standalone `tools.benchmarks.link_afd_device_flows` command is only needed to
upgrade a merged trace produced by an older version of the merge tool.

For a multi-host session, the merge tools require one reference host and one
clock-calibration file for every other host. They reject incomplete calibration
instead of silently assuming synchronized realtime clocks. Single-host merges
continue to use the local realtime anchor.
