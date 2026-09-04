# xnode32 attention layout DP3TP8 (+ DP8EP8 FFN) — e2e perf, 2026-09-04

Same 32-card dual-node AFD stack as `../async_sched/` (mbt=65536, CWS=1,
FLASHCOMM1=1, async-sched ON, ASYNC_MOE_SPLIT=request, HCCL_BUFFSIZE=4096),
only the attention layout differs: `ATTN_LAYOUT=dp3tp8` (DP3 x TP8 = 24 attn
ranks; node1 DP0-1 on 16 NPUs + API, node2 DP2 headless on 8 NPUs) instead of
the default dp6tp4.  FFN unchanged (DP8EP8, node2 NPUs 8-15).  attn_ranks_per_dp=8.

Pods recreated for this run (fresh rootfs, env rebuilt from NAS:
CAM vendor + umdk + vllm-ascend e19e14da7 artifacts + editable plugin):
v4f-base-2 = 33.215.116.167 (master), v4f-xnode-2 = 33.215.116.209 (same /22).

Driver: `tools/itask/xnode32_dp3tp8_run.sh`; one server bring-up, then
formal_1 at 1x / 1.5x / 2x.  All cells 512/512, 0 mismatched prompt tokens.

## Results (formal_1, offered 35,070 tok/s at 1x)

| rate | layout | ok | wall_s | drain_s | ttft p50/p95/p99/max (s) | eff tok/s |
|------|--------|----|--------|---------|--------------------------|-----------|
| 1x   | dp3tp8 | 512 | 157.2 |  6.8 | 4.71 / 28.91 / 32.79 / 35.79 | 33,555 |
| 1x   | dp6tp4 | 512 | 156.2 |  5.7 | 2.45 / 5.59 / 8.21 / 11.25   | 33,796 |
| 1.5x | dp3tp8 | 512 | 120.1 | 19.3 | 10.12 / 17.92 / 19.78 / 22.58 | 44,079 |
| 1.5x | dp6tp4 | 512 | 107.1 |  6.2 | 3.35 / 9.92 / 11.96 / 13.96  | 49,516 |
| 2x   | dp3tp8 | 512 | 116.1 | 40.3 | 18.46 / 34.87 / 38.48 / 41.48 | 45,644 |
| 2x   | dp6tp4 | 512 | 90.1  | 14.4 | 6.77 / 12.27 / 13.95 / 15.44 | 58,868 |

dp6tp4 reference = `../async_sched/xnode32_request_as_mbt65536_*.json`
(same async-sched ON, request split, mbt 65536 batch).

## Verdict

- **1x (supply-limited)**: throughput parity (-0.7%), but dp3tp8 TTFT tail is
  already 4x worse (p99 32.8s vs 8.2s) — 3 engines queue twice the requests each.
- **1.5x**: -11.0% eff (44.1K vs 49.5K), drain 3x worse.
- **2x (capacity-limited)**: **-22.5% eff (45.6K vs 58.9K)**, drain 40.3s vs
  14.4s, TTFT p50 18.5s — deep steady queueing.
- Device latency stays ~6ms p99 in all cells: the loss is scheduling/queueing,
  not kernel.  DP6TP4's 6 lighter engines strictly dominate for this
  long-prefill workload; TP8's wider attention does not pay for halving the
  engine count.

Conclusion: keep dp6tp4 as the default xnode32 layout; dp3tp8 is archived for
reference (launch knob `ATTN_LAYOUT=dp3tp8`).
