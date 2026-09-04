# Single-node 16-card AFD, attention DP4TP2 + FFN DP8EP8 — 2026-09-04

One pod (v4f-base-2 = 33.215.116.167), all 32-card AFD knobs on a single node:
mbt=65536, CWS=1, FLASHCOMM1=1, async-sched ON, ASYNC_MOE_SPLIT=request,
HCCL_BUFFSIZE=4096, attn_ranks_per_dp=2, util 0.80, mml 70000.
Attention DP4TP2 on NPUs 0-7 (API 8900), FFN DP8EP8 on NPUs 8-15 (8901).

Driver: `tools/itask/singlenode_dp4tp2_run.sh` (FFN-first 20s stagger,
slow2x -> slow1p33x -> 1x).  All cells 512/512.

## Buffer size note

HCCL_BUFFSIZE=1024 (the validated minimum for the old single-node DP2TP4 @
mbt65536) **crashes this layout**: EngineDeadError during bring-up.  4096
works.  Whether the extra need comes from DP4TP2 or from FLASHCOMM1=1 on this
topology is untested; 4096 is the go-to for single-node 65536 now.

## Results (formal_1; offered 17.5K / 26.3K / 35.1K tok/s)

| rate | ok | wall_s | plan_s | drain_s | ttft p50/p95/p99/max (s) | eff tok/s | eff/offered |
|------|----|--------|--------|---------|--------------------------|-----------|-------------|
| 0.5x  | 512 | 312.3 | 300 | 12.2 | 2.48 / 7.24 / 9.94 / 13.06 | 16,853 | 96.1% |
| 0.75x | 512 | 211.2 | 200 | 10.6 | 3.14 / 7.75 / 10.39 / 11.36 | 24,984 | 95.0% |
| 1x    | 512 | 161.2 | 150 | 10.7 | 5.52 / 13.72 / 15.82 / 18.82 | 32,742 | 93.4% |

## Reading

- The ~10s TTFT tail at 0.5x/0.75x is **intrinsic single-request prefill
  latency**, not queueing: a 63K-token prompt takes ~8-14s to prefill on one
  engine at this topology's per-engine speed.  The tail only starts growing
  from queueing at 1x (p50 5.52s, p99 15.82s).
- **1x keeps up (93.4% of offered, drain 10.7s)**: single-node 16-card
  capacity is at least ~33K tok/s — 1.8x the old single-node DP2TP4
  mbt10240 measurement (17.8K), and roughly on par with what dp6tp4 32-card
  showed at 1x (33.8K, also supply-limited).  Where dp4tp2 sits vs dp6tp4
  under overload (1.5x/2x) is untested.
- Device latency stays 6-8ms p99 through all cells.

## vs baseline DP4TP4EP16 (same pod, 16 cards, no AFD)

Baseline source: `../baseline1x/` (2026-09-02, single instance, mbt 8192 and
32768).  Each side runs its own best-known mbt: baseline 32768 (65536 not
viable for it), AFD 65536.  Note the baseline cells predate the async-sched
default flip, so baseline = async OFF while AFD = async ON — part of the 1x
throughput delta may be scheduler, not decoupling (async alone measured
+3.5-4.7% on other stacks).  The TTFT/drain columns are confound-free.
peak = max 15s-bucket service rate (tools/itask/bucket15_summary.py).

| rate | config | eff tok/s | ttft p50/p99 (s) | drain_s | peak_15s |
|-------+--------------------------+-----------+-----------------+---------+----------|
| 0.5x  | AFD dp4tp2+ep8 mbt65536  | 16,853 | 2.48 /  9.94 | 12.2 | 33,840 |
| 0.5x  | base mbt8192             | 16,961 | 3.02 / 10.54 | 10.2 | 27,190 |
| 0.5x  | base mbt32768            | 16,885 | 6.21 / 15.31 | 11.6 | 28,239 |
| 0.75x | AFD dp4tp2+ep8 mbt65536  | 24,984 | 3.14 / 10.39 | 10.6 | 40,712 |
| 0.75x | base mbt8192             | 23,885 | 3.98 / 15.03 | 19.9 | 34,883 |
| 0.75x | base mbt32768            | 24,644 | 5.99 / 16.83 | 13.5 | 43,128 |
| 1x    | AFD dp4tp2+ep8 mbt65536  | 32,742 | 5.52 / 15.82 | 10.7 | 42,090 |
| 1x    | base mbt8192             | 29,687 | 8.96 / 24.02 | 26.7 | 36,995 |
| 1x    | base mbt32768            | 31,276 | 7.74 / 22.76 | 18.2 | 45,118 |

Deltas vs baseline mbt32768: eff -0.2% / +1.4% / +4.7%;  TTFT p99 -35% / -38% / -30%.
Deltas vs baseline mbt8192:  eff -0.6% / +4.4% / +10.0%; TTFT p99 -6%  / -31% / -34%.

Reading: at sub-capacity rates throughput is parity (both follow supply) and
AFD's win is **TTFT tail -30~38% at every rate** plus flatter drain.  At 1x
AFD adds +4.7%/+10.0% eff with 40-60% less drain.  The p50 advantage
(2.5-3.1s vs 6.0s at 0.5x/0.75x) comes from mbt65536 single-chunk prefill.

Peak 15s burst: AFD 42.1K @1x vs baseline32768 45.1K (-7%) and baseline8192
37.0K (+14%).  AFD's burst profile is flatter across rates (33.8 -> 40.7 ->
42.1K) — the ubatch pipeline smooths output instead of stacking burst chunks.

Sustained capacity: **knee not measured** — 1x was still supply-following
(93.4%, drain 10.7s), so capacity is only bounded below by ~33K tok/s.
Needs 1.5x/2x overload cells to locate.
