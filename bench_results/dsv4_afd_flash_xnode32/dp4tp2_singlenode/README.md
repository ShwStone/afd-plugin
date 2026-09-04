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

Baseline source: `../baseline1x/` (2026-09-02, single instance, best cell =
mbt32768).  Each side runs its own best-known mbt: baseline 32768 (65536 not
viable for it), AFD 65536.  Note the baseline cells predate the async-sched
default flip, so baseline = async OFF while AFD = async ON — part of the 1x
throughput delta may be scheduler, not decoupling (async alone measured
+3.5-4.7% on other stacks).  The TTFT/drain columns are confound-free.

| rate | AFD eff | base32k eff | Δeff | AFD p99 | base32k p99 | Δp99 | AFD drain | base drain |
|-------|---------|-------------|------|---------|-------------|------|-----------|------------|
| 0.5x  | 16,853 | 16,885 | -0.2% | 9.94  | 15.31 | -35% | 12.2 | 11.6 |
| 0.75x | 24,984 | 24,644 | +1.4% | 10.39 | 16.83 | -38% | 10.6 | 13.5 |
| 1x    | 32,742 | 31,276 | +4.7% | 15.82 | 22.76 | -30% | 10.7 | 18.2 |

(vs baseline mbt8192: 1x eff 29,770 -> AFD +10.0%, p99 24.02 -> -34%.)

Reading: at sub-capacity rates throughput is parity (both follow supply) and
AFD's win is **TTFT tail -30~38% at every rate** plus flatter drain.  At 1x
AFD adds +4.7% eff with 40% less drain.  The p50 advantage (2.5-3.1s vs 6.0s
at 0.5x/0.75x) comes from mbt65536 single-chunk prefill.
