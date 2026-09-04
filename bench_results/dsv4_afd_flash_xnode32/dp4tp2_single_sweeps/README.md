# Single-node 16-card comparison set — baseline vs AFD dp4tp2, 2026-09-04

Four-run matrix on the two rebuilt pods (identical image/env, same /22):
base-2 = baseline line, xnode-2 = AFD line. formal_1, 512 reqs, rates
0.5x/0.75x/1x/1.25x/1.5x (offered 17.5K/26.3K/35.1K/43.8K/52.6K tok/s),
all cells 512/512. Common knobs everywhere: async-sched ON, CWS on, util 0.80,
mml 70000. Per-series config:

- **AFD tok 65536** — DP4TP2+EP8, ASYNC_MOE_SPLIT=token, FLASHCOMM1, HCCL_BUFFSIZE=4096
- **AFD req 65536** — same but split=request (0.5x-1x from `../dp4tp2_singlenode/`,
  1.25x/1.5x completed here after that server died on a manual 1.5x attempt)
- **base-async 8192 / 32768** — stock DP4TP4EP16, ASYNC_SCHEDULING=1 (retest of
  `../baseline1x/` which predated the async flip)
- **base-noasync** — the 2026-09-02 cells, kept for the async effect

peak = max 15s-bucket service rate. Files: `base1xas_*.json` (baseline async),
`dp4tp2tok_*.json`, `dp4tp2req_*.json`.

## eff tok/s (supply-following % / TTFT p99 s / peak K)

| series | 0.5x | 0.75x | 1x | 1.25x | 1.5x |
|--------|------|-------|----|-------|------|
| AFD tok 65536 | 17,075 (97/9.0/30K) | 25,230 (96/10.7/41K) | 32,830 (94/15.6/45K) | 37,321 (85/25.6/47K) | **40,096 (76/34.3/47K)** |
| AFD req 65536 | 16,853 (96/9.9/34K) | 24,984 (95/10.4/41K) | 32,742 (93/15.8/42K) | 36,856 (84/27.1/44K) | 36,837 (70/36.0/44K) |
| base-async 32768 | 16,796 (96/18.8/31K) | 23,928 (91/20.5/43K) | 31,340 (89/21.8/45K) | 35,022 (80/26.4/46K) | 36,925 (70/36.6/45K) |
| base-async 8192 | 16,735 (95/11.3/27K) | 24,425 (93/13.2/36K) | 31,058 (89/20.0/39K) | 32,645 (75/35.7/39K) | 30,917 (59/59.2/38K) |
| base-noasync 32768 | 16,885 | 24,644 | 31,276 | — | — |
| base-noasync 8192 | 16,961 | 23,922 | 29,770 | — | — |

## Verdicts

1. **AFD token-split mbt65536 is the best single-node config at every rate.**
   vs the best baseline (async 32768): +0-2% at sub-capacity rates, **+4.7% at
   1x, +6.6% at 1.25x, +8.6% at 1.5x**, and TTFT p99 -28% at 1x (15.6 vs 21.8s).
2. **AFD sustained capacity ≈ 38-40K tok/s vs baseline ≈ 37K** (16 cards).
   Request-split plateaus at 36.8K; token-split keeps climbing to 40.1K and
   degrades slower (drain 31s vs 43s at 1.5x). Token > request at every rate
   (+0.3% .. +8.9%).
3. **The async confound is resolved**: async ON vs OFF on baseline-32768 moves
   1x eff by +0.2% (and none of the earlier +4.7% can be scheduler). On 8192
   async helps more (1x +4.3%, p99 24.0->20.0s) but 8192 saturates first
   regardless (1.5x eff collapses to 59% of supply, p99 59s).
4. mbt8192 is the wrong operating point for this long-prefill workload on
   stock baseline; 32768 tracks AFD's shape but flat-worse.
5. Peak 15s burst is a wash (AFD 44-47K vs baseline 43-46K) — the win is in
   sustained rate and TTFT tail, not burst.
