# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Estimate the max sustainable ``max_num_batched_tokens`` for Stage-3 btsweep.

Model: ``max_num_batched_tokens`` is a scheduler per-DP-replica token budget.
Raising it does not grow the KV block pool (vLLM pre-allocates it from
``gpu_memory_utilization`` at startup), so the *binding* constraints are:

1. **Per-step activation memory** must fit the headroom left after the KV
   reservation (``HBM * (1 - gpu_memory_utilization)`` minus runtime buffers).
2. **Scheduler structural cap** = ``max_num_seqs * max_model_len``.
3. **KV block pool** (computed for completeness; it is not binding here).
4. **FFN EP throughput** is a *soft* ceiling for AFD (EP8 = 1/4 of baseline
   EP32 rank-resource), which calibration confirms rather than this model.

Model constants come from the reduced DeepSeek-V3.2 model
(``deepseek-v3.2-reduced``): MLA KV = ``(kv_lora_rank + qk_rope_head_dim)``
bf16 elements per token per layer.

Usage:
  python3 -m tools.benchmarks.estimate_batch_capacity \
    [--model-config config.json] [--hbm-gib 64] \
    [--output bench_results/prefill_stage3/00_plan/capacity_estimate.json]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

GIB = 1024 ** 3
DTYPE_BYTES = 2  # bf16

# Fallbacks match /a3_inference/.../deepseek-v3.2-reduced/config.json.
DEFAULT_MODEL = {
    "hidden_size": 7168,
    "num_hidden_layers": 10,
    "intermediate_size": 18432,      # dense MLP
    "moe_intermediate_size": 2048,
    "num_experts_per_tok": 8,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "first_k_dense_replace": 3,
}

# max-num-seqs / max-model-len from the launch scripts (Stage-2 topology).
SYSTEMS = {
    "baseline": {
        "dp": 4,
        "tp": 8,
        "ep": 32,          # experts parallel across all 32 ranks
        "gpu_util": 0.90,
        "max_num_seqs": 8,
        "max_model_len": 70000,
    },
    "afd": {
        # Attention ranks: DP3 x TP8 (24), FFN: EP8 (8). Per-rank util applies
        # to the attention ranks (FFN sets no gpu-memory-utilization).
        "dp": 3,
        "tp": 8,
        "ep": 8,
        "gpu_util": 0.80,
        "max_num_seqs": 32,
        "max_model_len": 70000,
    },
}


def _load_model_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return DEFAULT_MODEL
    raw = json.loads(path.read_text(encoding="utf-8"))
    resolved = dict(DEFAULT_MODEL)
    for key in resolved:
        if raw.get(key) is not None:
            resolved[key] = raw[key]
    return resolved


def per_token_kv_bytes(model: dict[str, object]) -> float:
    """MLA KV cache bytes per token (compressed latent + rope), all layers."""
    per_layer = (
        int(model["kv_lora_rank"]) + int(model["qk_rope_head_dim"])
    ) * DTYPE_BYTES
    return float(per_layer * int(model["num_hidden_layers"]))


def kv_tokens_per_rank(
    hbm_bytes: float,
    gpu_util: float,
    weights_bytes_per_rank: float,
    kv_bytes_per_token: float,
    tp: int,
) -> float:
    """Global KV-block pool capacity (tokens) reachable by the scheduler.

    With FlashComm1/SP each TP rank stores ``1/tp`` of every active sequence's
    KV, so a rank's block pool covers ``tp * (kv_bytes / kv_per_token)``
    globally-scheduled tokens.
    """
    kv_reserved = hbm_bytes * gpu_util - weights_bytes_per_rank
    kv_reserved = max(kv_reserved, 0.0)
    return tp * kv_reserved / kv_bytes_per_token


def activation_bytes_per_step(
    batch_tokens: int,
    system: dict[str, object],
    model: dict[str, object],
) -> dict[str, float]:
    """Rough per-rank per-step peak activation (bytes) at a batch budget.

    Conservative upper bound: only one model layer's tensors are alive at a
    time (activations free each layer), and a DP replica's ``batch_tokens``
    are split across its TP ranks (SP) for attention/dense.
    """
    hidden = int(model["hidden_size"])
    tp = int(system["tp"])
    ep = int(system["ep"])
    moe_inter = int(model["moe_intermediate_size"])
    dense_inter = int(model["intermediate_size"])
    experts_per_tok = int(model["num_experts_per_tok"])

    tokens_per_rank = batch_tokens / tp
    # attention: q,k,v,o + residual/out buffers (~6 full hidden tensors).
    attention = tokens_per_rank * hidden * 6.0 * DTYPE_BYTES
    # dense MLP (only the first `first_k_dense_replace` layers).
    dense = tokens_per_rank * dense_inter * 3.0 * DTYPE_BYTES
    # routed MoE: each token fans out to experts_per_tok experts across EP
    # ranks; per rank it processes its share of expert tokens.
    routed = (batch_tokens * experts_per_tok / ep) * moe_inter * 3.0 * DTYPE_BYTES
    return {
        "attention_bytes": attention,
        "dense_mlp_bytes": dense,
        "routed_moe_bytes": routed,
        "total_bytes": attention + dense + routed,
    }


def estimate(
    model: dict[str, object],
    hbm_gib: float,
    weights_gib: float,
    runtime_overhead_gib: float,
    candidates: Sequence[int],
) -> dict[str, object]:
    hbm_bytes = hbm_gib * GIB
    kv_per_token = per_token_kv_bytes(model)
    output: dict[str, object] = {
        "model": model,
        "hbm_gib": hbm_gib,
        "kv_bytes_per_token": kv_per_token,
        "kv_gib_per_1m_tokens": kv_per_token * 1_000_000 / GIB,
        "systems": {},
        "recommended_sweep": {},
    }
    for name, spec in SYSTEMS.items():
        tp = int(spec["tp"])
        ep = int(spec["ep"])
        ranks = int(spec["dp"]) * tp
        weights_per_rank = weights_gib * GIB / ranks
        free_for_activations = hbm_bytes * (1.0 - spec["gpu_util"]) - (
            runtime_overhead_gib * GIB
        )
        kv_capacity = kv_tokens_per_rank(
            hbm_bytes,
            spec["gpu_util"],
            weights_per_rank,
            kv_per_token,
            tp,
        )
        structural_cap = int(spec["max_num_seqs"]) * int(spec["max_model_len"])

        rows: list[dict[str, object]] = []
        for batch in candidates:
            act = activation_bytes_per_step(batch, spec, model)
            total_gib = act["total_bytes"] / GIB
            rows.append(
                {
                    "max_num_batched_tokens": batch,
                    "activation_gib_per_rank_per_step": round(total_gib, 2),
                    "activation_margin_gib": round(
                        free_for_activations / GIB - total_gib, 2
                    ),
                    "oom_risk": total_gib > free_for_activations / GIB,
                }
            )
        oom_rows = [r for r in rows if r["oom_risk"]]
        ceiling_activation = (
            int(rows[0]["max_num_batched_tokens"])
            if oom_rows and rows[0]["oom_risk"]
            else max(
                (
                    int(r["max_num_batched_tokens"])
                    for r in rows
                    if not r["oom_risk"]
                ),
                default=None,
            )
        )
        output["systems"][name] = {
            "dp": spec["dp"],
            "tp": spec["tp"],
            "ep": spec["ep"],
            "ranks": ranks,
            "gpu_util": spec["gpu_util"],
            "weights_gib_per_rank": round(weights_per_rank / GIB, 2),
            "free_activation_gib_per_rank": round(
                free_for_activations / GIB, 2
            ),
            "kv_pool_capacity_tokens_global": int(kv_capacity),
            "scheduler_structural_cap_tokens": structural_cap,
            "rows": rows,
            "ceiling_activation": ceiling_activation,
        }
        output["recommended_sweep"][name] = {
            "max_sustainable_estimate": ceiling_activation,
            "note": (
                "AFD FFN EP8 is compute-bound (soft ceiling); calibration "
                "decides the real max. Baseline is activation-memory-bound."
                if name == "afd"
                else "Activation-memory-bound above this estimate."
            ),
        }
    return output


def _print_report(report: dict[str, object]) -> None:
    print("=== Batch-token capacity estimate ===")
    print(f"KV cache per token (10 layers, MLA): "
          f"{report['kv_bytes_per_token']:.0f} B"
          f"  (~{report['kv_gib_per_1m_tokens']:.2f} GiB / 1M tokens)")
    for name in ("baseline", "afd"):
        sys_info = report["systems"][name]
        print(f"\n[{name}] dp{sys_info['dp']} tp{sys_info['tp']} "
              f"ep{sys_info['ep']} util={sys_info['gpu_util']}  "
              f"free-activation/rank={sys_info['free_activation_gib_per_rank']} GiB")
        print(f"  KV pool (global) ~{sys_info['kv_pool_capacity_tokens_global']:,} "
              f"tokens — NOT binding")
        print(f"  scheduler structural cap "
              f"~{sys_info['scheduler_structural_cap_tokens']:,} tokens")
        print(f"  activation/rank/step (GiB) vs OOM:")
        for row in sys_info["rows"]:
            flag = "  <-- OOM RISK" if row["oom_risk"] else ""
            print(f"    mbt={row['max_num_batched_tokens']:>7}  "
                  f"act={row['activation_gib_per_rank_per_step']:>5.2f} GiB  "
                  f"margin={row['activation_margin_gib']:>+6.2f} GiB{flag}")
    print("\n=== Recommended sweep (max sustainable) ===")
    for name, rec in report["recommended_sweep"].items():
        print(f"  {name}: <= {rec['max_sustainable_estimate']}  — {rec['note']}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--hbm-gib", type=float, default=64.0)
    parser.add_argument("--weights-gib", type=float, default=85.0,
                        help="on-disk model size (quantized) in GiB")
    parser.add_argument("--runtime-overhead-gib", type=float, default=1.5)
    parser.add_argument("--mbs", type=int, nargs="*", default=[
        4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144,
    ])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    model = _load_model_config(args.model_config)
    report = estimate(
        model,
        args.hbm_gib,
        args.weights_gib,
        args.runtime_overhead_gib,
        args.mbs,
    )
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
