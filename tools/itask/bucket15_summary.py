#!/usr/bin/env python3
"""Per-cell summary for replay result JSONs: effective throughput, TTFT
percentiles, drain, and peak 15s-bucket service rate (input tokens of
requests completing inside each 15s window)."""
import json
import sys


def summarize(path: str) -> dict:
    d = json.load(open(path))
    reqs = [r for r in d["requests"] if r["success"]]
    s = d["summary"]
    buckets: dict[int, int] = {}
    for r in reqs:
        b = int(r["completed_s"] // 15)
        buckets[b] = buckets.get(b, 0) + int(r.get("prompt_tokens_reported") or r["input_length"])
    peak_bucket_rate = max(v / 15 for v in buckets.values()) if buckets else 0.0
    ttft = s.get("ttft_s", {})
    return {
        "file": path.split("/")[-1],
        "ok": f"{s['successful']}/{s['request_count']}",
        "failed": s["failed"],
        "wall_s": round(s["wall_s"], 1),
        "eff_tok_s": round(s["actual_send_token_rate"] and s["total_input_tokens"] / s["wall_s"] or 0),
        "ttft_p50": round(ttft.get("p50", -1), 2),
        "ttft_p95": round(ttft.get("p95", -1), 2),
        "ttft_p99": round(ttft.get("p99", -1), 2),
        "ttft_max": round(ttft.get("max", -1), 2),
        "drain_s": round(s.get("queue_drain_s", -1), 1),
        "peak_15s_tok_s": round(peak_bucket_rate),
        "tokens_mismatch": s.get("prompt_tokens_mismatched"),
    }


if __name__ == "__main__":
    rows = [summarize(p) for p in sys.argv[1:]]
    hdr = ["file", "ok", "failed", "wall_s", "eff_tok_s", "ttft_p50", "ttft_p95",
           "ttft_p99", "ttft_max", "drain_s", "peak_15s_tok_s", "tokens_mismatch"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[h]) for h in hdr))
