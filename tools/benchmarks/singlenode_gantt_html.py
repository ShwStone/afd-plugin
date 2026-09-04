#!/usr/bin/env python3
"""Build the interactive per-request Gantt for the 2026-09-04 single-node
comparison set (plus the morning dual-node DP3TP8 run), excluding DP8TP2.

Reuses the rendering template from xnode32_gantt_html.py unchanged; only the
system set, run files, and load tabs differ.

Systems (all async-sched ON, CWS, formal_1):
  dp3tp8   dual-node 32c DP3TP8+EP8        mbt=65536 (1x/1.5x/2x)
  sreq     single-node AFD DP4TP2+EP8      request-split mbt=65536
  stok     single-node AFD DP4TP2+EP8      token-split   mbt=65536
  b8k      single-node baseline DP4TP4EP16 mbt=8192
  b32k     single-node baseline DP4TP4EP16 mbt=32768

Usage: python3 singlenode_gantt_html.py <result_dir>
Output: <result_dir>/gantt_20260904.html
"""
import json
import sys
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "g", Path(__file__).with_name("xnode32_gantt_html.py"))
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)

SYSTEMS = {  # key -> (label, color); Okabe-Ito, CVD-checked (dE>=12 all pairs)
    "dp3tp8": ("双机 DP3TP8+EP8 mbt=65536",        "#0072B2"),
    "sreq":   ("单机 AFD DP4TP2+EP8 request 65536", "#D55E00"),
    "stok":   ("单机 AFD DP4TP2+EP8 token 65536",   "#009E73"),
    "b8k":    ("单机 baseline DP4TP4EP16 mbt=8192",  "#CC79A7"),
    "b32k":   ("单机 baseline DP4TP4EP16 mbt=32768", "#E69F00"),
}
DEFAULT_ON = ["stok", "sreq", "b8k", "b32k"]
DOUBLED = {}  # no virtual systems in this set
LOADS = ["0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"]

RUNS = {
    ("dp3tp8", "1x"):    "dp3tp8/xnode32_dp3tp8_mbt65536_1x.json",
    ("dp3tp8", "1.5x"):  "dp3tp8/xnode32_dp3tp8_mbt65536_fast1p5x.json",
    ("dp3tp8", "2x"):    "dp3tp8/xnode32_dp3tp8_mbt65536_fast2x.json",
    ("sreq", "0.5x"):    "dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_slow2x.json",
    ("sreq", "0.75x"):   "dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_slow1p33x.json",
    ("sreq", "1x"):      "dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_1x.json",
    ("sreq", "1.25x"):   "dp4tp2_single_sweeps/dp4tp2req_mbt65536_1p25x.json",
    ("sreq", "1.5x"):    "dp4tp2_single_sweeps/dp4tp2req_mbt65536_1p5x.json",
    ("stok", "0.5x"):    "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_0p5x.json",
    ("stok", "0.75x"):   "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_0p75x.json",
    ("stok", "1x"):      "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1x.json",
    ("stok", "1.25x"):   "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1p25x.json",
    ("stok", "1.5x"):    "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1p5x.json",
    ("b8k", "0.5x"):     "dp4tp2_single_sweeps/base1xas_mbt8192_0p5x.json",
    ("b8k", "0.75x"):    "dp4tp2_single_sweeps/base1xas_mbt8192_0p75x.json",
    ("b8k", "1x"):       "dp4tp2_single_sweeps/base1xas_mbt8192_1x.json",
    ("b8k", "1.25x"):    "dp4tp2_single_sweeps/base1xas_mbt8192_1p25x.json",
    ("b8k", "1.5x"):     "dp4tp2_single_sweeps/base1xas_mbt8192_1p5x.json",
    ("b32k", "0.5x"):    "dp4tp2_single_sweeps/base1xas_mbt32768_0p5x.json",
    ("b32k", "0.75x"):   "dp4tp2_single_sweeps/base1xas_mbt32768_0p75x.json",
    ("b32k", "1x"):      "dp4tp2_single_sweeps/base1xas_mbt32768_1x.json",
    ("b32k", "1.25x"):   "dp4tp2_single_sweeps/base1xas_mbt32768_1p25x.json",
    ("b32k", "1.5x"):    "dp4tp2_single_sweeps/base1xas_mbt32768_1p5x.json",
}


def main():
    result_dir = Path(sys.argv[1])
    data = {l: {} for l in LOADS}
    summaries = {l: {} for l in LOADS}
    for (syskey, load), rel in RUNS.items():
        d = json.load(open(result_dir / rel))
        s = d["summary"]
        summaries[load][syskey] = {
            "eff": round(s["total_input_tokens"] / s["wall_s"]),
            "drain": round(s["queue_drain_s"], 1),
            "p50": round(s["ttft_s"]["p50"], 2),
            "p99": round(s["ttft_s"]["p99"], 2),
        }
        recs = {}
        for r in d["requests"]:
            if not r["success"]:
                continue
            recs[_g.rid_num(r["request_id"])] = [
                round(r["actual_send_s"], 3),
                round(r["ttft_s"], 3),
                round(r["e2el_s"], 3),
                r["input_length"],
            ]
        data[load][syskey] = recs

    payload = {
        "systems": {k: {"label": v[0], "color": v[1]} for k, v in SYSTEMS.items()},
        "defaultOn": DEFAULT_ON,
        "loads": LOADS,
        "data": data,
        "summaries": summaries,
    }
    html = _g.TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    out = result_dir / "gantt_20260904.html"
    out.write_text(html)
    print(f"saved {out} ({out.stat().st_size//1024} KiB)")


if __name__ == "__main__":
    main()
