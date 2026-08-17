#!/usr/bin/env python3
"""Show why attn_output rel_l2 explodes: per-token rel_l2 + reference L2 norm."""
import glob
import re
import sys

import torch

CAP = sys.argv[1] if len(sys.argv) > 1 else "bench_results/precision/v026-safe-l8"
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
boundary = sys.argv[3] if len(sys.argv) > 3 else "attn_output"
tensor = sys.argv[4] if len(sys.argv) > 4 else "attn_output"
NOUB = f"{CAP}/no_ubatch"
TOK = f"{CAP}/token"


def load_shards(mode_dir, layer, boundary, tensor, stage=None):
    pat = (
        f"{mode_dir}/attention_dp0_tp*_t2_l{layer}_s{stage}_"
        if stage is not None
        else f"{mode_dir}/attention_dp0_tp*_t2_l{layer}_s0_"
    )
    files = sorted(glob.glob(pat + boundary + "_" + tensor + ".pt"))
    files.sort(key=lambda f: int(re.search(r"_tp(\d+)_", f).group(1)))
    return torch.cat([torch.load(f, map_location="cpu", weights_only=True) for f in files], dim=0)


def tokvec(tok0, tok1, t):
    return tok0[t] if t < 53 else tok1[t - 53]


noub = load_shards(NOUB, layer, boundary, tensor)
# is this a MoE layer (has stage1 in token) or dense?
has_s1 = len(glob.glob(f"{TOK}/attention_dp0_tp*_t2_l{layer}_s1_{boundary}_{tensor}.pt")) > 0
tok0 = load_shards(TOK, layer, boundary, tensor, 0)
tok1 = load_shards(TOK, layer, boundary, tensor, 1) if has_s1 else None

n_real = 105
rel = []
ref_l2 = []
for t in range(n_real):
    a = noub[t].float()
    b = (tokvec(tok0, tok1, t) if has_s1 else tok0[t]).float()
    denom = b.pow(2).sum().sqrt().item()
    diff = (a - b).abs()
    num = diff.pow(2).sum().sqrt().item()
    rel.append(num / (denom + 1e-12))
    ref_l2.append(denom)

rel = torch.tensor(rel)
ref_l2 = torch.tensor(ref_l2)
idx = rel.argmax().item()
print(f"layer={layer} boundary={boundary} tensor={tensor} staged={has_s1}")
print(f"max rel_l2 = {rel.max().item():.4f} @ token {idx}")
print(f"  that token: ref ||b||_2 = {ref_l2[idx].item():.6f}, diff ||a-b||_2 = {rel[idx].item()*ref_l2[idx].item():.6f}")
print(f"  ref ||b||_2 distribution: min={ref_l2.min().item():.6f} max={ref_l2.max().item():.6f} median={ref_l2.median().item():.6f}")
print(f"  #tokens with rel_l2>1: {(rel>1).sum().item()}, >10: {(rel>10).sum().item()}")
print(f"  tokens where ref_l2 < 0.01: {(ref_l2<0.01).sum().item()} / {n_real}")
