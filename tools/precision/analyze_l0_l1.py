#!/usr/bin/env python3
"""Compare layer-0 attention output vs layer-1 attention input divergence
between the no_ubatch and token capture dirs (per real token)."""
import glob
import re
import sys

import torch

CAP = sys.argv[1] if len(sys.argv) > 1 else "bench_results/precision/v026-safe-l8"
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


def per_tok_diff(noub, tok0, tok1, n_real=105, staged=True):
    a = noub[:n_real].float()
    if not staged:
        # dense layers: token capture is a single full-batch stage (no split)
        return (a - tok0[:n_real].float()).abs().max(dim=1).values
    b = torch.stack([tokvec(tok0, tok1, t).float() for t in range(n_real)])
    return (a - b).abs().max(dim=1).values


def summ(name, d, ref):
    print(f"  {name}: max={d.max().item():.6f}@tok{int(d.argmax())} "
          f"mean={d.mean().item():.6f} rel_max={d.max().item()/ref*100:.1f}%"
          f" rel_mean={d.mean().item()/ref*100:.1f}%")


# layer 0 attn_output (dense -> no stages)
l0_n = load_shards(NOUB, 0, "attn_output", "attn_output")
l0_t = load_shards(TOK, 0, "attn_output", "attn_output")
d0 = (l0_n[:105].float() - l0_t[:105].float()).abs().max(dim=1).values
ref0 = l0_n[:105].float().pow(2).mean().sqrt().item()

# layer 1 layer_input hidden_states + residual (dense: token is single stage)
l1h_n = load_shards(NOUB, 1, "layer_input", "hidden_states")
l1h_t0 = load_shards(TOK, 1, "layer_input", "hidden_states", 0)
d1h = per_tok_diff(l1h_n, l1h_t0, None, staged=False)
ref1h = l1h_n[:105].float().pow(2).mean().sqrt().item()

l1r_n = load_shards(NOUB, 1, "layer_input", "residual")
l1r_t0 = load_shards(TOK, 1, "layer_input", "residual", 0)
d1r = per_tok_diff(l1r_n, l1r_t0, None, staged=False)
ref1r = l1r_n[:105].float().pow(2).mean().sqrt().item()

print("=== layer 0 attn OUTPUT ===")
print(f"  ref RMS={ref0:.6f}")
summ("attn_output", d0, ref0)
print("=== layer 1 attn INPUT (layer_input) ===")
print(f"  hidden ref RMS={ref1h:.6f}, residual ref RMS={ref1r:.6f}")
summ("hidden_states", d1h, ref1h)
summ("residual", d1r, ref1r)
print("=== per-token correlation ===")
print(f"  corr(l0_attn, l1_hidden) = {torch.corrcoef(torch.stack([d0, d1h]))[0,1].item():.4f}")
print(f"  corr(l0_attn, l1_resid)  = {torch.corrcoef(torch.stack([d0, d1r]))[0,1].item():.4f}")
