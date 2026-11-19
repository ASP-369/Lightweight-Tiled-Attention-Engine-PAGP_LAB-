import math

import torch
import torch.nn.functional as F


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Materialized scaled dot-product attention.

    q, k, v: (batch, heads, seq_len, head_dim)
    Returns: (batch, heads, seq_len, head_dim)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N) -- the O(N^2) materialization
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def _check_correctness(batch=2, heads=4, seq_len=512, head_dim=64, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    out = naive_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)

    max_err = (out - ref).abs().max().item()
    print(f"[naive_attention] seq_len={seq_len} max abs error vs SDPA reference: {max_err:.3e}")
    assert max_err < 1e-5, f"correctness check failed: max_err={max_err}"
    return max_err


if __name__ == "__main__":
    for n in (64, 256, 512, 1024, 2048):
        _check_correctness(seq_len=n)
    print("ALL CORRECTNESS CHECKS PASSED")
