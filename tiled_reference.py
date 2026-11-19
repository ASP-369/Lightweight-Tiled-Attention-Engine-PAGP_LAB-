import math

import torch

from naive_attention import naive_attention


def tiled_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_size: int = 128) -> torch.Tensor:
    """Online-softmax tiled attention, looped in plain PyTorch (no custom kernel).

    Validates the algorithm itself: at no point is the full (N, N) score
    matrix materialized -- only one (N, tile_size) tile of scores exists
    at a time.

    q, k, v: (batch, heads, seq_len, head_dim)
    """
    batch, heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    o = torch.zeros_like(q)
    m = torch.full((batch, heads, seq_len, 1), float("-inf"), device=q.device, dtype=q.dtype)
    l = torch.zeros((batch, heads, seq_len, 1), device=q.device, dtype=q.dtype)

    for start in range(0, seq_len, tile_size):
        end = min(start + tile_size, seq_len)
        k_j = k[:, :, start:end, :]
        v_j = v[:, :, start:end, :]

        s_j = (q @ k_j.transpose(-2, -1)) * scale  # (B, H, N, tile)
        m_new = torch.maximum(m, s_j.amax(dim=-1, keepdim=True))
        p_j = torch.exp(s_j - m_new)
        alpha = torch.exp(m - m_new)

        l = alpha * l + p_j.sum(dim=-1, keepdim=True)
        o = alpha * o + p_j @ v_j
        m = m_new

    return o / l


def _check_correctness(batch=2, heads=4, seq_len=512, head_dim=64, tile_size=128, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    out = tiled_attention_reference(q, k, v, tile_size=tile_size)
    ref = naive_attention(q, k, v)

    max_err = (out - ref).abs().max().item()
    print(f"[tiled_reference] seq_len={seq_len} tile_size={tile_size} max abs error vs naive: {max_err:.3e}")
    assert max_err < 1e-4, f"correctness check failed: max_err={max_err}"
    return max_err


if __name__ == "__main__":
    for n in (64, 256, 512, 1024, 2048):
        for tile in (16, 32, 64, 128):
            if tile <= n:
                _check_correctness(seq_len=n, tile_size=tile)
    print("ALL CORRECTNESS CHECKS PASSED")
