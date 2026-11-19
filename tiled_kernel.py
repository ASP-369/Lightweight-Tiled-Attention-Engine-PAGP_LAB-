import math

import torch
import triton
import triton.language as tl

from naive_attention import naive_attention


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_ptr = q_ptr + pid_bh * stride_qh
    k_ptr = k_ptr + pid_bh * stride_kh
    v_ptr = v_ptr + pid_bh * stride_vh
    o_ptr = o_ptr + pid_bh * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale  # (BLOCK_M, BLOCK_N)
        s_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        s = tl.where(s_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=False)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc, mask=o_mask)


def tiled_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """Triton tiled attention with online softmax.

    q, k, v: (batch, heads, seq_len, head_dim), contiguous, fp32, cuda.
    """
    batch, heads, seq_len, head_dim = q.shape
    assert k.shape == q.shape and v.shape == q.shape
    assert q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    block_d = triton.next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)

    grid = (triton.cdiv(seq_len, block_m), batch * heads)

    _attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seq_len, head_dim,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_stages=1,
        num_warps=4,
    )
    return o


def _check_correctness(batch=2, heads=4, seq_len=512, head_dim=64,
                        block_m=64, block_n=64, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    out = tiled_attention_triton(q, k, v, block_m=block_m, block_n=block_n)
    ref = naive_attention(q, k, v)

    max_err = (out - ref).abs().max().item()
    print(f"[tiled_kernel] seq_len={seq_len} block_m={block_m} block_n={block_n} "
          f"head_dim={head_dim} max abs error vs naive: {max_err:.3e}")
    assert max_err < 1e-4, f"correctness check failed: max_err={max_err}"
    return max_err


if __name__ == "__main__":
    # Query block (BLOCK_M) fixed at 64; this is large enough to saturate the
    # SM's shared-memory budget on a 6GB Ada GPU in combination with BLOCK_N=128,
    # so BLOCK_M itself is not part of the tile-size sweep (see tiled_kernel.py notes).
    for n in (64, 128, 256, 512, 1024, 2048):
        for block_n in (16, 32, 64, 128):
            if block_n <= n:
                _check_correctness(seq_len=n, block_m=64, block_n=block_n)
    for hd in (32, 64, 128):
        _check_correctness(seq_len=512, head_dim=hd, block_m=64, block_n=64)
    print("ALL CORRECTNESS CHECKS PASSED")
