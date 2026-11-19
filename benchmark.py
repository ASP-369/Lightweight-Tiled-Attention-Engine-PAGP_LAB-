import csv
import gc
import sys
from dataclasses import dataclass, asdict

import torch

from naive_attention import naive_attention
from tiled_kernel import tiled_attention_triton


@dataclass
class RunResult:
    variant: str
    seq_len: int
    tile_size: int
    batch: int
    heads: int
    head_dim: int
    latency_ms: float
    peak_memory_mb: float
    max_abs_error: float
    status: str  # "ok" or "oom"


def _make_qkv(batch, heads, seq_len, head_dim, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)
    return q, k, v


def _time_cuda(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def run_one(variant: str, batch: int, heads: int, seq_len: int, head_dim: int,
            tile_size: int = 64, device: str = "cuda") -> RunResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    try:
        q, k, v = _make_qkv(batch, heads, seq_len, head_dim, device)

        if variant == "naive":
            fn = lambda: naive_attention(q, k, v)
        elif variant == "tiled":
            fn = lambda: tiled_attention_triton(q, k, v, block_m=64, block_n=tile_size)
        else:
            raise ValueError(f"unknown variant: {variant}")

        out = fn()
        latency_ms = _time_cuda(fn)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Correctness check runs after latency/memory are already captured,
        # and in its own try: at large N, computing the naive reference can
        # itself OOM even when the variant under test (tiled) comfortably
        # fits, and that must not be reported as the tested variant's OOM.
        max_err = 0.0
        if variant != "naive":
            try:
                torch.cuda.empty_cache()
                ref = naive_attention(q, k, v)
                max_err = (out - ref).abs().max().item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                max_err = float("nan")  # reference too large to compute at this N

        return RunResult(variant, seq_len, tile_size, batch, heads, head_dim,
                          latency_ms, peak_mb, max_err, "ok")

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return RunResult(variant, seq_len, tile_size, batch, heads, head_dim,
                          float("nan"), float("nan"), float("nan"), "oom")


def run_sweep(seq_lens, out_csv, batch=1, heads=4, head_dim=64, tile_size=64):
    results = []
    for n in seq_lens:
        for variant in ("naive", "tiled"):
            r = run_one(variant, batch, heads, n, head_dim, tile_size)
            status_str = "OOM" if r.status == "oom" else f"{r.latency_ms:.3f} ms, {r.peak_memory_mb:.1f} MB"
            print(f"N={n:>6} {variant:>6}: {status_str}")
            results.append(r)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nwrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    small_sweep = [64, 128, 256, 512, 1024]
    run_sweep(small_sweep, "results_sanity.csv")
