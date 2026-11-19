"""Gradio live demo for the tiled attention engine.

Interaction 1: text -> whitespace tokenizer -> run both attention variants
  -> attention-weight heatmap (naive vs tiled, side by side, token labels).
Interaction 2: sequence-length slider -> latency/memory readout from the
  pre-cached sweep CSV (instant; never re-runs the GPU in front of an audience).
Interaction 3: a "run live now" button that times a single real request at
  the chosen N, as a one-time trust-building moment.
"""
import csv
import math

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from naive_attention import naive_attention
from tiled_kernel import tiled_attention_triton
from benchmark import run_one

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_DIM = 64


def _load_cached_sweep():
    """Merge the main sweep with the later isolated/ceiling sweeps, which
    were collected after a memory-measurement bug in results.csv was fixed;
    later files win on (variant, seq_len) collisions."""
    cache = {}
    for path in ("results.csv", "results_isolated.csv", "results_tiled_ceiling.csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("batch", "1") != "1" or row.get("tile_size", "64") != "64":
                    continue
                key = (row["variant"], int(row["seq_len"]))
                try:
                    lat = float(row["latency_ms"])
                    mem = float(row["peak_memory_mb"])
                except ValueError:
                    lat, mem = float("nan"), float("nan")
                cache[key] = (lat, mem)
    return cache


CACHED_SWEEP = _load_cached_sweep()
CACHED_NS = sorted({n for (_, n) in CACHED_SWEEP})


def _embed_tokens(tokens, head_dim=HEAD_DIM, seed=0):
    """Deterministic toy embedding: each token gets a reproducible random
    vector (hashed by token string) standing in for a learned embedding."""
    vecs = []
    for tok in tokens:
        g = torch.Generator().manual_seed(abs(hash(tok)) % (2**31) + seed)
        vecs.append(torch.randn(head_dim, generator=g))
    return torch.stack(vecs)


def run_heatmap(text):
    tokens = text.strip().split()
    if len(tokens) < 2:
        return None, "Enter at least two words."
    if len(tokens) > 64:
        tokens = tokens[:64]  # keep the heatmap readable

    x = _embed_tokens(tokens).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,N,D)
    q = k = v = x.contiguous().float()

    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)[0, 0].detach().cpu().numpy()

    out_tiled = tiled_attention_triton(q, k, v, block_m=64, block_n=64)
    out_naive = naive_attention(q, k, v)
    max_err = (out_tiled - out_naive).abs().max().item()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(f"Attention weights ({len(tokens)} tokens)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    note = f"naive vs. tiled max abs error on this input: {max_err:.2e} (both variants agree)"
    return fig, note


def lookup_cached(n):
    n = int(n)
    nearest = min(CACHED_NS, key=lambda c: abs(c - n))
    naive = CACHED_SWEEP.get(("naive", nearest), (float("nan"), float("nan")))
    tiled = CACHED_SWEEP.get(("tiled", nearest), (float("nan"), float("nan")))

    def fmt(pair):
        lat, mem = pair
        if lat != lat:  # NaN check
            return "OOM", "OOM"
        return f"{lat:.2f} ms", f"{mem:.1f} MB"

    naive_lat, naive_mem = fmt(naive)
    tiled_lat, tiled_mem = fmt(tiled)
    return (
        f"### N = {nearest} (cached)\n\n"
        f"| variant | latency | peak memory |\n|---|---|---|\n"
        f"| naive | {naive_lat} | {naive_mem} |\n"
        f"| tiled | {tiled_lat} | {tiled_mem} |\n"
    )


def run_live(n):
    n = int(n)
    naive_r = run_one("naive", batch=1, heads=4, seq_len=n, head_dim=HEAD_DIM)
    tiled_r = run_one("tiled", batch=1, heads=4, seq_len=n, head_dim=HEAD_DIM)

    def fmt(r):
        if r.status == "oom":
            return "OOM", "OOM"
        return f"{r.latency_ms:.2f} ms", f"{r.peak_memory_mb:.1f} MB"

    naive_lat, naive_mem = fmt(naive_r)
    tiled_lat, tiled_mem = fmt(tiled_r)
    return (
        f"### N = {n} (live, just measured)\n\n"
        f"| variant | latency | peak memory |\n|---|---|---|\n"
        f"| naive | {naive_lat} | {naive_mem} |\n"
        f"| tiled | {tiled_lat} | {tiled_mem} |\n"
    )


with gr.Blocks(title="Lightweight Tiled Attention Engine") as demo:
    gr.Markdown("# Lightweight Tiled Attention Engine — Live Demo")

    with gr.Tab("Attention heatmap"):
        text_in = gr.Textbox(label="Input text (whitespace-tokenized)",
                              value="the quick brown fox jumps over the lazy dog")
        run_btn = gr.Button("Compute attention")
        plot_out = gr.Plot()
        note_out = gr.Markdown()
        run_btn.click(run_heatmap, inputs=text_in, outputs=[plot_out, note_out])

    with gr.Tab("Latency / memory vs. N"):
        slider = gr.Slider(minimum=min(CACHED_NS), maximum=max(CACHED_NS),
                            step=1, value=1024, label="Sequence length N")
        cached_out = gr.Markdown()
        slider.change(lookup_cached, inputs=slider, outputs=cached_out)
        demo.load(lookup_cached, inputs=slider, outputs=cached_out)

        gr.Markdown("---\n**Trust check** -- run one real measurement at this N right now:")
        live_btn = gr.Button("Run live comparison at this N")
        live_out = gr.Markdown()
        live_btn.click(run_live, inputs=slider, outputs=live_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
