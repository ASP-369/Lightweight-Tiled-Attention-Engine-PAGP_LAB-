# Lightweight Tiled Attention Engine — Project Plan

*PAGP Lab External (EL) Project | Target hardware: Lenovo LOQ, Windows 11, RTX 4050 (6GB VRAM), 24GB RAM, CUDA 11.8*

## 1. Project Overview

**Goal:** Implement and compare two GPU attention mechanisms — conventional (materialized) attention and a tiled, online-softmax implementation — and use the comparison to answer a concrete research question about memory-compute tradeoffs.

**Research question:** At what sequence lengths does tiled attention outperform materialized attention in latency, and how does memory reduction scale with sequence length?

**Core claim to prove (framed honestly):** Tiled attention trades a small constant-factor overhead at short sequences for asymptotically better memory scaling (O(N) vs O(N²)), letting it run at sequence lengths where materialized attention runs out of memory entirely.

**Deliverables:**

1. Two correct, verified attention implementations (naive + tiled)
2. A measurement harness producing CSV data across multiple sweeps
3. Tables and graphs (latency, memory, correctness, tile-size sensitivity)
4. A live interactive demo (text input, attention heatmap, sequence-length slider with live latency/memory readout)
5. A short written report explaining the online-softmax algorithm and the results

## 2. System Specifications

| Component | Spec | Implication |
| --- | --- | --- |
| GPU | RTX 4050 Laptop, 6 GB VRAM, Ada Lovelace, compute capability 8.9 | Fully supported by modern PyTorch/Triton/Numba; 6GB sets your OOM crossover point (\~N=32k–45k for single-head fp32, lower with batching) |
| CPU/RAM | 24 GB system RAM | Plenty for data prep, CSV handling, plotting; not a bottleneck |
| OS | Windows 11 | Native Triton support is weaker than Linux; plan accounts for this (Section 3) |
| System CUDA | 11.8 | Does not block you — PyTorch's pip wheels bundle their own CUDA runtime, so the system toolkit version is mostly irrelevant for PyTorch itself. Only matters if you compile raw CUDA C++ code directly (not part of this plan) |

## 3. Toolchain Decision

**Chosen stack:** PyTorch (CUDA 11.8 build) for the reference implementation and measurement, **Triton** for the tiled GPU kernel, with **Numba CUDA** as a fallback if Triton setup stalls on Windows.

**Why this combination:**

- Triton is Python-based, compiles to real GPU kernels, and is what the original FlashAttention research prototyped in — legitimate to cite.
- Raw CUDA C++ is explicitly avoided: with a 1–2 week timeline, debugging shared-memory tiling bugs in raw `.cu` files is the single highest risk to not finishing. Triton gets you the same tiling/SRAM concepts with far less failure surface.
- Numba CUDA is the safety net since it installs reliably on native Windows and matches a tool already named in your syllabus.

**Critical Day 1 decision — Windows native vs WSL2:** Triton's official pip packages have historically been more reliable on Linux than native Windows. Since Triton support quality on Windows changes often, Day 1's first task is a quick go/no-go test:

1. Try native Windows first: `pip install torch --index-url https://download.pytorch.org/whl/cu118` then `pip install triton` (or check if your torch version bundles a Windows-compatible Triton already).
2. Run a 5-line "vector add" Triton kernel as a smoke test.
3. **If it fails or is unstable within the first hour:** switch immediately to **WSL2 (Ubuntu)**. Your NVIDIA Windows driver already supports CUDA passthrough into WSL2 — you do NOT need to install a separate Linux driver, only the CUDA toolkit inside WSL. This is the standard, reliable fallback and still uses your RTX 4050 directly.
4. **If both stall beyond hour 2:** drop to the Numba CUDA path for the tiled kernel instead of Triton, and note this as a scoping decision in your report (this is a legitimate and explainable engineering tradeoff, not a failure).

Do not spend more than half of Day 1 on this decision — pick a path and move on.

## 4. System Architecture

```
[Input: text or synthetic Q,K,V tensors]
        |
        v
[Tokenizer (simple whitespace/BPE)] --> [Q, K, V matrices]
        |
        +--------------------+--------------------+
        v                                         v
[Naive Attention (PyTorch)]              [Tiled Attention (Triton kernel)]
  - QK^T (full matrix)                      - Loop over K/V tiles
  - softmax (materialized)                  - Online softmax (running max + sum)
  - x V                                      - Accumulate output incrementally
        |                                         |
        +--------------------+--------------------+
                             v
                [Measurement Harness]
          - CUDA event timing (latency)
          - torch.cuda.max_memory_allocated (peak memory)
          - Correctness check (max abs error vs reference)
                             v
              [CSV results] --> [Tables + Graphs]
                             v
                    [Gradio Live Demo]
          - Attention heatmap for entered text
          - Sequence-length slider -> live latency/memory chart
          - Pre-cached sweep data for instant slider response
```

## 5. Day-by-Day Execution Plan (10–12 working days)

| Day | Task | Output |
| --- | --- | --- |
| 1 | Toolchain go/no-go test (Section 3). Confirm PyTorch CUDA 11.8 build works, Triton or WSL2 fallback decided. Run trivial Triton/Numba kernel successfully. | Working dev environment |
| 2 | Implement naive attention in pure PyTorch (QK^T, softmax, xV). Verify against `torch.nn.functional.scaled_dot_product_attention`, max error < 1e-5. | `naive_attention.py`, correctness log |
| 3 | Derive the online-softmax recurrence on paper (see Appendix). Implement a pure-Python looped tiled version (no GPU kernel yet) to validate the algorithm logic itself. Verify numerically matches naive output. | `tiled_reference.py`, correctness log |
| 4–5 | Port the tiled version to a real Triton kernel (or Numba CUDA fallback). Focus on correctness first, not speed. Debug against the Day 3 reference. | `tiled_kernel.py`, passing correctness tests |
| 6 | Build the measurement harness: CUDA event timing for latency, `torch.cuda.max_memory_allocated()` for peak memory (reset before each run). Run first small sweep (N = 64 to 1024) to sanity-check the harness itself. | `benchmark.py`, first CSV |
| 7 | Run the full experiment matrix: sequence-length sweep (64 → as high as the GPU allows, pushing until naive attention OOMs), tile-size sweep (16/32/64/128), batch-size sweep (1, 4, 8). Record the exact N where naive attention crashes. | Full `results.csv` |
| 8 | Build tables and all graphs (Section 7) from the CSV using matplotlib/pandas. | `plots/` folder, `results_table.md` |
| 9 | Build the Gradio live demo: text box -> tokenize -> attention heatmap; slider for N -> live latency/memory readout (using pre-cached sweep data so the slider feels instant). | `demo_app.py` |
| 10 | Full dry run of the demo end-to-end. Record a backup video in case of live failure on presentation day. Start the written report. | Backup video, report draft |
| 11–12 (buffer) | Fix issues found in dry run, finish report (algorithm explanation + honest results discussion), prepare poster/slides, rehearse explaining *why* tiling reduces memory and the online-softmax trick. | Final report, poster, rehearsed demo |

## 6. Experiment Design

Run all experiments with the correctness check enabled (max abs error vs. PyTorch's built-in attention) so every data point also doubles as a correctness proof.

| Sweep | Values | Purpose |
| --- | --- | --- |
| Sequence length (N) | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, (push further until OOM) | Main latency/memory crossover data |
| Tile size | 16, 32, 64, 128 | Shows tiled kernel's sensitivity to block size — a second, richer result axis |
| Batch size | 1, 4, 8 | Shows how memory pressure scales with batch; likely shifts the OOM point to a lower, faster-to-demo N |
| Head dimension | 32, 64, 128 (optional, time-permitting) | Shows arithmetic-intensity effects |

For each run: record wall-clock latency (averaged over multiple repeated runs after a warm-up run), peak memory allocated, and max absolute numerical error vs. the reference.

## 7. Metrics, Tables, and Graphs (final deliverable checklist)

1. **Table 1 — Main comparison**: N, naive latency (ms), tiled latency (ms), speedup ratio, naive peak memory (MB), tiled peak memory (MB), memory reduction ratio.
2. **Graph 1 — Latency vs. N** (log-log, two lines: naive, tiled) — shows the crossover point.
3. **Graph 2 — Peak memory vs. N** (log-linear) — shows O(N²) vs O(N) divergence; mark the N where naive OOMs.
4. **Graph 3 — Tile-size sweep**: latency vs. tile size, for 2–3 fixed N values.
5. **Graph 4 — Batch-size effect**: peak memory vs. N, one line per batch size.
6. **Table 2 — Correctness proof**: max absolute error between tiled and naive outputs, across all tested N (should stay near-zero, proving the tiled kernel computes the same result).

## 8. Live Demo Design

- **Tool:** Gradio (fastest path to a slider + chart UI with minimal frontend code).
- **Interaction 1:** Text box -> simple tokenizer -> run both attention variants -> display an attention-weight heatmap (token labels on both axes).
- **Interaction 2:** Sequence-length slider -> instantly display latency and peak memory for naive vs. tiled at that N, pulled from the **pre-cached** sweep CSV (not a live re-run) so the demo never stalls waiting on GPU kernel launches in front of an audience.
- **Interaction 3 (stretch):** Toggle button to run a *live* side-by-side timing at a chosen N, for the audience to see real-time numbers once, as a trust-building moment — but only after the cached version is working and reliable.
- **Backup plan:** a recorded screen-capture video of the full demo, ready to play if the live GPU demo has any issue on presentation day.

## 9. Risk Mitigation

| Risk | Mitigation |
| --- | --- |
| Triton install fails on Windows | WSL2 fallback (Section 3); decide within Day 1 |
| Tiled kernel numerically wrong | Day 3's pure-Python reference catches algorithm bugs before any GPU kernel is written |
| Naive attention OOMs mid-sweep unexpectedly | Wrap sweep loop in try/except, log the OOM as data (it's a real result, not an error) |
| Live demo fails on presentation day | Pre-cached results + recorded backup video |
| Running out of time before Day 10 | Cut the batch-size and head-dimension sweeps first (Section 6) — the N sweep and memory graph are the non-negotiable core result |

## 10. Framing the Results Honestly (important for the panel)

On a 6GB laptop GPU, naive attention may still be *faster* than tiled attention at small N (the tiling loop has overhead that only pays off once O(N²) memory traffic dominates) — this matches the real FlashAttention paper's own findings and is not a failure. Present the conclusion as:

> "Tiled attention incurs a small constant-factor overhead at short sequences but scales memory linearly instead of quadratically, allowing it to run at sequence lengths where materialized attention runs out of memory entirely. On our hardware, materialized attention fails at N ≈ \[X\], while tiled attention continues to run."

This is a true, defensible, and research-flavored conclusion regardless of exactly where your measured crossover point lands.

## Appendix: Online Softmax Recurrence

For each new K/V tile, maintain a running max `m` and running sum `l`, and update the output accumulator `O` incrementally:

```
For each tile j of K, V:
  S_j = Q @ K_j^T * scale              # partial attention scores
  m_new = max(m_old, rowmax(S_j))      # update running max
  P_j = exp(S_j - m_new)               # stabilized exponentials
  l_new = exp(m_old - m_new) * l_old + rowsum(P_j)   # update running sum
  O_new = exp(m_old - m_new) * O_old + P_j @ V_j      # rescale old output, add new contribution
  m_old, l_old, O_old = m_new, l_new, O_new

Final: O = O_old / l_old
```

This avoids ever materializing the full N×N attention matrix — only one tile of S is in memory at a time, which is the entire basis for the memory-reduction result in Section 7, Graph 2.
