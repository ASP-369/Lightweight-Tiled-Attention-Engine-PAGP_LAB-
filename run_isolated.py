"""Run a single (variant, N, batch, tile_size) benchmark point in its own
process. A failed cudaMalloc for one variant can leave the caching allocator
fragmented for the rest of that process, so OOM results near the memory
ceiling are only trustworthy when each point starts from a clean CUDA context.
"""
import csv
import json
import subprocess
import sys
from dataclasses import asdict

_WORKER = """
import sys, json
from benchmark import run_one
r = run_one(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
from dataclasses import asdict
print("RESULT_JSON:" + json.dumps(asdict(r)))
"""


def run_isolated(variant, batch, heads, seq_len, head_dim, tile_size=64):
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, variant, str(batch), str(heads), str(seq_len), str(head_dim), str(tile_size)],
        capture_output=True, text=True, timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    # process crashed (e.g. hard OOM kill) rather than catching OutOfMemoryError cleanly
    if proc.stderr.strip():
        print(f"  [stderr] {proc.stderr.strip()[-800:]}", file=sys.stderr)
    return {
        "variant": variant, "seq_len": seq_len, "tile_size": tile_size,
        "batch": batch, "heads": heads, "head_dim": head_dim,
        "latency_ms": float("nan"), "peak_memory_mb": float("nan"),
        "max_abs_error": float("nan"), "status": "oom",
    }


def main():
    results = []
    seq_lens = [65536, 98304, 131072, 196608]
    fieldnames = ["variant", "seq_len", "tile_size", "batch", "heads", "head_dim",
                  "latency_ms", "peak_memory_mb", "max_abs_error", "status"]
    with open("results_tiled_ceiling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in seq_lens:
            r = run_isolated("tiled", batch=1, heads=4, seq_len=n, head_dim=64, tile_size=64)
            print(f"[isolated] N={n:>6} tiled: status={r['status']} "
                  f"lat={r['latency_ms']:.3f}ms mem={r['peak_memory_mb']:.1f}MB err={r['max_abs_error']:.2e}")
            results.append(r)
            writer.writerow(r)
            f.flush()
    print(f"\nwrote {len(results)} rows to results_tiled_ceiling.csv")


if __name__ == "__main__":
    main()
