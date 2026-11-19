from benchmark import run_one
import csv
from dataclasses import asdict


def main():
    results = []

    # --- Sweep 1: sequence length (main latency/memory crossover) ---
    seq_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152]
    for n in seq_lens:
        for variant in ("naive", "tiled"):
            r = run_one(variant, batch=1, heads=4, seq_len=n, head_dim=64, tile_size=64)
            print(f"[seqlen] N={n:>6} {variant:>6}: status={r.status} "
                  f"lat={r.latency_ms:.3f}ms mem={r.peak_memory_mb:.1f}MB err={r.max_abs_error:.2e}")
            results.append(r)
            if variant == "naive" and r.status == "oom":
                print(f"  -> naive OOM confirmed at N={n}; recording tiled-only for larger N not needed here")

    # --- Sweep 2: tile size sensitivity (tiled only, fixed N) ---
    for n in (1024, 4096, 8192):
        for tile in (16, 32, 64, 128):
            r = run_one("tiled", batch=1, heads=4, seq_len=n, head_dim=64, tile_size=tile)
            print(f"[tilesize] N={n:>6} tile={tile:>4}: status={r.status} "
                  f"lat={r.latency_ms:.3f}ms mem={r.peak_memory_mb:.1f}MB err={r.max_abs_error:.2e}")
            results.append(r)

    # --- Sweep 3: batch size effect on memory ---
    for batch in (1, 4, 8):
        for n in (1024, 4096, 8192, 16384):
            for variant in ("naive", "tiled"):
                r = run_one(variant, batch=batch, heads=4, seq_len=n, head_dim=64, tile_size=64)
                print(f"[batch] B={batch} N={n:>6} {variant:>6}: status={r.status} "
                      f"lat={r.latency_ms:.3f}ms mem={r.peak_memory_mb:.1f}MB err={r.max_abs_error:.2e}")
                results.append(r)

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nwrote {len(results)} rows to results.csv")


if __name__ == "__main__":
    main()
