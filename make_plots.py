import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_num(v):
    try:
        x = float(v)
        return x if x == x else None  # filter NaN
    except (TypeError, ValueError):
        return None


def graph1_latency_vs_n():
    rows = load_rows("results.csv")
    by_variant = {"naive": {}, "tiled": {}}
    for r in rows:
        if r["batch"] != "1" or r["tile_size"] != "64":
            continue
        n = int(r["seq_len"])
        lat = to_num(r["latency_ms"])
        if lat is not None:
            by_variant[r["variant"]][n] = lat

    plt.figure(figsize=(6, 5))
    for variant, marker in (("naive", "o"), ("tiled", "s")):
        xs = sorted(by_variant[variant])
        ys = [by_variant[variant][x] for x in xs]
        plt.plot(xs, ys, marker=marker, label=variant)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length N")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs. sequence length")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph1_latency_vs_n.png", dpi=150)
    plt.close()


def graph2_memory_vs_n():
    rows = load_rows("results.csv") + load_rows("results_isolated.csv") + load_rows("results_tiled_ceiling.csv")
    by_variant = {"naive": {}, "tiled": {}}
    for r in rows:
        if r.get("batch", "1") != "1" or r.get("tile_size", "64") != "64":
            continue
        n = int(r["seq_len"])
        mem = to_num(r["peak_memory_mb"])
        if mem is not None:
            by_variant[r["variant"]][n] = mem

    oom_n_naive = min((int(r["seq_len"]) for r in rows
                        if r["variant"] == "naive" and r["status"] == "oom"
                        and r.get("batch", "1") == "1"), default=None)

    plt.figure(figsize=(6, 5))
    for variant, marker in (("naive", "o"), ("tiled", "s")):
        xs = sorted(by_variant[variant])
        ys = [by_variant[variant][x] for x in xs]
        plt.plot(xs, ys, marker=marker, label=variant)
    if oom_n_naive:
        plt.axvline(oom_n_naive, color="red", linestyle="--", alpha=0.6,
                     label=f"naive OOM (N={oom_n_naive})")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence length N")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Peak memory vs. sequence length")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph2_memory_vs_n.png", dpi=150)
    plt.close()


def graph3_tile_size_sweep():
    rows = load_rows("results.csv")
    by_n = {}
    for r in rows:
        if r["variant"] != "tiled" or r["batch"] != "1":
            continue
        n = int(r["seq_len"])
        tile = int(r["tile_size"])
        lat = to_num(r["latency_ms"])
        if lat is not None and n in (1024, 4096, 8192):
            by_n.setdefault(n, {})[tile] = lat

    plt.figure(figsize=(6, 5))
    for n, marker in zip(sorted(by_n), ("o", "s", "^")):
        tiles = sorted(by_n[n])
        plt.plot(tiles, [by_n[n][t] for t in tiles], marker=marker, label=f"N={n}")
    plt.xscale("log", base=2)
    plt.xlabel("K/V tile size")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs. tile size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph3_tile_size_sweep.png", dpi=150)
    plt.close()


def graph4_batch_size_effect():
    rows = load_rows("results.csv")
    by_batch = {}
    for r in rows:
        if r["variant"] != "tiled" or r["tile_size"] != "64":
            continue
        batch = int(r["batch"])
        n = int(r["seq_len"])
        mem = to_num(r["peak_memory_mb"])
        if mem is not None:
            by_batch.setdefault(batch, {})[n] = mem

    plt.figure(figsize=(6, 5))
    for batch, marker in zip(sorted(by_batch), ("o", "s", "^")):
        ns = sorted(by_batch[batch])
        plt.plot(ns, [by_batch[batch][n] for n in ns], marker=marker, label=f"batch={batch}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length N")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Tiled attention: memory vs. N, by batch size")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graph4_batch_size_effect.png", dpi=150)
    plt.close()


def _load_merged(variant_filter=None):
    """Load results.csv then overlay the isolated/ceiling sweeps, which were
    collected after a memory-measurement bug in results.csv was fixed; later
    files win on (variant, seq_len, batch, tile_size) collisions."""
    merged = {}
    for path in ("results.csv", "results_isolated.csv", "results_tiled_ceiling.csv"):
        for r in load_rows(path):
            key = (r["variant"], r["seq_len"], r.get("batch", "1"), r.get("tile_size", "64"))
            merged[key] = r
    return list(merged.values())


def table1_main_comparison():
    rows = _load_merged()
    by_n = {}
    for r in rows:
        if r["batch"] != "1" or r["tile_size"] != "64":
            continue
        n = int(r["seq_len"])
        by_n.setdefault(n, {})[r["variant"]] = r

    lines = ["| N | naive latency (ms) | tiled latency (ms) | speedup | naive mem (MB) | tiled mem (MB) | mem reduction |",
              "|---|---|---|---|---|---|---|"]
    for n in sorted(by_n):
        naive_r = by_n[n].get("naive")
        tiled_r = by_n[n].get("tiled")
        nl = to_num(naive_r["latency_ms"]) if naive_r else None
        tl = to_num(tiled_r["latency_ms"]) if tiled_r else None
        nm = to_num(naive_r["peak_memory_mb"]) if naive_r else None
        tm = to_num(tiled_r["peak_memory_mb"]) if tiled_r else None

        nl_s = f"{nl:.3f}" if nl is not None else "OOM"
        tl_s = f"{tl:.3f}" if tl is not None else "OOM"
        speedup = f"{nl / tl:.2f}x" if (nl is not None and tl is not None) else "-"
        nm_s = f"{nm:.1f}" if nm is not None else "OOM"
        tm_s = f"{tm:.1f}" if tm is not None else "OOM"
        reduction = f"{nm / tm:.2f}x" if (nm is not None and tm is not None) else "-"

        lines.append(f"| {n} | {nl_s} | {tl_s} | {speedup} | {nm_s} | {tm_s} | {reduction} |")

    with open("results_table.md", "w") as f:
        f.write("## Table 1 -- Main comparison\n\n")
        f.write("\n".join(lines))
        f.write("\n\n## Table 2 -- Correctness proof (max abs error, tiled vs naive)\n\n")
        f.write("| N | tile size | max abs error |\n|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (int(r["seq_len"]), int(r["tile_size"]))):
            if r["variant"] == "tiled":
                err = to_num(r["max_abs_error"])
                if err is not None:
                    f.write(f"| {r['seq_len']} | {r['tile_size']} | {err:.3e} |\n")
    print("wrote results_table.md")


if __name__ == "__main__":
    graph1_latency_vs_n()
    graph2_memory_vs_n()
    graph3_tile_size_sweep()
    graph4_batch_size_effect()
    table1_main_comparison()
    print(f"all plots written to {OUT_DIR}/")
