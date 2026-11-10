#!/usr/bin/env python3
"""Plot per-query time vs nprobe for the IVF-CPU baseline and Rummy on GPU.

Usage: python3 scripts/plot_cpu_vs_gpu.py [results_dir]
Reads  results/sift1m.csv (CPU) and results/sift1m_gpu.csv (GPU)
Writes results/sift1m_cpu_vs_gpu.png

The three lines are told apart by line style, marker shape and an end-of-line
label, not by color alone (the chart stays readable for color-vision deficiency
and in grayscale print).
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Each series: color (validated categorical slots 1-3), line style, marker, label.
SERIES = [
    ("cpu", "#2a78d6", "-", "o", "CPU (i7-9700, 8 thr)"),
    ("gpu512", "#eb6834", (0, (6, 3)), "s", "GPU, pool 512 MiB"),
    ("gpu128", "#1baf7a", (0, (1.5, 2.5)), "^", "GPU, pool 128 MiB"),
]


def load():
    data = {}
    with open(os.path.join(RES, "sift1m.csv")) as f:
        for r in csv.DictReader(f):
            data.setdefault(("cpu", int(r["bs"])), []).append((int(r["nprobe"]), float(r["query_ms"])))
    with open(os.path.join(RES, "sift1m_gpu.csv")) as f:
        for r in csv.DictReader(f):
            key = ("gpu" + r["pool_mib"], int(r["bs"]))
            data.setdefault(key, []).append((int(r["nprobe"]), float(r["query_ms"])))
    for k in data:
        data[k].sort()
    return data


def spread_labels(ys_log, min_gap):
    """Nudge label positions (log10 space) apart so they never overlap."""
    order = sorted(range(len(ys_log)), key=lambda i: ys_log[i])
    pos = [ys_log[i] for i in order]
    for _ in range(50):
        moved = False
        for j in range(1, len(pos)):
            gap = pos[j] - pos[j - 1]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                pos[j - 1] -= shift
                pos[j] += shift
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys_log)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def main():
    data = load()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, facecolor=SURFACE)
    ymin, ymax = 0.012, 8.0
    for ax, bs in zip(axes, (8, 2048)):
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 41)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"batch = {bs}", color=INK, fontsize=12, loc="left")
        ax.set_xlabel("nprobe (clusters searched, of 256)")
        ends = []
        for key, color, ls, marker, label in SERIES:
            pts = data[(key, bs)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, marker=marker, markersize=6.5,
                    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    solid_capstyle="round", dash_capstyle="round", label=label)
            ends.append((xs[-1], ys[-1], marker, label))
        # Direct labels at the right end of each line, nudged apart if they would collide.
        span = math.log10(ymax) - math.log10(ymin)
        placed = spread_labels([math.log10(e[1]) for e in ends], min_gap=span * 0.085)
        glyph = {"o": "●", "s": "■", "^": "▲"}
        short = {"o": "CPU", "s": "GPU 512", "^": "GPU 128"}
        for (x, y, marker, label), ly in zip(ends, placed):
            ax.annotate(f"{glyph[marker]} {short[marker]}  {y:.2f} ms", xy=(x, y),
                        xytext=(x + 1.2, 10 ** ly), textcoords="data", va="center", ha="left",
                        fontsize=8.8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6, shrinkA=0, shrinkB=4))
        ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel("time per query (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945),
               handlelength=3.6, fontsize=9.5)
    fig.suptitle("SIFT1M, IVF256,Flat, top-10: IVF-CPU baseline vs Rummy on GPU", color=INK, fontsize=13, y=0.995)
    fig.text(0.5, 0.012,
             "CPU: lab PC, i7-9700, 8 threads.  GPU: Kaggle Tesla T4, Rummy main branch, 4 threads.  "
             "pool = GPU memory Rummy may use (base set ~488 MiB).  Different machines: compare trends, not absolute values.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = os.path.join(RES, "sift1m_cpu_vs_gpu.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
