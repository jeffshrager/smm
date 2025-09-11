#!/usr/bin/env python3
"""
plot_sweep.py — plot lines from a "wide" sweep TSV (one column per parameter value).

Input TSV format (from sweep_reduce.py):
    row<TAB><val1><TAB><val2>...
    0  <TAB> y_0_1 <TAB> y_0_2 ...
    1  <TAB> y_1_1 <TAB> y_1_2 ...
    ...

Usage:
  python3 plot_sweep.py path/to/learning_rate_preds.tsv
  python3 plot_sweep.py path/to/learning_rate_preds.tsv --label "Learning rate sweep"
  python3 plot_sweep.py path/to/learning_rate_preds.tsv --out plot.png

Examples:
# Plot with default title (filename)
python3 plot_sweep.py results/lr_sweep/learning_rate_preds.tsv

# Plot with a custom title
python3 plot_sweep.py results/lr_sweep/learning_rate_preds.tsv --label "Learning rate sweep"

# Save instead of showing
python3 plot_sweep.py results/lr_sweep/learning_rate_preds.tsv --label "Learning rate sweep" --out sweep.png

python3 plot_sweep.py results/addstart_sweep/... --label "Addition Start Sweep"

"""

import argparse
import csv
import math
import os
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Plot a sweep wide-TSV as multiple lines.")
    ap.add_argument("tsv", help="Path to wide TSV, e.g., learning_rate_preds.tsv")
    ap.add_argument("--label", default=None,
                    help="Plot title (shown at the top of the graph)")
    ap.add_argument("--xcol", default="row", help="Name of x column (default: row)")
    ap.add_argument("--out", default=None, help="Optional output image path (e.g., plot.png)")
    return ap.parse_args()

def to_float_or_nan(x: str) -> float:
    try:
        if x == "" or x.lower() == "nan":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def read_wide_tsv(path: str, xcol: str):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"Empty TSV: {path}")

    header = rows[0]
    if xcol not in header:
        raise SystemExit(f"'{xcol}' column not found. Columns: {header}")
    x_idx = header.index(xcol)
    col_idxs = [i for i, name in enumerate(header) if i != x_idx]
    col_names = [header[i] for i in col_idxs]

    series = [[] for _ in col_idxs]
    for r in rows[1:]:
        for j, i in enumerate(col_idxs):
            val = r[i] if i < len(r) else ""
            series[j].append(to_float_or_nan(val))

    max_len = max((len(s) for s in series), default=0)
    for s in series:
        if len(s) < max_len:
            s += [float("nan")] * (max_len - len(s))

    xs = list(range(max_len))
    return xs, series, col_names

def main():
    args = parse_args()
    xs, series, col_names = read_wide_tsv(args.tsv, args.xcol)

    fig, ax = plt.subplots()
    for y, label in zip(series, col_names):
        ax.plot(xs, y, label=label)

    ax.set_xlabel(args.xcol)
    ax.set_ylabel("value")

    if args.label:
        ax.set_title(args.label)
    else:
        ax.set_title(os.path.basename(args.tsv))

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ok] saved {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
