#!/usr/bin/env python3
"""
python an1.py --tsv my.tsv --window 200 --x step --outdir .

Moving-window correctness for two operation types ("->" and "+") plotted on the
full timeline. The window is over the last N global rows; within each window
we compute correctness for each operator using only that operator’s rows.

Outputs:
  - Combined PNG of both operators over the full timeline
  - Per-operator PNGs
  - CSV with the rolling series

Usage:
  python smm_window_correctness.py --tsv PATH --window 200 --x step --outdir out/
Options:
  --x {row,step,timestamp}   X-axis to use (default: step if present, else row)
  --no-ffill                 Do NOT forward-fill NaNs (default is to ffill)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OPS = ["->", "+"]

def choose_x(df: pd.DataFrame, x_choice: str | None):
    if x_choice is None:
        x_choice = "step" if "step" in df.columns else "row"
    if x_choice == "row":
        x = pd.RangeIndex(start=0, stop=len(df), step=1)
        x_label = "Row (global)"
    elif x_choice == "step":
        if "step" not in df.columns:
            raise ValueError("Requested x='step' but 'step' column is missing.")
        x = df["step"].values
        x_label = "Step (global)"
    elif x_choice == "timestamp":
        if "timestamp" not in df.columns:
            raise ValueError("Requested x='timestamp' but 'timestamp' column is missing.")
        x = pd.to_datetime(df["timestamp"])
        x_label = "Timestamp"
    else:
        raise ValueError(f"Unknown --x value: {x_choice}")
    return x, x_label, x_choice

def rolling_correctness_full_timeline(df: pd.DataFrame, window: int, ffill: bool):
    """For each operator op:
       mw_op[t] = sum(correct_i * 1{op_i==op} over window) / sum(1{op_i==op} over window).
       If an operator has no rows in a window, value is NaN (optionally ffilled)."""
    df = df.copy()
    df["correct"] = (df["predicted"] == df["target"]).astype(float)

    results = {}
    for op in OPS:
        is_op = (df["operator"] == op).astype(float)
        num = (df["correct"] * is_op).rolling(window=window, min_periods=1).sum()
        den = is_op.rolling(window=window, min_periods=1).sum()
        mw = num / den
        if ffill:
            mw = mw.ffill()
        results[op] = mw
    return pd.DataFrame(results)

def analyze(tsv_path: str, window: int = 100, outdir: str = "analysis_out",
            x_choice: str | None = None, ffill: bool = True):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(tsv_path, sep="\t", engine="python")
    run_id = Path(tsv_path).stem
    x, x_label, x_choice = choose_x(df, x_choice)
    mw_df = rolling_correctness_full_timeline(df, window, ffill)
    mw_df["x"] = x

    # Combined plot
    plt.figure()
    for op in OPS:
        label = "next (->)" if op == "->" else "add (+)"
        plt.plot(mw_df["x"], mw_df[op], label=label)
    plt.ylabel("Moving-window correctness (0.0–1.0)")
    plt.xlabel(x_label)
    plt.title(f"Moving-window correctness over full timeline (window={window})")
    plt.legend()
    plt.tight_layout()

    combined_png = os.path.join(outdir, f'{run_id}.png')
    plt.savefig(combined_png, dpi=150)
    plt.close()

    # Per-operator plots
    per_paths = {}
    for op in OPS:
        plt.figure()
        plt.plot(mw_df["x"], mw_df[op])
        plt.ylabel("Moving-window correctness (0.0–1.0)")
        plt.xlabel(x_label)
        optitle = "next (->)" if op == "->" else "add (+)"
        plt.title(f"{optitle}: full timeline (window={window})")
        plt.tight_layout()
        fname = f"{run_id}_{'next' if op=='->' else 'add'}_timeline.png"
        p = os.path.join(outdir, fname)
        plt.savefig(p, dpi=150)
        plt.close()
        per_paths[op] = p

    # CSV export
    export = pd.DataFrame({
        "x": mw_df["x"],
        "mw_next": mw_df["->"],
        "mw_add": mw_df["+"]
    })
    csv_path = os.path.join(outdir, f'{run_id}.csv')
    export.to_csv(csv_path, index=False)

    return {
        "combined_plot": combined_png,
        "per_operator_plots": {"next": per_paths["->"], "add": per_paths["+"]},
        "csv": csv_path,
        "x_choice": x_choice
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Path to input TSV file")
    ap.add_argument("--window", type=int, default=100, help="Window size in rows (global)")
    ap.add_argument("--outdir", default=".", help="Directory for outputs")
    ap.add_argument("--x", choices=["row", "step", "timestamp"], default=None,
                    help="X-axis to use (default: step if present, else row)")
    ap.add_argument("--no-ffill", action="store_true",
                    help="Do not forward-fill NaNs where an operator has no data in the window")
    args = ap.parse_args()

    info = analyze(
        tsv_path=args.tsv,
        window=args.window,
        outdir=args.outdir,
        x_choice=args.x,
        ffill=not args.no_ffill
    )
    print("Wrote:")
    print("  combined:", info["combined_plot"])
    print("  next    :", info["per_operator_plots"]["next"])
    print("  add     :", info["per_operator_plots"]["add"])
    print("  csv     :", info["csv"])
    print("  x-axis  :", info["x_choice"])

if __name__ == "__main__":
    main()
