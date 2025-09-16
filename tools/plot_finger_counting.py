#!/usr/bin/env python3
# plot_finger_fraction_like_sweep_reduce.py
import argparse, os, sys, glob, re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def is_param_dir(name: str, param: str) -> bool:
    # Matches "<PARAM>=<VALUE>" with anything after "="
    return name.startswith(f"{param}=") and "=" in name

def parse_param_value(dirname: str, param: str) -> str:
    # Returns the <VALUE> from "<PARAM>=<VALUE>"
    try:
        p, v = dirname.split("=", 1)
        if p != param:
            raise ValueError
        return v
    except Exception:
        return dirname  # fallback

def find_latest_tsv(run_dir: str, tsv_glob: str) -> Optional[str]:
    # Recursively find TSVs by glob (default **/*.tsv) and take the newest by mtime
    paths = glob.glob(os.path.join(run_dir, tsv_glob), recursive=True)
    paths = [p for p in paths if p.lower().endswith(".tsv") and os.path.isfile(p)]
    if not paths: 
        return None
    return max(paths, key=os.path.getmtime)

def coerce_bool_series(s: pd.Series) -> pd.Series:
    # Accept True/False, 1/0, "true"/"false", "t"/"f", etc.
    x = s.astype(str).str.strip().str.lower()
    truthy = {"true","t","1","yes","y"}
    falsy  = {"false","f","0","no","n",""}
    out = []
    for val in x:
        if val in truthy: out.append(1)
        elif val in falsy: out.append(0)
        else:
            # try numeric
            try:
                out.append(1 if float(val)!=0 else 0)
            except:
                out.append(0)
    return pd.Series(out, index=s.index, dtype="int64")

def compute_windowed_fraction(df: pd.DataFrame, window: int) -> pd.Series:
    vals = coerce_bool_series(df["used_finger_counting"])
    # rolling mean over all rows; denominator is the window length
    return vals.rolling(window=window, min_periods=1).mean()

def main():
    ap = argparse.ArgumentParser(
        description="Plot windowed fraction of `used_finger_counting` from sweep outputs."
    )
    ap.add_argument("--sweep-dir", required=True, help="Sweep root containing <PARAM>=<VALUE> subdirs")
    ap.add_argument("--param", required=True, help="Sweep parameter name (e.g., addition_start_step)")
    ap.add_argument("--glob", default="**/*.tsv", help="Glob to find TSVs within each run dir (default: **/*.tsv)")
    ap.add_argument("--window", type=int, default=2000, help="Rolling window size (rows). Default 2000")
    ap.add_argument("--label", default=None, help="Title for the plot")
    ap.add_argument("--output", default=None, help="Optional path to save PNG/SVG; otherwise show()")
    ap.add_argument("--export-tsv", action="store_true", help="Also write per-run *_finger_fraction.tsv series")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir
    param     = args.param
    tsv_glob  = args.glob
    window    = args.window

    if not os.path.isdir(sweep_dir):
        sys.exit(f"Not a directory: {sweep_dir}")

    # Discover run subdirs like "<PARAM>=<VALUE>"
    subdirs = [
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and is_param_dir(d, param)
    ]
    if not subdirs:
        sys.exit(f"No run subdirectories matching '{param}=*' under {sweep_dir}")

    plt.figure()
    plotted_any = False

    for d in sorted(subdirs):
        run_dir = os.path.join(sweep_dir, d)
        value_str = parse_param_value(d, param)
        tsv_path = find_latest_tsv(run_dir, tsv_glob)

        if tsv_path is None:
            print(f"[warn] No TSVs in {run_dir} matching {tsv_glob}; skipping.", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(tsv_path, sep="\t", engine="python")
        except Exception as e:
            print(f"[warn] Failed to read {tsv_path}: {e}; skipping.", file=sys.stderr)
            continue

        if "used_finger_counting" not in df.columns:
            print(f"[warn] {tsv_path} missing 'used_finger_counting'; skipping.", file=sys.stderr)
            continue

        frac = compute_windowed_fraction(df, window)

        if args.export_tsv:
            out_tsv = os.path.splitext(tsv_path)[0] + "_finger_fraction.tsv"
            pd.DataFrame({"finger_fraction": frac}).to_csv(out_tsv, sep="\t", index=False)

        plt.plot(frac.values, label=str(value_str))
        plotted_any = True

    if not plotted_any:
        sys.exit("Nothing plotted (no usable files).")

    plt.xlabel("Row index")
    plt.ylabel(f"Finger-count fraction (window={window})")
    plt.title(args.label if args.label else f"Windowed finger-count fraction by {param}")
    plt.legend(title=param, loc="best")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"[ok] Saved plot to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
