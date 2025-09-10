#!/usr/bin/env python3
"""
sweep_reduce.py — aggregate per-run TSVs from a sweep into a single wide TSV.

Given a sweep directory containing subdirectories named like
   <PARAM>=<VALUE>/
each holding a training TSV (created by train.py), this script:

  1) Locates the latest *.tsv file in each run directory.
  2) Applies a user-provided computation to that TSV (edit compute_vector()).
  3) Collects the returned vectors (all the same length) into a new TSV at the
     sweep root named:  <PARAM>_<COMPUTATION>.tsv

The output TSV has one column per parameter value (sorted numerically when possible).
A 'row' index column is added as the first column.

Usage:
  python3 sweep_reduce.py --sweep-dir <DIR> --param <PARAM>
Optional:
  --glob '*.tsv'            # pattern to find per-run TSVs (default: '*.tsv')
  --strict-length           # error if vectors differ in length (default on)
  --allow-truncate          # truncate to min length across runs (disables strict)
  --allow-pad               # pad to max length across runs with '' (disables strict)

Example:

python3 sweep_reduce.py --sweep-dir results/addstart_sweep --param addition_start_step --allow-truncate

"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import List, Tuple, Dict, Any, Optional

# -------------------------
# EDIT THIS COMPUTATION
# -------------------------
def compute_vector(tsv_path: str) -> Tuple[str, List[Any]]:
    """
    Given a single run TSV, return (computation_name, vector).

    Default example:
      - Read the TSV (tab-delimited)
      - Return the 'predicted' column as a list
      - Name the computation 'preds'

    Replace this with your desired computation; just keep the signature.
    """
    comp_name = "preds"
    vector: List[Any] = []

    with open(tsv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # normalize fieldnames lowercased for robust matching
        field_map = {name.lower(): name for name in reader.fieldnames or []}
        if "predicted" not in field_map:
            raise RuntimeError(f"'predicted' column not found in {tsv_path}. "
                               f"Columns: {reader.fieldnames}")
        colname = field_map["predicted"]
        for row in reader:
            vector.append(row.get(colname, ""))

    return comp_name, vector


# -------------------------
# Helpers
# -------------------------
def is_param_dir(name: str, param: str) -> bool:
    # Match "<param>=<value>" exactly at the start
    return name.startswith(f"{param}=")

def parse_param_value(dirname: str, param: str) -> str:
    # Extract the part after the first '='
    try:
        return dirname.split("=", 1)[1]
    except Exception:
        return dirname  # fallback to raw name

def numeric_key(x: str):
    # Try to sort numerically if possible; else string sort
    try:
        return float(x)
    except Exception:
        return x

def find_latest_tsv(run_dir: str, pattern: str) -> Optional[str]:
    # Find all *.tsv (or user pattern) and pick most recently modified
    paths = glob.glob(os.path.join(run_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def write_wide_tsv(out_path: str, header_vals: List[str], matrix: List[List[Any]]) -> None:
    """
    Write a wide TSV:
      row\t<val1>\t<val2>\t...
      0\t...\t...
      1\t...\t...
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["row"] + header_vals)
        for i in range(len(matrix[0]) if matrix else 0):
            row = [i] + [col[i] for col in matrix]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Aggregate sweep TSVs into a wide TSV.")
    ap.add_argument("--sweep-dir", required=True, help="Top-level sweep directory")
    ap.add_argument("--param", required=True, help="Parameter name (e.g., learning_rate)")
    ap.add_argument("--glob", default="*.tsv", help="Glob for per-run TSVs (default: *.tsv)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--strict-length", action="store_true", help="Require identical vector lengths (default)")
    grp.add_argument("--allow-truncate", action="store_true", help="Truncate to MIN length")
    grp.add_argument("--allow-pad", action="store_true", help="Pad to MAX length with ''")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir
    param = args.param
    tsv_glob = args.glob

    if not os.path.isdir(sweep_dir):
        sys.exit(f"Not a directory: {sweep_dir}")

    # Collect run dirs of the form "<param>=<value>"
    subdirs = [
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and is_param_dir(d, param)
    ]
    if not subdirs:
        sys.exit(f"No run subdirectories matching '{param}=*' under {sweep_dir}")

    # Map: value_str -> (tsv_path, vector)
    results: Dict[str, Tuple[str, List[Any]]] = {}

    comp_name: Optional[str] = None
    for d in sorted(subdirs):
        run_dir = os.path.join(sweep_dir, d)
        value_str = parse_param_value(d, param)

        tsv_path = find_latest_tsv(run_dir, tsv_glob)
        if tsv_path is None:
            print(f"[warn] No TSVs in {run_dir} matching {tsv_glob}; skipping.", file=sys.stderr)
            continue

        name, vec = compute_vector(tsv_path)
        if comp_name is None:
            comp_name = name
        elif comp_name != name:
            sys.exit(f"Computation name mismatch: got '{name}' for {tsv_path}, expected '{comp_name}'")

        results[value_str] = (tsv_path, vec)

    if not results:
        sys.exit("No results gathered — nothing to write.")

    # Ensure consistent lengths or reconcile as requested
    lengths = [len(v) for (_, v) in results.values()]
    min_len = min(lengths)
    max_len = max(lengths)

    strict = (not args.allow_truncate) and (not args.allow_pad) or args.strict_length

    if strict and (min_len != max_len):
        detail = ", ".join(f"{k}:{len(v)}" for k, (_, v) in results.items())
        sys.exit(f"Vector lengths differ (strict mode). Lengths: {detail}")

    # Harmonize vectors if needed
    if args.allow_truncate and (min_len != max_len):
        for k in list(results.keys()):
            tsv_path, vec = results[k]
            results[k] = (tsv_path, vec[:min_len])
    elif args.allow_pad and (min_len != max_len):
        for k in list(results.keys()):
            tsv_path, vec = results[k]
            pad_n = max_len - len(vec)
            if pad_n > 0:
                vec = vec + ([""] * pad_n)
                results[k] = (tsv_path, vec)

    # Sort columns by numeric value when possible
    sorted_vals = sorted(results.keys(), key=numeric_key)

    # Build matrix: columns correspond to parameter values (in sorted order)
    columns: List[List[Any]] = [results[v][1] for v in sorted_vals]

    # Compose output file path
    out_name = f"{param}_{comp_name or 'comp'}.tsv"
    out_path = os.path.join(sweep_dir, out_name)

    write_wide_tsv(out_path, sorted_vals, columns)
    print(f"[ok] Wrote {out_path}")
    print(f"[info] Columns: {', '.join(sorted_vals)}")
    # Optional: show source files per column
    for v in sorted_vals:
        print(f"[src] {v}: {results[v][0]}")

if __name__ == "__main__":
    main()
