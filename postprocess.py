import argparse, pandas as pd, numpy as np, os, json

"""
Post-run quick statistics
-------------------------
Reads a TSV log file from training and computes simple summary metrics
over the last N rows (default 5000). Designed for fast inspection of
final model behavior without plotting.

Metrics per operator (currently hard-coded for '+' and '->'):
- acc_<op>: mean accuracy = fraction of correct predictions.
- conf_<op>: mean confidence (normalized entropy).
- finger_<op>: mean fraction of examples where finger counting was used.

Usage:
    python postprocess.py run.tsv --tail 5000

Arguments:
- tsv: path to training log TSV.
- --tail: number of most recent rows to include (default 5000).

Output:
- Prints a JSON dictionary with metrics per operator.

Notes:
- Expects TSV columns from train.py (target, predicted, operator,
  confidence, used_finger_counting).
- Operators are currently fixed to { '+', '->' } but can be extended.
- Complements eval.py (plots full trajectories) by offering a quick,
  tail-focused snapshot of end-of-training performance.
"""

def quick_stats(tsv_path, tail_steps=5000):
    df = pd.read_csv(tsv_path, sep="\t")
    df['is_correct'] = (df['target'] == df['predicted']).astype(int)
    if len(df) > tail_steps:
        tail = df.tail(tail_steps)
    else:
        tail = df
    res = {}
    for op in ['->', '+']:
        sub = tail[tail['operator']==op]
        if len(sub):
            res[f'acc_{op}'] = float(sub['is_correct'].mean())
            res[f'conf_{op}'] = float(sub['confidence'].mean())
            res[f'finger_{op}'] = float(sub['used_finger_counting'].mean())
        else:
            res[f'acc_{op}'] = None
            res[f'conf_{op}'] = None
            res[f'finger_{op}'] = None
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=str)
    ap.add_argument("--tail", type=int, default=5000)
    args = ap.parse_args()
    res = quick_stats(args.tsv, tail_steps=args.tail)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
