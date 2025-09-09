import argparse, pandas as pd, numpy as np, os, json

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
