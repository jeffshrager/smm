import argparse, pandas as pd, numpy as np, os, json
import matplotlib.pyplot as plt

def summarize(tsv_path, steps_per_epoch=1000, save_plots=True, outdir=None):
    df = pd.read_csv(tsv_path, sep="\t")
    df['is_correct'] = (df['target'] == df['predicted']).astype(int)
    df['is_add'] = (df['operator'] == '+').astype(int)
    df['is_next'] = (df['operator'] == '->').astype(int)

    step = df['step'].values
    epoch = (step // steps_per_epoch).astype(int)
    df['epoch'] = epoch

    grp = df.groupby(['epoch','operator']).agg(
        acc=('is_correct','mean'),
        finger=('used_finger_counting','mean'),
        conf=('confidence','mean'),
        n=('is_correct','size')
    ).reset_index()

    if save_plots:
        if outdir is None:
            outdir = os.path.dirname(tsv_path)
        base = os.path.splitext(os.path.basename(tsv_path))[0]

        # accuracy
        plt.figure()
        for op in ['->', '+']:
            g = grp[grp['operator']==op]
            plt.plot(g['epoch'], g['acc'], label=f'Accuracy {op}')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy over time')
        acc_png = os.path.join(outdir, f"{base}_acc.png")
        plt.savefig(acc_png); plt.close()

        # finger rate (addition)
        plt.figure()
        g = grp[grp['operator']=='+']
        if len(g):
            plt.plot(g['epoch'], g['finger'])
        plt.xlabel('Epoch'); plt.ylabel('Finger-use rate (addition)'); plt.title('Finger counting usage (addition)')
        fing_png = os.path.join(outdir, f"{base}_finger.png")
        plt.savefig(fing_png); plt.close()

        # confidence
        plt.figure()
        for op in ['->', '+']:
            g = grp[grp['operator']==op]
            plt.plot(g['epoch'], g['conf'], label=f'Conf {op}')
        plt.xlabel('Epoch'); plt.ylabel('Mean confidence'); plt.legend(); plt.title('Confidence over time')
        conf_png = os.path.join(outdir, f"{base}_conf.png")
        plt.savefig(conf_png); plt.close()

        return {'acc_png':acc_png, 'finger_png':fing_png, 'conf_png':conf_png}

    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=str)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = summarize(args.tsv, steps_per_epoch=args.steps_per_epoch, save_plots=(not args.no_plots))
    if out:
        import json
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
