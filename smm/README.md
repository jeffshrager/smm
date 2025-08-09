# Small Math Model (SMM) — Content-Gated + Finger Counting

This codebase reproduces the experiments we agreed on:
- **Counting fix**: no conflicting “extended counting to 7–12” items.
- **Domain-agnostic content gate**: `gate = sigmoid(x @ A + b)`, correct gradients.
- **Finger counting**: three-phase procedure as recursive subcalls to prevent forgetting.
- **Curriculum**: Gaussian schedule with delayed addition.
- **Confidence floor**: keeps finger counting active early.

## Install
```
pip install numpy pandas matplotlib
```

## Run (long run, JSON config)
```
python -m train --config smm/configs/longrun_50k.json --outdir smm/results
```
Override any param on CLI:
```
python -m train --config smm/configs/longrun_50k.json   --override learning_rate=0.005 addition_start_step=15000 checkpoint_interval=5000
```

## Evaluate & plots
```
python -m eval smm/results/2025xxxxxxxx.tsv --steps-per-epoch 1000
python -m postprocess smm/results/2025xxxxxxxx.tsv --tail 5000
```

## Config knobs (JSON)
- `total_steps` (default 50000)
- `learning_rate` (default 0.005), `learning_rate_floor` (0.002), `lr_decay` (0.9999)
- `hidden_size` (64)
- `gate_freeze_until_step` (3000)
- `addition_start_step` (12000), `counting_fade_rate` (0.0001)
- `confidence_criterion_start` (0.9), `confidence_floor` (0.75)
- `checkpoint_interval` (10000), `resume_from_checkpoint` (path or null)

## Notes
- Outputs are TSV and OUT under `results/` with timestamps.
- Checkpoints are `.npz` files you can resume from via `--override resume_from_checkpoint=...`.
- For reproducibility across seeds, invoke `--override` with a seed and set your RNGs before constructing the model or curriculum (this minimal version uses fixed init inside `SMM`; feel free to propagate your own seed).

## Next steps
When you’re ready, we’ll spin a separate branch that adds **multiple strategies per operator** and a small **policy head** to predict strategies (FARRA/UMA).
