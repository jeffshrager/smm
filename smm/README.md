# Small Math Model (SMM) — Embeddings, Quasi-Attention, and Finger Counting

The Small Math Model (SMM) is a pedagogical analogue to modern Large Language and Math Models (LLMs/LMMs). It is designed to study how embeddings, attention-like gating, curriculum learning, and fallback reasoning strategies (finger counting) interact in a controlled setting. This project modernizes and extends earlier work by Shrager et al. by explicitly incorporating embeddings and quasi-attention.

## Key Features
- **Embeddings**: Numbers (1–12) and operators (`+`, `->`) are mapped into learned vector embeddings, analogous to token embeddings in LLMs.
- **Quasi-Attention**: Inputs are concatenated and passed through a per-dimension sigmoid gate (`gate = sigmoid(x @ A + b)`), allowing the model to emphasize or suppress input dimensions. This acts as a simplified form of attention.
- **Finger Counting**: A three-phase counting procedure serves as an external reasoning tool. When the model’s confidence is low (based on entropy of the output distribution), it falls back to finger counting, and the result is fed back into training.
- **Curriculum Learning**: A Gaussian schedule controls task difficulty and the mix of counting vs. addition problems, delaying addition until later in training.
- **Confidence Annealing**: A confidence floor ensures finger counting remains active early in training, then gradually phases it out.

## Installation
```bash
pip install numpy pandas matplotlib

##Running Training

Example (long run with JSON config):

python -m train --config smm/configs/longrun_50k.json --outdir smm/results


##Override any parameter on the command line:

python -m train --config smm/configs/longrun_50k.json \
  --override learning_rate=0.005 addition_start_step=15000 checkpoint_interval=5000

##Evaluation and Summarization
python -m eval smm/results/2025xxxxxxxx.tsv --steps-per-epoch 1000
python -m postprocess smm/results/2025xxxxxxxx.tsv --tail 5000

##Configuration Parameters (JSON)

total_steps (default 50000)

learning_rate (default 0.005), learning_rate_floor (0.002), lr_decay (0.9999)

hidden_size (64)

embed_size (8)

gate_freeze_until_step (3000)

addition_start_step (12000), counting_fade_rate (0.0001)

confidence_criterion_start (0.9), confidence_floor (0.75)

checkpoint_interval (10000), resume_from_checkpoint (path or null)

##Outputs

TSV logs and .out files under results/ (timestamped).

Optional .npz checkpoints (resumable).

Plots for accuracy, confidence, and finger counting usage (via eval.py).

Notes

##Analysis

See an1.py in /results

##Model initialization currently uses a fixed RNG seed; reproducibility can be extended by overriding with your own seed.

This codebase is intended both as a modernization of Shrager et al.’s early symbolic–connectionist work and as a stepping stone toward richer analogues of LLM training.

##Next Steps

Future extensions will explore:

Multiple strategies per operator with a learned policy head (e.g., FARRA/UMA).

Replacement of quasi-attention with a true Transformer block.

Richer vocabularies and multi-digit tokenization.
