# smm
Small Math Model
# Small Math Model (SMM)

A simplified neural network that learns basic arithmetic, designed to study how neural networks handle conflicting learned behaviors. The SMM demonstrates the tension between counting sequences and addition through continuous learning with Gaussian curriculum scheduling.

## Overview

The Small Math Model is a three-layer neural network that learns to process three-word sentences of the form "addend1 operator addend2" where:
- **Addends**: Numbers 1-5 
- **Operators**: `+` (addition) or `->` (counting/next)
- **Outputs**: Numbers 1-12

### Key Research Features

**Counting vs Addition Conflict**: The model first learns counting sequences (1→2, 2→3, 3→4, etc.) then must learn addition. This creates intentional conflicts like learning "2 → 3" but later "2 + 3 = 5", forcing the network to use the operator to determine which pattern applies.

**Gaussian Curriculum Learning**: Three-level hierarchical system:
- **Time Flow (TF)**: Controls overall pacing through training
- **Complexity Flow (CF)**: Gets mean from TF, determines current complexity focus  
- **Complexity (CX)**: Gets mean from CF, samples actual problems

**Confidence & Finger Counting**: Uses entropy-based confidence calculation. When confidence falls below threshold, engages "finger counting" - a fallback representing external mechanical computation.

## Architecture

- **Input Layer**: One-hot encoding (addend1 + operator + addend2) = 12 dimensions
- **Hidden Layer**: 64 ReLU units with simple attention mechanism
- **Output Layer**: 12 softmax units (results 1-12)
- **Learning**: Continuous online learning (one problem per step)

## Installation

```bash
# Required packages
pip install numpy matplotlib pandas

Usage
Basic Training
bashpython smm2.py
This will:

Initialize the SMM with Gaussian curriculum
Train for 10,000 continuous learning steps
Log all training data to results/TIMESTAMP.tsv
Output progress to results/TIMESTAMP.out
Test the model comprehensively at the end

Analyzing Results
Use the visualization tool to plot learning progress:
bash# Basic 3D histogram plot
python plot_smm_progress.py results/20231207123456.tsv

# Custom epoch grouping (500 steps per epoch)
python plot_smm_progress.py results/20231207123456.tsv --steps-per-epoch 500

# Save plot to file
python plot_smm_progress.py results/20231207123456.tsv --output progress.png
Configuration
Model Parameters
In smm2.py, you can adjust:
python# Network architecture
smm = SMM(hidden_size=64, learning_rate=0.02)
smm.confidence_criterion = 0.9  # Threshold for finger counting

# Curriculum parameters
curriculum = GaussianCurriculum()
curriculum.total_steps = 10000           # Total training steps
curriculum.addition_start_step = 3000    # When addition begins
curriculum.tf_rate = 0.0001             # Curriculum speed
Curriculum Learning Schedule

Steps 0-3000: Pure counting focus
Steps 3000-10000: Gradual transition to addition
Continuous mixing: Smooth Gaussian transitions, no hard phase boundaries

Output Files
TSV Log (TIMESTAMP.tsv)
Detailed step-by-step training log with columns:

timestamp, phase, step
addend1, operator, addend2, target, predicted
confidence, used_finger_counting, loss
confidence_criterion, learning_rate

Console Output (TIMESTAMP.out)
Human-readable training progress including:

Curriculum configuration
Step-by-step progress (every 1000 steps)
Comprehensive final testing
Counting vs addition conflict analysis

Example Results
The model learns to resolve the counting-addition conflict:
Counting vs Addition conflict analysis:
  2 -> ? = 3 (expected 3) conf:0.892
  2 + 2 = 4 (expected 4) conf:0.856
    Conflict resolved: True (counting: True, addition: True)
Performance typically reaches:

Counting: 90-100% accuracy
Addition: 70-90% accuracy (depending on complexity)

Research Applications
This model is useful for studying:

Curriculum learning effects in neural networks
Catastrophic forgetting and conflict resolution
Confidence estimation in neural networks
Continuous vs batch learning differences
Attention mechanisms in simple arithmetic

File Structure
smm/
├── smm2.py                 # Main SMM implementation
├── plot_smm_progress.py    # Visualization tool
├── results/                # Generated logs and outputs
│   ├── TIMESTAMP.tsv      # Training data logs
│   └── TIMESTAMP.out      # Console output logs
└── README.md              # This file
Contributing
The SMM is designed to be easily extensible:

Add new operators or number ranges
Modify the curriculum learning schedule
Experiment with different architectures
Add new conflict scenarios

Citation
If you use this code in research, please cite the Small Math Model project and describe the specific configuration used.
