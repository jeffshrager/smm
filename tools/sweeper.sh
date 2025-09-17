#!/usr/bin/env bash
# sweeper.sh
#
# Run a sweep, reduce, and plot with parameterized a/s/b and label.
#
# Examples:
#   ./sweeper.sh -a 0.6 -s 0.025 -b 0.8 -l confidence_floor
#   ./sweeper.sh -a 0.3 -s 0.010 -b 0.9 -l "addition start step"
#
# Optional knobs (with sensible defaults) let you tweak paths/opts.

set -euo pipefail

# --- Defaults (tweak if your layout differs) ---
TEMPLATE="params.json"
RESULTS_DIR="results"
PYTHON="python3"
TRAIN="../../repo/model/train.py"
SWEEP_CFG="../../repo/tools/sweep_config.sh"
SWEEP_REDUCE="../../repo/tools/sweep_reduce.py"
PLOT_SWEEP="../../repo/tools/plot_sweep.py"
PLOT_FINGERS="../../repo/tools/plot_finger_counting.py"
OP="add"
WINDOW=2000

# --- Required args ---
A=""
S=""
B=""
LABEL=""

usage() {
  cat <<EOF
Usage: $(basename "$0") -a <A> -s <S> -b <B> -l <label> [options]

Required:
  -a <A>           Value for -a (e.g., 0.6)
  -s <S>           Value for -s (e.g., 0.025)
  -b <B>           Value for -b (e.g., 0.8)
  -l <label>       Label/param name (used for -p/--param and in plot text)

Options:
  -t <template>    Sweep template JSON (default: ${TEMPLATE})
  -o <outdir>      Results directory (default: ${RESULTS_DIR})
  -w <window>      Window size for correctness/finger counting (default: ${WINDOW})
  -O <op>          Operator for reduce step (default: ${OP})
  -P <python>      Python executable (default: ${PYTHON})
  -T <train>       Train script path (default: ${TRAIN})
  -h               Show this help

Notes:
- The label is passed as -p to sweep_config and --param to reducers/plots.
- The label is also embedded in plot titles; a slugified version names files.
EOF
}

# Parse args
while getopts ":a:s:b:l:t:o:w:O:P:T:h" opt; do
  case "$opt" in
    a) A="$OPTARG" ;;
    s) S="$OPTARG" ;;
    b) B="$OPTARG" ;;
    l) LABEL="$OPTARG" ;;
    t) TEMPLATE="$OPTARG" ;;
    o) RESULTS_DIR="$OPTARG" ;;
    w) WINDOW="$OPTARG" ;;
    O) OP="$OPTARG" ;;
    P) PYTHON="$OPTARG" ;;
    T) TRAIN="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

# Validate required
if [[ -z "$A" || -z "$S" || -z "$B" || -z "$LABEL" ]]; then
  echo "Error: -a, -s, -b, and -l are required." >&2
  usage
  exit 2
fi

# Slugify label for filenames (lowercase, non-alnum -> _)
SLUG=$(echo "$LABEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')

mkdir -p "$RESULTS_DIR"

# 1) Sweep
"$SWEEP_CFG" \
  -t "$TEMPLATE" \
  -p "$LABEL" \
  -a "$A" \
  -s "$S" \
  -b "$B" \
  -o "$RESULTS_DIR" \
  --python "$PYTHON" \
  --train "$TRAIN"

# 2) Reduce
"$PYTHON" "$SWEEP_REDUCE" \
  --sweep-dir "$RESULTS_DIR" \
  --param "$LABEL" \
  --op "$OP" \
  --allow-truncate

# 3) Plot correctness
CORR_TSV="${RESULTS_DIR}/${SLUG}_${OP}_correctness${WINDOW}.tsv"
CORR_PNG="${SLUG}_${OP}_correctness${WINDOW}.png"
if [[ -f "$CORR_TSV" ]]; then
  "$PYTHON" "$PLOT_SWEEP" \
    "$CORR_TSV" \
    --label "${LABEL} x Sweep Correctness ${WINDOW}" \
    --out "$CORR_PNG"
else
  echo "Warning: Missing TSV for correctness plot: $CORR_TSV" >&2
fi

# 4) Plot finger counting usage
FINGERS_PNG="${SLUG}_${OP}_finger_counting.png"
"$PYTHON" "$PLOT_FINGERS" \
  --sweep-dir "$RESULTS_DIR" \
  --param "$LABEL" \
  --window "$WINDOW" \
  --out "$FINGERS_PNG"

echo "Done."
echo "Outputs (if available):"
echo "  Correctness TSV: $CORR_TSV"
echo "  Correctness PNG: $CORR_PNG"
echo "  Finger-counting PNG: $FINGERS_PNG"
