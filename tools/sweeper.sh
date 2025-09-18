#!/usr/bin/env bash
# sweeper.sh — run a parameter sweep, reduce the outputs, and plot results.
# Supports skipping the sweep with --norerun or --rerun=no to reuse existing runs.

# ./sweeper.sh -a 0.6 -s 0.025 -b 0.8 -l confidence_floor
# Reuse runs only (no rerun):
# ./sweeper.sh --norerun -l confidence_floor -O "count" -w 250

set -euo pipefail

########################################
# Defaults (adjust for your repo layout)
########################################
A="0.6"
S="0.025"
B="0.8"
LABEL="confidence_floor"

# Curriculum / analytics
OP="add"            # reduce operator: e.g., "add", "count", "->", etc.
WINDOW="2000"       # window size for correctness / finger counting

# Files & tools
TEMPLATE="params.json"
RESULTS_DIR="results"
PYTHON="python3"
TRAIN="../../repo/model/train.py"

SWEEP_CFG="../../repo/tools/sweep_config.sh"
REDUCE_SCRIPT="../../repo/tools/sweep_reduce.py"
PLOT_SWEEP_SCRIPT="../../repo/tools/plot_sweep.py"
PLOT_FINGER_SCRIPT="../../repo/tools/plot_finger_counting.py"

# Control
RERUN="yes"         # default: do the sweep; use --norerun or --rerun=no to skip

########################################
usage() {
  cat <<EOF
Usage: $0 [options]

Sweep + reduce + plot pipeline. Use --norerun (or --rerun=no) to skip re-running
the sweep and just regenerate reductions/plots from existing runs in RESULTS_DIR.

Options:
  -a <min>          Sweep 'a' lower bound (default: ${A})
  -s <step>         Sweep step size (default: ${S})
  -b <max>          Sweep 'b' upper bound (default: ${B})
  -l <label>        Parameter label (default: ${LABEL})
  -t <template>     Sweep template JSON (default: ${TEMPLATE})
  -o <results>      Results directory (default: ${RESULTS_DIR})
  -O <op>           Operator for reduce (default: ${OP})
  -w <window>       Window size for correctness/finger counting (default: ${WINDOW})
  -P <python>       Python executable (default: ${PYTHON})
  -T <train>        Train script path (default: ${TRAIN})
  -C <sweep_cfg>    sweep_config.sh path (default: ${SWEEP_CFG})
  -R <reduce_py>    sweep_reduce.py path (default: ${REDUCE_SCRIPT})
  -S <plot_py>      plot_sweep.py path (default: ${PLOT_SWEEP_SCRIPT})
  -F <finger_py>    plot_finger_counting.py path (default: ${PLOT_FINGER_SCRIPT})
  --norerun         Skip the sweep step; only do reduce/plots (default is to run)
  --rerun yes|no    Explicitly control whether to run the sweep (default: yes)
  -h                Show this help and exit

Examples:
  # Normal end-to-end run:
  $0 -a 0.6 -s 0.025 -b 0.8 -l confidence_floor

  # Reuse prior runs; redo reduce/plots for a different operator/window:
  $0 --norerun -l confidence_floor -O "count" -w 250
EOF
}

########################################
# Pre-parse long options so macOS/BSD getopts won't choke
########################################
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --norerun)
      RERUN="no"; shift ;;
    --rerun)
      RERUN="${2:-yes}"; shift 2 ;;
    --rerun=*)
      RERUN="${1#*=}"; shift ;;
    --) shift; break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      args+=("$1"); shift ;;
  esac
done
set -- "${args[@]}"

########################################
# Parse short options
########################################
while getopts ":a:s:b:l:t:o:O:w:P:T:C:R:S:F:h" opt; do
  case "$opt" in
    a) A="$OPTARG" ;;
    s) S="$OPTARG" ;;
    b) B="$OPTARG" ;;
    l) LABEL="$OPTARG" ;;
    t) TEMPLATE="$OPTARG" ;;
    o) RESULTS_DIR="$OPTARG" ;;
    O) OP="$OPTARG" ;;
    w) WINDOW="$OPTARG" ;;
    P) PYTHON="$OPTARG" ;;
    T) TRAIN="$OPTARG" ;;
    C) SWEEP_CFG="$OPTARG" ;;
    R) REDUCE_SCRIPT="$OPTARG" ;;
    S) PLOT_SWEEP_SCRIPT="$OPTARG" ;;
    F) PLOT_FINGER_SCRIPT="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Missing arg for -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

########################################
# Sanity checks
########################################
if [[ ! -x "$SWEEP_CFG" && ! -f "$SWEEP_CFG" ]]; then
  echo "Warning: sweep_config.sh not found at: $SWEEP_CFG (only needed if rerun=yes)" >&2
fi
if [[ ! -f "$REDUCE_SCRIPT" ]]; then
  echo "Error: sweep_reduce.py not found at: $REDUCE_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$PLOT_SWEEP_SCRIPT" ]]; then
  echo "Error: plot_sweep.py not found at: $PLOT_SWEEP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$PLOT_FINGER_SCRIPT" ]]; then
  echo "Error: plot_finger_counting.py not found at: $PLOT_FINGER_SCRIPT" >&2
  exit 2
fi

########################################
# Helpers
########################################
to_slug() {
  # lowercase, replace non-alnum by _, trim leading/trailing _
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g'
}

SLUG="$(to_slug "$LABEL")"
mkdir -p "$RESULTS_DIR"

########################################
# 1) Sweep (optional)
########################################
do_sweep=1
case "$RERUN" in
  [Nn]|[Nn][Oo]) do_sweep=0 ;;
esac

if [[ "$do_sweep" -eq 1 ]]; then
  echo "[sweep] Running sweep_config.sh with a=${A}..${B} step=${S} label=${LABEL}"
  "${SWEEP_CFG}" \
    -t "$TEMPLATE" \
    -p "$LABEL" \
    -a "$A" \
    -s "$S" \
    -b "$B" \
    -o "$RESULTS_DIR" \
    --python "$PYTHON" \
    --train "$TRAIN"
else
  echo "[sweep] Skipping (rerun=no). Using existing runs in: ${RESULTS_DIR}"
fi

########################################
# 2) Reduce
########################################
echo "[reduce] Reducing sweeps in ${RESULTS_DIR} for param=${LABEL} op=${OP} window=${WINDOW}"
"${PYTHON}" "${REDUCE_SCRIPT}" \
  --sweep-dir "${RESULTS_DIR}" \
  --param "${LABEL}" \
  --op "${OP}" \
  --window "${WINDOW}" \
  --allow-truncate

TSV_PATH="${RESULTS_DIR}/${SLUG}_${OP}_correctness${WINDOW}.tsv"
if [[ ! -f "${TSV_PATH}" ]]; then
  echo "Warning: Expected TSV not found: ${TSV_PATH}"
  echo "         (Name template may differ in your reduce script.)"
fi

########################################
# 3) Plot sweep correctness
########################################
OUT_CORR="${SLUG}_${OP}_correctness${WINDOW}.png"
PLOT_LABEL="${LABEL} x Sweep Correctness ${WINDOW}"
echo "[plot] Plotting correctness -> ${OUT_CORR}"
"${PYTHON}" "${PLOT_SWEEP_SCRIPT}" \
  "${TSV_PATH}" \
  --label "${PLOT_LABEL}" \
  --out "${OUT_CORR}"

########################################
# 4) Plot finger counting usage
########################################
OUT_FC="${SLUG}_${OP}_finger_counting.png"
echo "[plot] Plotting finger counting -> ${OUT_FC}"
"${PYTHON}" "${PLOT_FINGER_SCRIPT}" \
  --sweep-dir "${RESULTS_DIR}" \
  --param "${LABEL}" \
  --window "${WINDOW}" \
  --out "${OUT_FC}"

echo "[done] Results:"
echo "  TSV:  ${TSV_PATH}"
echo "  PNG:  ${OUT_CORR}"
echo "  PNG:  ${OUT_FC}"
