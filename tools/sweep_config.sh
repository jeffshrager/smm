#!/usr/bin/env bash
#
# sweep_config.sh — run multiple train.py jobs by sweeping one JSON config param
# macOS/Bash 3.2 compatible (no readarray/mapfile)
#
# EXAMPLES:
#   ./sweep_config.sh -t longrun_50k_add_start_3000.json -p learning_rate \
#       -a 0.003 -s 0.001 -b 0.007 -o results/lr_sweep
#
#   ./sweep_config.sh -t longrun_50k_add_start_3000.json -p addition_start_step \
#       -a 3000 -s 2000 -b 15000 -o results/addstart_sweep --train ./train.py

set -euo pipefail

TEMPLATE=""
PARAM=""
START=""
STEP=""
END=""
BASE_OUTDIR="results_sweep"
PYBIN="python3"
TRAIN="./train.py"
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage:
  sweep_config.sh -t <template.json> -p <param> -a <start> -s <step> -b <end> [-o outdir] [--python pybin] [--train train.py] [-n]

Flags:
  -t, --template   Template JSON config (required)
  -p, --param      JSON key to sweep (required)
  -a, --start      Start value inclusive (required)
  -s, --step       Step (required)
  -b, --end        End value inclusive (required)
  -o, --outdir     Base output dir (default: results_sweep)
      --python     Python executable (default: python3)
      --train      Path to train.py (default: ./train.py)
  -n, --dry-run    Print commands without running
  -h, --help       Show help
USAGE
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--template) TEMPLATE="$2"; shift 2 ;;
    -p|--param)    PARAM="$2"; shift 2 ;;
    -a|--start)    START="$2"; shift 2 ;;
    -s|--step)     STEP="$2"; shift 2 ;;
    -b|--end)      END="$2"; shift 2 ;;
    -o|--outdir)   BASE_OUTDIR="$2"; shift 2 ;;
    --python)      PYBIN="$2"; shift 2 ;;
    --train)       TRAIN="$2"; shift 2 ;;
    -n|--dry-run)  DRY_RUN=true; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# --- validation ---
[[ -n "$TEMPLATE" && -f "$TEMPLATE" ]] || { echo "Template not found: $TEMPLATE" >&2; exit 2; }
[[ -n "$PARAM" && -n "$START" && -n "$STEP" && -n "$END" ]] || { echo "Missing sweep args." >&2; usage; exit 2; }
[[ -f "$TRAIN" ]] || { echo "train.py not found at: $TRAIN" >&2; exit 2; }

mkdir -p "$BASE_OUTDIR"
TMPDIR=".sweep_tmp"
mkdir -p "$TMPDIR"

# --- generate values into a temp file (no readarray) ---
VALUES_FILE="${TMPDIR}/values_${PARAM}_$$.txt"
"$PYBIN" - "$START" "$STEP" "$END" > "$VALUES_FILE" <<'PY'
import sys, math
start = float(sys.argv[1]); step = float(sys.argv[2]); end = float(sys.argv[3])
if step == 0: 
    print(start); sys.exit(0)
vals = []
x = start
# include end (tolerate tiny fp error)
if step > 0:
    while x <= end + abs(step)*1e-12:
        vals.append(x); x += step
else:
    while x >= end - abs(step)*1e-12:
        vals.append(x); x += step
for v in vals:
    # print ints cleanly
    if abs(v - int(round(v))) < 1e-12:
        print(int(round(v)))
    else:
        # cap precision to avoid ugly fp tails
        print(("{:.12g}").format(v))
PY

have_jq=false
if command -v jq >/dev/null 2>&1; then have_jq=true; fi

write_cfg() {
  local src="$1" key="$2" val="$3" dst="$4"
  if $have_jq; then
    if echo "$val" | grep -Eq '^-?[0-9]+(\.[0-9]+)?$'; then
      jq --argjson v "$val" ".\"$key\" = \$v" "$src" > "$dst"
    else
      jq --arg v "$val" ".\"$key\" = \$v" "$src" > "$dst"
    fi
  else
    "$PYBIN" - "$src" "$key" "$val" "$dst" <<'PY'
import json, sys
src, key, sval, dst = sys.argv[1:]
with open(src) as f: cfg = json.load(f)
# numeric if possible
try:
    v = float(sval); 
    if abs(v - int(round(v))) < 1e-12: v = int(round(v))
except Exception:
    v = sval
cfg[key] = v
with open(dst, 'w') as f: json.dump(cfg, f, indent=2, sort_keys=True)
PY
  fi
}

echo "[sweep] Template: $TEMPLATE"
echo "[sweep] Param:    $PARAM"
echo "[sweep] Range:    $START .. $END (step $STEP)"
echo "[sweep] Out base: $BASE_OUTDIR"
echo

# --- main loop (no arrays) ---
# shellcheck disable=SC2162
while read v; do
  [ -z "${v:-}" ] && continue
  vtag="${v//./p}"; vtag="${vtag//-/_}"
  cfg_tmp="$TMPDIR/cfg_${PARAM}=${vtag}.json"
  run_out="${BASE_OUTDIR}/${PARAM}=${v}"

  write_cfg "$TEMPLATE" "$PARAM" "$v" "$cfg_tmp"
  echo "[sweep] v=$v -> cfg=$cfg_tmp outdir=$run_out"
  mkdir -p "$run_out"

  cmd=( "$PYBIN" "$TRAIN" --config "$cfg_tmp" --outdir "$run_out" )
  if $DRY_RUN; then
    printf 'DRY-RUN: %q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done < "$VALUES_FILE"

echo "[sweep] done."
