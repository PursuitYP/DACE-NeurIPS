#!/usr/bin/env bash
### dace: Table 3 GCG column — white-box GCG on the merged defender, then judge with run_judge_gcg.sh ###
# Usage: LOCAL_MODEL_PATH=<merged_defender_dir> MODEL_NAME=... DEFENDER_API_BASE_URL=... ATTACKER_API_BASE_URL=... \
#        OPENAI_API_KEY=... bash eval/OpenRT/run_gcg.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LOCAL_MODEL_PATH:?please set LOCAL_MODEL_PATH (merged HF dir of the defender) for GCG}"
export LOCAL_MODEL_PATH
export ATTACKS=gcg
export RESULTS_DIR=${RESULTS_DIR:-$SCRIPT_DIR/results/gcg-${MODEL_NAME:-${DEFENDER_API_MODEL:-orm}}-$(date +%Y%m%d_%H%M%S)}
exec bash "$SCRIPT_DIR/run_eval.sh" "$@"
