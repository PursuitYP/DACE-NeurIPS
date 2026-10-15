#!/usr/bin/env bash
### dace: Table 3 Self-Eval baseline — OpenRT wraps the target when its model name starts with "selfeval-" ###
# Usage: same env vars as run_eval.sh (serve the base instruct model as the target).
# GCG column: ATTACKS=gcg LOCAL_MODEL_PATH=<base instruct model>, then run_judge_gcg.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_MODEL="selfeval-${DEFENDER_API_MODEL:-orm}"
export MODEL_NAME=${MODEL_NAME:-selfeval-defense}
export ATTACKER_ANSWER_EXTRACT=${ATTACKER_ANSWER_EXTRACT:-false}   # baselines were run without answer extraction
exec bash "$SCRIPT_DIR/run_eval.sh" "$@"
