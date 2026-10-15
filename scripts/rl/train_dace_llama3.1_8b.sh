#!/usr/bin/env bash
### dace: DACE co-evolution on Llama-3.1-8B-Instruct (attacker = Phase-1 SFT checkpoint, defender = Llama-3.1-8B-Instruct) ###
# Override any default via env vars, e.g. START_AGENT=defender bash scripts/rl/train_dace_llama3.1_8b.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ATTACKER_MODEL_PATH=${ATTACKER_MODEL_PATH:-$PROJECT_DIR/src/360-LLaMA-Factory/saves/dace_sft/llama3d1-8b_v4_full_sft_dsz2}
export DEFENDER_MODEL_PATH=${DEFENDER_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
export EXP_NAME=${EXP_NAME:-DACE-Llama3.1_8B_Instruct}

exec bash "${SCRIPT_DIR}/train_dace.sh" "$@"
