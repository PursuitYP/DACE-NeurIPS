#!/usr/bin/env bash
### dace: DACE co-evolution on Qwen2.5-14B-Instruct (attacker = Phase-1 SFT checkpoint, defender = Qwen2.5-14B-Instruct) ###
# Override any default via env vars, e.g. START_AGENT=defender bash scripts/rl/train_dace_qwen2.5_14b.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export ATTACKER_MODEL_PATH=${ATTACKER_MODEL_PATH:-$PROJECT_DIR/src/360-LLaMA-Factory/saves/dace_sft/qwen2d5-14b_v4_full_sft_dsz2}
export DEFENDER_MODEL_PATH=${DEFENDER_MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}
export EXP_NAME=${EXP_NAME:-DACE-Qwen2.5_14B_Instruct}

exec bash "${SCRIPT_DIR}/train_dace.sh" "$@"
