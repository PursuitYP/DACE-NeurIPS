#!/usr/bin/env bash
### dace: Table 3 — OOD automated red-teaming on HarmBench (320 test behaviors) via OpenRT, GPT-4o judge ###
#
# Serve the merged defender (target) and the attacker LLM with an OpenAI-compatible server, e.g.
#   vllm serve <merged_defender_dir> --served-model-name orm --port <PORT>
# then:
#   MODEL_NAME=DACE-Qwen2.5-7B \
#   DEFENDER_API_BASE_URL=http://<DEFENDER_HOST>:<PORT>/v1 ATTACKER_API_BASE_URL=http://<ATTACKER_HOST>:<PORT>/v1 \
#   OPENAI_API_KEY=<key> bash eval/OpenRT/run_eval.sh
#
# Other scripts reuse this one: run_gcg.sh (GCG), run_self_eval.sh / run_smoothllm_eval.sh (inference-time defenses).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# user-supplied relative paths are resolved against the caller's directory
abspath() { if [[ -n "$1" && "$1" != /* && -e "$1" ]]; then echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; else echo "$1"; fi; }
LOCAL_MODEL_PATH="$(abspath "${LOCAL_MODEL_PATH:-}")"
[[ -n "${DATASET_PATH:-}" ]] && DATASET_PATH="$(abspath "$DATASET_PATH")"
[[ -n "${RESULTS_DIR:-}" && "$RESULTS_DIR" != /* ]] && RESULTS_DIR="$PWD/$RESULTS_DIR"
cd "$SCRIPT_DIR"

# ---- target (defender) and attacker endpoints ----
DEFENDER_API_BASE_URL=${DEFENDER_API_BASE_URL:-http://<DEFENDER_HOST>:<PORT>/v1}
DEFENDER_API_KEY=${DEFENDER_API_KEY:-EMPTY}
DEFENDER_API_MODEL=${DEFENDER_API_MODEL:-orm}
ATTACKER_API_BASE_URL=${ATTACKER_API_BASE_URL:-http://<ATTACKER_HOST>:<PORT>/v1}
ATTACKER_API_KEY=${ATTACKER_API_KEY:-EMPTY}
ATTACKER_API_MODEL=${ATTACKER_API_MODEL:-orm}
ATTACKER_ANSWER_EXTRACT=${ATTACKER_ANSWER_EXTRACT:-true}

# ---- judge / embedding (OpenAI-compatible) ----
: "${OPENAI_API_KEY:?please set OPENAI_API_KEY (GPT-4o judge + embeddings)}"
OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
export OPENAI_API_KEY OPENAI_BASE_URL
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-3-small}

# ---- run settings (values used in our experiments) ----
TARGET_MODEL=${TARGET_MODEL:-$DEFENDER_API_MODEL}
ATTACKER_MODEL=${ATTACKER_MODEL:-$ATTACKER_API_MODEL}
MODEL_NAME=${MODEL_NAME:-$TARGET_MODEL}
ATTACKS=${ATTACKS:-"no_revision pair tree autodan autodan_turbo_r"}
# LOCAL_MODEL_PATH: required by GCG (white-box), merged HF dir of the defender
DATASET_PATH=${DATASET_PATH:-./seed/harmbench/harmbench_behaviors_text_test.csv}
NUM_SAMPLES=${NUM_SAMPLES:-}               # empty = all prompts
MAX_TURNS=${MAX_TURNS:-3}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-5}
RESULTS_DIR=${RESULTS_DIR:-./results/${MODEL_NAME}-$(date +%Y%m%d_%H%M%S)}

if [[ "$DEFENDER_API_BASE_URL" == *"<"* ]]; then
  echo "[OpenRT] ERROR: please export DEFENDER_API_BASE_URL (OpenAI-compatible endpoint of the target)." >&2
  exit 2
fi
if [[ " $ATTACKS " =~ \ (pair|tree|autodan_turbo|autodan_turbo_r)\  && "$ATTACKER_API_BASE_URL" == *"<"* ]]; then
  echo "[OpenRT] ERROR: PAIR / TAP / AutoDAN-Turbo need an attacker LLM; please export ATTACKER_API_BASE_URL." >&2
  exit 2
fi

echo "----------------------------------------------------------------"
echo "OpenRT | target=$TARGET_MODEL @ $DEFENDER_API_BASE_URL | attacker=$ATTACKER_MODEL | judge=$JUDGE_MODEL"
echo "OpenRT | attacks=$ATTACKS | dataset=$DATASET_PATH | results=$RESULTS_DIR"
echo "----------------------------------------------------------------"

ARGS=(
    --target-api-key "$DEFENDER_API_KEY"
    --target-base-url "$DEFENDER_API_BASE_URL"
    --attacker-api-key "$ATTACKER_API_KEY"
    --attacker-base-url "$ATTACKER_API_BASE_URL"
    --judge-api-key "$OPENAI_API_KEY"
    --judge-base-url "$OPENAI_BASE_URL"
    --target-model "$TARGET_MODEL"
    --attacker-model "$ATTACKER_MODEL"
    --judge-model "$JUDGE_MODEL"
    --embedding-model "$EMBEDDING_MODEL"
    --dataset-path "$DATASET_PATH"
    --max-turns "$MAX_TURNS"
    --judge-threshold "$JUDGE_THRESHOLD"
    --results-dir "$RESULTS_DIR"
)
[[ -n "$LOCAL_MODEL_PATH" ]] && ARGS+=(--local-model-path "$LOCAL_MODEL_PATH")
[[ -n "$NUM_SAMPLES" ]] && ARGS+=(--num-samples "$NUM_SAMPLES")
[[ "$ATTACKER_ANSWER_EXTRACT" == "true" ]] && ARGS+=(--attacker-answer-extract)

# shellcheck disable=SC2086
python3 unified_eval.py "${ARGS[@]}" --attacks $ATTACKS
