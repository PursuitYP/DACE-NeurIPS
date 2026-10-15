#!/usr/bin/env bash
### dace: evaluations that query the defender through an OpenAI-compatible API — Table 1 StrongREJECT, Table 2 IFEval / AlpacaEval 2 ###
#
#   [1/3] safety-eval-fork : StrongREJECT (GPT-4o autograder)
#   [2/3] olmes            : IFEval
#   [3/3] safety-eval-fork : AlpacaEval 2 (GPT-4o annotator)
#
# Serve the merged defender first, e.g.
#   vllm serve <merged_defender_dir> --served-model-name orm --port <PORT>
# Usage:
#   MODEL_NAME=DACE-Qwen2.5-7B DEFENDER_API_BASE_URL=http://<DEFENDER_HOST>:<PORT>/v1 \
#   OPENAI_API_KEY=<key> [OPENAI_BASE_URL=<gateway>] bash eval/run_api_defender_all.sh
set -o pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME=${MODEL_NAME:-DACE-defender}
export DEFENDER_API_BASE_URL=${DEFENDER_API_BASE_URL:-http://<DEFENDER_HOST>:<PORT>/v1}
export DEFENDER_API_KEY=${DEFENDER_API_KEY:-EMPTY}
export DEFENDER_API_MODEL=${DEFENDER_API_MODEL:-orm}
: "${OPENAI_API_KEY:?please set OPENAI_API_KEY (GPT-4o judge for StrongREJECT / AlpacaEval)}"
export OPENAI_API_KEY
[[ -n "${OPENAI_BASE_URL:-}" ]] && export OPENAI_BASE_URL
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
SAFETY_EVAL_ENV=${SAFETY_EVAL_ENV:-safety-eval}
OLMES_ENV=${OLMES_ENV:-olmes}
if [[ "$DEFENDER_API_BASE_URL" == *"<"* ]]; then
  echo "[eval] ERROR: please export DEFENDER_API_BASE_URL=http://<host>:<port>/v1" >&2
  exit 2
fi
export HF_DATASETS_VERBOSITY=info

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
activate() { command -v conda >/dev/null 2>&1 || return 0; conda activate "$1" || { echo "[eval] ERROR: conda env '$1' not found" >&2; exit 1; }; }

echo "=========================================================="
echo "[API-DEFENDER] MODEL_NAME            = $MODEL_NAME"
echo "[API-DEFENDER] DEFENDER_API_BASE_URL = $DEFENDER_API_BASE_URL"
echo "[API-DEFENDER] DEFENDER_API_MODEL    = $DEFENDER_API_MODEL"
echo "=========================================================="

# ------------------------------------------------------------
# [1/3] Table 1 · StrongREJECT (safety-eval-fork)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/safety-eval-fork"
    activate "$SAFETY_EVAL_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

    TASKS="strongreject"
    RESULTS_DIR="$PWD/results/$TASKS-$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    timeout 7200 python -u evaluation/eval.py generators \
        --model_name_or_path "$MODEL_NAME" \
        --model_input_template_path_or_name "hf" \
        --tasks "$TASKS" \
        --report_output_path "$RESULTS_DIR/metrics.json" \
        --save_individual_results_path "$RESULTS_DIR/all_results.json" \
        --use_defender_api \
        --use_gpt4_autograder \
        --batch_size 1 2>&1 | tee "$RESULTS_DIR/eval.log"
    rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] && echo "[1/3] StrongREJECT done: $RESULTS_DIR" || echo "[1/3] StrongREJECT FAILED (exit $rc), see $RESULTS_DIR"
)

# ------------------------------------------------------------
# [2/3] Table 2 · IFEval (olmes, via defender API)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/olmes"
    activate "$OLMES_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    # Tip (offline nodes): export TIKTOKEN_CACHE_DIR=$CONDA_PREFIX/lib/python3.10/site-packages/litellm/litellm_core_utils/tokenizers

    OUTPUT_DIR="$PWD/results/ifeval/$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    python api_eval.py \
        --task ifeval \
        --api-key "$DEFENDER_API_KEY" \
        --base-url "$DEFENDER_API_BASE_URL" \
        --model "$DEFENDER_API_MODEL" \
        --output-dir "$OUTPUT_DIR"
    rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] && echo "[2/3] IFEval done: $OUTPUT_DIR" || echo "[2/3] IFEval FAILED (exit $rc), see $OUTPUT_DIR"
)

# ------------------------------------------------------------
# [3/3] Table 2 · AlpacaEval 2 (safety-eval-fork, GPT-4o annotator)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/safety-eval-fork"
    activate "$SAFETY_EVAL_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

    export ALPACA_EVAL_ANNOTATORS_CONFIG="${ALPACA_EVAL_ANNOTATORS_CONFIG:-$PWD/evaluation/tasks/generation/alpacaeval/weighted_alpaca_eval_gpt4o}"
    export ALPACA_EVAL_EXTRA_RETRIES=3
    export ALPACA_EVAL_RETRY_SLEEP_SECONDS=5
    export ALPACA_EVAL_SKIP_CONTENT_FILTER=1

    TASKS="alpacaeval"
    RESULTS_DIR="$PWD/results/$TASKS-$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    timeout 72000 python -u evaluation/eval.py generators \
        --model_name_or_path "$MODEL_NAME" \
        --model_input_template_path_or_name "hf" \
        --tasks "$TASKS" \
        --report_output_path "$RESULTS_DIR/metrics.json" \
        --save_individual_results_path "$RESULTS_DIR/all_results.json" \
        --use_defender_api \
        --batch_size 1 2>&1 | tee "$RESULTS_DIR/eval.log"
    rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] && echo "[3/3] AlpacaEval 2 done: $RESULTS_DIR" || echo "[3/3] AlpacaEval 2 FAILED (exit $rc), see $RESULTS_DIR"
)

echo "[DONE] API-based defender evaluation finished."
