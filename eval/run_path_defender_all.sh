#!/usr/bin/env bash
### dace: evaluations that need a local (merged HF) defender checkpoint — Table 1 safety, Table 2 ARC-C / GPQA / MMLU ###
#
#   [1/3] safety-eval-fork : WildGuardTest / WildJailbreak / HarmBench / DAN / OR-Bench / XSTest (judge: Qwen3Guard)
#   [2/3] olmes            : ARC-Challenge / GPQA
#   [3/3] safety-eval-fork : MMLU
#
# Usage:
#   MODEL_PATH=checkpoints/Game-separated/<exp>/global_step_300/defender/actor/huggingface \
#   WILDGUARD_API_ENDPOINT=http://<GUARD_HOST>:<PORT>/v1 \
#   bash eval/run_path_defender_all.sh
#
# Each stage runs in its own subshell / conda env; a failing stage does not stop the next one.
set -o pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:?please set MODEL_PATH (merged HF checkpoint of the defender)}"
[[ -d "$MODEL_PATH" ]] && MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"   # local dir -> absolute path (HF hub ids are kept as-is)
if [[ -z "${MODEL_NAME:-}" ]]; then
  # .../<experiment_name>/global_step_<N>/defender/actor/huggingface -> <experiment_name>-step<N>
  if [[ "$(basename "$MODEL_PATH")" == huggingface ]]; then
    _step_dir="$(dirname "$(dirname "$(dirname "$MODEL_PATH")")")"
    MODEL_NAME="$(basename "$(dirname "$_step_dir")")-step$(basename "$_step_dir" | sed 's/global_step_//')"
  else
    MODEL_NAME="$(basename "$MODEL_PATH")"
  fi
fi
MODEL_TEMPLATE=${MODEL_TEMPLATE:-game_defender}   # chat template used for Table 1 (all DACE backbones)
# Note: Table 1 judges the full defender output (answer extraction disabled via --no_extract_answer True).
SAFETY_EVAL_ENV=${SAFETY_EVAL_ENV:-safety-eval}
OLMES_ENV=${OLMES_ENV:-olmes}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

export WILDGUARD_API_ENDPOINT=${WILDGUARD_API_ENDPOINT:-http://<GUARD_HOST>:<PORT>/v1}
export WILDGUARD_API_KEY=${WILDGUARD_API_KEY:-EMPTY}
if [[ "$WILDGUARD_API_ENDPOINT" == *"<"* ]]; then
  echo "[eval] ERROR: please export WILDGUARD_API_ENDPOINT (Qwen3Guard served with --served-model-name orm)." >&2
  exit 2
fi
export HF_DATASETS_VERBOSITY=info
# Optional: export HF_TOKEN=<your_hf_token>   (GPQA is a gated dataset)

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
activate() { command -v conda >/dev/null 2>&1 || return 0; conda activate "$1" || { echo "[eval] ERROR: conda env '$1' not found" >&2; exit 1; }; }

echo "=========================================================="
echo "[PATH-DEFENDER] MODEL_PATH = $MODEL_PATH"
echo "[PATH-DEFENDER] MODEL_NAME = $MODEL_NAME"
echo "=========================================================="

# ------------------------------------------------------------
# [1/3] Table 1 · safety benchmarks (safety-eval-fork)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/safety-eval-fork"
    activate "$SAFETY_EVAL_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    export OPENAI_API_KEY=${OPENAI_API_KEY:-fake-key-for-safety-eval}
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    TASKS="wildguardtest,harmbench_precompute,wildjailbreak:benign,wildjailbreak:harmful,do_anything_now,harmbench,or_bench:hard-1k,or_bench:toxic,xstest"
    CLASSIFIER="Qwen3GuardAPI"
    RESULTS_DIR="$PWD/results/$MODEL_NAME-$CLASSIFIER-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    timeout 10800 python -u evaluation/eval.py generators \
        --model_name_or_path "$MODEL_PATH" \
        --model_input_template_path_or_name "$MODEL_TEMPLATE" \
        --tasks "$TASKS" \
        --report_output_path "$RESULTS_DIR/metrics.json" \
        --save_individual_results_path "$RESULTS_DIR/all_results.json" \
        --use_vllm \
        --classifier_model_name "$CLASSIFIER" \
        --no_extract_answer True \
        --batch_size 1 2>&1 | tee "$RESULTS_DIR/eval.log"
    rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] && echo "[1/3] safety done: $RESULTS_DIR" || echo "[1/3] safety FAILED (exit $rc), see $RESULTS_DIR"
)

# ------------------------------------------------------------
# [2/3] Table 2 · ARC-C / GPQA (olmes)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/olmes"
    activate "$OLMES_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    # Tip (offline nodes): point tiktoken to a pre-populated cache, e.g. litellm's bundled cl100k_base:
    # export TIKTOKEN_CACHE_DIR=$CONDA_PREFIX/lib/python3.10/site-packages/litellm/litellm_core_utils/tokenizers

    RESULTS_BASE_DIR="$PWD/results/$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
    failed=0
    for TASK in arc_challenge gpqa; do
        case $TASK in
            arc_challenge) TASK_ARGS="--task arc_challenge --model-type hf --batch-size 16" ;;
            gpqa)          TASK_ARGS="--task gpqa --model-type hf --batch-size 4" ;;
        esac
        TASK_RESULTS_DIR="$RESULTS_BASE_DIR/$TASK"
        mkdir -p "$TASK_RESULTS_DIR"
        timeout 7200 python -m oe_eval.run_eval \
            --model-path "$MODEL_PATH" \
            $TASK_ARGS \
            --output-dir "$TASK_RESULTS_DIR" 2>&1 | tee "$TASK_RESULTS_DIR/eval.log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { failed=1; echo "[2/3] $TASK FAILED, see $TASK_RESULTS_DIR"; }
    done
    [[ $failed -eq 0 ]] && echo "[2/3] olmes done: $RESULTS_BASE_DIR" || echo "[2/3] olmes finished with failures: $RESULTS_BASE_DIR"
)

# ------------------------------------------------------------
# [3/3] Table 2 · MMLU (safety-eval-fork)
# ------------------------------------------------------------
(
    cd "$EVAL_DIR/safety-eval-fork"
    activate "$SAFETY_EVAL_ENV"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

    TASKS="mmlu"
    RESULTS_DIR="$PWD/results/$TASKS-$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    python -u evaluation/eval.py generators \
        --model_name_or_path "$MODEL_PATH" \
        --model_input_template_path_or_name "hf" \
        --tasks "$TASKS" \
        --report_output_path "$RESULTS_DIR/metrics.json" \
        --save_individual_results_path "$RESULTS_DIR/all_results.json" \
        --batch_size 1 2>&1 | tee "$RESULTS_DIR/eval.log"
    rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] && echo "[3/3] MMLU done: $RESULTS_DIR" || echo "[3/3] MMLU FAILED (exit $rc), see $RESULTS_DIR"
)

echo "[DONE] path-based defender evaluation finished."
