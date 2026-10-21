#!/usr/bin/env bash
### dace: Phase-2 DACE co-evolution launcher (attacker/defender GRPO + coverage-gain diversity reward + Bayesian replay pool) ###
#
# Usually called through one of the backbone wrappers:
#   bash scripts/rl/train_dace_qwen2.5_7b.sh
#   bash scripts/rl/train_dace_llama3.1_8b.sh
#   bash scripts/rl/train_dace_qwen2.5_14b.sh
#
# Required env vars:
#   ATTACKER_MODEL_PATH      Phase-1 SFT attacker (output_dir of the Phase-1 SFT yaml)
#   DEFENDER_MODEL_PATH      defender base model (HF hub id or local path)
#   WILDGUARD_API_ENDPOINT   OpenAI-compatible endpoint of the safety judge (Qwen3Guard served as "orm")
# Optional env vars:
#   EXP_NAME, START_AGENT (attacker|defender), CHECKPOINT_DIR, NUM_GPUS, WANDB_API_KEY, WANDB_MODE, ...
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

: "${ATTACKER_MODEL_PATH:?please set ATTACKER_MODEL_PATH (Phase-1 SFT attacker)}"
: "${DEFENDER_MODEL_PATH:?please set DEFENDER_MODEL_PATH (defender base model)}"
EXP_NAME=${EXP_NAME:-DACE}
START_AGENT=${START_AGENT:-attacker}            # attacker-first (paper default) | defender
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints}
NUM_GPUS=${NUM_GPUS:-8}                         # split evenly into an attacker pool and a defender pool
GPUS_PER_AGENT=$((NUM_GPUS / 2))
RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}

# ---- safety judge (Qwen3Guard-Gen served with vLLM, --served-model-name orm) ----
export WILDGUARD_API_ENDPOINT=${WILDGUARD_API_ENDPOINT:-http://<GUARD_HOST>:<PORT>/v1}
export WILDGUARD_API_KEY=${WILDGUARD_API_KEY:-EMPTY}
if [[ "$WILDGUARD_API_ENDPOINT" == *"<"* ]]; then
  echo "[DACE] ERROR: please export WILDGUARD_API_ENDPOINT=http://<host>:<port>/v1 (see README: Judge Model)." >&2
  exit 2
fi

# ---- reward shaping (values used in the paper) ----
export SAFETY_SCORE_MODE=classifier
export REWARD_HARM=1
export REWARD_REFUSAL=0.5
export REWARD_LABEL=0
export USE_Q_SAFE_LABEL_FOR_REFUSAL=1
export REWARD_MODEL_USE_ANSWER_TAGS=0
export FORMAT_REWARD_VALUE=1
export ENABLE_REVISION_REWARD=0          # must be disabled explicitly (code default is 1)
export ENABLE_DEFENDER_QUALITY_REWARD=0  # must be disabled explicitly (code default is 1)
export REWARD_SCORE_MAX_WORKERS=${REWARD_SCORE_MAX_WORKERS:-1}
export REWARD_SCORE_TIMEOUT_S=${REWARD_SCORE_TIMEOUT_S:-30}

# ---- agent switching ----
ATTACKER_RATIO=${ATTACKER_RATIO:-1}
DEFENDER_RATIO=${DEFENDER_RATIO:-1}
UPDATE_FREQ=${UPDATE_FREQ:-15}
SWITCH_MODE=${SWITCH_MODE:-ratio}               # ratio | metric
SWITCH_METRIC_NAME=${SWITCH_METRIC_NAME:-reward/response_harm}
SWITCH_METRIC_LOW=${SWITCH_METRIC_LOW:-0.80}
SWITCH_METRIC_HIGH=${SWITCH_METRIC_HIGH:-0.90}
SWITCH_METRIC_WINDOW=${SWITCH_METRIC_WINDOW:-3}

# ---- logging (W&B key is only read from your environment) ----
export WANDB_MODE=${WANDB_MODE:-offline}

export PYTHONPATH=$PROJECT_DIR/src/verl:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}

if [[ ! -f data/safety/train_dace.parquet || ! -f data/safety/test_wjb_dace.parquet ]]; then
  echo "[DACE] ERROR: RL data not found. Run first:" >&2
  echo "  python data/safety/preprocess_dace.py --local_dir data/safety --write_test_parquet" >&2
  exit 2
fi

timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
project_name=game
experiment_name="${EXP_NAME}-${START_AGENT}1st-${timestamp}"
mkdir -p "logs/${project_name}"

ray stop --force
ray start --head --port="$RAY_MASTER_PORT" --num-gpus "$NUM_GPUS"
sleep 30

SWITCH_ARGS=()
if [[ "${SWITCH_MODE}" == "metric" ]]; then
    SWITCH_ARGS=(
        "algorithm.switch_agent.mode=metric"
        "+algorithm.switch_agent.metric_name=${SWITCH_METRIC_NAME}"
        "+algorithm.switch_agent.metric_low=${SWITCH_METRIC_LOW}"
        "+algorithm.switch_agent.metric_high=${SWITCH_METRIC_HIGH}"
        "+algorithm.switch_agent.metric_window=${SWITCH_METRIC_WINDOW}"
    )
else
    SWITCH_ARGS=(
        "algorithm.switch_agent.mode=ratio"
        "algorithm.switch_agent.freq=${UPDATE_FREQ}"
        "algorithm.switch_agent.update_ratio={}"
        "+algorithm.switch_agent.update_ratio.attacker=${ATTACKER_RATIO}"
        "+algorithm.switch_agent.update_ratio.defender=${DEFENDER_RATIO}"
    )
fi

PYTHONUNBUFFERED=1 python -m verl.separated_trainer.main_ppo \
    trainer.project_name=Game-separated \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/Game-separated/${experiment_name}" \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${GPUS_PER_AGENT} \
    data.train_files=data/safety/train_dace.parquet \
    data.val_files=data/safety/test_wjb_dace.parquet \
    data.val_batch_size=256 \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=6144 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.rollout.max_num_turns=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.stop_when_truncated=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.stop_when_truncated_roles=[defender] \
    +trainer.val_before_train=False \
    +trainer.val_only=False \
    +trainer.save_val_generations=True \
    +trainer.save_train_generations=True \
    trainer.test_freq=15 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=300 \
    algorithm.adv_estimator=grpo \
    algorithm.switch_agent.model_paths=[${ATTACKER_MODEL_PATH},${DEFENDER_MODEL_PATH}] \
    algorithm.switch_agent.agent_roles=[attacker,defender] \
    algorithm.switch_agent.train_roles=[attacker,defender] \
    algorithm.switch_agent.start_agent=${START_AGENT} \
    "${SWITCH_ARGS[@]}" \
    reward_model.reward_manager=game \
    reward_model.mask_unfinished_reward=True \
    +reward_model.use_format_reward=True \
    +reward_model.format_reward_roles=[attacker] \
    algorithm.filter_groups.enable=False \
    trainer.logger=[console,wandb] \
    algorithm.diversity.enable=True \
    algorithm.diversity.use_dace_format=True \
    algorithm.diversity.lambda_success=1.0 \
    algorithm.diversity.lambda_fail=0.5 \
    algorithm.diversity.epsilon=1e-8 \
    algorithm.replay_pool.enable=True \
    algorithm.replay_pool.replay_batch_size=32 \
    algorithm.replay_pool.gamma_decay=0.90 \
    algorithm.replay_pool.prune_threshold=0.45 \
    algorithm.replay_pool.prune_min_trials=2.0 \
    algorithm.replay_pool.max_pool_size=4000 \
    algorithm.replay_pool.alpha_prior=1.0 \
    algorithm.replay_pool.beta_prior=1.0 \
    algorithm.replay_pool.clip_negative=True \
    2>&1 | tee "${PROJECT_DIR}/logs/${project_name}/${experiment_name}.log"

ray stop --force
