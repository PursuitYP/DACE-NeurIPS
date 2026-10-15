#!/usr/bin/env bash
### dace: merge FSDP actor shards of DACE checkpoints into HuggingFace format (needed before evaluation) ###
#
# Usage:
#   CKPT_DIR=checkpoints/Game-separated/<experiment_name> bash scripts/model_merge.sh
#   CKPT_DIR=... STEPS="250 300" ROLES="attacker defender" bash scripts/model_merge.sh
#
# The merged model of <role> at <step> is written to:
#   $CKPT_DIR/global_step_<step>/<role>/actor/huggingface
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGER_SCRIPT="$PROJECT_DIR/src/verl/scripts/model_merger.py"

: "${CKPT_DIR:?please set CKPT_DIR, e.g. checkpoints/Game-separated/<experiment_name>}"
STEPS=${STEPS:-300}
ROLES=${ROLES:-defender}

failed=0
for step in $STEPS; do
  for role in $ROLES; do
    actor_dir="$CKPT_DIR/global_step_${step}/${role}/actor"
    if [[ ! -d "$actor_dir" ]]; then
      echo "[merge] skip: $actor_dir not found" >&2
      failed=1
      continue
    fi
    echo "[merge] step=${step} role=${role} -> ${actor_dir}/huggingface"
    if ! PYTHONPATH="$PROJECT_DIR/src/verl${PYTHONPATH:+:$PYTHONPATH}" python3 "$MERGER_SCRIPT" --local_dir "$actor_dir"; then
      echo "[merge] failed: $actor_dir" >&2
      failed=1
    fi
  done
done
exit $failed
