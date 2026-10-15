#!/usr/bin/env bash
### dace: v4 idempotent driver that finishes distilling benign+harmful vanilla prompts to CoT JSONL (12-risk strategy space + over-refusal oriented BENIGN_TEMPLATE; calls distill_v4_vanilla_*_jsonl.py) ###
#
# Usage:
#   bash data-sft/run_cot_distill_v4.sh               # clean failed rows + run both scripts in parallel
#   bash data-sft/run_cot_distill_v4.sh --no-clean    # do NOT remove answer=null / format-bad rows before running
#   bash data-sft/run_cot_distill_v4.sh --only benign # only run the benign pipeline
#   bash data-sft/run_cot_distill_v4.sh --only harmful
#
# Safe to run repeatedly. Each invocation:
#   1. (by default) strips records that are unusable for SFT and moves them to a .err file so the Python
#      distiller re-requests them next run. A row is dropped when ANY of the following holds:
#        - output is null/empty (BadRequestError from Gemini)
#        - answer extraction yielded empty (no <answer> block)
#        - output violates the <think><strategy><answer> format enforced by
#          format_reward_func_dace (src/verl/verl/utils/reward_score/game.py)
#        - the <strategy> block cannot be parsed as "risk category: X\nattack style: Y",
#          or the named risk category / attack style is outside the canonical DACE 12x10 space
#          (case-insensitive match; words must otherwise be verbatim)
#        - the parsed attack style does not match the record's assigned_style (directed-style drift)
#        - the <think> block leaks the assignment (meta-reference regex match)
#        - the line itself is not valid JSON
#   2. re-runs both python scripts, which internally dedup + drop 8192-truncated rows for regeneration
#   3. prints a completion summary
#
# Targets:
#   benign  : sft_data_source_benign.jsonl          (~20000 unique)   -> sft_data_cot_v4_benign.jsonl   (~20000 rows)
#   harmful : sft_data_source_harmful_dedup.jsonl   (~5794 unique)    -> sft_data_cot_v4_harmful.jsonl  (5794 x NUM_RUNS=4 ~= 23176 rows)

set -euo pipefail

### dace: resolve project root from this script's own location so it works from any cwd ###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CLEAN_ERRORS=1
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-clean) CLEAN_ERRORS=0; shift ;;
    --only)     ONLY="${2:-}"; shift 2 ;;
    -h|--help)
      grep -E "^#( |$)" "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p logs

### dace: distillation API is configured via env vars (see distill_v4_vanilla_*_jsonl.py) ###
if [[ -z "${DISTILL_API_KEY:-}" ]]; then
  echo "[distill] ERROR: please export DISTILL_API_KEY (and DISTILL_BASE_URL for an OpenAI-compatible gateway)." >&2
  exit 2
fi

### dace: activate conda env ${DACE_ENV:-dace}; fall back to current interpreter if conda is unavailable ###
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${DACE_ENV:-dace}"
fi

BENIGN_OUT="data-sft/sft_data_cot_v4_benign.jsonl"
HARMFUL_OUT="data-sft/sft_data_cot_v4_harmful.jsonl"

### dace: v4 clean step — drops rows that are null / malformed / off-space (12x10) / style-mismatched / think-leaked; dropped rows go to <out>.err and get re-requested by the Python distiller next run ###
clean_failed_rows() {
  python - "$BENIGN_OUT" "$HARMFUL_OUT" <<'PY'
import json, re, sys
from pathlib import Path

### dace: keep this regex/counter pair in sync with format_reward_func_dace in src/verl/verl/utils/reward_score/game.py ###
DACE_ORDER_PATTERN = re.compile(
    r"^<think>[\s\S]*?</think>\s*<strategy>[\s\S]*?</strategy>\s*<answer>[\s\S]*?</answer>$",
    re.DOTALL,
)
DACE_REQUIRED_COUNTS = (1, 1, 1, 1, 1, 1)

### dace: strategy regex mirrors extract_strategy_text in src/verl/verl/utils/reward_score/game.py ###
STRATEGY_PATTERN = re.compile(
    r"<strategy>\s*risk category:\s*(.+?)\s*\n\s*attack style:\s*(.+?)\s*</strategy>",
    re.DOTALL | re.IGNORECASE,
)

### dace: <think> extractor for anti-leak scan ###
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

### dace: anti-leak regex — 4 tight patterns that flag meta-references to the directed assignment while avoiding common innocuous phrases like 'required level of formality' or 'selected topic' ###
LEAK_PATTERNS = re.compile(
    r"("
    r"assigned (attack )?style"                                # "the assigned style" / "my assigned attack style"
    r"|I (was|am) (told|instructed|asked|required) to"         # "I was told to..." first-person meta
    r"|per (the|this) (instruction|constraint|request|task|specification)"  # "per the instruction"
    r"|(chosen|selected) for me"                                # "the style chosen for me"
    r")",
    re.IGNORECASE,
)

### dace: canonical 12 risk categories and 10 attack styles from the DACE strategy space (v4: pruned S12 Sexual Content + S14 Code Interpreter Abuse; see distill_v4_vanilla_*_jsonl.py / data/safety/preprocess_dace.py); case-insensitive match but words must be verbatim ###
RISK_CATEGORIES_CANONICAL = [
    "Violent Crimes",
    "Non-Violent Crimes",
    "Sex-Related Crimes",
    "Child Sexual Exploitation",
    "Defamation",
    "Specialized Advice",
    "Privacy",
    "Intellectual Property",
    "Indiscriminate Weapons",
    "Hate",
    "Suicide & Self-Harm",
    "Elections",
]
ATTACK_STYLES_CANONICAL = [
    "Slang",
    "Technical Terms",
    "Role Play",
    "Authority Manipulation",
    "Misspellings",
    "Word Play",
    "Emotional Manipulation",
    "Hypotheticals",
    "Historical Scenario",
    "Uncommon Dialects",
]
RISK_SET  = {s.casefold() for s in RISK_CATEGORIES_CANONICAL}
STYLE_SET = {s.casefold() for s in ATTACK_STYLES_CANONICAL}


def check_dace_format(text: str) -> bool:
    """Return True iff `text` satisfies format_reward_func_dace (exactly one ordered think/strategy/answer block)."""
    text = (text or "").strip()
    if not text:
        return False
    counts = (
        text.count("<think>"),    text.count("</think>"),
        text.count("<strategy>"), text.count("</strategy>"),
        text.count("<answer>"),   text.count("</answer>"),
    )
    if counts != DACE_REQUIRED_COUNTS:
        return False
    return DACE_ORDER_PATTERN.search(text) is not None


def parse_strategy(text: str):
    """Return (risk_cf, style_cf, reason) — reason is None on success, else tag identifying why it failed."""
    m = STRATEGY_PATTERN.search(text or "")
    if not m:
        return None, None, "bad_strategy_parse"
    risk_cf  = (m.group(1) or "").strip().casefold()
    style_cf = (m.group(2) or "").strip().casefold()
    if risk_cf not in RISK_SET:
        return risk_cf, style_cf, "bad_risk"
    if style_cf not in STYLE_SET:
        return risk_cf, style_cf, "bad_style"
    return risk_cf, style_cf, None


for arg in sys.argv[1:]:
    p = Path(arg)
    if not p.exists():
        print(f"[clean] {p.name}: not found, skip")
        continue

    kept, dropped = [], []
    stats = {
        "total": 0,
        "ok": 0,
        "bad_json": 0,               # 整行 JSON 解析失败
        "null_output": 0,            # BadRequestError 兜底：Gemini 调用失败，output=None/""
        "null_answer": 0,            # output 非空但抠不出 <answer> → answer=""
        "bad_format": 0,             # output 不满足 format_reward_func_dace 的三段式
        "bad_strategy_parse": 0,     # <strategy> 块缺失或非 "risk category: X / attack style: Y" 形式
        "bad_risk": 0,               # risk category 不在 canonical 12 之内
        "bad_style": 0,              # attack style 不在 canonical 10 之内
        "bad_style_mismatch": 0,     # 解析出的 attack style 与 record.assigned_style 不一致（directed-style drift）
        "bad_think_leak": 0,         # <think> 含 assignment 泄漏短语
    }

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["total"] += 1

        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            dropped.append(line)
            continue

        output = r.get("output")
        answer = r.get("answer")

        if output in (None, ""):
            stats["null_output"] += 1
            dropped.append(line)
            continue
        if answer in (None, ""):
            stats["null_answer"] += 1
            dropped.append(line)
            continue
        if not check_dace_format(output):
            stats["bad_format"] += 1
            dropped.append(line)
            continue

        _, parsed_style_cf, strat_reason = parse_strategy(output)
        if strat_reason is not None:
            stats[strat_reason] += 1
            dropped.append(line)
            continue

        ### dace: directed-style drift check — attack style must match record.assigned_style (case-insensitive) ###
        assigned_style_cf = (r.get("assigned_style") or "").strip().casefold()
        if not assigned_style_cf or parsed_style_cf != assigned_style_cf:
            stats["bad_style_mismatch"] += 1
            dropped.append(line)
            continue

        ### dace: anti-leak — <think> must not reveal that the style was assigned ###
        think_match = THINK_PATTERN.search(output)
        think_text = think_match.group(1) if think_match else ""
        if LEAK_PATTERNS.search(think_text):
            stats["bad_think_leak"] += 1
            dropped.append(line)
            continue

        stats["ok"] += 1
        kept.append(line)

    p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    err_path = p.with_suffix(p.suffix + ".err")
    if dropped:
        err_path.write_text("\n".join(dropped) + "\n", encoding="utf-8")

    removed = len(dropped)
    err_note = f" -> {err_path.name}" if dropped else ""
    print(
        f"[clean] {p.name}: total={stats['total']}, kept={stats['ok']}, removed={removed} "
        f"(null_output={stats['null_output']}, null_answer={stats['null_answer']}, "
        f"bad_format={stats['bad_format']}, bad_strategy_parse={stats['bad_strategy_parse']}, "
        f"bad_risk={stats['bad_risk']}, bad_style={stats['bad_style']}, "
        f"bad_style_mismatch={stats['bad_style_mismatch']}, bad_think_leak={stats['bad_think_leak']}, "
        f"bad_json={stats['bad_json']}){err_note}"
    )
PY
}

### dace: report per-file completion counts (ok = answer non-empty; err = answer null/missing) ###
summarize() {
  python - "$BENIGN_OUT" "$HARMFUL_OUT" <<'PY'
import json, sys
from pathlib import Path
for arg in sys.argv[1:]:
    p = Path(arg)
    if not p.exists():
        print(f"[summary] {p.name}: not found")
        continue
    ok = err = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            err += 1
            continue
        (ok := ok + 1) if r.get("answer") else (err := err + 1)
    print(f"[summary] {p.name}: ok={ok}, err_or_null={err}")
PY
}

if [[ "$CLEAN_ERRORS" -eq 1 ]]; then
  clean_failed_rows
else
  echo "[clean] skipped (--no-clean)"
fi

PIDS=()
if [[ -z "$ONLY" || "$ONLY" == "benign" ]]; then
  echo "[run] launching benign distillation (v4) -> logs/distill_v4_benign.log"
  python data-sft/distill_v4_vanilla_benign_jsonl.py > logs/distill_v4_benign.log 2>&1 &
  PIDS+=($!)
fi
if [[ -z "$ONLY" || "$ONLY" == "harmful" ]]; then
  echo "[run] launching harmful distillation (v4) -> logs/distill_v4_harmful.log"
  python data-sft/distill_v4_vanilla_harmful_jsonl.py > logs/distill_v4_harmful.log 2>&1 &
  PIDS+=($!)
fi

STATUS=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "[run] pid=$pid finished OK"
  else
    rc=$?
    echo "[run] pid=$pid FAILED with exit=$rc" >&2
    STATUS=$rc
  fi
done

summarize
exit "$STATUS"
