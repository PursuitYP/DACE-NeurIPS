### dace: v4 convert — transform distilled CoT jsonl (v4) into Alpaca-style JSON for 360-LLaMA-Factory SFT ###
"""
v4 Convert: directly produce Alpaca-style SFT JSON from v4 distilled CoT JSONL.

Unlike v2/v3 (which composed v1 rare-risk + v2 directed-style samples), v4 is
distilled end-to-end under the 12-risk strategy space + over-refusal BENIGN_TEMPLATE,
so no cross-version compose is needed.

Inputs  (defaults, same dir as this script):
    sft_data_cot_v4_benign.jsonl
    sft_data_cot_v4_harmful.jsonl

Outputs (defaults in src/360-LLaMA-Factory/data/ so LLaMA-Factory can find them):
    game_cot_dace_v4_benign.json    (Alpaca 5-key, benign-only, dedup by input)
    game_cot_dace_v4_harmful.json   (Alpaca 5-key, harmful-only, dedup by (input, output) — preserves NUM_RUNS multi-pass CoT)
    game_cot_dace_v4_all.json       (Alpaca 5-key, concat benign + harmful)

Alpaca 5-key schema (matches existing game_cot_* registrations in
src/360-LLaMA-Factory/data/dataset_info.json):
    instruction / input / output / answer / system

Dedup policy:
    - benign  : dedup by `input` (same vanilla → single rewrite)
    - harmful : dedup by (`input`, `output`) (preserves the NUM_RUNS=4 multi-pass CoT
                diversity from distill_v4_vanilla_harmful_jsonl.py)

Drop policy (both sides):
    - drop rows where output / answer / input is null/empty (Gemini BadRequestError)
      unless --no-require-answer is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_KEYS = ("instruction", "input", "output", "answer", "system")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return records


def _slim(rec: dict) -> dict:
    """Pick only the 5 Alpaca keys, normalize missing → empty string."""
    return {k: (rec.get(k) if rec.get(k) is not None else "") for k in TARGET_KEYS}


def is_usable(rec: dict) -> bool:
    return bool(rec.get("input")) and bool(rec.get("output")) and bool(rec.get("answer"))


def dedup_benign(records: list[dict], require_answer: bool) -> tuple[list[dict], dict]:
    """benign: dedup by `input` (first occurrence wins)."""
    stats = {"total": 0, "kept": 0, "drop_empty": 0, "drop_dup": 0}
    out: list[dict] = []
    seen: set[str] = set()
    for rec in records:
        stats["total"] += 1
        if require_answer and not is_usable(rec):
            stats["drop_empty"] += 1
            continue
        key = rec.get("input") or ""
        if key in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(key)
        out.append(rec)
        stats["kept"] += 1
    return out, stats


def dedup_harmful(records: list[dict], require_answer: bool) -> tuple[list[dict], dict]:
    """harmful: dedup by (input, output) — preserves NUM_RUNS multi-pass CoT."""
    stats = {"total": 0, "kept": 0, "drop_empty": 0, "drop_dup": 0}
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        stats["total"] += 1
        if require_answer and not is_usable(rec):
            stats["drop_empty"] += 1
            continue
        key = (rec.get("input") or "", rec.get("output") or "")
        if key in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(key)
        out.append(rec)
        stats["kept"] += 1
    return out, stats


def write_json_array(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [_slim(r) for r in records]
    path.write_text(json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(slim)


def _format_stats(label: str, stats: dict) -> str:
    return (
        f"[{label}] total={stats['total']} kept={stats['kept']} "
        f"drop_empty={stats['drop_empty']} drop_dup={stats['drop_dup']}"
    )


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    lf_data = here.parent / "src" / "360-LLaMA-Factory" / "data"
    p = argparse.ArgumentParser(
        description="v4 convert: distilled CoT v4 JSONL → Alpaca-style SFT JSON (no compose).",
    )
    # inputs
    p.add_argument("--benign-src",  type=Path, default=here / "sft_data_cot_v4_benign.jsonl")
    p.add_argument("--harmful-src", type=Path, default=here / "sft_data_cot_v4_harmful.jsonl")
    # outputs (default to LLaMA-Factory data dir)
    p.add_argument("--benign-out",  type=Path, default=lf_data / "game_cot_dace_v4_benign.json")
    p.add_argument("--harmful-out", type=Path, default=lf_data / "game_cot_dace_v4_harmful.json")
    p.add_argument("--merged-out",  type=Path, default=lf_data / "game_cot_dace_v4_all.json")
    # flags
    p.add_argument("--only", choices=["benign", "harmful", "both"], default="both")
    p.add_argument(
        "--no-require-answer", action="store_true",
        help="Keep rows even when output/answer/input is empty (default: drop them).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    require_answer = not args.no_require_answer

    benign_out: list[dict] = []
    harmful_out: list[dict] = []

    if args.only in ("benign", "both"):
        b_recs = load_jsonl(args.benign_src)
        benign_out, bs = dedup_benign(b_recs, require_answer)
        b_rows = write_json_array(args.benign_out, benign_out)
        print(_format_stats("benign",  bs))
        print(f"  -> {args.benign_out}  ({b_rows} rows)")

    if args.only in ("harmful", "both"):
        h_recs = load_jsonl(args.harmful_src)
        harmful_out, hs = dedup_harmful(h_recs, require_answer)
        h_rows = write_json_array(args.harmful_out, harmful_out)
        print(_format_stats("harmful", hs))
        print(f"  -> {args.harmful_out} ({h_rows} rows)")

    if args.only == "both":
        merged = benign_out + harmful_out
        m_rows = write_json_array(args.merged_out, merged)
        print(f"[merged] total={len(merged)} (benign={len(benign_out)} + harmful={len(harmful_out)})")
        print(f"  -> {args.merged_out}  ({m_rows} rows)")


if __name__ == "__main__":
    main()


### dace: v4 example usage ###
"""Example usage:

# Default — reads v4 jsonl from data-sft/, writes Alpaca json into src/360-LLaMA-Factory/data/
python data-sft/convert_v4_cot_to_game_format.py

# Custom outputs (e.g. stay in data-sft/ for local inspection first)
python data-sft/convert_v4_cot_to_game_format.py \
    --benign-out  data-sft/game_cot_dace_v4_benign.json \
    --harmful-out data-sft/game_cot_dace_v4_harmful.json \
    --merged-out  data-sft/game_cot_dace_v4_all.json

python data-sft/convert_v4_cot_to_game_format.py \
    --benign-out  src/360-LLaMA-Factory/data/game_cot_dace_v4_benign.json \
    --harmful-out src/360-LLaMA-Factory/data/game_cot_dace_v4_harmful.json \
    --merged-out  src/360-LLaMA-Factory/data/game_cot_dace_v4_all.json

# Only one side
python data-sft/convert_v4_cot_to_game_format.py --only benign
"""
