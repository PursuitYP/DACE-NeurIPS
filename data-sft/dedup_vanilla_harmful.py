### dace: deduplicate vanilla_harmful jsonl by vanilla field, keep first occurrence ###
import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "sft_data_source_harmful.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "sft_data_source_harmful_dedup.jsonl")


def main():
    seen = set()
    kept_by_source = Counter()
    dropped_by_source = Counter()
    total_in = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total_in += 1
            vanilla = item.get("vanilla")
            source = item.get("data_source", "unknown")
            if vanilla in seen:
                dropped_by_source[source] += 1
                continue
            seen.add(vanilla)
            kept_by_source[source] += 1
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_out = sum(kept_by_source.values())
    total_dropped = sum(dropped_by_source.values())

    print(f"Input:  {total_in} records from {INPUT_FILE}")
    print(f"Output: {total_out} records to {OUTPUT_FILE}")
    print(f"Dropped (duplicates): {total_dropped}")
    print("\nKept by data_source:")
    for src, n in kept_by_source.items():
        print(f"  {src}: {n}")
    print("\nDropped by data_source:")
    for src, n in dropped_by_source.items():
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()


# Usage (from repo root): python data-sft/dedup_vanilla_harmful.py
