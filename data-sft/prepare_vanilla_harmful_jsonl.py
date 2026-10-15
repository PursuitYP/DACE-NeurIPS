### dace: build vanilla_harmful jsonl by concatenating goal fields from three SFT/RL sources ###
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    ("wildteam_with_think-v5.json", "JBR1-SFT"),
    ("warmup_attack_target.json", "JBR1-RL-S1"),
    ("train_target_total_5062.json", "JBR1-RL-S2"),
]

OUTPUT_FILE = os.path.join(BASE_DIR, "sft_data_source_harmful.jsonl")


def load_goals(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["goal"] for item in data if "goal" in item]


def main():
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for filename, data_source in SOURCES:
            path = os.path.join(BASE_DIR, filename)
            goals = load_goals(path)
            for goal in goals:
                record = {
                    "vanilla": goal,
                    "data_source": data_source,
                    "data_type": "vanilla_harmful",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{data_source}] {filename}: {len(goals)} records")
            total += len(goals)
    print(f"Total: {total} records written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


# Usage (from repo root): python data-sft/prepare_vanilla_harmful_jsonl.py
