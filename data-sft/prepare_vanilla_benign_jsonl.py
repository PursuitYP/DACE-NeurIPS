### dace: build vanilla_benign jsonl from WJB-Benign vanilla fields ###
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

### dace: reuse the bundled WJB benign seeds under data/safety ###
INPUT_FILE = os.path.join(BASE_DIR, "..", "data", "safety", "vanilla_benign_dataset_origin.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "sft_data_source_benign.jsonl")

DATA_SOURCE = "WJB-Benign"
DATA_TYPE = "vanilla_benign"


def main():
    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vanilla = item.get("vanilla")
            if vanilla is None:
                continue
            record = {
                "vanilla": vanilla,
                "data_source": DATA_SOURCE,
                "data_type": DATA_TYPE,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
    print(f"[{DATA_SOURCE}] {total} records written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


# Usage (from repo root): python data-sft/prepare_vanilla_benign_jsonl.py
