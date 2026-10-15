# DACE Evaluation

This directory reproduces the evaluation of the paper (Tables 1–3). Each tool runs in its own conda environment:

- **safety-eval-fork** (fork of [Ai2 safety-eval](https://github.com/allenai/safety-eval)) — Table 1 safety benchmarks, MMLU and AlpacaEval 2
- **olmes** ([OLMES](https://github.com/allenai/olmes)) — ARC-Challenge, GPQA and IFEval
- **OpenRT** ([OpenRT](https://github.com/AI45Lab/OpenRT)) — Table 3 OOD automated red-teaming (with DACE additions in `unified_eval.py` / `dace_prompts.py`)

| Paper table | Benchmarks | Script |
|---|---|---|
| Table 1 | WildGuardTest, WildJailbreak, HarmBench (vanilla / adversarial), DAN, OR-Bench-Toxic, XSTest | `run_path_defender_all.sh` (stage 1/3) |
| Table 1 | StrongREJECT | `run_api_defender_all.sh` (stage 1/3) |
| Table 2 | ARC-C, GPQA, MMLU | `run_path_defender_all.sh` (stages 2/3, 3/3) |
| Table 2 | IFEval, AlpacaEval 2 | `run_api_defender_all.sh` (stages 2/3, 3/3) |
| Table 3 | no-rev, PAIR, TAP, AutoDAN, AutoDAN-Turbo | `OpenRT/run_eval.sh` |
| Table 3 | GCG | `OpenRT/run_gcg.sh` + `OpenRT/run_judge_gcg.sh` |
| Table 3 | Self-Eval / SmoothLLM baselines | `OpenRT/run_self_eval.sh`, `OpenRT/run_smoothllm_eval.sh` |

## Environment setup

### Safety Eval
```bash
cd eval/safety-eval-fork
conda create -n safety-eval python=3.10
conda activate safety-eval
pip install -e .
pip install -r requirements.txt   # includes vllm==0.8.5
```

### OLMES
```bash
cd eval/olmes
conda create -n olmes python=3.10
conda activate olmes
pip install -e .
```

### OpenRT
```bash
cd eval/OpenRT
conda create -n OpenRT python=3.10
conda activate OpenRT
pip install -e .
```

The wrapper scripts activate these environments by name; override with `SAFETY_EVAL_ENV`, `OLMES_ENV` if you use different names.

## Preparation

1. **Merge the defender checkpoint** (FSDP shards -> HuggingFace):
   ```bash
   CKPT_DIR=checkpoints/Game-separated/<experiment_name> STEPS=300 ROLES=defender bash scripts/model_merge.sh
   # -> checkpoints/Game-separated/<experiment_name>/global_step_300/defender/actor/huggingface
   ```
2. **Serve the safety judge** (Qwen3Guard) for Table 1:
   ```bash
   vllm serve Qwen/Qwen3Guard-Gen-8B --served-model-name orm --port <PORT>
   export WILDGUARD_API_ENDPOINT=http://<GUARD_HOST>:<PORT>/v1
   ```
3. **Serve the defender** for the API-based evaluations and OpenRT, and (for PAIR / TAP / AutoDAN-Turbo) an attacker LLM on another port:
   ```bash
   vllm serve <merged_defender_dir> --served-model-name orm --port <PORT>
   vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name orm --port <PORT2>   # OpenRT attacker
   ```
   Use `DEFENDER_API_MODEL` / `ATTACKER_API_MODEL` if you serve them under other names.
4. **GPT-4o judge** (StrongREJECT autograder, AlpacaEval 2 annotator, OpenRT judge): set `OPENAI_API_KEY` and, for an OpenAI-compatible gateway, `OPENAI_BASE_URL`.

## Table 1 & Table 2

```bash
# local checkpoint: safety benchmarks (Qwen3Guard judge) + ARC-C / GPQA + MMLU
MODEL_NAME=DACE-Qwen2.5-7B \
MODEL_PATH=checkpoints/Game-separated/<experiment_name>/global_step_300/defender/actor/huggingface \
WILDGUARD_API_ENDPOINT=http://<GUARD_HOST>:<PORT>/v1 \
bash eval/run_path_defender_all.sh

# defender API: StrongREJECT + IFEval + AlpacaEval 2
MODEL_NAME=DACE-Qwen2.5-7B \
DEFENDER_API_BASE_URL=http://<DEFENDER_HOST>:<PORT>/v1 \
OPENAI_API_KEY=<API_KEY> \
bash eval/run_api_defender_all.sh
```

Notes:

- Table 1 uses the `game_defender` chat template for all DACE defenders (`MODEL_TEMPLATE`); MMLU, StrongREJECT and AlpacaEval 2 use the `hf` template.
- GPQA is a gated HF dataset: `export HF_TOKEN=<your_hf_token>` beforehand. MMLU data is downloaded automatically on first use.
- Results are written to `eval/safety-eval-fork/results/` and `eval/olmes/results/`.
- Some resources are fetched on first use: MMLU data, ARC-C / GPQA / IFEval from the HF Hub, AlpacaEval's `df_gamed.csv` (`tatsu-lab/alpaca_eval`) and NLTK `punkt` for IFEval. On offline nodes, pre-download them and set `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`; `tiktoken` may also hang, so point `TIKTOKEN_CACHE_DIR` at a pre-populated cache (e.g. litellm's bundled `cl100k_base` for olmes).

## Table 3: OOD automated red-teaming (OpenRT)

All attacks use the 320 HarmBench test behaviors in `OpenRT/seed/harmbench/` and a GPT-4o judge (score 1–5, success threshold 5).
Run these scripts inside the `OpenRT` environment (`conda activate OpenRT`).

```bash
# black-box attacks: no_revision, pair (PAIR), tree (TAP), autodan, autodan_turbo_r (AutoDAN-Turbo)
MODEL_NAME=DACE-Qwen2.5-7B \
DEFENDER_API_BASE_URL=http://<DEFENDER_HOST>:<PORT>/v1 \
ATTACKER_API_BASE_URL=http://<ATTACKER_HOST>:<PORT>/v1 \
OPENAI_API_KEY=<API_KEY> \
bash eval/OpenRT/run_eval.sh

# white-box GCG: optimizes suffixes on LOCAL_MODEL_PATH and queries the served defender; then judge the GCG histories
LOCAL_MODEL_PATH=<merged_defender_dir> MODEL_NAME=DACE-Qwen2.5-7B \
DEFENDER_API_BASE_URL=... OPENAI_API_KEY=... \
bash eval/OpenRT/run_gcg.sh
OPENAI_API_KEY=... bash eval/OpenRT/run_judge_gcg.sh eval/OpenRT/results/gcg-<MODEL_NAME>-<timestamp>/gcg

# inference-time defense baselines (serve the base instruct model as the target; GCG column: ATTACKS=gcg LOCAL_MODEL_PATH=<base model>)
DEFENDER_API_BASE_URL=... ATTACKER_API_BASE_URL=... OPENAI_API_KEY=... bash eval/OpenRT/run_self_eval.sh
DEFENDER_API_BASE_URL=... ATTACKER_API_BASE_URL=... OPENAI_API_KEY=... bash eval/OpenRT/run_smoothllm_eval.sh
```

Notes:

- PAIR, TAP and AutoDAN-Turbo need the attacker LLM behind `ATTACKER_API_BASE_URL`; no-rev, AutoDAN and GCG only need the target and the judge.
- Run settings follow our experiments: `MAX_TURNS=3`, `JUDGE_THRESHOLD=5`, `--attacker-answer-extract`, embeddings `text-embedding-3-small`. All of them can be overridden by env vars (see the top of `run_eval.sh`).
- Self-Eval / SmoothLLM are enabled by prefixing the target model name (`selfeval-<model>`, `smoothllm-<model>`).
- Results are written to `eval/OpenRT/results/`; relative paths passed via env vars are resolved against your current directory.

## Acknowledgements

```bibtex
@article{wang2026openrt,
  title={OpenRT: An Open-Source Red Teaming Framework for Multimodal LLMs},
  author={Wang, Xin and Chen, Yunhao and Li, Juncheng and Wang, Yixu and Yao, Yang and Gu, Tianle and Li, Jie and Teng, Yan and Wang, Yingchun and Hu, Xia},
  journal={arXiv preprint arXiv:2601.01592},
  year={2026}
}

@inproceedings{gu2025olmes,
  title={Olmes: A standard for language model evaluations},
  author={Gu, Yuling and Tafjord, Oyvind and Kuehl, Bailey and Haddad, Dany and Dodge, Jesse and Hajishirzi, Hannaneh},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={5005--5033},
  year={2025}
}

@misc{wildteaming2024,
  title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
  author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Yejin Choi and Nouha Dziri},
  year={2024},
  eprint={2406.18510},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{wildguard2024,
  title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs},
  author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
  year={2024},
  eprint={2406.18495},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
