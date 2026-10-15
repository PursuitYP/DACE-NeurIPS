import abc
import json
import os
import re
import time
from abc import ABC

from alpaca_eval import evaluate as alpaca_farm_evaluate
import openai

from evaluation.schemas import OpenEndedTaskBase, GeneratorModelBase


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_ALPACA_EVAL_OPENAI_PATCHED = False
_ALPACA_EVAL_SKIP_STATS = {
    "content_filter_retries": 0,
    "content_filter_skipped_requests": 0,
    "content_filter_skipped_examples": 0,
}


def _is_content_filter_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "content_filter" in message
        or "content filter" in message
        or "content management policy" in message
    )


def _patch_alpaca_eval_openai_decoder() -> None:
    global _ALPACA_EVAL_OPENAI_PATCHED
    if _ALPACA_EVAL_OPENAI_PATCHED:
        return

    import alpaca_eval.decoders.openai as alpaca_openai_decoder

    original_helper = alpaca_openai_decoder._openai_completion_helper

    def robust_openai_completion_helper(*args, **kwargs):
        global _ALPACA_EVAL_SKIP_STATS
        retry_count = int(os.getenv("ALPACA_EVAL_EXTRA_RETRIES", "3"))
        retry_sleep = float(os.getenv("ALPACA_EVAL_RETRY_SLEEP_SECONDS", "5"))
        skip_on_content_filter = os.getenv("ALPACA_EVAL_SKIP_CONTENT_FILTER", "1") == "1"

        prompt_batch = args[0][0]
        batch_size = len(prompt_batch)
        last_error = None

        for attempt in range(retry_count + 1):
            try:
                return original_helper(*args, **kwargs)
            except openai.BadRequestError as exc:
                last_error = exc
                if _is_content_filter_error(exc):
                    if attempt < retry_count:
                        _ALPACA_EVAL_SKIP_STATS["content_filter_retries"] += 1
                        print(
                            f"[AlpacaEval] Content filter on judge request, retry {attempt + 1}/{retry_count} "
                            f"after {retry_sleep}s."
                        )
                        time.sleep(retry_sleep)
                        continue
                    if skip_on_content_filter:
                        _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_requests"] += 1
                        _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_examples"] += batch_size
                        print(
                            "[AlpacaEval] Content filter persists after retries; "
                            "returning empty completion for this sample."
                        )
                        return [dict(text="", total_tokens=0)] * batch_size
                raise
            except Exception as exc:
                last_error = exc
                if _is_content_filter_error(exc):
                    if attempt < retry_count:
                        _ALPACA_EVAL_SKIP_STATS["content_filter_retries"] += 1
                        print(
                            f"[AlpacaEval] Content filter-like judge failure ({type(exc).__name__}), "
                            f"retry {attempt + 1}/{retry_count} after {retry_sleep}s."
                        )
                        time.sleep(retry_sleep)
                        continue
                    if skip_on_content_filter:
                        _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_requests"] += 1
                        _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_examples"] += batch_size
                        print(
                            "[AlpacaEval] Content filter-like judge failure persists after retries; "
                            "returning empty completion for this sample."
                        )
                        return [dict(text="", total_tokens=0)] * batch_size
                if attempt < retry_count:
                    print(
                        f"[AlpacaEval] Judge request failed ({type(exc).__name__}), "
                        f"retry {attempt + 1}/{retry_count} after {retry_sleep}s."
                    )
                    time.sleep(retry_sleep)
                    continue
                raise

        if skip_on_content_filter and last_error is not None and _is_content_filter_error(last_error):
            return [dict(text="", total_tokens=0)] * batch_size
        raise last_error

    alpaca_openai_decoder._openai_completion_helper = robust_openai_completion_helper
    _ALPACA_EVAL_OPENAI_PATCHED = True


def extract_answer(text: str) -> str:
    """
    extract answer
    """
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    # fallback: 去掉 <think>...</think>
    text_wo_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text_wo_think.strip()


class AlpacaEvalBase(OpenEndedTaskBase, ABC):
    def __init__(self):
        super().__init__()
        self.max_new_tokens, self.temperature, self.top_p = self.prepare_hparams()

    @abc.abstractmethod
    def prepare_hparams(self):
        raise NotImplementedError

    def required_input_fields(self) -> list[str]:
        return ["instruction"]


class AlpacaEval2_0(AlpacaEvalBase):
    def prepare_hparams(self):
        max_new_tokens = 8192
        temperature = 0
        top_p = 1.0
        return max_new_tokens, temperature, top_p

    def _get_eval_data_path(self) -> str:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "alpaca_eval.json")

    def _get_reference_outputs_path(self) -> str:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        gpt4_baseline = os.path.join(current_dir, "alpaca_eval_gpt4_baseline.json")
        if os.path.exists(gpt4_baseline):
            return gpt4_baseline
        return os.path.join(current_dir, "alpaca_eval.json")

    def load(self) -> list[dict]:

        data_path = self._get_eval_data_path()
        print(f"[AlpacaEval] Loading data from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            alpaca_eval_data = json.load(f)
        instructions = [{"instruction": row["instruction"]} for row in alpaca_eval_data]
        print(f"[AlpacaEval] Loaded {len(instructions)} instructions")
        return instructions

    def _evaluate(self, model: GeneratorModelBase) -> tuple[dict, list[dict]]:
        global _ALPACA_EVAL_SKIP_STATS
        _ALPACA_EVAL_SKIP_STATS = {
            "content_filter_retries": 0,
            "content_filter_skipped_requests": 0,
            "content_filter_skipped_examples": 0,
        }
        _patch_alpaca_eval_openai_decoder()
        inputs = [{"instruction": row["instruction"]} for row in self.data]
        completions = model.generate_completions(
            inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        assert len(completions) == len(self.data)

        model_id = "_".join(model.model_name_or_path.split("/"))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(current_dir, "cache"), exist_ok=True)
        output_path = os.path.join(current_dir, "results", model_id)
        os.makedirs(output_path, exist_ok=True)

        # 1. Extract <answer>
        print(f"\n[AlpacaEval] Extracting <answer> tags from {len(completions)} responses...")
        model_results = []
        raw_results = []
        extracted_count = 0
        fallback_count = 0
        
        for example, raw_output in zip(self.data, completions):
            answer = extract_answer(raw_output)
            if "<answer>" in raw_output:
                extracted_count += 1
            else:
                fallback_count += 1
            
            model_results.append(
                {
                    "instruction": example["instruction"],
                    "output": answer,
                    "generator": model_id,
                }
            )
            raw_results.append(
                {
                    "instruction": example["instruction"],
                    "raw_output": raw_output,
                    "generator": model_id,
                }
            )
        
        print(f"[AlpacaEval] Successfully extracted {extracted_count} answers, {fallback_count} fallbacks")

        raw_output_path = os.path.join(output_path, "raw_outputs.json")
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        print(f"[AlpacaEval] Saved raw outputs to: {raw_output_path}")

        # 2. GPT-4 baseline
        reference_outputs_path = self._get_reference_outputs_path()
        print(f"[AlpacaEval] Using reference outputs: {reference_outputs_path}")

        # 3. evaluate
        print(f"\n[AlpacaEval] Starting GPT-4 evaluation...")
        annotators_config = os.environ.get(
            "ALPACA_EVAL_ANNOTATORS_CONFIG",
            os.path.join(current_dir, "weighted_alpaca_eval_gpt4o"),
        )
        print(f"[AlpacaEval] Using annotator config: {annotators_config}")
        df_leaderboard, _ = alpaca_farm_evaluate(
            model_outputs=model_results,
            reference_outputs=reference_outputs_path,
            annotators_config=annotators_config,
            fn_metric="get_length_controlled_winrate",
            output_path=output_path,
            is_return_instead_of_print=True,
            caching_path=os.path.join(
                current_dir, "cache", "alpaca_eval_annotator_cache.json"
            ),
        )

        # 4. get annotations
        if isinstance(annotators_config, str) and "/" not in annotators_config:
            annotation_path = os.path.join(output_path, annotators_config, "annotations.json")
        else:
            annotation_path = os.path.join(output_path, "annotations.json")
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        for annotation, original_data in zip(annotations, self.data):
            annotation["id"] = original_data["id"]

        selected_row = df_leaderboard[df_leaderboard.index == model_id]
        selected_row = selected_row.to_dict(orient="records")[0]
        selected_row["content_filter_retries"] = _ALPACA_EVAL_SKIP_STATS["content_filter_retries"]
        selected_row["content_filter_skipped_requests"] = _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_requests"]
        selected_row["content_filter_skipped_examples"] = _ALPACA_EVAL_SKIP_STATS["content_filter_skipped_examples"]

        skip_stats_path = os.path.join(output_path, "skip_stats.json")
        with open(skip_stats_path, "w", encoding="utf-8") as f:
            json.dump(_ALPACA_EVAL_SKIP_STATS, f, indent=2, ensure_ascii=False)
        
        print(f"\n[AlpacaEval] Evaluation complete!")
        print(f"[AlpacaEval] LC Win Rate: {selected_row.get('length_controlled_winrate', 'N/A'):.2f}%")
        print(f"[AlpacaEval] Win Rate: {selected_row.get('win_rate', 'N/A'):.2f}%")
        print(f"[AlpacaEval] Avg Length: {selected_row.get('avg_length', 'N/A'):.0f} tokens")
        print(
            f"[AlpacaEval] Content-filter skipped examples: "
            f"{selected_row['content_filter_skipped_examples']}"
        )

        return selected_row, annotations
