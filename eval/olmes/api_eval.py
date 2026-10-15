#!/usr/bin/env python3
"""
API-based Evaluation Framework for OLMES Benchmarks
Reuses local datasets and metric calculation logic from OLMES.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from openai import OpenAI

# Add OLMES to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from oe_eval.metrics.metric import IFEvalMetric, ExactMatch, MC1MC2Accuracy
from oe_eval.tasks.oe_eval_tasks.ifeval import IFEval
from oe_eval.tasks.oe_eval_tasks.arc import ARCChallenge
from oe_eval.tasks.oe_eval_tasks.gpqa import GPQA
from oe_eval.tasks.oe_eval_tasks.truthfulqa import TruthfulQA
from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU_OneTurnCoT
from oe_eval.tasks.oe_eval_tasks.gsm8k import GSM8K
from oe_eval.tasks.oe_eval_tasks.bbh import GenericBBH
from oe_eval.utils import extract_answer_from_tags


class APIEvaluator:
    """Base class for API-based evaluation"""
    
    def __init__(self, api_key: str, base_url: str, model: str, output_dir: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def call_api(self, messages: List[Dict[str, str]], max_tokens: int = 1280, 
                 temperature: float = 0.0, stop: List[str] = None) -> str:
        """Call the API and return the generated text"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or []
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def call_api_with_extraction(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1280,
        temperature: float = 0.0,
        stop: List[str] = None,
    ) -> tuple[str, str]:
        raw_text = self.call_api(messages, max_tokens=max_tokens, temperature=temperature, stop=stop)
        extracted_text = extract_answer_from_tags(raw_text)
        return raw_text, extracted_text
    
    def evaluate(self, task_name: str, limit: int = None) -> Dict[str, float]:
        """Main evaluation method to be implemented by subclasses"""
        raise NotImplementedError


class IFEvalAPIEvaluator(APIEvaluator):
    """IFEval task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize task to get default configs
        self.task = IFEval(task_name="ifeval")
        self.task_config = self.task.TASK_CONFIG_DEFAULTS
        
    def evaluate(self, task_name: str = "ifeval", limit: int = None) -> Dict[str, float]:
        """Evaluate IFEval task"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Load dataset from local cache
        dataset = load_dataset(
            self.task_config["dataset_path"],
            split=self.task_config["split"],
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Initialize metric
        metric = IFEvalMetric(**self.task_config["metric_kwargs"])
        
        results = []
        predictions = []
        references = []
        
        for idx, doc in enumerate(tqdm(dataset, desc="Processing IFEval")):
            # Use the task's doc_to_text method to get the prompt
            prompt = self.task.doc_to_text(doc)
            
            # Call API with simple user message (no system prompt for IFEval)
            messages = [{"role": "user", "content": prompt}]
            
            raw_text, generated_text = self.call_api_with_extraction(
                messages=messages,
                max_tokens=self.task_config["generation_kwargs"]["max_gen_toks"],
                temperature=self.task_config["generation_kwargs"]["temperature"],
                stop=self.task_config["generation_kwargs"]["stop_sequences"]
            )
            
            # Prepare prediction in the format expected by IFEvalMetric
            # IFEvalMetric.process_one_doc expects group_lst with model_resps and doc
            prediction = {
                "doc_id": idx,
                "native_id": doc.get("key", idx),
                "idx": 0,
                "doc": doc,
                "model_resps": {"continuation": generated_text},
                "label": None
            }
            
            predictions.append(prediction)
            references.append(doc)
            
            # Save intermediate results
            results.append({
                "doc_id": idx,
                "prompt": prompt,
                "generated": generated_text,
                "raw_generated": raw_text,
                "instruction_id_list": doc["instruction_id_list"]
            })
        
        # Calculate metrics using OLMES's IFEvalMetric
        # IFEvalMetric expects to call compute_for_docs and aggregate_to_task
        metric.compute_for_docs(predictions)
        metric_results = metric.aggregate_to_task(primary_metric=self.task_config["primary_metric"])
        
        # Save detailed results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "num_samples": len(results),
                "metrics": metric_results,
                "predictions": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Metrics: {metric_results}")
        
        return metric_results


class ARCAPIEvaluator(APIEvaluator):
    """ARC-Challenge task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = ARCChallenge(task_name="arc_challenge")
        self.task_config = self.task.TASK_CONFIG_DEFAULTS
        
    def evaluate(self, task_name: str = "arc_challenge", limit: int = None) -> Dict[str, float]:
        """Evaluate ARC-Challenge task"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Load dataset
        dataset = load_dataset(
            self.task_config["dataset_path"],
            self.task_config["dataset_name"],
            split=self.task_config["split"],
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Process docs
        processed_docs = []
        for idx, doc in enumerate(dataset):
            processed_doc = self.task._process_doc(doc, idx)
            processed_docs.append(processed_doc)
        
        results = []
        correct = 0
        
        for idx, doc in enumerate(tqdm(processed_docs, desc="Processing ARC-Challenge")):
            # Get the question text
            query_text = self.task.doc_to_text(doc)
            
            # For multiple choice, we need to get loglikelihoods for each choice
            # Since API doesn't provide loglikelihoods, we'll use generation approach
            # Ask the model to generate the answer letter
            prompt = query_text + "\nAnswer with only the letter (A, B, C, D, etc.):"
            
            messages = [{"role": "user", "content": prompt}]
            raw_text, generated_text = self.call_api_with_extraction(
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            
            # Extract the predicted letter
            predicted_letter = generated_text.strip().upper()
            if len(predicted_letter) > 1:
                # Try to extract just the letter
                import re
                match = re.search(r'\b([A-E])\b', predicted_letter)
                if match:
                    predicted_letter = match.group(1)
                else:
                    predicted_letter = predicted_letter[0] if predicted_letter else ""
            
            # Get gold answer
            gold_letter = doc["gold"]
            
            is_correct = (predicted_letter == gold_letter)
            if is_correct:
                correct += 1
            
            results.append({
                "doc_id": idx,
                "question": query_text,
                "choices": doc["choices"],
                "predicted": predicted_letter,
                "gold": gold_letter,
                "correct": is_correct,
                "raw_generated": raw_text,
            })
        
        accuracy = correct / len(results) if results else 0.0
        metric_results = {"acc": accuracy, "acc_norm": accuracy}
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "num_samples": len(results),
                "metrics": metric_results,
                "predictions": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Metrics: {metric_results}")
        
        return metric_results


class GPQAAPIEvaluator(APIEvaluator):
    """GPQA task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = GPQA(task_name="gpqa")
        self.task_config = self.task.TASK_CONFIG_DEFAULTS
        
    def evaluate(self, task_name: str = "gpqa", limit: int = None) -> Dict[str, float]:
        """Evaluate GPQA task"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Load dataset
        dataset = load_dataset(
            self.task_config["dataset_path"],
            self.task_config["dataset_name"],
            split=self.task_config["split"],
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Initialize metric
        metric = ExactMatch(
            extract_pred_fn=self.task._extract_answer,
            ignore_case=True,
            ignore_punctuation=True,
            **self.task_config["metric_kwargs"]
        )
        
        results = []
        predictions = []
        references = []
        
        for idx, doc in enumerate(tqdm(dataset, desc="Processing GPQA")):
            # Process doc
            processed_doc = self.task._process_doc(doc, idx)
            
            # Get prompt text
            prompt = self.task.doc_to_text(processed_doc)
            
            # Call API
            messages = [{"role": "user", "content": prompt}]
            raw_text, generated_text = self.call_api_with_extraction(
                messages=messages,
                max_tokens=self.task_config["generation_kwargs"]["max_gen_toks"],
                temperature=self.task_config["generation_kwargs"]["temperature"]
            )
            
            # Prepare for metric calculation in OLMES format
            prediction = {
                "doc_id": idx,
                "native_id": idx,
                "idx": 0,
                "doc": processed_doc,
                "model_resps": {"continuation": generated_text},
                "label": processed_doc["answer_text"]
            }
            
            predictions.append(prediction)
            references.append(processed_doc)
            
            results.append({
                "doc_id": idx,
                "question": prompt,
                "generated": generated_text,
                "raw_generated": raw_text,
                "gold": processed_doc["answer_text"]
            })
        
        # Calculate metrics for GPQA
        metric.compute_for_docs(predictions)
        metric_results = metric.aggregate_to_task(primary_metric=self.task_config["primary_metric"])
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "num_samples": len(results),
                "metrics": metric_results,
                "predictions": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Metrics: {metric_results}")
        
        return metric_results


class TruthfulQAAPIEvaluator(APIEvaluator):
    """TruthfulQA task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = TruthfulQA(task_name="truthfulqa")
        self.task_config = self.task.TASK_CONFIG_DEFAULTS
        
    def evaluate(self, task_name: str = "truthfulqa", limit: int = None) -> Dict[str, float]:
        """Evaluate TruthfulQA task"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Load dataset
        dataset = load_dataset(
            self.task_config["dataset_path"],
            self.task_config["dataset_name"],
            split=self.task_config["split"],
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        results = []
        correct_mc1 = 0
        correct_mc2 = 0
        
        for idx, doc in enumerate(tqdm(dataset, desc="Processing TruthfulQA")):
            # Process doc
            processed_doc = self.task._process_doc(doc, idx)
            
            # For API-based evaluation, we'll use a simplified approach
            # Generate answer and check if it matches the correct choices
            question = processed_doc["question"]
            mc1_targets = processed_doc["mc1_targets"]
            mc2_targets = processed_doc["mc2_targets"]
            
            # Create a multiple choice prompt
            all_choices = mc1_targets["choices"] if "choices" in mc1_targets else []
            
            if all_choices:
                prompt = f"Question: {question}\n\nChoose the best answer from the following options:\n"
                for i, choice in enumerate(all_choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nAnswer with only the letter:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            messages = [{"role": "user", "content": prompt}]
            raw_text, generated_text = self.call_api_with_extraction(
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )
            
            # For simplicity, we'll record the results
            # Full MC1/MC2 metrics require loglikelihood which API doesn't provide directly
            results.append({
                "doc_id": idx,
                "question": question,
                "generated": generated_text,
                "raw_generated": raw_text,
                "mc1_targets": mc1_targets,
                "mc2_targets": mc2_targets
            })
        
        # Note: Proper MC1/MC2 evaluation requires loglikelihoods
        # This is a simplified version
        metric_results = {
            "note": "TruthfulQA requires loglikelihood evaluation, API-based evaluation is approximate"
        }
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "num_samples": len(results),
                "metrics": metric_results,
                "predictions": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Metrics: {metric_results}")
        
        return metric_results


class MMLUAPIEvaluator(APIEvaluator):
    """MMLU task evaluator using API (CoT version)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a sample subject for initialization
        from oe_eval.data.mmlu_tasks import MMLU_SUBJECTS
        self.subjects = MMLU_SUBJECTS
        
    def evaluate(self, task_name: str = "mmlu", limit: int = None, 
                 subjects: List[str] = None) -> Dict[str, float]:
        """Evaluate MMLU task (sample subjects or all)"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Evaluate a subset of subjects for demonstration
        if subjects is None:
            subjects = ["abstract_algebra", "anatomy", "astronomy"]  # Sample subjects
        
        all_results = {}
        
        for subject in subjects:
            print(f"\n--- Evaluating {subject} ---")
            
            # Create task instance for this subject
            from oe_eval.tasks.oe_eval_tasks.mmlu import create_mmlu_cot_task
            task_class = create_mmlu_cot_task(subject)
            task = task_class()
            
            # Load dataset
            dataset = load_dataset(
                "cais/mmlu",
                subject,
                split="test",
                cache_dir=os.getenv("HF_DATASETS_CACHE")
            )
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            # Initialize metric
            metric = ExactMatch(
                extract_pred_fn=task._extract_answer,
                ignore_case=True,
                ignore_punctuation=True,
                **task.task_config["metric_kwargs"]
            )
            
            results = []
            predictions = []
            references = []
            
            for idx, doc in enumerate(tqdm(dataset, desc=f"Processing {subject}")):
                # Process doc
                processed_doc = task._process_doc(doc, idx)
                
                # Get prompt
                # For CoT, we need few-shot examples
                fewshot_docs = task.fewshot_examples(5, None, processed_doc)
                
                # Build few-shot prompt
                prompt = ""
                description = task.task_config["context_kwargs"].get("description", "")
                if description:
                    prompt += description + "\n\n"
                
                for fs_doc in fewshot_docs[:5]:
                    prompt += f"{task.doc_to_text(fs_doc)}\n"
                    if task.doc_to_target(fs_doc):
                        prompt += f"{task.doc_to_target(fs_doc)}\n\n"
                
                prompt += task.doc_to_text(processed_doc)
                
                # Call API
                messages = [{"role": "user", "content": prompt}]
                raw_text, generated_text = self.call_api_with_extraction(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0
                )
                
                # Prepare for metric
                prediction = {
                    "doc_id": idx,
                    "native_id": idx,
                    "idx": 0,
                    "doc": processed_doc,
                    "model_resps": {"continuation": generated_text},
                    "label": processed_doc["answer_text"]
                }
                
                predictions.append(prediction)
                references.append(processed_doc)
                
                results.append({
                    "doc_id": idx,
                    "question": processed_doc["query"],
                    "generated": generated_text,
                    "raw_generated": raw_text,
                    "gold": processed_doc["answer_text"]
                })
            
            # Calculate metrics for MMLU subject
            subject_metrics = {}
            if predictions:
                metric.compute_for_docs(predictions)
                subject_metrics = metric.aggregate_to_task(primary_metric=task.task_config["primary_metric"])
            all_results[subject] = subject_metrics
            
            print(f"{subject} metrics: {subject_metrics}")
        
        # Average across subjects
        if all_results:
            avg_metrics = {}
            for key in list(all_results.values())[0].keys():
                values = [result[key] for result in all_results.values() if key in result]
                avg_metrics[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
            
            all_results["average"] = avg_metrics
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "subjects": subjects,
                "metrics": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Average metrics: {all_results.get('average', {})}")
        
        return all_results


class GSM8KAPIEvaluator(APIEvaluator):
    """GSM8K task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = GSM8K(task_name="gsm8k")
        self.task_config = self.task.TASK_CONFIG_DEFAULTS
        
    def evaluate(self, task_name: str = "gsm8k", limit: int = None) -> Dict[str, float]:
        """Evaluate GSM8K task"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Load dataset
        dataset = load_dataset(
            self.task_config["dataset_path"],
            self.task_config["dataset_name"],
            split="test",
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Initialize metric
        metric = ExactMatch(
            extract_pred_fn=self.task._extract_answer,
            ignore_case=True,
            ignore_punctuation=False,
            **self.task_config["metric_kwargs"]
        )
        
        # Get few-shot examples
        train_dataset = load_dataset(
            self.task_config["dataset_path"],
            self.task_config["dataset_name"],
            split="train",
            cache_dir=os.getenv("HF_DATASETS_CACHE")
        )
        fewshot_docs = [self.task._process_doc(doc, i) for i, doc in enumerate(list(train_dataset)[:8])]
        
        results = []
        predictions = []
        references = []
        
        for idx, doc in enumerate(tqdm(dataset, desc="Processing GSM8K")):
            # Process doc
            processed_doc = self.task._process_doc(doc, idx)
            
            # Build few-shot prompt (8-shot CoT)
            prompt = ""
            for fs_doc in fewshot_docs:
                prompt += self.task.doc_to_text(fs_doc)
                prompt += self.task.doc_to_target(fs_doc)
                prompt += "\n\n"
            
            prompt += self.task.doc_to_text(processed_doc)
            
            # Call API
            messages = [{"role": "user", "content": prompt}]
            raw_text, generated_text = self.call_api_with_extraction(
                messages=messages,
                max_tokens=self.task_config["generation_kwargs"]["max_gen_toks"],
                temperature=self.task_config["generation_kwargs"]["temperature"],
                stop=self.task_config["generation_kwargs"]["stop_sequences"]
            )
            
            # Prepare for metric in OLMES format
            prediction = {
                "doc_id": idx,
                "native_id": idx,
                "idx": 0,
                "doc": processed_doc,
                "model_resps": {"continuation": generated_text},
                "label": self.task._extract_answer(processed_doc["short_answer"])
            }
            
            predictions.append(prediction)
            references.append(processed_doc)
            
            results.append({
                "doc_id": idx,
                "question": processed_doc["question"],
                "generated": generated_text,
                "raw_generated": raw_text,
                "gold": processed_doc["short_answer"]
            })
        
        # Calculate metrics for GSM8K
        metric.compute_for_docs(predictions)
        metric_results = metric.aggregate_to_task(primary_metric=self.task_config["primary_metric"])
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "num_samples": len(results),
                "metrics": metric_results,
                "predictions": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Metrics: {metric_results}")
        
        return metric_results


class BBHAPIEvaluator(APIEvaluator):
    """BBH task evaluator using API"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from oe_eval.data.bbh_tasks import BBH_TASKS
        self.bbh_tasks = BBH_TASKS
        
    def evaluate(self, task_name: str = "bbh", limit: int = None,
                 tasks: List[str] = None) -> Dict[str, float]:
        """Evaluate BBH tasks"""
        print(f"\n=== Evaluating {task_name} ===")
        
        # Evaluate a subset of tasks for demonstration
        if tasks is None:
            tasks = ["boolean_expressions", "causal_judgement", "date_understanding"]  # Sample tasks
        
        all_results = {}
        
        for bbh_task_name in tasks:
            print(f"\n--- Evaluating {bbh_task_name} ---")
            
            # Create task instance
            from oe_eval.tasks.oe_eval_tasks.bbh import create_bbh_task
            task_class = create_bbh_task(bbh_task_name)
            task = task_class()
            
            # Load dataset
            dataset = load_dataset(
                "lukaemon/bbh",
                bbh_task_name,
                split="test",
                cache_dir=os.getenv("HF_DATASETS_CACHE")
            )
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            # Initialize metric
            metric = ExactMatch(
                extract_pred_fn=task._extract_answer,
                ignore_case=True,
                ignore_punctuation=(bbh_task_name != "dyck_languages"),
                **task.task_config["metric_kwargs"]
            )
            
            # Get few-shot examples (3-shot)
            fewshot_docs = [task._process_doc(doc, i) for i, doc in enumerate(list(dataset)[:3])]
            
            results = []
            predictions = []
            references = []
            
            for idx, doc in enumerate(tqdm(dataset, desc=f"Processing {bbh_task_name}")):
                # Process doc
                processed_doc = task._process_doc(doc, idx)
                
                # Build few-shot prompt
                prompt = ""
                description = task.task_config["context_kwargs"].get("description", "")
                if description:
                    prompt += description + "\n\n"
                
                for fs_doc in fewshot_docs[:3]:
                    prompt += task.doc_to_text(fs_doc)
                    if task.doc_to_target(fs_doc):
                        prompt += task.doc_to_target(fs_doc)
                    prompt += "\n\n"
                
                prompt += task.doc_to_text(processed_doc)
                
                # Call API
                messages = [{"role": "user", "content": prompt}]
                raw_text, generated_text = self.call_api_with_extraction(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                    stop=["\n\n"]
                )
                
                # Prepare for metric in OLMES format
                prediction = {
                    "doc_id": idx,
                    "native_id": idx,
                    "idx": 0,
                    "doc": processed_doc,
                    "model_resps": {"continuation": generated_text},
                    "label": processed_doc["answer"]
                }
                
                predictions.append(prediction)
                references.append(processed_doc)
                
                results.append({
                    "doc_id": idx,
                    "input": processed_doc["input"],
                    "generated": generated_text,
                    "raw_generated": raw_text,
                    "gold": processed_doc["answer"]
                })
            
            # Calculate metrics for BBH task
            task_metrics = {}
            if predictions:
                metric.compute_for_docs(predictions)
                task_metrics = metric.aggregate_to_task(primary_metric=task.task_config["primary_metric"])
            all_results[bbh_task_name] = task_metrics
            
            print(f"{bbh_task_name} metrics: {task_metrics}")
        
        # Average across tasks
        if all_results:
            avg_metrics = {}
            for key in list(all_results.values())[0].keys():
                values = [result[key] for result in all_results.values() if key in result]
                avg_metrics[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
            
            all_results["average"] = avg_metrics
        
        # Save results
        output_file = self.output_dir / f"{task_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "tasks": tasks,
                "metrics": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Average metrics: {all_results.get('average', {})}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="API-based evaluation for OLMES benchmarks")
    parser.add_argument("--task", type=str, required=True,
                       choices=["ifeval", "arc_challenge", "gpqa", "truthfulqa", 
                               "mmlu", "gsm8k", "bbh", "all"],
                       help="Task to evaluate")
    parser.add_argument("--api-key", type=str, default="FAKE_API_KEY",
                       help="API key")
    ### dace: defender endpoint from env var (DEFENDER_API_BASE_URL); placeholder default ###
    parser.add_argument("--base-url", type=str, 
                       default=os.environ.get("DEFENDER_API_BASE_URL", "http://<DEFENDER_HOST>:<PORT>/v1"),
                       help="API base URL")
    parser.add_argument("--model", type=str, default="orm",
                       help="Model name for API")
    parser.add_argument("--output-dir", type=str, default="./api_eval_results",
                       help="Output directory for results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Create evaluator mapping
    evaluators = {
        "ifeval": IFEvalAPIEvaluator,
        "arc_challenge": ARCAPIEvaluator,
        "gpqa": GPQAAPIEvaluator,
        "truthfulqa": TruthfulQAAPIEvaluator,
        "mmlu": MMLUAPIEvaluator,
        "gsm8k": GSM8KAPIEvaluator,
        "bbh": BBHAPIEvaluator
    }
    
    # Run evaluation
    if args.task == "all":
        all_results = {}
        for task_name, evaluator_class in evaluators.items():
            print(f"\n{'='*60}")
            print(f"Running {task_name}")
            print(f"{'='*60}")
            
            evaluator = evaluator_class(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                output_dir=args.output_dir
            )
            
            try:
                results = evaluator.evaluate(task_name, limit=args.limit)
                all_results[task_name] = results
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary
        summary_file = Path(args.output_dir) / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"All evaluations complete. Summary saved to {summary_file}")
        print(f"{'='*60}")
    else:
        evaluator_class = evaluators[args.task]
        evaluator = evaluator_class(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            output_dir=args.output_dir
        )
        evaluator.evaluate(args.task, limit=args.limit)


if __name__ == "__main__":
    main()
