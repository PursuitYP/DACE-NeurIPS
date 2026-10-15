from OpenRT.models import BaseModel
from OpenRT.datasets import BaseDataset
from OpenRT.attacks import BaseAttack, AttackResult
from OpenRT.evaluators import BaseEvaluator, EvaluationMetrics
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
from dataclasses import asdict

class Orchestrator:
    """
    Experiment workflow orchestrator (The "Conductor of the Orchestra").

    This class has a very focused and important responsibility: it receives all already-instantiated modules
    (model, dataset, attack, evaluator) and executes a complete, reproducible experiment in a fixed order.
    It does not contain any specific attack or evaluation logic itself, thereby achieving high decoupling
    between the core workflow and specific implementations.
    """
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        attack: BaseAttack,
        evaluator: BaseEvaluator,
        max_workers: int = 16,
        ### dace: incremental checkpoint + running success rate (quota-safe records) ###
        checkpoint_path: str = None,
        running_every: int = 5,
    ):
        """
        Initialize the orchestrator.

        Args:
            model: The already instantiated model object.
            dataset: The already instantiated dataset object.
            attack: The already instantiated attack object.
            evaluator: The already instantiated evaluator object.
            max_workers: Maximum number of threads when executing attacks in parallel.
        """
        self.model = model
        self.dataset = dataset
        self.attack = attack
        self.evaluator = evaluator
        self.max_workers = max_workers
        ### dace: checkpoint config ###
        self.checkpoint_path = checkpoint_path
        self.running_every = max(1, int(running_every))

    def run(self) -> Tuple[EvaluationMetrics, List[AttackResult]]:
        """
        Execute the complete experiment workflow.

        Returns:
            A tuple containing the final evaluation metrics and a detailed list of results from each attack.
        """
        # 1. Print experiment configuration information for easy recording and debugging.
        print("--- Starting Experiment ---")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Dataset: {self.dataset.__class__.__name__} (Size: {len(self.dataset)})")
        print(f"Attack: {self.attack.__class__.__name__}")
        print(f"Evaluator: {self.evaluator.__class__.__name__}")
        print("---------------------------")

        # 2. Execute attacks in parallel using multiple threads
        all_results: List[AttackResult] = [None] * len(self.dataset)

        ### dace: open checkpoint sink (append JSONL) + running-progress path ###
        ckpt_f = None
        progress_path = None
        if self.checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
            ckpt_f = open(self.checkpoint_path, "a", encoding="utf-8")
            progress_path = self.checkpoint_path + ".progress.json"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all attack tasks
            future_to_index = {
                executor.submit(self.attack.attack, target): i
                for i, target in enumerate(self.dataset)
            }

            # Collect results
            in_success=0
            completed_count = 0
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
                index = future_to_index[future]
                target = self.dataset[index]

                try:
                    result = future.result()
                    all_results[index] = result
                    completed_count += 1
                    in_success+=result.success
                    # print(f"\n[ Completed target {completed_count}/{len(self.dataset)}: '{str(target)}...' ]")
                    # print(f"  > Attack Outcome: {'Success' if result.success else 'Failure'}")
                except Exception as e:
                    import traceback
                    print(f"\n[ Error attacking target {index+1}/{len(self.dataset)}: '{str(target)}...' ]")
                    print(f"  > Error: {e}")
                    print(f"  > Traceback: {traceback.format_exc()}")
                    # Create a failed result
                    all_results[index] = AttackResult(success=False, target=target)

                ### dace: persist each completed behavior + emit running success every N ###
                if ckpt_f is not None:
                    try:
                        rec = asdict(all_results[index])
                    except Exception:
                        r = all_results[index]
                        rec = {"target": str(getattr(r, "target", target)),
                               "success": bool(getattr(r, "success", False))}
                    ckpt_f.write(json.dumps({"index": index, "result": rec}, default=str) + "\n")
                    ckpt_f.flush()
                if completed_count % self.running_every == 0 or completed_count == len(self.dataset):
                    running_asr = in_success / completed_count if completed_count else 0.0
                    print(f"[running] {completed_count}/{len(self.dataset)} done | "
                          f"in-loop success={in_success} ({running_asr:.2%})")
                    if progress_path:
                        with open(progress_path, "w", encoding="utf-8") as pf:
                            json.dump({"completed": completed_count,
                                       "total": len(self.dataset),
                                       "in_loop_success": int(in_success),
                                       "running_in_loop_asr": running_asr}, pf)

        if ckpt_f is not None:
            ckpt_f.close()


        if completed_count == 0:
            print("No attacks were completed successfully.")
            return EvaluationMetrics(attack_success_rate=0), all_results

        # 3. After all attacks complete, call the evaluator for final evaluation.
        print("\n--- Evaluating Full Experiment Results ---")
        metrics = self.evaluator.evaluate(all_results)

        if isinstance(metrics, EvaluationMetrics):
            metrics.insucess_rate = in_success/len(self.dataset)
            print(f"Final Attack Success Rate: {max(metrics.attack_success_rate,metrics.insucess_rate):.2%}")
        elif isinstance(metrics, dict) and "overall" in metrics and isinstance(metrics["overall"], EvaluationMetrics):
            metrics["overall"].insucess_rate = in_success/len(self.dataset)
            print(f"Final Attack Success Rate: {max(metrics['overall'].attack_success_rate,metrics['overall'].insucess_rate):.2%}")
        print("--- Experiment Finished ---\n")

        # 4. Return results for the caller to perform post-processing (e.g., save to file).
        return metrics, all_results