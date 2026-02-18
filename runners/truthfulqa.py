import os
import csv
import logging
import json
import ast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import click
from datetime import datetime

from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog

from runners.base import BaseExperimentRunner, ExperimentConfig
from tasks.task_adaptive_truthfulqa import (
    truthfulqa_initial,
    adaptive_truthfulqa_refactor,
    re_evaluate_adaptive_truthfulqa,
)
from utils_elicitation.novelty import write_novelty_results

# Turn off verbose logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("inspect_ai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


@dataclass
class TruthfulQAExperimentConfig(ExperimentConfig):
    """Configuration for TruthfulQA experiments"""
    use_cot: bool = False
    randomize_sampling: bool = False
    judge_model_name: Optional[str] = None
    n_positive_samples: int = 1
    n_negative_samples: int = 2
    original_eval_model_name: Optional[str] = None
    self_check_model_name: Optional[str] = None
    use_embeddings: bool = False
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    similarity_threshold: float = 0.6
    score_threshold: int = 4
    max_attempts: int = 5
    use_eval_model_for_checker: bool = False
    include_previous_reasoning: bool = False
    previous_reasoning_limit: int = 0


class TruthfulQAExperimentRunner(BaseExperimentRunner):
    """
    Experiment runner for TruthfulQA experiments.
    Extends BaseExperimentRunner with TruthfulQA-specific functionality.
    """
    
    def __init__(self, config: TruthfulQAExperimentConfig):
        super().__init__(config)
        self.config: TruthfulQAExperimentConfig = config
        
        # TruthfulQA-specific directories
        for dir_path in ["logs/truthfulqa/initial", "logs/truthfulqa/adaptive", "outputs/truthfulqa"]:
            os.makedirs(dir_path, exist_ok=True)
    
    def extract_accuracy_metrics(self, eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract TruthfulQA-specific metrics from evaluation log.
        Returns: (accuracy, count, status)
        """
        if eval_log.status != "success":
            return None, None, eval_log.status
        
        scores = []
        for sample in eval_log.samples:
            if hasattr(sample, 'score') and sample.score is not None:
                if hasattr(sample.score, 'value'):
                    scores.append(sample.score.value)
                else:
                    scores.append(float(sample.score))
        
        if not scores:
            return None, None, "no_scores"
        
        # Calculate accuracy as mean of scores
        accuracy = sum(scores) / len(scores)
        count = len(scores)
        
        return accuracy, count, eval_log.status
    
    def run_initial_evaluation(
        self,
        eval_models: List[str],
        log_dir: str = "logs/truthfulqa/initial"
    ) -> Dict[str, EvalLog]:
        """Run initial TruthfulQA evaluation for multiple models."""
        results = {}
        cached_logs = self.read_eval_cache()
        
        for eval_model_name in eval_models:
            # Check cache first
            cache_key = f"truthfulqa_initial_{eval_model_name.replace('/', '_')}"
            if cache_key in cached_logs and os.path.exists(cached_logs[cache_key]):
                try:
                    from inspect_ai.log import read_eval_log
                    log = read_eval_log(cached_logs[cache_key])[0]
                    if log and log.status == "success":
                        print(f"Using cached initial evaluation for {eval_model_name}")
                        results[eval_model_name] = log
                        continue
                except Exception as e:
                    print(f"Error reading cached log for {eval_model_name}: {e}")
            
            # Run new evaluation
            print(f"Running initial TruthfulQA evaluation for {eval_model_name}")
            task = truthfulqa_initial(
                use_cot=self.config.use_cot,
                debug=False,
            )
            
            log = eval(task, log_dir=log_dir, model=eval_model_name)[0]
            
            # Cache the result
            self.write_eval_cache(cache_key, log.location)
            results[eval_model_name] = log
        
        return results
    
    def run_adaptive_evaluation(
        self,
        initial_log_path: str,
        generator_model_name: str,
        eval_model_name: str,
        log_dir: str = "logs/truthfulqa/adaptive"
    ) -> Optional[EvalLog]:
        """Run adaptive TruthfulQA evaluation."""
        print(f"Running adaptive TruthfulQA evaluation: generator={generator_model_name}, eval={eval_model_name}")
        
        task = adaptive_truthfulqa_refactor(
            initial_log_path=initial_log_path,
            n_positive_samples=self.config.n_positive_samples,
            n_negative_samples=self.config.n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            self_check_model_name=self.config.self_check_model_name,
            use_embeddings=self.config.use_embeddings,
            embeddings_model_name=self.config.embeddings_model_name,
            similarity_threshold=self.config.similarity_threshold,
            score_threshold=self.config.score_threshold,
            max_attempts=self.config.max_attempts,
            randomize_sampling=self.config.randomize_sampling,
            use_cot_generator=self.config.use_cot,
            use_cot_evaluator=self.config.use_cot,
            original_eval_model_name=self.config.original_eval_model_name,
            judge_model_name=self.config.judge_model_name,
            use_eval_model_for_checker=self.config.use_eval_model_for_checker,
            include_previous_reasoning=self.config.include_previous_reasoning,
            previous_reasoning_limit=self.config.previous_reasoning_limit,
        )
        
        log = eval(task, log_dir=log_dir, model=eval_model_name, epochs=Epochs(5, ["mean"]))[0]
        return log
    
    def run_re_evaluation(
        self,
        adaptive_log_path: str,
        eval_model_name: str,
        log_dir: str = "logs/truthfulqa/re_eval"
    ) -> Optional[EvalLog]:
        """Re-evaluate adaptive TruthfulQA questions."""
        print(f"Re-evaluating adaptive TruthfulQA questions with {eval_model_name}")
        
        task = re_evaluate_adaptive_truthfulqa(
            adaptive_log_path=adaptive_log_path,
            use_cot=self.config.use_cot,
            filter_by_incorrect=False,
        )
        
        log = eval(task, log_dir=log_dir, model=eval_model_name)[0]
        return log
    
    def write_results(self, eval_log: EvalLog, **additional_data):
        """Write TruthfulQA-specific experiment results."""
        accuracy, count, status = self.extract_accuracy_metrics(eval_log)
        
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "eval_model": additional_data.get("eval_model", ""),
            "generator_model": additional_data.get("generator_model", ""),
            "accuracy": f"{accuracy:.4f}" if accuracy is not None else "",
            "count": str(count) if count is not None else "",
            "status": status,
            "use_cot": str(self.config.use_cot),
            "n_positive_samples": str(self.config.n_positive_samples),
            "n_negative_samples": str(self.config.n_negative_samples),
            "use_embeddings": str(self.config.use_embeddings),
            "similarity_threshold": str(self.config.similarity_threshold),
            "score_threshold": str(self.config.score_threshold),
            "max_attempts": str(self.config.max_attempts),
            "log_path": eval_log.location if eval_log else "",
        }
        
        # Add any additional data
        results_data.update(additional_data)
        
        # Write to CSV
        fieldnames = list(results_data.keys())
        file_exists = os.path.exists(self.config.get_results_path())
        
        with open(self.config.get_results_path(), mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_data)


@click.command()
@click.option(
    "--models-for-transfer",
    multiple=True,
    default=["openai/gpt-4o-mini"],
    help="Models to use for initial evaluation and transfer experiments",
)
@click.option(
    "--models-for-generation",
    multiple=True,
    default=["openai/gpt-4o", "openai/gpt-4o-mini"],
    help="Models to use for question generation",
)
@click.option("--use-cot", is_flag=True, default=False, help="Use chain-of-thought reasoning")
@click.option("--randomize-sampling", is_flag=True, default=False, help="Randomize sample selection")
@click.option("--judge-model", default=None, help="Judge model for evaluation")
@click.option("--self-check-model", default=None, help="Self-check model for validation")
@click.option("--n-positive-samples", default=1, type=int, help="Number of positive samples for adaptation")
@click.option("--n-negative-samples", default=2, type=int, help="Number of negative samples for adaptation")
@click.option("--use-embeddings", is_flag=True, default=False, help="Use embeddings for similarity checking")
@click.option("--embeddings-model", default="sentence-transformers/all-mpnet-base-v2", help="Embeddings model to use")
@click.option("--similarity-threshold", default=0.6, type=float, help="Similarity threshold for embeddings")
@click.option("--score-threshold", default=4, type=int, help="Score threshold for acceptance")
@click.option("--max-attempts", default=5, type=int, help="Maximum generation attempts")
@click.option("--use-eval-model-for-checker", is_flag=True, default=False, help="Use eval model for checking")
@click.option("--include-previous-reasoning", is_flag=True, default=False, help="Include previous reasoning in context")
@click.option("--previous-reasoning-limit", default=0, type=int, help="Number of previous reasoning traces to include")
@click.option("--results-path", default="results/truthfulqa_results.csv", help="Path to save results")
@click.option("--cache-path", default="results/truthfulqa_cache.csv", help="Path to save cache")
@click.option("--run-initial", is_flag=True, default=True, help="Run initial evaluations")
@click.option("--run-adaptive", is_flag=True, default=True, help="Run adaptive evaluations")
@click.option("--run-re-eval", is_flag=True, default=False, help="Run re-evaluations")
def main(
    models_for_transfer,
    models_for_generation,
    use_cot,
    randomize_sampling,
    judge_model,
    self_check_model,
    n_positive_samples,
    n_negative_samples,
    use_embeddings,
    embeddings_model,
    similarity_threshold,
    score_threshold,
    max_attempts,
    use_eval_model_for_checker,
    include_previous_reasoning,
    previous_reasoning_limit,
    results_path,
    cache_path,
    run_initial,
    run_adaptive,
    run_re_eval,
):
    """
    Run TruthfulQA experiments using the refactored base class:
    1. Run initial TruthfulQA evaluations (if --run-initial)
    2. Run adaptive evaluations (if --run-adaptive)
    3. Run re-evaluations (if --run-re-eval)
    """
    config = TruthfulQAExperimentConfig(
        use_cot=use_cot,
        randomize_sampling=randomize_sampling,
        judge_model_name=judge_model,
        self_check_model_name=self_check_model,
        n_positive_samples=n_positive_samples,
        n_negative_samples=n_negative_samples,
        use_embeddings=use_embeddings,
        embeddings_model_name=embeddings_model,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
        max_attempts=max_attempts,
        use_eval_model_for_checker=use_eval_model_for_checker,
        include_previous_reasoning=include_previous_reasoning,
        previous_reasoning_limit=previous_reasoning_limit,
        results_path=results_path,
        cache_path=cache_path,
    )
    
    runner = TruthfulQAExperimentRunner(config)
    
    # Run initial evaluations
    initial_logs = {}
    if run_initial:
        print(f"Running initial evaluations for models: {models_for_transfer}")
        initial_logs = runner.run_initial_evaluation(list(models_for_transfer))
        
        # Write initial results
        for model_name, log in initial_logs.items():
            runner.write_results(log, eval_model=model_name, experiment_type="initial")
    
    # Run adaptive evaluations
    if run_adaptive:
        print(f"Running adaptive evaluations")
        cached_logs = runner.read_eval_cache()
        
        for eval_model in models_for_transfer:
            for generator_model in models_for_generation:
                # Get initial log
                cache_key = f"truthfulqa_initial_{eval_model.replace('/', '_')}"
                if cache_key in cached_logs:
                    initial_log_path = cached_logs[cache_key]
                elif eval_model in initial_logs:
                    initial_log_path = initial_logs[eval_model].location
                else:
                    print(f"No initial log found for {eval_model}, skipping adaptive")
                    continue
                
                # Set original eval model for transfer experiments
                config.original_eval_model_name = eval_model
                
                adaptive_log = runner.run_adaptive_evaluation(
                    initial_log_path=initial_log_path,
                    generator_model_name=generator_model,
                    eval_model_name=eval_model,
                )
                
                if adaptive_log:
                    runner.write_results(
                        adaptive_log,
                        eval_model=eval_model,
                        generator_model=generator_model,
                        experiment_type="adaptive",
                        initial_log_path=initial_log_path,
                    )
                    
                    # Write novelty results if available
                    try:
                        write_novelty_results(
                            log=adaptive_log,
                            task_type="truthfulqa",
                            task_name="truthfulqa",
                            generator_model_name=generator_model,
                            eval_model_name=eval_model,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        )
                    except Exception as e:
                        print(f"Error writing novelty results: {e}")


if __name__ == "__main__":
    main()
