import os
import csv
import logging
import json
from typing import Dict, List, Optional, Tuple
import click
from dataclasses import dataclass
from datetime import datetime

from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog
from inspect_ai import Task

from runners.base import BaseExperimentRunner, ExperimentConfig
from tasks.task_politeness import adaptive_politeness, politeness_n_shot, re_evaluate_adaptive_politeness
from utils_elicitation.novelty import write_novelty_results

# Disable all logging output
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('inspect_ai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)


@dataclass
class PolitenessExperimentConfig(ExperimentConfig):
    """Configuration for Politeness experiments"""
    use_cot: bool = False
    randomize_sampling: bool = False
    judge_model_name: Optional[str] = None
    n_positive_samples: int = 1
    n_negative_samples: int = 2
    original_eval_model_name: Optional[str] = None
    n_shot: int = 3  # For n-shot initial evaluation


class PolitenessExperimentRunner(BaseExperimentRunner):
    """
    Experiment runner for Politeness experiments.
    Extends BaseExperimentRunner with politeness-specific functionality.
    """
    
    def __init__(self, config: PolitenessExperimentConfig):
        super().__init__(config)
        self.config: PolitenessExperimentConfig = config
        
        # Politeness-specific directories
        for dir_path in ["logs/politeness/initial", "logs/politeness/adaptive", "outputs/politeness"]:
            os.makedirs(dir_path, exist_ok=True)
    
    def extract_accuracy_metrics(self, eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract politeness-specific metrics from evaluation log.
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
        log_dir: str = "logs/politeness/initial"
    ) -> Dict[str, EvalLog]:
        """Run initial politeness evaluation for multiple models."""
        results = {}
        cached_logs = self.read_eval_cache()
        
        for eval_model_name in eval_models:
            # Check cache first
            cache_key = f"politeness_initial_{eval_model_name.replace('/', '_')}"
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
            print(f"Running initial politeness evaluation for {eval_model_name}")
            task = politeness_n_shot(
                n_shot=self.config.n_shot,
                use_cot=self.config.use_cot,
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
        log_dir: str = "logs/politeness/adaptive"
    ) -> Optional[EvalLog]:
        """Run adaptive politeness evaluation."""
        print(f"Running adaptive politeness evaluation: generator={generator_model_name}, eval={eval_model_name}")
        
        task = adaptive_politeness(
            initial_log_path=initial_log_path,
            n_positive_samples=self.config.n_positive_samples,
            n_negative_samples=self.config.n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            use_cot_generator=self.config.use_cot,
            use_cot_evaluator=self.config.use_cot,
            randomize_sampling=self.config.randomize_sampling,
            judge_model_name=self.config.judge_model_name,
            original_eval_model_name=self.config.original_eval_model_name,
        )
        
        log = eval(task, log_dir=log_dir, model=eval_model_name, epochs=Epochs(30, "mean"))[0]
        return log
    
    def run_re_evaluation(
        self,
        adaptive_log_path: str,
        eval_model_name: str,
        log_dir: str = "logs/politeness/re_eval"
    ) -> Optional[EvalLog]:
        """Re-evaluate adaptive politeness questions."""
        print(f"Re-evaluating adaptive politeness questions with {eval_model_name}")
        
        task = re_evaluate_adaptive_politeness(
            adaptive_log_path=adaptive_log_path,
            use_cot=self.config.use_cot,
            filter_by_incorrect=False,
        )
        
        log = eval(task, log_dir=log_dir, model=eval_model_name)[0]
        return log
    
    def write_results(self, eval_log: EvalLog, **additional_data):
        """Write politeness-specific experiment results."""
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
            "n_shot": str(self.config.n_shot),
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
@click.option("--n-positive-samples", default=1, type=int, help="Number of positive samples for adaptation")
@click.option("--n-negative-samples", default=2, type=int, help="Number of negative samples for adaptation")
@click.option("--n-shot", default=3, type=int, help="Number of shots for initial evaluation")
@click.option("--results-path", default="results/politeness_results.csv", help="Path to save results")
@click.option("--cache-path", default="results/politeness_cache.csv", help="Path to save cache")
@click.option("--run-initial", is_flag=True, default=True, help="Run initial evaluations")
@click.option("--run-adaptive", is_flag=True, default=True, help="Run adaptive evaluations")
@click.option("--run-re-eval", is_flag=True, default=False, help="Run re-evaluations")
def main(
    models_for_transfer,
    models_for_generation,
    use_cot,
    randomize_sampling,
    judge_model,
    n_positive_samples,
    n_negative_samples,
    n_shot,
    results_path,
    cache_path,
    run_initial,
    run_adaptive,
    run_re_eval,
):
    """
    Run Politeness experiments using the refactored base class:
    1. Run initial politeness evaluations (if --run-initial)
    2. Run adaptive evaluations (if --run-adaptive)
    3. Run re-evaluations (if --run-re-eval)
    """
    config = PolitenessExperimentConfig(
        use_cot=use_cot,
        randomize_sampling=randomize_sampling,
        judge_model_name=judge_model,
        n_positive_samples=n_positive_samples,
        n_negative_samples=n_negative_samples,
        n_shot=n_shot,
        results_path=results_path,
        cache_path=cache_path,
    )
    
    runner = PolitenessExperimentRunner(config)
    
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
                cache_key = f"politeness_initial_{eval_model.replace('/', '_')}"
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
                            task_type="politeness",
                            task_name="politeness",
                            generator_model_name=generator_model,
                            eval_model_name=eval_model,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        )
                    except Exception as e:
                        print(f"Error writing novelty results: {e}")


if __name__ == "__main__":
    main()
