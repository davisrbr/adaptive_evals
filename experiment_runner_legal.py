'''
Example usage:
python experiment_runner_legal.py --models-for-transfer openai/gpt-4o --models-for-transfer openai/gpt-4o-mini
'''
import os
import csv
from typing import Dict, List, Optional, Tuple
import click
from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog, read_eval_log
from tasks.task_adaptive_legal import adaptive_legal, legalbench_initial_aggregated, re_evaluate_adaptive_legal
import logging
import json
import ast
from datetime import datetime
from inspect_ai import Task
from utils_elicitation.novelty import write_novelty_results
from solvers.adaptive_utils import multiple_choice_save_cot
from inspect_ai.scorer import choice
from inspect_ai.dataset import MemoryDataset

# Disable all logging output
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('inspect_ai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

def read_eval_cache(cache_csv: str) -> Dict[str, str]:
    """
    Returns a dict of {model_name: log_path} previously saved.
    """
    if not os.path.exists(cache_csv):
        return {}
    output = {}
    with open(cache_csv, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]
            log_path = row["log_path"]
            output[model_name] = log_path
    return output


def write_eval_cache(cache_csv: str, model_name: str, log_path: str) -> None:
    """
    Appends the given (model_name, log_path) to the CSV cache.
    Creates the file with headers if it does not exist.
    """
    file_exists = os.path.exists(cache_csv)
    with open(cache_csv, mode="a", newline="") as f:
        fieldnames = ["model_name", "log_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model_name": model_name, "log_path": log_path})


class ExperimentConfig:
    """Configuration class to handle experiment settings and paths"""
    def __init__(
        self,
        use_example: bool = True,
        use_cot_target: bool = True,
        use_cot_in_context_attacker: bool = False,
        num_epochs: int = 100,
    ):
        self.use_example = use_example
        self.use_cot_target = use_cot_target
        self.use_cot_in_context_attacker = use_cot_in_context_attacker
        self.num_epochs = num_epochs

    @property
    def experiment_id(self) -> str:
        """Generate a readable experiment identifier"""
        components = []
        if self.use_example:
            components.append("with_examples")
        if self.use_cot_target:
            components.append("cot_target")
        if self.use_cot_in_context_attacker:
            components.append("cot_attacker")
        if self.num_epochs:
            components.append(f"num_epochs_{self.num_epochs}")
        return "_".join(components) or "base"
    
    def get_cache_path(self) -> str:
        """Get the path for the cache CSV"""
        return os.path.join("cache", f"initial_eval_{self.experiment_id}.csv")
    
    def get_initial_log_dir(self) -> str:
        """Get the directory for initial evaluation logs"""
        return os.path.join("logs", "legalbench", "initial", self.experiment_id)
    
    def get_adaptive_log_dir(self) -> str:
        """Get the directory for adaptive evaluation logs"""
        return os.path.join("logs", "legalbench", "adaptive", self.experiment_id)

def extract_accuracy_metrics(eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Parse the log results and return (accuracy, accuracy_judged, scorer_name_used).
    
    We stop at the first ScoreItem that provides either metric.
    """
    if not eval_log or not eval_log.results:
        return None, None, None

    for score_item in (eval_log.results.scores or []):
        # Look for metrics in the score item
        metrics = score_item.metrics or {}
        
        # Get accuracy values if they exist
        accuracy = metrics.get("accuracy")
        accuracy_judged = metrics.get("accuracy_judged")
        
        # If we found either metric, return the values
        if accuracy is not None or accuracy_judged is not None:
            return (
                accuracy.value if accuracy else None,
                accuracy_judged.value if accuracy_judged else None,
                score_item.scorer,
            )

    return None, None, None


def parse_re_eval_counts(eval_log: EvalLog) -> Tuple[int, int, int]:
    """
    Returns a 3-tuple:
      (a) number of samples that passed judge filter,
      (b) number of samples that were labeled incorrect in the adaptive pass,
      (c) number of samples that were incorrect for the re-evaluation model.
    """
    if not eval_log or not eval_log.samples:
        return 0, 0, 0

    total_passed_judge = len(eval_log.samples)
    initially_incorrect = 0
    re_eval_incorrect = 0

    for sample in eval_log.samples:
        # Get the original metadata from the adaptive run
        orig_md_str = sample.metadata.get("original_metadata", "")
        if orig_md_str:
            try:
                parsed_md = json.loads(orig_md_str)
            except json.JSONDecodeError:
                try:
                    parsed_md = ast.literal_eval(orig_md_str)
                except Exception:
                    parsed_md = None

            # Check if originally incorrect in adaptive pass
            if isinstance(parsed_md, dict) and parsed_md.get("score") == "I":
                initially_incorrect += 1

        # Check if re-eval was incorrect
        if hasattr(sample, 'score') and sample.score and sample.score.value == "I":
            re_eval_incorrect += 1
        elif sample.store.get("score") == "I":  # Backup check in store
            re_eval_incorrect += 1

    return total_passed_judge, initially_incorrect, re_eval_incorrect


def write_experiment_log(
    experiment_csv: str,
    eval_model_name: str,
    generator_model_name: str,
    re_eval_model_name: str,
    initial_log_path: str,
    adaptive_log_path: str,
    re_eval_log_path: Optional[str],
    use_example: bool,
    use_cot_target: bool,
    use_cot_in_context_attacker: bool,
    num_epochs: int,
    adaptive_accuracy: Optional[float],
    adaptive_accuracy_judged: Optional[float],
    adaptive_scorer_name: Optional[str],
    re_eval_accuracy: Optional[float],
    re_eval_accuracy_judged: Optional[float],
    re_eval_scorer_name: Optional[str],
    judge_model_name: Optional[str],
    positive_samples: int,
    negative_samples: int,
    passed_judge_count: Optional[int] = None,
    adaptive_incorrect_count: Optional[int] = None,
    re_eval_incorrect_count: Optional[int] = None,
    novelty_results_file: Optional[str] = None,
) -> None:
    """
    Appends a single row to the experiment CSV containing:
      - all hyperparameters,
      - the logs,
      - metrics extracted from the logs,
      - counts of judge filter and re-eval counts,
      - and the novelty filtering results file path.
    """
    os.makedirs(os.path.dirname(experiment_csv), exist_ok=True)
    file_exists = os.path.exists(experiment_csv)
    print(f"Writing experiment log to {experiment_csv}")
    with open(experiment_csv, mode="a", newline="") as f:
        fieldnames = [
            "eval_model_name",
            "generator_model_name",
            "re_eval_model_name",
            "initial_log_path",
            "adaptive_log_path",
            "re_eval_log_path",
            "use_example",
            "use_cot_target",
            "use_cot_in_context_attacker",
            "num_epochs",
            "adaptive_accuracy",
            "adaptive_accuracy_judged",
            "adaptive_scorer_name",
            "re_eval_accuracy",
            "re_eval_accuracy_judged",
            "re_eval_scorer_name",
            "judge_model_name",
            "positive_samples",
            "negative_samples",
            "passed_judge_count",
            "adaptive_incorrect_count",
            "re_eval_incorrect_count",
            "novelty_results_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "eval_model_name": eval_model_name,
                "generator_model_name": generator_model_name,
                "re_eval_model_name": re_eval_model_name,
                "initial_log_path": initial_log_path,
                "adaptive_log_path": adaptive_log_path,
                "re_eval_log_path": re_eval_log_path,
                "use_example": use_example,
                "use_cot_target": use_cot_target,
                "use_cot_in_context_attacker": use_cot_in_context_attacker,
                "num_epochs": num_epochs,
                "adaptive_accuracy": adaptive_accuracy,
                "adaptive_accuracy_judged": adaptive_accuracy_judged,
                "adaptive_scorer_name": adaptive_scorer_name,
                "re_eval_accuracy": re_eval_accuracy,
                "re_eval_accuracy_judged": re_eval_accuracy_judged,
                "re_eval_scorer_name": re_eval_scorer_name,
                "judge_model_name": judge_model_name,
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
                "passed_judge_count": passed_judge_count,
                "adaptive_incorrect_count": adaptive_incorrect_count,
                "re_eval_incorrect_count": re_eval_incorrect_count,
                "novelty_results_file": novelty_results_file,
            }
        )

class TransferLegalExperimentRunner:
    def __init__(self, cache_csv: Optional[str] = None, experiment_csv: Optional[str] = None, **kwargs):
        self.config = ExperimentConfig(**kwargs)
        self.cache_csv = cache_csv or self.config.get_cache_path()
        self.experiment_csv = experiment_csv or os.path.join("results", f"legal_experiment_results_{self.config.experiment_id}.csv")
        print(f"Using experiment CSV: {self.experiment_csv}")
        print(f"Using cache CSV: {self.cache_csv}")
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.experiment_csv), exist_ok=True)
        os.makedirs(self.config.get_initial_log_dir(), exist_ok=True)
        os.makedirs(self.config.get_adaptive_log_dir(), exist_ok=True)
        
        # Models for the initial pass
        self.initial_eval_models = [
            # "openai/gpt-4o",
            "openai/gpt-4o-mini",
            # "openai/o1-mini",
            # "together/deepseek-ai/DeepSeek-V3",
            # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # "anthropic/claude-3-5-sonnet-latest",
        ]
        self.generator_models = [
            "openai/gpt-4o-mini",
            # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # "openai/gpt-4o",
            # "anthropic/claude-3-5-sonnet-latest",
            # "openai/o1-mini",
        ]
        # Models on which we'll perform the adaptive evaluation
        self.adaptive_evaluated_models = [
            # "openai/gpt-4o",
            "openai/gpt-4o-mini",
            # "together/deepseek-ai/DeepSeek-V3",
            # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # "openai/o1-mini",
            # "anthropic/claude-3-5-haiku-latest",
            # "anthropic/claude-3-5-sonnet-latest",
        ]

        self.task_names = [
            # "maud_ability_to_consummate_concept_is_subject_to_mae_carveouts",
            # "maud_financial_point_of_view_is_the_sole_consideration",
            "maud_accuracy_of_fundamental_target_rws_bringdown_standard",
            # "maud_accuracy_of_target_general_rw_bringdown_timing_answer",
        ]

    def run_initial_experiment_for_task(
        self,
        task_name: str,
        model_name: str,
        log_dir: Optional[str] = None,
    ) -> Optional[EvalLog]:
        """
        Runs the initial experiment for a single task and model.
        Returns the EvalLog if successful, None otherwise.
        """
        log_dir = log_dir or os.path.join(self.config.get_initial_log_dir(), task_name)
        os.makedirs(log_dir, exist_ok=True)
        
        cache_key = f"{model_name}_{task_name}"
        cache_data = read_eval_cache(self.cache_csv)
        
        if cache_key in cache_data:
            # Already evaluated, just re-load the log
            cached_log_path = cache_data[cache_key]
            task_log = read_eval_log(cached_log_path)
            if task_log.status == "success":
                return task_log
            else:
                print(f"Warning: cached log for {cache_key} had status '{task_log.status}' so re-running.")
        
        # Not in cache or needs re-running, run the initial experiment
        task = legalbench_initial_aggregated(
            task_names=[task_name],
            debug=False,
            use_example=self.config.use_example,
            use_cot=self.config.use_cot_target,
            use_claude="claude" in model_name.lower(),
        )
        
        result_logs = eval(
            task,
            epochs=Epochs(1, "max"),
            max_connections=10000,
            log_dir=log_dir,
            model=model_name,
            log_level="critical",
        )

        if not result_logs:
            print(f"No logs returned for {model_name} on task {task_name}. Skipping.")
            return None

        task_log = result_logs[0]
        if task_log.status != "success":
            print(f"Initial experiment task {task_name} with {model_name} failed.")
            return None

        # Cache the new log path
        write_eval_cache(self.cache_csv, cache_key, task_log.location)
        print(f"Initial experiment task {task_name} with {model_name} completed successfully.")
        
        return task_log

    def run_initial_experiments(
        self,
        task_name: str,
        models: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, EvalLog]:
        """
        1. Reads CSV cache to see if any models have already been run for this task.
        2. Runs initial experiments only on models not in cache.
        3. Updates the cache with newly completed log paths.
        Returns dict of {model_name: EvalLog}.
        """
        models = models or self.initial_eval_models
        logs_by_model: Dict[str, EvalLog] = {}
        
        task_log_dir = log_dir or os.path.join(self.config.get_initial_log_dir(), task_name)
        os.makedirs(task_log_dir, exist_ok=True)
        
        for model_name in models:
            task_log = self.run_initial_experiment_for_task(task_name, model_name, task_log_dir)
            if task_log:
                logs_by_model[model_name] = task_log

        return logs_by_model

    def run_rewording_experiments(
        self,
        log_dir: str = "legalbench_logs/rewording",
    ):
        pass

    def run_adaptive_experiment(
        self,
        initial_log_path: str,
        original_eval_model_name: str,
        task_name: str,
        positive_samples_list: List[int] = [1],
        negative_samples_list: List[int] = [8],
        log_dir: Optional[str] = None,
        cot_in_context: bool = False,
        use_cot: bool = True,
        similarity_threshold: float = 0.6,
        use_embeddings: bool = False,
        num_epochs: int = 100,
    ):
        """
        Uses a single (initial_log_path) from the original model, 
        then runs adaptive experiments with each generator model 
        on each of the adaptive_eval_models.

        NOTE: USE_COT IS FOR THE EVALUATOR MODEL, NOT THE GENERATOR MODEL.
        """
        task_specific_log_dir = log_dir or os.path.join(self.config.get_adaptive_log_dir(), task_name)
        os.makedirs(task_specific_log_dir, exist_ok=True)
        
        for pos_samples in positive_samples_list:
            for neg_samples in negative_samples_list:
                for generator_model in self.generator_models:
                    for eval_model in self.adaptive_evaluated_models:
                        task = adaptive_legal(
                            initial_log_path=initial_log_path,
                            task_name=task_name,
                            n_positive_samples=pos_samples,
                            n_negative_samples=neg_samples,
                            generator_model_name=generator_model,
                            eval_model_name=eval_model,
                            use_cot_generator=True,
                            use_cot_evaluator=use_cot,
                            cot_in_context=cot_in_context,
                            randomize_sampling=False,
                            original_eval_model_name=original_eval_model_name,
                            judge_model_name="anthropic/claude-3-5-sonnet-latest",
                        )
                        
                        # Create a specific log directory for this combination
                        combination_log_dir = os.path.join(
                            task_specific_log_dir,
                            f"{eval_model.replace('/', '_')}_{generator_model.replace('/', '_')}_pos{pos_samples}_neg{neg_samples}"
                        )
                        os.makedirs(combination_log_dir, exist_ok=True)
                        
                        adaptive_logs = eval(
                            task,
                            epochs=Epochs(num_epochs, "mean"),
                            max_connections=1000,
                            log_dir=combination_log_dir,
                            model=eval_model,
                            temperature=0,
                            log_level="critical",
                        )
                        
                        if not adaptive_logs or adaptive_logs[0].status != "success":
                            print(f"Adaptive experiment failed for {eval_model} with generator {generator_model} on task {task_name}")
                            continue
                            
                        adaptive_log = adaptive_logs[0]
                        adaptive_accuracy, adaptive_accuracy_judged, adaptive_scorer = extract_accuracy_metrics(adaptive_log)
                        
                        # Compute novelty results
                        results_dir = os.path.join(os.path.dirname(self.experiment_csv), task_name)
                        os.makedirs(results_dir, exist_ok=True)
                        novelty_file = os.path.join(
                            results_dir,
                            f"novelty_legal_{task_name}_{eval_model.replace('/', '_')}_{generator_model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
                        
                        try:
                            write_novelty_results(
                                adaptive_log_path=adaptive_log.location,
                                embeddings_model_name="sentence-transformers/all-mpnet-base-v2",
                                similarity_threshold=similarity_threshold,
                                output_file=novelty_file,
                                use_embeddings=use_embeddings,
                            )
                        except Exception as exc:
                            print(f"Novelty filtering failed for (task={task_name}, eval={eval_model}, gen={generator_model}). Error: {exc}")
                            novelty_file = None
                            
                        # Re-evaluation step for each model (transfer)
                        for re_eval_model in self.adaptive_evaluated_models:
                            print(f"Re-evaluating model {re_eval_model} on questions from {eval_model} generated by {generator_model} for task {task_name}")
                            
                            # Create re-evaluation task
                            re_eval_task = re_evaluate_adaptive_legal(
                                adaptive_log_path=adaptive_log.location,
                                use_cot=use_cot,
                                filter_by_incorrect=True,
                            )
                            
                            # Create a specific log directory for re-evaluation
                            re_eval_log_dir = os.path.join(combination_log_dir, "re_eval", re_eval_model.replace("/", "_"))
                            os.makedirs(re_eval_log_dir, exist_ok=True)
                            
                            re_eval_log_path: Optional[str] = None
                            re_eval_acc: Optional[float] = None
                            re_eval_acc_judged: Optional[float] = None
                            re_eval_scorer: Optional[str] = None
                            
                            re_eval_logs = eval(
                                re_eval_task,
                                epochs=Epochs(1, "max"),
                                log_dir=re_eval_log_dir,
                                model=re_eval_model,
                                log_level="critical",
                            )
                            
                            if re_eval_logs and re_eval_logs[0].status == "success":
                                re_eval_log_path = re_eval_logs[0].location
                                print(
                                    f"Re-evaluation success for (task={task_name}, eval={eval_model}, "
                                    f"gen={generator_model}, re-eval={re_eval_model})."
                                )
                                re_eval_acc, re_eval_acc_judged, re_eval_scorer = extract_accuracy_metrics(re_eval_logs[0])
                                passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = parse_re_eval_counts(
                                    re_eval_logs[0]
                                )
                            else:
                                print(
                                    f"Re-evaluation returned no logs or non-success status for "
                                    f"(task={task_name}, eval={eval_model}, gen={generator_model}, re-eval={re_eval_model})."
                                )
                                passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = (0, 0, 0)

                            
                            # Create a task-specific experiment CSV
                            task_experiment_csv = os.path.join(
                                os.path.dirname(self.experiment_csv), 
                                f"{task_name}_experiment_results_{self.config.experiment_id}.csv"
                            )
                            
                            write_experiment_log(
                                experiment_csv=task_experiment_csv,
                                eval_model_name=eval_model,
                                generator_model_name=generator_model,
                                re_eval_model_name=re_eval_model,
                                initial_log_path=initial_log_path,
                                adaptive_log_path=adaptive_log.location,
                                re_eval_log_path=re_eval_log_path,
                                use_example=self.config.use_example,
                                use_cot_target=self.config.use_cot_target,
                                use_cot_in_context_attacker=self.config.use_cot_in_context_attacker,
                                adaptive_accuracy=adaptive_accuracy,
                                adaptive_accuracy_judged=adaptive_accuracy_judged,
                                adaptive_scorer_name=adaptive_scorer,
                                re_eval_accuracy=re_eval_acc,
                                re_eval_accuracy_judged=re_eval_acc_judged,
                                re_eval_scorer_name=re_eval_scorer,
                                judge_model_name="anthropic/claude-3-5-sonnet-latest",
                                positive_samples=pos_samples,
                                negative_samples=neg_samples,
                                passed_judge_count=passed_judge,
                                adaptive_incorrect_count=adaptive_incorrect_cnt,
                                re_eval_incorrect_count=re_eval_incorrect_cnt,
                                novelty_results_file=novelty_file,
                                num_epochs=num_epochs,
                            )
                            
                            # Also write to the main experiment CSV for aggregated analysis
                            write_experiment_log(
                                experiment_csv=self.experiment_csv,
                                eval_model_name=eval_model,
                                generator_model_name=generator_model,
                                re_eval_model_name=re_eval_model,
                                initial_log_path=initial_log_path,
                                adaptive_log_path=adaptive_log.location,
                                re_eval_log_path=re_eval_log_path,
                                use_example=self.config.use_example,
                                use_cot_target=self.config.use_cot_target,
                                use_cot_in_context_attacker=self.config.use_cot_in_context_attacker,
                                adaptive_accuracy=adaptive_accuracy,
                                adaptive_accuracy_judged=adaptive_accuracy_judged,
                                adaptive_scorer_name=adaptive_scorer,
                                re_eval_accuracy=re_eval_acc,
                                re_eval_accuracy_judged=re_eval_acc_judged,
                                re_eval_scorer_name=re_eval_scorer,
                                judge_model_name="anthropic/claude-3-5-sonnet-latest",
                                positive_samples=pos_samples,
                                negative_samples=neg_samples,
                                passed_judge_count=passed_judge,
                                adaptive_incorrect_count=adaptive_incorrect_cnt,
                                re_eval_incorrect_count=re_eval_incorrect_cnt,
                                novelty_results_file=novelty_file,
                                num_epochs=num_epochs,
                            )

    def run_task_pipeline(
        self,
        task_name: str,
        models_for_transfer: List[str],
        similarity_threshold: float = 0.6,
        use_embeddings: bool = False,
    ):
        """
        Runs the complete pipeline for a single task:
        1. Run initial experiments for all models
        2. For each transfer model, run adaptive experiments
        
        All results are saved in task-specific directories.
        """
        print(f"Starting pipeline for task: {task_name}")
        
        # Create task-specific directories
        task_initial_log_dir = os.path.join(self.config.get_initial_log_dir(), task_name)
        task_adaptive_log_dir = os.path.join(self.config.get_adaptive_log_dir(), task_name)
        task_results_dir = os.path.join(os.path.dirname(self.experiment_csv), task_name)
        
        os.makedirs(task_initial_log_dir, exist_ok=True)
        os.makedirs(task_adaptive_log_dir, exist_ok=True)
        os.makedirs(task_results_dir, exist_ok=True)
        
        # 1. Run initial experiments for this task
        logs_by_model = self.run_initial_experiments(task_name=task_name, log_dir=task_initial_log_dir)
        
        # 2. For each transfer model, run adaptive experiments
        for model_for_transfer in models_for_transfer:
            if model_for_transfer not in logs_by_model:
                print(f"Warning: No log found for {model_for_transfer} on task {task_name}. Available: {list(logs_by_model.keys())}")
                continue
            
            chosen_log = logs_by_model[model_for_transfer]
            print(f"Running adaptive experiment with {model_for_transfer} on task {task_name}")
            
            self.run_adaptive_experiment(
                initial_log_path=chosen_log.location,
                original_eval_model_name=model_for_transfer,
                task_name=task_name,
                positive_samples_list=[1],
                negative_samples_list=[8],
                log_dir=task_adaptive_log_dir,
                cot_in_context=self.config.use_cot_in_context_attacker,
                use_cot=self.config.use_cot_target,
                similarity_threshold=similarity_threshold,
                use_embeddings=use_embeddings,
                num_epochs=self.config.num_epochs,
            )
        
        print(f"Completed pipeline for task: {task_name}")

@click.command()
@click.option("--models-for-transfer", default=["openai/gpt-4o-mini"], multiple=True, help="Models' logs to use for adaptive question generation.")
@click.option("--experiment-csv", default=None, help="Path to the CSV file where experiment results will be stored.")
@click.option("--cache-csv", default=None, help="Path to the CSV file where cached logs will be stored.")
@click.option("--similarity-threshold", default=0.6, type=float, help="Question similarity threshold for novelty checking.")
@click.option("--use-embeddings", is_flag=True, help="If True, use embeddings to compare question similarity.")
@click.option("--tasks", default=None, multiple=True, help="Specific tasks to run. If not provided, all default tasks will be used.")
@click.option("--num-epochs", default=100, type=int, help="Number of epochs to run for adaptive question generation.")
def main(
    models_for_transfer: List[str],
    experiment_csv: Optional[str],
    cache_csv: Optional[str],
    similarity_threshold: float,
    use_embeddings: bool,
    tasks: Optional[List[str]],
    num_epochs: int,
):
    # Run experiments with different configurations
    # for cot_in_context in [True, False]:
        # for use_example in [True, False]:
        #     for use_cot_target in [True, False]:
    for cot_in_context in [True]:
        for use_example in [True]:
            for use_cot_target in [True, False]:
                runner = TransferLegalExperimentRunner(
                    cache_csv=cache_csv, 
                    experiment_csv=experiment_csv,
                    use_example=use_example, 
                    use_cot_target=use_cot_target, 
                    use_cot_in_context_attacker=cot_in_context,
                    num_epochs=num_epochs,
                )
                
                # If specific tasks were provided, override the default task list
                if tasks:
                    runner.task_names = list(tasks)
                
                # Run the complete pipeline for each task
                for task_name in runner.task_names:
                    runner.run_task_pipeline(
                        task_name=task_name,
                        models_for_transfer=list(models_for_transfer),
                        similarity_threshold=similarity_threshold,
                        use_embeddings=use_embeddings,
                    )


if __name__ == "__main__":
    main() 