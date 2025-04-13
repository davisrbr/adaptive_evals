import os
import csv
from typing import Dict, List, Optional, Tuple
import click
from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog, read_eval_log
from tasks.task_politeness import adaptive_politeness, politeness_n_shot, re_evaluate_adaptive_politeness
import logging
import json
from datetime import datetime
from inspect_ai import Task
from utils_elicitation.novelty import write_novelty_results

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


def extract_accuracy_metrics(eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Extract overall accuracy metrics from the eval log.
    Returns (accuracy, accuracy_judged, scorer_name)
    """
    if not eval_log:
        return None, None, None
    
    # Check if metrics are available in the results section
    if not hasattr(eval_log, 'results') or not eval_log.results:
        return None, None, None
    
    # Normal accuracy
    accuracy = None
    scorer = None
    
    # Try to find accuracy in the results
    for result in eval_log.results:
        if not hasattr(result, 'metrics') or not result.metrics:
            continue
            
        for scorer_name, metrics in result.metrics.items():
            if "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                scorer = scorer_name
                break
        
        if accuracy is not None:
            break
    
    # Judge-filtered accuracy if available
    accuracy_judged = None
    for result in eval_log.results:
        if not hasattr(result, 'metrics') or not result.metrics:
            continue
            
        for scorer_name, metrics in result.metrics.items():
            if "accuracy_judged" in metrics:
                accuracy_judged = metrics["accuracy_judged"]
                break
                
        if accuracy_judged is not None:
            break
    
    return accuracy, accuracy_judged, scorer


def parse_re_eval_counts(eval_log: EvalLog) -> Tuple[int, int, int]:
    """
    Parse counts from re-eval logs:
    - Number of samples that passed the judge
    - Number of samples that were incorrect in the adaptive pass
    - Number of samples that were incorrect in the re-eval
    """
    if not eval_log:
        return 0, 0, 0
    
    # Check if store attribute exists and has re_eval_stats
    metadata = getattr(eval_log, "store", {}).get("re_eval_stats", {})
    if not metadata and hasattr(eval_log, "results") and eval_log.results:
        # Try results.metadata if store isn't available
        metadata = eval_log.results.metadata or {}
        metadata = metadata.get("re_eval_stats", {})
    
    return (
        metadata.get("passed_judge_count", 0),
        metadata.get("adaptive_incorrect_count", 0),
        metadata.get("re_eval_incorrect_count", 0),
    )


def write_experiment_log(
    experiment_csv: str,
    eval_model_name: str,
    generator_model_name: str,
    re_eval_model_name: str,
    initial_log_path: str,
    adaptive_log_path: str,
    re_eval_log_path: Optional[str],
    use_cot: bool,
    similarity_threshold: float,
    score_threshold: int,
    use_embeddings: bool,
    filter_incorrect: bool,
    adaptive_accuracy: Optional[float],
    adaptive_accuracy_judged: Optional[float],
    adaptive_scorer_name: Optional[str],
    re_eval_accuracy: Optional[float],
    re_eval_accuracy_judged: Optional[float],
    re_eval_scorer_name: Optional[str],
    judge_model_name: Optional[str],
    positive_samples: int,
    negative_samples: int,
    passed_judge_count: int,
    adaptive_incorrect_count: int,
    re_eval_incorrect_count: int,
    novelty_results_file: Optional[str],
    num_epochs: int = 1,
) -> None:
    """
    Write an experiment log entry to the CSV file.
    Creates the file with headers if it does not exist.
    """
    file_exists = os.path.exists(experiment_csv)
    with open(experiment_csv, mode="a", newline="") as f:
        fieldnames = [
            "timestamp",
            "eval_model",
            "generator_model",
            "re_eval_model",
            "initial_log_path",
            "adaptive_log_path",
            "re_eval_log_path",
            "use_cot",
            "similarity_threshold",
            "score_threshold",
            "use_embeddings",
            "filter_incorrect",
            "adaptive_accuracy",
            "adaptive_accuracy_judged",
            "adaptive_scorer",
            "re_eval_accuracy",
            "re_eval_accuracy_judged",
            "re_eval_scorer",
            "judge_model",
            "positive_samples",
            "negative_samples",
            "passed_judge_count",
            "adaptive_incorrect_count", 
            "re_eval_incorrect_count",
            "novelty_results_file",
            "num_epochs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "eval_model": eval_model_name,
            "generator_model": generator_model_name,
            "re_eval_model": re_eval_model_name,
            "initial_log_path": initial_log_path,
            "adaptive_log_path": adaptive_log_path,
            "re_eval_log_path": re_eval_log_path or "",
            "use_cot": str(use_cot),
            "similarity_threshold": str(similarity_threshold),
            "score_threshold": str(score_threshold),
            "use_embeddings": str(use_embeddings),
            "filter_incorrect": str(filter_incorrect),
            "adaptive_accuracy": str(adaptive_accuracy) if adaptive_accuracy is not None else "",
            "adaptive_accuracy_judged": str(adaptive_accuracy_judged) if adaptive_accuracy_judged is not None else "",
            "adaptive_scorer": adaptive_scorer_name or "",
            "re_eval_accuracy": str(re_eval_accuracy) if re_eval_accuracy is not None else "",
            "re_eval_accuracy_judged": str(re_eval_accuracy_judged) if re_eval_accuracy_judged is not None else "",
            "re_eval_scorer": re_eval_scorer_name or "",
            "judge_model": judge_model_name or "",
            "positive_samples": str(positive_samples),
            "negative_samples": str(negative_samples),
            "passed_judge_count": str(passed_judge_count),
            "adaptive_incorrect_count": str(adaptive_incorrect_count),
            "re_eval_incorrect_count": str(re_eval_incorrect_count),
            "novelty_results_file": novelty_results_file or "",
            "num_epochs": str(num_epochs),
        })


def check_experiment_already_run(
    experiment_csv: str,
    eval_model_name: str,
    generator_model_name: str,
    re_eval_model_name: str,
    positive_samples: int,
    negative_samples: int,
    use_cot: bool,
    similarity_threshold: float,
    use_embeddings: bool,
    filter_incorrect: bool,
) -> bool:
    """
    Checks if a specific experiment configuration has already been run by looking at the CSV.
    Returns True if the experiment has been run successfully, False otherwise.
    """
    if not os.path.exists(experiment_csv):
        return False
    
    with open(experiment_csv, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Check if all parameters match
                if (row["eval_model_name"] == eval_model_name and
                    row["generator_model_name"] == generator_model_name and
                    row["re_eval_model_name"] == re_eval_model_name and
                    int(row["positive_samples"] if row["positive_samples"] else 0) == positive_samples and
                    int(row["negative_samples"] if row["negative_samples"] else 0) == negative_samples and
                    str(row["use_cot"]).lower() == str(use_cot).lower() and
                    float(row["similarity_threshold"] if row["similarity_threshold"] else 0) == similarity_threshold and
                    str(row["use_embeddings"]).lower() == str(use_embeddings).lower() and
                    str(row["filter_incorrect"]).lower() == str(filter_incorrect).lower()):
                    
                    # Check if the experiment has valid paths and results
                    if row["adaptive_log_path"] and row["adaptive_accuracy"] is not None:
                        print(f"Experiment already run for {eval_model_name} with generator {generator_model_name} and re-eval {re_eval_model_name}")
                        return True
            except (ValueError, KeyError, TypeError):
                # Skip rows with missing or invalid data
                continue
    
    return False


class ExperimentConfig:
    """Configuration class for Politeness Experiment Runner"""
    
    def __init__(
        self,
        use_cot: bool = False,
        use_cot_target: bool = False,
        use_cot_in_context_attacker: bool = False,
        use_example: bool = True,
        experiment_id: Optional[str] = None,
    ):
        self.use_cot = use_cot
        self.use_cot_target = use_cot_target
        self.use_cot_in_context_attacker = use_cot_in_context_attacker
        self.use_example = use_example
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiment_id = experiment_id
    
    @property
    def experiment_id(self) -> str:
        """Generate a readable experiment identifier"""
        if self._experiment_id:
            return self._experiment_id
            
        components = []
        if self.use_cot:
            components.append("cot")
        if self.use_cot_target:
            components.append("cot_target")
        if self.use_cot_in_context_attacker:
            components.append("cot_attacker")
        if self.use_example:
            components.append("with_examples")
        components.append(self.timestamp)
        
        return "_".join(components) or "base"
    
    def get_cache_path(self) -> str:
        """Get the path for the cache CSV"""
        return os.path.join("cache", f"politeness_initial_{self.experiment_id}.csv")
    
    def get_initial_log_dir(self) -> str:
        """Get the directory for initial evaluation logs"""
        return os.path.join("logs", "politeness", "initial", self.experiment_id)
    
    def get_adaptive_log_dir(self) -> str:
        """Get the directory for adaptive evaluation logs"""
        return os.path.join("logs", "politeness", "adaptive", self.experiment_id)
    
    def get_results_dir(self) -> str:
        """Get the directory for detailed results"""
        return os.path.join("results", "politeness", self.experiment_id)


class TransferPolitenessExperimentRunner:
    def __init__(
        self,
        use_cot: bool = False,
        use_cot_target: bool = False,
        use_cot_in_context_attacker: bool = False,
        use_example: bool = True,
        experiment_id: Optional[str] = None,
        cache_csv: Optional[str] = None,
        experiment_csv: Optional[str] = None,
        randomize_sampling: bool = False,
    ):
        self.config = ExperimentConfig(
            use_cot=use_cot,
            use_cot_target=use_cot_target,
            use_cot_in_context_attacker=use_cot_in_context_attacker,
            use_example=use_example,
            experiment_id=experiment_id,
        )
        
        # Use provided CSV paths or generate from config
        self.cache_csv = cache_csv or self.config.get_cache_path()
        self.experiment_csv = experiment_csv or os.path.join(
            self.config.get_results_dir(), 
            f"politeness_experiment_results_{self.config.experiment_id}.csv"
        )
        
        print(f"Using experiment CSV: {self.experiment_csv}")
        print(f"Using cache CSV: {self.cache_csv}")
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.experiment_csv), exist_ok=True)
        os.makedirs(self.config.get_initial_log_dir(), exist_ok=True)
        os.makedirs(self.config.get_adaptive_log_dir(), exist_ok=True)
        os.makedirs(self.config.get_results_dir(), exist_ok=True)
        
        # Models for the initial pass
        self.initial_eval_models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/o3-mini",
            "anthropic/claude-3-5-sonnet-latest",
        ]
        self.generator_models = [
            "openai/gpt-4o",
            # "openai/o3-mini",
        ]
        # Models on which we'll perform the adaptive evaluation
        self.adaptive_evaluated_models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/o3-mini",
            "anthropic/claude-3-5-sonnet-latest",
        ]
        self.randomize_sampling = randomize_sampling

    def run_initial_experiment(
        self,
        n_examples: int = 3,
        num_samples: int = 100,
        cot: bool = False,
        log_dir: Optional[str] = None,
    ) -> Dict[str, EvalLog]:
        """
        Runs the initial politeness evaluation task on each model in self.initial_eval_models.
        Returns a dict of {model_name: eval_log}.
        """
        log_dir = log_dir or self.config.get_initial_log_dir()
        os.makedirs(log_dir, exist_ok=True)

        # Check cache for existing logs
        cache = read_eval_cache(self.cache_csv)
        
        # Set up the task
        task = politeness_n_shot(n_examples=5, cot=cot, max_samples=num_samples)
        
        # Evaluate each model
        logs = {}
        for model_name in self.initial_eval_models:
            # Check if this (model_name, task) combo is already in the cache
            if model_name in cache:
                cached_log_path = cache[model_name]
                if os.path.exists(cached_log_path):
                    # Load the cached log
                    try:
                        log = read_eval_log(cached_log_path)
                        print(f"Using cached log for {model_name} from {cached_log_path}")
                        logs[model_name] = log
                        continue
                    except Exception as exc:
                        print(f"Failed to load cached log for {model_name}: {exc}")
            
            # Run the evaluation
            print(f"Running initial evaluation for {model_name}")
            model_log_dir = os.path.join(log_dir, model_name.replace("/", "_"))
            os.makedirs(model_log_dir, exist_ok=True)
            
            eval_logs = eval(
                task,
                epochs=Epochs(1, "mean"),
                max_connections=50,
                log_dir=model_log_dir,
                model=model_name,
                temperature=0,
                log_level="critical",
            )
            
            if not eval_logs or eval_logs[0].status != "success":
                print(f"Initial evaluation failed for {model_name}")
                continue
                
            eval_log = eval_logs[0]
            logs[model_name] = eval_log
            
            # Cache the log
            write_eval_cache(self.cache_csv, model_name, eval_log.location)
            
        return logs

    def run_adaptive_experiment(
        self,
        initial_log_path: str,
        original_eval_model_name: str,
        positive_samples_list: List[int] = [1],
        negative_samples_list: List[int] = [8],
        log_dir: Optional[str] = None,
        cot_in_context: bool = False,
        use_cot: bool = True,
        similarity_threshold: float = 0.6,
        use_embeddings: bool = False,
        num_epochs: int = 1,
    ):
        """
        Uses a single (initial_log_path) from the original model, 
        then runs adaptive experiments with each generator model 
        on each of the adaptive_eval_models.
        """
        task_specific_log_dir = log_dir or self.config.get_adaptive_log_dir()
        os.makedirs(task_specific_log_dir, exist_ok=True)
        
        # Create a task-specific experiment CSV
        task_experiment_csv = os.path.join(
            os.path.dirname(self.experiment_csv), 
            f"politeness_experiment_results_{self.config.experiment_id}.csv"
        )
        
        for pos_samples in positive_samples_list:
            for neg_samples in negative_samples_list:
                for generator_model in self.generator_models:
                    for eval_model in self.adaptive_evaluated_models:
                        # Check both main and task-specific experiment CSVs for existing runs
                        if check_experiment_already_run(
                            experiment_csv=self.experiment_csv,
                            eval_model_name=eval_model,
                            generator_model_name=generator_model,
                            re_eval_model_name=eval_model,  # First check with same re-eval model
                            positive_samples=pos_samples,
                            negative_samples=neg_samples,
                            use_cot=use_cot,
                            similarity_threshold=similarity_threshold,
                            use_embeddings=use_embeddings,
                            filter_incorrect=True,
                        ) or check_experiment_already_run(
                            experiment_csv=task_experiment_csv,
                            eval_model_name=eval_model,
                            generator_model_name=generator_model,
                            re_eval_model_name=eval_model,  # First check with same re-eval model
                            positive_samples=pos_samples,
                            negative_samples=neg_samples,
                            use_cot=use_cot,
                            similarity_threshold=similarity_threshold,
                            use_embeddings=use_embeddings,
                            filter_incorrect=True,
                        ):
                            print(f"Skipping already completed experiment for {eval_model} with generator {generator_model}")
                            continue
                            
                        task = adaptive_politeness(
                            initial_log_path=initial_log_path,
                            n_positive_samples=pos_samples,
                            n_negative_samples=neg_samples,
                            generator_model_name=generator_model,
                            eval_model_name=eval_model,
                            use_cot_generator=True,
                            use_cot_evaluator=use_cot,
                            cot_in_context=cot_in_context,
                            randomize_sampling=self.randomize_sampling,
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
                            max_connections=50,
                            log_dir=combination_log_dir,
                            model=eval_model,
                            temperature=0,
                            log_level="critical",
                        )
                        
                        if not adaptive_logs or adaptive_logs[0].status != "success":
                            print(f"Adaptive experiment failed for {eval_model} with generator {generator_model}")
                            continue
                            
                        adaptive_log = adaptive_logs[0]
                        adaptive_accuracy, adaptive_accuracy_judged, adaptive_scorer = extract_accuracy_metrics(adaptive_log)
                        
                        # Compute novelty results
                        results_dir = os.path.join(os.path.dirname(self.experiment_csv))
                        os.makedirs(results_dir, exist_ok=True)
                        novelty_file = os.path.join(
                            results_dir,
                            f"novelty_politeness_{eval_model.replace('/', '_')}_{generator_model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
                            print(f"Novelty filtering failed for (eval={eval_model}, gen={generator_model}). Error: {exc}")
                            novelty_file = None
                            
                        # Re-evaluation step for each model (transfer)
                        for re_eval_model in self.adaptive_evaluated_models:
                            # Check if this specific re-evaluation experiment has already been run
                            if check_experiment_already_run(
                                experiment_csv=self.experiment_csv,
                                eval_model_name=eval_model,
                                generator_model_name=generator_model,
                                re_eval_model_name=re_eval_model,
                                positive_samples=pos_samples,
                                negative_samples=neg_samples,
                                use_cot=use_cot,
                                similarity_threshold=similarity_threshold,
                                use_embeddings=use_embeddings,
                                filter_incorrect=True,
                            ) or check_experiment_already_run(
                                experiment_csv=task_experiment_csv,
                                eval_model_name=eval_model,
                                generator_model_name=generator_model,
                                re_eval_model_name=re_eval_model,
                                positive_samples=pos_samples,
                                negative_samples=neg_samples,
                                use_cot=use_cot,
                                similarity_threshold=similarity_threshold,
                                use_embeddings=use_embeddings,
                                filter_incorrect=True,
                            ):
                                print(f"Skipping already completed re-evaluation for {re_eval_model} on questions from {eval_model} generated by {generator_model}")
                                continue
                            
                            print(f"Re-evaluating model {re_eval_model} on questions from {eval_model} generated by {generator_model}")
                            
                            # Create re-evaluation task
                            re_eval_task = re_evaluate_adaptive_politeness(
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
                                    f"Re-evaluation success for (eval={eval_model}, "
                                    f"gen={generator_model}, re-eval={re_eval_model})."
                                )
                                re_eval_acc, re_eval_acc_judged, re_eval_scorer = extract_accuracy_metrics(re_eval_logs[0])
                                passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = parse_re_eval_counts(
                                    re_eval_logs[0]
                                )
                            else:
                                print(
                                    f"Re-evaluation returned no logs or non-success status for "
                                    f"(eval={eval_model}, gen={generator_model}, re-eval={re_eval_model})."
                                )
                                passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = (0, 0, 0)

                            
                            # Create a task-specific experiment CSV
                            task_experiment_csv = os.path.join(
                                os.path.dirname(self.experiment_csv), 
                                f"politeness_experiment_results_{self.config.experiment_id}.csv"
                            )
                            
                            write_experiment_log(
                                experiment_csv=task_experiment_csv,
                                eval_model_name=eval_model,
                                generator_model_name=generator_model,
                                re_eval_model_name=re_eval_model,
                                initial_log_path=initial_log_path,
                                adaptive_log_path=adaptive_log.location,
                                re_eval_log_path=re_eval_log_path,
                                use_cot=use_cot,
                                similarity_threshold=similarity_threshold,
                                score_threshold=4,  # Default score threshold for politeness
                                use_embeddings=use_embeddings,
                                filter_incorrect=True,
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
                                use_cot=use_cot,
                                similarity_threshold=similarity_threshold,
                                score_threshold=4,  # Default score threshold for politeness
                                use_embeddings=use_embeddings,
                                filter_incorrect=True,
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
        models_for_transfer: List[str],
        positive_samples_list: List[int] = [1],
        negative_samples_list: List[int] = [8],
        num_epochs: int = 1,
        n_examples: int = 3,
        num_samples: int = 500,
        cot: bool = False,
        cot_in_context: bool = False,
        similarity_threshold: float = 0.6,
        use_embeddings: bool = False,
    ):
        """
        Runs the complete pipeline for politeness evaluations:
        1. Initial evaluation of each model
        2. Adaptive evaluation with each generator model
        3. Re-evaluation with models_for_transfer
        """
        # Update models for adaptive evaluation and re-evaluation
        self.adaptive_evaluated_models = models_for_transfer
        
        # 1. Run initial evaluation
        initial_logs = self.run_initial_experiment(
            n_examples=n_examples,
            num_samples=num_samples,
            cot=cot
        )
        
        if not initial_logs:
            print("No successful initial evaluation logs. Cannot proceed.")
            return
        
        # 2 & 3. Run adaptive experiments and re-evaluations for each model
        for model_name, log in initial_logs.items():
            print(f"\nRunning adaptive experiments for initial model {model_name}\n")
            
            self.run_adaptive_experiment(
                initial_log_path=log.location,
                original_eval_model_name=model_name,
                positive_samples_list=positive_samples_list,
                negative_samples_list=negative_samples_list,
                cot_in_context=cot_in_context,
                use_cot=cot,
                similarity_threshold=similarity_threshold,
                use_embeddings=use_embeddings,
                num_epochs=num_epochs,
            )


@click.command()
@click.option('--models-for-transfer', default=[
                "openai/gpt-4o",
                "openai/gpt-4o-mini",
                "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "openai/o3-mini",
                "anthropic/claude-3-5-sonnet-latest",
            ], multiple=True, help='Models to use for transfer evaluation')
@click.option('--positive-samples', default=[1], multiple=True, type=int, help='Number of positive samples to use')
@click.option('--negative-samples', default=[8], multiple=True, type=int, help='Number of negative samples to use')
@click.option('--num-epochs', default=1, help='Number of epochs to run for adaptive evaluation')
@click.option('--num-samples', default=1000, help='Number of samples to use for initial evaluation')
@click.option('--n-examples', default=5, help='Number of in-context examples to use')
@click.option('--use-cot', is_flag=True, help='Use chain-of-thought for solver prompts')
@click.option('--cot-in-context', is_flag=True, help='Use chain-of-thought in context examples')
@click.option('--similarity-threshold', default=0.6, type=float, help='Similarity threshold for novelty scoring')
@click.option('--use-embeddings', is_flag=True, help='Use embeddings for similarity checking')
@click.option('--experiment-id', default=None, help='Custom experiment ID')
@click.option('--experiment-csv', default=None, help='Path to the CSV file where experiment results will be stored')
@click.option('--cache-csv', default=None, help='Path to the CSV file where cached logs will be stored')
@click.option('--randomize-sampling', is_flag=True, help='If True, use random sampling when generating questions.')
def main(
    models_for_transfer,
    positive_samples,
    negative_samples,
    num_epochs,
    num_samples,
    n_examples,
    use_cot,
    cot_in_context,
    similarity_threshold,
    use_embeddings,
    experiment_id,
    experiment_csv,
    cache_csv,
    randomize_sampling,
):
    """Run politeness transfer experiments"""
    
    # Convert tuple options to lists
    models_for_transfer = list(models_for_transfer) or ["openai/gpt-4o-mini"]
    positive_samples = list(positive_samples) or [1]
    negative_samples = list(negative_samples) or [8]
    
    print(f"Running politeness transfer experiments with models: {models_for_transfer}")
    print(f"Positive samples: {positive_samples}")
    print(f"Negative samples: {negative_samples}")
    print(f"Num epochs: {num_epochs}")
    print(f"Using CoT: {use_cot}")
    print(f"CoT in context: {cot_in_context}")
    
    runner = TransferPolitenessExperimentRunner(
        use_cot=use_cot,
        use_cot_target=use_cot,
        use_cot_in_context_attacker=cot_in_context,
        experiment_id=experiment_id,
        experiment_csv=experiment_csv,
        cache_csv=cache_csv,
        randomize_sampling=randomize_sampling,
    )
    
    runner.run_task_pipeline(
        models_for_transfer=models_for_transfer,
        positive_samples_list=positive_samples,
        negative_samples_list=negative_samples,
        num_epochs=num_epochs,
        num_samples=num_samples,
        n_examples=n_examples,
        cot=use_cot,
        cot_in_context=cot_in_context,
        similarity_threshold=similarity_threshold,
        use_embeddings=use_embeddings,
    )


if __name__ == "__main__":
    main()
