import os
import csv
import json
import logging
from typing import Dict, List, Optional, Tuple
import click

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai import eval, Epochs
from tasks.task_politeness import politeness_n_shot, adaptive_politeness

# Turn off verbose logging from various libraries
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("inspect_ai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def read_eval_cache(cache_csv: str) -> Dict[str, str]:
    """
    Returns a dict of {model_name: log_path} previously saved
    for initial evaluations.
    """
    if not os.path.exists(cache_csv):
        return {}
    output: Dict[str, str] = {}
    with open(cache_csv, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            output[row["model_name"]] = row["log_path"]
    return output


def write_eval_cache(cache_csv: str, model_name: str, log_path: str) -> None:
    """
    Appends the given (model_name, log_path) to the CSV cache.
    Creates the file with headers if it does not exist.
    """
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    file_exists = os.path.exists(cache_csv)
    with open(cache_csv, mode="a", newline="") as f:
        fieldnames = ["model_name", "log_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model_name": model_name, "log_path": log_path})


def parse_accuracy_metrics(eval_log: EvalLog) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse the log results and return (accuracy, scorer_name).
    Because we might have multiple scorers, we look for the first that
    provides 'accuracy' in its metrics.
    """
    if not eval_log or not eval_log.results:
        return None, None

    for score_item in (eval_log.results.scores or []):
        metrics = score_item.metrics or {}
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            return (accuracy.value, score_item.scorer)
    return None, None


def parse_judge_metadata(eval_log: EvalLog) -> Tuple[str, str]:
    """
    Aggregate language and judge_choice metadata across samples.
    Return (language_counts_json, judge_choice_counts_json).
    """
    if not eval_log or not eval_log.samples:
        return "{}", "{}"

    language_counts: Dict[str, int] = {}
    judge_choice_counts: Dict[str, int] = {}

    for sample in eval_log.samples:
        sample_judge = sample.store.get("generated_sample")
        lang = sample_judge.get("metadata", {}).get("language")
        choice = sample_judge.get("metadata", {}).get("judge_choice")

        if lang:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        if choice:
            judge_choice_counts[choice] = judge_choice_counts.get(choice, 0) + 1

    return json.dumps(language_counts), json.dumps(judge_choice_counts)


def write_experiment_log(
    experiment_csv: str,
    eval_model_name: str,
    generator_model_name: Optional[str],
    initial_log_path: Optional[str],
    adaptive_log_path: Optional[str],
    use_cot: bool,
    similarity_threshold: float,
    score_threshold: int,
    use_embeddings: bool,
    accuracy: Optional[float],
    scorer_name: Optional[str],
    judge_language_counts: str,
    judge_choice_counts: str,
    n_datapoints: int,
    max_attempts: int,
) -> None:
    """
    Write a single row into the experiment CSV, capturing these metadata.
    """
    file_exists = os.path.exists(experiment_csv)
    fieldnames = [
        "eval_model_name",
        "generator_model_name",
        "initial_log_path",
        "adaptive_log_path",
        "use_cot",
        "similarity_threshold",
        "score_threshold",
        "use_embeddings",
        "accuracy",
        "scorer_name",
        "judge_language_counts",
        "judge_choice_counts",
        "n_datapoints",
        "max_attempts",
    ]
    os.makedirs(os.path.dirname(experiment_csv), exist_ok=True)
    with open(experiment_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "eval_model_name": eval_model_name,
                "generator_model_name": generator_model_name,
                "initial_log_path": initial_log_path,
                "adaptive_log_path": adaptive_log_path,
                "use_cot": use_cot,
                "similarity_threshold": similarity_threshold,
                "score_threshold": score_threshold,
                "use_embeddings": use_embeddings,
                "accuracy": accuracy,
                "scorer_name": scorer_name,
                "judge_language_counts": judge_language_counts,
                "judge_choice_counts": judge_choice_counts,
                "n_datapoints": n_datapoints,
                "max_attempts": max_attempts,
            }
        )


class PolitenessExperimentRunner:
    """
    An experiment runner for politeness tasks:
      1) Evaluate a set of models using politeness_n_shot as the initial step.
      2) For each of those, run adaptive_politeness with a set of generator models.
      3) Store results in a CSV cache and a separate experiment CSV log.
    """

    def __init__(
        self,
        use_cot: bool = False,
        similarity_threshold: float = 0.6,
        score_threshold: int = 4,
        use_embeddings: bool = False,
        n_datapoints: int = 40,
        max_attempts: int = 5,
        logs_dir: str = "logs_politeness",
        experiment_csv: str = "results/politeness_experiment_results.csv",
        cache_csv: str = "results/politeness_experiment_cache.csv",
    ):
        self.use_cot = use_cot
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.use_embeddings = use_embeddings
        self.n_datapoints = n_datapoints
        self.max_attempts = max_attempts
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.experiment_csv = experiment_csv
        self.cache_csv = cache_csv
        os.makedirs(os.path.dirname(self.experiment_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)

    def run_initial_experiments(self, models: List[str]) -> Dict[str, EvalLog]:
        """
        Runs or loads politeness_n_shot for each model in 'models'.
        Returns {model_name: EvalLog} for each successful evaluation.
        """
        logs_by_model: Dict[str, EvalLog] = {}
        cached = read_eval_cache(self.cache_csv)

        for model_name in models:
            if model_name in cached:
                maybe_log = read_eval_log(cached[model_name])
                if maybe_log and maybe_log.status == "success":
                    logs_by_model[model_name] = maybe_log
                    print(f"[Initial Politeness] Using cached log for: {model_name}")
                    continue

            print(f"[Initial Politeness] Running politeness_n_shot for: {model_name}")
            task_n_shot = politeness_n_shot(n_examples=5, debug=self.n_datapoints, cot=self.use_cot)
            init_logdir = os.path.join(self.logs_dir, f"initial_{model_name.replace('/', '_')}")
            os.makedirs(init_logdir, exist_ok=True)
            init_logs = eval(
                task_n_shot,
                epochs=Epochs(1, "max"),
                log_dir=init_logdir,
                model=model_name,
                log_level="critical",
            )
            if not init_logs or init_logs[0].status != "success":
                print(f"[Initial Politeness] No success log for {model_name}, skipping.")
                continue

            logs_by_model[model_name] = init_logs[0]
            write_eval_cache(self.cache_csv, model_name, init_logs[0].location)

        return logs_by_model

    def run_adaptive_eval(
        self,
        initial_log_path: str,
        eval_model_name: str,
        generator_models: List[str],
    ) -> None:
        """
        For each generator model, run adaptive_politeness on the new questions
        and log the results.
        """
        for generator_name in generator_models:
            print(f"[Adaptive Politeness] eval={eval_model_name}, generator={generator_name}")
            adaptive_task = adaptive_politeness(
                initial_log_path=initial_log_path,
                n_positive_samples=2,
                n_negative_samples=8,
                generator_model_name=generator_name,
                eval_model_name=eval_model_name,
                use_embeddings=self.use_embeddings,
                similarity_threshold=self.similarity_threshold,
                score_threshold=self.score_threshold,
                max_attempts=self.max_attempts,
                cot_in_context=self.use_cot,
                use_cot_generator=self.use_cot,
                use_cot_evaluator=self.use_cot,
                judge_model_name="together/deepseek-ai/DeepSeek-V3",
            )
            adaptive_logdir = os.path.join(
                self.logs_dir, f"adaptive_{eval_model_name.replace('/', '_')}__{generator_name.replace('/', '_')}"
            )
            os.makedirs(adaptive_logdir, exist_ok=True)
            adaptive_logs = eval(
                adaptive_task,
                epochs=Epochs(self.n_datapoints, "max"),
                log_dir=adaptive_logdir,
                model=eval_model_name,
                log_level="critical",
            )

            if not adaptive_logs or adaptive_logs[0].status != "success":
                print(f"[Adaptive Politeness] No success log (eval={eval_model_name}, gen={generator_name}). Skipped.")
                continue

            adaptive_log = adaptive_logs[0]
            accuracy, scorer = parse_accuracy_metrics(adaptive_log)
            language_counts, judge_choice_counts = parse_judge_metadata(adaptive_log)

            write_experiment_log(
                experiment_csv=self.experiment_csv,
                eval_model_name=eval_model_name,
                generator_model_name=generator_name,
                initial_log_path=initial_log_path,
                adaptive_log_path=adaptive_log.location,
                use_cot=self.use_cot,
                similarity_threshold=self.similarity_threshold,
                score_threshold=self.score_threshold,
                use_embeddings=self.use_embeddings,
                accuracy=accuracy,
                scorer_name=scorer,
                judge_language_counts=language_counts,
                judge_choice_counts=judge_choice_counts,
                n_datapoints=self.n_datapoints,
                max_attempts=self.max_attempts,
            )

    def run_all(self, eval_models: List[str], generator_models: List[str]) -> None:
        """
        1) Runs or loads initial politeness_n_shot logs for each model in eval_models.
        2) For each of those logs, runs adaptive_politeness with each generator model.
        """
        logs_by_model = self.run_initial_experiments(eval_models)
        for eval_model_name, init_log in logs_by_model.items():
            self.run_adaptive_eval(
                initial_log_path=init_log.location,
                eval_model_name=eval_model_name,
                generator_models=generator_models,
            )


@click.command()
@click.option(
    "--experiment-csv",
    default="results/politeness_experiment_results.csv",
    help="Path to the CSV file where experiment results will be stored.",
)
@click.option(
    "--cache-csv",
    default="results/politeness_experiment_cache.csv",
    help="Path to the CSV file where cached logs will be stored.",
)
@click.option(
    "--eval-models",
    default=["openai/gpt-4o-mini"],
    multiple=True,
    help="Models to run the initial politeness_n_shot evaluations.",
)
@click.option(
    "--generator-models",
    default=["openai/gpt-4o-mini"],
    multiple=True,
    help="Models to generate new utterances in the adaptive step.",
)
@click.option("--n-datapoints", default=40, type=int, help="Number of datapoints for both initial and adaptive tasks.")
@click.option("--use-cot", is_flag=True, help="Use chain-of-thought for solver prompts.")
@click.option(
    "--similarity-threshold",
    default=0.3,
    type=float,
    help="Utterance similarity threshold for novelty filtering in the adaptive step.",
)
@click.option(
    "--score-threshold",
    default=4,
    type=int,
    help="Score threshold for deciding if the model was correct or not (per the self-check logic).",
)
@click.option(
    "--use-embeddings",
    is_flag=True,
    help="If True, use embeddings to compare utterance similarity for novelty filtering.",
)
@click.option("--max-attempts", default=5, type=int, help="Max attempts to generate a valid new utterance.")
def main(
    experiment_csv: str,
    cache_csv: str,
    eval_models: List[str],
    generator_models: List[str],
    n_datapoints: int,
    use_cot: bool,
    similarity_threshold: float,
    score_threshold: int,
    use_embeddings: bool,
    max_attempts: int,
) -> None:
    """
    1) Runs politeness_n_shot for each model in --eval-models.
    2) Then runs adaptive_politeness for each (eval-model, generator-model) pair.
    3) Stores logs and metrics, including judge language metadata, into a CSV.
    """
    runner = PolitenessExperimentRunner(
        use_cot=use_cot,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
        use_embeddings=use_embeddings,
        n_datapoints=n_datapoints,
        max_attempts=max_attempts,
        experiment_csv=experiment_csv,
        cache_csv=cache_csv,
    )
    runner.run_all(
        eval_models=list(eval_models),
        generator_models=list(generator_models),
    )


if __name__ == "__main__":
    main() 