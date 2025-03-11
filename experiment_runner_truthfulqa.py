import os
import csv
import logging
import json
from typing import Dict, List, Optional, Tuple
import click
import ast
from datetime import datetime

from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog, read_eval_log

# Import the tasks we need
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


def read_eval_cache(cache_csv: str) -> Dict[str, str]:
    """
    Returns a dict of {model_name: log_path} previously saved
    for the initial evaluations.
    """
    if not os.path.exists(cache_csv):
        return {}
    output = {}
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


def extract_accuracy_metrics(eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Parse the log results and return (accuracy, accuracy_judged, scorer_name_used).

    Because re_evaluate_adaptive_truthfulqa uses multiple_choice_save_cot with a "mean" reducer,
    we need to look for:
    - "accuracy" in the metrics for regular accuracy
    - "accuracy_judged" in the metrics for accuracy on judged samples only
    
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
    n_datapoints: int,
    max_attempts: int,
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
    file_exists = os.path.exists(experiment_csv)
    with open(experiment_csv, mode="a", newline="") as f:
        fieldnames = [
            "eval_model_name",
            "generator_model_name",
            "re_eval_model_name",
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
            "adaptive_scorer_name",
            "re_eval_accuracy",
            "re_eval_accuracy_judged",
            "re_eval_scorer_name",
            "judge_model_name",
            "n_datapoints",
            "max_attempts",
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
                "use_cot": use_cot,
                "similarity_threshold": similarity_threshold,
                "score_threshold": score_threshold,
                "use_embeddings": use_embeddings,
                "filter_incorrect": filter_incorrect,
                "adaptive_accuracy": adaptive_accuracy,
                "adaptive_accuracy_judged": adaptive_accuracy_judged,
                "adaptive_scorer_name": adaptive_scorer_name,
                "re_eval_accuracy": re_eval_accuracy,
                "re_eval_accuracy_judged": re_eval_accuracy_judged,
                "re_eval_scorer_name": re_eval_scorer_name,
                "judge_model_name": judge_model_name,
                "n_datapoints": n_datapoints,
                "max_attempts": max_attempts,
                "passed_judge_count": passed_judge_count,
                "adaptive_incorrect_count": adaptive_incorrect_count,
                "re_eval_incorrect_count": re_eval_incorrect_count,
                "novelty_results_file": novelty_results_file,
            }
        )



class TruthfulQAExperimentRunner:
    """
    An experiment runner to:
      1) Evaluate a set of "adaptive evaluator" models using @truthfulqa_initial
      2) For each "adaptive evaluator" model and each generator model, run
         @adaptive_truthfulqa_refactor to generate new questions
      3) For each newly generated set of questions, run @re_evaluate_adaptive_truthfulqa
         with a set of "re-eval" models, to see how they do on those questions
         (transfer).
      4) Store experiment results in a CSV cache and a separate experiment CSV log.
    """

    def __init__(
        self,
        use_cot: bool = False,
        similarity_threshold: float = 0.6,
        score_threshold: int = 4,
        use_embeddings: bool = False,
        n_pos: int = 12,
        n_neg: int = 2,
        n_datapoints: int = 40,
        max_attempts: int = 10,
        re_eval_filter_incorrect: bool = False,
        logs_dir: str = "logs",
        experiment_csv: str = "results/experiment_results.csv",
        cache_csv: str = "results/experiment_cache.csv",
        judge_model_name: Optional[str] = "anthropic/claude-3-5-sonnet-latest",
    ):
        self.use_cot = use_cot
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.use_embeddings = use_embeddings
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.re_eval_filter_incorrect = re_eval_filter_incorrect
        self.max_attempts = max_attempts
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        self.judge_model_name = judge_model_name
        self.n_datapoints = n_datapoints
        self.experiment_csv = experiment_csv
        self.cache_csv = cache_csv
        os.makedirs(os.path.dirname(self.experiment_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)

    def run_initial_experiments(self, adaptive_eval_models: List[str]) -> Dict[str, EvalLog]:
        """
        Runs or loads the results of truthfulqa_initial for each model in adaptive_eval_models.
        Returns a dict {model_name: EvalLog} for each successful evaluation.
        """
        from inspect_ai.log import read_eval_log
        logs_by_model: Dict[str, EvalLog] = {}

        cached = read_eval_cache(self.cache_csv)

        for model_name in adaptive_eval_models:
            if model_name in cached:
                # Load existing
                maybe_log = read_eval_log(cached[model_name])
                if maybe_log and maybe_log.status == "success":
                    logs_by_model[model_name] = maybe_log
                    print(f"[Initial TQA] Using cached log for: {model_name}")
                    continue

            print(f"[Initial TQA] Running truthfulqa_initial for: {model_name}")
            tqa_task = truthfulqa_initial(
                target="mc1",
                use_cot=self.use_cot,
                debug=False,
            )
            init_log_dir = os.path.join(
                self.logs_dir, f"initial_{model_name.replace('/', '_')}"
            )
            os.makedirs(init_log_dir, exist_ok=True)
            init_logs = eval(
                tqa_task,
                epochs=Epochs(1, "max"),
                log_dir=init_log_dir,
                model=model_name,
                log_level="critical",
                max_messages=1000,
            )
            if not init_logs or init_logs[0].status != "success":
                print(f"[Initial TQA] No success log for {model_name}, skipping.")
                continue
            logs_by_model[model_name] = init_logs[0]
            write_eval_cache(self.cache_csv, model_name, init_logs[0].location)
        return logs_by_model

    def run_adaptive_and_transfer_eval(
        self,
        initial_log_path: str,
        adaptive_eval_model_name: str,
        generator_models: List[str],
        re_eval_models: List[str],
    ) -> None:
        """
        Runs adaptive_truthfulqa_refactor for each generator model, then
        re-evaluates those newly generated questions on each re_eval model.
        """
        for generator_name in generator_models:
            print(f"[Adaptive Step] Working on eval={adaptive_eval_model_name}, generator={generator_name}")
            adaptive_task = adaptive_truthfulqa_refactor(
                initial_log_path=initial_log_path,
                n_positive_samples=self.n_pos,
                n_negative_samples=self.n_neg,
                generator_model_name=generator_name,
                eval_model_name=adaptive_eval_model_name,
                self_check_model_name=None,
                target="mc1",
                use_embeddings=self.use_embeddings,
                embeddings_model_name="sentence-transformers/all-mpnet-base-v2",
                similarity_threshold=self.similarity_threshold,
                score_threshold=self.score_threshold,
                max_attempts=self.max_attempts,
                randomize_sampling=False,
                cot_in_context=True,
                use_cot_generator=True,
                use_cot_evaluator=False,
                original_eval_model_name=None,
                judge_model_name=self.judge_model_name,
                use_eval_model_for_checker=True,
            )
            adaptive_logdir = os.path.join(
                self.logs_dir,
                f"adaptive_{adaptive_eval_model_name.replace('/', '_')}__{generator_name.replace('/', '_')}"
            )
            os.makedirs(adaptive_logdir, exist_ok=True)

            adaptive_logs = eval(
                adaptive_task,
                epochs=Epochs(self.n_datapoints, "max"),
                log_dir=adaptive_logdir,
                model=adaptive_eval_model_name,
                log_level="critical",
            )
            if not adaptive_logs or adaptive_logs[0].status != "success":
                print(
                    f"[Adaptive Step] No success log for (eval={adaptive_eval_model_name}, gen={generator_name}). Skipped."
                )
                continue

            adaptive_log = adaptive_logs[0]
            adaptive_accuracy, adaptive_accuracy_judged, adaptive_scorer = extract_accuracy_metrics(adaptive_log)

            # Compute a unique filename for this experiment's novelty filtering results.
            results_dir = os.path.dirname(self.experiment_csv)  # results/ folder
            novelty_file = os.path.join(
                results_dir,
                f"novelty_{adaptive_eval_model_name.replace('/', '_')}_{generator_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            try:
                write_novelty_results(
                    adaptive_log_path=adaptive_log.location,
                    embeddings_model_name="sentence-transformers/all-mpnet-base-v2",
                    similarity_threshold=self.similarity_threshold,
                    output_file=novelty_file,
                    use_embeddings=self.use_embeddings,
                )
            except Exception as exc:
                print(f"[Adaptive Step] Novelty filtering failed for (eval={adaptive_eval_model_name}, gen={generator_name}). Error: {exc}")
                novelty_file = None

            # Re-evaluation step for each re_eval model
            for re_eval_model in re_eval_models:
                print(
                    f"[Re-Eval] Checking how re_eval_model={re_eval_model} does "
                    f"on questions from (eval={adaptive_eval_model_name}, gen={generator_name})."
                )
                re_eval_task = re_evaluate_adaptive_truthfulqa(
                    adaptive_log_path=adaptive_log.location,
                    use_cot=self.use_cot,
                    filter_by_incorrect=self.re_eval_filter_incorrect,
                )
                re_eval_logdir = os.path.join(adaptive_logdir, "re_eval", re_eval_model.replace("/", "_"))
                os.makedirs(re_eval_logdir, exist_ok=True)
                re_eval_log_path: Optional[str] = None
                re_eval_acc: Optional[float] = None
                re_eval_acc_judged: Optional[float] = None
                re_eval_scorer: Optional[str] = None

                try:
                    re_eval_logs = eval(
                        re_eval_task,
                        epochs=Epochs(1, "max"),
                        log_dir=re_eval_logdir,
                        model=re_eval_model,
                        log_level="critical",
                    )
                    if re_eval_logs and re_eval_logs[0].status == "success":
                        re_eval_log_path = re_eval_logs[0].location
                        print(
                            f"[Re-Eval] Success for (eval={adaptive_eval_model_name}, "
                            f"gen={generator_name}, re-eval={re_eval_model})."
                        )
                        re_eval_acc, re_eval_acc_judged, re_eval_scorer = extract_accuracy_metrics(re_eval_logs[0])
                        passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = parse_re_eval_counts(
                            re_eval_logs[0]
                        )
                    else:
                        print(
                            f"[Re-Eval] Returned no logs or non-success status for "
                            f"(eval={adaptive_eval_model_name}, gen={generator_name}, re-eval={re_eval_model})."
                        )
                        passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = (0, 0, 0)
                except Exception as exc:
                    print(
                        f"[Re-Eval] Step failed for (eval={adaptive_eval_model_name}, "
                        f"gen={generator_name}, re-eval={re_eval_model}). Error: {exc}"
                    )
                    passed_judge, adaptive_incorrect_cnt, re_eval_incorrect_cnt = (0, 0, 0)

                # Log everything about this triple, including the novelty results file.
                write_experiment_log(
                    experiment_csv=self.experiment_csv,
                    eval_model_name=adaptive_eval_model_name,
                    generator_model_name=generator_name,
                    re_eval_model_name=re_eval_model,
                    initial_log_path=initial_log_path,
                    adaptive_log_path=adaptive_log.location,
                    re_eval_log_path=re_eval_log_path,
                    use_cot=self.use_cot,
                    similarity_threshold=self.similarity_threshold,
                    score_threshold=self.score_threshold,
                    use_embeddings=self.use_embeddings,
                    filter_incorrect=self.re_eval_filter_incorrect,
                    adaptive_accuracy=adaptive_accuracy,
                    adaptive_accuracy_judged=adaptive_accuracy_judged,
                    adaptive_scorer_name=adaptive_scorer,
                    re_eval_accuracy=re_eval_acc,
                    re_eval_accuracy_judged=re_eval_acc_judged,
                    re_eval_scorer_name=re_eval_scorer,
                    judge_model_name=self.judge_model_name,
                    n_datapoints=self.n_datapoints,
                    max_attempts=self.max_attempts,
                    passed_judge_count=passed_judge,
                    adaptive_incorrect_count=adaptive_incorrect_cnt,
                    re_eval_incorrect_count=re_eval_incorrect_cnt,
                    novelty_results_file=novelty_file,
                )

    def run_all(
        self,
        adaptive_eval_models: List[str],
        generator_models: List[str],
        re_eval_models: List[str],
    ) -> None:
        """
        1) Runs or loads initial TQA logs for each model in adaptive_eval_models.
        2) For each of those, runs adaptive expansions with the generator_models.
        3) Then re-evaluates the newly generated questions with each model in re_eval_models.
        """
        logs_by_model = self.run_initial_experiments(adaptive_eval_models)

        for eval_model_name, init_log in logs_by_model.items():
            self.run_adaptive_and_transfer_eval(
                initial_log_path=init_log.location,
                adaptive_eval_model_name=eval_model_name,
                generator_models=generator_models,
                re_eval_models=re_eval_models,
            )


@click.command()
@click.option(
    "--experiment-csv",
    default="results/experiment_results.csv",
    help="Path to the CSV file where experiment results will be stored.",
)
@click.option(
    "--cache-csv",
    default="results/experiment_cache.csv",
    help="Path to the CSV file where cached logs will be stored.",
)
@click.option(
    "--adaptive-eval-models",
    default=[
            # "openai/gpt-4o",
            # "openai/gpt-4o-mini",
            # "together/deepseek-ai/DeepSeek-V3",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ],
    multiple=True,
    help="Models to run initial TQA and to evaluate adaptively in the new questions step.",
)
@click.option(
    "--n-datapoints",
    default=40,
    type=int,
    help="Number of datapoints to use for initial TQA.",
)
@click.option(
    "--generator-models",
    default=["openai/gpt-4o"],
    multiple=True,
    help="Models to generate new questions during the adaptive step.",
)
@click.option(
    "--re-eval-models",
    default=[
            # "openai/gpt-4o",
            "openai/gpt-4o-mini",
            # "openai/o1-mini",
            # "together/deepseek-ai/DeepSeek-V3",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # "anthropic/claude-3-5-sonnet-latest",
        ],
    multiple=True,
    help="Models to re-evaluate the newly generated questions (transfer).",
)
@click.option("--use-cot", is_flag=True, help="Use chain-of-thought for solver prompts.")
@click.option(
    "--similarity-threshold",
    default=0.3,
    type=float,
    help="Question similarity threshold for new question checking.",
)
@click.option(
    "--score-threshold",
    default=4,
    type=int,
    help="Score threshold for negative/positive classification",
)
@click.option(
    "--use-embeddings",
    is_flag=True,
    help="If True, use embeddings to compare question similarity",
)
@click.option("--n-pos", default=12, type=int, help="Number of positive samples to generate")
@click.option("--n-neg", default=2, type=int, help="Number of negative samples to generate")
@click.option("--max-attempts", default=10, type=int, help="Maximum number of attempts to generate a question")
@click.option(
    "--re-eval-filter-incorrect",
    is_flag=True,
    help="If set, re-evaluation only uses previously incorrectly answered questions.",
)
def main(
    experiment_csv: str,
    cache_csv: str,
    n_datapoints: int,
    adaptive_eval_models: List[str],
    generator_models: List[str],
    re_eval_models: List[str],
    use_cot: bool,
    similarity_threshold: float,
    score_threshold: int,
    use_embeddings: bool,
    n_pos: int,
    n_neg: int,
    max_attempts: int,
    re_eval_filter_incorrect: bool,
) -> None:
    """
    1) Runs initial TruthfulQA for each model in --adaptive-eval-models.
    2) Then runs an adaptive expansion for each (adaptive-eval-model, generator-model) pair.
    3) Finally, re-evaluates the newly generated questions with each model in --re-eval-models
       to test knowledge transfer or other cross-model performance differences.

    We also parse:
      - accuracy / accuracy_judged
      - which scorer provided it
      - how many samples passed judge filter
      - how many were incorrect in adaptive pass
      - how many were incorrect in re-evaluation
    and store these in the experiment CSV.
    """
    runner = TruthfulQAExperimentRunner(
        use_cot=use_cot,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
        use_embeddings=use_embeddings,
        n_pos=n_pos,
        n_neg=n_neg,
        re_eval_filter_incorrect=re_eval_filter_incorrect,
        n_datapoints=n_datapoints,
        max_attempts=max_attempts,
        experiment_csv=experiment_csv,
        cache_csv=cache_csv,
    )
    runner.run_all(
        adaptive_eval_models=list(adaptive_eval_models),
        generator_models=list(generator_models),
        re_eval_models=list(re_eval_models),
    )


if __name__ == "__main__":
    main() 
