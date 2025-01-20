'''
Example usage:
python experiment_runner_legal.py --models-for-transfer openai/gpt-4o --models-for-transfer openai/gpt-4o-mini
'''
import os
import csv
from typing import Dict, List, Optional
import click
from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog, read_eval_log
from tasks.task_adaptive_legal import adaptive_legal, legalbench_initial_aggregated
import logging

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
        use_cot_in_context_attacker: bool = False
    ):
        self.use_example = use_example
        self.use_cot_target = use_cot_target
        self.use_cot_in_context_attacker = use_cot_in_context_attacker
        
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

class TransferLegalExperimentRunner:
    def __init__(self, cache_csv: Optional[str] = None, **kwargs):
        self.config = ExperimentConfig(**kwargs)
        self.cache_csv = cache_csv or self.config.get_cache_path()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
        os.makedirs(self.config.get_initial_log_dir(), exist_ok=True)
        os.makedirs(self.config.get_adaptive_log_dir(), exist_ok=True)
        
        # Models for the initial pass
        self.initial_eval_models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "together/deepseek-ai/DeepSeek-V3",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "anthropic/claude-3-5-sonnet-latest",
        ]
        self.generator_models = [
            # "openai/gpt-4o-mini",
            # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/gpt-4o",
            # "anthropic/claude-3-5-sonnet-latest",
            # "openai/o1-mini",
        ]
        # Models on which we’ll perform the adaptive evaluation
        self.adaptive_evaluated_models = [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "together/deepseek-ai/DeepSeek-V3",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
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

    def run_initial_experiments(
        self,
        log_dir: Optional[str] = None,
    ) -> Dict[str, EvalLog]:
        """
        1. Reads CSV cache to see if any models have already been run.
        2. Runs initial experiments only on models not in cache.
        3. Updates the cache with newly completed log paths.
        Returns dict of {model_name: EvalLog}.
        """
        logs_by_model: Dict[str, EvalLog] = {}
        cache_data = read_eval_cache(self.cache_csv)
        log_dir = log_dir or self.config.get_initial_log_dir()

        for model_name in self.initial_eval_models:
            if model_name in cache_data:
                # Already evaluated, just re-load the log
                cached_log_path = cache_data[model_name]
                task_log = read_eval_log(cached_log_path)
                if task_log.status == "success":
                    logs_by_model[model_name] = task_log
                else:
                    print(f"Warning: cached log for {model_name} had status '{task_log.status}' so re-running.")
                continue

            # Not in cache, run the initial experiment
            task = legalbench_initial_aggregated(
                task_names=self.task_names,
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
                print(f"No logs returned for {model_name}. Skipping.")
                continue

            task_log = result_logs[0]
            if task_log.status != "success":
                print(f"Initial experiment tasks {self.task_names} with {model_name} failed.")
                continue

            # Cache the new log path
            write_eval_cache(self.cache_csv, model_name, task_log.location)
            logs_by_model[model_name] = task_log
            print(f"Initial experiment tasks {self.task_names} with {model_name} completed successfully.")

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
        task_name: str = "maud_specific_performance",
        positive_samples_list: List[int] = [1],
        negative_samples_list: List[int] = [8],
        log_dir: Optional[str] = None,
        cot_in_context: bool = False,
        use_cot: bool = True,
    ):
        """
        Uses a single (initial_log_path) from the original model, 
        then runs adaptive experiments with each generator model 
        on each of the adaptive_eval_models.

        NOTE: USE_COT IS FOR THE EVALUATOR MODEL, NOT THE GENERATOR MODEL.
        """
        log_dir = log_dir or self.config.get_adaptive_log_dir()
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
                        eval(
                            task,
                            epochs=Epochs(100, "mean"),
                            max_connections=1000,
                            log_dir=log_dir,
                            model=eval_model,
                            temperature=0,
                            log_level="critical",
                        )


@click.command()
@click.option("--models-for-transfer", default=["openai/gpt-4o"], multiple=True, help="Models' logs to use for adaptive question generation.")
def main(models_for_transfer: List[str]):
    # 1. Run or load the initial experiments for all "initial_eval_models."
    for cot_in_context in [True, False]:
        for use_example in [True, False]:
            for use_cot_target in [True, False]:
                runner = TransferLegalExperimentRunner(cache_csv=None, use_example=use_example, use_cot_target=use_cot_target, use_cot_in_context_attacker=cot_in_context)
                logs_by_model = runner.run_initial_experiments()

                # 2. Run adaptive experiments for each specified model
                for model_for_transfer in models_for_transfer:
                    if model_for_transfer not in logs_by_model:
                        print(f"Warning: No log found for {model_for_transfer} in logs_by_model. Available: {list(logs_by_model.keys())}")
                        continue

                    chosen_log = logs_by_model[model_for_transfer]
                    print(f"Running adaptive experiment with {model_for_transfer}")
                    runner.run_adaptive_experiment(
                        initial_log_path=chosen_log.location,
                        original_eval_model_name=model_for_transfer,
                        positive_samples_list=[1],
                        negative_samples_list=[8],
                        cot_in_context=cot_in_context,
                        use_cot=True,
                    )


if __name__ == "__main__":
    main() 