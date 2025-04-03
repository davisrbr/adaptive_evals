import os
from inspect_ai import Epochs, eval
from inspect_ai.log import read_eval_log
from tasks.task_adaptive_legal import legalbench_initial, adaptive_legal, legalbench_initial_aggregated
from experiment_runner_legal import LegalExperimentRunner

def main():
    runner = LegalExperimentRunner()
    
    # Example usage for adaptive experiments
    initial_log_path = "/Users/davisbrown/adaptive_evals/logs/2025-01-09T14-16-12-05-00_legalbench-initial-aggregated_HhCUKSpNNWrGJ8MRfhJpGj.eval"
    runner.run_adaptive_experiment(
        initial_log_path=initial_log_path,
        positive_samples_list=[1],
        negative_samples_list=[8],
        log_dir="logs/adaptive_legal_test"
    )

if __name__ == "__main__":
    main() 