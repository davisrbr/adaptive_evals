import os
from inspect_ai import Epochs, eval
from inspect_ai.log import read_eval_log
from tasks.task_adaptive_legal import legalbench_initial, adaptive_legal, legalbench_initial_aggregated

class LegalExperimentRunner:
    def __init__(self):
        self.model_list_generator = [
            "openai/gpt-4o",
            # "openai/gpt-4o-mini",
            # "anthropic/claude-3-5-sonnet-20240620",
            # "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            # "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        ]
        self.model_list_eval = [
            # "openai/gpt-4o",
            "openai/gpt-4o-mini",
            # "anthropic/claude-3-5-sonnet-20240620",
        ]
        self.task_names = [
            'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
            'maud_financial_point_of_view_is_the_sole_consideration', 
            'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
            'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
        ]

    def run_adaptive_experiment(
        self,
        initial_log_path: str,
        task_name: str = "maud_specific_performance",
        positive_samples_list: list[int] = [1],
        negative_samples_list: list[int] = [4, 8, 16, 32, 64],
        log_dir: str = "logs/",
    ):
        """Run adaptive experiments with different sample configurations."""
        for positive_samples in positive_samples_list:
            for negative_samples in negative_samples_list:
                for generator_model in self.model_list_generator:
                    task = adaptive_legal(
                        initial_log_path=initial_log_path,
                        task_name=task_name,
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name="openai/gpt-4o",
                        use_cot_generator=True,
                        use_cot_evaluator=False,
                        randomize_sampling=False,
                        judge_model_name="openai/o1-preview",
                    )
                    eval(
                        task,
                        epochs=Epochs(30, "mean"),
                        max_connections=1000,
                        log_dir=log_dir,
                        model="openai/gpt-4o",
                        temperature=0,
                    )

    def run_initial_experiments(self):
        """Run initial experiments for each task and evaluation model."""
        for task_name in self.task_names:
            for eval_model in self.model_list_eval:
                log_dir = f"legalbench_logs/initial_legal_{task_name}_{eval_model.replace('/', '_')}"
                
                # Check for existing results
                if os.path.exists(log_dir):
                    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
                else:
                    json_files = []
                    os.makedirs(log_dir)

                if json_files:
                    try:
                        initial_log_path = os.path.join(
                            log_dir,
                            max(json_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
                        )
                        task_log = read_eval_log(initial_log_path)
                        print(f"Skipping initial legal task for {eval_model} (exists in {initial_log_path})")
                        continue
                    except Exception as e:
                        print(f"Error reading existing log: {e}")

                # Run initial evaluation if no valid results exist
                task = legalbench_initial(task_name=task_name)
                task_log = eval(
                    task,
                    epochs=Epochs(1, "max"),
                    max_connections=10000,
                    log_dir=log_dir,
                    model=eval_model,
                    log_level="error"
                )[0]

                if task_log.status != "success":
                    print(f"Task {task_name} with {eval_model} failed.")
                    continue

                print(f"Task {task_name} with {eval_model} completed successfully.")

def main():
    runner = LegalExperimentRunner()
    
    # Example usage for adaptive experiments
    initial_log_path = "logs/2024-12-06T14-38-07-05-00_legalbench-initial-aggregated_CP3Juo4BSyFjLMzfJ8VYyU.json"
    runner.run_adaptive_experiment(
        initial_log_path=initial_log_path,
        positive_samples_list=[1],
        negative_samples_list=[4, 8, 16, 32, 64]
    )

    # Uncomment to run initial experiments
    # runner.run_initial_experiments()

if __name__ == "__main__":
    main() 