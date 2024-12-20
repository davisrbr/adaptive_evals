import os
import pandas as pd
import re
from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from typing import Any, Literal, Optional
from solvers.solver_adaptive_legal import adaptive_legal_scorer, adaptive_legal_solver, adaptive_legal_judge_solver, adaptive_legal_judge_scorer
from datasets import load_dataset
import sys
sys.path.append("..")
from legalbench.utils import generate_prompts
from legalbench.tasks import TASKS
from solvers.solver_adaptive_legal import rewording_legal_solver

@task
def legalbench_initial(task_name: str = "maud_accuracy_of_target_general_rw_bringdown_timing_answer") -> Task:
    """
    Initial evaluation task for the LegalBench dataset.

    Args:
        task_name (str): The name of the LegalBench task to evaluate.
    """

    # Load the prompt template for the specified task
    prompt_template_path = f"../legalbench/tasks/{task_name}/base_prompt.txt"
    with open(prompt_template_path) as in_file:
        prompt_template = in_file.read()

    # Define a function to convert dataset records to Sample objects
    def record_to_sample(record: dict[str, Any]) -> Sample:
        # Convert the single record into a pandas DataFrame
        df = pd.DataFrame([record])

        # Generate the prompt using the template and record data
        prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
        prompt = prompts[0]

        # Extract the options from the prompt
        option_pattern = r"Option ([A-Z]): (.*)"
        options_matches = re.findall(option_pattern, prompt)

        # Create choices based on options extracted
        choices = [match[1] for match in options_matches]

        # The target is the letter of the correct answer
        target = record["answer"].strip()

        return Sample(
            input=prompt,
            choices=choices,
            target=target,
            metadata={"task_name": task_name}
        )

    # Load the LegalBench dataset
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=record_to_sample,
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=True,
        shuffle=True,
    )

    return Task(
        dataset=dataset,
        solver=[
            multiple_choice(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

@task
def legalbench_initial_aggregated(task_names: list[str] = [
    'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    'maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer',
    'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
], debug: bool = False) -> Task:
    """
    Initial evaluation task for multiple LegalBench datasets aggregated into a single dataset.

    Args:
        task_names (list[str]): The list of LegalBench task names to evaluate.
    """

    # Initialize an empty list to collect samples from all tasks
    all_samples = []

    for task_name in task_names:
        # Load the prompt template for the specified task
        prompt_template_path = f"../legalbench/tasks/{task_name}/base_prompt.txt"
        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        # Define a function to convert dataset records to Sample objects
        def record_to_sample(record: dict[str, Any]) -> Sample:
            # Convert the single record into a pandas DataFrame
            df = pd.DataFrame([record])

            # Generate the prompt using the template and record data
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            # Extract the options from the prompt
            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)

            # Create choices based on options extracted
            choices = [match[1] for match in options_matches]

            # The target is the letter of the correct answer
            target = record["answer"].strip()

            return Sample(
                input=prompt,
                choices=choices,
                target=target,
                metadata={"task_name": task_name}
            )

        # Load the LegalBench dataset for this task
        dataset = hf_dataset(
            path="nguha/legalbench",
            name=task_name,
            sample_fields=record_to_sample,
            split="test",  # Use "test" or "validation" as appropriate
            auto_id=True,
            shuffle=True if not debug else False,
        )

        # Collect samples from this dataset
        samples = dataset.samples
        all_samples.extend(samples)

    # Create a combined dataset
    combined_dataset = MemoryDataset(name="legalbench_aggregated", samples=all_samples)
    if debug:
        combined_dataset = combined_dataset[:50]

    return Task(
        dataset=combined_dataset,
        solver=[
            multiple_choice(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

@task
def adaptive_legal(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 1,
    n_negative_samples: int = 2,
    generator_model_name: str = "openai/gpt-4o",
    eval_model_name: str = "openai/gpt-4o",
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    randomize_sampling: bool = False,
    judge_model_name: Optional[str] = None,
) -> Task:
    """
    Adaptive evaluation task for the LegalBench dataset.

    Args:
        initial_log_path (str): Path to the initial evaluation log.
        task_name (str): The name of the LegalBench task to evaluate.
        n_positive_samples (int): Number of correctly answered samples to use for adaptation.
        n_negative_samples (int): Number of incorrectly answered samples to use for adaptation.
        generator_model_name (str): Name of the model used to generate new questions.
        eval_model_name (str): Name of the model used to evaluate the new questions.
        use_cot (bool): Whether to use chain-of-thought prompting.
        judge_model_name (str): Name of the model used to score generated questions.
    """

    return Task(
        dataset=MemoryDataset(name="adaptive_legal", samples=[]),  # We'll populate this in the solver
        solver=[
            adaptive_legal_solver(
                initial_log_path=initial_log_path,
                task_name=task_name,
                n_positive_samples=n_positive_samples,
                n_negative_samples=n_negative_samples,
                generator_model_name=generator_model_name,
                eval_model_name=eval_model_name,
                use_cot_generator=use_cot_generator,
                use_cot_evaluator=use_cot_evaluator,
                randomize_sampling=randomize_sampling,
            ),
            adaptive_legal_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            ),
        ],
        scorer=[
            adaptive_legal_scorer(),
            adaptive_legal_judge_scorer(),
        ],
    )

@task
def legalbench_reworded(
    task_name: str = "maud_accuracy_of_target_general_rw_bringdown_timing_answer",
    model_name: str = "openai/gpt-4o-mini",
    rewording_model_name: str = "openai/gpt-4o-mini",
    cot: bool = False,
    debug: bool = False
) -> Task:
    """
    Evaluation task for the LegalBench dataset with reworded questions.

    Args:
        task_name (str): The name of the LegalBench task to evaluate.
        model_name (str): The name of the model used to answer the question.
        rewording_model_name (str): The name of the model used to reword the question.
        debug (bool): If True, use only a 5 question subset for debugging.
    """

    # Load the prompt template for the specified task
    prompt_template_path = f"../legalbench/tasks/{task_name}/base_prompt.txt"
    with open(prompt_template_path) as in_file:
        prompt_template = in_file.read()

    # Define a function to convert dataset records to Sample objects
    def record_to_sample(record: dict[str, Any]) -> Sample:
        # Convert the single record into a pandas DataFrame
        df = pd.DataFrame([record])

        # Generate the prompt using the template and record data
        prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
        prompt = prompts[0]

        # Extract the options from the prompt
        option_pattern = r"Option ([A-Z]): (.*)"
        options_matches = re.findall(option_pattern, prompt)

        # Create choices based on options extracted
        choices = [match[1] for match in options_matches]

        # The target is the letter of the correct answer
        target = record["answer"].strip()

        return Sample(
            input=prompt,
            choices=choices,
            target=target,
            metadata={"task_name": task_name}
        )

    # Load the LegalBench dataset
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=record_to_sample,
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=True,
        shuffle=True,
    )

    # If in debug mode, limit to 5 samples
    if debug:
        dataset = dataset[:5]

    return Task(
        dataset=dataset,
        solver=[
            rewording_legal_solver(model_name=model_name, rewording_model_name=rewording_model_name, cot=cot),
            multiple_choice(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

@task
def legalbench_reworded_aggregated(
    task_names: list[str] = [
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
        'maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ],
    model_name: str = "openai/gpt-4o-mini",
    rewording_model_name: str = "openai/gpt-4o-mini",
    cot: bool = False,
    debug: bool = False
) -> Task:
    """
    Aggregated evaluation task for multiple LegalBench datasets with reworded questions.

    Args:
        task_names (list[str]): The list of LegalBench task names to evaluate.
        model_name (str): The name of the model used to answer the question.
        rewording_model_name (str): The name of the model used to reword the question.
        cot (bool): Whether to use chain-of-thought prompting.
        debug (bool): If True, use only a 5 question subset for debugging.
    """
    # Initialize an empty list to collect samples from all tasks
    all_samples = []

    for task_name in task_names:
        # Load the prompt template for the specified task
        prompt_template_path = f"../legalbench/tasks/{task_name}/base_prompt.txt"
        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        # Define a function to convert dataset records to Sample objects
        def record_to_sample(record: dict[str, Any]) -> Sample:
            df = pd.DataFrame([record])
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            choices = [match[1] for match in options_matches]
            target = record["answer"].strip()

            return Sample(
                input=prompt,
                choices=choices,
                target=target,
                metadata={"task_name": task_name}
            )

        # Load the LegalBench dataset for this task
        dataset = hf_dataset(
            path="nguha/legalbench",
            name=task_name,
            sample_fields=record_to_sample,
            split="test",
            auto_id=True,
            shuffle=True if not debug else False,
        )

        # Collect samples from this dataset
        samples = dataset.samples
        all_samples.extend(samples)

    # Create a combined dataset
    combined_dataset = MemoryDataset(name="legalbench_reworded_aggregated", samples=all_samples)

    # If in debug mode, limit to 30 samples
    if debug:
        combined_dataset = combined_dataset[:30]

    return Task(
        dataset=combined_dataset,
        solver=[
            rewording_legal_solver(model_name=model_name, rewording_model_name=rewording_model_name, cot=cot),
            multiple_choice(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

if __name__ == "__main__":
    # List of models to use for generation and evaluation
    model_list_generator = [
        "openai/gpt-4o",
        # "openai/gpt-4o-mini",
        # "anthropic/claude-3-5-sonnet-20240620",
        # "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        # "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ]
    model_list_eval = [
        # "openai/gpt-4o",
        "openai/gpt-4o-mini",
        # "anthropic/claude-3-5-sonnet-20240620",
    ]

    # Specify the LegalBench tasks to evaluate
    task_names = [
        'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
        'maud_financial_point_of_view_is_the_sole_consideration', 
        'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ]
    # task_names = [task for task in TASKS if task.startswith("maud_")]

    # for this run, we are always evaluation gpt-4o, and sweep across generator models and n_positive_samples/n_negative_samples
    # this is for an adaptive evaluation with the aggregated task
    initial_log_path = "logs/2024-12-06T14-38-07-05-00_legalbench-initial-aggregated_CP3Juo4BSyFjLMzfJ8VYyU.json"
    # initial_log_path = "logs/2024-10-30T10-44-42-04-00_legalbench-initial-aggregated_gQXZisrb6539kQGbrUWSMC.json"
    # initial_log_path = "logs/2024-10-30T10-35-29-04-00_legalbench-initial-aggregated_m6nH9qBkc6hrgRkcuRfGqx.json"
    for positive_samples in [1]:
        for negative_samples in [4, 8, 16, 32, 64]:
        # for negative_samples in [32]:
    # for positive_samples in [4]:
    #     for negative_samples in [60]:
    # for positive_samples in [2]:
        # for negative_samples in [10]:
            for generator_model in model_list_generator:
                log_dir = "logs/"
                # task = adaptive_legal(
                #     initial_log_path=initial_log_path,
                #     task_name="maud_specific_performance",
                #     n_positive_samples=positive_samples,
                #     n_negative_samples=negative_samples,
                #     generator_model_name=generator_model,
                #     eval_model_name="openai/gpt-4o",
                #     use_cot_generator=False,
                #     use_cot_evaluator=False,
                # )
                # eval(
                #     task,
                #     epochs=Epochs(30, "mean"),
                #     max_connections=10000,
                #     log_dir=log_dir,
                #     model="openai/gpt-4o-mini",
                #     temperature=0,
                # )
                # task = adaptive_legal(
                #     initial_log_path=initial_log_path,
                #     task_name="maud_specific_performance",
                #     n_positive_samples=positive_samples,
                #     n_negative_samples=negative_samples,
                #     generator_model_name=generator_model,
                #     eval_model_name="openai/gpt-4o",
                #     use_cot_generator=True,
                #     use_cot_evaluator=False,
                # )
                # eval(
                #     task,
                #     epochs=Epochs(30, "mean"),
                #     max_connections=10000,
                #     log_dir=log_dir,
                #     model="openai/gpt-4o-mini",
                #     temperature=0,
                # )
                task = adaptive_legal(
                    initial_log_path=initial_log_path,
                    task_name="maud_specific_performance",
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
    # for task_name in task_names:
    #     for eval_model in model_list_eval:
    #         # Run the initial LegalBench task
    #         task = legalbench_initial(task_name=task_name)
    #         log_dir = f"legalbench_logs/initial_legal_{task_name}_{eval_model.replace('/', '_')}"
    #         if os.path.exists(log_dir):
    #             # check if any json exists in log_dir
    #             json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    #         else:
    #             json_files = []

    #         if json_files:
    #             try:
    #                 initial_log_path = os.path.join(log_dir, max(
    #                     json_files,
    #                     key=lambda x: os.path.getctime(os.path.join(log_dir, x))
    #                 ))  
    #                 task_log = read_eval_log(initial_log_path)
    #                 print(f"Skipping initial legal task for {eval_model} because it already exists, in {log_dir}, called {initial_log_path}")
    #             except Exception as e:
    #                 print(f"An error occurred while retrieving the latest JSON file: {e}")
    #                 # Handle the error as needed, possibly continue or exit
    #         else:
    #             print(f"No JSON files found in {log_dir}. Proceeding with the initial truthfulqa task.")
    #             task_log = eval(task, epochs=Epochs(1, "max"), max_connections=10000, log_dir=log_dir, model=eval_model, log_level="error")[0]
    #             initial_log_path = os.path.join(log_dir, max(
    #                 [f for f in os.listdir(log_dir) if f.endswith('.json')],
    #                 key=lambda x: os.path.getctime(os.path.join(log_dir, x))
    #             ))
    #         # check if the task was successful
    #         if task_log.status == "success":
    #             print(f"Task {task_name} with {eval_model} completed successfully.")

    #             for generator_model in model_list_generator:
    #                 # Run the adaptive LegalBench task
    #                 for positive_samples in [2]:
    #                     for negative_samples in [2, 4, 16]:
    #                         log_dir = f"legalbench_logs/adaptive_legal_{task_name}_{eval_model.replace('/', '_')}"
    #                         if not os.path.exists(log_dir):
    #                             os.makedirs(log_dir)
    #                             task_log = None
    #                         else:
    #                             json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    #                             if json_files:
    #                                 task_logs = [read_eval_log(os.path.join(log_dir, f)) for f in json_files]
    #                             else:
    #                                 print(f"No JSON files found in {log_dir}. Proceeding with the adaptive legal task.")
    #                                 task_log = None
    #                             # Start of Selection
    #                             if not task_log or any((log.status != "success" or log.use_cot_generator) for log in task_logs):
    #                                 task = adaptive_legal(
    #                                     initial_log_path=initial_log_path,
    #                                     task_name=task_name,
    #                                     n_positive_samples=positive_samples,
    #                                     n_negative_samples=negative_samples,
    #                                     generator_model_name=generator_model,
    #                                     eval_model_name=eval_model,
    #                                     use_cot_generator=False,
    #                                     use_cot_evaluator=False,
    #                                 )
    #                                 eval(
    #                                     task,
    #                                     epochs=Epochs(30, "mean"),
    #                                     max_connections=10000,
    #                                     log_dir=log_dir,
    #                                     model=eval_model,
    #                                     temperature=0,
    #                                 )
    #                             elif not task_log or any((log.status != "success" or not log.use_cot_generator) for log in task_logs):
    #                                 task = adaptive_legal(
    #                                     initial_log_path=initial_log_path,
    #                                     task_name=task_name,
    #                                     n_positive_samples=positive_samples,
    #                                     n_negative_samples=negative_samples,
    #                                     generator_model_name=generator_model,
    #                                     eval_model_name=eval_model,
    #                                     use_cot_generator=True,
    #                                     use_cot_evaluator=False,
    #                                 )
    #                                 eval(
    #                                     task,
    #                                     epochs=Epochs(30, "mean"),
    #                                     max_connections=10000,
    #                                     log_dir=log_dir,
    #                                     model=eval_model,
    #                                     temperature=0,
    #                                 )
    #         else:
    #             print(f"Task {task_name} with {eval_model} failed.")