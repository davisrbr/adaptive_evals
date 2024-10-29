import os
import pandas as pd
import re
from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from typing import Any, Literal
from solver_adaptive_legal import adaptive_legal_scorer, adaptive_legal_solver
from datasets import load_dataset
from legalbench.utils import generate_prompts

@task
# def legalbench_initial(task_name: str = "maud_accuracy_of_fundamental_target_rws_bringdown_standard") -> Task:
def legalbench_initial(task_name: str = "maud_accuracy_of_target_general_rw_bringdown_timing_answer") -> Task:
    """
    Initial evaluation task for the LegalBench dataset.

    Args:
        task_name (str): The name of the LegalBench task to evaluate.
    """

    # Load the prompt template for the specified task
    prompt_template_path = f"legalbench/tasks/{task_name}/base_prompt.txt"
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
def adaptive_legal(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 1,
    n_negative_samples: int = 2,
    generator_model_name: str = "openai/gpt-4",
    eval_model_name: str = "openai/gpt-4",
    use_cot: bool = False,
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
                use_cot=use_cot,
            ),
        ],
        scorer=adaptive_legal_scorer(),
    )

if __name__ == "__main__":
    # List of models to use for generation and evaluation
    model_list_generator = [
        "openai/gpt-4",
        "anthropic/claude-v1",
        "openai/gpt-3.5-turbo",
    ]
    model_list_eval = [
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
    ]

    # Specify the LegalBench tasks to evaluate
    task_names = [
        "maud_accuracy_of_fundamental_target_rws_bringdown_standard",
        # Add other task names as needed
    ]

    for task_name in task_names:
        for eval_model in model_list_eval:
            # Run the initial LegalBench task
            task = legalbench_initial(task_name=task_name)
            log_dir = f"logs/initial_legal_{task_name}_{eval_model.replace('/', '_')}"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            eval(task, epochs=Epochs(3, "max"), max_connections=10000, log_dir=log_dir, model=eval_model)

            initial_log_path = log_dir

            for generator_model in model_list_generator:
                # Run the adaptive LegalBench task
                for positive_samples in [2, 4, 8]:
                    for negative_samples in [4, 8, 16]:
                        log_dir = f"logs/adaptive_legal_{task_name}_{eval_model.replace('/', '_')}"
                        if not os.path.exists(log_dir):
                            os.makedirs(log_dir)
                        task = adaptive_legal(
                            initial_log_path=initial_log_path,
                            task_name=task_name,
                            n_positive_samples=positive_samples,
                            n_negative_samples=negative_samples,
                            generator_model_name=generator_model,
                            eval_model_name=eval_model,
                            use_cot=False,
                        )
                        eval(
                            task,
                            epochs=Epochs(30, "mean"),
                            max_connections=10000,
                            log_dir=log_dir,
                            model=eval_model,
                            temperature=0,
                        )
