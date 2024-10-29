import os
from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, solver
from typing import Any, Literal
from eval_dump_inspect import labels_to_positions
from solver_adaptive_truthfulqa import adaptive_truthfulqa_scorer, adaptive_truthfulqa_solver

@task
def truthfulqa_initial(target: Literal["mc1", "mc2"] = "mc1") -> Task:
    """
    TruthfulQA: Measuring How Models Mimic Human Falsehoods

    Stephanie Lin, Jacob Hilton, Owain Evans
    https://arxiv.org/abs/2109.07958v2

    # Eval truthful QA with reference questions (mc1 - single true answer)
    inspect eval truthfulqa.py

    # Eval against the control questions (mc2 - multi-true answer)
    inspect eval truthfulqa.py -T target=mc2
    Inspect Task implementation for the TruthfulQA benchmark

    Args:
        target (Literal["mc1", "mc2"]): Whether to use the mc1 or mc2 targets
    """

    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
            metadata={"embedding": record["embedding"]},
        )

    dataset = hf_dataset(
        path="davisrbr/truthfulqa-embeddings",
        name="default",
        sample_fields=record_to_sample,
        split="validation",
        auto_id=True,
        shuffle=True,
    )
    if target == "mc1":
        multiple_correct = False
    else:
        multiple_correct = True

    return Task(
        dataset=dataset,
        solver=[multiple_choice(multiple_correct=multiple_correct, shuffle=True)],
        scorer=choice(),
    )
@task
def adaptive_truthfulqa(
    initial_log_path: str,
    n_positive_samples: int = 8,
    n_negative_samples: int = 8,
    generator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    target: Literal["mc1", "mc2"] = "mc1",
    use_cot: bool = False,
) -> Task:
    """
    Adaptive TruthfulQA task that generates new questions based on model errors and evaluates the model on them.
    """

    return Task(
        dataset=MemoryDataset(name="adaptive_truthfulqa", samples=[]),  # We'll populate this in the solver
        solver=[
            adaptive_truthfulqa_solver(
                initial_log_path=initial_log_path,
                n_positive_samples=n_positive_samples,
                n_negative_samples=n_negative_samples,
                generator_model_name=generator_model_name,
                eval_model_name=eval_model_name,
                target=target,
                use_cot=use_cot,
            ),
        ],
        scorer=adaptive_truthfulqa_scorer(),
    )


if __name__ == "__main__":
    model_list_generator = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o-mini", "together/mistralai/Mixtral-8x22B-Instruct-v0.1"]
    model_list_eval = ["openai/gpt-4o", "openai/gpt-4o-mini", "together/mistralai/Mixtral-8x22B-Instruct-v0.1"][1:]

    for eval_model in model_list_eval:
        # first, run the initial truthfulqa task
        task = truthfulqa_initial(target="mc1")
        log_dir = f"new_logs/initial_truthfulqa_log_{eval_model.replace('/', '_')}"
        if os.path.exists(log_dir):
            # check if any json exists in log_dir
            json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        else:
            json_files = []

        if json_files:
            try:
                initial_log_path = max(
                    json_files,
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x))
                )
                print(f"Skipping initial truthfulqa task for {eval_model} because it already exists, in {log_dir}, called {initial_log_path}")
            except Exception as e:
                print(f"An error occurred while retrieving the latest JSON file: {e}")
                # Handle the error as needed, possibly continue or exit
        else:
            print(f"No JSON files found in {log_dir}. Proceeding with the initial truthfulqa task.")
            eval(task, epochs=Epochs(3, "max"), max_connections=10000, log_dir=log_dir, model=eval_model)[0]
            initial_log_path = lo

        for generator_model in model_list_generator:
            # then, run the adaptive truthfulqa task
            for positive_samples in [2, 4, 8]:
                for negative_samples in [4, 8, 16]:
                    log_dir = f"new_logs/initial_adaptive_truthfulqa_log_{eval_model.replace('/', '_')}"
                    # get latest json in log_dir
                    task = adaptive_truthfulqa(
                        initial_log_path=f"new_logs/initial_truthfulqa_log_{eval_model.replace('/', '_')}/{initial_log_path}",
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=False,
                    )
                    eval(task, epochs=Epochs(30, "mean"), max_connections=10000, log_dir=log_dir, model=eval_model, temperature=0)[0] 

                    task = adaptive_truthfulqa(
                        initial_log_path=f"new_logs/initial_truthfulqa_log_{eval_model.replace('/', '_')}/{initial_log_path}",
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=True,
                    )
                    eval(task, epochs=Epochs(30, "mean"), max_connections=10000, log_dir=log_dir, model=eval_model, temperature=0)[0] 

