from inspect_ai import Task, task
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
        )

    dataset = hf_dataset(
        path="truthful_qa",
        name="multiple_choice",
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
    answer_model_name: str = "openai/gpt-4o-mini",
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
                answer_model_name=answer_model_name,
                target=target,
                use_cot=use_cot,
            ),
        ],
        scorer=adaptive_truthfulqa_scorer(),
    )

@task
def evaluate_adaptive_truthfulqa(
    adaptive_log_path: str,
    target: Literal["mc1", "mc2"] = "mc1",
) -> Task:
    """
    Task to evaluate the model on the newly generated adaptive dataset.
    """
    # Load the adaptive log to get the generated samples
    eval_log = read_eval_log(adaptive_log_path)
    generated_samples = eval_log.state.store.get('generated_samples', [])

    multiple_correct = target != "mc1"

    return Task(
        dataset=MemoryDataset(name="adaptive_truthfulqa", samples=generated_samples),
        solver=[
            multiple_choice(multiple_correct=multiple_correct, shuffle=True),
        ],
        scorer=choice(),
    )
