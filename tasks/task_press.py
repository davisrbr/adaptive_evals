from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import solver
from typing import Literal

from solvers.solver_press import press_solver, press_scorer


@task
def press_task(
    initial_log_path: str,
    n_iterations: int = 5,
    quiz_length: int = 10,
    evaluator_model_name: str = "openai/gpt-4o-mini",
    target: Literal["mc1", "mc2"] = "mc1",
    dataset_name: str = "truthfulqa",
) -> Task:
    """
    PRESS Method task that generates report cards analyzing model performance.
    
    This is a generalized version that works with any dataset in the repository.
    
    Args:
        initial_log_path: Path to the initial evaluation log
        n_iterations: Number of PRESS iterations (default: 5)
        quiz_length: Number of questions per quiz batch (default: 10)
        evaluator_model_name: Model to use for evaluation
        target: Target type for multiple choice questions
        dataset_name: Name of the dataset being analyzed
    
    PRESS only generates report cards, not questions.
    """
    return Task(
        dataset=MemoryDataset(
            name=f"press_{dataset_name}", 
            samples=[Sample(input="Generate report cards", choices=["A"], target="A")]
        ),
        solver=[
            press_solver(
                initial_log_path=initial_log_path,
                n_iterations=n_iterations,
                quiz_length=quiz_length,
                evaluator_model_name=evaluator_model_name,
                target=target,
                dataset_name=dataset_name,
            ),
        ],
        scorer=press_scorer(),
    )