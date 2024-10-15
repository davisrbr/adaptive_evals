from inspect_ai import Task, task, Epochs, Task, eval
from inspect_ai.solver._task_state import Sample
from inspect_ai.scorer import at_least
from data_inspect import jb_behaviors_dataset, jb_behaviors_dataset_embeddings
from solvers_inspect import pair_solver, pair_solver_adaptive, decomposition_attack_solver
from scorers_inspect import final_scorer

import numpy as np

# randomly sample from the dataset
np.random.seed(42)

jb_reducer = at_least(1, 10)
jb_behaviors_dataset = [
    Sample(input=item.input, target=item.target, metadata=item.metadata, sandbox=item.sandbox if hasattr(item, 'sandbox') else None)
    for item in jb_behaviors_dataset
][:1]

@task
def task_decomposition_task(
    dataset: list = jb_behaviors_dataset,
    epochs: int = 10
):
    """
    Task Decomposition task within Inspect
    """
    return Task(
        dataset=dataset,
        plan=[
            decomposition_attack_solver(
            )
        ],
        scorer=final_scorer(),
        epochs=Epochs(epochs, jb_reducer)
    )

if __name__ == "__main__":
    for dataset in jb_behaviors_dataset:
        task = task_decomposition_task(dataset=[dataset])
        eval(task, epochs=Epochs(2, "max"), max_connections=10000, log_dir="decomposition", model="together/mistralai/Mixtral-8x22B-Instruct-v0.1")[0]