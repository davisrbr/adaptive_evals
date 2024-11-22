from inspect_ai import Task, task
from inspect_ai import Epochs, Task, eval
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.solver._task_state import Sample
from inspect_ai.scorer import at_least
from data.data_inspect import jb_behaviors_dataset, jb_behaviors_dataset_embeddings
from solvers.solver_inspect import pair_solver, pair_solver_adaptive, decomposition_attack_solver
from scorers.scorer_inspect import final_scorer, jailbreakbench_scorer, secondary_scorer

import numpy as np


# randomly sample from the dataset
np.random.seed(42)

# create initial PAIR run for adaptive examples
# jb_behaviors_dataset = jb_behaviors_dataset[::5]
# jb_behaviors_dataset = jb_behaviors_dataset[:2]

# Filter the dataset to keep 4 out of every 5 examples, and randomly sample 30 examples
# jb_behaviors_dataset = [example for i, example in enumerate(jb_behaviors_dataset) if i % 5 != 0]
# jb_behaviors_dataset = np.random.choice(jb_behaviors_dataset, 30)
# jb_behaviors_dataset = jb_behaviors_dataset[8*7:8*8] # privacy
# attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",


jb_reducer = at_least(1, 1)

@task
def pair_task(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    max_iterations: int = 3,
    n_last_messages: int = 2, 
    epochs: int = 2, 
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    dataset: list = jb_behaviors_dataset,
):
    """
    PAIR task within Inspect
    """
    return Task(
        dataset=dataset,
        plan=[
            pair_solver(
                max_iterations=max_iterations,
                target_model_name=target_model_name,
                judge_model_name=judge_model_name,
                attack_model_name=attack_model_name,
                n_last_messages=n_last_messages,
                use_strongreject_scorer=use_strongreject_scorer,
                heirarchal_scorer=heirarchal_scorer,
            ),
        ],
        # scorer=secondary_scorer(judge_model="openai/gpt-4o"),
        scorer=final_scorer(),
        epochs=Epochs(epochs, jb_reducer)
    )

@task
def pair_task_adaptive(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    judge_model_name: str = "openai/gpt-4o-mini", 
    max_iterations: int = 3,
    n_last_messages: int = 2, 
    percentiles_and_samples: list = [(1.0, 2), (0.5, 2)],  # List of (percentile, n_samples) tuples
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    use_embeddings: bool = True,
    epochs: int = 2, 
    dataset: list = jb_behaviors_dataset_embeddings,
):
    """
    PAIR task within Inspect
    Args:
        percentiles_and_samples: List of tuples [(percentile, n_samples), ...] 
            where percentile is between 0 and 1, and n_samples is the number of examples to sample
            at that percentile
    """
    return Task(
        dataset=dataset,
        plan=[
            pair_solver_adaptive(
                max_iterations=max_iterations,
                target_model_name=target_model_name,
                attack_model_name=attack_model_name,
                judge_model_name=judge_model_name,
                n_last_messages=n_last_messages,
                percentiles_and_samples=percentiles_and_samples,
                heirarchal_scorer=heirarchal_scorer,
                use_strongreject_scorer=use_strongreject_scorer,
                use_embeddings=use_embeddings
            ),
        ],
        scorer=final_scorer(),
        epochs=Epochs(epochs, jb_reducer)
    )

jb_behaviors_dataset = [
    Sample(input=item.input, target=item.target, metadata=item.metadata, sandbox=item.sandbox if hasattr(item, 'sandbox') else None)
    for item in jb_behaviors_dataset
]


if __name__ == "__main__":
    # run both pair and pair adaptive tasks
    # loop over just different percentiles and samples
    for percentiles_and_samples in [[(1.0, 2), (0.5, 2)], [(1.0, 2), (0.8, 2)], [(1.0, 2), (0.2, 2)]]:
    # for percentiles_and_samples in [[(1.0, 4), (0.5, 4)], [(1.0, 4), (0.8, 4)], [(1.0, 4), (0.2, 4)]]:
        task = pair_task_adaptive(epochs=10, percentiles_and_samples=percentiles_and_samples)
        eval(task, epochs=Epochs(1, "max"), max_connections=10000, log_dir=f"logs/pair_task_adaptive_log_{percentiles_and_samples}")[0]  
    task = pair_task(epochs=10)
    eval(task, epochs=Epochs(1, "max"), max_connections=10000, log_dir="logs/pair_task_log" )[0]  