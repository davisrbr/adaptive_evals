from inspect_ai import Task, task, Epochs, eval
from data.data_inspect import jb_behaviors_dataset
from solvers.solvers_inspect import pair_solver
from scorers.scorers_inspect import final_scorer
from inspect_ai.scorer import mean, stderr

@task
def pair_task(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    max_iterations: int = 6,
    n_last_messages: int = 5, 
    epochs: int = 25, 
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    dataset: list = jb_behaviors_dataset[:2],
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
        scorer=final_scorer(),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs, reducer="max")
    )
