from inspect_ai import Task, task, Epochs
from solvers.solvers_cyberbullying import cyberbullying_basic_solver
from data.data_cyberbullying import cyberbullying_personas
from scorers.scorers_inspect import final_scorer
from inspect_ai.scorer import mean, stderr


@task
def cyberbullying_task(
    target_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    epochs: int = 1, 
    use_strongreject_scorer: bool = True,
    heirarchal_scorer: bool = False,
    judge_calibration: bool = False,
    dataset: list = cyberbullying_personas,
):
    """
    PAIR task within Inspect
    """
    return Task(
        dataset=dataset,
        plan=[
            cyberbullying_basic_solver(
                target_model_name=target_model_name,
                judge_model_name=judge_model_name,
                use_strongreject_scorer=use_strongreject_scorer,
                heirarchal_scorer=heirarchal_scorer,
                judge_calibration=judge_calibration,
            ),
        ],
        scorer=final_scorer(),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs, reducer="max")
    )