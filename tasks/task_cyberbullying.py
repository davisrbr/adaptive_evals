from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import json_dataset
from solvers.solvers_cyberbullying import cyberbullying_basic_solver
from data.data_cyberbullying import process_sample_cyberbullying
from scorers.scorers_inspect import final_scorer
from inspect_ai.scorer import mean, stderr


@task
def cyberbullying_task(
    target_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    epochs: int = 1, 
    use_strongreject_scorer: bool = True,
    judge_calibration: bool = False,
    num_samples: int | bool = 25,
):
    """
    PAIR task within Inspect
    """
    
    dataset = json_dataset(
        json_file="/Users/davisbrown/adaptive_evals/data/cyberbullying_personas.json",
        sample_fields=process_sample_cyberbullying,
        shuffle=True,
        seed=42,
        limit=num_samples,
    )
    return Task(
        dataset=dataset,
        plan=[
            cyberbullying_basic_solver(
                target_model_name=target_model_name,
                judge_model_name=judge_model_name,
                use_strongreject_scorer=use_strongreject_scorer,
                judge_calibration=judge_calibration,
            ),
        ],
        scorer=final_scorer(cutoff=0),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs, reducer="max")
    )