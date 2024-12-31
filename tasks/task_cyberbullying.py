from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import json_dataset
from solvers.solvers_cyberbullying import cyberbullying_basic_solver, cyberbullying_pair_solver
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

@task
def cyberbullying_jailbreak_task(
    target_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    n_streams: int = 1, 
    use_strongreject_scorer: bool = False,
    judge_calibration: bool = False,
    num_samples: int | bool = 25,
    max_iterations: int = 3,
    n_last_messages: int = 3,
    cutoff: int = 0,
    heirarchal_scorer: bool = False,
    use_vanilla_prompt: bool = True,
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
            cyberbullying_pair_solver(
                target_model_name=target_model_name,
                attack_model_name=attack_model_name,
                judge_model_name=judge_model_name,
                use_strongreject_scorer=use_strongreject_scorer,
                judge_calibration=judge_calibration,
                n_last_messages=n_last_messages,
                cutoff=cutoff,
                heirarchal_scorer=heirarchal_scorer,
                max_iterations=max_iterations,
                use_vanilla_prompt=use_vanilla_prompt,
            ),
        ],
        scorer=final_scorer(cutoff=0),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs=n_streams, reducer="max")
    )