from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import MemoryDataset, json_dataset
from inspect_ai.log import read_eval_log
from solvers.solvers_cyberbullying import cyberbullying_adaptive_solver, cyberbullying_basic_solver, cyberbullying_pair_solver
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
    Task designed to get a baseline for cyberbullying / calibrate the judge.
    """
    
    dataset = json_dataset(
        json_file="../data/cyberbullying_personas.json",
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
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf",
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
        json_file="../data/cyberbullying_personas.json",
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


@task
def cyberbullying_adaptive_task(
    target_model_name: str = "together/meta-llama/Llama-2-7b-chat-hf",
    judge_model_name: str = "openai/gpt-4o-mini",
    attack_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
    max_iterations: int = 5,
    n_last_messages: int = 3,
    n_positive_samples: int = 5,
    n_negative_samples: int = 5,
    randomize_sampling: bool = False,
    use_strongreject_scorer: bool = False,
    heirarchal_scorer: bool = False,
    judge_calibration: bool = False,
    use_vanilla_prompt: bool = True,
    # For adaptive retrieval:
    percentiles_and_samples: list = [(1.0, 2), (0.5, 2)],  # Example: top 100% get 2 samples, top 50% get 2
    use_embeddings: bool = False,
    # Filter criteria for retrieving only successful outputs from prior runs of the basic solver:
    initial_log_path: str = "../logs/2024-12-31T12-11-26-05-00_cyberbullying-jailbreak-task_VJoc6WaBg3QG4W2ySinAS4.json",
    num_samples: int | bool = 25,
    n_streams: int = 1,
):
    dataset = json_dataset(
        json_file="../data/cyberbullying_personas.json",
        sample_fields=process_sample_cyberbullying,
        shuffle=True,
        seed=42,
        limit=num_samples,
    )
    return Task(
        dataset=dataset,
        plan=[cyberbullying_adaptive_solver(
            max_iterations=max_iterations,
            target_model_name=target_model_name,
            judge_model_name=judge_model_name,
            attack_model_name=attack_model_name,
            n_last_messages=n_last_messages,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            randomize_sampling=randomize_sampling,
            use_strongreject_scorer=use_strongreject_scorer,
            heirarchal_scorer=heirarchal_scorer,
            judge_calibration=judge_calibration,
            use_vanilla_prompt=use_vanilla_prompt,
            percentiles_and_samples=percentiles_and_samples,
            use_embeddings=use_embeddings,
            initial_log_path=initial_log_path,
        )],
        scorer=final_scorer(cutoff=0),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs=n_streams, reducer="max")
    )
