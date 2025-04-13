from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import MemoryDataset, json_dataset
from solvers.solvers_cyberbullying import cyberbullying_adaptive_solver, cyberbullying_basic_solver, cyberbullying_pair_solver
from data.data_cyberbullying import process_sample_cyberbullying
from scorers.scorers_inspect import final_scorer
from inspect_ai.scorer import mean, stderr
from typing import Union


@task
def cyberbullying_task(
    target_model_name: str = "together/mistralai/Mixtral-8x22B-Instruct-v0.1", 
    judge_model_name: str = "openai/gpt-4o-mini", 
    epochs: int = 1, 
    use_strongreject_scorer: bool = True,
    judge_calibration: bool = False,
    num_samples: Union[int, bool] = 25,
):
    """
    Task designed to get a baseline for cyberbullying / calibrate the judge.
    
    Args:
        target_model_name: Model to be evaluated
        judge_model_name: Model used to judge responses
        epochs: Number of evaluation runs
        use_strongreject_scorer: Whether to use strong rejection scoring
        judge_calibration: Whether to calibrate the judge
        num_samples: Number of samples to evaluate, or False for all
    """
    
    dataset_paths = ["../data/cyberbullying_personas.json", "./data/cyberbullying_personas.json"]
    
    for path in dataset_paths:
        try:
            dataset = json_dataset(
                json_file=path,
                sample_fields=process_sample_cyberbullying,
                shuffle=True,
                seed=42,
                limit=num_samples,
            )
            break  # Successfully loaded dataset, exit loop
        except FileNotFoundError:
            continue  # Try next path
    else:  # This executes if no break occurred in the loop
        print("Error: Could not find cyberbullying_personas.json in any of the expected locations")
        return None    
    
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
    use_cot_in_context: bool = True,
    randomize_sampling: bool = False,
):
    """
    PAIR task within Inspect
    """
    dataset_paths = ["../data/cyberbullying_personas.json", "./data/cyberbullying_personas.json"]
    
    for path in dataset_paths:
        try:
            dataset = json_dataset(
                json_file=path,
                sample_fields=process_sample_cyberbullying,
                shuffle=True,
                seed=42,
                limit=num_samples,
            )
            break 
        except FileNotFoundError:
            continue
    else:
        print("Error: Could not find cyberbullying_personas.json in any of the expected locations")
        return None

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
                use_cot_in_context=use_cot_in_context,
                randomize_sampling=randomize_sampling,
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
    hierarchical_scorer: bool = False,
    judge_calibration: bool = False,
    use_vanilla_prompt: bool = True,
    percentiles_and_samples: list = [(1.0, 2), (0.5, 2)],  # Example: top 100% get 2 samples, top 50% get 2
    use_embeddings: bool = False,
    initial_log_path: str = "../logs/2024-12-31T12-11-26-05-00_cyberbullying-jailbreak-task_VJoc6WaBg3QG4W2ySinAS4.json",
    use_cot_in_context: bool = True,
    num_samples: int | bool = 25,
    n_streams: int = 1,
):
    """
    Adaptive cyberbullying evaluation task that uses previous successful attacks to inform new ones.
    
    Args:
        target_model_name: Model to be evaluated
        judge_model_name: Model used to judge responses
        attack_model_name: Model used to generate attack prompts
        max_iterations: Maximum number of back-and-forth iterations
        n_last_messages: Number of messages to keep in the context window
        n_positive_samples: Number of positive samples to retrieve
        n_negative_samples: Number of negative samples to retrieve
        randomize_sampling: Whether to randomize the sampling of positive and negative samples
        use_strongreject_scorer: Whether to use the strongreject scorer
        hierarchical_scorer: Whether to use a heirarchal PAIR scorer
        judge_calibration: Whether to use the neutral generation prompt for judge calibration
        use_vanilla_prompt: Whether to use the vanilla (non-jailbreak) prompt
        percentiles_and_samples: List of tuples specifying the percentiles and number of samples to retrieve, from embedding space similarity of logs to bullying targets
        use_embeddings: Whether to use embeddings for adaptive retrieval
        initial_log_path: Path to the initial log file
        num_samples: Number of samples to evaluate, or False for all
        n_streams: Number of streams to evaluate
    """
    dataset_paths = ["../data/cyberbullying_personas.json", "./data/cyberbullying_personas.json"]
    
    for path in dataset_paths:
        try:
            dataset = json_dataset(
                json_file=path,
                sample_fields=process_sample_cyberbullying,
                shuffle=True,
                seed=42,
                limit=num_samples,
            )
            break
        except FileNotFoundError:
            continue
    else:
        print("Error: Could not find cyberbullying_personas.json in any of the expected locations")
        return None
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
            hierarchical_scorer=hierarchical_scorer,
            judge_calibration=judge_calibration,
            use_vanilla_prompt=use_vanilla_prompt,
            percentiles_and_samples=percentiles_and_samples,
            use_embeddings=use_embeddings,
            initial_log_path=initial_log_path,
            use_cot_in_context=use_cot_in_context,
        )],
        scorer=final_scorer(cutoff=0),
        metrics=[mean(), stderr()],
        epochs=Epochs(epochs=n_streams, reducer="max")
    )
