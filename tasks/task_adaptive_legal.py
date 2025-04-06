import os
from inspect_ai.log import read_eval_log
import pandas as pd
import re
from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, chain_of_thought
from typing import Any, Callable, Optional
import sys
import json
import logging

from solvers.adaptive_utils import multiple_choice_save_cot
from solvers.solver_adaptive_legal_refactor import adaptive_legal_solver_refactor
from utils_elicitation.novelty import novelty_filter_judged_only
sys.path.append("..")
sys.path.append("../legalbench")
from legalbench.utils import generate_prompts

from scorers.scorers_rewording import choice_judged, judge_scoring
from solvers.solver_adaptive_legal import adaptive_legal_solver, adaptive_legal_judge_solver, rewording_legal_solver, rewording_legal_judge_solver
from scorers.scorers_legal import (
    adaptive_legal_scorer_judged,
    adaptive_legal_scorer,
    adaptive_legal_judge_scorer,
)


def get_record_to_sample(
    prompt_template: str, 
    task_name: str, 
    debug: bool = False, 
    use_cot: bool = False
) -> Callable[[dict[str, Any]], Sample]:
    '''Returns a function that converts a dataset record into a Sample object.'''
    def record_to_sample(record: dict[str, Any],) -> Sample:
        # Convert the single record into a pandas DataFrame
        df = pd.DataFrame([record])

        # Generate the prompt using the template and record data
        prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
        prompt = prompts[0]

        # Extract the options from the prompt
        # Try to find options with letters first
        option_pattern = r"Option ([A-Z]): (.*)"
        options_matches = re.findall(option_pattern, prompt)
        
        # If no matches found, try numbered options
        if not options_matches:
            option_pattern = r"(\d+): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            # replace all with this format in prompt with letters
            prompt = re.sub(r"\d+", lambda m: chr(ord('A') + int(m.group(0))), prompt)

        # Create choices based on options extracted
        choices = [match[1] for match in options_matches]

        # The target is the letter of the correct answer
        target = record["answer"].strip()
        # Convert numeric answer to letter if needed
        if target.isdigit():
            target = chr(ord('A') + int(target))

        return Sample(
            input=prompt,
            choices=choices,
            target=target,
            metadata={"task_name": task_name, "debug": debug, "use_cot": use_cot, "original_prompt": prompts[0]}
        )

    return record_to_sample

def get_prompt_template(
    task_name: str,
    use_claude: bool = False,
    use_example: bool = True
) -> str:
    """
    Get the prompt template for a specific LegalBench task.
    
    Args:
        task_name: The name of the LegalBench task
        use_claude: Whether to use Claude-specific prompts
        use_example: Whether to include examples in the prompt
        
    Returns:
        The prompt template as a string
    """
    # Define the template filename based on conditions
    if use_claude:
        template_filename = "claude_prompt.txt"
    elif use_example:
        template_filename = "base_prompt.txt"
    else:
        template_filename = "base_prompt_wo_example.txt"
        
    # Try both relative paths (.. and .) to handle different execution contexts
    for base_path in ["../legalbench", "./legalbench"]:
        try:
            prompt_template_path = f"{base_path}/tasks/{task_name}/{template_filename}"
            with open(prompt_template_path) as in_file:
                return in_file.read()
        except FileNotFoundError:
            continue
            
    raise FileNotFoundError(f"Could not find prompt template for task {task_name}")

@task
def legalbench_initial(task_name: str = "maud_accuracy_of_target_general_rw_bringdown_timing_answer", use_cot: bool = False, debug: bool = False) -> Task:
    """
    Initial evaluation task for the LegalBench dataset.

    Args:
        task_name (str): The name of the LegalBench task to evaluate.
    """

    # Load the prompt template for the specified task
    prompt_template = get_prompt_template(task_name)

    # Load the LegalBench dataset
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=get_record_to_sample(prompt_template, task_name, debug, use_cot),
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=True,
        shuffle=not debug,
    )
    if debug:
        dataset = dataset[:5]

    return Task(
        dataset=dataset,
        solver=[
            multiple_choice_save_cot(multiple_correct=False, shuffle=False, cot=use_cot)
        ],
        scorer=choice(),
    )

@task
def legalbench_initial_aggregated(task_names: list[str] = [
    # 'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
    'maud_financial_point_of_view_is_the_sole_consideration', 
    # 'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
    # 'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
], debug: bool = False, use_example: bool = True, use_cot: bool = False, use_claude: bool = False) -> Task:
    """
    Initial evaluation task for multiple LegalBench datasets aggregated into a single dataset.

    Args:
        task_names (list[str]): The list of LegalBench task names to evaluate.
        use_few_shot (bool): Whether to use few-shot prompting.
        use_cot (bool): Whether to use chain-of-thought prompting.
    """

    # Initialize an empty list to collect samples from all tasks
    all_samples = []

    for task_name in task_names:
        # Load the prompt template for the specified task
        try:
            prompt_template = get_prompt_template(task_name, use_claude, use_example)
        except FileNotFoundError as e:
            logging.error(f"Skipping task {task_name}: {e}")
            continue

        # Load the LegalBench dataset for this task
        dataset = hf_dataset(
            path="nguha/legalbench",
            name=task_name,
            sample_fields=get_record_to_sample(prompt_template, task_name, debug, use_cot),
            split="test",  # Use "test" or "validation" as appropriate
            auto_id=False,
            shuffle=True if not debug else False,
            trust=True
        )

        if debug:
            dataset = dataset[:10]

        # Collect samples from this dataset
        samples = dataset.samples
        all_samples.extend(samples)

    # Create a combined dataset
    combined_dataset = MemoryDataset(name="legalbench_aggregated", samples=all_samples)
    if debug:
        combined_dataset = combined_dataset[:50]

    solvers = [
       multiple_choice_save_cot(multiple_correct=False, shuffle=True, cot=use_cot)
    ]

    return Task(
        dataset=combined_dataset,
        solver=solvers,
        scorer=choice(),
    )

@task
def adaptive_legal(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 1,
    n_negative_samples: int = 2,
    generator_model_name: str = "openai/gpt-4o",
    eval_model_name: str = "openai/gpt-4o",
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    use_claude: bool = False,
    use_example: bool = True,
    randomize_sampling: bool = False,
    judge_model_name: Optional[str] = None,
    cot_in_context: bool = False,
    original_eval_model_name: Optional[str] = None,
) -> Task:
    """
    Adaptive evaluation task for the LegalBench dataset.

    Args:
        initial_log_path (str): Path to the initial evaluation log.
        task_name (str): The name of the LegalBench task to evaluate.
        n_positive_samples (int): Number of correctly answered samples to use for adaptation.
        n_negative_samples (int): Number of incorrectly answered samples to use for adaptation.
        generator_model_name (str): Name of the model used to generate new questions.
        eval_model_name (str): Name of the model used to evaluate the new questions.
        use_cot_generator (bool): Whether to use chain-of-thought prompting for generation.
        use_cot_evaluator (bool): Whether to use chain-of-thought prompting for evaluation.
        randomize_sampling (bool): Whether to shuffle sample selection.
        judge_model_name (Optional[str]): Name of the model used to judge correctness.
        use_claude (bool): Whether to use Claude for generation and evaluation.
        use_example (bool): Whether to use examples in the prompt.
        original_eval_model_name (Optional[str]): Name of the model used to evaluate the original questions-- facilitates transfer experiment logging.
    Returns:
        Task: An Inspect Task instance that runs an adaptive solver pipeline
        and includes the judged scorer if a judge_model_name is provided.
    """
    # Get the prompt template
    prompt_template = get_prompt_template(task_name, use_claude, use_example)
    
    # Load the LegalBench dataset for this task
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=get_record_to_sample(prompt_template, task_name, debug=False, use_cot=use_cot_evaluator),
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=False,
        shuffle=randomize_sampling,
        trust=True
    )[:1]

    # We always include the standard solver chain: the main adaptive solver and
    # possibly a judge solver. Add them to the 'solver' list:
    solver_list = [
        adaptive_legal_solver(
            initial_log_path=initial_log_path,
            task_name=task_name,
            cot_in_context=cot_in_context,
            use_claude=use_claude,
            use_example=use_example,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            use_cot_generator=use_cot_generator,
            use_cot_evaluator=use_cot_evaluator,
            randomize_sampling=randomize_sampling,
            original_eval_model_name=original_eval_model_name,
        )
    ]
    # If we have a judge model, include the judge solver
    if judge_model_name:
        solver_list.append(
            adaptive_legal_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            )
        )

    # Now assemble the appropriate scorers. The standard scorers are:
    #   1) adaptive_legal_scorer
    #   2) adaptive_legal_judge_scorer
    # If we have a judge model, we also include adaptive_legal_scorer_judged
    scorer_list = [
        adaptive_legal_scorer(),
        adaptive_legal_judge_scorer(),
    ]
    if judge_model_name:
        scorer_list.append(adaptive_legal_scorer_judged())

    return Task(
        dataset=dataset,
        solver=solver_list,
        scorer=scorer_list,
    )

@task
def adaptive_legal_refactor(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 1,
    n_negative_samples: int = 2,
    generator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    self_check_model_name: Optional[str] = None,
    use_embeddings: bool = False,
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_threshold: float = 0.6,
    score_threshold: int = 4,
    max_attempts: int = 5,
    randomize_sampling: bool = False,
    cot_in_context: bool = False,
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    use_claude: bool = False,
    use_example: bool = True,
    original_eval_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    use_eval_model_for_checker: bool = False,
    include_previous_reasoning: bool = False,
    previous_reasoning_limit: int = 0,
) -> Task:
    """
    Refactored adaptive evaluation task for the LegalBench dataset.

    Args:
        initial_log_path (str): Path to the initial evaluation log.
        task_name (str): The name of the LegalBench task to evaluate.
        n_positive_samples (int): Number of correctly answered samples to use for adaptation.
        n_negative_samples (int): Number of incorrectly answered samples to use for adaptation.
        generator_model_name (str): Name of the model used to generate new questions.
        eval_model_name (str): Name of the model used to evaluate the new questions.
        self_check_model_name (Optional[str]): Name of the model used for self-checking.
        use_embeddings (bool): Whether to use embeddings for similarity checking.
        embeddings_model_name (str): Name of the embeddings model to use.
        similarity_threshold (float): Threshold for similarity checking.
        score_threshold (int): Minimum score required for acceptance.
        max_attempts (int): Maximum number of generation attempts.
        randomize_sampling (bool): Whether to shuffle sample selection.
        cot_in_context (bool): Whether to include chain of thought in context.
        use_cot_generator (bool): Whether to use chain of thought for generation.
        use_cot_evaluator (bool): Whether to use chain of thought for evaluation.
        use_claude (bool): Whether to use Claude-specific prompts.
        use_example (bool): Whether to include examples in prompts.
        original_eval_model_name (Optional[str]): Name of original evaluation model.
        judge_model_name (Optional[str]): Name of the model used to judge correctness.
        use_eval_model_for_checker (bool): Whether to use eval model for checking.
        include_previous_reasoning (bool): Whether to include previous failure mode reasoning traces in context for the generator.
        previous_reasoning_limit (int): Number of previous failure mode reasoning traces to include in context.
"""
    # Load the prompt template for the specified task
    prompt_template = get_prompt_template(task_name)

    # Load the LegalBench dataset
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=get_record_to_sample(prompt_template, task_name, debug=False, use_cot=use_cot_evaluator),
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=True,
    )[:1]
    # Construct the solver list: the main adaptive solver plus (if specified) a judge solver
    solver_list = [
        adaptive_legal_solver_refactor(
            initial_log_path=initial_log_path,
            task_name=task_name,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            self_check_model_name=self_check_model_name,
            use_embeddings=use_embeddings,
            embeddings_model_name=embeddings_model_name,
            similarity_threshold=similarity_threshold,
            score_threshold=score_threshold,
            max_attempts=max_attempts,
            randomize_sampling=randomize_sampling,
            cot_in_context=cot_in_context,
            use_cot_generator=use_cot_generator,
            use_cot_evaluator=use_cot_evaluator,
            use_claude=use_claude,
            use_example=use_example,
            original_eval_model_name=original_eval_model_name,
            use_eval_model_for_checker=use_eval_model_for_checker,
            include_previous_reasoning=include_previous_reasoning,
            previous_reasoning_limit=previous_reasoning_limit,
        ),
    ]

    # If we have a separate judge model, append the judge solver
    if judge_model_name:
        solver_list.append(
            adaptive_legal_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            )
        )

    # Build up the scorers list
    scorer_list = [
        adaptive_legal_scorer(),
        adaptive_legal_judge_scorer(),
    ]
    # If we have a judge model, also include the judged scorer
    if judge_model_name:
        scorer_list.append(adaptive_legal_scorer_judged())

    return Task(
        dataset=dataset,
        solver=solver_list,
        scorer=scorer_list,
        epochs=Epochs(
            epochs=5,
            reducer=[
                "mean",
                novelty_filter_judged_only(
                    embeddings_model_name=embeddings_model_name,
                    similarity_threshold=similarity_threshold,
                    scorer_id="adaptive_legal_scorer_judged"  # only apply novelty check on final judged scores
                )
            ]
        ),
    )

@task
def legalbench_reworded(
    task_name: str = "maud_accuracy_of_target_general_rw_bringdown_timing_answer",
    model_name: str = "openai/gpt-4o-mini",
    rewording_model_name: str = "openai/gpt-4o-mini",
    cot: bool = False,
    debug: bool = False
) -> Task:
    """
    Evaluation task for the LegalBench dataset with reworded questions.

    Args:
        task_name (str): The name of the LegalBench task to evaluate.
        model_name (str): The name of the model used to answer the question.
        rewording_model_name (str): The name of the model used to reword the question.
        debug (bool): If True, use only a 5 question subset for debugging.
    """

    # Load the prompt template for the specified task
    prompt_template = get_prompt_template(task_name)

    # Load the LegalBench dataset
    dataset = hf_dataset(
        path="nguha/legalbench",
        name=task_name,
        sample_fields=get_record_to_sample(prompt_template, task_name, debug, cot),
        split="test",  # Use "test" or "validation" as appropriate
        auto_id=True,
        shuffle=True,
    )

    # If in debug mode, limit to 5 samples
    if debug:
        dataset = dataset[:5]

    return Task(
        dataset=dataset,
        solver=[
            rewording_legal_solver(model_name=model_name, rewording_model_name=rewording_model_name, cot=cot),
            multiple_choice_save_cot(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

@task
def legalbench_reworded_aggregated(
    task_names: list[str] = [
        'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
        'maud_financial_point_of_view_is_the_sole_consideration', 
        'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ],
    model_name: str = "openai/gpt-4o-mini",
    rewording_model_name: str = "openai/gpt-4o-mini",
    cot: bool = False,
    use_claude: bool = False,
    use_example: bool = True,
    debug: bool = False
) -> Task:
    """
    Aggregated evaluation task for multiple LegalBench datasets with reworded questions.

    Args:
        task_names (list[str]): The list of LegalBench task names to evaluate.
        model_name (str): The name of the model used to answer the question.
        rewording_model_name (str): The name of the model used to reword the question.
        cot (bool): Whether to use chain-of-thought prompting.
        use_claude (bool): Whether to use Claude for rewording.
        use_example (bool): Whether to use examples in the prompt.
        debug (bool): If True, use only a 5 question subset for debugging.
    """
    # Initialize an empty list to collect samples from all tasks
    all_samples = []

    for task_name in task_names:
        # Load the prompt template for the specified task
        try:
            prompt_template = get_prompt_template(task_name, use_claude, use_example)
        except FileNotFoundError as e:
            logging.error(f"Skipping task {task_name}: {e}")
            continue

        # Define a function to convert dataset records to Sample objects
        def record_to_sample(record: dict[str, Any]) -> Sample:
            df = pd.DataFrame([record])
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            choices = [match[1] for match in options_matches]
            target = record["answer"].strip()

            sample_id = str(hash(task_name + prompt + target))

            return Sample(
                input=prompt,
                choices=choices,
                target=target,
                id=sample_id,
                metadata={"task_name": task_name}
            )

        # Load the LegalBench dataset for this task
        dataset = hf_dataset(
            path="nguha/legalbench",
            name=task_name,
            sample_fields=record_to_sample,
            split="test", 
            shuffle=True if not debug else False,
        )

        # Collect samples from this dataset
        # If in debug mode, limit to 30 samples
        if debug:
            dataset = dataset[:30]
        samples = dataset.samples
        all_samples.extend(samples)

    # Create a combined dataset
    combined_dataset = MemoryDataset(name="legalbench_reworded_aggregated", samples=all_samples)

    return Task(
        dataset=combined_dataset,
        solver=[
            rewording_legal_solver(model_name=model_name, rewording_model_name=rewording_model_name, cot=cot),
            multiple_choice_save_cot(multiple_correct=False, shuffle=True)
        ],
        scorer=choice(),
    )

@task
def legalbench_reworded_judged(
    task_names: list[str] = [
        'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
        'maud_financial_point_of_view_is_the_sole_consideration', 
        'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ],
    model_name: str = "openai/gpt-4o-mini",
    rewording_model_name: Optional[str] = "openai/gpt-4o-mini",
    judge_model_name: Optional[str] = None,
    cot: bool = False,
    debug: bool = False,
) -> Task:
    """
    Evaluation task for the LegalBench dataset with reworded questions and a 
    judge solver to verify that the reworded questions preserve original legal content.

    Args:
        task_names: The list of LegalBench task names to evaluate.
        model_name: The name of the model used to answer the question.
        rewording_model_name: The name of the model used to reword the question.
        judge_model_name: The name of the model used to judge the correctness of rewording.
        cot: Whether to use chain-of-thought prompting for the question solver.
        debug: If True, use only a small subset for debugging.
    """
    all_samples = []
    for task_name in task_names:
        prompt_template_path = f"../legalbench/tasks/{task_name}/base_prompt.txt"
        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        def record_to_sample(record: dict[str, Any]) -> Sample:
            df = pd.DataFrame([record])
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            choices = [match[1] for match in options_matches]

            target = record["answer"].strip()

            prompt_text = prompt
            question_hash = str(hash(prompt_text))
            sample_id = f"{task_name}_{question_hash}_{target}"

            return Sample(
                input=prompt,
                choices=choices,
                target=target,
                id=sample_id,
                metadata={"task_name": task_name},
            )

        dataset = hf_dataset(
            path="nguha/legalbench",
            name=task_name,
            sample_fields=record_to_sample,
            split="test",
            shuffle=True if not debug else False,
        )
        if debug:
            dataset = dataset[:5]
        samples = dataset.samples
        all_samples.extend(samples)

    return Task(
        dataset=MemoryDataset(name="legalbench_reworded_judged", samples=all_samples),
        solver=[
            rewording_legal_solver(
                model_name=model_name,
                rewording_model_name=rewording_model_name,
                cot=cot,
            ),
            multiple_choice(
                multiple_correct=False,
                shuffle=True,
            ),
            rewording_legal_judge_solver(
                judge_model_name=judge_model_name,
                num_attempts=3,
            ),
        ],
        scorer=[
            choice(),
            judge_scoring(),
            choice_judged(),
        ],
    )

@task
def re_evaluate_adaptive_legal(
    adaptive_log_path: str,
    use_cot: bool = False,
    filter_by_incorrect: bool = False,
) -> Task:
    """
    Re-evaluates adaptive legal questions that passed judge filtering.
    The model will be inferred from the task configuration.
    
    Args:
        adaptive_log_path: Path to the log from an adaptive legal experiment
        use_cot: Whether to use chain-of-thought reasoning in the evaluation
        filter_by_incorrect: If True, only re-evaluate questions that were answered incorrectly
    
    Returns:
        A Task that will re-evaluate the questions from the adaptive experiment
    """
    logger = logging.getLogger(__name__)
    
    # Load and validate log
    eval_log = read_eval_log(adaptive_log_path)
    if not eval_log.samples:
        raise ValueError("No samples found in the adaptive legal log.")
    logger.info(f"Found {len(eval_log.samples)} non-filtered samples in {adaptive_log_path}")
    
    # Filter samples based on judge approval and prepare prompts
    filtered_samples = []
    for sample_item in eval_log.samples:
        generated_sample = sample_item.store.get("generated_sample")
        if not generated_sample:
            continue
            
        # Get question data
        question_str = generated_sample.get("input", "").strip()
        if not question_str:
            print(f"Skipping sample because it has no question: {generated_sample}")
            continue
        target = generated_sample.get("target", None)
        if not target:
            print(f"Skipping sample because it has no target: {generated_sample}")
            continue

        # extract the choices from the question prompt
        option_pattern = r"Option ([A-Z]): (.*)"
        options_matches = re.findall(option_pattern, question_str)
        choices = [match[1] for match in options_matches]
            
        # If filter_by_incorrect is True, only include samples that were incorrect
        if filter_by_incorrect:
            score = generated_sample.get("metadata", {}).get("score", "")
            judge_choice = generated_sample.get("metadata", {}).get("judge_choice", "")
            if score != "I" or judge_choice == "C":  # Only include incorrect samples or samples that were judged as valid
                continue
                
        # Create sample with prepared prompt
        filtered_samples.append(Sample(
            input=str(question_str),
            choices=choices,
            target=target,
            metadata={
                "original_question": question_str,
                "original_choices": choices,
                "original_target": target,
                "original_metadata": json.dumps(generated_sample.get("metadata", {})),
                "original_eval_model": eval_log.eval.task_args.get("eval_model_name", "unknown"),
                "original_generator_model": eval_log.eval.task_args.get("generator_model_name", "unknown"),
                "original_judge_model": eval_log.eval.task_args.get("judge_model_name", "unknown"),
                "original_embeddings_model": eval_log.eval.task_args.get("embeddings_model_name", "sentence-transformers/all-mpnet-base-v2"),
            }
        ))
    
    if not filtered_samples:
        logger.info("No samples passed filtering. Nothing to evaluate.")
        raise ValueError("No samples passed filtering from the adaptive solver log.")
    logger.info(f"Found {len(filtered_samples)} samples post-filtering.")
    
    dataset = MemoryDataset(
        name="re_evaluate_adaptive_legal",
        samples=filtered_samples
    )
    
    return Task(
        dataset=dataset,
        solver=[multiple_choice_save_cot(multiple_correct=False, shuffle=True, cot=use_cot)],
        scorer=choice(),
        reducer=("mean",),
    )

if __name__ == "__main__":
    # List of models to use for generation and evaluation
    model_list_generator = [
        "openai/gpt-4o",
        # "openai/gpt-4o-mini",
        # "anthropic/claude-3-5-sonnet-20240620",
        # "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        # "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ]
    model_list_eval = [
        # "openai/gpt-4o",
        "openai/gpt-4o-mini",
        # "anthropic/claude-3-5-sonnet-20240620",
    ]

    # Specify the LegalBench tasks to evaluate
    task_names = [
        'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
        'maud_financial_point_of_view_is_the_sole_consideration', 
        'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ]
    # task_names = [task for task in TASKS if task.startswith("maud_")]

    # for this run, we are always evaluation gpt-4o, and sweep across generator models and n_positive_samples/n_negative_samples
    # this is for an adaptive evaluation with the aggregated task
    initial_log_path = "logs/2024-12-06T14-38-07-05-00_legalbench-initial-aggregated_CP3Juo4BSyFjLMzfJ8VYyU.json"
    # initial_log_path = "logs/2024-10-30T10-44-42-04-00_legalbench-initial-aggregated_gQXZisrb6539kQGbrUWSMC.json"
    # initial_log_path = "logs/2024-10-30T10-35-29-04-00_legalbench-initial-aggregated_m6nH9qBkc6hrgRkcuRfGqx.json"
    for positive_samples in [1]:
        for negative_samples in [4, 8, 16, 32, 64]:
        # for negative_samples in [32]:
    # for positive_samples in [4]:
    #     for negative_samples in [60]:
    # for positive_samples in [2]:
        # for negative_samples in [10]:
            for generator_model in model_list_generator:
                log_dir = "logs/"
                # task = adaptive_legal(
                #     initial_log_path=initial_log_path,
                #     task_name="maud_specific_performance",
                #     n_positive_samples=positive_samples,
                #     n_negative_samples=negative_samples,
                #     generator_model_name=generator_model,
                #     eval_model_name="openai/gpt-4o",
                #     use_cot_generator=False,
                #     use_cot_evaluator=False,
                # )
                # eval(
                #     task,
                #     epochs=Epochs(30, "mean"),
                #     max_connections=50,
                #     log_dir=log_dir,
                #     model="openai/gpt-4o-mini",
                #     temperature=0,
                # )
                # task = adaptive_legal(
                #     initial_log_path=initial_log_path,
                #     task_name="maud_specific_performance",
                #     n_positive_samples=positive_samples,
                #     n_negative_samples=negative_samples,
                #     generator_model_name=generator_model,
                #     eval_model_name="openai/gpt-4o",
                #     use_cot_generator=True,
                #     use_cot_evaluator=False,
                # )
                # eval(
                #     task,
                #     epochs=Epochs(30, "mean"),
                #     max_connections=50,
                #     log_dir=log_dir,
                #     model="openai/gpt-4o-mini",
                #     temperature=0,
                # )
                task = adaptive_legal(
                    initial_log_path=initial_log_path,
                    task_name="maud_specific_performance",
                    n_positive_samples=positive_samples,
                    n_negative_samples=negative_samples,
                    generator_model_name=generator_model,
                    eval_model_name="openai/gpt-4o",
                    use_cot_generator=True,
                    use_cot_evaluator=False,
                    randomize_sampling=False,
                    judge_model_name="openai/o1-preview",
                )
                eval(
                    task,
                    epochs=Epochs(30, "mean"),
                    max_connections=1000,
                    log_dir=log_dir,
                    model="openai/gpt-4o",
                    temperature=0,
                )
    # for task_name in task_names:
    #     for eval_model in model_list_eval:
    #         # Run the initial LegalBench task
    #         task = legalbench_initial(task_name=task_name)
    #         log_dir = f"legalbench_logs/initial_legal_{task_name}_{eval_model.replace('/', '_')}"
    #         if os.path.exists(log_dir):
    #             # check if any json exists in log_dir
    #             json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    #         else:
    #             json_files = []

    #         if json_files:
    #             try:
    #                 initial_log_path = os.path.join(log_dir, max(
    #                     json_files,
    #                     key=lambda x: os.path.getctime(os.path.join(log_dir, x))
    #                 ))  
    #                 task_log = read_eval_log(initial_log_path)
    #                 print(f"Skipping initial legal task for {eval_model} because it already exists, in {log_dir}, called {initial_log_path}")
    #             except Exception as e:
    #                 print(f"An error occurred while retrieving the latest JSON file: {e}")
    #                 # Handle the error as needed, possibly continue or exit
    #         else:
    #             print(f"No JSON files found in {log_dir}. Proceeding with the initial truthfulqa task.")
    #             task_log = eval(task, epochs=Epochs(1, "max"), max_connections=50, log_dir=log_dir, model=eval_model, log_level="error")[0]
    #             initial_log_path = os.path.join(log_dir, max(
    #                 [f for f in os.listdir(log_dir) if f.endswith('.json')],
    #                 key=lambda x: os.path.getctime(os.path.join(log_dir, x))
    #             ))
    #         # check if the task was successful
    #         if task_log.status == "success":
    #             print(f"Task {task_name} with {eval_model} completed successfully.")

    #             for generator_model in model_list_generator:
    #                 # Run the adaptive LegalBench task
    #                 for positive_samples in [2]:
    #                     for negative_samples in [2, 4, 16]:
    #                         log_dir = f"legalbench_logs/adaptive_legal_{task_name}_{eval_model.replace('/', '_')}"
    #                         if not os.path.exists(log_dir):
    #                             os.makedirs(log_dir)
    #                             task_log = None
    #                         else:
    #                             json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    #                             if json_files:
    #                                 task_logs = [read_eval_log(os.path.join(log_dir, f)) for f in json_files]
    #                             else:
    #                                 print(f"No JSON files found in {log_dir}. Proceeding with the adaptive legal task.")
    #                                 task_log = None
    #                             # Start of Selection
    #                             if not task_log or any((log.status != "success" or log.use_cot_generator) for log in task_logs):
    #                                 task = adaptive_legal(
    #                                     initial_log_path=initial_log_path,
    #                                     task_name=task_name,
    #                                     n_positive_samples=positive_samples,
    #                                     n_negative_samples=negative_samples,
    #                                     generator_model_name=generator_model,
    #                                     eval_model_name=eval_model,
    #                                     use_cot_generator=False,
    #                                     use_cot_evaluator=False,
    #                                 )
    #                                 eval(
    #                                     task,
    #                                     epochs=Epochs(30, "mean"),
    #                                     max_connections=50,
    #                                     log_dir=log_dir,
    #                                     model=eval_model,
    #                                     temperature=0,
    #                                 )
    #                             elif not task_log or any((log.status != "success" or not log.use_cot_generator) for log in task_logs):
    #                                 task = adaptive_legal(
    #                                     initial_log_path=initial_log_path,
    #                                     task_name=task_name,
    #                                     n_positive_samples=positive_samples,
    #                                     n_negative_samples=negative_samples,
    #                                     generator_model_name=generator_model,
    #                                     eval_model_name=eval_model,
    #                                     use_cot_generator=True,
    #                                     use_cot_evaluator=False,
    #                                 )
    #                                 eval(
    #                                     task,
    #                                     epochs=Epochs(30, "mean"),
    #                                     max_connections=50,
    #                                     log_dir=log_dir,
    #                                     model=eval_model,
    #                                     temperature=0,
    #                                 )
    #         else:
    #             print(f"Task {task_name} with {eval_model} failed.")