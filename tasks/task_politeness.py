"""
Multilingual politness initial task and adaptive version
"""
from typing import Any, Optional, Literal
import torch
import logging


import inspect_ai
from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import Generate, Solver, TaskState, generate, multiple_choice, solver
from inspect_ai.log import read_eval_log

from utils_elicitation.novelty import novelty_filter_judged_only

try:
    from solvers.adaptive_utils import multiple_choice_save_cot
except ImportError:
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from solvers.adaptive_utils import multiple_choice_save_cot

from solvers.solver_adaptive_politeness import (
    adaptive_politeness_solver,
    adaptive_politeness_judge_solver
)

from scorers.scorers_politeness import (
    adaptive_politeness_scorer,
    adaptive_politeness_judge_scorer,
    adaptive_politeness_scorer_judged,
)


TASK_POLITENESS_MULTISHOT_PROMPT_TEMPLATE = r"""
The following are examples of utterances and their politeness ratings on a scale from -2 (very impolite) to +2 (very polite). At the end of these examples, you will be given a new utterance to rate:

{dev_examples}

Please rate the following utterance:
{test_utterance}
""".strip()

TASK_POLITENESS_EXAMPLE_TEMPLATE = r"""
Utterance: {utterance}
(A) -2
(B) -1
(C) 0
(D) +1
(E) +2
Answer: {answer}
""".strip()

@task
def politeness_n_shot(n_examples: int = 5, debug: int = -1, cot: bool = False, max_samples: int = None) -> Task:
    """
    Multilingual politeness task with n-shot prompting
    n_examples: number of examples to use for demonstrations
    debug: number of examples to use for debugging (default: -1, which uses all examples)
    cot: whether to use chain-of-thought prompting
    max_samples: maximum number of samples to evaluate (default: None, which uses all examples)
    """
    dataset = get_politeness_dataset(split="test", shuffle=True)
    if debug > 0:
        dataset = dataset[:debug]
    elif max_samples and max_samples > 0:
        dataset = dataset[:max_samples]

    return Task(
        dataset=dataset,
        solver=[
            politeness_n_shot_solver(n_examples=n_examples),
            multiple_choice_save_cot(shuffle=True, cot=cot),
        ],
        scorer=choice(),
    )


@solver
def politeness_n_shot_solver(n_examples: int = 5) -> Solver:
    """A custom solver for politeness n-shot.
    Uses the first n examples from the dataset as demonstrations.
    """
    dev_dataset = get_politeness_dataset(split="train", shuffle=False)[:n_examples]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dev_examples = "\n\n".join([
            format_politeness_example(
                utterance=str(sample.input),
                answer=str(sample.target),
            )
            for sample in dev_dataset
        ])

        state.user_prompt.text = f"""The following are examples of utterances and their politeness ratings on a scale from -2 (very impolite) to +2 (very polite):

{dev_examples}

Rate the following utterance:
{state.input}"""

        return state

    return solve

@task  
def adaptive_politeness(
    initial_log_path: str,
    n_positive_samples: int = 8,
    n_negative_samples: int = 8,
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
    original_eval_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    use_eval_model_for_checker: bool = False,
) -> Task:
    """
    Adaptive Politeness task that generates new questions based on model errors and evaluates the model on them.
    """
    question_embedding_map = None
    if use_embeddings:
        def record_to_sample(record: dict[str, Any]) -> Sample:
            return Sample(
                input=record["Utterance"],
                choices=[str(i) for i in [-2, -1, 0, 1, 2]],
                target=chr(ord('A') + [str(i) for i in [-2, -1, 0, 1, 2]].index(str(round(record["politeness"])))),
                metadata={"embedding": record["embedding"]},
            )
        ds = hf_dataset(
            path="davisrbr/politeness-embeddings",
            name="default",
            sample_fields=record_to_sample,
            split="validation",
            auto_id=True,
            shuffle=True,
        )

        # Build a mapping from question text to embeddings
        question_embedding_map = {
            s.input: torch.tensor(s.metadata["embedding"]) for s in ds
        }

    # Construct the solver list: the main adaptive solver plus (if specified) a judge solver
    solver_list = [
        adaptive_politeness_solver(
            initial_log_path=initial_log_path,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            self_check_model_name=self_check_model_name,
            use_embeddings=use_embeddings,
            embeddings_model_name=embeddings_model_name,
            question_embedding_map=question_embedding_map,
            similarity_threshold=similarity_threshold,
            score_threshold=score_threshold,
            max_attempts=max_attempts,
            randomize_sampling=randomize_sampling,
            cot_in_context=cot_in_context,
            use_cot_generator=use_cot_generator,
            use_cot_evaluator=use_cot_evaluator,
            original_eval_model_name=original_eval_model_name,
            use_eval_model_for_checker=use_eval_model_for_checker,
        ),
    ]

    # if we have a separate judge model, append the judge solver
    if judge_model_name:
        solver_list.append(
            adaptive_politeness_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            )
        )

    scorer_list = [
        adaptive_politeness_scorer(),
    ]
    # if we have a judge model, also include judge-based scorers
    if judge_model_name:
        scorer_list.append(adaptive_politeness_judge_scorer())
        scorer_list.append(adaptive_politeness_scorer_judged())

    return Task(
        dataset=get_politeness_dataset(split="test", shuffle=True)[:1], # placeholder dataset, is not used
        solver=solver_list,
        scorer=scorer_list,
        epochs=Epochs(
            epochs=5,
            reducer=[
                "mean",
                novelty_filter_judged_only(
                    embeddings_model_name=embeddings_model_name,
                    similarity_threshold=similarity_threshold,
                    scorer_id="adaptive_politeness_scorer_judged"  # so we only apply the novelty check/compute the embeddings on the final judged scores
                )
            ]
        ),
    )

def format_politeness_example(utterance: str, answer: str = "") -> str:
    """Format a politeness example with utterance and optional answer."""
    choices_text = "\n".join([
        "(A) -2 (very impolite)",
        "(B) -1 (somewhat impolite)", 
        "(C) 0 (neutral)",
        "(D) +1 (somewhat polite)",
        "(E) +2 (very polite)",
    ])
    
    return f"""Utterance: {utterance}
{choices_text}
Answer: {answer}""".strip()

def get_politeness_dataset(
    split: str = "train",
    shuffle: bool = False,
) -> Dataset:
    """Get the politeness dataset with the specified split."""
    choices = [str(i) for i in [-2, -1, 0, 1, 2]]
    
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["Utterance"],
            choices=choices,
            target=chr(ord('A') + choices.index(str(round(record["politeness"])))),
        )

    return hf_dataset(
        path="shreyahavaldar/multilingual_politeness",
        name="default",
        sample_fields=record_to_sample,
        split=split,
        auto_id=True,
        shuffle=shuffle,
    )

@task
def re_evaluate_adaptive_politeness(
    adaptive_log_path: str,
    use_cot: bool = False,
    filter_by_incorrect: bool = False,
) -> Task:
    """
    Re-evaluates adaptive politeness utterances that passed judge filtering.
    The model used for re-evaluation will be inferred from the task configuration.
    
    Args:
        adaptive_log_path: Path to the log file from a previous adaptive politeness evaluation
        use_cot: Whether to use chain-of-thought prompting for re-evaluation
        filter_by_incorrect: If True, only re-evaluate utterances that were previously answered incorrectly
    """
    logger = logging.getLogger(__name__)

    # Load and validate log
    eval_log = read_eval_log(adaptive_log_path)
    if not eval_log.samples:
        raise ValueError("No samples found in the adaptive politeness log.")
    logger.info(f"Found {len(eval_log.samples)} non-filtered samples in {adaptive_log_path}")

    # Filter samples based on judge approval and prepare prompts
    filtered_samples = []
    for sample_item in eval_log.samples:
        generated_sample = sample_item.store.get("generated_sample")
        if not generated_sample:
            continue
        
        # Check judge rating if available (A/B = approved, C = rejected)
        judge_choice = generated_sample["metadata"].get("judge_choice", "")
        if not judge_choice or judge_choice not in ["A", "B"]:
            # We don't have a valid judge choice
            continue
        else:
            print(f"DEBUG: Judge choice is {judge_choice} for sample {generated_sample}")
            
        # Filter by incorrect responses if requested
        if filter_by_incorrect and generated_sample['metadata'].get("score", "") != "I":
            print(f"DEBUG: Sample {generated_sample['metadata'].get('score')} is incorrect, skipping")
            continue

        # Get utterance data
        utterance = generated_sample['input']
        choices = generated_sample['choices']
        if not utterance or not choices:
            print(f"DEBUG: No utterance or choices for sample {generated_sample}")
            continue
        target = generated_sample['target']
        if not target:
            print(f"DEBUG: No target for sample {generated_sample}")
            continue

        # Create sample for re-evaluation
        filtered_samples.append(Sample(
            input=utterance,
            choices=choices,
            target=target,
            metadata={
                "original_utterance": utterance,
                "original_metadata": str(generated_sample['metadata']),
                "original_eval_model": eval_log.eval.task_args.get("eval_model_name", "gpt-4o-mini"),
                "original_generator_model": eval_log.eval.task_args.get("generator_model_name", "gpt-4o-mini"),
                "original_judge_model": eval_log.eval.task_args.get("judge_model_name", ""),
                "original_embeddings_model": eval_log.eval.task_args.get("embeddings_model_name", "sentence-transformers/all-mpnet-base-v2"),
                "language": generated_sample['metadata'].get("language", "unknown"),
            }
        ))

    if not filtered_samples:
        logger.info("No samples passed filtering criteria. Nothing to evaluate.")
        raise ValueError("No samples passed filtering from the adaptive politeness log.")
    
    logger.info(f"Found {len(filtered_samples)} samples after filtering.")
    dataset = MemoryDataset(
        name="re_evaluate_adaptive_politeness",
        samples=filtered_samples
    )

    return Task(
        dataset=dataset,
        solver=[multiple_choice_save_cot(shuffle=True, cot=use_cot)],
        scorer=choice(),
        reducer=("mean",),
    )
