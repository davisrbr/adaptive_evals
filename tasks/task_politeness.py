"""
Multilingual politness initial task and adaptive version
"""

from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import Generate, Solver, TaskState, generate, multiple_choice, solver

from solvers.adaptive_utils import multiple_choice_save_cot


TASK_POLITENESS_MULTISHOT_PROMPT_TEMPLATE = r"""
The following are examples of utterances and their politeness ratings on a scale from -2 (very impolite) to +2 (very polite). At the end of these examples, you will be given a new utterance to rate:

{dev_examples}

Please rate the following utterance:
{test_utterance}
""".strip()

TASK_POLITENESS_EXAMPLE_TEMPLATE = r"""
Utterance: {utterance}
(A) -2 (very impolite)
(B) -1 (somewhat impolite)
(C) 0 (neutral)
(D) +1 (somewhat polite) 
(E) +2 (very polite)
Answer: {answer}
""".strip()

@task
def politeness_n_shot(n_examples: int = 5, debug: int = -1, cot: bool = False) -> Task:
    """
    Multilingual politeness task with n-shot prompting
    n_examples: number of examples to use for demonstrations
    debug: number of examples to use for debugging (default: -1, which uses all examples)
    cot: whether to use chain-of-thought prompting
    """
    dataset = get_politeness_dataset(split="test", shuffle=True)
    if debug:
        dataset = dataset[:debug]

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
