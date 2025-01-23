"""
Multilingual politness initial task and adaptive version
"""
from typing import Any, Optional, Literal
import torch
import numpy as np
import pandas as pd

import inspect_ai
from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Dataset, Sample, hf_dataset, MemoryDataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import Generate, Solver, TaskState, generate, multiple_choice, solver

from datasets import load_dataset, Dataset

try:
    from solvers.adaptive_utils import multiple_choice_save_cot
except ImportError:
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from solvers.adaptive_utils import multiple_choice_save_cot


TASK_CULTURE_MULTISHOT_PROMPT_TEMPLATE = r"""
The following are questions involving cultural reasoning and their corresponding answers. Please use these examples to help you answer the following question:
{dev_examples}

Answer the following question:
{test_question}
""".strip()

TASK_CULTURE_EXAMPLE_TEMPLATE = r"""
{question}
(A) Option 1
(B) Option 2
(C) Option 3
(D) Option 4
Answer: {answer}
""".strip()

@task
def culture_n_shot(n_examples: int = 5, debug: int = -1, cot: bool = False) -> Task:
    """
    Multilingual culture task with n-shot prompting
    n_examples: number of examples to use for demonstrations
    debug: number of examples to use for debugging (default: -1, which uses all examples)
    cot: whether to use chain-of-thought prompting
    """
    dataset = get_culture_dataset(split="train", shuffle=True)
    if debug:
        dataset = dataset[:debug]

    return Task(
        dataset=dataset,
        solver=[
            culture_n_shot_solver(n_examples=n_examples),
            multiple_choice_save_cot(shuffle=True, cot=cot),
        ],
        scorer=choice(),
    )


@solver
def culture_n_shot_solver(n_examples: int = 5) -> Solver:
    """A custom solver for culture n-shot.
    Uses the first n examples from the dataset as demonstrations.
    """
    dev_dataset = get_culture_dataset(split="train", shuffle=False)[:n_examples]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dev_examples = "\n\n".join([
            format_culture_example(
                input=str(sample.input),
                answer=str(sample.target),
            )
            for sample in dev_dataset
        ])

        state.user_prompt.text = f"""The following are questions involving cultural reasoning and their corresponding answers. Please use these examples to help you answer the following question:

{dev_examples}

Answer the following question:
{state.input}"""

        return state

    return solve


def format_culture_example(input: str, answer: str = "") -> str:
    """Format a culture example with input and optional answer."""
    
    return f"""{input}
    Answer: {answer}""".strip()

def get_culture_dataset(
    split: str = "train",
    shuffle: bool = False,
) -> Dataset:
    """Get the culture dataset with the specified split."""
    choices = ["A", "B", "C", "D"]
    
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["prompt_question"] + "\n" + record["answer_choices"],
            choices=choices,
            target=record["answer"],
        )

    return hf_dataset(
        path="shreyahavaldar/CulturalBench-MC",
        name="default",
        sample_fields=record_to_sample,
        split=split,
        auto_id=True,
        shuffle=shuffle,
    )

def process_culture_dataset():
    hf_dataset = pd.read_csv("hf://datasets/kellycyy/CulturalBench/CulturalBench-Hard.csv")
    processed_dataset = pd.DataFrame(columns=["question_idx", "prompt_question", "answer_choices", "answer"])
    processed_idx = []
    processed_question = []
    processed_choices = []
    processed_answer = []
    question_idxs = np.unique(hf_dataset["question_idx"])
    for qidx in question_idxs:
        question_rows = hf_dataset[hf_dataset["question_idx"] == qidx].reset_index()
        answers = list(question_rows["answer"])
        if(np.count_nonzero(answers) != 1): continue
        question_text = question_rows.iloc[0]["prompt_question"]
        choices = ""
        answer = ""
        answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        for idx, row in question_rows.iterrows():
            choices += "({}) {}\n".format(answer_mapping[idx], row["prompt_option"])
            if(row["answer"] == 1):
                answer = answer_mapping[idx]

        processed_idx.append(qidx)
        processed_question.append(question_text)
        processed_choices.append(choices.strip())
        processed_answer.append(answer)

    processed_dataset["question_idx"] = processed_idx
    processed_dataset["prompt_question"] = processed_question
    processed_dataset["answer_choices"] = processed_choices
    processed_dataset["answer"] = processed_answer
    
    #upload to huggingface
    final_dataset = Dataset.from_pandas(processed_dataset)
    final_dataset.push_to_hub("shreyahavaldar/CulturalBench-MC")