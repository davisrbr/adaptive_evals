
"""
Multilingual politness initial task and adaptive version
"""

from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from solvers.adaptive_utils import multiple_choice_save_cot


@task
def politeness_initial(debug: bool = False) -> Task:
    """
    Multilingual politeness initial task
    """

    choices = [str(i) for i in [-2, -1, 0, 1, 2]]
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["Utterance"],
            choices=choices,
            target=str(round(record["politeness"])),
        )

    dataset = hf_dataset(
        path="shreyahavaldar/multilingual_politeness",
        name="default",
        sample_fields=record_to_sample,
        split="train",
        auto_id=True,
        shuffle=True,
    )
    if debug:
        dataset = dataset[:10]

    return Task(
        dataset=dataset,
        solver=[multiple_choice_save_cot(shuffle=True)],
        scorer=choice(),
    )
