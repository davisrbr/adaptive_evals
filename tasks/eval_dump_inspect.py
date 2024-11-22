"""
Measuring Massive Multitask Language Understanding

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300

Based on: https://github.com/openai/simple-evals/blob/main/mmlu_eval.py

# eval all subjects w/ 500 randomly selected samples
inspect eval inspect_evals/mmlu --limit 500

# add chain of thought
inspect eval inspect_evals/mmlu --limit 500 -T cot=true

# eval selected subjects
inspect eval inspect_evals/mmlu -T subjects=anatomy
inspect eval inspect_evals/mmlu -T subjects=astronomy
inspect eval inspect_evals/mmlu -T subjects=anatomy,astronomy
"""

from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


# map records to inspect sample
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["A"]),
            str(record["B"]),
            str(record["C"]),
            str(record["D"]),
        ],
        target=record["Answer"],
        metadata={"subject": record["Subject"]},
    )


# read dataset globally so it can be shared by all of the tasks
# (shuffle so that --limit draws from multiple subjects)
dataset = csv_dataset(
    csv_file="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    sample_fields=record_to_sample,
    auto_id=True,
    shuffle=True,
)


@task
def mmlu(subjects: str | list[str] = [], cot: bool = False) -> Task:
    """
    Inspect Task implementation for MMLU

    Args:
        subjects (str | list[str]): Subjects to filter to
        cot (bool): Whether to use chain of thought
    """
    # filter dataset if requested
    subjects = subjects if isinstance(subjects, list) else [subjects]
    if len(subjects) > 0:
        task_dataset = dataset.filter(
            name=f"{dataset.name}-{'-'.join(subjects)}",
            predicate=lambda sample: sample.metadata is not None
            and sample.metadata.get("subject") in subjects,
        )
    else:
        task_dataset = dataset

    solver = multiple_choice(cot=cot)

    return Task(
        dataset=task_dataset,
        solver=solver,
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )

@task
def truthfulqa(target: Literal["mc1", "mc2"] = "mc1") -> Task:
    """
    TruthfulQA: Measuring How Models Mimic Human Falsehoods

    Stephanie Lin, Jacob Hilton, Owain Evans
    https://arxiv.org/abs/2109.07958v2

    # Eval truthful QA with reference questions (mc1 - single true answer)
    inspect eval truthfulqa.py

    # Eval against the control questions (mc2 - multi-true answer)
    inspect eval truthfulqa.py -T target=mc2
    Inspect Task implementation for the TruthfulQA benchmark

    Args:
        target (Literal["mc1", "mc2"]): Whether to use the mc1 or mc2 targets
    """

    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
        )

    dataset = hf_dataset(
        path="truthful_qa",
        name="multiple_choice",
        sample_fields=record_to_sample,
        split="validation",
        auto_id=True,
        shuffle=True,
    )

    # As per the reference (), mc1 only has a single true answer whereas there
    # can be multiple true answers in mc2. See:
    # https://github.com/sylinrl/TruthfulQA/blob/fdd8ad1c0d00a478cf8b0bb41a3ad8378c16293b/README.md#multiple-choice
    if target == "mc1":
        multiple_correct = False
    else:
        multiple_correct = True

    return Task(
        dataset=dataset,
        solver=[multiple_choice(multiple_correct=multiple_correct, shuffle=True)],
        scorer=choice(),
    )


# The dataset uses a binary list for each target, where 1 indicates an answer is
# correct and 0 is incorrect. For example, if there are three options and the
# second is correct, the target would be [0, 1, 0].
#
# This function converts that to a list of letters corresponding to the correct
# answers, which allows us to use the `choice("letter")` scorer.
#     e.g. [0, 1, 1] -> ["B", "C"]
def labels_to_positions(labels: list[int]) -> list[str]:
    return [chr(ord("A") + i) for i, label in enumerate(labels) if label == 1]