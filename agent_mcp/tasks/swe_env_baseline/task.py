"""SWE-Bench environment sensitivity eval.

Tests whether a coding agent can handle environment variations.
Each sample is a small bug-fix task run inside a Docker sandbox.
We can re-run the same samples in different environments (Python
versions, available packages, etc.) to test environment sensitivity.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import scorer, accuracy, Score, CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, python, think


@scorer(metrics=[accuracy()])
def patch_checker():
    """Check if the agent produced a correct fix by running the test."""
    async def score(state: TaskState, target: Target) -> Score:
        for msg in state.messages:
            text = ""
            if hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    text = " ".join(getattr(b, "text", "") or "" for b in msg.content)

            if "passed" in text.lower() and ("PASSED" in text or "passed" in text):
                return Score(value=CORRECT, explanation="Tests passed")
            if text.strip().endswith("OK") and "test" in text.lower():
                return Score(value=CORRECT, explanation="Tests passed (OK)")

        return Score(value=INCORRECT, explanation="Tests did not pass")
    return score


SAMPLES = [
    Sample(
        id="division_by_zero_fix",
        input=(
            "The file /testbed/mathlib.py has a bug in the `summarize` function. "
            "When called with an empty list, it raises a ZeroDivisionError instead of "
            "returning a proper result. Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_mathlib.py -v"
        ),
        target="tests pass",
        metadata={"hypothesis": "baseline", "category": "division_by_zero"},
    ),
    Sample(
        id="unicode_handling_fix",
        input=(
            "The file /testbed/textproc.py has a bug in the `clean_text` function. "
            "It strips unicode characters (emoji, accented letters) instead of "
            "preserving them. Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_textproc.py -v"
        ),
        target="tests pass",
        metadata={"hypothesis": "baseline", "category": "unicode_handling"},
    ),
    Sample(
        id="recursive_glob_fix",
        input=(
            "The file /testbed/fileutil.py has a bug in the `find_configs` function. "
            "It fails to find .yaml files in subdirectories. "
            "Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_fileutil.py -v"
        ),
        target="tests pass",
        metadata={"hypothesis": "baseline", "category": "path_handling"},
    ),
    Sample(
        id="data_merge_fix",
        input=(
            "The file /testbed/dataproc.py has a bug in the `merge_records` function. "
            "When merging records with duplicate keys, it silently drops data instead "
            "of combining values. Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_dataproc.py -v"
        ),
        target="tests pass",
        metadata={"hypothesis": "baseline", "category": "data_merge"},
    ),
]


@task
def swe_env_sensitivity():
    """SWE-Bench-style eval testing environment sensitivity."""
    return Task(
        dataset=MemoryDataset(SAMPLES),
        agent=react(
            tools=[bash(timeout=60), python(timeout=60), think()],
            prompt=(
                "You are an expert software engineer. Fix the bug described below. "
                "The repository is at /testbed. Read the code, understand the bug, "
                "write a fix, and run the tests to verify."
            ),
        ),
        scorer=patch_checker(),
        sandbox=("docker", "compose.yaml"),
        message_limit=20,
    )
