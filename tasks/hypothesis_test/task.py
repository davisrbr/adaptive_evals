"""Hypothesis-driven eval: testing tool confusion and investigation loops.

Hypotheses from baseline analysis (50% accuracy on swe_env_baseline):
- H1 (tool_confusion): Model tries vim/nano/editors instead of bash heredoc/python,
  wastes turns on unavailable tools, especially for multi-line edits
- H2 (investigation_loop): Model over-investigates simple bugs instead of
  applying the obvious fix, hitting message limit
- H3 (regex_complexity): Bugs requiring regex changes are harder because the
  model struggles with sed/inline replacement of special chars

Controls: Simple logic bugs (like baseline passing samples) that should pass.
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
    # === H1: Tool confusion (multi-line edits tempt editor usage) ===
    Sample(
        id="csv_quoted_fields_fix",
        input=(
            "The file /testbed/csvparser.py has a bug in the `parse_csv` function. "
            "It uses naive comma splitting which breaks on quoted fields containing "
            "commas (e.g. '\"New York, NY\"'). The fix requires replacing the simple "
            "split with proper CSV-aware parsing. Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_csvparser.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H1_tool_confusion",
            "category": "csv_parsing",
            "trigger": "multi-line replacement needed",
        },
    ),
    Sample(
        id="email_regex_fix",
        input=(
            "The file /testbed/validator.py has a bug in the `validate_email` function. "
            "The regex pattern rejects valid emails with '+' in the local part "
            "(like user+tag@example.com) and emails with subdomains "
            "(like user@mail.example.com). Fix the regex and verify with: "
            "cd /testbed && python -m unittest test_validator.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H3_regex_complexity",
            "category": "regex_fix",
            "trigger": "regex special chars in sed/replacement",
        },
    ),
    # === H2: Investigation loop (obvious fix, distractor complexity) ===
    Sample(
        id="list_format_fix",
        input=(
            "The file /testbed/formatter.py has a bug in the `format_report` function. "
            "When a value is a list, it should be displayed as comma-separated items "
            "(e.g. 'python, coding') but instead shows the Python list repr "
            "(e.g. \"['python', 'coding']\"). Fix the bug and verify with: "
            "cd /testbed && python -m unittest test_formatter.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H2_investigation_loop",
            "category": "string_formatting",
            "trigger": "obvious fix but model may over-investigate",
        },
    ),
    # === Control: Simple bug, should pass ===
    Sample(
        id="word_count_case_fix",
        input=(
            "The file /testbed/counter.py has a bug in the `count_words` function. "
            "It should count words case-insensitively but doesn't lowercase them, "
            "so 'Hello' and 'hello' are counted separately. Fix the bug and verify "
            "with: cd /testbed && python -m unittest test_counter.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "control",
            "category": "case_sensitivity",
            "trigger": "none - simple one-line fix",
        },
    ),
]


@task
def swe_hypothesis_test():
    """Test hypotheses about tool confusion and investigation loops."""
    return Task(
        dataset=MemoryDataset(SAMPLES),
        solver=react(
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
