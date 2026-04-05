"""Round 3: Deep dive into regex editing failures.

Refined hypothesis from Round 2:
- H3a (special_char_regex): Model fails specifically when the correct regex
  contains shell-special characters (+, ?, #, &) because bash/sed mangles them
- H3b (regex_logic): Model fails on regex fixes that require understanding
  multiple alternative patterns (OR groups, optional sections)
- H3c_control (simple_regex): Model succeeds on regex fixes that don't involve
  shell-special chars and have straightforward patterns

Design:
- url_parser: Fix requires adding ?#& chars to regex (H3a trigger)
- log_parser: Fix requires case-insensitive flag + level normalization (H3b)
- phone_parser: Fix requires adding alternative pattern with no special chars (H3c control)
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
    # H3a: regex with shell-special chars (?, #, &)
    Sample(
        id="url_regex_special_chars",
        input=(
            "The file /testbed/url_parser.py has a bug in the `extract_urls` function. "
            "The regex pattern doesn't match URLs with query parameters "
            "(like ?q=hello&lang=en) or fragment identifiers (like #section). "
            "Fix the regex to capture full URLs including query strings and fragments. "
            "Verify with: cd /testbed && python -m unittest test_url_parser.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H3a_special_char_regex",
            "category": "regex_special_chars",
            "trigger": "regex fix needs ? # & in pattern",
        },
    ),
    # H3b: regex needing case-insensitive + normalization logic
    Sample(
        id="log_level_case_fix",
        input=(
            "The file /testbed/log_parser.py has a bug in the `parse_log_levels` function. "
            "It only matches uppercase log levels like [INFO] but misses lowercase or "
            "mixed-case like [info] or [Warning]. The function should match case-insensitively "
            "and normalize all levels to uppercase in the output dict. "
            "Fix the bug and verify with: cd /testbed && python -m unittest test_log_parser.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H3b_regex_logic",
            "category": "regex_case_handling",
            "trigger": "needs re.IGNORECASE flag + .upper() normalization",
        },
    ),
    # H3c control: regex fix without shell-special chars
    Sample(
        id="phone_format_fix",
        input=(
            "The file /testbed/phone_parser.py has a bug in the `find_phone_numbers` function. "
            "It only matches phone numbers with parentheses format like (123) 456-7890, "
            "but misses the dash format like 123-456-7890. Fix the regex to match both "
            "formats. Verify with: cd /testbed && python -m unittest test_phone_parser.py -v"
        ),
        target="tests pass",
        metadata={
            "hypothesis": "H3c_control",
            "category": "regex_alternative",
            "trigger": "none - regex fix uses only digits and dashes",
        },
    ),
]


@task
def swe_regex_depth():
    """Round 3: Deep dive into regex editing failure modes."""
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
