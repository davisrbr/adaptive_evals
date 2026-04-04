"""
Coding Agent Evaluation

An agentic eval where the model must solve programming challenges using
bash and python tools. Tests the model's ability to write, debug, and
test code in a sandboxed environment.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import scorer, accuracy, Score, Scorer, Target, CORRECT, INCORRECT
from inspect_ai.tool import bash, python


DEFAULT_SYSTEM_PROMPT = """\
You are an expert programmer. Solve the given programming challenge.
Write your solution, test it, and make sure it passes all test cases.
Write the final answer to /tmp/solution_output.txt.
"""

_CHALLENGES = [
    Sample(
        input=(
            "Write a Python function that finds the longest palindromic substring "
            "in a given string. Test it with the inputs: 'babad' (expected: 'bab' or 'aba'), "
            "'cbbd' (expected: 'bb'), and 'racecar' (expected: 'racecar'). "
            "Write 'PASS' to /tmp/solution_output.txt if all tests pass, 'FAIL' otherwise."
        ),
        target="PASS",
        id="coding_palindrome",
        metadata={"category": "algorithms", "difficulty": "medium"},
    ),
    Sample(
        input=(
            "Write a Python script that reads a CSV file at /data/input.csv, "
            "groups rows by the 'category' column, calculates the mean of the 'value' "
            "column for each group, and writes the results to /tmp/solution_output.txt "
            "as 'category: mean_value' lines, sorted alphabetically by category."
        ),
        target="A: 15.0\nB: 25.0\nC: 35.0",
        id="coding_csv",
        files={
            "/data/input.csv": "category,value\nA,10\nA,20\nB,20\nB,30\nC,30\nC,40\n",
        },
        metadata={"category": "data_processing", "difficulty": "easy"},
    ),
    Sample(
        input=(
            "There is a buggy Python script at /data/buggy.py. It's supposed to "
            "implement a binary search function, but it has bugs. Fix the bugs and "
            "verify the fixed version works correctly. Write 'FIXED' to "
            "/tmp/solution_output.txt when done."
        ),
        target="FIXED",
        id="coding_debug",
        files={
            "/data/buggy.py": (
                "def binary_search(arr, target):\n"
                "    left, right = 0, len(arr)\n"  # Bug: should be len(arr) - 1
                "    while left < right:\n"  # Bug: should be left <= right
                "        mid = (left + right) / 2\n"  # Bug: should be //
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid\n"  # Bug: should be mid + 1
                "        else:\n"
                "            right = mid\n"  # Bug: should be mid - 1
                "    return -1\n"
            ),
        },
        metadata={"category": "debugging", "difficulty": "medium"},
    ),
    Sample(
        input=(
            "Implement a simple REST API using Python's built-in http.server module "
            "that supports GET /items (list items), POST /items (add item), and "
            "GET /items/<id> (get specific item). Start the server on port 9090, "
            "then test it with curl commands. Write 'API_WORKS' to "
            "/tmp/solution_output.txt if all endpoints work correctly."
        ),
        target="API_WORKS",
        id="coding_api",
        metadata={"category": "web", "difficulty": "hard"},
    ),
]


@scorer(metrics=[accuracy()])
def output_file_scorer() -> Scorer:
    """Score based on the contents of /tmp/solution_output.txt."""
    async def score(state, target: Target) -> Score:
        # Check agent's output for the solution
        completion = state.output.completion
        # Also try to check via the last tool output that read the file
        for msg in reversed(state.messages):
            if msg.role == "tool":
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                if target.text in content:
                    return Score(value=CORRECT, explanation="Target found in tool output")

        if target.text in completion:
            return Score(value=CORRECT, explanation="Target found in agent output")

        return Score(value=INCORRECT, explanation=f"Expected '{target.text}' not found")
    return score


@task
def coding_challenge(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_messages: int = 40,
    max_tokens: int = 80000,
    categories: str = "",
    difficulty: str = "",
):
    """
    Coding challenge agent evaluation.

    Args:
        system_prompt: System prompt for the agent
        max_messages: Maximum agent messages
        max_tokens: Maximum tokens for agent
        categories: Comma-separated category filter
        difficulty: Difficulty filter: 'easy', 'medium', 'hard'
    """
    samples = list(_CHALLENGES)

    if categories:
        cats = set(c.strip() for c in categories.split(","))
        samples = [s for s in samples if s.metadata.get("category") in cats]

    if difficulty:
        samples = [s for s in samples if s.metadata.get("difficulty") == difficulty]

    return Task(
        dataset=MemoryDataset(samples),
        agent=react(
            tools=[bash(timeout=60), python(timeout=60)],
            prompt=system_prompt,
        ),
        scorer=output_file_scorer(),
        sandbox="docker",
        message_limit=max_messages,
        token_limit=max_tokens,
    )
