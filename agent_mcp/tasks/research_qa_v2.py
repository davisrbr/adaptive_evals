"""
Research QA v2 - Improved evaluation that actually requires tool use.

Improvements over v1:
- Questions require computation/code execution, not just knowledge recall
- Uses exact-match scoring where possible (not LLM grader)
- Mix of difficulty levels to get signal across capability range
"""

import hashlib

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import scorer, accuracy, Score, Scorer, Target, CORRECT, INCORRECT
from inspect_ai.tool import bash, python


@scorer(metrics=[accuracy()])
def answer_contains_scorer() -> Scorer:
    """Score by checking if specific required strings appear in the output."""
    async def score(state, target: Target) -> Score:
        # Collect all text from the conversation
        full_text = state.output.completion.lower()
        for msg in state.messages:
            if msg.role == "tool":
                full_text += " " + str(getattr(msg, "content", "")).lower()
            elif msg.role == "assistant":
                full_text += " " + str(getattr(msg, "content", "")).lower()

        required = [t.strip().lower() for t in target.text.split("|")]
        found = [r for r in required if r in full_text]

        if len(found) == len(required):
            return Score(value=CORRECT, explanation=f"All required terms found: {found}")
        elif len(found) > 0:
            return Score(value=INCORRECT, explanation=f"Partial: found {found}, missing {set(required)-set(found)}")
        return Score(value=INCORRECT, explanation=f"None of {required} found in output")
    return score


# Pre-compute targets
_HASH_TARGET = hashlib.sha256(b"adaptive evaluation").hexdigest()


def _primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


_P20 = _primes(20)
_PRIME_SUM_TARGET = str(sum(_P20[i] for i in range(0, 20, 2)))


_SAMPLES = [
    # EASY: Requires running code to compute
    Sample(
        input=(
            'Using Python, compute the SHA-256 hash of the string "adaptive evaluation" '
            "(encoded as UTF-8). Report the hex digest."
        ),
        target=_HASH_TARGET,
        id="compute_hash",
        metadata={"category": "computation", "difficulty": "easy", "requires_tools": True},
    ),
    # MEDIUM: Requires writing and executing multi-step code
    Sample(
        input=(
            "Write a Python program that generates the first 20 prime numbers, "
            "then computes the sum of primes at even indices (0-indexed: index 0, 2, 4, ...). "
            "Report the final sum as a single number."
        ),
        target=_PRIME_SUM_TARGET,
        id="prime_sum",
        metadata={"category": "computation", "difficulty": "medium", "requires_tools": True},
    ),
    # MEDIUM: Requires data analysis code
    Sample(
        input=(
            "Using Python, generate a list of 100 random integers between 1 and 1000 "
            "using random.seed(42). Then compute: (1) the mean, (2) the median, "
            "(3) the standard deviation. Report all three values rounded to 2 decimal places."
        ),
        target="mean|median|std",
        id="stats_computation",
        metadata={"category": "data_analysis", "difficulty": "medium", "requires_tools": True},
    ),
    # HARD: Requires multi-step reasoning AND code execution
    Sample(
        input=(
            "Using Python, implement a function that checks whether a given string of "
            'parentheses is balanced. Test it on these inputs: "(())", "(()", ")(", '
            '"((())())", and "". Report True/False for each, one per line.'
        ),
        target="true|false",
        id="balanced_parens",
        metadata={"category": "coding", "difficulty": "hard", "requires_tools": True},
    ),
    # EASY: Requires knowledge AND computation
    Sample(
        input=(
            "The Fibonacci sequence starts with F(0)=0, F(1)=1. "
            "Using Python, compute F(50) exactly (not approximately). Report the exact integer."
        ),
        target="12586269025",
        id="fibonacci_50",
        metadata={"category": "computation", "difficulty": "easy", "requires_tools": True},
    ),
]


@task
def research_qa_v2(
    system_prompt: str = "You are a capable assistant. Use the bash and python tools to solve problems. Always execute code rather than just describing it.",
    max_messages: int = 20,
    max_tokens: int = 100000,
    difficulty: str = "",
):
    """
    Research QA v2 - computation-focused eval requiring tool use.
    """
    samples = list(_SAMPLES)
    if difficulty:
        samples = [s for s in samples if s.metadata.get("difficulty") == difficulty]

    return Task(
        dataset=MemoryDataset(samples),
        agent=react(
            tools=[bash(timeout=30), python(timeout=30)],
            prompt=system_prompt,
        ),
        scorer=answer_contains_scorer(),
        message_limit=max_messages,
        token_limit=max_tokens,
    )
