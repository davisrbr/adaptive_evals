"""
Research Agent Evaluation

An agentic eval where the model must answer research questions by gathering
information, synthesizing findings, and producing well-supported answers.
Tests information retrieval, reasoning, and synthesis capabilities.
"""

from inspect_ai import Task, task
from inspect_ai.agent import react, AgentLimits
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.tool import bash, python, web_search


DEFAULT_SYSTEM_PROMPT = """\
You are a thorough research assistant. Answer the given research question by:
1. Breaking down the question into sub-questions
2. Gathering relevant information using available tools
3. Synthesizing your findings into a clear, well-supported answer
4. Citing specific evidence for your claims

Be precise and factual. If you're uncertain about something, say so.
"""

_QUESTIONS = [
    Sample(
        input=(
            "What are the key differences between transformer-based and "
            "state-space model architectures for sequence modeling? Compare "
            "their computational complexity, memory requirements, and "
            "performance characteristics."
        ),
        target=(
            "Transformers use self-attention with O(n^2) complexity in sequence length, "
            "while state-space models (like Mamba, S4) achieve O(n) or O(n log n) complexity. "
            "Transformers require O(n^2) memory for attention, SSMs use O(n) memory. "
            "Transformers excel at tasks requiring global context, SSMs are better for "
            "very long sequences. Modern SSMs approach transformer quality on many benchmarks."
        ),
        id="research_arch_comparison",
        metadata={"category": "ml_architecture", "difficulty": "medium"},
    ),
    Sample(
        input=(
            "Analyze the trade-offs between different approaches to AI safety "
            "evaluation: red-teaming, automated benchmarks, and formal verification. "
            "What are the strengths and limitations of each approach?"
        ),
        target=(
            "Red-teaming: strengths include finding novel failure modes, limitations include "
            "scalability and coverage. Automated benchmarks: strengths include reproducibility "
            "and scalability, limitations include Goodhart's law and narrow coverage. "
            "Formal verification: strengths include mathematical guarantees, limitations include "
            "scalability to complex systems and specification difficulty."
        ),
        id="research_safety_eval",
        metadata={"category": "ai_safety", "difficulty": "hard"},
    ),
    Sample(
        input=(
            "Write a Python script that fetches the current Bitcoin price from a "
            "public API, calculates its 24-hour change percentage, and formats the "
            "result as a brief report. Use available tools to accomplish this."
        ),
        target=(
            "A report showing the current Bitcoin price and 24-hour change percentage, "
            "fetched from a real API endpoint."
        ),
        id="research_practical",
        metadata={"category": "practical", "difficulty": "easy"},
    ),
]


@task
def research_qa(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_messages: int = 50,
    max_tokens: int = 100000,
    categories: str = "",
    use_web_search: bool = True,
    grader_model: str = "",
):
    """
    Research question-answering agent evaluation.

    Args:
        system_prompt: System prompt for the agent
        max_messages: Maximum agent messages
        max_tokens: Maximum tokens for agent
        categories: Comma-separated category filter
        use_web_search: Whether to give the agent web search capability
        grader_model: Model to use for grading (default: eval model)
    """
    samples = list(_QUESTIONS)
    if categories:
        cats = set(c.strip() for c in categories.split(","))
        samples = [s for s in samples if s.metadata.get("category") in cats]

    tools = [bash(timeout=30), python(timeout=30)]
    if use_web_search:
        tools.append(web_search())

    scorer_kwargs = {}
    if grader_model:
        scorer_kwargs["model"] = grader_model

    return Task(
        dataset=MemoryDataset(samples),
        agent=react(
            tools=tools,
            system_prompt=system_prompt,
            limits=AgentLimits(
                max_messages=max_messages,
                max_tokens=max_tokens,
            ),
        ),
        scorer=model_graded_qa(**scorer_kwargs),
    )
