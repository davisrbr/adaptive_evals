"""
Example: Run adaptive evals programmatically via the Agent SDK.

This shows how to use the AdaptiveEvalAgent class to run a full
adaptive evaluation loop with custom control over each iteration.
"""

import asyncio
import json
from pathlib import Path

# Ensure the project root is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_mcp.harness.agent_sdk import AdaptiveEvalAgent, AdaptiveEvalResult


async def run_cybersec_eval():
    """Run an adaptive eval on the cybersec CTF task."""
    agent = AdaptiveEvalAgent(
        model="claude-sonnet-4-20250514",
        eval_task="agent_mcp/tasks/cybersec_agent.py",
        eval_model="anthropic/claude-sonnet-4-20250514",
        log_dir="./logs/agent_sdk_demo",
        max_turns_per_iteration=25,
    )

    # Option 1: Run the full loop automatically
    results = await agent.run_adaptive_loop(
        iterations=3,
        focus="sandbox interaction and tool error recovery",
    )

    print(f"\nBaseline accuracy: {results.baseline_accuracy}")
    print(f"Final accuracy: {results.final_accuracy}")
    print(f"\n--- Final Report ---\n{results.final_report}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/agent_sdk_cybersec.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)


async def run_custom_loop():
    """Run an adaptive eval with custom iteration control."""
    agent = AdaptiveEvalAgent(
        model="claude-sonnet-4-20250514",
        eval_task="agent_mcp/tasks/coding_agent.py",
        eval_model="anthropic/claude-sonnet-4-20250514",
        log_dir="./logs/agent_sdk_custom",
    )

    # Run baseline
    print("Running baseline...")
    baseline = await agent.run_baseline()
    print(f"Baseline: {baseline.summary.get('accuracy', 'N/A')}")

    # Custom iteration: focus on debugging challenges
    print("\nRunning targeted iteration on debugging...")
    iteration1 = await agent.run_adaptive_iteration(
        iteration=1,
        previous=baseline,
        focus="debugging category challenges - improve the agent's ability to identify and fix bugs",
    )
    print(f"Iteration 1: {iteration1.summary.get('accuracy', 'N/A')}")

    # Another iteration: focus on the hardest remaining failures
    print("\nRunning iteration focused on remaining failures...")
    iteration2 = await agent.run_adaptive_iteration(
        iteration=2,
        previous=iteration1,
        focus="the most persistent failures that didn't improve in the previous iteration",
    )
    print(f"Iteration 2: {iteration2.summary.get('accuracy', 'N/A')}")


async def compare_models():
    """Compare multiple models on the same task."""
    models_to_eval = [
        "anthropic/claude-sonnet-4-20250514",
        "openai/gpt-4o",
    ]

    all_results: list[AdaptiveEvalResult] = []

    for eval_model in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {eval_model}")
        print(f"{'='*60}")

        agent = AdaptiveEvalAgent(
            model="claude-sonnet-4-20250514",  # Orchestrator model
            eval_task="agent_mcp/tasks/research_agent.py",
            eval_model=eval_model,
            log_dir=f"./logs/comparison/{eval_model.replace('/', '_')}",
        )

        results = await agent.run_adaptive_loop(iterations=2)
        all_results.append(results)

        print(f"  Baseline: {results.baseline_accuracy}")
        print(f"  Final: {results.final_accuracy}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r.eval_model}: {r.baseline_accuracy} -> {r.final_accuracy}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=["cybersec", "custom", "compare"],
                       default="cybersec")
    args = parser.parse_args()

    if args.example == "cybersec":
        asyncio.run(run_cybersec_eval())
    elif args.example == "custom":
        asyncio.run(run_custom_loop())
    elif args.example == "compare":
        asyncio.run(compare_models())
