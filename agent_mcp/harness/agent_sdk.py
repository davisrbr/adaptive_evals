"""
Claude Agent SDK Integration

Build programmatic adaptive evaluation pipelines using the Claude Agent SDK.
This provides a Python-native way to run the adaptive eval agent, with full
control over the loop, state management, and integration with other systems.

Usage:
    from agent_mcp.harness.agent_sdk import AdaptiveEvalAgent

    agent = AdaptiveEvalAgent(
        model="claude-sonnet-4-20250514",
        eval_task="agent_mcp/tasks/cybersec_agent.py",
        eval_model="anthropic/claude-sonnet-4-20250514",
    )
    results = await agent.run_adaptive_loop(iterations=3)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import Agent, AgentConfig, MCPServerConfig


@dataclass
class EvalIteration:
    """Results from a single iteration of the adaptive eval loop."""
    iteration: int
    log_path: str
    summary: dict[str, Any]
    failures: dict[str, Any]
    patterns: dict[str, Any]
    modifications_applied: dict[str, str] = field(default_factory=dict)


@dataclass
class AdaptiveEvalResult:
    """Complete results from an adaptive evaluation run."""
    eval_task: str
    eval_model: str
    iterations: list[EvalIteration]
    final_report: str = ""

    @property
    def baseline_accuracy(self) -> str:
        if self.iterations:
            return self.iterations[0].summary.get("accuracy", "N/A")
        return "N/A"

    @property
    def final_accuracy(self) -> str:
        if self.iterations:
            return self.iterations[-1].summary.get("accuracy", "N/A")
        return "N/A"

    def to_dict(self) -> dict:
        return {
            "eval_task": self.eval_task,
            "eval_model": self.eval_model,
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": self.final_accuracy,
            "num_iterations": len(self.iterations),
            "iterations": [
                {
                    "iteration": it.iteration,
                    "log_path": it.log_path,
                    "summary": it.summary,
                    "modifications": it.modifications_applied,
                }
                for it in self.iterations
            ],
            "final_report": self.final_report,
        }


class AdaptiveEvalAgent:
    """
    Programmatic adaptive eval agent using the Claude Agent SDK.

    Wraps the Claude Agent SDK to create an agent that uses the adaptive eval
    MCP server tools to run evaluations, analyze results, and iterate.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        eval_task: str = "",
        eval_model: str = "anthropic/claude-sonnet-4-20250514",
        project_dir: str = ".",
        log_dir: str = "./logs",
        max_turns_per_iteration: int = 20,
        system_prompt: str = "",
    ):
        self.model = model
        self.eval_task = eval_task
        self.eval_model = eval_model
        self.project_dir = project_dir
        self.log_dir = log_dir
        self.max_turns_per_iteration = max_turns_per_iteration

        from .claude_code_config import ADAPTIVE_EVAL_SYSTEM_PROMPT
        self.system_prompt = system_prompt or ADAPTIVE_EVAL_SYSTEM_PROMPT

        self._mcp_config = MCPServerConfig(
            name="adaptive-eval-agent",
            command="python",
            args=["-m", "agent_mcp.server.main"],
            env={"INSPECT_LOG_DIR": log_dir},
        )

    def _create_agent(self, prompt: str) -> Agent:
        """Create a Claude Agent SDK agent instance."""
        return Agent(
            config=AgentConfig(
                model=self.model,
                system_prompt=self.system_prompt,
                max_turns=self.max_turns_per_iteration,
                mcp_servers=[self._mcp_config],
            ),
            prompt=prompt,
        )

    async def run_baseline(self) -> EvalIteration:
        """Run the initial baseline evaluation."""
        prompt = (
            f"Run a baseline evaluation of model `{self.eval_model}` using task `{self.eval_task}`. "
            f"After the eval completes, provide a summary using `summarize_eval`, analyze any "
            f"failures with `analyze_failures`, and extract agent patterns with `extract_agent_patterns`. "
            f"Return the results as JSON with keys: log_path, summary, failures, patterns."
        )

        agent = self._create_agent(prompt)
        result = await agent.run()

        return self._parse_iteration_result(0, result)

    async def run_adaptive_iteration(
        self,
        iteration: int,
        previous: EvalIteration,
        focus: str = "",
    ) -> EvalIteration:
        """Run an adaptive iteration based on previous results."""
        prompt = (
            f"Based on the previous evaluation at `{previous.log_path}`, the model "
            f"achieved {previous.summary.get('accuracy', 'unknown')} accuracy.\n\n"
            f"Key failures: {json.dumps(previous.failures.get('failure_categories', {}), indent=2)}\n\n"
        )

        if focus:
            prompt += f"Focus on: {focus}\n\n"

        prompt += (
            f"Design and run a follow-up evaluation that probes the identified weaknesses more deeply. "
            f"You can modify the system prompt, change task parameters, or create a task variant. "
            f"Then analyze the new results and compare with the baseline using `diff_eval_runs`. "
            f"Return results as JSON with keys: log_path, summary, failures, patterns, modifications_applied."
        )

        agent = self._create_agent(prompt)
        result = await agent.run()

        return self._parse_iteration_result(iteration, result)

    async def run_adaptive_loop(
        self,
        iterations: int = 3,
        focus: str = "",
    ) -> AdaptiveEvalResult:
        """Run the full adaptive evaluation loop."""
        result = AdaptiveEvalResult(
            eval_task=self.eval_task,
            eval_model=self.eval_model,
            iterations=[],
        )

        # Run baseline
        baseline = await self.run_baseline()
        result.iterations.append(baseline)

        # Run adaptive iterations
        previous = baseline
        for i in range(1, iterations):
            iteration = await self.run_adaptive_iteration(i, previous, focus)
            result.iterations.append(iteration)
            previous = iteration

        # Generate final report
        result.final_report = await self._generate_report(result)

        return result

    async def _generate_report(self, result: AdaptiveEvalResult) -> str:
        """Generate a final summary report."""
        log_paths = [it.log_path for it in result.iterations if it.log_path]

        prompt = (
            f"Generate a final report for an adaptive evaluation of `{self.eval_model}` "
            f"on task `{self.eval_task}`. There were {len(result.iterations)} iterations.\n\n"
        )

        for it in result.iterations:
            prompt += (
                f"Iteration {it.iteration}: accuracy={it.summary.get('accuracy', 'N/A')}, "
                f"modifications={it.modifications_applied}\n"
            )

        if len(log_paths) >= 2:
            prompt += (
                f"\nUse `diff_eval_runs` to compare the first and last runs "
                f"(`{log_paths[0]}` vs `{log_paths[-1]}`). "
            )

        prompt += (
            "\nProvide a concise report covering: overall findings, key failure patterns, "
            "how the model responded to adaptations, and recommendations."
        )

        agent = self._create_agent(prompt)
        report_result = await agent.run()
        return str(report_result.output) if hasattr(report_result, "output") else str(report_result)

    def _parse_iteration_result(self, iteration: int, agent_result) -> EvalIteration:
        """Parse agent output into an EvalIteration."""
        output = str(agent_result.output) if hasattr(agent_result, "output") else str(agent_result)

        # Try to extract JSON from the output
        try:
            # Find JSON in the output
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(output[start:end])
                return EvalIteration(
                    iteration=iteration,
                    log_path=data.get("log_path", ""),
                    summary=data.get("summary", {}),
                    failures=data.get("failures", {}),
                    patterns=data.get("patterns", {}),
                    modifications_applied=data.get("modifications_applied", {}),
                )
        except json.JSONDecodeError:
            pass

        # Fallback: return raw output
        return EvalIteration(
            iteration=iteration,
            log_path="",
            summary={"raw_output": output[:500]},
            failures={},
            patterns={},
        )


async def main():
    """Example usage of the adaptive eval agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run adaptive evals via Agent SDK")
    parser.add_argument("--task", required=True, help="Task to evaluate")
    parser.add_argument("--eval-model", default="anthropic/claude-sonnet-4-20250514",
                       help="Model to evaluate")
    parser.add_argument("--agent-model", default="claude-sonnet-4-20250514",
                       help="Claude model for the agent")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--focus", default="")
    parser.add_argument("--output", default="")

    args = parser.parse_args()

    agent = AdaptiveEvalAgent(
        model=args.agent_model,
        eval_task=args.task,
        eval_model=args.eval_model,
    )

    results = await agent.run_adaptive_loop(
        iterations=args.iterations,
        focus=args.focus,
    )

    output = results.to_dict()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to: {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
