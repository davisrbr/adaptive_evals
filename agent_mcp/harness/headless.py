"""
Claude Code Headless Wrapper

Run adaptive evaluation loops using Claude Code in headless mode.
Claude Code acts as the orchestrating agent, using the MCP server tools
to run evals, analyze results, and iterate.

Usage:
    # Run a single adaptive evaluation loop
    python -m agent_mcp.harness.headless \
        --task agent_mcp/tasks/cybersec_agent.py \
        --model anthropic/claude-sonnet-4-20250514 \
        --iterations 3

    # Run with a custom analysis prompt
    python -m agent_mcp.harness.headless \
        --prompt "Evaluate the model on cybersec CTF tasks, focusing on ..."

    # Resume from previous results
    python -m agent_mcp.harness.headless \
        --resume-from ./logs/previous_run.json \
        --focus "failures"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .claude_code_config import ADAPTIVE_EVAL_SYSTEM_PROMPT


def build_headless_prompt(
    task: str,
    model: str,
    iterations: int = 3,
    focus: str = "",
    resume_from: str = "",
    custom_prompt: str = "",
) -> str:
    """Build the prompt for Claude Code headless mode."""

    if custom_prompt:
        return custom_prompt

    prompt_parts = [ADAPTIVE_EVAL_SYSTEM_PROMPT, "\n## Your Current Assignment\n"]

    if resume_from:
        prompt_parts.append(
            f"Resume analysis from a previous eval run at: {resume_from}\n"
            f"Use `read_eval_log` to examine the previous results, then continue "
            f"with adaptive evaluation.\n"
        )
        if focus:
            prompt_parts.append(f"Focus your analysis on: {focus}\n")
    else:
        prompt_parts.append(
            f"Run an adaptive evaluation of model `{model}` using the task at `{task}`.\n\n"
            f"Perform {iterations} iteration(s) of the adaptive loop:\n"
            f"1. Run the baseline evaluation\n"
            f"2. Analyze the results thoroughly\n"
            f"3. Identify the most interesting failure patterns\n"
            f"4. Design and run follow-up evaluations that probe those weaknesses\n"
            f"5. Compare results across iterations and summarize findings\n"
        )
        if focus:
            prompt_parts.append(f"\nFocus particularly on: {focus}\n")

    prompt_parts.append(
        "\nProvide a final summary report of your findings, including:\n"
        "- Overall performance metrics across iterations\n"
        "- Key failure patterns identified\n"
        "- How performance changed with different configurations\n"
        "- Recommendations for further evaluation\n"
    )

    return "\n".join(prompt_parts)


def run_headless(
    prompt: str,
    project_dir: str = ".",
    output_file: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Run Claude Code in headless mode with the adaptive eval MCP server.

    Returns a dict with the session results.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--model", model,
        "--max-turns", str(max_turns),
        "--prompt", prompt,
    ]

    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["INSPECT_LOG_DIR"] = str(Path(project_dir) / "logs")

    if verbose:
        print(f"Running: {' '.join(cmd[:6])}...", file=sys.stderr)

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )

    output = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    # Try to parse JSON output
    try:
        output["parsed"] = json.loads(result.stdout)
    except json.JSONDecodeError:
        output["parsed"] = None

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"Results written to: {output_file}", file=sys.stderr)

    return output


def run_headless_streaming(
    prompt: str,
    project_dir: str = ".",
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 50,
) -> subprocess.Popen:
    """
    Run Claude Code in headless mode with streaming output.
    Returns the Popen object for the caller to manage.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--model", model,
        "--max-turns", str(max_turns),
        "--prompt", prompt,
    ]

    env = os.environ.copy()
    env["INSPECT_LOG_DIR"] = str(Path(project_dir) / "logs")

    return subprocess.Popen(
        cmd,
        cwd=project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive evaluations using Claude Code headless mode"
    )
    parser.add_argument("--task", help="Task file or module path to evaluate")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514",
                       help="Model to evaluate")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of adaptive iterations")
    parser.add_argument("--focus", default="",
                       help="Focus area for analysis")
    parser.add_argument("--resume-from", default="",
                       help="Previous eval log to resume from")
    parser.add_argument("--prompt", default="",
                       help="Custom prompt (overrides other options)")
    parser.add_argument("--output", default="",
                       help="Output file for results")
    parser.add_argument("--project-dir", default=".",
                       help="Project directory")
    parser.add_argument("--claude-model", default="claude-sonnet-4-20250514",
                       help="Claude model for the harness agent")
    parser.add_argument("--max-turns", type=int, default=50,
                       help="Max conversation turns")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.task and not args.prompt and not args.resume_from:
        parser.error("Must specify --task, --prompt, or --resume-from")

    prompt = build_headless_prompt(
        task=args.task or "",
        model=args.model,
        iterations=args.iterations,
        focus=args.focus,
        resume_from=args.resume_from,
        custom_prompt=args.prompt,
    )

    if args.verbose:
        print("=" * 60, file=sys.stderr)
        print("PROMPT:", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    result = run_headless(
        prompt=prompt,
        project_dir=args.project_dir,
        output_file=args.output,
        model=args.claude_model,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    if result["returncode"] != 0:
        print(f"Claude Code exited with code {result['returncode']}", file=sys.stderr)
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr)
        sys.exit(1)

    # Print the agent's final output
    if result.get("parsed"):
        print(json.dumps(result["parsed"], indent=2))
    else:
        print(result["stdout"])


if __name__ == "__main__":
    main()
