"""
Claude Code MCP Configuration Generator

Generates the .mcp.json configuration and system prompt needed to use
the adaptive eval MCP server with Claude Code (interactive mode).

Usage:
    python -m agent_mcp.harness.claude_code_config setup
    python -m agent_mcp.harness.claude_code_config print-prompt
"""

import json
import os
import sys
from pathlib import Path

# The MCP server configuration for Claude Code's .mcp.json
MCP_SERVER_CONFIG = {
    "adaptive-eval-agent": {
        "command": "python",
        "args": ["-m", "agent_mcp.server.main"],
        "env": {
            "INSPECT_LOG_DIR": "./logs",
        },
    }
}

# System prompt to guide Claude Code when used as the adaptive eval agent
ADAPTIVE_EVAL_SYSTEM_PROMPT = """\
You are an adaptive evaluation agent. Your job is to systematically evaluate
AI models using inspect-ai, analyze their performance, identify weaknesses,
and iteratively improve the evaluation to probe deeper.

## Your Workflow

1. **Discover**: Use `list_eval_logs` and `list_tasks` to understand what
   evaluations are available and what has been run before.

2. **Run**: Use `run_eval` to execute evaluations. Start with a baseline run
   to establish initial performance.

3. **Analyze**: After each run:
   - Use `summarize_eval` for a high-level overview
   - Use `analyze_failures` to understand what went wrong
   - Use `extract_agent_patterns` to see behavioral patterns
   - Use `scan_transcripts` to detect issues like eval awareness or refusals

4. **Iterate**: Based on your analysis:
   - Use `run_eval_with_modified_prompt` to test prompt changes
   - Use `run_eval_subset` to re-run only failures
   - Use `create_task_variant` to create modified versions of tasks
   - Use `diff_eval_runs` to measure the impact of changes

5. **Report**: Summarize your findings, including:
   - Overall model capabilities and limitations
   - Specific failure patterns and their root causes
   - How the model responds to different prompts/configurations
   - Recommendations for further evaluation

## Key Principles

- **Be systematic**: Don't just run evals randomly. Form hypotheses about model
  behavior and design experiments to test them.
- **Be thorough**: Look at individual sample transcripts, not just aggregate scores.
- **Be adaptive**: If you find an interesting failure pattern, design follow-up
  evals that probe it more deeply.
- **Track your progress**: Keep notes on what you've tried and what you've learned.
"""


def generate_mcp_config(project_dir: str = ".") -> dict:
    """Generate complete .mcp.json content for a project."""
    config = {"mcpServers": dict(MCP_SERVER_CONFIG)}

    # Add Inspect Scout MCP if available
    # (Scout doesn't have MCP natively, but we wrap it in our server)

    return config


def write_mcp_config(project_dir: str = "."):
    """Write .mcp.json to the project directory."""
    config = generate_mcp_config(project_dir)
    config_path = Path(project_dir) / ".mcp.json"

    # Merge with existing config if present
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        existing.setdefault("mcpServers", {}).update(config["mcpServers"])
        config = existing

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote MCP config to: {config_path}")
    return config_path


def write_claude_md(project_dir: str = "."):
    """Write/update CLAUDE.md with the adaptive eval system prompt."""
    claude_md_path = Path(project_dir) / "CLAUDE.md"

    content = f"""\
# Adaptive Eval Agent

{ADAPTIVE_EVAL_SYSTEM_PROMPT}

## Available MCP Tools

The `adaptive-eval-agent` MCP server provides these tool groups:

### Discovery
- `list_eval_logs` - List previous eval runs
- `read_eval_log` - Read eval log details
- `read_eval_samples` - Read individual sample transcripts
- `compare_eval_runs` - Compare metrics across runs

### Execution
- `run_eval` - Run an inspect-ai evaluation
- `run_eval_with_modified_prompt` - Re-run with changes
- `run_eval_subset` - Re-run specific samples
- `get_eval_status` - Check running eval status

### Analysis
- `analyze_failures` - Categorize failure patterns
- `summarize_eval` - High-level run summary
- `scan_transcripts` - Run Scout scanners on transcripts
- `extract_agent_patterns` - Extract behavioral patterns
- `diff_eval_runs` - Compare two runs in detail

### Configuration
- `list_tasks` - List available eval tasks
- `create_task_variant` - Create modified task versions
- `write_task` - Write new eval tasks
- `read_task_source` - Read task source code
- `get_inspect_docs` - Get inspect-ai documentation

## Example Tasks

- `agent_mcp/tasks/cybersec_agent.py` - CTF challenges (easy/medium/hard)
- `agent_mcp/tasks/coding_agent.py` - Programming challenges
- `agent_mcp/tasks/research_agent.py` - Research QA with web search
"""

    with open(claude_md_path, "w") as f:
        f.write(content)

    print(f"Wrote CLAUDE.md to: {claude_md_path}")
    return claude_md_path


def setup(project_dir: str = "."):
    """Full setup: write .mcp.json and CLAUDE.md."""
    write_mcp_config(project_dir)
    write_claude_md(project_dir)
    print("\nSetup complete! Start Claude Code in this directory to use the adaptive eval agent.")
    print("  claude")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "print-prompt":
        print(ADAPTIVE_EVAL_SYSTEM_PROMPT)
    elif len(sys.argv) > 1 and sys.argv[1] == "setup":
        project_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        setup(project_dir)
    else:
        print("Usage:")
        print("  python -m agent_mcp.harness.claude_code_config setup [project_dir]")
        print("  python -m agent_mcp.harness.claude_code_config print-prompt")
