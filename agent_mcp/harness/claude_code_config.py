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
You are an adaptive evaluation agent. Your job is to run evaluations, discover
failure patterns using Scout, and create new eval questions that test those
patterns more precisely. Each iteration must produce new eval samples.

## The Adaptive Loop

Each iteration has 5 steps. You are NOT done until step 5 is complete.

1. **Run**: Execute an evaluation with `run_eval`. On the first iteration,
   this is a baseline run. On later iterations, this runs the new eval you
   created in the previous step 5.

2. **Scan**: Run `scan_transcripts` on the eval log. The default scanner suite
   classifies every transcript by failure_mode, tool_usage_pattern,
   reasoning_quality, abstention_judgment, behavioral_tags,
   environment_vs_agent, and patch_quality. You can also pass
   `custom_questions` to scan for hypothesis-specific patterns.

3. **Hypothesize**: Look at the scanner distributions. What patterns appear?
   Form specific, testable hypotheses. Examples:
   - "The model calls get_stock_price for anything with a ticker-like name"
   - "output_misinterpretation is the top failure mode — the model writes
     correct patches but doesn't iterate when tests fail"
   - "3/8 samples hit the message limit — more turns might help"
   A good hypothesis is specific enough that you can design samples to
   confirm or reject it.

4. **Test**: Create eval samples that test each hypothesis. Use `write_task`
   or edit the task file directly. Include:
   - Samples designed to trigger the hypothesized failure (if it's real,
     these should fail)
   - Control samples (similar structure but without the trigger — these
     should pass)
   - Document which hypothesis each sample tests in the metadata

5. **Run the new eval**: Execute the expanded eval with `run_eval` and
   `scan_transcripts`. Compare results to the previous round. For each
   hypothesis, state whether it was confirmed, rejected, or refined.
   Then return to step 3 with the new scan results.

## Completion Criteria

The loop is complete when EITHER:
- You have confirmed a precise failure mode and created eval samples that
  reliably trigger it (not just "the model sometimes fails" — you need
  "the model fails specifically when X because Y")
- The user tells you to stop

The loop is NOT complete when:
- You have scan results but haven't created new eval samples yet
- You have hypotheses but haven't tested them
- You confirmed a hypothesis but haven't refined it into a more precise one

## Key Principles

- **Every round must produce new samples.** Analysis without new eval creation
  is incomplete.
- **Hypotheses should get more specific each round.** Round 1: "model over-calls
  tools." Round 2: "model over-calls financial tools." Round 3: "model treats
  any 3-letter uppercase string as a stock ticker."
- **Use controls.** When testing "does X cause failure?", also include samples
  where X is absent to make sure the model still passes.
- **Scanner results drive sample design.** Don't guess what to test next — look
  at the failure_mode and behavioral_tags distributions.
- **Custom scanners test custom hypotheses.** If you hypothesize "the model
  confuses bash and python tool selection", write a custom_question scanner
  for it before creating samples.
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
