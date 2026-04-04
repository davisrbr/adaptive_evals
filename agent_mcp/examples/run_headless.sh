#!/bin/bash
# Example: Run the adaptive eval agent in headless mode
#
# This runs a fully automated adaptive evaluation loop using
# Claude Code headless mode. Claude Code acts as the orchestrating
# agent, using the MCP server tools to run evals and iterate.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# Ensure MCP config exists
python -m agent_mcp.harness.claude_code_config setup .

# Run headless adaptive eval on the cybersec CTF task
python -m agent_mcp.harness.headless \
    --task agent_mcp/tasks/cybersec_agent.py \
    --model anthropic/claude-sonnet-4-20250514 \
    --iterations 3 \
    --focus "tool usage errors and sandbox interaction failures" \
    --output results/headless_cybersec_run.json \
    --verbose

echo ""
echo "Results saved to: results/headless_cybersec_run.json"
