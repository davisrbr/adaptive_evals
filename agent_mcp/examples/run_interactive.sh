#!/bin/bash
# Example: Run the adaptive eval agent interactively with Claude Code
#
# Prerequisites:
#   pip install -e ".[agent-mcp]"
#   # Set API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
#
# This sets up the MCP server config and launches Claude Code.
# Claude Code will have access to all the eval tools via MCP.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# Setup MCP config and CLAUDE.md
python -m agent_mcp.harness.claude_code_config setup .

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start Claude Code with:"
echo "  cd $PROJECT_DIR && claude"
echo ""
echo "Then try prompts like:"
echo '  "List available evaluation tasks and run the cybersec CTF eval on claude-sonnet-4-20250514"'
echo '  "Analyze the failures from the last eval run and design a follow-up evaluation"'
echo '  "Compare the baseline and modified eval runs and summarize your findings"'
echo ""
