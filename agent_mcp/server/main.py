"""
Adaptive Eval Agent MCP Server

An MCP server that gives Claude Code (or any MCP client) rich tools for running,
analyzing, and iterating on agentic inspect-ai evaluations.

Usage:
    python -m agent_mcp.server.main
    # or via the CLI entry point:
    adaptive-eval-mcp
"""

import asyncio
import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import run_stdio

from .tools_discovery import register_discovery_tools
from .tools_execution import register_execution_tools
from .tools_analysis import register_analysis_tools
from .tools_configuration import register_configuration_tools

logger = logging.getLogger(__name__)


def create_server() -> Server:
    """Create and configure the MCP server with all tool groups."""
    server = Server("adaptive-eval-agent")

    register_discovery_tools(server)
    register_execution_tools(server)
    register_analysis_tools(server)
    register_configuration_tools(server)

    return server


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = create_server()
    logger.info("Starting adaptive-eval-agent MCP server")
    await run_stdio(server)


if __name__ == "__main__":
    asyncio.run(main())
