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
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import tools_discovery
from . import tools_execution
from . import tools_analysis
from . import tools_configuration

logger = logging.getLogger(__name__)

# All tool modules that provide get_tools() and handle_call()
_TOOL_MODULES = [
    tools_discovery,
    tools_execution,
    tools_analysis,
    tools_configuration,
]


def create_server() -> Server:
    """Create and configure the MCP server with all tool groups."""
    server = Server("adaptive-eval-agent")

    @server.list_tools()
    async def list_all_tools():
        tools = []
        for module in _TOOL_MODULES:
            tools.extend(module.get_tools())
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        for module in _TOOL_MODULES:
            result = await module.handle_call(name, arguments)
            if result is not None:
                return result
        return []

    return server


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = create_server()
    logger.info("Starting adaptive-eval-agent MCP server")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
