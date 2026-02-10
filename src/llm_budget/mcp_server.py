"""
Model Context Protocol (MCP) server for llm-budget tools.

Exposes estimate_cost, compare_models, and check_budget as MCP tools.

Run as:
    python -m llm_budget.mcp_server
    llm-budget serve-mcp
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def create_mcp_server() -> Any:
    """Create and configure an MCP server instance.

    Returns:
        Configured MCP Server instance.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install with: pip install llm-budget[mcp]"
        )

    from .tools import cost_tools_openai, cost_context

    server = Server("llm-budget")

    # Build MCP Tool list from OpenAI schemas
    openai_tools = cost_tools_openai()
    mcp_tools: List[Tool] = []
    for tool_def in openai_tools:
        func = tool_def["function"]
        mcp_tools.append(
            Tool(
                name=func["name"],
                description=func["description"],
                inputSchema=func["parameters"],
            )
        )

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return mcp_tools

    @server.call_tool()
    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        try:
            result = cost_context({"name": name, "arguments": arguments})
            return [TextContent(type="text", text=result)]
        except Exception as exc:
            error_msg = f"Error executing {name}: {exc}"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]

    return server


async def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install with: pip install llm-budget[mcp]"
        )

    server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for ``python -m llm_budget.mcp_server``."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
