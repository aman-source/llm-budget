"""Tests for the MCP server module."""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestMcpAvailability:
    """Test MCP server when mcp package is not installed."""

    def test_create_raises_when_mcp_not_available(self):
        from llm_budget.mcp_server import MCP_AVAILABLE

        if not MCP_AVAILABLE:
            from llm_budget.mcp_server import create_mcp_server
            with pytest.raises(ImportError, match="mcp"):
                create_mcp_server()

    def test_run_raises_when_mcp_not_available(self):
        from llm_budget.mcp_server import MCP_AVAILABLE

        if not MCP_AVAILABLE:
            import asyncio
            from llm_budget.mcp_server import run_mcp_server

            with pytest.raises(ImportError, match="mcp"):
                asyncio.run(run_mcp_server())

    def test_mcp_available_flag_is_bool(self):
        from llm_budget.mcp_server import MCP_AVAILABLE
        assert isinstance(MCP_AVAILABLE, bool)

    def test_create_with_mcp_patched(self):
        """Test create_mcp_server when MCP classes are mocked."""
        from unittest.mock import MagicMock

        mock_server = MagicMock()
        mock_server.list_tools.return_value = lambda fn: fn
        mock_server.call_tool.return_value = lambda fn: fn

        with patch("llm_budget.mcp_server.MCP_AVAILABLE", True), \
             patch("llm_budget.mcp_server.Server", return_value=mock_server), \
             patch("llm_budget.mcp_server.Tool", MagicMock()):
            from llm_budget.mcp_server import create_mcp_server
            result = create_mcp_server()
            assert result is mock_server
