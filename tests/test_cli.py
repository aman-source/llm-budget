"""Tests for the CLI module."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llm_budget.cli import cli
from llm_budget.tracker import Tracker


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


class TestStatus:
    def test_empty_status(self, runner, tracker):
        with patch("llm_budget.cli.get_tracker", return_value=tracker):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "LLM Budget Status" in result.output

    def test_with_spend(self, runner, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        with patch("llm_budget.cli.get_tracker", return_value=tracker):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "$" in result.output


class TestEstimate:
    def test_basic(self, runner):
        result = runner.invoke(cli, ["estimate", "Hello world", "-m", "gpt-4o"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "$" in result.output

    def test_with_output_tokens(self, runner):
        result = runner.invoke(
            cli, ["estimate", "Hello", "-m", "gpt-4o", "-o", "500"]
        )
        assert result.exit_code == 0


class TestCompare:
    def test_basic(self, runner):
        result = runner.invoke(
            cli, ["compare", "Hello world", "--models", "gpt-4o,gpt-4o-mini"]
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_multiple_models(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                "Explain quantum computing",
                "--models",
                "gpt-4o,gpt-4o-mini,deepseek-chat",
            ],
        )
        assert result.exit_code == 0


class TestHistory:
    def test_empty(self, runner, tracker):
        with patch("llm_budget.cli.get_tracker", return_value=tracker):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No history" in result.output

    def test_with_records(self, runner, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        with patch("llm_budget.cli.get_tracker", return_value=tracker):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output


class TestModels:
    def test_list_all(self, runner):
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_filter_provider(self, runner):
        result = runner.invoke(cli, ["models", "-p", "openai"])
        assert result.exit_code == 0
        assert "openai" in result.output


class TestVersion:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestUpdatePrices:
    def test_update_success(self, runner, tmp_path):
        """Test update-prices with mocked HTTP response and file system."""
        import json
        from pathlib import Path

        # Read the actual models.json to get current data
        real_models_path = Path(__file__).parent.parent / "src" / "llm_budget" / "models.json"
        models_data = json.loads(real_models_path.read_text())

        # Mock the HTTP response with data that matches tracked models
        mock_data = {
            "gpt-4o": {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.2e-05,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "litellm_provider": "openai",
            }
        }

        class MockResp:
            def read(self):
                return json.dumps(mock_data).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        # Copy models.json to tmp_path so we don't modify the real one
        tmp_models = tmp_path / "models.json"
        tmp_models.write_text(json.dumps(models_data))

        with patch("llm_budget.cli.urllib.request.urlopen", return_value=MockResp()), \
             patch("llm_budget.cli.Path", wraps=Path) as mock_path_cls:
            # We need to patch the specific Path(__file__).parent / "models.json"
            # Easier approach: just mock open to redirect file writes
            original_open = open

            def mock_open_fn(path, *args, **kwargs):
                path_str = str(path)
                if path_str.endswith("models.json") and "llm_budget" in path_str:
                    return original_open(str(tmp_models), *args, **kwargs)
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=mock_open_fn), \
                 patch("llm_budget.cli.get_registry") as mock_reg:
                mock_reg.return_value.reload.return_value = None
                result = runner.invoke(cli, ["update-prices"])

        assert "Fetching" in result.output
        assert result.exit_code == 0

    def test_update_network_error(self, runner):
        """Test update-prices handles network errors gracefully."""
        with patch("llm_budget.cli.urllib.request.urlopen", side_effect=Exception("network error")):
            result = runner.invoke(cli, ["update-prices"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestServeMcp:
    def test_serve_mcp_no_package(self, runner):
        """Test serve-mcp command when mcp not installed."""
        with patch("llm_budget.mcp_server.MCP_AVAILABLE", False):
            result = runner.invoke(cli, ["serve-mcp"])
        assert result.exit_code == 1


class TestModelsNoResults:
    def test_no_models_found(self, runner):
        """Test models command with provider that has no models."""
        result = runner.invoke(cli, ["models", "-p", "nonexistent_provider"])
        assert result.exit_code == 0
        assert "No models found" in result.output
