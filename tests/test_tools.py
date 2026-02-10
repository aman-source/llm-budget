"""Tests for tools module."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_budget.tools import (
    _handle_check_budget,
    _handle_compare_models,
    _handle_estimate_cost,
    cost_context,
    cost_prompt,
    cost_tools,
    cost_tools_anthropic,
    cost_tools_openai,
)
from llm_budget.tracker import Tracker


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


# ── cost_tools_openai ────────────────────────────────────────────────────────

class TestCostToolsOpenAI:
    def test_returns_three_tools(self):
        tools = cost_tools_openai()
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_valid_openai_schema(self):
        tools = cost_tools_openai()
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_tool_names(self):
        tools = cost_tools_openai()
        names = {t["function"]["name"] for t in tools}
        assert names == {"estimate_cost", "compare_models", "check_budget"}


# ── cost_tools_anthropic ─────────────────────────────────────────────────────

class TestCostToolsAnthropic:
    def test_returns_three_tools(self):
        tools = cost_tools_anthropic()
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_valid_anthropic_schema(self):
        tools = cost_tools_anthropic()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names(self):
        tools = cost_tools_anthropic()
        names = {t["name"] for t in tools}
        assert names == {"estimate_cost", "compare_models", "check_budget"}


# ── cost_tools ───────────────────────────────────────────────────────────────

class TestCostTools:
    def test_returns_three_tools(self):
        tools = cost_tools()
        assert isinstance(tools, list)
        assert len(tools) == 3

    def test_has_name_and_description(self):
        for tool in cost_tools():
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool


# ── cost_context with different formats ──────────────────────────────────

class TestHandleToolCallFormats:
    def test_openai_format(self):
        """OpenAI tool_call object with .function.name and .function.arguments."""
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="estimate_cost",
                arguments=json.dumps({
                    "prompt": "Hello world",
                    "model": "gpt-4o",
                }),
            )
        )
        result = cost_context(tool_call)
        assert isinstance(result, str)
        assert "gpt-4o" in result
        assert "$" in result

    def test_anthropic_format(self):
        """Anthropic tool_use block with .name and .input."""
        tool_call = SimpleNamespace(
            name="estimate_cost",
            input={"prompt": "Hello world", "model": "gpt-4o"},
        )
        result = cost_context(tool_call)
        assert isinstance(result, str)
        assert "gpt-4o" in result

    def test_dict_format_with_arguments(self):
        """Plain dict with 'name' and 'arguments'."""
        result = cost_context({
            "name": "estimate_cost",
            "arguments": {"prompt": "Hello", "model": "gpt-4o"},
        })
        assert isinstance(result, str)
        assert "gpt-4o" in result

    def test_dict_format_with_input(self):
        """Plain dict with 'name' and 'input'."""
        result = cost_context({
            "name": "estimate_cost",
            "input": {"prompt": "Hello", "model": "gpt-4o"},
        })
        assert isinstance(result, str)

    def test_dict_format_with_json_string(self):
        """Plain dict with JSON string arguments."""
        result = cost_context({
            "name": "estimate_cost",
            "arguments": json.dumps({"prompt": "Hello", "model": "gpt-4o"}),
        })
        assert isinstance(result, str)

    def test_unknown_tool_returns_error_string(self):
        result = cost_context({"name": "nonexistent_tool", "arguments": {}})
        assert isinstance(result, str)
        assert "Error" in result
        assert "Unknown tool" in result

    def test_missing_name_returns_error_string(self):
        result = cost_context({"arguments": {}})
        assert isinstance(result, str)
        assert "Error" in result
        assert "name" in result

    def test_unsupported_format_returns_error_string(self):
        result = cost_context(42)  # type: ignore[arg-type]
        assert isinstance(result, str)
        assert "Error" in result
        assert "Unsupported" in result

    def test_invalid_json_returns_error_string(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="estimate_cost",
                arguments="{invalid json",
            )
        )
        result = cost_context(tool_call)
        assert isinstance(result, str)
        assert "Error" in result
        assert "Invalid JSON" in result

    def test_unknown_model_returns_error_string(self):
        result = cost_context({
            "name": "estimate_cost",
            "arguments": {"prompt": "hello", "model": "fake-model-xyz"},
        })
        assert isinstance(result, str)
        assert "Error" in result
        assert "not found" in result

    def test_missing_required_arg_returns_error_string(self):
        result = cost_context({
            "name": "estimate_cost",
            "arguments": {"prompt": "hello"},  # missing model
        })
        assert isinstance(result, str)
        assert "Error" in result
        assert "model" in result

    def test_compare_skips_unknown_models(self):
        result = cost_context({
            "name": "compare_models",
            "arguments": {
                "prompt": "test",
                "models": ["gpt-4o", "fake-model", "gpt-4o-mini"],
            },
        })
        assert isinstance(result, str)
        assert "gpt-4o" in result
        assert "fake-model" in result  # mentioned in skipped note

    def test_negative_output_tokens_clamped(self):
        result = cost_context({
            "name": "estimate_cost",
            "arguments": {
                "prompt": "hello",
                "model": "gpt-4o",
                "expected_output_tokens": -100,
            },
        })
        assert isinstance(result, str)
        assert "$-" not in result  # no negative costs


# ── estimate_cost handler ────────────────────────────────────────────────────

class TestEstimateCostHandler:
    def test_basic_output(self):
        result = _handle_estimate_cost("Hello world", "gpt-4o")
        assert "Estimated cost for gpt-4o" in result
        assert "Input:" in result
        assert "Output:" in result
        assert "$" in result

    def test_complex_task_gpt4o_well_suited(self):
        result = _handle_estimate_cost(
            "Analyze this complex problem step by step",
            "gpt-4o",
            task_type="complex",
        )
        assert "Well-suited" in result

    def test_extraction_task_gpt4o_overkill(self):
        result = _handle_estimate_cost(
            "Extract the name from: John Smith",
            "gpt-4o",
            task_type="extraction",
        )
        assert "Overkill" in result

    def test_complex_task_gpt4o_mini_below_recommended(self):
        result = _handle_estimate_cost(
            "Analyze this complex problem step by step",
            "gpt-4o-mini",
            task_type="complex",
        )
        assert "Below recommended" in result

    def test_no_task_type_still_works(self):
        result = _handle_estimate_cost("Hello", "gpt-4o")
        assert "Estimated cost" in result
        # Should NOT contain task fit assessment
        assert "Task fit" not in result

    def test_shows_capability_tier(self):
        result = _handle_estimate_cost("Hello", "gpt-4o")
        assert "Frontier" in result
        assert "95/100" in result

    def test_with_expected_output_tokens(self):
        result = _handle_estimate_cost(
            "Hello", "gpt-4o", expected_output_tokens=100
        )
        assert "100" in result


# ── compare_models handler ───────────────────────────────────────────────────

class TestCompareModelsHandler:
    def test_complex_task_frontier_in_recommended(self):
        result = _handle_compare_models(
            "Analyze this complex problem",
            task_type="complex",
            models=["gpt-4o", "gpt-4o-mini", "deepseek-chat"],
        )
        assert "RECOMMENDED" in result
        assert "gpt-4o" in result

    def test_simple_task_frontier_in_overkill(self):
        result = _handle_compare_models(
            "Extract the name",
            task_type="simple",
            models=["gpt-4o", "gpt-4o-mini", "deepseek-chat"],
        )
        assert "OVERKILL" in result

    def test_no_task_type_neutral_comparison(self):
        result = _handle_compare_models(
            "Hello world",
            models=["gpt-4o", "gpt-4o-mini"],
        )
        assert "Tip" in result
        assert "task_type" in result

    def test_default_models_used_when_none_provided(self):
        result = _handle_compare_models("Hello world")
        # Should include some of the default models
        assert "gpt-4o" in result

    def test_includes_best_value(self):
        result = _handle_compare_models(
            "Test prompt",
            task_type="moderate",
            models=["gpt-4o", "gpt-4o-mini", "deepseek-chat"],
        )
        assert "Best value" in result or "Cheapest" in result


# ── check_budget handler ─────────────────────────────────────────────────────

class TestCheckBudgetHandler:
    def test_healthy_budget(self, tracker):
        # Record small spend (~20%)
        tracker.record("gpt-4o", 100, 50, 1.00)
        result = _handle_check_budget(
            "today",
            tracker=tracker,
            budget_config={"daily": 5.00},
        )
        assert "Healthy" in result
        assert "Budget Status" in result

    def test_tightening_budget(self, tracker):
        # Record ~75% spend
        tracker.record("gpt-4o", 100, 50, 3.75)
        result = _handle_check_budget(
            "today",
            tracker=tracker,
            budget_config={"daily": 5.00},
        )
        assert "tightening" in result

    def test_critical_budget(self, tracker):
        # Record ~92% spend
        tracker.record("gpt-4o", 100, 50, 4.60)
        result = _handle_check_budget(
            "today",
            tracker=tracker,
            budget_config={"daily": 5.00},
        )
        assert "CRITICAL" in result

    def test_no_budget_config(self, tracker):
        tracker.record("gpt-4o", 100, 50, 1.00)
        result = _handle_check_budget("today", tracker=tracker)
        assert "Budget Status" in result
        assert "No budget limit" in result

    def test_all_time_period_maps_to_total(self, tracker):
        tracker.record("gpt-4o", 100, 50, 1.00)
        result = _handle_check_budget("all_time", tracker=tracker)
        assert "all_time" in result

    def test_shows_model_breakdown(self, tracker):
        tracker.record("gpt-4o", 100, 50, 0.50)
        tracker.record("gpt-4o-mini", 100, 50, 0.02)
        tracker.record("gpt-4o", 100, 50, 0.30)
        result = _handle_check_budget("today", tracker=tracker)
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_shows_call_counts(self, tracker):
        tracker.record("gpt-4o", 100, 50, 0.50)
        tracker.record("gpt-4o", 100, 50, 0.30)
        result = _handle_check_budget("today", tracker=tracker)
        assert "2 calls" in result

    def test_empty_tracker(self, tracker):
        result = _handle_check_budget("today", tracker=tracker)
        assert "No spending recorded" in result


# ── cost_context routing ─────────────────────────────────────────────────

class TestHandleToolCallRouting:
    def test_estimate_cost_route(self):
        result = cost_context({
            "name": "estimate_cost",
            "arguments": {"prompt": "Test", "model": "gpt-4o"},
        })
        assert "Estimated cost" in result

    def test_compare_models_route(self):
        result = cost_context({
            "name": "compare_models",
            "arguments": {
                "prompt": "Test",
                "models": ["gpt-4o", "gpt-4o-mini"],
            },
        })
        assert "gpt-4o" in result

    def test_check_budget_route(self):
        result = cost_context({
            "name": "check_budget",
            "arguments": {"period": "today"},
        })
        assert "Budget Status" in result

    def test_check_budget_default_period(self):
        result = cost_context({
            "name": "check_budget",
            "arguments": {},
        })
        assert "Budget Status" in result


# ── System prompt ────────────────────────────────────────────────────────────

class TestCostPrompt:
    def test_balanced_strategy(self):
        prompt = cost_prompt(strategy="balanced")
        assert "Decision framework" in prompt
        assert "estimate_cost" in prompt
        assert "compare_models" in prompt
        assert "check_budget" in prompt

    def test_cost_first_strategy(self):
        prompt = cost_prompt(strategy="cost_first")
        assert "MINIMIZE COST" in prompt

    def test_quality_first_strategy(self):
        prompt = cost_prompt(strategy="quality_first")
        assert "PRIORITIZE QUALITY" in prompt

    def test_includes_budget_info(self):
        prompt = cost_prompt(
            daily_budget=5.00,
            strategy="balanced",
        )
        assert "$5.00" in prompt
        assert "Budget" in prompt

    def test_multiple_budget_periods(self):
        prompt = cost_prompt(
            daily_budget=5.00,
            weekly_budget=25.00,
            monthly_budget=100.00,
        )
        assert "$5.00" in prompt
        assert "$25.00" in prompt
        assert "$100.00" in prompt
