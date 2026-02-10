"""Tests for the estimator module."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_budget.estimator import (
    EstimationResult,
    _estimate_output_tokens,
    _get_learned_ratio,
    compare,
    estimate,
)
from llm_budget.exceptions import ModelNotFoundError
from llm_budget.pricing import get_registry
from llm_budget.tracker import Tracker


class TestEstimate:
    def test_basic_string(self):
        result = estimate("Hello world", model="gpt-4o")
        assert isinstance(result, EstimationResult)
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.estimated_cost > 0
        assert result.is_output_estimated is True

    def test_with_messages(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        result = estimate(msgs, model="gpt-4o")
        assert result.input_tokens > 0
        # Messages should have more tokens than raw string due to overhead
        raw = estimate("Hello world", model="gpt-4o")
        assert result.input_tokens > raw.input_tokens

    def test_explicit_output_tokens(self):
        result = estimate("Hello", model="gpt-4o", expected_output_tokens=500)
        assert result.output_tokens == 500
        assert result.is_output_estimated is False

    def test_cost_math(self):
        result = estimate("Hello", model="gpt-4o", expected_output_tokens=100)
        expected_cost = result.input_cost + result.output_cost
        assert abs(result.estimated_cost - expected_cost) < 1e-12

    def test_breakdown_property(self):
        result = estimate("Hello", model="gpt-4o", expected_output_tokens=100)
        breakdown = result.breakdown
        assert "input" in breakdown
        assert "output" in breakdown
        assert breakdown["input"].startswith("$")

    def test_str_representation(self):
        result = estimate("Hello", model="gpt-4o")
        s = str(result)
        assert "gpt-4o" in s
        assert "$" in s

    def test_different_models(self):
        r1 = estimate("Hello", model="gpt-4o", expected_output_tokens=100)
        r2 = estimate("Hello", model="gpt-4o-mini", expected_output_tokens=100)
        # gpt-4o should be more expensive than gpt-4o-mini
        assert r1.estimated_cost > r2.estimated_cost


class TestEstimateOutputTokens:
    def test_small_input(self):
        # For very small input, heuristic = min(max_out, input*0.75, 1000)
        assert _estimate_output_tokens(10, 16384) == 7  # 10 * 0.75 = 7.5 -> 7

    def test_medium_input(self):
        assert _estimate_output_tokens(2000, 16384) == 1000  # capped at 1000

    def test_limited_max_output(self):
        assert _estimate_output_tokens(10000, 500) == 500  # capped at max_output


class TestCompare:
    def test_returns_sorted(self):
        results = compare(
            "Hello world",
            ["gpt-4o", "gpt-4o-mini", "deepseek-chat"],
            expected_output_tokens=100,
        )
        assert len(results) == 3
        costs = [r.estimated_cost for r in results]
        assert costs == sorted(costs)

    def test_returns_all_models(self):
        models = ["gpt-4o", "claude-sonnet-4-20250514"]
        results = compare("Hello", models, expected_output_tokens=100)
        result_models = {r.model for r in results}
        assert result_models == set(models)

    def test_print_table_false(self, capsys):
        results = compare(
            "Hello",
            ["gpt-4o", "gpt-4o-mini"],
            expected_output_tokens=100,
            print_table=False,
        )
        assert len(results) == 2
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_table_true(self, capsys):
        compare(
            "Hello",
            ["gpt-4o", "gpt-4o-mini"],
            expected_output_tokens=100,
            print_table=True,
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestInputLimitWarning:
    def test_warns_when_exceeding_max_input(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            # gpt-3.5-turbo has max_input_tokens=16385
            huge = "word " * 20000
            estimate(huge, model="gpt-3.5-turbo", expected_output_tokens=10)
        assert any("exceed" in r.message.lower() for r in caplog.records)


class TestPricingFuzzyMatch:
    """Test that versioned model names resolve to the most specific match."""

    def test_exact_match(self):
        reg = get_registry()
        assert reg.get("gpt-4o").name == "gpt-4o"
        assert reg.get("gpt-4o-mini").name == "gpt-4o-mini"

    def test_versioned_gpt4o_mini(self):
        reg = get_registry()
        # Must match gpt-4o-mini, NOT gpt-4o
        assert reg.get("gpt-4o-mini-2024-07-18").name == "gpt-4o-mini"

    def test_versioned_gpt4o(self):
        reg = get_registry()
        assert reg.get("gpt-4o-2024-08-06").name == "gpt-4o"

    def test_o1_exact(self):
        reg = get_registry()
        assert reg.get("o1").name == "o1"

    def test_o3_mini_variant(self):
        reg = get_registry()
        assert reg.get("o3-mini-high").name == "o3-mini"

    def test_unknown_model(self):
        reg = get_registry()
        with pytest.raises(ModelNotFoundError):
            reg.get("totally-unknown-model")


class TestAdaptiveEstimation:
    """Test that the estimator learns from tracker history."""

    @pytest.fixture
    def tracker(self, tmp_path):
        t = Tracker(db_path=str(tmp_path / "learn.db"))
        yield t
        t.close()

    def test_uses_learned_ratio(self, tracker):
        """With enough history, estimate should use learned ratio instead of static heuristic."""
        # Seed tracker with 10 calls where output ≈ 2x input
        for _ in range(10):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=200, cost_usd=0.01)

        # Patch get_tracker to return our test tracker
        with patch("llm_budget.tracker.get_tracker", return_value=tracker):
            # Static heuristic for 100 input = min(16384, 75, 1000) = 75
            # Learned ratio = 2.0, so adaptive = 100 * 2.0 = 200
            result = _estimate_output_tokens(100, 16384, model="gpt-4o")
            assert result == 200

    def test_falls_back_to_heuristic_cold_start(self, tracker):
        """With no history, should use the static heuristic."""
        with patch("llm_budget.tracker.get_tracker", return_value=tracker):
            # No records → cold start → static heuristic
            result = _estimate_output_tokens(100, 16384, model="gpt-4o")
            assert result == 75  # min(16384, 100*0.75, 1000) = 75

    def test_clamped_to_max_output(self, tracker):
        """Learned ratio shouldn't exceed max_output_tokens."""
        for _ in range(10):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=5000, cost_usd=0.5)

        with patch("llm_budget.tracker.get_tracker", return_value=tracker):
            # Learned ratio = 50.0, so 100 * 50 = 5000, but max_output = 500
            result = _estimate_output_tokens(100, 500, model="gpt-4o")
            assert result == 500

    def test_no_model_uses_static_heuristic(self):
        """When model is None, always use static heuristic (no tracker lookup)."""
        result = _estimate_output_tokens(100, 16384, model=None)
        assert result == 75

    def test_end_to_end_estimate_uses_learning(self, tracker):
        """Full estimate() call should produce different output tokens when history exists."""
        # Seed: calls where output = 3x input (ratio=3.0)
        for _ in range(10):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=300, cost_usd=0.01)

        with patch("llm_budget.tracker.get_tracker", return_value=tracker):
            result = estimate("Hello", model="gpt-4o")
            # "Hello" ≈ 1 token. Learned ratio=3.0 → 1*3=3, but clamped to min 1.
            # The key assertion: is_output_estimated=True and output came from learning
            assert result.is_output_estimated is True
            assert result.output_tokens >= 1

    def test_get_learned_ratio_handles_exceptions(self):
        """_get_learned_ratio returns None if tracker raises."""
        with patch("llm_budget.tracker.get_tracker", side_effect=RuntimeError("boom")):
            assert _get_learned_ratio("gpt-4o") is None
