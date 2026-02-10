"""Tests for the decorators module."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_budget.decorators import _extract_usage, budget, estimate_decorator
from llm_budget.exceptions import BudgetExceeded
from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


@pytest.fixture(autouse=True)
def _patch_tracker(tracker, monkeypatch):
    """Patch the default tracker so decorators use our test tracker."""
    monkeypatch.setattr("llm_budget.decorators.get_tracker", lambda: tracker)


class TestExtractUsage:
    def test_openai_format(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        inp, out = _extract_usage(response)
        assert inp == 100
        assert out == 50

    def test_anthropic_format(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=200, output_tokens=100)
        )
        inp, out = _extract_usage(response)
        assert inp == 200
        assert out == 100

    def test_dict_format(self):
        response = {"usage": {"prompt_tokens": 50, "completion_tokens": 25}}
        inp, out = _extract_usage(response)
        assert inp == 50
        assert out == 25

    def test_no_usage(self):
        inp, out = _extract_usage("plain string")
        assert inp is None
        assert out is None

    def test_none_response(self):
        inp, out = _extract_usage(None)
        assert inp is None
        assert out is None


class TestBudgetDecorator:
    def test_basic_call(self, tracker):
        fake_response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        )

        @budget(max_cost=10.0, period="total", track_model="gpt-4o")
        def my_func(model, messages):
            return fake_response

        result = my_func(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result is fake_response

        # Verify spend was recorded
        spend = tracker.get_spend(period="total")
        assert spend > 0

    def test_budget_exceeded_raises(self, tracker):
        # Pre-fill tracker past the limit so any new call exceeds it
        tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.00)

        @budget(max_cost=10.0, period="total", on_exceed="raise", track_model="gpt-4o")
        def my_func(model, messages):
            return None

        with pytest.raises(BudgetExceeded):
            my_func(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_budget_skip_returns_none(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.00)

        @budget(max_cost=10.0, period="total", on_exceed="skip", track_model="gpt-4o")
        def my_func(model, messages):
            return "should not reach here"

        result = my_func(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result is None

    def test_no_model_runs_untracked(self, tracker):
        @budget(max_cost=10.0, period="total")
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"
        assert tracker.get_spend(period="total") == 0.0

    def test_downgrade_swaps_model(self, tracker):
        """When on_exceed='downgrade:X', the decorator should catch BudgetExceeded
        and call the function with the downgraded model."""
        tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.00)

        received_models = []

        @budget(max_cost=10.0, period="total", on_exceed="downgrade:gpt-4o-mini", track_model="gpt-4o")
        def my_func(model, messages):
            received_models.append(model)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            )

        result = my_func(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result is not None
        assert received_models == ["gpt-4o-mini"]

    def test_downgrade_with_positional_args(self, tracker):
        """Downgrade also works when model is passed positionally."""
        tracker.record(model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.00)

        received_models = []

        @budget(max_cost=10.0, period="total", on_exceed="downgrade:gpt-4o-mini", track_model="gpt-4o")
        def my_func(model, messages):
            received_models.append(model)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            )

        result = my_func("gpt-4o", [{"role": "user", "content": "Hello"}])
        assert result is not None
        assert received_models == ["gpt-4o-mini"]


class TestEstimateDecorator:
    def test_logs_estimate(self, caplog):
        import logging

        @estimate_decorator(model="gpt-4o", log_level=logging.INFO)
        def my_func(model, messages):
            return "result"

        with caplog.at_level(logging.INFO):
            result = my_func(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert result == "result"
        assert any("Estimated cost" in r.message for r in caplog.records)

    def test_no_model_still_runs(self):
        @estimate_decorator()
        def my_func():
            return "ok"

        assert my_func() == "ok"
