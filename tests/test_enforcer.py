"""Tests for the enforcer module."""
from __future__ import annotations

import warnings

import pytest

from llm_budget.enforcer import BudgetEnforcer, BudgetPolicy, EnforcementResult
from llm_budget.estimator import EstimationResult
from llm_budget.exceptions import BudgetExceeded
from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


def _make_estimation(cost: float = 0.01) -> EstimationResult:
    """Create a dummy EstimationResult."""
    return EstimationResult(
        model="gpt-4o",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        estimated_cost=cost,
        input_cost=cost * 0.4,
        output_cost=cost * 0.6,
        max_input_tokens=128000,
        max_output_tokens=16384,
        is_output_estimated=True,
    )


class TestUnderBudget:
    def test_allowed(self, tracker):
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total")
        result = enforcer.check(_make_estimation(0.01), policy)
        assert result.allowed is True
        assert result.remaining_budget == 1.0
        assert result.warning is None

    def test_with_existing_spend(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.50)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total")
        result = enforcer.check(_make_estimation(0.01), policy)
        assert result.allowed is True
        assert abs(result.current_spend - 0.50) < 1e-9


class TestOverBudgetRaise:
    def test_raises(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.99)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total", on_exceed="raise")
        with pytest.raises(BudgetExceeded) as exc_info:
            enforcer.check(_make_estimation(0.05), policy)
        assert exc_info.value.period == "total"
        assert exc_info.value.estimated_cost == 0.05


class TestOverBudgetWarn:
    def test_allowed_with_warning(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.99)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total", on_exceed="warn")
        result = enforcer.check(_make_estimation(0.05), policy)
        assert result.allowed is True
        assert result.warning is not None


class TestOverBudgetSkip:
    def test_not_allowed(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.99)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total", on_exceed="skip")
        result = enforcer.check(_make_estimation(0.05), policy)
        assert result.allowed is False


class TestOverBudgetDowngrade:
    def test_suggests_model(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.99)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(
            max_cost=1.0, period="total", on_exceed="downgrade:gpt-4o-mini"
        )
        with pytest.raises(BudgetExceeded) as exc_info:
            enforcer.check(_make_estimation(0.05), policy)
        assert exc_info.value.suggested_model == "gpt-4o-mini"


class TestAlertThreshold:
    def test_alert_at_threshold(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.85)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total", alert_at=0.8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = enforcer.check(_make_estimation(0.01), policy)
        assert result.warning is not None
        assert len(w) == 1
        assert "85%" in str(w[0].message)

    def test_no_alert_below_threshold(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.50)
        enforcer = BudgetEnforcer(tracker)
        policy = BudgetPolicy(max_cost=1.0, period="total", alert_at=0.8)
        result = enforcer.check(_make_estimation(0.01), policy)
        assert result.warning is None
