"""Tests for the CrewAI integration module."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_budget.exceptions import BudgetExceeded
from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


# Mock CrewAI hook registration
_mock_before_hooks: list = []
_mock_after_hooks: list = []


def _mock_register_before(fn):
    _mock_before_hooks.append(fn)


def _mock_register_after(fn):
    _mock_after_hooks.append(fn)


@pytest.fixture(autouse=True)
def _reset_hooks():
    _mock_before_hooks.clear()
    _mock_after_hooks.clear()
    yield
    _mock_before_hooks.clear()
    _mock_after_hooks.clear()


# Shared patch for all tests needing CrewAI mocked
_crewai_patches = [
    patch("llm_budget.integrations.crewai.CREWAI_AVAILABLE", True),
    patch(
        "llm_budget.integrations.crewai.register_before_llm_call_hook",
        _mock_register_before,
        create=True,
    ),
    patch(
        "llm_budget.integrations.crewai.register_after_llm_call_hook",
        _mock_register_after,
        create=True,
    ),
]


class TestCrewaiAvailability:
    def test_availability_flag_is_bool(self):
        from llm_budget.integrations.crewai import CREWAI_AVAILABLE

        assert isinstance(CREWAI_AVAILABLE, bool)

    def test_raises_when_crewai_not_available(self):
        with patch("llm_budget.integrations.crewai.CREWAI_AVAILABLE", False):
            from llm_budget.integrations.crewai import BudgetHooks

            with pytest.raises(ImportError, match="crewai"):
                BudgetHooks(max_cost=1.0)


class TestNormalizePrompt:
    def test_string(self):
        from llm_budget.integrations.crewai import _normalize_crewai_prompt

        assert _normalize_crewai_prompt("hello") == "hello"

    def test_list_of_strings(self):
        from llm_budget.integrations.crewai import _normalize_crewai_prompt

        result = _normalize_crewai_prompt(["hello", "world"])
        assert result == "hello\nworld"

    def test_list_of_dicts(self):
        from llm_budget.integrations.crewai import _normalize_crewai_prompt

        result = _normalize_crewai_prompt(
            [{"content": "hello"}, {"content": "world"}]
        )
        assert result == "hello\nworld"

    def test_object_with_content(self):
        from llm_budget.integrations.crewai import _normalize_crewai_prompt

        msg = SimpleNamespace(content="hello")
        result = _normalize_crewai_prompt([msg])
        assert result == "hello"

    def test_non_string(self):
        from llm_budget.integrations.crewai import _normalize_crewai_prompt

        assert _normalize_crewai_prompt(42) == "42"


class TestBudgetHooks:
    @pytest.fixture(autouse=True)
    def _patch_crewai(self):
        with patch(
            "llm_budget.integrations.crewai.CREWAI_AVAILABLE", True
        ), patch(
            "llm_budget.integrations.crewai.register_before_llm_call_hook",
            _mock_register_before,
            create=True,
        ), patch(
            "llm_budget.integrations.crewai.register_after_llm_call_hook",
            _mock_register_after,
            create=True,
        ):
            yield

    def test_before_call_runs_estimation(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.before_call("Hello, how are you?")
        assert hooks.last_estimation is not None
        assert hooks.last_estimation.model == "gpt-4o"

    def test_after_call_records_spend(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks._last_model = "gpt-4o"

        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        hooks.after_call(response)

        assert hooks.call_count == 1
        assert hooks.total_input_tokens == 100
        assert hooks.total_output_tokens == 50
        assert hooks.total_cost > 0
        assert tracker.get_spend(period="total") > 0

    def test_budget_exceeded_raises(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        tracker.record(
            model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.0
        )

        hooks = BudgetHooks(
            max_cost=10.0,
            period="total",
            on_exceed="raise",
            tracker=tracker,
            default_model="gpt-4o",
        )
        with pytest.raises(BudgetExceeded):
            hooks.before_call("Hello")

    def test_register_adds_hooks(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.register()
        assert len(_mock_before_hooks) == 1
        assert len(_mock_after_hooks) == 1
        assert hooks.is_registered

    def test_unregister(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.register()
        hooks.unregister()
        assert not hooks.is_registered

    def test_context_manager(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        with BudgetHooks(
            tracker=tracker, default_model="gpt-4o"
        ) as hooks:
            assert hooks.is_registered
        assert not hooks.is_registered

    def test_model_from_kwargs(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.before_call("Hello", model="gpt-4o-mini")
        assert hooks._last_model == "gpt-4o-mini"

    def test_fallback_to_estimation_on_no_usage(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.before_call("Hello world, tell me about AI")
        hooks.after_call("plain string response")
        assert hooks.total_cost > 0

    def test_tracking_only_mode(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks.before_call("Hello")
        assert hooks.last_estimation is not None
        assert hooks._enforcer is None

    def test_properties_accumulate(self, tracker):
        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(tracker=tracker, default_model="gpt-4o")
        hooks._last_model = "gpt-4o"

        for _ in range(3):
            response = SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            )
            hooks.after_call(response)

        assert hooks.call_count == 3
        assert hooks.total_input_tokens == 30
        assert hooks.total_output_tokens == 15
        assert hooks.total_tokens == 45


class TestTrackCrewai:
    @pytest.fixture(autouse=True)
    def _patch_crewai(self):
        with patch(
            "llm_budget.integrations.crewai.CREWAI_AVAILABLE", True
        ), patch(
            "llm_budget.integrations.crewai.register_before_llm_call_hook",
            _mock_register_before,
            create=True,
        ), patch(
            "llm_budget.integrations.crewai.register_after_llm_call_hook",
            _mock_register_after,
            create=True,
        ):
            yield

    def test_context_manager(self, tracker):
        from llm_budget.integrations.crewai import track_crewai

        with track_crewai(max_cost=5.0, tracker=tracker) as hooks:
            assert hooks.is_registered
            assert hooks.total_cost == 0.0
        assert not hooks.is_registered

    def test_tracks_cost_through_lifecycle(self, tracker):
        from llm_budget.integrations.crewai import track_crewai

        with track_crewai(tracker=tracker) as hooks:
            hooks.before_call("Hello world")
            response = SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=25)
            )
            hooks.after_call(response)
        assert hooks.total_cost > 0
        assert hooks.call_count == 1
