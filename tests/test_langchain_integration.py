"""Tests for the LangChain integration module."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_budget.exceptions import BudgetExceeded
from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


# -- Mock LangChain types ----------------------------------------------------


class _FakeLLMResult:
    def __init__(self, token_usage=None, usage=None, model_name=None):
        self.llm_output = {}
        if token_usage:
            self.llm_output["token_usage"] = token_usage
        if usage:
            self.llm_output["usage"] = usage
        if model_name:
            self.llm_output["model_name"] = model_name
        self.generations = []


class _FakeHumanMessage:
    type = "human"

    def __init__(self, content):
        self.content = content


class _FakeAIMessage:
    type = "ai"

    def __init__(self, content):
        self.content = content


class _FakeSystemMessage:
    type = "system"

    def __init__(self, content):
        self.content = content


# -- Tests --------------------------------------------------------------------


class TestLangchainAvailability:
    def test_availability_flag_is_bool(self):
        from llm_budget.integrations.langchain import LANGCHAIN_AVAILABLE

        assert isinstance(LANGCHAIN_AVAILABLE, bool)

    def test_raises_when_langchain_not_available(self):
        with patch(
            "llm_budget.integrations.langchain.LANGCHAIN_AVAILABLE", False
        ):
            from llm_budget.integrations.langchain import BudgetCallbackHandler

            with pytest.raises(ImportError, match="langchain-core"):
                BudgetCallbackHandler(max_cost=1.0)


class TestMessageConversion:
    def test_human_message(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        result = _langchain_messages_to_dicts([_FakeHumanMessage("Hello")])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_ai_message(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        result = _langchain_messages_to_dicts([_FakeAIMessage("Hi there")])
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_system_message(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        result = _langchain_messages_to_dicts(
            [_FakeSystemMessage("You are helpful")]
        )
        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_string_message(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        result = _langchain_messages_to_dicts(["plain string"])
        assert result == [{"role": "user", "content": "plain string"}]

    def test_dict_passthrough(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        msg = {"role": "user", "content": "hi"}
        result = _langchain_messages_to_dicts([msg])
        assert result == [msg]

    def test_mixed_messages(self):
        from llm_budget.integrations.langchain import _langchain_messages_to_dicts

        msgs = [
            _FakeSystemMessage("Be helpful"),
            _FakeHumanMessage("Hello"),
            _FakeAIMessage("Hi there"),
        ]
        result = _langchain_messages_to_dicts(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


class TestModelExtraction:
    def test_from_serialized_kwargs_model_name(self):
        from llm_budget.integrations.langchain import (
            _extract_model_from_serialized,
        )

        result = _extract_model_from_serialized(
            {"kwargs": {"model_name": "gpt-4o"}}
        )
        assert result == "gpt-4o"

    def test_from_serialized_kwargs_model(self):
        from llm_budget.integrations.langchain import (
            _extract_model_from_serialized,
        )

        result = _extract_model_from_serialized(
            {"kwargs": {"model": "claude-sonnet-4-20250514"}}
        )
        assert result == "claude-sonnet-4-20250514"

    def test_from_invocation_params(self):
        from llm_budget.integrations.langchain import (
            _extract_model_from_serialized,
        )

        result = _extract_model_from_serialized(
            {"kwargs": {}},
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        assert result == "gpt-4o-mini"

    def test_returns_none_when_not_found(self):
        from llm_budget.integrations.langchain import (
            _extract_model_from_serialized,
        )

        result = _extract_model_from_serialized({})
        assert result is None


class TestBudgetCallbackHandler:
    @pytest.fixture(autouse=True)
    def _patch_langchain_available(self):
        with patch(
            "llm_budget.integrations.langchain.LANGCHAIN_AVAILABLE", True
        ):
            yield

    def test_on_llm_start_runs_estimation(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello, how are you?"],
        )
        assert handler.last_estimation is not None
        assert handler.last_estimation.model == "gpt-4o"

    def test_on_llm_end_records_spend(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler._last_model = "gpt-4o"

        response = _FakeLLMResult(
            token_usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        handler.on_llm_end(response)

        assert handler.call_count == 1
        assert handler.total_input_tokens == 100
        assert handler.total_output_tokens == 50
        assert handler.total_cost > 0
        assert tracker.get_spend(period="total") > 0

    def test_budget_exceeded_raises(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        tracker.record(
            model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.0
        )

        handler = BudgetCallbackHandler(
            max_cost=10.0, period="total", on_exceed="raise", tracker=tracker
        )
        with pytest.raises(BudgetExceeded):
            handler.on_llm_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                prompts=["Hello, how are you?"],
            )

    def test_budget_warn_mode_allows(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        tracker.record(
            model="gpt-4o", input_tokens=1000, output_tokens=500, cost_usd=10.0
        )

        handler = BudgetCallbackHandler(
            max_cost=10.0, period="total", on_exceed="warn", tracker=tracker
        )
        # warn mode should NOT raise, just warn
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello"],
        )
        assert handler.last_estimation is not None

    def test_tracking_only_mode(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello"],
        )
        assert handler.last_estimation is not None
        assert handler._enforcer is None

    def test_on_chat_model_start(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            messages=[[_FakeHumanMessage("Hello")]],
        )
        assert handler.last_estimation is not None

    def test_on_llm_error_is_noop(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler.on_llm_error(ValueError("test error"))

    def test_model_override(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker, model="gpt-4o-mini")
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello"],
        )
        assert handler._last_model == "gpt-4o-mini"
        assert handler.last_estimation.model == "gpt-4o-mini"

    def test_properties_accumulate(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler._last_model = "gpt-4o"

        for _ in range(3):
            response = _FakeLLMResult(
                token_usage={"prompt_tokens": 10, "completion_tokens": 5}
            )
            handler.on_llm_end(response)

        assert handler.call_count == 3
        assert handler.total_input_tokens == 30
        assert handler.total_output_tokens == 15
        assert handler.total_tokens == 45

    def test_anthropic_token_format_in_token_usage(self, tracker):
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler._last_model = "gpt-4o"

        response = _FakeLLMResult(
            token_usage={"input_tokens": 80, "output_tokens": 40}
        )
        handler.on_llm_end(response)

        assert handler.total_input_tokens == 80
        assert handler.total_output_tokens == 40

    def test_anthropic_usage_dict_format(self, tracker):
        """Anthropic via LangChain puts usage in llm_output['usage'], not 'token_usage'."""
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler._last_model = "claude-sonnet-4-20250514"

        response = _FakeLLMResult(
            usage={"input_tokens": 15, "output_tokens": 12}
        )
        handler.on_llm_end(response)

        assert handler.total_input_tokens == 15
        assert handler.total_output_tokens == 12
        assert handler.total_cost > 0

    def test_model_name_from_llm_output(self, tracker):
        """on_llm_end extracts actual model name from llm_output."""
        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(tracker=tracker)
        handler._last_model = "gpt-4o"  # pre-flight guess

        response = _FakeLLMResult(
            usage={"input_tokens": 10, "output_tokens": 5},
            model_name="claude-sonnet-4-20250514",  # actual model from response
        )
        handler.on_llm_end(response)

        # Cost should be calculated against the actual model, not the guess
        assert handler.call_count == 1
        assert handler.total_input_tokens == 10


class TestBudgetCallbackContextManager:
    @pytest.fixture(autouse=True)
    def _patch_langchain_available(self):
        with patch(
            "llm_budget.integrations.langchain.LANGCHAIN_AVAILABLE", True
        ):
            yield

    def test_context_manager_yields_handler(self, tracker):
        from llm_budget.integrations.langchain import budget_callback

        with budget_callback(max_cost=5.0, tracker=tracker) as handler:
            assert handler is not None
            assert handler.total_cost == 0.0

    def test_context_manager_tracks_cost(self, tracker):
        from llm_budget.integrations.langchain import budget_callback

        with budget_callback(tracker=tracker) as handler:
            handler._last_model = "gpt-4o"
            response = _FakeLLMResult(
                token_usage={"prompt_tokens": 50, "completion_tokens": 25}
            )
            handler.on_llm_end(response)
        assert handler.total_cost > 0
