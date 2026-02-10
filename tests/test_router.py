"""Tests for the router module — smart_route() and @cost_aware."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_budget.router import (
    _detect_provider,
    _normalize_task,
    _pick_model,
    cost_aware,
    smart_route,
)
from llm_budget.tracker import Tracker


# ---------------------------------------------------------------------------
# Fake clients (same pattern as test_middleware.py)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeChat:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response):
        self.chat = _FakeChat(response)


class _FakeAnthropicMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeAnthropicMessages(response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


def _openai_response(in_tok=100, out_tok=50):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        model="gpt-4o-mini",
    )


def _anthropic_response(in_tok=100, out_tok=50):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
        model="claude-3-5-haiku-20241022",
    )


# ---------------------------------------------------------------------------
# TestDetectProvider
# ---------------------------------------------------------------------------


class TestDetectProvider:
    def test_detects_openai_by_attribute(self):
        client = _FakeOpenAIClient(_openai_response())
        assert _detect_provider(client) == "openai"

    def test_detects_anthropic_by_attribute(self):
        client = _FakeAnthropicClient(_anthropic_response())
        # FakeAnthropicClient has .messages but not .chat
        assert _detect_provider(client) == "anthropic"

    def test_unknown_returns_none(self):
        client = object()
        assert _detect_provider(client) is None


# ---------------------------------------------------------------------------
# TestNormalizeTask
# ---------------------------------------------------------------------------


class TestNormalizeTask:
    def test_string_becomes_messages(self):
        msgs, text = _normalize_task("Hello world")
        assert msgs == [{"role": "user", "content": "Hello world"}]
        assert text == "Hello world"

    def test_messages_list_passed_through(self):
        original = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        msgs, text = _normalize_task(original)
        assert msgs is original
        assert "Be helpful" in text
        assert "Hello" in text

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be str or list"):
            _normalize_task(123)


# ---------------------------------------------------------------------------
# TestPickModel
# ---------------------------------------------------------------------------


class TestPickModel:
    def test_priority_cheapest_safe_first(self):
        recs = {
            "cheapest_safe": "model-a",
            "best_value": "model-b",
            "cheapest_risky": "model-c",
        }
        assert _pick_model(recs) == "model-a"

    def test_falls_back_to_best_value(self):
        recs = {
            "cheapest_safe": None,
            "best_value": "model-b",
            "cheapest_risky": "model-c",
        }
        assert _pick_model(recs) == "model-b"

    def test_falls_back_to_cheapest_risky(self):
        recs = {
            "cheapest_safe": None,
            "best_value": None,
            "cheapest_risky": "model-c",
        }
        assert _pick_model(recs) == "model-c"

    def test_all_none_returns_none(self):
        recs = {
            "cheapest_safe": None,
            "best_value": None,
            "cheapest_risky": None,
        }
        assert _pick_model(recs) is None


# ---------------------------------------------------------------------------
# TestSmartRoute
# ---------------------------------------------------------------------------


class TestSmartRoute:
    def test_string_task_to_messages(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        smart_route(
            client, "Hello world",
            provider="anthropic", tracker=tracker,
        )

        kwargs = client.messages.last_kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "Hello world"}]

    def test_messages_list_forwarded(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)
        msgs = [{"role": "user", "content": "Hi"}]

        smart_route(
            client, msgs,
            provider="anthropic", tracker=tracker,
        )

        kwargs = client.messages.last_kwargs
        assert kwargs["messages"] == msgs

    def test_explicit_model_skips_routing(self, tracker):
        resp = _openai_response()
        client = _FakeOpenAIClient(resp)

        smart_route(
            client, "test",
            model="gpt-4o", provider="openai", tracker=tracker,
        )

        kwargs = client.chat.completions.last_kwargs
        assert kwargs["model"] == "gpt-4o"

    def test_extraction_picks_efficient_model(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        smart_route(
            client, "Extract dates from this text",
            task_type="extraction", provider="anthropic", tracker=tracker,
        )

        from llm_budget.pricing import get_registry
        model_used = client.messages.last_kwargs["model"]
        pricing = get_registry().get(model_used)
        assert pricing.capability_tier == "efficient"

    def test_complex_picks_frontier_model(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        smart_route(
            client, "Analyze the trade-offs between microservices and monoliths",
            task_type="complex", provider="anthropic", tracker=tracker,
        )

        from llm_budget.pricing import get_registry
        model_used = client.messages.last_kwargs["model"]
        pricing = get_registry().get(model_used)
        assert pricing.capability_tier in ("frontier", "reasoning")

    def test_cost_tracked(self, tracker):
        resp = _anthropic_response(in_tok=100, out_tok=50)
        client = _FakeAnthropicClient(resp)

        smart_route(
            client, "test",
            provider="anthropic", tracker=tracker,
        )

        assert tracker.get_spend(period="total") > 0

    def test_api_kwargs_forwarded(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        smart_route(
            client, "test",
            provider="anthropic", tracker=tracker,
            temperature=0.5, system="Be helpful",
        )

        kwargs = client.messages.last_kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["system"] == "Be helpful"

    def test_returns_response_unchanged(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        result = smart_route(
            client, "test",
            provider="anthropic", tracker=tracker,
        )

        assert result is resp

    def test_no_task_raises(self, tracker):
        client = _FakeAnthropicClient(_anthropic_response())
        with pytest.raises(ValueError, match="task.*required"):
            smart_route(client, provider="anthropic", tracker=tracker)

    def test_openai_client_works(self, tracker):
        resp = _openai_response()
        client = _FakeOpenAIClient(resp)

        result = smart_route(
            client, "Hello",
            provider="openai", tracker=tracker,
        )

        assert result is resp
        assert client.chat.completions.last_kwargs["model"] is not None
        assert tracker.get_spend(period="total") > 0

    def test_metadata_attached(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        result = smart_route(
            client, "test",
            task_type="extraction", provider="anthropic", tracker=tracker,
        )

        assert hasattr(result, "_llm_budget")
        assert result._llm_budget["task_type"] == "extraction"
        assert result._llm_budget["was_routed"] is True

    def test_explicit_model_not_routed(self, tracker):
        resp = _anthropic_response()
        client = _FakeAnthropicClient(resp)

        result = smart_route(
            client, "test",
            model="claude-3-5-haiku-20241022",
            provider="anthropic", tracker=tracker,
        )

        assert result._llm_budget["was_routed"] is False


# ---------------------------------------------------------------------------
# TestCostAware
# ---------------------------------------------------------------------------


class TestCostAware:
    def test_swaps_model_for_extraction(self, tracker):
        calls = []

        @cost_aware(budget=5.00, tracker=tracker)
        def my_fn(task, model="claude-sonnet-4-20250514", task_type="moderate"):
            calls.append(model)
            return SimpleNamespace(usage=None)

        my_fn("Extract dates", model="claude-sonnet-4-20250514", task_type="extraction")

        assert len(calls) == 1
        used = calls[0]
        # Should have been swapped to a cheaper model
        from llm_budget.pricing import get_registry
        pricing = get_registry().get(used)
        assert pricing.provider == "anthropic"
        assert pricing.capability_tier == "efficient"

    def test_keeps_frontier_for_complex(self, tracker):
        calls = []

        @cost_aware(budget=5.00, tracker=tracker)
        def my_fn(task, model="claude-sonnet-4-20250514", task_type="moderate"):
            calls.append(model)
            return SimpleNamespace(usage=None)

        my_fn("Deep analysis", model="claude-sonnet-4-20250514", task_type="complex")

        assert len(calls) == 1
        used = calls[0]
        from llm_budget.pricing import get_registry
        pricing = get_registry().get(used)
        assert pricing.capability_tier in ("frontier", "reasoning")

    def test_tracks_cost(self, tracker):
        @cost_aware(budget=5.00, tracker=tracker)
        def my_fn(task, model="claude-sonnet-4-20250514", task_type="moderate"):
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

        my_fn("test", model="claude-sonnet-4-20250514", task_type="simple")

        assert tracker.get_spend(period="total") > 0

    def test_no_model_param_noop(self, tracker):
        @cost_aware(budget=5.00, tracker=tracker)
        def my_fn(task):
            return "done"

        result = my_fn("hello")
        assert result == "done"

    def test_preserves_return_value(self, tracker):
        expected = SimpleNamespace(usage=None, data="important")

        @cost_aware(budget=5.00, tracker=tracker)
        def my_fn(task, model="claude-sonnet-4-20250514", task_type="moderate"):
            return expected

        result = my_fn("test", model="claude-sonnet-4-20250514")
        assert result is expected
