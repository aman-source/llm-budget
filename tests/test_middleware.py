"""Tests for the middleware module."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_budget.middleware import (
    _TrackedAnthropicClient,
    _TrackedChatCompletions,
    _TrackedOpenAIClient,
    track_anthropic,
    track_openai,
)
from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    t = Tracker(db_path=str(tmp_path / "test.db"))
    yield t
    t.close()


# ---------------------------------------------------------------------------
# OpenAI middleware
# ---------------------------------------------------------------------------

class _FakeCompletions:
    """Minimal stand-in for client.chat.completions."""

    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeChat:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response):
        self.chat = _FakeChat(response)
        self.api_key = "sk-test"


class TestTrackedOpenAI:
    def test_records_spend(self, tracker):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        client = _FakeOpenAIClient(response)
        tracked = track_openai(client, tracker=tracker)

        result = tracked.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        assert result is response
        spend = tracker.get_spend(period="total")
        assert spend > 0

    def test_cost_calculation(self, tracker):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500)
        )
        client = _FakeOpenAIClient(response)
        tracked = track_openai(client, tracker=tracker)
        tracked.chat.completions.create(model="gpt-4o", messages=[])

        # gpt-4o: input $2.5e-6/tok, output $1.0e-5/tok
        expected = 1000 * 2.5e-6 + 500 * 1.0e-5
        spend = tracker.get_spend(period="total")
        assert abs(spend - expected) < 1e-9

    def test_no_usage(self, tracker):
        response = SimpleNamespace(usage=None)
        client = _FakeOpenAIClient(response)
        tracked = track_openai(client, tracker=tracker)
        tracked.chat.completions.create(model="gpt-4o", messages=[])

        assert tracker.get_spend(period="total") == 0.0

    def test_missing_usage_attr(self, tracker):
        response = SimpleNamespace()  # no .usage at all
        client = _FakeOpenAIClient(response)
        tracked = track_openai(client, tracker=tracker)
        tracked.chat.completions.create(model="gpt-4o", messages=[])

        assert tracker.get_spend(period="total") == 0.0

    def test_delegates_attributes(self, tracker):
        response = SimpleNamespace(usage=None)
        client = _FakeOpenAIClient(response)
        tracked = track_openai(client, tracker=tracker)

        # api_key should be delegated to original client
        assert tracked.api_key == "sk-test"

    def test_uses_default_tracker_when_none(self):
        response = SimpleNamespace(usage=None)
        client = _FakeOpenAIClient(response)
        # Should not raise — uses get_tracker() internally
        tracked = track_openai(client)
        assert tracked is not None


# ---------------------------------------------------------------------------
# Anthropic middleware
# ---------------------------------------------------------------------------

class _FakeAnthropicMessages:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeAnthropicMessages(response)
        self.api_key = "sk-ant-test"


class TestTrackedAnthropic:
    def test_records_spend(self, tracker):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=200, output_tokens=100)
        )
        client = _FakeAnthropicClient(response)
        tracked = track_anthropic(client, tracker=tracker)

        result = tracked.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is response
        spend = tracker.get_spend(period="total")
        assert spend > 0

    def test_cost_calculation(self, tracker):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500)
        )
        client = _FakeAnthropicClient(response)
        tracked = track_anthropic(client, tracker=tracker)
        tracked.messages.create(model="claude-sonnet-4-20250514", messages=[])

        # claude-sonnet-4: input $3.0e-6/tok, output $1.5e-5/tok
        expected = 1000 * 3.0e-6 + 500 * 1.5e-5
        spend = tracker.get_spend(period="total")
        assert abs(spend - expected) < 1e-9

    def test_no_usage(self, tracker):
        response = SimpleNamespace(usage=None)
        client = _FakeAnthropicClient(response)
        tracked = track_anthropic(client, tracker=tracker)
        tracked.messages.create(model="claude-sonnet-4-20250514", messages=[])

        assert tracker.get_spend(period="total") == 0.0

    def test_delegates_attributes(self, tracker):
        response = SimpleNamespace(usage=None)
        client = _FakeAnthropicClient(response)
        tracked = track_anthropic(client, tracker=tracker)

        assert tracked.api_key == "sk-ant-test"
