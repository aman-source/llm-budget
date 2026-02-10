"""Transparent client wrappers for OpenAI and Anthropic."""
from __future__ import annotations

import logging
from typing import Any, Optional

from .pricing import get_pricing
from .tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI wrappers
# ---------------------------------------------------------------------------

class _TrackedChatCompletions:
    """Proxy for OpenAI client.chat.completions that tracks spend."""

    def __init__(self, original: Any, tracker: Tracker) -> None:
        self._original = original
        self._tracker = tracker

    def create(self, **kwargs: Any) -> Any:
        """Wrap client.chat.completions.create() to record spend."""
        response = self._original.create(**kwargs)

        model = kwargs.get("model", "unknown")
        try:
            usage = response.usage
            if usage:
                pricing = get_pricing(model)
                cost = (
                    usage.prompt_tokens * pricing.input_cost_per_token
                    + usage.completion_tokens * pricing.output_cost_per_token
                )
                self._tracker.record(
                    model=model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cost_usd=cost,
                )
        except Exception as exc:
            logger.warning("Failed to record OpenAI spend: %s", exc)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _TrackedChat:
    """Proxy for OpenAI client.chat."""

    def __init__(self, original: Any, tracker: Tracker) -> None:
        self._original = original
        self.completions = _TrackedChatCompletions(
            original.completions, tracker
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _TrackedOpenAIClient:
    """Proxy for an OpenAI client that tracks all chat.completions calls."""

    def __init__(self, client: Any, tracker: Tracker) -> None:
        self._client = client
        self.chat = _TrackedChat(client.chat, tracker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Anthropic wrappers
# ---------------------------------------------------------------------------

class _TrackedAnthropicMessages:
    """Proxy for Anthropic client.messages that tracks spend."""

    def __init__(self, original: Any, tracker: Tracker) -> None:
        self._original = original
        self._tracker = tracker

    def create(self, **kwargs: Any) -> Any:
        """Wrap client.messages.create() to record spend."""
        response = self._original.create(**kwargs)

        model = kwargs.get("model", "unknown")
        try:
            usage = response.usage
            if usage:
                pricing = get_pricing(model)
                cost = (
                    usage.input_tokens * pricing.input_cost_per_token
                    + usage.output_tokens * pricing.output_cost_per_token
                )
                self._tracker.record(
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                )
        except Exception as exc:
            logger.warning("Failed to record Anthropic spend: %s", exc)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _TrackedAnthropicClient:
    """Proxy for an Anthropic client that tracks all messages calls."""

    def __init__(self, client: Any, tracker: Tracker) -> None:
        self._client = client
        self.messages = _TrackedAnthropicMessages(client.messages, tracker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def track_openai(
    client: Any,
    tracker: Optional[Tracker] = None,
) -> _TrackedOpenAIClient:
    """Wrap an OpenAI client to automatically track spend.

    Args:
        client: An ``openai.OpenAI()`` instance.
        tracker: Optional Tracker instance. Uses the default if not given.

    Returns:
        A wrapped client that behaves identically but records spend.
    """
    return _TrackedOpenAIClient(client, tracker or get_tracker())


def track_anthropic(
    client: Any,
    tracker: Optional[Tracker] = None,
) -> _TrackedAnthropicClient:
    """Wrap an Anthropic client to automatically track spend.

    Args:
        client: An ``anthropic.Anthropic()`` instance.
        tracker: Optional Tracker instance. Uses the default if not given.

    Returns:
        A wrapped client that behaves identically but records spend.
    """
    return _TrackedAnthropicClient(client, tracker or get_tracker())
