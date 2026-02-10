"""Smart routing: automatic model selection based on task type and budget.

Two APIs for zero-friction cost optimization:
- smart_route(): One function call, picks the cheapest suitable model.
- @cost_aware: Decorator that auto-swaps models per task type.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .capabilities import recommend_models
from .decorators import _extract_usage
from .pricing import get_pricing, get_registry
from .tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "deepseek": "deepseek-chat",
    "google": "gemini-2.0-flash",
    "mistral": "mistral-small",
}


# ── Public API ──────────────────────────────────────────────────────────────


def smart_route(
    client: Any,
    task: Union[str, List[Dict[str, Any]], None] = None,
    *,
    task_type: str = "moderate",
    budget: Optional[float] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    provider: Optional[str] = None,
    tracker: Optional[Tracker] = None,
    **api_kwargs: Any,
) -> Any:
    """Route a task to the cheapest suitable model automatically.

    Detects your client (OpenAI/Anthropic), picks the best model for the
    task type, makes the API call, and tracks the cost. One function call.

    Args:
        client: OpenAI() or Anthropic() client instance.
        task: Task as a string or messages list.
        task_type: simple/moderate/complex/coding/math/creative/extraction.
        budget: Budget in USD. Influences model selection when low.
        model: Explicit model override (skips routing, still tracks cost).
        max_tokens: Max output tokens (default 1024).
        provider: Explicit provider override ("openai"/"anthropic").
        tracker: Explicit Tracker instance. Uses default if None.
        **api_kwargs: Forwarded to the API call (temperature, system, etc).

    Returns:
        The raw API response — same type as client.messages.create() or
        client.chat.completions.create() would return.
    """
    if task is None:
        raise ValueError("'task' is required (string or messages list).")

    messages, prompt_text = _normalize_task(task)
    resolved_provider = provider or _detect_provider(client)
    resolved_tracker = tracker or get_tracker()

    # Model selection
    if model is not None:
        selected_model = model
    else:
        selected_model = _select_model(
            prompt_text, task_type, budget, resolved_provider, resolved_tracker
        )

    # Make the API call
    response = _call_client(
        client, resolved_provider, selected_model, messages,
        max_tokens, **api_kwargs,
    )

    # Track cost (never crash on tracking failure)
    _record_cost(resolved_tracker, selected_model, response)

    # Attach metadata
    try:
        response._llm_budget = {
            "model_used": selected_model,
            "task_type": task_type,
            "was_routed": model is None,
        }
    except (AttributeError, TypeError):
        pass

    return response


def cost_aware(
    budget: Optional[float] = None,
    task_type: str = "moderate",
    provider: Optional[str] = None,
    tracker: Optional[Tracker] = None,
) -> Callable:
    """Decorator that auto-selects the cheapest suitable model per call.

    The decorated function must accept ``model`` as a keyword argument
    (or first positional arg). Optionally accepts ``task_type`` to guide
    model selection per-call.

    Example::

        @cost_aware(budget=5.00)
        def my_step(task, model="claude-sonnet-4-20250514", task_type="moderate"):
            return client.messages.create(
                model=model,
                messages=[{"role": "user", "content": task}],
                max_tokens=1024,
            )

        # Automatically uses haiku for extraction:
        result = my_step("Extract dates...", task_type="extraction")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract model (same pattern as @budget in decorators.py:41-42)
            original_model = kwargs.get("model") or (
                args[0] if args else None
            )
            if not original_model or not isinstance(original_model, str):
                return func(*args, **kwargs)

            # Extract task_type from kwargs (fall back to decorator default)
            call_task_type = kwargs.get("task_type", task_type)

            # Detect provider from model name
            resolved_provider = provider
            if resolved_provider is None:
                try:
                    resolved_provider = get_pricing(original_model).provider
                except Exception:
                    return func(*args, **kwargs)

            resolved_tracker = tracker or get_tracker()

            # Select model
            prompt_text = str(kwargs.get("task", "")) or str(
                args[1] if len(args) > 1 else "sample"
            )
            new_model = _select_model(
                prompt_text, call_task_type, budget,
                resolved_provider, resolved_tracker,
            )

            # Swap model if different (decorators.py:77-80 pattern)
            if new_model and new_model != original_model:
                if "model" in kwargs:
                    kwargs["model"] = new_model
                elif args:
                    args = (new_model,) + args[1:]

            # Call the original function
            response = func(*args, **kwargs)

            # Track cost
            used_model = new_model or original_model
            _record_cost(resolved_tracker, used_model, response)

            return response

        return wrapper

    return decorator


# ── Internal helpers ────────────────────────────────────────────────────────


def _detect_provider(client: Any) -> Optional[str]:
    """Auto-detect provider from client type."""
    # Method 1: module path (most reliable)
    mod = (getattr(type(client), "__module__", "") or "").lower()

    # Handle our own tracked wrappers from middleware.py
    if "llm_budget" in mod:
        inner = getattr(client, "_client", None)
        if inner is not None:
            mod = (getattr(type(inner), "__module__", "") or "").lower()

    if mod.startswith("openai"):
        return "openai"
    if mod.startswith("anthropic"):
        return "anthropic"

    # Method 2: duck typing fallback
    has_chat = hasattr(client, "chat")
    has_messages = hasattr(client, "messages")

    if has_chat:
        return "openai"
    if has_messages and not has_chat:
        return "anthropic"

    return None


def _normalize_task(
    task: Union[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Convert task to (messages_list, prompt_text)."""
    if isinstance(task, str):
        return [{"role": "user", "content": task}], task

    if isinstance(task, list):
        parts = []
        for msg in task:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return task, " ".join(parts)

    raise TypeError(f"task must be str or list, got {type(task).__name__}")


def _pick_model(recs: Dict[str, Any]) -> Optional[str]:
    """Extract the best model name from recommend_models() result."""
    return (
        recs.get("cheapest_safe")
        or recs.get("best_value")
        or recs.get("cheapest_risky")
    )


def _select_model(
    prompt_text: str,
    task_type: str,
    budget: Optional[float],
    provider: Optional[str],
    tracker: Tracker,
) -> str:
    """Select the best model for the task and provider."""
    registry = get_registry()

    # Filter to provider's models
    if provider:
        available = [m.name for m in registry.list_models(provider=provider)]
    else:
        available = None  # all models

    # Budget remaining
    budget_remaining = None
    if budget is not None:
        spent = tracker.get_spend(period="today")
        budget_remaining = max(0.0, budget - spent)

    recs = recommend_models(
        prompt=prompt_text,
        task_type=task_type,
        budget_remaining=budget_remaining,
        available_models=available,
    )

    selected = _pick_model(recs)

    if selected is None:
        selected = _PROVIDER_DEFAULTS.get(provider or "", "gpt-4o-mini")
        logger.warning(
            "No model recommended for task_type=%s provider=%s; "
            "falling back to %s",
            task_type, provider, selected,
        )

    return selected


def _call_client(
    client: Any,
    provider: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
) -> Any:
    """Dispatch the API call to the correct client method."""
    # Anthropic: system is a top-level kwarg, not in messages
    if provider == "anthropic" or (
        provider is None
        and hasattr(client, "messages")
        and not hasattr(client, "chat")
    ):
        return client.messages.create(
            model=model, messages=messages,
            max_tokens=max_tokens, **kwargs,
        )

    # OpenAI (default)
    if hasattr(client, "chat"):
        return client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, **kwargs,
        )

    raise TypeError(
        f"Unsupported client: {type(client).__name__}. "
        f"Expected OpenAI or Anthropic client."
    )


def _record_cost(
    tracker: Tracker,
    model: str,
    response: Any,
) -> None:
    """Track cost from a response. Never raises."""
    try:
        actual_in, actual_out = _extract_usage(response)
        if actual_in is not None and actual_out is not None:
            pricing = get_pricing(model)
            cost = (
                actual_in * pricing.input_cost_per_token
                + actual_out * pricing.output_cost_per_token
            )
            tracker.record(
                model=model,
                input_tokens=actual_in,
                output_tokens=actual_out,
                cost_usd=cost,
            )
    except Exception:
        logger.debug("Failed to record cost for %s", model, exc_info=True)
