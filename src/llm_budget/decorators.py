"""@budget() and @estimate() decorators."""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

from .enforcer import BudgetEnforcer, BudgetPolicy
from .estimator import estimate as estimate_cost
from .exceptions import BudgetExceeded
from .pricing import get_pricing
from .tracker import get_tracker

logger = logging.getLogger(__name__)


def budget(
    max_cost: float,
    period: str = "daily",
    on_exceed: str = "raise",
    alert_at: float = 0.8,
    track_model: Optional[str] = None,
) -> Callable:
    """Decorator that enforces a budget on a function making LLM calls.

    The decorated function should accept 'model' and 'messages' as keyword
    arguments (or as the first two positional arguments).

    Args:
        max_cost: Maximum allowed spend for the period.
        period: 'hourly', 'daily', 'weekly', 'monthly', or 'total'.
        on_exceed: 'raise', 'warn', 'skip', or 'downgrade:model_name'.
        alert_at: Fraction (0-1) at which to emit a warning.
        track_model: If set, only track/enforce budget for this model.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract model and messages from args/kwargs
            model = track_model or kwargs.get("model") or (
                args[0] if args else None
            )
            messages = kwargs.get("messages") or (
                args[1] if len(args) > 1 else None
            )

            if not model or not messages:
                logger.warning(
                    "Could not extract model/messages; running without budget check"
                )
                return func(*args, **kwargs)

            # Pre-flight estimation
            estimation = estimate_cost(messages, model)

            # Budget enforcement
            tracker = get_tracker()
            enforcer = BudgetEnforcer(tracker)
            policy = BudgetPolicy(
                max_cost=max_cost,
                period=period,
                on_exceed=on_exceed,
                alert_at=alert_at,
                model=track_model,
            )
            try:
                result = enforcer.check(estimation, policy)
            except BudgetExceeded as exc:
                if exc.suggested_model:
                    # Downgrade mode: swap model and proceed
                    logger.info(
                        "Downgrading from %s to %s (budget exceeded)",
                        model, exc.suggested_model,
                    )
                    model = exc.suggested_model
                    if "model" in kwargs:
                        kwargs["model"] = model
                    elif args:
                        args = (model,) + args[1:]
                else:
                    raise
            else:
                if not result.allowed:
                    logger.info("Skipping call to %s (budget exceeded)", model)
                    return None

            # Execute the wrapped function
            response = func(*args, **kwargs)

            # Post-flight: record actual cost
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
            else:
                # Fallback: record estimated cost
                tracker.record(
                    model=model,
                    input_tokens=estimation.input_tokens,
                    output_tokens=estimation.output_tokens,
                    cost_usd=estimation.estimated_cost,
                )

            return response

        return wrapper

    return decorator


def estimate_decorator(
    model: Optional[str] = None,
    log_level: int = logging.INFO,
) -> Callable:
    """Decorator that logs a cost estimate before each call.

    Lighter weight than @budget — no enforcement, no tracking.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_model = model or kwargs.get("model") or (
                args[0] if args else None
            )
            messages = kwargs.get("messages") or (
                args[1] if len(args) > 1 else None
            )
            if call_model and messages:
                est = estimate_cost(messages, call_model)
                logger.log(log_level, "Estimated cost: %s", est)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _extract_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    """Try to extract token usage from an API response.

    Supports OpenAI, Anthropic, and dict responses.
    Returns (input_tokens, output_tokens) or (None, None).
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None, None

        # OpenAI format
        inp = getattr(usage, "prompt_tokens", None)
        out = getattr(usage, "completion_tokens", None)
        if inp is not None:
            return int(inp), int(out or 0)

        # Anthropic format
        inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "output_tokens", None)
        if inp is not None:
            return int(inp), int(out or 0)

        # Dict format
        if isinstance(usage, dict):
            inp = usage.get("prompt_tokens") or usage.get("input_tokens")
            out = usage.get("completion_tokens") or usage.get("output_tokens")
            if inp is not None:
                return int(inp), int(out or 0)
    except Exception:
        logger.debug("Failed to extract usage from response", exc_info=True)

    return None, None
