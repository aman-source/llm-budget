"""CrewAI integration: budget-aware hook registration."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, List, Optional

from ..enforcer import BudgetEnforcer, BudgetPolicy
from ..estimator import EstimationResult, estimate as estimate_cost
from ..exceptions import BudgetExceeded
from ..pricing import get_pricing
from ..tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)

try:
    from crewai.hooks import (
        register_before_llm_call_hook,
        register_after_llm_call_hook,
    )

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_crewai_prompt(prompt: Any) -> str:
    """Normalize a CrewAI prompt to a string for token counting."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("content", str(item)))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(prompt)


# ---------------------------------------------------------------------------
# Main hook class
# ---------------------------------------------------------------------------


class BudgetHooks:
    """CrewAI budget hooks for pre-flight enforcement and post-flight tracking.

    Usage::

        from llm_budget.integrations.crewai import BudgetHooks

        hooks = BudgetHooks(max_cost=10.0, period="daily")
        hooks.register()

        # ... run CrewAI agents ...

        hooks.unregister()
        print(f"Total cost: ${hooks.total_cost:.4f}")
    """

    def __init__(
        self,
        max_cost: Optional[float] = None,
        period: str = "monthly",
        on_exceed: str = "raise",
        alert_at: float = 0.8,
        tracker: Optional[Tracker] = None,
        model: Optional[str] = None,
        default_model: str = "gpt-4o",
    ) -> None:
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI integration requires 'crewai'. "
                "Install with: pip install llm-budget[crewai]"
            )
        self._tracker = tracker or get_tracker()
        self._model_override = model
        self._default_model = default_model
        self._last_estimation: Optional[EstimationResult] = None
        self._last_model: Optional[str] = None
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._call_count: int = 0
        self._registered: bool = False

        self._policy: Optional[BudgetPolicy] = None
        self._enforcer: Optional[BudgetEnforcer] = None
        if max_cost is not None:
            self._policy = BudgetPolicy(
                max_cost=max_cost,
                period=period,
                on_exceed=on_exceed,
                alert_at=alert_at,
                model=model,
            )
            self._enforcer = BudgetEnforcer(self._tracker)

    # -- Properties -----------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def tracker(self) -> Tracker:
        return self._tracker

    @property
    def last_estimation(self) -> Optional[EstimationResult]:
        return self._last_estimation

    @property
    def is_registered(self) -> bool:
        return self._registered

    # -- Hook methods ---------------------------------------------------------

    def before_call(self, prompt: Any, **kwargs: Any) -> None:
        """Pre-flight hook: estimate cost and enforce budget."""
        model = (
            self._model_override
            or kwargs.get("model")
            or self._default_model
        )
        self._last_model = model
        prompt_text = _normalize_crewai_prompt(prompt)

        try:
            estimation = estimate_cost(prompt_text, model)
            self._last_estimation = estimation
            logger.debug("CrewAI pre-flight estimate: %s", estimation)

            if self._enforcer and self._policy:
                result = self._enforcer.check(estimation, self._policy)
                if not result.allowed:
                    raise BudgetExceeded(
                        f"Budget exceeded (skip mode). "
                        f"Spent: ${result.current_spend:.4f} of "
                        f"${self._policy.max_cost:.2f}",
                        estimated_cost=estimation.estimated_cost,
                        remaining_budget=result.remaining_budget,
                        period=self._policy.period,
                    )
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.warning("CrewAI pre-flight estimation failed: %s", exc)

    def after_call(self, response: Any, **kwargs: Any) -> None:
        """Post-flight hook: extract usage and record to tracker."""
        from ..decorators import _extract_usage

        model = self._last_model or self._default_model
        self._call_count += 1

        try:
            input_tokens, output_tokens = _extract_usage(response)
            if input_tokens is not None and output_tokens is not None:
                pricing = get_pricing(model)
                cost = (
                    input_tokens * pricing.input_cost_per_token
                    + output_tokens * pricing.output_cost_per_token
                )
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self._total_cost += cost
                self._tracker.record(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            elif self._last_estimation:
                est = self._last_estimation
                self._total_cost += est.estimated_cost
                self._tracker.record(
                    model=model,
                    input_tokens=est.input_tokens,
                    output_tokens=est.output_tokens,
                    cost_usd=est.estimated_cost,
                )
        except Exception as exc:
            logger.warning("Failed to record CrewAI spend: %s", exc)

    # -- Registration ---------------------------------------------------------

    def register(self) -> None:
        """Register hooks with CrewAI's global hook system."""
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI integration requires 'crewai'. "
                "Install with: pip install llm-budget[crewai]"
            )
        register_before_llm_call_hook(self.before_call)
        register_after_llm_call_hook(self.after_call)
        self._registered = True
        logger.debug("CrewAI budget hooks registered")

    def unregister(self) -> None:
        """Mark hooks as unregistered.

        Note: CrewAI may not support selective unregistration of
        previously registered hooks. This is a best-effort operation.
        """
        self._registered = False
        logger.debug("CrewAI budget hooks marked as unregistered")

    # -- Context manager ------------------------------------------------------

    def __enter__(self) -> BudgetHooks:
        self.register()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unregister()
        logger.debug(
            "CrewAI budget session ended. Total cost: $%.4f, calls: %d",
            self.total_cost,
            self.call_count,
        )


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------


@contextmanager
def track_crewai(
    max_cost: Optional[float] = None,
    period: str = "monthly",
    on_exceed: str = "raise",
    alert_at: float = 0.8,
    tracker: Optional[Tracker] = None,
    model: Optional[str] = None,
    default_model: str = "gpt-4o",
):
    """Context manager that creates BudgetHooks and registers them.

    Usage::

        from llm_budget.integrations.crewai import track_crewai

        with track_crewai(max_cost=10.0) as hooks:
            crew = Crew(agents=[...], tasks=[...])
            crew.kickoff()
            print(f"Total cost: ${hooks.total_cost:.4f}")
    """
    hooks = BudgetHooks(
        max_cost=max_cost,
        period=period,
        on_exceed=on_exceed,
        alert_at=alert_at,
        tracker=tracker,
        model=model,
        default_model=default_model,
    )
    with hooks:
        yield hooks
