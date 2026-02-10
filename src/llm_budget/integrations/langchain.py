"""LangChain integration: budget-aware callback handler."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

from ..enforcer import BudgetEnforcer, BudgetPolicy
from ..estimator import EstimationResult, estimate as estimate_cost
from ..exceptions import BudgetExceeded
from ..pricing import get_pricing
from ..tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stub so class definition works without langchain installed."""

    class LLMResult:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _langchain_messages_to_dicts(
    messages: Sequence[Any],
) -> List[Dict[str, str]]:
    """Convert LangChain message objects to standard dict format.

    Handles HumanMessage, AIMessage, SystemMessage, ChatMessage, plain
    strings, and dicts.
    """
    result: List[Dict[str, str]] = []
    role_map = {"human": "user", "ai": "assistant", "system": "system"}

    for msg in messages:
        if isinstance(msg, str):
            result.append({"role": "user", "content": msg})
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            role = role_map.get(msg.type, msg.type)
            result.append({"role": role, "content": str(msg.content)})
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"role": "user", "content": str(msg)})
    return result


def _extract_model_from_serialized(
    serialized: Dict[str, Any],
    **kwargs: Any,
) -> Optional[str]:
    """Extract model name from LangChain's serialized dict or invocation params."""
    ser_kwargs = serialized.get("kwargs", {})
    model = ser_kwargs.get("model_name") or ser_kwargs.get("model")
    if model:
        return model

    inv_params = kwargs.get("invocation_params", {})
    model = inv_params.get("model_name") or inv_params.get("model")
    return model or None


# ---------------------------------------------------------------------------
# Main callback handler
# ---------------------------------------------------------------------------


class BudgetCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler with budget estimation and enforcement.

    Usage::

        from llm_budget.integrations.langchain import BudgetCallbackHandler

        handler = BudgetCallbackHandler(max_cost=5.0, period="daily")
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        llm.invoke("Hello")

        print(handler.total_cost)
        print(handler.total_tokens)
    """

    def __init__(
        self,
        max_cost: Optional[float] = None,
        period: str = "monthly",
        on_exceed: str = "raise",
        alert_at: float = 0.8,
        tracker: Optional[Tracker] = None,
        model: Optional[str] = None,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain integration requires 'langchain-core'. "
                "Install with: pip install llm-budget[langchain]"
            )
        super().__init__()
        self.raise_error = True  # Propagate exceptions (e.g. BudgetExceeded)
        self._tracker = tracker or get_tracker()
        self._model_override = model
        self._last_estimation: Optional[EstimationResult] = None
        self._last_model: Optional[str] = None
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._call_count: int = 0

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

    # -- LangChain lifecycle hooks --------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called before an LLM call. Runs pre-flight estimation + enforcement."""
        model = (
            self._model_override
            or _extract_model_from_serialized(serialized, **kwargs)
        )
        if not model:
            logger.warning(
                "Could not extract model name from LangChain serialized dict; "
                "skipping pre-flight budget check"
            )
            return

        self._last_model = model
        messages_text = "\n".join(prompts) if prompts else ""
        self._run_preflight(messages_text, model)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Called before a chat model call (structured messages)."""
        model = (
            self._model_override
            or _extract_model_from_serialized(serialized, **kwargs)
        )
        if not model:
            logger.warning(
                "Could not extract model name; skipping pre-flight check"
            )
            return

        self._last_model = model
        flat_messages = messages[0] if messages else []
        dict_messages = _langchain_messages_to_dicts(flat_messages)
        self._run_preflight(dict_messages, model)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called after an LLM call. Records actual cost to tracker."""
        # Prefer actual model name from response (providers may differ from request)
        llm_output = getattr(response, "llm_output", None) or {}
        model = (
            llm_output.get("model_name")
            or llm_output.get("model")
            or self._last_model
            or "unknown"
        )
        self._call_count += 1

        try:
            input_tokens, output_tokens, cost = self._extract_llm_result_usage(
                response, model
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
        except Exception as exc:
            logger.warning("Failed to record LangChain LLM spend: %s", exc)
            if self._last_estimation:
                est = self._last_estimation
                self._total_cost += est.estimated_cost
                self._tracker.record(
                    model=model,
                    input_tokens=est.input_tokens,
                    output_tokens=est.output_tokens,
                    cost_usd=est.estimated_cost,
                )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called on LLM error. No-op."""
        logger.debug("LLM call errored: %s", error)

    # -- Internal helpers -----------------------------------------------------

    def _run_preflight(
        self,
        messages: Any,
        model: str,
    ) -> None:
        """Run pre-flight estimation and budget enforcement."""
        try:
            estimation = estimate_cost(messages, model)
            self._last_estimation = estimation
            logger.debug("Pre-flight estimate: %s", estimation)

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
            logger.warning("Pre-flight estimation failed: %s", exc)

    def _extract_llm_result_usage(
        self,
        response: Any,
        model: str,
    ) -> tuple[int, int, float]:
        """Extract token usage from LLMResult.

        Checks multiple locations since different providers store usage
        differently:
        1. llm_output["token_usage"] (OpenAI via LangChain)
        2. llm_output["usage"] (Anthropic via LangChain)
        3. generations[0][0].message.usage_metadata (universal fallback)
        """
        input_tokens = 0
        output_tokens = 0

        llm_output = getattr(response, "llm_output", None) or {}

        # OpenAI format: llm_output["token_usage"]
        token_usage = llm_output.get("token_usage", {})
        if token_usage:
            input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)

        # Anthropic format: llm_output["usage"]
        if input_tokens == 0 and output_tokens == 0:
            usage = llm_output.get("usage", {})
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        # Universal fallback: generations[0][0].message.usage_metadata
        if input_tokens == 0 and output_tokens == 0:
            generations = getattr(response, "generations", None)
            if generations and generations[0]:
                gen = generations[0][0]
                msg = getattr(gen, "message", None)
                if msg:
                    um = getattr(msg, "usage_metadata", None)
                    if um and isinstance(um, dict):
                        input_tokens = um.get("input_tokens", 0)
                        output_tokens = um.get("output_tokens", 0)

        pricing = get_pricing(model)
        cost = (
            input_tokens * pricing.input_cost_per_token
            + output_tokens * pricing.output_cost_per_token
        )
        return input_tokens, output_tokens, cost


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def budget_callback(
    max_cost: Optional[float] = None,
    period: str = "monthly",
    on_exceed: str = "raise",
    alert_at: float = 0.8,
    tracker: Optional[Tracker] = None,
    model: Optional[str] = None,
):
    """Context manager that creates a BudgetCallbackHandler.

    Usage::

        from llm_budget.integrations.langchain import budget_callback

        with budget_callback(max_cost=1.0, period="daily") as handler:
            llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
            result = llm.invoke("Hello")
            print(f"Cost so far: ${handler.total_cost:.4f}")
    """
    handler = BudgetCallbackHandler(
        max_cost=max_cost,
        period=period,
        on_exceed=on_exceed,
        alert_at=alert_at,
        tracker=tracker,
        model=model,
    )
    try:
        yield handler
    finally:
        logger.debug(
            "Budget callback session ended. Total cost: $%.4f, calls: %d",
            handler.total_cost,
            handler.call_count,
        )
