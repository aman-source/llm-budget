"""Custom exceptions for llm-budget."""
from __future__ import annotations

from typing import Optional


class LLMBudgetError(Exception):
    """Base exception for all llm-budget errors."""


class BudgetExceeded(LLMBudgetError):
    """Raised when a request would exceed the configured budget."""

    def __init__(
        self,
        message: str,
        *,
        estimated_cost: float = 0.0,
        remaining_budget: float = 0.0,
        period: str = "",
        suggested_model: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.estimated_cost = estimated_cost
        self.remaining_budget = remaining_budget
        self.period = period
        self.suggested_model = suggested_model


class ModelNotFoundError(LLMBudgetError):
    """Raised when a model is not found in the pricing database."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' not found in pricing database")
        self.model = model


class TokenizationError(LLMBudgetError):
    """Raised when token counting fails."""
