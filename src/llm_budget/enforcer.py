"""Budget enforcement logic."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

from .estimator import EstimationResult
from .exceptions import BudgetExceeded
from .tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)


@dataclass
class BudgetPolicy:
    """A budget policy configuration."""

    max_cost: float
    period: str = "monthly"
    on_exceed: str = "raise"  # "raise" | "warn" | "skip" | "downgrade:model"
    alert_at: float = 0.8
    model: Optional[str] = None  # restrict enforcement to a specific model


@dataclass
class EnforcementResult:
    """Result of a budget check."""

    allowed: bool
    current_spend: float
    remaining_budget: float
    estimated_cost: float
    warning: Optional[str] = None
    downgrade_model: Optional[str] = None


class BudgetEnforcer:
    """Checks estimated costs against budget policies."""

    def __init__(self, tracker: Optional[Tracker] = None) -> None:
        self._tracker = tracker or get_tracker()

    def check(
        self,
        estimation: EstimationResult,
        policy: BudgetPolicy,
    ) -> EnforcementResult:
        """Check if an estimated call is within budget.

        Returns EnforcementResult. Raises BudgetExceeded if on_exceed='raise'
        and the budget is exceeded.
        """
        current_spend = self._tracker.get_spend(
            period=policy.period,
            model=policy.model,
        )
        remaining = policy.max_cost - current_spend

        # Early warning at alert threshold
        warn_msg: Optional[str] = None
        if policy.max_cost > 0 and current_spend >= policy.max_cost * policy.alert_at:
            pct = (current_spend / policy.max_cost) * 100
            warn_msg = (
                f"Budget {pct:.0f}% used "
                f"(${current_spend:.4f} of ${policy.max_cost:.2f})"
            )
            logger.warning(warn_msg)
            warnings.warn(warn_msg, UserWarning, stacklevel=3)

        # Budget exceeded?
        if estimation.estimated_cost > remaining:
            return self._handle_exceeded(
                estimation, policy, current_spend, remaining, warn_msg
            )

        return EnforcementResult(
            allowed=True,
            current_spend=current_spend,
            remaining_budget=remaining,
            estimated_cost=estimation.estimated_cost,
            warning=warn_msg,
        )

    def _handle_exceeded(
        self,
        estimation: EstimationResult,
        policy: BudgetPolicy,
        current_spend: float,
        remaining: float,
        warn_msg: Optional[str],
    ) -> EnforcementResult:
        on_exceed = policy.on_exceed

        if on_exceed == "raise":
            raise BudgetExceeded(
                f"{policy.period.capitalize()} budget of ${policy.max_cost:.2f} "
                f"exceeded. Spent: ${current_spend:.2f}, "
                f"estimated next call: ${estimation.estimated_cost:.4f}",
                estimated_cost=estimation.estimated_cost,
                remaining_budget=remaining,
                period=policy.period,
            )

        if on_exceed == "warn":
            msg = (
                f"Over budget but proceeding (warn mode). "
                f"Spent: ${current_spend:.4f} of ${policy.max_cost:.2f}"
            )
            logger.warning(msg)
            return EnforcementResult(
                allowed=True,
                current_spend=current_spend,
                remaining_budget=remaining,
                estimated_cost=estimation.estimated_cost,
                warning=msg,
            )

        if on_exceed == "skip":
            logger.info(
                "Skipping call — budget exceeded "
                "(${%.4f} of $%.2f)", current_spend, policy.max_cost
            )
            return EnforcementResult(
                allowed=False,
                current_spend=current_spend,
                remaining_budget=remaining,
                estimated_cost=estimation.estimated_cost,
            )

        if on_exceed.startswith("downgrade:"):
            downgrade_to = on_exceed.split(":", 1)[1]
            logger.info("Downgrading to %s due to budget constraints", downgrade_to)
            raise BudgetExceeded(
                f"Budget exceeded. Suggesting downgrade to {downgrade_to}",
                estimated_cost=estimation.estimated_cost,
                remaining_budget=remaining,
                period=policy.period,
                suggested_model=downgrade_to,
            )

        # Unknown on_exceed mode — default to raise
        raise BudgetExceeded(
            f"Budget exceeded (unknown mode: {on_exceed})",
            estimated_cost=estimation.estimated_cost,
            remaining_budget=remaining,
            period=policy.period,
        )
