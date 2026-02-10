"""Pre-flight cost estimation engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from .pricing import get_pricing, ModelPricing
from .tokenizer import Messages, count_tokens

logger = logging.getLogger(__name__)


@dataclass
class EstimationResult:
    """Result of a cost estimation."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    input_cost: float
    output_cost: float
    max_input_tokens: int
    max_output_tokens: int
    is_output_estimated: bool

    @property
    def breakdown(self) -> dict[str, str]:
        return {
            "input": f"${self.input_cost:.4f}",
            "output": f"${self.output_cost:.4f}",
        }

    def __str__(self) -> str:
        est_marker = " (estimated)" if self.is_output_estimated else ""
        return (
            f"{self.model}: ~{self.input_tokens} input + "
            f"~{self.output_tokens} output{est_marker} tokens = "
            f"${self.estimated_cost:.6f}"
        )


def _get_learned_ratio(model: str) -> Optional[float]:
    """Try to get a learned output/input ratio from the tracker.

    Uses lazy import to avoid circular dependency (estimator ↔ tracker).
    Returns None if the tracker has insufficient data (cold start).
    """
    try:
        from .tracker import get_tracker
        tracker = get_tracker()
        return tracker.get_output_ratio(model)
    except Exception:
        return None


def _estimate_output_tokens(
    input_tokens: int,
    max_output: int,
    model: Optional[str] = None,
) -> int:
    """Estimate output tokens, using learned history when available.

    Strategy:
      1. If tracker has enough history for this model, use the learned
         median output/input ratio (adaptive).
      2. Otherwise, fall back to static heuristic:
         min(max_output_tokens, input_tokens * 0.75, 1000).

    The result is always clamped to [1, max_output].
    """
    learned_ratio = _get_learned_ratio(model) if model else None

    if learned_ratio is not None and input_tokens > 0:
        estimated = int(input_tokens * learned_ratio)
        logger.debug(
            "Adaptive estimate for %s: ratio=%.3f → %d output tokens",
            model, learned_ratio, estimated,
        )
    else:
        estimated = int(min(max_output, input_tokens * 0.75, 1000))

    return max(1, min(estimated, max_output))


def estimate(
    messages: Union[str, Messages],
    model: str = "gpt-4o",
    expected_output_tokens: Optional[int] = None,
) -> EstimationResult:
    """Estimate the cost of an LLM API call before making it.

    Args:
        messages: Input text or chat messages.
        model: Model name (must be in the pricing database).
        expected_output_tokens: If None, uses a heuristic estimate.

    Returns:
        EstimationResult with full cost breakdown.
    """
    pricing = get_pricing(model)
    input_tokens = count_tokens(messages, model)

    is_estimated = expected_output_tokens is None
    if is_estimated:
        output_tokens = _estimate_output_tokens(
            input_tokens, pricing.max_output_tokens, model=model
        )
    else:
        output_tokens = expected_output_tokens

    if input_tokens > pricing.max_input_tokens:
        logger.warning(
            "Input tokens (%d) exceed model %s max (%d) — API call will likely fail",
            input_tokens, model, pricing.max_input_tokens,
        )

    input_cost = input_tokens * pricing.input_cost_per_token
    output_cost = output_tokens * pricing.output_cost_per_token

    return EstimationResult(
        model=model,
        provider=pricing.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        max_input_tokens=pricing.max_input_tokens,
        max_output_tokens=pricing.max_output_tokens,
        is_output_estimated=is_estimated,
    )


def compare(
    messages: Union[str, Messages],
    models: list[str],
    expected_output_tokens: Optional[int] = None,
    print_table: bool = True,
) -> list[EstimationResult]:
    """Compare costs across multiple models for the same input.

    Args:
        messages: Input text or chat messages.
        models: List of model names to compare.
        expected_output_tokens: If None, uses a heuristic estimate.
        print_table: Whether to print a formatted comparison table.

    Returns:
        Results sorted by estimated_cost (cheapest first).
    """
    results = [
        estimate(messages, model, expected_output_tokens) for model in models
    ]
    results.sort(key=lambda r: r.estimated_cost)
    if print_table:
        _print_comparison_table(results)
    return results


def _print_comparison_table(results: list[EstimationResult]) -> None:
    """Print a comparison table. Uses rich if available, else plain text."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Model Cost Comparison")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Provider", style="dim")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Est. Output", justify="right")
        table.add_column("Input Cost", justify="right")
        table.add_column("Output Cost", justify="right")
        table.add_column("Total Cost", justify="right", style="green")

        for r in results:
            table.add_row(
                r.model,
                r.provider,
                str(r.input_tokens),
                f"~{r.output_tokens}",
                f"${r.input_cost:.6f}",
                f"${r.output_cost:.6f}",
                f"${r.estimated_cost:.6f}",
            )

        Console().print(table)
    except ImportError:
        header = (
            f"{'Model':<30} {'Input Tok':>10} {'Est. Output':>12} "
            f"{'Est. Cost':>12}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.model:<30} {r.input_tokens:>10} "
                f"{'~' + str(r.output_tokens):>12} "
                f"${r.estimated_cost:>11.6f}"
            )
