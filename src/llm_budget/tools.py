"""
Agent tool interface for llm-budget.

Exposes estimate_cost, compare_models, and check_budget as callable tools
that LLM agents can use to make cost-quality-aware decisions.

Supports OpenAI function calling, Anthropic tool_use, and generic dict formats.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .capabilities import (
    TASK_REQUIREMENTS,
    TIER_HIERARCHY,
    calculate_cost_quality_score,
    get_capability_summary,
    is_model_suitable,
    recommend_models,
)
from .estimator import EstimationResult, compare, estimate
from .exceptions import ModelNotFoundError
from .pricing import get_registry
from .tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)

# ── Tool parameter schema (shared) ──────────────────────────────────────────

_TASK_TYPE_ENUM = [
    "simple", "moderate", "complex", "coding", "math", "creative", "extraction",
]

_ESTIMATE_COST_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "The prompt or message content you plan to send"
            ),
        },
        "model": {
            "type": "string",
            "description": (
                "The model to estimate cost for "
                "(e.g. gpt-4o, gpt-4o-mini, claude-sonnet-4-20250514)"
            ),
        },
        "task_type": {
            "type": "string",
            "enum": _TASK_TYPE_ENUM,
            "description": (
                "The type of task. 'simple' = formatting, extraction, "
                "translation. 'moderate' = summarization, Q&A. "
                "'complex' = multi-step reasoning, analysis. "
                "'coding' = code generation/review. "
                "'math' = calculations, proofs, science. "
                "'creative' = writing, brainstorming. "
                "'extraction' = parsing, structured output. "
                "Determines whether the model is a good fit for the task."
            ),
        },
        "expected_output_tokens": {
            "type": "integer",
            "description": (
                "Estimated output tokens. Use 200 for short, 500 for medium, "
                "1000+ for long. Optional — a heuristic is used if not provided."
            ),
        },
    },
    "required": ["prompt", "model"],
}

_COMPARE_MODELS_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The prompt or message content to compare across models",
        },
        "task_type": {
            "type": "string",
            "enum": _TASK_TYPE_ENUM,
            "description": (
                "The type of task. Determines which models are recommended "
                "vs flagged as overkill or underpowered."
            ),
        },
        "models": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Models to compare. If not provided, compares a smart default "
                "set of 6 models (2 frontier + 3 efficient + 1 reasoning)."
            ),
        },
        "expected_output_tokens": {
            "type": "integer",
            "description": "Estimated output tokens. Optional.",
        },
    },
    "required": ["prompt"],
}

_CHECK_BUDGET_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "period": {
            "type": "string",
            "enum": ["today", "this_week", "this_month", "all_time"],
            "description": "Time period to check. Defaults to 'today'.",
        },
    },
}

DEFAULT_COMPARE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
    "deepseek-chat",
    "gemini-2.0-flash",
]


# ── Schema generators ───────────────────────────────────────────────────────

def cost_tools_openai() -> List[Dict[str, Any]]:
    """Return cost tool schemas in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "estimate_cost",
                "description": (
                    "Estimate the cost of an LLM API call BEFORE making it. "
                    "Returns cost breakdown, model capability info, task fit "
                    "assessment, and alternative model suggestions. Use this "
                    "to make informed cost-quality decisions before every "
                    "API call."
                ),
                "parameters": _ESTIMATE_COST_PARAMS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_models",
                "description": (
                    "Compare cost AND capability of multiple models for a "
                    "specific task. Returns models split into recommended "
                    "(meets quality bar) and budget-friendly (cheaper but "
                    "risky) groups, sorted by best value. Use this when "
                    "deciding which model to use for a task."
                ),
                "parameters": _COMPARE_MODELS_PARAMS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_budget",
                "description": (
                    "Check remaining budget, spending history, and get "
                    "adaptive recommendations. Returns current spend, "
                    "remaining budget, spending projections, and model "
                    "strategy recommendations based on how much budget "
                    "is left."
                ),
                "parameters": _CHECK_BUDGET_PARAMS,
            },
        },
    ]


def cost_tools_anthropic() -> List[Dict[str, Any]]:
    """Return cost tool schemas in Anthropic tool_use format."""
    return [
        {
            "name": "estimate_cost",
            "description": (
                "Estimate the cost of an LLM API call BEFORE making it. "
                "Returns cost breakdown, model capability info, task fit "
                "assessment, and alternative model suggestions."
            ),
            "input_schema": _ESTIMATE_COST_PARAMS,
        },
        {
            "name": "compare_models",
            "description": (
                "Compare cost AND capability of multiple models for a "
                "specific task. Returns models split into recommended "
                "and budget-friendly groups, sorted by best value."
            ),
            "input_schema": _COMPARE_MODELS_PARAMS,
        },
        {
            "name": "check_budget",
            "description": (
                "Check remaining budget, spending history, and get "
                "adaptive recommendations based on budget status."
            ),
            "input_schema": _CHECK_BUDGET_PARAMS,
        },
    ]


def cost_tools() -> List[Dict[str, Any]]:
    """Return framework-agnostic cost tool definitions.

    Adapt these to CrewAI, AutoGen, LangChain, etc.
    """
    return [
        {
            "name": "estimate_cost",
            "description": (
                "Estimate the cost of an LLM API call before making it."
            ),
            "parameters": _ESTIMATE_COST_PARAMS,
        },
        {
            "name": "compare_models",
            "description": (
                "Compare cost and capability of multiple models for a task."
            ),
            "parameters": _COMPARE_MODELS_PARAMS,
        },
        {
            "name": "check_budget",
            "description": (
                "Check remaining budget and get adaptive recommendations."
            ),
            "parameters": _CHECK_BUDGET_PARAMS,
        },
    ]


# ── Tool handlers ────────────────────────────────────────────────────────────

def _handle_estimate_cost(
    prompt: str,
    model: str,
    task_type: Optional[str] = None,
    expected_output_tokens: Optional[int] = None,
) -> str:
    """Handle estimate_cost tool call and return formatted string."""
    # Clamp negative output tokens
    if expected_output_tokens is not None and expected_output_tokens < 0:
        expected_output_tokens = 0

    try:
        result = estimate(prompt, model, expected_output_tokens)
        pricing = get_registry().get(model)
    except ModelNotFoundError:
        return (
            f"Error: Model '{model}' not found in pricing database. "
            f"Use compare_models to see available models."
        )

    lines = [
        f"Estimated cost for {model}: ${result.estimated_cost:.4f}",
        f"- Input: {result.input_tokens:,} tokens (${result.input_cost:.4f})",
    ]
    out_note = " [estimated]" if result.is_output_estimated else ""
    lines.append(
        f"- Output: ~{result.output_tokens:,} tokens "
        f"(${result.output_cost:.4f}){out_note}"
    )
    lines.append(
        f"- Model: {pricing.capability_tier.capitalize()} tier, "
        f"quality {pricing.quality_score}/100"
    )

    if task_type is not None and task_type in TASK_REQUIREMENTS:
        suitable, reason = is_model_suitable(model, task_type)
        req = TASK_REQUIREMENTS[task_type]
        min_quality = req["min_quality_score"]

        if suitable:
            # Check if overkill
            min_tier = req["min_tier"]
            min_tier_rank = TIER_HIERARCHY.get(min_tier, 1)
            model_tier_rank = TIER_HIERARCHY.get(pricing.capability_tier, 1)
            if (
                model_tier_rank > min_tier_rank + 1
                and task_type in ("simple", "extraction")
            ):
                lines.append(
                    f"- Task fit: \u26a1 Overkill for {task_type} tasks"
                )
                # Find cheaper alternative
                alt = _find_cheaper_alternative(
                    prompt, model, task_type, expected_output_tokens
                )
                if alt:
                    savings_pct = (
                        (1 - alt["cost"] / result.estimated_cost) * 100
                    )
                    lines.append("")
                    lines.append(
                        f"\U0001f4a1 Recommended: {alt['model']} at "
                        f"${alt['cost']:.4f} ({savings_pct:.0f}% cheaper)"
                    )
                    lines.append(
                        f"{alt['model']} scores {alt['quality']}/100 which "
                        f"exceeds the {min_quality}/100 needed for "
                        f"{task_type} tasks. Same quality results, "
                        f"fraction of the cost."
                    )
            else:
                lines.append(
                    f"- Task fit: \u2705 Well-suited for {task_type}"
                )
                # Suggest budget-friendly alternative
                alt = _find_cheaper_alternative(
                    prompt, model, task_type, expected_output_tokens
                )
                if alt and alt["cost"] < result.estimated_cost * 0.5:
                    savings_pct = (
                        (1 - alt["cost"] / result.estimated_cost) * 100
                    )
                    lines.append("")
                    lines.append(
                        f"Budget-friendly alternative: {alt['model']} at "
                        f"${alt['cost']:.4f} ({savings_pct:.0f}% cheaper)"
                    )
                    lines.append(
                        f"\u26a0\ufe0f Quality tradeoff: {alt['model']} scores "
                        f"{alt['quality']}/100 and may struggle with "
                        f"{task_type}. Use only if budget is tight."
                    )
        else:
            lines.append(
                f"- Task fit: \u26a0\ufe0f Below recommended quality "
                f"({min_quality}) for {task_type}"
            )
            # Find better alternative
            alt = _find_better_alternative(
                prompt, model, task_type, expected_output_tokens
            )
            if alt:
                cost_diff = alt["cost"] - result.estimated_cost
                lines.append("")
                lines.append(
                    f"\U0001f53c Recommended upgrade: {alt['model']} at "
                    f"${alt['cost']:.4f}"
                )
                lines.append(
                    f"For {task_type}, {alt['model']} "
                    f"(quality {alt['quality']}) significantly outperforms "
                    f"{model}. The ${cost_diff:.4f} difference is worth "
                    f"it for reliable results."
                )
                lines.append(
                    f"Proceed with {model} only if you accept potential "
                    f"quality issues."
                )

    return "\n".join(lines)


def _find_cheaper_alternative(
    prompt: str,
    current_model: str,
    task_type: str,
    expected_output_tokens: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Find a cheaper model that still meets the task requirements."""
    registry = get_registry()
    current = registry.get(current_model)
    current_est = estimate(prompt, current_model, expected_output_tokens)

    req = TASK_REQUIREMENTS.get(task_type)
    if req is None:
        return None

    best = None
    for m in registry.list_models():
        if m.name == current_model:
            continue
        est = estimate(prompt, m.name, expected_output_tokens)
        if est.estimated_cost >= current_est.estimated_cost:
            continue
        suitable, _ = is_model_suitable(m.name, task_type)
        # For cheaper alternative: include even marginally suitable ones
        if suitable or m.quality_score >= req["min_quality_score"] - 10:
            if best is None or est.estimated_cost < best["cost"]:
                best = {
                    "model": m.name,
                    "cost": est.estimated_cost,
                    "quality": m.quality_score,
                }
    return best


def _find_better_alternative(
    prompt: str,
    current_model: str,
    task_type: str,
    expected_output_tokens: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Find a better-quality model for the given task type."""
    registry = get_registry()
    req = TASK_REQUIREMENTS.get(task_type)
    if req is None:
        return None

    best = None
    for m in registry.list_models():
        if m.name == current_model:
            continue
        suitable, _ = is_model_suitable(m.name, task_type)
        if not suitable:
            continue
        est = estimate(prompt, m.name, expected_output_tokens)
        score = calculate_cost_quality_score(m.name, task_type, est.estimated_cost)
        if best is None or score > best["score"]:
            best = {
                "model": m.name,
                "cost": est.estimated_cost,
                "quality": m.quality_score,
                "score": score,
            }
    return best


def _value_stars(score: float, max_score: float) -> str:
    """Convert a value score to 1-5 star rating."""
    if max_score <= 0:
        return "\u2b50"
    ratio = score / max_score
    if ratio >= 0.9:
        return "\u2b50\u2b50\u2b50\u2b50\u2b50"
    elif ratio >= 0.7:
        return "\u2b50\u2b50\u2b50\u2b50"
    elif ratio >= 0.5:
        return "\u2b50\u2b50\u2b50"
    elif ratio >= 0.3:
        return "\u2b50\u2b50"
    else:
        return "\u2b50"


def _handle_compare_models(
    prompt: str,
    task_type: Optional[str] = None,
    models: Optional[List[str]] = None,
    expected_output_tokens: Optional[int] = None,
) -> str:
    """Handle compare_models tool call and return formatted string."""
    # Clamp negative output tokens
    if expected_output_tokens is not None and expected_output_tokens < 0:
        expected_output_tokens = 0

    model_list = models if models else DEFAULT_COMPARE_MODELS

    # Filter out unknown models instead of crashing
    registry = get_registry()
    skipped: List[str] = []
    valid_models: List[str] = []
    for m in model_list:
        try:
            registry.get(m)
            valid_models.append(m)
        except ModelNotFoundError:
            skipped.append(m)

    if not valid_models:
        return (
            f"Error: No recognized models in the list. "
            f"Skipped: {', '.join(skipped)}. "
            f"Use compare_models without a models list to see defaults."
        )

    results = compare(prompt, valid_models, expected_output_tokens, print_table=False)

    if not results:
        return "No valid models to compare."

    first = results[0]
    out_note = " [estimated]" if first.is_output_estimated else ""

    lines = [
        f"Cost-Quality Comparison ({first.input_tokens:,} input tokens, "
        f"~{first.output_tokens:,} output tokens{out_note})"
    ]

    if task_type is not None and task_type in TASK_REQUIREMENTS:
        req = TASK_REQUIREMENTS[task_type]
        min_quality = req["min_quality_score"]
        lines.append(f"Task type: {task_type}")
        lines.append("")

        # Split into recommended, budget-friendly, overkill
        recommended = []
        budget_friendly = []
        overkill_list = []

        min_tier = req["min_tier"]
        min_tier_rank = TIER_HIERARCHY.get(min_tier, 1)

        for r in results:
            try:
                p = registry.get(r.model)
                quality = p.quality_score
                tier = p.capability_tier
            except Exception:
                quality = 0
                tier = "unknown"

            model_tier_rank = TIER_HIERARCHY.get(tier, 1)
            meets_quality = quality >= min_quality
            meets_tier = model_tier_rank >= min_tier_rank

            score = calculate_cost_quality_score(
                r.model, task_type, r.estimated_cost
            )

            entry = {
                "result": r,
                "quality": quality,
                "tier": tier,
                "score": score,
            }

            if meets_quality and meets_tier:
                if (
                    model_tier_rank > min_tier_rank + 1
                    and task_type in ("simple", "extraction")
                ):
                    overkill_list.append(entry)
                else:
                    recommended.append(entry)
            else:
                budget_friendly.append(entry)

        # Sort recommended by value score
        recommended.sort(key=lambda x: x["score"], reverse=True)
        max_score = recommended[0]["score"] if recommended else 1.0

        if task_type in ("simple", "extraction"):
            # For simple tasks: RECOMMENDED + OVERKILL
            if recommended:
                lines.append(
                    f"\u2705 RECOMMENDED (all meet quality bar of "
                    f"{min_quality}+ for {task_type}):"
                )
                lines.append(
                    f"{'Model':<23}| {'Cost':>8} | {'Quality':>7} | "
                    f"{'Fit':>6} | Value"
                )
                lines.append("-" * 65)
                for e in recommended:
                    r = e["result"]
                    stars = _value_stars(e["score"], max_score)
                    lines.append(
                        f"{r.model:<23}| ${r.estimated_cost:>7.4f} | "
                        f"{e['quality']:>4}/100 | \u2705     | {stars}"
                    )

            if overkill_list:
                lines.append("")
                lines.append(
                    "\U0001f4b0 OVERKILL (work fine but unnecessarily "
                    "expensive):"
                )
                lines.append(
                    f"{'Model':<23}| {'Cost':>8} | {'Quality':>7} | "
                    f"Savings if downgraded"
                )
                lines.append("-" * 65)
                cheapest_rec = (
                    min(recommended, key=lambda x: x["result"].estimated_cost)
                    if recommended
                    else None
                )
                for e in overkill_list:
                    r = e["result"]
                    if cheapest_rec:
                        cr = cheapest_rec["result"]
                        savings = r.estimated_cost - cr.estimated_cost
                        pct = (savings / r.estimated_cost) * 100
                        note = (
                            f"Save ${savings:.4f} ({pct:.0f}%) using "
                            f"{cr.model}"
                        )
                    else:
                        note = ""
                    lines.append(
                        f"{r.model:<23}| ${r.estimated_cost:>7.4f} | "
                        f"{e['quality']:>4}/100 | {note}"
                    )
        else:
            # For complex tasks: RECOMMENDED + BUDGET-FRIENDLY
            if recommended:
                lines.append(
                    f"\u2705 RECOMMENDED (meets quality bar of "
                    f"{min_quality}+ for {task_type}):"
                )
                lines.append(
                    f"{'Model':<23}| {'Cost':>8} | {'Quality':>7} | "
                    f"{'Fit':>6} | Value"
                )
                lines.append("-" * 65)
                for e in recommended:
                    r = e["result"]
                    stars = _value_stars(e["score"], max_score)
                    lines.append(
                        f"{r.model:<23}| ${r.estimated_cost:>7.4f} | "
                        f"{e['quality']:>4}/100 | \u2705     | {stars}"
                    )

            if budget_friendly:
                lines.append("")
                lines.append(
                    f"\u26a0\ufe0f BUDGET-FRIENDLY (cheaper but quality "
                    f"below {min_quality} threshold):"
                )
                lines.append(
                    f"{'Model':<23}| {'Cost':>8} | {'Quality':>7} | Risk"
                )
                lines.append("-" * 65)
                for e in budget_friendly:
                    r = e["result"]
                    risk = _get_quality_risk(e["quality"], task_type)
                    lines.append(
                        f"{r.model:<23}| ${r.estimated_cost:>7.4f} | "
                        f"{e['quality']:>4}/100 | {risk}"
                    )

        # Summary lines
        lines.append("")
        all_entries = recommended + budget_friendly + overkill_list
        if recommended:
            best = recommended[0]
            lines.append(
                f"Best value: {best['result'].model} "
                f"(${best['result'].estimated_cost:.4f})"
            )
        if recommended:
            cheapest_safe_entry = min(
                recommended, key=lambda x: x["result"].estimated_cost
            )
            cr = cheapest_safe_entry["result"]
            lines.append(
                f"Cheapest safe option: {cr.model} (${cr.estimated_cost:.4f})"
            )
        if budget_friendly:
            cheapest_risky = min(
                budget_friendly, key=lambda x: x["result"].estimated_cost
            )
            cr = cheapest_risky["result"]
            lines.append(
                f"Cheapest risky option: {cr.model} "
                f"(${cr.estimated_cost:.4f})"
            )

    else:
        # No task_type: neutral comparison sorted by cost
        lines.append("")
        lines.append(
            f"{'Model':<23}| {'Cost':>8} | {'Quality':>7} | "
            f"{'Tier':<10} | Best For"
        )
        lines.append("-" * 80)
        for r in results:
            try:
                p = registry.get(r.model)
                quality = f"{p.quality_score}/100"
                tier = p.capability_tier
                best_for = ", ".join(p.strengths[:3])
            except Exception:
                quality = "N/A"
                tier = "unknown"
                best_for = ""
            lines.append(
                f"{r.model:<23}| ${r.estimated_cost:>7.4f} | "
                f"{quality:>7} | {tier:<10} | {best_for}"
            )
        lines.append("")
        lines.append(
            'Tip: Specify task_type for personalized recommendations '
            '(e.g. task_type="coding").'
        )

    if skipped:
        lines.append("")
        lines.append(
            f"Note: Skipped unknown models: {', '.join(skipped)}"
        )

    return "\n".join(lines)


def _get_quality_risk(quality: int, task_type: str) -> str:
    """Return a short risk description for a below-threshold model."""
    risk_map = {
        "complex": "May struggle with multi-step logic",
        "coding": "Less reliable code output",
        "math": "May produce incorrect calculations",
        "creative": "Less nuanced creative output",
        "moderate": "Generally adequate but less thorough",
        "simple": "Should handle simple tasks fine",
        "extraction": "Should handle extraction fine",
    }
    return risk_map.get(task_type, "Quality below recommended threshold")


def _handle_check_budget(
    period: str = "today",
    tracker: Optional[Tracker] = None,
    budget_config: Optional[Dict[str, float]] = None,
) -> str:
    """Handle check_budget tool call and return formatted string."""
    if tracker is None:
        tracker = get_tracker()

    # Map period for tracker compatibility
    tracker_period = "total" if period == "all_time" else period

    total_spend = tracker.get_spend(period=tracker_period)
    breakdown = tracker.get_spend_breakdown(period=tracker_period)
    call_counts = tracker.get_call_count_breakdown(period=tracker_period)
    total_calls = tracker.get_total_call_count(period=tracker_period)

    # Determine budget limit from config
    budget_limit = None
    if budget_config:
        period_to_key = {
            "today": "daily",
            "this_week": "weekly",
            "this_month": "monthly",
            "all_time": "monthly",  # fallback
        }
        config_key = period_to_key.get(period, "monthly")
        budget_limit = budget_config.get(config_key)

    lines = [f"Budget Status ({period}):"]

    if budget_limit is not None and budget_limit > 0:
        pct_used = (total_spend / budget_limit) * 100
        remaining = budget_limit - total_spend
        lines.append(
            f"- Spent: ${total_spend:.2f} / ${budget_limit:.2f} "
            f"{_period_label(period)} limit ({pct_used:.1f}%)"
        )
        lines.append(f"- Remaining: ${remaining:.2f}")
    else:
        lines.append(f"- Spent: ${total_spend:.4f}")

    lines.append(f"- Calls made: {total_calls}")

    # Per-model breakdown
    if breakdown:
        lines.append("")
        lines.append("Breakdown by model:")
        for model_name, cost in breakdown.items():
            count = call_counts.get(model_name, 0)
            if budget_limit and budget_limit > 0:
                pct = (cost / total_spend * 100) if total_spend > 0 else 0
                lines.append(
                    f"- {model_name}: ${cost:.2f} ({pct:.0f}%) "
                    f"\u2014 {count} calls"
                )
            else:
                lines.append(
                    f"- {model_name}: ${cost:.4f} \u2014 {count} calls"
                )
    else:
        lines.append("")
        lines.append("No spending recorded for this period.")

    # Adaptive recommendations based on budget %
    if budget_limit is not None and budget_limit > 0:
        pct_used = (total_spend / budget_limit) * 100
        remaining = budget_limit - total_spend

        lines.append("")

        if pct_used < 60:
            # Healthy
            if total_calls > 0:
                avg_cost = total_spend / total_calls
                if avg_cost > 0:
                    hours_left = remaining / (
                        total_spend / max(1, _hours_elapsed(period))
                    ) if total_spend > 0 else float("inf")
                    if hours_left != float("inf"):
                        lines.append(
                            f"Projection: At current rate, budget lasts "
                            f"~{hours_left:.1f} more hours."
                        )
            lines.append(
                "Status: \u2705 Healthy. You can use frontier models for "
                "complex tasks."
            )
        elif pct_used < 85:
            # Tightening
            if total_calls > 0:
                avg_cost = total_spend / total_calls
                if avg_cost > 0:
                    hours_left = remaining / (
                        total_spend / max(1, _hours_elapsed(period))
                    ) if total_spend > 0 else 0
                    lines.append(
                        f"Projection: At current rate, limit hit in "
                        f"~{hours_left:.1f} hours."
                    )
            lines.append(
                "Status: \u26a0\ufe0f Budget tightening. Recommendations:"
            )
            lines.append(
                "- Switch simple/moderate tasks to gpt-4o-mini "
                "(saves ~$0.009/call vs gpt-4o)"
            )
            lines.append(
                "- Reserve remaining budget for complex tasks "
                "needing frontier models"
            )
            _append_capacity_lines(lines, remaining)
        else:
            # Critical
            lines.append("Projection: Budget nearly exhausted.")
            lines.append(
                "Status: \U0001f534 CRITICAL. Recommendations:"
            )
            lines.append(
                "- Switch ALL tasks to cheapest model "
                "(deepseek-chat $0.0003/call or gpt-4o-mini $0.0004/call)"
            )
            _append_capacity_lines(lines, remaining)
            lines.append(
                "- Only use frontier models if task absolutely requires it "
                "and failure cost exceeds model cost"
            )
    else:
        lines.append("")
        lines.append(
            "Note: No budget limit configured. Use budget_config to set "
            "daily/weekly/monthly limits."
        )

    return "\n".join(lines)


def _period_label(period: str) -> str:
    """Return human-readable label for a period."""
    return {
        "today": "daily",
        "this_week": "weekly",
        "this_month": "monthly",
        "all_time": "total",
    }.get(period, period)


def _hours_elapsed(period: str) -> float:
    """Rough estimate of hours elapsed in the period (for projection)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if period == "today":
        return max(1, now.hour + now.minute / 60)
    elif period == "this_week":
        return max(1, now.weekday() * 24 + now.hour)
    elif period == "this_month":
        return max(1, (now.day - 1) * 24 + now.hour)
    else:
        return 24.0  # fallback


def _append_capacity_lines(
    lines: List[str],
    remaining: float,
) -> None:
    """Append remaining call capacity estimates."""
    # Estimate calls remaining for different model tiers
    # Using approximate per-call costs
    gpt4o_cost = 0.0096  # ~$0.01 per call
    mini_cost = 0.0004
    gpt4o_calls = int(remaining / gpt4o_cost) if gpt4o_cost > 0 else 0
    mini_calls = int(remaining / mini_cost) if mini_cost > 0 else 0
    lines.append(
        f"- Remaining capacity: ~{gpt4o_calls} gpt-4o calls OR "
        f"~{mini_calls} gpt-4o-mini calls"
    )


# ── Universal handler ────────────────────────────────────────────────────────

def cost_context(
    tool_call: Any,
    tracker: Optional[Tracker] = None,
    budget_config: Optional[Dict[str, float]] = None,
) -> str:
    """Universal handler for tool calls from any agent framework.

    Accepts:
    - OpenAI tool_call object (.function.name, .function.arguments as JSON)
    - Anthropic tool_use block (.name, .input as dict)
    - Plain dict: {"name": "...", "arguments": {...}}
        or {"name": "...", "input": {...}}

    Args:
        tool_call: The tool call in any supported format
        tracker: Optional Tracker instance for check_budget
        budget_config: Optional dict {"daily": 5.0, "weekly": 25.0, ...}

    Returns:
        str: Always returns a string. Never raises exceptions.
        On error, returns a string starting with "Error: ".
    """
    try:
        name, args = _parse_tool_call(tool_call)
        return _route_tool(name, args, tracker, budget_config)
    except Exception as exc:
        logger.error("cost_context failed: %s", exc)
        return f"Error: {exc}"


# Backward-compatible alias (others at end of file)
handle_tool_call = cost_context


def _parse_tool_call(tool_call: Any) -> tuple:
    """Extract (name, args_dict) from various tool call formats."""
    # OpenAI format: has .function attribute
    if hasattr(tool_call, "function"):
        func = tool_call.function
        name = func.name
        raw_args = func.arguments
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool arguments: {e}")
        else:
            args = raw_args or {}
        return name, args

    # Anthropic format: has .name and .input attributes
    if hasattr(tool_call, "name") and hasattr(tool_call, "input"):
        return tool_call.name, tool_call.input or {}

    # Dict format
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if name is None:
            raise ValueError(
                "Dict tool_call must have a 'name' key. "
                "Got keys: " + ", ".join(tool_call.keys())
            )
        args = (
            tool_call.get("arguments")
            or tool_call.get("input")
            or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool arguments: {e}")
        return name, args

    raise ValueError(
        f"Unsupported tool_call format: {type(tool_call).__name__}. "
        f"Expected OpenAI tool_call, Anthropic tool_use block, or dict."
    )


def _route_tool(
    name: str,
    args: Dict[str, Any],
    tracker: Optional[Tracker],
    budget_config: Optional[Dict[str, float]],
) -> str:
    """Route to the appropriate tool handler."""
    if name == "estimate_cost":
        # Validate required args
        if "prompt" not in args:
            return "Error: 'prompt' is required for estimate_cost."
        if "model" not in args:
            return "Error: 'model' is required for estimate_cost."
        return _handle_estimate_cost(
            prompt=args["prompt"],
            model=args["model"],
            task_type=args.get("task_type"),
            expected_output_tokens=args.get("expected_output_tokens"),
        )
    elif name == "compare_models":
        if "prompt" not in args:
            return "Error: 'prompt' is required for compare_models."
        return _handle_compare_models(
            prompt=args["prompt"],
            task_type=args.get("task_type"),
            models=args.get("models"),
            expected_output_tokens=args.get("expected_output_tokens"),
        )
    elif name == "check_budget":
        return _handle_check_budget(
            period=args.get("period", "today"),
            tracker=tracker,
            budget_config=budget_config,
        )
    else:
        valid_names = "estimate_cost, compare_models, check_budget"
        raise ValueError(
            f"Unknown tool: '{name}'. Valid tools are: {valid_names}"
        )


# ── System prompt helper ─────────────────────────────────────────────────────

def cost_prompt(
    daily_budget: Optional[float] = None,
    weekly_budget: Optional[float] = None,
    monthly_budget: Optional[float] = None,
    strategy: str = "balanced",
) -> str:
    """Generate a system prompt that makes the agent cost-aware.

    Args:
        daily_budget: Optional daily budget in USD
        weekly_budget: Optional weekly budget in USD
        monthly_budget: Optional monthly budget in USD
        strategy: "balanced" | "cost_first" | "quality_first"

    Returns:
        System prompt string with live budget data.
    """
    tracker = get_tracker()

    # Build budget line
    budget_parts = []
    if daily_budget is not None:
        spent = tracker.get_spend("today")
        budget_parts.append(
            f"${daily_budget:.2f}/day | Current spend: ${spent:.2f} | "
            f"Remaining: ${daily_budget - spent:.2f}"
        )
    if weekly_budget is not None:
        spent = tracker.get_spend("this_week")
        budget_parts.append(
            f"${weekly_budget:.2f}/week | Current spend: ${spent:.2f} | "
            f"Remaining: ${weekly_budget - spent:.2f}"
        )
    if monthly_budget is not None:
        spent = tracker.get_spend("this_month")
        budget_parts.append(
            f"${monthly_budget:.2f}/month | Current spend: ${spent:.2f} | "
            f"Remaining: ${monthly_budget - spent:.2f}"
        )

    budget_line = ""
    if budget_parts:
        budget_line = "\n\nBudget: " + " | ".join(budget_parts)

    if strategy == "cost_first":
        return (
            "You have access to cost estimation tools. MINIMIZE COST while "
            "maintaining acceptable quality.\n"
            f"{budget_line}\n\n"
            "Rules:\n"
            "1. Always start with the cheapest model for every task\n"
            "2. Only upgrade if the task is clearly complex AND the cheap "
            "model would likely fail\n"
            "3. Call compare_models before every API call to find cheapest "
            "viable option\n"
            "4. Check budget after every 10 calls\n"
            "5. If a cheaper model gives 80%+ quality for the task, use it\n"
            "\n"
            "Available tools: estimate_cost, compare_models, check_budget"
        )
    elif strategy == "quality_first":
        return (
            "You have access to cost estimation tools. PRIORITIZE QUALITY "
            "but avoid obvious waste on simple tasks.\n"
            f"{budget_line}\n\n"
            "Rules:\n"
            "1. Use frontier models for reasoning, coding, analysis, and "
            "creative tasks\n"
            "2. Only use cheap models for truly mechanical tasks "
            "(extraction, formatting)\n"
            "3. Use estimate_cost to sanity-check, not to penny-pinch\n"
            "4. Check budget periodically \u2014 if running low, prioritize "
            "remaining budget for critical tasks\n"
            "5. Quality failures cost more than model costs \u2014 when in "
            "doubt, use the better model\n"
            "\n"
            "Available tools: estimate_cost, compare_models, check_budget"
        )
    else:
        # balanced (default)
        return (
            "You have access to cost estimation tools for LLM API calls. "
            "Use them to make intelligent cost-quality tradeoffs.\n"
            f"{budget_line}\n\n"
            "Decision framework:\n"
            "1. Before each LLM call, classify the task complexity:\n"
            "   - Simple (extraction, formatting, translation) \u2192 "
            "efficient models ($0.0003-0.0005/call)\n"
            "   - Moderate (summarization, Q&A) \u2192 efficient or mid-tier "
            "models ($0.0004-0.002/call)\n"
            "   - Complex (reasoning, analysis, coding) \u2192 frontier "
            "models ($0.005-0.01/call)\n"
            "2. Call compare_models with task_type to get specific "
            "recommendations\n"
            "3. Budget > 50% remaining: use best model for complex tasks, "
            "cheap for simple\n"
            "4. Budget < 20% remaining: switch to cheapest models for "
            "ALL tasks\n"
            "5. Never use frontier models for tasks that efficient models "
            "handle equally well\n"
            "\n"
            "Available tools: estimate_cost, compare_models, check_budget"
        )


# ── Backward-compatible aliases ─────────────────────────────────────────────
get_openai_tools = cost_tools_openai
get_anthropic_tools = cost_tools_anthropic
get_tool_definitions = cost_tools
get_cost_aware_system_prompt = cost_prompt
