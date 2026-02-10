"""
Cost-quality tradeoff engine.

Maps task types to capability requirements and recommends models
that balance cost and quality based on task complexity.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .pricing import get_registry, ModelPricing

logger = logging.getLogger(__name__)

# Task type to required capability mapping
TASK_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "simple": {
        "description": "Formatting, extraction, translation, classification",
        "min_tier": "efficient",
        "preferred_tiers": ["efficient", "mid"],
        "min_quality_score": 50,
        "relevant_strengths": [
            "extraction", "formatting", "classification", "translation", "general",
        ],
    },
    "moderate": {
        "description": "Summarization, Q&A, content generation, general tasks",
        "min_tier": "efficient",
        "preferred_tiers": ["efficient", "mid"],
        "min_quality_score": 65,
        "relevant_strengths": [
            "summarization", "qa", "general", "creative",
        ],
    },
    "complex": {
        "description": "Multi-step reasoning, deep analysis, research synthesis",
        "min_tier": "mid",
        "preferred_tiers": ["frontier"],
        "min_quality_score": 85,
        "relevant_strengths": [
            "reasoning", "analysis", "long_context",
        ],
    },
    "coding": {
        "description": "Code generation, debugging, code review, refactoring",
        "min_tier": "mid",
        "preferred_tiers": ["frontier", "reasoning"],
        "min_quality_score": 80,
        "relevant_strengths": [
            "coding", "reasoning",
        ],
    },
    "math": {
        "description": "Mathematical reasoning, proofs, calculations, science",
        "min_tier": "mid",
        "preferred_tiers": ["reasoning", "frontier"],
        "min_quality_score": 85,
        "relevant_strengths": [
            "math", "reasoning", "science",
        ],
    },
    "creative": {
        "description": "Creative writing, brainstorming, marketing copy, storytelling",
        "min_tier": "mid",
        "preferred_tiers": ["frontier"],
        "min_quality_score": 80,
        "relevant_strengths": [
            "creative", "instruction_following",
        ],
    },
    "extraction": {
        "description": "Data extraction, parsing, structured output, JSON generation",
        "min_tier": "efficient",
        "preferred_tiers": ["efficient"],
        "min_quality_score": 50,
        "relevant_strengths": [
            "extraction", "formatting", "instruction_following",
        ],
    },
}

TIER_HIERARCHY: Dict[str, int] = {
    "efficient": 1,
    "mid": 2,
    "frontier": 3,
    "reasoning": 3,  # same level as frontier, different specialty
}


def recommend_models(
    prompt: str,
    task_type: str = "moderate",
    budget_remaining: Optional[float] = None,
    available_models: Optional[List[str]] = None,
    expected_output_tokens: int = 500,
) -> Dict[str, Any]:
    """Recommend models for a task based on cost-quality tradeoff.

    Returns dict with:
    - recommended: list of models that meet quality bar, sorted by value
    - budget_friendly: list of cheaper models with quality warnings
    - overkill: list of models more expensive than needed for this task
    - best_value: single best cost/quality model for this task
    - cheapest_safe: cheapest model that meets quality bar
    - cheapest_risky: absolute cheapest with quality risk note
    - warning: string if budget is low or task/model mismatch
    """
    registry = get_registry()
    all_models = registry.list_models()

    if available_models is not None:
        available_set = set(available_models)
        all_models = [m for m in all_models if m.name in available_set]

    req = TASK_REQUIREMENTS.get(task_type)
    if req is None:
        # Unknown task type — treat as moderate
        req = TASK_REQUIREMENTS["moderate"]

    min_quality = req["min_quality_score"]
    min_tier = req["min_tier"]
    min_tier_rank = TIER_HIERARCHY.get(min_tier, 1)
    preferred_tiers = req["preferred_tiers"]

    # Count input tokens for cost estimation
    from .tokenizer import count_tokens
    input_tokens = count_tokens(prompt, "gpt-4o")

    recommended = []
    budget_friendly = []
    overkill = []
    all_with_cost = []

    for model in all_models:
        cost = (
            input_tokens * model.input_cost_per_token
            + expected_output_tokens * model.output_cost_per_token
        )
        model_tier_rank = TIER_HIERARCHY.get(model.capability_tier, 1)
        meets_tier = model_tier_rank >= min_tier_rank
        meets_quality = model.quality_score >= min_quality

        entry = {
            "model": model.name,
            "cost": cost,
            "quality_score": model.quality_score,
            "tier": model.capability_tier,
            "value_score": calculate_cost_quality_score(
                model.name, task_type, cost
            ),
        }
        all_with_cost.append(entry)

        if meets_tier and meets_quality:
            # Check if overkill: model tier much higher than needed for simple tasks
            if (
                model_tier_rank > min_tier_rank + 1
                and task_type in ("simple", "extraction")
            ):
                overkill.append(entry)
            else:
                recommended.append(entry)
        else:
            budget_friendly.append(entry)

    # Sort recommended by value_score descending (best value first)
    recommended.sort(key=lambda x: x["value_score"], reverse=True)
    # Sort budget_friendly by cost ascending (cheapest first)
    budget_friendly.sort(key=lambda x: x["cost"])
    # Sort overkill by cost ascending
    overkill.sort(key=lambda x: x["cost"])
    # Sort all by cost for cheapest_risky
    all_with_cost.sort(key=lambda x: x["cost"])

    best_value = recommended[0]["model"] if recommended else None
    cheapest_safe = (
        min(recommended, key=lambda x: x["cost"])["model"]
        if recommended
        else None
    )
    cheapest_risky = all_with_cost[0]["model"] if all_with_cost else None

    warning = None
    if budget_remaining is not None and budget_remaining <= 0:
        warning = "Budget exhausted. All calls will exceed budget."
    elif budget_remaining is not None:
        # Check if any recommended models fit in budget
        affordable_recs = [
            r for r in recommended if r["cost"] <= budget_remaining
        ]
        if not affordable_recs and recommended:
            warning = (
                f"Budget remaining (${budget_remaining:.4f}) is too low for "
                f"recommended models. Consider budget-friendly alternatives."
            )
    if not recommended and not warning:
        warning = f"No models meet the quality bar for '{task_type}' tasks."

    return {
        "recommended": [r["model"] for r in recommended],
        "budget_friendly": [r["model"] for r in budget_friendly],
        "overkill": [r["model"] for r in overkill],
        "best_value": best_value,
        "cheapest_safe": cheapest_safe,
        "cheapest_risky": cheapest_risky,
        "warning": warning,
    }


def calculate_cost_quality_score(
    model: str,
    task_type: str,
    cost_usd: float,
) -> float:
    """Calculate a combined cost-quality score for ranking.

    Higher = better value.

    Formula:
    - Start with quality_score
    - Bonus if model's strengths match task's relevant_strengths
    - Penalty proportional to cost (normalized across compared models)
    - Heavy penalty if model tier < task's min_tier
    """
    try:
        pricing = get_registry().get(model)
    except Exception:
        return 0.0

    score = float(pricing.quality_score)

    # Bonus for matching strengths
    req = TASK_REQUIREMENTS.get(task_type)
    if req is not None:
        relevant = set(req["relevant_strengths"])
        model_strengths = set(pricing.strengths)
        matching = relevant & model_strengths
        # Up to +15 bonus for strength overlap
        if relevant:
            score += 15.0 * (len(matching) / len(relevant))

        # Heavy penalty if tier is below task's min_tier
        min_tier = req["min_tier"]
        min_tier_rank = TIER_HIERARCHY.get(min_tier, 1)
        model_tier_rank = TIER_HIERARCHY.get(pricing.capability_tier, 1)
        if model_tier_rank < min_tier_rank:
            score -= 30.0

    # Cost penalty: scale cost to a 0-30 range penalty
    # $0.01 = moderate penalty, $0.001 = small, $0.10 = heavy
    cost_penalty = cost_usd * 3000.0  # $0.01 => 30 penalty points
    score -= cost_penalty

    return max(score, 0.0)


def get_capability_summary(model: str) -> str:
    """Return a one-line human-readable capability summary.

    e.g. "gpt-4o: Frontier model. Best for reasoning, coding, creative. Quality: 95/100"
    """
    try:
        pricing = get_registry().get(model)
    except Exception:
        return f"Unknown model: {model}"

    tier_label = pricing.capability_tier.capitalize()
    strengths_str = ", ".join(pricing.strengths[:4])
    return (
        f"{pricing.name}: {tier_label} model. "
        f"Best for {strengths_str}. "
        f"Quality: {pricing.quality_score}/100"
    )


def is_model_suitable(
    model: str,
    task_type: str,
) -> Tuple[bool, str]:
    """Check if a model is suitable for a task type.

    Returns (suitable: bool, reason: str)
    """
    try:
        pricing = get_registry().get(model)
    except Exception:
        return (False, f"Unknown model: {model}")

    req = TASK_REQUIREMENTS.get(task_type)
    if req is None:
        return (True, f"Unknown task type '{task_type}', assuming suitable.")

    min_quality = req["min_quality_score"]
    min_tier = req["min_tier"]
    min_tier_rank = TIER_HIERARCHY.get(min_tier, 1)
    model_tier_rank = TIER_HIERARCHY.get(pricing.capability_tier, 1)

    # Check tier
    if model_tier_rank < min_tier_rank:
        return (
            False,
            f"{pricing.name} (quality: {pricing.quality_score}) is below "
            f"the recommended quality threshold of {min_quality} for "
            f"{task_type} tasks. May produce unreliable results.",
        )

    # Check quality score
    if pricing.quality_score < min_quality:
        return (
            False,
            f"{pricing.name} (quality: {pricing.quality_score}) is below "
            f"the recommended quality threshold of {min_quality} for "
            f"{task_type} tasks. May produce unreliable results.",
        )

    # Check if overkill
    preferred_tiers = req["preferred_tiers"]
    if (
        pricing.capability_tier not in preferred_tiers
        and model_tier_rank > min_tier_rank + 1
        and task_type in ("simple", "extraction")
    ):
        return (
            True,
            f"{pricing.name} is well-suited for {task_type} "
            f"(quality: {pricing.quality_score}, tier: {pricing.capability_tier}) "
            f"but may be overkill. Consider a cheaper alternative.",
        )

    return (
        True,
        f"{pricing.name} is well-suited for {task_type} "
        f"(quality: {pricing.quality_score}, tier: {pricing.capability_tier})",
    )
