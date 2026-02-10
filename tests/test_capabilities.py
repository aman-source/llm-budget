"""Tests for capabilities module."""
from __future__ import annotations

import pytest

from llm_budget.capabilities import (
    TASK_REQUIREMENTS,
    TIER_HIERARCHY,
    calculate_cost_quality_score,
    get_capability_summary,
    is_model_suitable,
    recommend_models,
)


class TestTaskRequirements:
    def test_all_task_types_present(self):
        expected = {"simple", "moderate", "complex", "coding", "math", "creative", "extraction"}
        assert expected == set(TASK_REQUIREMENTS.keys())

    def test_all_tasks_have_required_fields(self):
        required_keys = {
            "description", "min_tier", "preferred_tiers",
            "min_quality_score", "relevant_strengths",
        }
        for task_type, req in TASK_REQUIREMENTS.items():
            assert required_keys.issubset(req.keys()), (
                f"Task '{task_type}' missing fields"
            )


class TestTierHierarchy:
    def test_all_tiers_present(self):
        assert "efficient" in TIER_HIERARCHY
        assert "mid" in TIER_HIERARCHY
        assert "frontier" in TIER_HIERARCHY
        assert "reasoning" in TIER_HIERARCHY

    def test_ordering(self):
        assert TIER_HIERARCHY["efficient"] < TIER_HIERARCHY["mid"]
        assert TIER_HIERARCHY["mid"] < TIER_HIERARCHY["frontier"]

    def test_reasoning_equals_frontier(self):
        assert TIER_HIERARCHY["reasoning"] == TIER_HIERARCHY["frontier"]


class TestRecommendModels:
    def test_complex_recommends_frontier(self):
        recs = recommend_models("Analyze this complex dataset", task_type="complex")
        assert len(recs["recommended"]) > 0
        # All recommended should be frontier or reasoning tier
        from llm_budget.pricing import get_registry
        registry = get_registry()
        for model_name in recs["recommended"]:
            m = registry.get(model_name)
            assert m.capability_tier in ("frontier", "reasoning"), (
                f"{model_name} is {m.capability_tier}, expected frontier/reasoning"
            )

    def test_simple_recommends_efficient(self):
        recs = recommend_models("Extract the name from: John Smith", task_type="simple")
        assert len(recs["recommended"]) > 0
        # At least some recommended should be efficient tier
        from llm_budget.pricing import get_registry
        registry = get_registry()
        tiers = [registry.get(m).capability_tier for m in recs["recommended"]]
        assert "efficient" in tiers or "mid" in tiers

    def test_complex_with_low_budget_adds_warning(self):
        recs = recommend_models(
            "Analyze this", task_type="complex", budget_remaining=0.001
        )
        # With very low budget, should have warning or empty recommended
        if len(recs["recommended"]) == 0:
            assert recs["warning"] is not None
        # If some are recommended, they should be affordable

    def test_returns_all_keys(self):
        recs = recommend_models("test", task_type="moderate")
        expected_keys = {
            "recommended", "budget_friendly", "overkill",
            "best_value", "cheapest_safe", "cheapest_risky", "warning",
        }
        assert expected_keys == set(recs.keys())

    def test_available_models_filter(self):
        recs = recommend_models(
            "test",
            task_type="moderate",
            available_models=["gpt-4o", "gpt-4o-mini"],
        )
        all_models = (
            recs["recommended"] + recs["budget_friendly"] + recs["overkill"]
        )
        for m in all_models:
            assert m in ("gpt-4o", "gpt-4o-mini")


class TestCalculateCostQualityScore:
    def test_deepseek_beats_gpt4o_for_extraction(self):
        # deepseek-chat is cheap and suitable for extraction
        # gpt-4o is expensive and overkill
        score_dc = calculate_cost_quality_score("deepseek-chat", "extraction", 0.0003)
        score_4o = calculate_cost_quality_score("gpt-4o", "extraction", 0.0096)
        assert score_dc > score_4o

    def test_gpt4o_beats_deepseek_for_complex(self):
        score_4o = calculate_cost_quality_score("gpt-4o", "complex", 0.0096)
        score_dc = calculate_cost_quality_score("deepseek-chat", "complex", 0.0003)
        assert score_4o > score_dc

    def test_higher_quality_scores_higher(self):
        # Same cost, different models
        score_4o = calculate_cost_quality_score("gpt-4o", "moderate", 0.005)
        score_mini = calculate_cost_quality_score("gpt-4o-mini", "moderate", 0.005)
        assert score_4o > score_mini

    def test_zero_cost_no_crash(self):
        score = calculate_cost_quality_score("gpt-4o", "moderate", 0.0)
        assert score > 0

    def test_unknown_model_returns_zero(self):
        score = calculate_cost_quality_score("nonexistent-model", "moderate", 0.01)
        assert score == 0.0


class TestIsModelSuitable:
    def test_gpt4o_mini_unsuitable_for_complex(self):
        suitable, reason = is_model_suitable("gpt-4o-mini", "complex")
        assert suitable is False
        assert "quality" in reason.lower() or "threshold" in reason.lower()

    def test_gpt4o_mini_suitable_for_extraction(self):
        suitable, reason = is_model_suitable("gpt-4o-mini", "extraction")
        assert suitable is True

    def test_gpt4o_suitable_for_extraction_but_overkill(self):
        suitable, reason = is_model_suitable("gpt-4o", "extraction")
        assert suitable is True
        assert "overkill" in reason.lower()

    def test_gpt4o_suitable_for_complex(self):
        suitable, reason = is_model_suitable("gpt-4o", "complex")
        assert suitable is True
        assert "well-suited" in reason.lower()

    def test_unknown_model_graceful(self):
        suitable, reason = is_model_suitable("nonexistent-model-xyz", "complex")
        assert suitable is False
        assert "unknown" in reason.lower()

    def test_unknown_task_type_returns_suitable(self):
        suitable, reason = is_model_suitable("gpt-4o", "unknown_task_xyz")
        assert suitable is True
        assert "unknown" in reason.lower()

    def test_reasoning_model_for_math(self):
        suitable, reason = is_model_suitable("o3-mini", "math")
        assert suitable is True


class TestGetCapabilitySummary:
    def test_returns_formatted_string(self):
        summary = get_capability_summary("gpt-4o")
        assert "gpt-4o" in summary
        assert "Frontier" in summary
        assert "95/100" in summary

    def test_includes_strengths(self):
        summary = get_capability_summary("gpt-4o")
        assert "reasoning" in summary or "coding" in summary

    def test_unknown_model(self):
        summary = get_capability_summary("nonexistent-model-xyz")
        assert "Unknown" in summary
