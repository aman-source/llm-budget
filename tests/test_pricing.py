"""Tests for the pricing module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_budget.pricing import PricingRegistry, get_pricing, get_registry, ModelPricing
from llm_budget.exceptions import ModelNotFoundError


@pytest.fixture
def tmp_models(tmp_path):
    """Create a temporary models.json for testing."""
    data = {
        "test-model": {
            "provider": "test",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
            "max_input_tokens": 4096,
            "max_output_tokens": 2048,
            "tokenizer": "cl100k_base",
            "capability_tier": "efficient",
            "strengths": ["general"],
            "quality_score": 60,
        },
        "test-model-v2": {
            "provider": "test",
            "input_cost_per_token": 2e-06,
            "output_cost_per_token": 4e-06,
            "max_input_tokens": 8192,
            "max_output_tokens": 4096,
            "tokenizer": "cl100k_base",
        },
    }
    p = tmp_path / "models.json"
    p.write_text(json.dumps(data))
    return p


class TestPricingRegistry:
    def test_load_from_file(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        m = registry.get("test-model")
        assert m.provider == "test"
        assert m.capability_tier == "efficient"

    def test_defaults_for_missing_fields(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        m = registry.get("test-model-v2")
        assert m.capability_tier == "mid"  # default
        assert m.quality_score == 70  # default

    def test_prefix_match(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        m = registry.get("test-model-v2-extra")
        assert m.name == "test-model-v2"

    def test_model_not_found(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        with pytest.raises(ModelNotFoundError):
            registry.get("nonexistent-model")

    def test_list_models(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        all_models = registry.list_models()
        assert len(all_models) == 2

    def test_list_models_with_filter(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        filtered = registry.list_models(provider="test")
        assert len(filtered) == 2
        filtered_none = registry.list_models(provider="nonexistent")
        assert len(filtered_none) == 0

    def test_add_model(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        registry.add_model(
            "custom-model",
            provider="custom",
            input_cost_per_token=1e-06,
            output_cost_per_token=2e-06,
            max_input_tokens=4096,
            max_output_tokens=2048,
            tokenizer="cl100k_base",
            strengths=["coding", "reasoning"],
        )
        m = registry.get("custom-model")
        assert m.provider == "custom"
        assert isinstance(m.strengths, tuple)
        assert "coding" in m.strengths

    def test_add_model_defaults(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        registry.add_model(
            "minimal-model",
            provider="test",
            input_cost_per_token=1e-06,
            output_cost_per_token=2e-06,
            max_input_tokens=4096,
            max_output_tokens=2048,
            tokenizer="cl100k_base",
        )
        m = registry.get("minimal-model")
        assert m.capability_tier == "mid"
        assert m.quality_score == 70

    def test_reload(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        assert len(registry.list_models()) == 2

        # Write updated data
        data = {
            "new-model": {
                "provider": "new",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "max_input_tokens": 4096,
                "max_output_tokens": 2048,
                "tokenizer": "cl100k_base",
            },
        }
        tmp_models.write_text(json.dumps(data))
        registry.reload()
        assert len(registry.list_models()) == 1
        assert registry.get("new-model").provider == "new"

    def test_models_property(self, tmp_models):
        registry = PricingRegistry(tmp_models)
        models_dict = registry.models
        assert isinstance(models_dict, dict)
        assert "test-model" in models_dict


class TestGetPricing:
    def test_convenience_function(self):
        pricing = get_pricing("gpt-4o")
        assert pricing.provider == "openai"

    def test_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            get_pricing("totally-fake-model-xyz")


class TestGetRegistry:
    def test_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
