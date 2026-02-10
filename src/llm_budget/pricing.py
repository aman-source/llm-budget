"""Model pricing database loader and lookup."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """Pricing and capability data for a single model."""

    name: str
    provider: str
    input_cost_per_token: float
    output_cost_per_token: float
    max_input_tokens: int
    max_output_tokens: int
    tokenizer: str
    capability_tier: str = "mid"
    strengths: Tuple[str, ...] = ()
    quality_score: int = 70


class PricingRegistry:
    """Loads and queries the model pricing database."""

    def __init__(self, pricing_file: Optional[Path] = None) -> None:
        self._models: dict[str, ModelPricing] = {}
        self._path = pricing_file or Path(__file__).parent / "models.json"
        self._load(self._path)

    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, info in data.items():
            self._models[name] = ModelPricing(
                name=name,
                provider=info["provider"],
                input_cost_per_token=info["input_cost_per_token"],
                output_cost_per_token=info["output_cost_per_token"],
                max_input_tokens=info["max_input_tokens"],
                max_output_tokens=info["max_output_tokens"],
                tokenizer=info["tokenizer"],
                capability_tier=info.get("capability_tier", "mid"),
                strengths=tuple(info.get("strengths", ["general"])),
                quality_score=info.get("quality_score", 70),
            )

    def get(self, model: str) -> ModelPricing:
        """Get pricing for a model.

        Tries exact match first, then longest prefix match in both directions.
        Raises ModelNotFoundError if not found.
        """
        if model in self._models:
            return self._models[model]

        # Collect all prefix matches and pick the longest (most specific)
        best: Optional[ModelPricing] = None
        best_len = 0
        for name, pricing in self._models.items():
            if model.startswith(name) and len(name) > best_len:
                best = pricing
                best_len = len(name)
            elif name.startswith(model) and len(model) > best_len:
                best = pricing
                best_len = len(model)

        if best is not None:
            return best

        raise ModelNotFoundError(model)

    def list_models(self, provider: Optional[str] = None) -> list[ModelPricing]:
        """List all models, optionally filtered by provider."""
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        return models

    def add_model(self, name: str, **kwargs: object) -> None:
        """Register a custom model at runtime (not persisted to disk)."""
        defaults = {
            "capability_tier": "mid",
            "strengths": ("general",),
            "quality_score": 70,
        }
        merged = {**defaults, **kwargs}
        # Ensure strengths is a tuple even if caller passes a list
        if isinstance(merged.get("strengths"), list):
            merged["strengths"] = tuple(merged["strengths"])
        self._models[name] = ModelPricing(name=name, **merged)  # type: ignore[arg-type]

    def reload(self) -> None:
        """Reload pricing data from disk (e.g. after update-prices)."""
        self._models.clear()
        self._load(self._path)

    @property
    def models(self) -> dict[str, ModelPricing]:
        return dict(self._models)


_default_registry: Optional[PricingRegistry] = None


def get_registry() -> PricingRegistry:
    """Get or create the default pricing registry (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PricingRegistry()
    return _default_registry


def get_pricing(model: str) -> ModelPricing:
    """Convenience: get pricing for a model from the default registry."""
    return get_registry().get(model)
