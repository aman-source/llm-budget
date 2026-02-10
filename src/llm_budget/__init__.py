"""llm-budget: Pre-flight LLM cost estimation and budget enforcement."""
from __future__ import annotations

from ._version import __version__
from .capabilities import (
    TASK_REQUIREMENTS,
    calculate_cost_quality_score,
    get_capability_summary,
    is_model_suitable,
    recommend_models,
)
from .estimator import EstimationResult, compare, estimate
from .exceptions import (
    BudgetExceeded,
    LLMBudgetError,
    ModelNotFoundError,
    TokenizationError,
)
from .middleware import track_anthropic, track_openai
from .pricing import ModelPricing, PricingRegistry, get_pricing, get_registry
from .tokenizer import count_tokens
from .tools import (
    cost_context,
    cost_prompt,
    cost_tools,
    cost_tools_anthropic,
    cost_tools_openai,
    # Backward-compatible aliases
    get_anthropic_tools,
    get_cost_aware_system_prompt,
    get_openai_tools,
    get_tool_definitions,
    handle_tool_call,
)
from .tracker import Tracker, get_tracker
from .enforcer import BudgetEnforcer, BudgetPolicy, EnforcementResult
from .decorators import budget, estimate_decorator
from .router import cost_aware, smart_route

__all__ = [
    "__version__",
    # Estimator
    "estimate",
    "compare",
    "EstimationResult",
    # Pricing
    "get_pricing",
    "get_registry",
    "PricingRegistry",
    "ModelPricing",
    # Tokenizer
    "count_tokens",
    # Tracker
    "Tracker",
    "get_tracker",
    # Enforcer
    "BudgetEnforcer",
    "BudgetPolicy",
    "EnforcementResult",
    # Decorators
    "budget",
    "estimate_decorator",
    # Middleware
    "track_openai",
    "track_anthropic",
    # Capabilities
    "recommend_models",
    "is_model_suitable",
    "get_capability_summary",
    "calculate_cost_quality_score",
    "TASK_REQUIREMENTS",
    # Agent Tools
    "cost_tools_openai",
    "cost_tools_anthropic",
    "cost_tools",
    "cost_context",
    "cost_prompt",
    # Backward-compatible aliases
    "get_openai_tools",
    "get_anthropic_tools",
    "get_tool_definitions",
    "handle_tool_call",
    "get_cost_aware_system_prompt",
    # Router
    "smart_route",
    "cost_aware",
    # Exceptions
    "LLMBudgetError",
    "BudgetExceeded",
    "ModelNotFoundError",
    "TokenizationError",
    # Integration availability flags
    "LANGCHAIN_AVAILABLE",
    "CREWAI_AVAILABLE",
]

# Framework integrations (optional -- imported lazily)
try:
    from .integrations.langchain import LANGCHAIN_AVAILABLE
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from .integrations.crewai import CREWAI_AVAILABLE
except ImportError:
    CREWAI_AVAILABLE = False
