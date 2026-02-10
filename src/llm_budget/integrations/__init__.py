"""Framework integrations for llm-budget (LangChain, CrewAI)."""
from __future__ import annotations

__all__: list[str] = []

# Lazy imports -- each sub-module guards its framework import.
# Users should import directly:
#   from llm_budget.integrations.langchain import BudgetCallbackHandler
#   from llm_budget.integrations.crewai import BudgetHooks, track_crewai
