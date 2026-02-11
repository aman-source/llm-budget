# llm-budget

**Stop overpaying for LLM API calls.** One line of code, automatic model selection, same results.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-269%20passed-brightgreen.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](#)

> **One function. Auto-selects the cheapest model that fits your task. Tracks every dollar.**

---

## Before / After

```python
# BEFORE: You always use the expensive model. $9/day for 1,000 extraction calls.
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # $0.009/call
    messages=[{"role": "user", "content": "Extract dates from this text..."}],
    max_tokens=1024,
)

# AFTER: llm-budget picks the cheapest model that can handle the task.
response = llm_budget.smart_route(
    client, "Extract dates from this text...",
    task_type="extraction",
)
# -> Used claude-3-5-haiku ($0.003/call). 67% cheaper. Same result.
# -> $3/day instead of $9/day. That's $180/month saved.
```

**`smart_route()` returns the exact same response object as the native SDK.** Your `response.content[0].text`, `response.choices[0].message.content`, `response.usage` -- all unchanged. Your existing parsing code doesn't change. It's literally a 1-line swap.

---

## Install

```bash
pip install llm-budget
```

With provider SDKs:

```bash
pip install llm-budget[openai]      # OpenAI integration
pip install llm-budget[anthropic]   # Anthropic integration
pip install llm-budget[langchain]   # LangChain integration
pip install llm-budget[crewai]      # CrewAI integration
pip install llm-budget[all]         # Everything
```

---

## How It Works

`llm-budget` knows the pricing, capability tier, and quality score of 20 models across 7 providers. When you tell it what kind of task you're doing, it picks the cheapest model that won't compromise quality.

| Task Type | What It Means | Model Tier Selected |
|-----------|--------------|-------------------|
| `extraction` | Parsing, dates, JSON, structured output | Efficient (cheapest) |
| `simple` | Formatting, translation, classification | Efficient |
| `moderate` | Summarization, Q&A, general tasks | Efficient / Mid |
| `creative` | Writing, brainstorming, marketing copy | Frontier |
| `coding` | Code generation, debugging, review | Frontier / Reasoning |
| `complex` | Multi-step reasoning, deep analysis | Frontier |
| `math` | Calculations, proofs, science | Reasoning |

---

## Quick Start: `smart_route()` (Recommended)

One function. Auto-selects model, makes the API call, tracks cost.

### Anthropic

```python
from anthropic import Anthropic
import llm_budget

client = Anthropic()

# Extraction task -> llm-budget picks haiku (cheap, good enough)
response = llm_budget.smart_route(
    client, "Extract all dates from: 'Started March 15. Deadline June 30.'",
    task_type="extraction",
)
print(response.content[0].text)  # Works exactly like client.messages.create()

# Complex analysis -> llm-budget picks sonnet (frontier, worth the cost)
response = llm_budget.smart_route(
    client, "Analyze the trade-offs between microservices and monoliths",
    task_type="complex",
)
print(response.content[0].text)

# Check what you spent
print(f"Total: ${llm_budget.get_tracker().get_spend('today'):.4f}")
```

### OpenAI

```python
from openai import OpenAI
import llm_budget

client = OpenAI()

# Simple task -> picks gpt-4o-mini ($0.0004/call instead of $0.006 with gpt-4o)
response = llm_budget.smart_route(
    client, "Translate this to French: 'Hello, how are you?'",
    task_type="simple",
)
print(response.choices[0].message.content)  # Same as client.chat.completions.create()

# Coding task -> picks gpt-4o or o3-mini (needs reasoning capability)
response = llm_budget.smart_route(
    client, "Write a Python function to merge two sorted linked lists",
    task_type="coding",
)
```

### Full `smart_route()` API

```python
response = llm_budget.smart_route(
    client,                          # OpenAI() or Anthropic() client
    "Your prompt here",              # String or messages list
    task_type="moderate",            # simple/moderate/complex/coding/math/creative/extraction
    budget=5.00,                     # Daily budget in USD (influences model selection when low)
    model="gpt-4o",                  # Explicit override (skips routing, still tracks cost)
    max_tokens=1024,                 # Max output tokens
    temperature=0.7,                 # Forwarded to the API
    system="Be helpful",             # Forwarded to the API (Anthropic)
)
```

**Key behaviors:**
- `task_type` controls which models are eligible. `"extraction"` picks cheap models. `"complex"` picks powerful ones.
- `budget` makes routing budget-aware. When remaining budget is low, it favors cheaper models.
- `model` overrides routing entirely (but still tracks cost).
- All extra kwargs (`temperature`, `system`, `top_p`, etc.) are forwarded to the underlying API call.
- Returns the **exact same response** as the native SDK. No wrappers, no transformations.
- Cost is automatically recorded to the tracker after every call.

---

## More Control: Decorators

### `@cost_aware` -- Auto-swap models per call

Wrap your existing function. The decorator intercepts the `model` parameter and swaps it to a cheaper suitable model based on `task_type`.

```python
import llm_budget
from anthropic import Anthropic

client = Anthropic()

@llm_budget.cost_aware(budget=5.00)
def my_step(task, model="claude-sonnet-4-20250514", task_type="moderate"):
    return client.messages.create(
        model=model,
        messages=[{"role": "user", "content": task}],
        max_tokens=1024,
    )

# Decorator auto-swaps to haiku for extraction (saves 67%):
result = my_step("Extract dates from this invoice", task_type="extraction")

# Keeps sonnet for complex reasoning (worth the cost):
result = my_step("Analyze quarterly revenue trends", task_type="complex")
```

### `@budget` -- Hard budget enforcement

Pre-flight cost estimation + hard limits. Blocks calls that would exceed budget.

```python
from llm_budget import budget, BudgetExceeded

@budget(
    max_cost=5.00,           # $5 daily limit
    period="daily",          # Resets daily
    on_exceed="raise",       # "raise" | "warn" | "skip" | "downgrade:gpt-4o-mini"
    alert_at=0.8,            # Warn at 80% usage
    track_model="gpt-4o",
)
def my_llm_call(model, messages):
    return openai.chat.completions.create(model=model, messages=messages)

try:
    result = my_llm_call(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
except BudgetExceeded as e:
    print(f"Budget exceeded: {e}")
    if e.suggested_model:
        print(f"Try cheaper model: {e.suggested_model}")
```

**`on_exceed` modes:**
| Mode | Behavior |
|------|----------|
| `"raise"` | Raises `BudgetExceeded` exception |
| `"warn"` | Logs warning, proceeds anyway |
| `"skip"` | Returns `None`, skips the API call |
| `"downgrade:gpt-4o-mini"` | Auto-swaps to cheaper model, proceeds |

---

## Pre-Flight Cost Estimation

Know what you'll spend before you spend it. No API key required.

```python
from llm_budget import estimate, compare

# Estimate a single call
est = estimate(
    messages=[{"role": "user", "content": "Summarize this 10-page document..."}],
    model="gpt-4o",
    expected_output_tokens=500,
)
print(est)
# gpt-4o: ~42 input + ~500 output tokens = $0.005105

print(est.estimated_cost)   # 0.005105
print(est.input_cost)       # 0.000105
print(est.output_cost)      # 0.005000
print(est.input_tokens)     # 42
print(est.output_tokens)    # 500

# Compare across models (prints a sorted table)
results = compare(
    messages="Explain quantum computing in simple terms",
    models=["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "deepseek-chat"],
)
# Output:
# Model                         Input Tok   Est. Output    Est. Cost
# ---------------------------------------------------------------
# deepseek-chat                         8          ~8    $0.000006
# gpt-4o-mini                           8          ~8    $0.000006
# claude-3-5-haiku-20241022             8          ~8    $0.000048
# ...
```

---

## Spend Tracking

### Middleware (auto-tracking, zero code changes)

Wrap your client once. Every API call is automatically tracked.

```python
from openai import OpenAI
from llm_budget import track_openai, get_tracker

client = track_openai(OpenAI())  # Wrap once

# Use normally -- cost is recorded automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Check your spend
tracker = get_tracker()
print(f"Today: ${tracker.get_spend('today'):.4f}")
print(f"This week: ${tracker.get_spend('this_week'):.4f}")
print(f"This month: ${tracker.get_spend('this_month'):.4f}")
print(tracker.get_spend_breakdown("today"))  # {"gpt-4o": 0.0042, "gpt-4o-mini": 0.0001}
```

Works with Anthropic too:

```python
from anthropic import Anthropic
from llm_budget import track_anthropic

client = track_anthropic(Anthropic())
```

### Manual tracking

```python
from llm_budget import Tracker

tracker = Tracker()                      # SQLite-backed, persists across sessions
tracker = Tracker(db_path="my_app.db")   # Custom database path

tracker.record(
    model="gpt-4o",
    input_tokens=100,
    output_tokens=50,
    cost_usd=0.00075,
)

tracker.get_spend("today")              # Total spend today
tracker.get_spend("monthly", model="gpt-4o")  # Spend on gpt-4o this month
tracker.get_spend_breakdown("this_week")       # {"gpt-4o": 3.21, "gpt-4o-mini": 0.14}
tracker.get_history(last_n=20)                 # Recent call records
```

---

## Agent Tools (for AI agents that self-optimize)

Give your agent three tools. It calls them to decide which model to use, check budget, and compare options. This is the most powerful mode -- the agent itself makes cost-quality decisions.

### OpenAI Agent

```python
from openai import OpenAI
from llm_budget import cost_tools_openai, cost_context, cost_prompt

client = OpenAI()
tools = cost_tools_openai()       # 3 tools: estimate_cost, compare_models, check_budget
system = cost_prompt(daily_budget=5.00, strategy="balanced")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": system}, ...],
    tools=tools,
)

# When the agent calls a tool:
for tool_call in response.choices[0].message.tool_calls or []:
    result = cost_context(tool_call)  # Auto-detects format, returns string
```

### Anthropic Agent

```python
from anthropic import Anthropic
from llm_budget import cost_tools_anthropic, cost_context, cost_prompt

client = Anthropic()
tools = cost_tools_anthropic()
system = cost_prompt(monthly_budget=50.00, strategy="quality_first")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=system,
    tools=tools,
    messages=[...],
)

for block in response.content:
    if block.type == "tool_use":
        result = cost_context(block)  # Auto-detects Anthropic format
```

### The Three Agent Tools

| Tool | What It Does | When to Call |
|------|-------------|-------------|
| `estimate_cost` | Pre-flight cost + capability check + alternatives | Before every LLM call |
| `compare_models` | Rank models by cost-quality for a specific task type | When choosing which model to use |
| `check_budget` | Live spend, remaining budget, projections, strategy advice | Periodically, or when budget is tight |

### Strategy Presets

```python
# Balanced (default) -- smart tradeoffs
prompt = cost_prompt(strategy="balanced", daily_budget=5.00)

# Cost-first -- minimize spending, cheapest viable models
prompt = cost_prompt(strategy="cost_first", daily_budget=2.00)

# Quality-first -- best models, only save on trivial tasks
prompt = cost_prompt(strategy="quality_first", monthly_budget=100.00)
```

### MCP Server

Expose the tools via Model Context Protocol for Claude Desktop, Cursor, etc.

```bash
pip install llm-budget[mcp]
llm-budget serve-mcp
```

Or in your MCP config:

```json
{
  "mcpServers": {
    "llm-budget": {
      "command": "python",
      "args": ["-m", "llm_budget.mcp_server"]
    }
  }
}
```

---

## Framework Integrations

### LangChain

Budget-aware callback handler that plugs into any LangChain LLM.

```bash
pip install llm-budget[langchain] langchain-anthropic  # or langchain-openai
```

```python
from langchain_anthropic import ChatAnthropic
from llm_budget.integrations.langchain import BudgetCallbackHandler

# Tracking only (no enforcement)
handler = BudgetCallbackHandler()
llm = ChatAnthropic(model="claude-sonnet-4-20250514", callbacks=[handler])
result = llm.invoke("Explain quantum computing")

print(f"Cost: ${handler.total_cost:.6f}")
print(f"Tokens: {handler.total_tokens}")
```

With budget enforcement:

```python
from llm_budget.integrations.langchain import budget_callback
from llm_budget import BudgetExceeded

with budget_callback(max_cost=1.00, period="daily", on_exceed="raise") as cb:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", callbacks=[cb])
    try:
        for question in questions:
            result = llm.invoke(question)
    except BudgetExceeded:
        print(f"Budget hit after {cb.call_count} calls (${cb.total_cost:.4f})")
```

Works with any LangChain LLM: `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, etc.

### CrewAI

Budget hooks that register globally with CrewAI's hook system.

```bash
pip install llm-budget[crewai] crewai
```

```python
from llm_budget.integrations.crewai import track_crewai
from llm_budget import BudgetExceeded

with track_crewai(max_cost=10.00, period="daily") as hooks:
    crew = Crew(agents=[researcher, writer], tasks=[...])
    try:
        result = crew.kickoff()
    except BudgetExceeded:
        print(f"Budget hit: ${hooks.total_cost:.4f}")
    print(f"Crew run cost: ${hooks.total_cost:.4f} ({hooks.call_count} LLM calls)")
```

Or register hooks manually:

```python
from llm_budget.integrations.crewai import BudgetHooks

hooks = BudgetHooks(max_cost=10.00, period="daily")
hooks.register()
# ... run agents ...
hooks.unregister()
```

---

## CLI

```bash
# Check your spend
$ llm-budget status
LLM Budget Status
==================================================
  Daily      $0.45
  Weekly     $3.21
  Monthly    $12.54
  Total      $47.89

Breakdown by model (monthly):
  gpt-4o                              $9.80
  gpt-4o-mini                         $2.74

# Compare costs across models
$ llm-budget compare "Explain quantum computing" --models gpt-4o,gpt-4o-mini,claude-sonnet-4-20250514

# Estimate a single call
$ llm-budget estimate "Write a Python tutorial" --model gpt-4o --output-tokens 500

# List all 20 supported models with pricing
$ llm-budget models
$ llm-budget models --provider anthropic

# Update pricing from LiteLLM (stays current)
$ llm-budget update-prices

# Show recent API call history
$ llm-budget history --last 20
```

### CI Cost Gate (`check`)

Block deployments that exceed a cost threshold. Exit code 1 = over budget, exit code 0 = within budget.

```bash
# Basic: fail if estimated cost exceeds $0.50
$ llm-budget check "Summarize this document" -m gpt-4o --max-cost 0.50
[PASS] gpt-4o: $0.000021 estimated vs $0.500000 max

# Pipe prompt from file (stdin)
$ cat prompts/summarize.txt | llm-budget check -m gpt-4o --max-cost 0.10

# Machine-readable JSON output for CI parsing
$ llm-budget check "Hello" -m gpt-4o --max-cost 0.50 --json
{"model": "gpt-4o", "input_tokens": 2, "output_tokens": 2, "estimated_cost": 1.8e-05, "max_cost": 0.5, "passed": true}

# With explicit output token estimate
$ llm-budget check "Translate this book" -m gpt-4o --max-cost 1.00 -o 4000
```

**GitHub Actions example:**

```yaml
jobs:
  cost-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install llm-budget
      - name: Check prompt cost
        run: llm-budget check -m gpt-4o --max-cost 1.00 --json < prompts/main.txt
```

**Options:**

| Flag | Description |
|------|-------------|
| `PROMPT` | Prompt text (argument) or pipe via stdin |
| `-m, --model` | Model to estimate against (required) |
| `--max-cost` | Maximum allowed cost in USD (required) |
| `-o, --output-tokens` | Expected output tokens (otherwise heuristic) |
| `--json` | Output as JSON for machine parsing |

---

## Supported Models

20 models across 7 providers. Pricing auto-updates via `llm-budget update-prices`.

| Provider | Models | Tier |
|----------|--------|------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini | Frontier / Efficient / Reasoning |
| **Anthropic** | claude-sonnet-4, claude-opus-4, claude-3.5-haiku, claude-3.5-sonnet | Frontier / Efficient |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | Efficient / Reasoning |
| **Google** | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash | Frontier / Efficient |
| **Meta** | llama-3.1-70b, llama-3.1-8b | Mid / Efficient |
| **Mistral** | mistral-large, mistral-small | Mid / Efficient |

### Capability Tiers

| Tier | Quality Score | Cost Range (per call) | Best For |
|------|--------------|----------------------|----------|
| **Reasoning** | 88-98 | $0.001 - $0.01 | Math, proofs, science, multi-step logic |
| **Frontier** | 88-96 | $0.003 - $0.02 | Complex analysis, coding, creative writing |
| **Mid** | 75-78 | $0.001 - $0.003 | General tasks, moderate reasoning |
| **Efficient** | 55-73 | $0.0001 - $0.003 | Extraction, formatting, translation, Q&A |

---

## API Reference

### Smart Routing

- `smart_route(client, task, *, task_type, budget, model, max_tokens, **api_kwargs)` -- One-call routing + tracking
- `cost_aware(budget, task_type, provider, tracker)` -- Decorator for auto model selection

### Cost Estimation

- `estimate(messages, model, expected_output_tokens)` -- Pre-flight cost estimate
- `compare(messages, models, expected_output_tokens)` -- Compare costs across models
- `count_tokens(text_or_messages, model)` -- Count tokens

### Budget Enforcement

- `@budget(max_cost, period, on_exceed, alert_at, track_model)` -- Decorator
- `BudgetEnforcer(tracker)` -- Programmatic enforcement
- `BudgetPolicy(max_cost, period, on_exceed, alert_at)` -- Policy config
- `BudgetExceeded` -- Exception (has `.suggested_model`, `.remaining_budget`)

### Spend Tracking

- `Tracker(db_path=None)` -- SQLite spend tracker
- `get_tracker()` -- Default tracker singleton
- `tracker.record(model, input_tokens, output_tokens, cost_usd)` -- Record a call
- `tracker.get_spend(period, model)` -- Query spend
- `tracker.get_spend_breakdown(period)` -- Spend by model
- `tracker.get_history(last_n, model)` -- Recent records
- `track_openai(client, tracker)` -- Auto-tracking middleware for OpenAI
- `track_anthropic(client, tracker)` -- Auto-tracking middleware for Anthropic

### Model Intelligence

- `recommend_models(prompt, task_type, budget_remaining, available_models)` -- Get model recommendations
- `is_model_suitable(model, task_type)` -- Check if model fits task `(bool, reason)`
- `get_capability_summary(model)` -- One-line model summary
- `get_pricing(model)` -- Get `ModelPricing` object
- `get_registry()` -- Access the full pricing registry

### Agent Tools

- `cost_tools_openai()` -- Tool schemas in OpenAI function calling format
- `cost_tools_anthropic()` -- Tool schemas in Anthropic tool_use format
- `cost_tools()` -- Framework-agnostic tool definitions
- `cost_context(tool_call, tracker, budget_config)` -- Universal tool handler (auto-detects format)
- `cost_prompt(daily_budget, weekly_budget, monthly_budget, strategy)` -- System prompt generator

### Framework Integrations

**LangChain** (`from llm_budget.integrations.langchain import ...`):
- `BudgetCallbackHandler(max_cost, period, on_exceed, alert_at, tracker, model)` -- LangChain callback handler
- `budget_callback(...)` -- Context manager that creates + yields a handler

**CrewAI** (`from llm_budget.integrations.crewai import ...`):
- `BudgetHooks(max_cost, period, on_exceed, alert_at, tracker, model, default_model)` -- CrewAI hook class
- `track_crewai(...)` -- Context manager that creates + registers hooks

---

## Feature Comparison

| Feature | llm-budget | TokenCost | LiteLLM |
|---------|-----------|-----------|---------|
| **Automatic model routing** | Yes (`smart_route`) | No | No |
| **Pre-flight cost estimation** | Yes | No | No |
| **Budget enforcement** | Yes (decorator) | No | No |
| **Auto-downgrade on budget** | Yes | No | No |
| **Agent cost tools** | Yes (3 tools) | No | No |
| **LangChain integration** | Yes (callback handler) | No | No |
| **CrewAI integration** | Yes (hooks) | No | No |
| **MCP server** | Yes | No | No |
| **Standalone (no SDK required)** | Yes | Yes | No |
| **Multi-model comparison** | Yes | No | Partial |
| **Local SQLite tracking** | Yes | No | Yes (proxy) |
| **Zero cloud dependency** | Yes | Yes | No |
| **CLI** | Yes | No | Yes |

---

## Contributing

llm-budget is open source under the MIT license. Contributions are welcome!

```bash
git clone https://github.com/aman-source/llm-budget.git
cd llm-budget
pip install -e ".[dev]"
pytest
```

**Areas where we'd love help:**
- **Framework integrations** -- Add support for LlamaIndex, AutoGen, or other frameworks
- **Async support** -- LangChain `ainvoke()` and streaming response tracking
- **Provider coverage** -- Add pricing for new models and providers
- **Bug fixes** -- See [open issues](https://github.com/aman-source/llm-budget/issues)

**Running tests:**

```bash
pytest                          # All tests (269)
pytest --cov=llm_budget -q      # With coverage (target: 90%+)
pytest tests/test_langchain_integration.py -v  # Specific module
```

All framework integrations (LangChain, CrewAI, MCP) use optional dependencies with mocked tests -- you don't need to install the frameworks to run the test suite.

---

## License

MIT
