"""CLI commands for llm-budget."""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import click

from ._version import __version__
from .estimator import compare as compare_costs
from .estimator import estimate as estimate_cost
from .pricing import get_registry
from .tracker import Tracker, get_tracker

logger = logging.getLogger(__name__)

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Models we track in our local models.json
TRACKED_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
    "o1", "o1-mini", "o3-mini",
    "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022",
    "claude-opus-4-20250514", "claude-3-5-sonnet-20241022",
    "deepseek-chat", "deepseek-reasoner",
    "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
    "llama-3.1-70b", "llama-3.1-8b",
    "mistral-large", "mistral-small",
]


def _format_usd(amount: float) -> str:
    if amount < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.2f}"


@click.group()
@click.version_option(version=__version__, prog_name="llm-budget")
def cli() -> None:
    """llm-budget: Pre-flight LLM cost estimation and budget tracking."""


@cli.command()
@click.option(
    "--period",
    default="monthly",
    type=click.Choice(
        ["hourly", "daily", "weekly", "monthly", "total"],
        case_sensitive=False,
    ),
    help="Time period for spend summary.",
)
def status(period: str) -> None:
    """Show current spend status and budget breakdown."""
    tracker = get_tracker()

    periods = ["daily", "weekly", "monthly", "total"]
    click.echo("LLM Budget Status")
    click.echo("=" * 50)

    for p in periods:
        total = tracker.get_spend(period=p)
        label = p.capitalize().ljust(10)
        click.echo(f"  {label} {_format_usd(total)}")

    click.echo()
    breakdown = tracker.get_spend_breakdown(period=period)
    if breakdown:
        click.echo(f"Breakdown by model ({period}):")
        for model, cost in breakdown.items():
            click.echo(f"  {model:<35} {_format_usd(cost)}")
    else:
        click.echo("No spend recorded yet.")


@cli.command()
@click.argument("prompt")
@click.option(
    "--models", "-m",
    required=True,
    help="Comma-separated list of models to compare.",
)
@click.option("--output-tokens", "-o", default=None, type=int)
def compare(prompt: str, models: str, output_tokens: Optional[int]) -> None:
    """Compare estimated costs across models for a given prompt."""
    model_list = [m.strip() for m in models.split(",")]
    compare_costs(prompt, model_list, output_tokens)


@cli.command(name="estimate")
@click.argument("prompt")
@click.option("--model", "-m", required=True, help="Model name.")
@click.option("--output-tokens", "-o", default=None, type=int)
def estimate_cmd(
    prompt: str, model: str, output_tokens: Optional[int]
) -> None:
    """Estimate cost for a single model and prompt."""
    result = estimate_cost(prompt, model, output_tokens)
    click.echo(str(result))
    click.echo(f"  Input cost:  {result.breakdown['input']}")
    click.echo(f"  Output cost: {result.breakdown['output']}")


@cli.command()
@click.option("--last", "limit", default=20, help="Number of records to show.")
@click.option("--model", default=None, help="Filter by model name.")
def history(limit: int, model: Optional[str]) -> None:
    """Show recent API call history."""
    tracker = get_tracker()
    records = tracker.get_history(last_n=limit, model=model)

    if not records:
        click.echo("No history recorded yet.")
        return

    header = (
        f"{'Timestamp':<22} {'Model':<30} {'In Tok':>8} "
        f"{'Out Tok':>8} {'Cost':>12}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for r in records:
        click.echo(
            f"{r.timestamp:<22} {r.model:<30} {r.input_tokens:>8} "
            f"{r.output_tokens:>8} {_format_usd(r.cost_usd):>12}"
        )


@cli.command(name="update-prices")
def update_prices() -> None:
    """Fetch latest pricing from LiteLLM and update local models.json."""
    click.echo("Fetching latest pricing data from LiteLLM...")

    try:
        req = urllib.request.Request(LITELLM_PRICES_URL)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        click.echo(f"Error fetching pricing data: {exc}", err=True)
        raise SystemExit(1)

    # Load current models.json to preserve tokenizer mappings
    models_path = Path(__file__).parent / "models.json"
    with open(models_path, "r", encoding="utf-8") as f:
        current = json.load(f)

    updated = 0
    for model_name in TRACKED_MODELS:
        if model_name not in raw:
            continue
        info = raw[model_name]
        inp = info.get("input_cost_per_token")
        out = info.get("output_cost_per_token")
        max_in = info.get("max_input_tokens")
        max_out = info.get("max_output_tokens")

        if inp is None or out is None:
            continue

        # Preserve existing tokenizer or default to cl100k_base
        tokenizer = (
            current.get(model_name, {}).get("tokenizer", "cl100k_base")
        )
        provider = current.get(model_name, {}).get(
            "provider",
            info.get("litellm_provider", "unknown").split("/")[0],
        )

        existing = current.get(model_name, {})
        existing.update({
            "provider": provider,
            "input_cost_per_token": inp,
            "output_cost_per_token": out,
            "max_input_tokens": max_in or existing.get(
                "max_input_tokens", 128000
            ),
            "max_output_tokens": max_out or existing.get(
                "max_output_tokens", 4096
            ),
            "tokenizer": tokenizer,
        })
        current[model_name] = existing
        updated += 1

    with open(models_path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)

    # Reload the in-memory registry so subsequent calls use new prices
    get_registry().reload()

    click.echo(f"Updated {updated} model(s) in {models_path}")


@cli.command(name="models")
@click.option(
    "--provider", "-p",
    default=None,
    help="Filter by provider (openai, anthropic, google, etc.).",
)
def list_models(provider: Optional[str]) -> None:
    """List all supported models and their pricing."""
    registry = get_registry()
    models = registry.list_models(provider=provider)

    if not models:
        click.echo("No models found.")
        return

    header = (
        f"{'Model':<35} {'Provider':<12} "
        f"{'Input $/1M tok':>15} {'Output $/1M tok':>16}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for m in sorted(models, key=lambda x: x.provider):
        click.echo(
            f"{m.name:<35} {m.provider:<12} "
            f"${m.input_cost_per_token * 1_000_000:>14.2f} "
            f"${m.output_cost_per_token * 1_000_000:>15.2f}"
        )


@cli.command(name="serve-mcp")
def serve_mcp() -> None:
    """Start MCP server for agent integration (requires mcp package)."""
    from .mcp_server import MCP_AVAILABLE

    if not MCP_AVAILABLE:
        click.echo(
            "MCP server requires the 'mcp' package.\n"
            "Install with: pip install llm-budget[mcp]",
            err=True,
        )
        raise SystemExit(1)

    import asyncio
    from .mcp_server import run_mcp_server

    click.echo("Starting llm-budget MCP server...")
    asyncio.run(run_mcp_server())


def main() -> None:
    """Entry point for the CLI."""
    cli()
