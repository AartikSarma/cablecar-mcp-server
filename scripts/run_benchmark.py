#!/usr/bin/env python
"""Run the hypothesis discovery benchmark suite.

Supports both statistical agents and LLM agents (via the model-agnostic
tool-use harness).

Usage examples::

    # Statistical agent only, one quality/context combination
    uv run python scripts/run_benchmark.py \\
        --qualities perfect \\
        --contexts full_vignette

    # Full statistical matrix
    uv run python scripts/run_benchmark.py \\
        --qualities perfect partial naive \\
        --contexts full_vignette domain_hint blind \\
        --output results/benchmark_run.json

    # LLM agent (requires API key)
    uv run python scripts/run_benchmark.py \\
        --agent llm \\
        --provider anthropic \\
        --model claude-sonnet-4-5-20250514 \\
        --scenario easy \\
        --contexts full_vignette
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cablecar.evaluation.agent import BenchmarkHarness, StatisticalAgent
from cablecar.evaluation.benchmarks import BenchmarkScore, DiscoveryBenchmark
from cablecar.evaluation.dgp import ContextLevel, DGPSpec
from cablecar.evaluation.harness import ToolUseAgent
from cablecar.evaluation.scenarios import ALL_SCENARIOS


VALID_QUALITIES = ("perfect", "partial", "naive")
VALID_CONTEXTS = {
    "full_vignette": ContextLevel.FULL_VIGNETTE,
    "domain_hint": ContextLevel.DOMAIN_HINT,
    "blind": ContextLevel.BLIND,
}
VALID_PROVIDERS = ("anthropic", "google", "openai")


def _build_llm_agent(
    provider: str,
    model: str | None = None,
    max_turns: int = 20,
) -> tuple[str, Any]:
    """Build (label, agent) for an LLM provider."""
    if provider == "anthropic":
        from cablecar.evaluation.adapters.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter(model=model or "claude-sonnet-4-5-20250514")
    elif provider == "google":
        from cablecar.evaluation.adapters.google import GoogleAdapter
        adapter = GoogleAdapter(model=model or "gemini-2.0-flash")
    elif provider == "openai":
        from cablecar.evaluation.adapters.openai import OpenAIAdapter
        adapter = OpenAIAdapter(model=model or "gpt-4o")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    label = f"llm_{provider}_{adapter.model_name}"
    agent = ToolUseAgent(adapter, max_turns=max_turns)
    return label, agent


def _build_agents(
    qualities: list[str],
) -> list[tuple[str, Any]]:
    """Build (label, agent) pairs from CLI arguments."""
    agents: list[tuple[str, Any]] = []
    for quality in qualities:
        label = f"statistical_{quality}"
        agents.append((label, StatisticalAgent(quality=quality)))
    return agents


def _run_benchmark(
    agents: list[tuple[str, Any]],
    specs: list[DGPSpec],
    context_levels: list[ContextLevel],
) -> list[dict[str, Any]]:
    """Run all agent x spec x context combinations and return score dicts."""
    all_scores: list[dict[str, Any]] = []

    for agent_label, agent in agents:
        harness = BenchmarkHarness(agent)
        for spec in specs:
            for ctx_level in context_levels:
                print(
                    f"  Running {agent_label} on {spec.name} "
                    f"[{ctx_level.value}]...",
                    end="",
                    flush=True,
                )
                score = harness.run_scenario(spec, ctx_level)
                print(f" score={score.overall_score:.4f}")
                all_scores.append({
                    "agent": agent_label,
                    "scenario": spec.name,
                    "difficulty": spec.difficulty.value,
                    "context_level": ctx_level.value,
                    "overall_score": round(score.overall_score, 4),
                    "dimensions": {
                        d.name: round(d.score, 4) for d in score.dimension_scores
                    },
                    "feedback": score.feedback,
                })

    return all_scores


def _print_report(scores: list[dict[str, Any]]) -> None:
    """Print a formatted benchmark report to stdout."""
    print()
    print("=" * 72)
    print("  BENCHMARK REPORT")
    print("=" * 72)

    if not scores:
        print("  No results.")
        return

    overall = sum(s["overall_score"] for s in scores) / len(scores)
    print(f"\n  Overall Score: {overall:.4f}  ({len(scores)} runs)")

    # By difficulty
    by_diff: dict[str, list[float]] = {}
    for s in scores:
        by_diff.setdefault(s["difficulty"], []).append(s["overall_score"])
    print("\n  --- By Difficulty ---")
    for tier in sorted(by_diff):
        vals = by_diff[tier]
        print(f"    {tier:12s}: {sum(vals)/len(vals):.4f}")

    # By context
    by_ctx: dict[str, list[float]] = {}
    for s in scores:
        by_ctx.setdefault(s["context_level"], []).append(s["overall_score"])
    print("\n  --- By Context Level ---")
    for ctx in sorted(by_ctx):
        vals = by_ctx[ctx]
        print(f"    {ctx:16s}: {sum(vals)/len(vals):.4f}")

    # By agent
    by_agent: dict[str, list[float]] = {}
    for s in scores:
        by_agent.setdefault(s["agent"], []).append(s["overall_score"])
    print("\n  --- By Agent ---")
    for agent in sorted(by_agent):
        vals = by_agent[agent]
        print(f"    {agent:24s}: {sum(vals)/len(vals):.4f}")

    # Per-run breakdown
    print(f"\n  --- Per-Run Breakdown ---")
    print(f"    {'Agent':<24s} {'Scenario':<30s} {'Context':<16s} {'Score':>6s}")
    print("    " + "-" * 78)
    for s in sorted(scores, key=lambda x: (x["agent"], x["difficulty"], x["context_level"])):
        print(
            f"    {s['agent']:<24s} {s['scenario']:<30s} "
            f"{s['context_level']:<16s} {s['overall_score']:>6.4f}"
        )

    print("\n" + "=" * 72)


def _build_output(scores: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Build the JSON output structure."""
    if not scores:
        return {"run_id": run_id, "agents": [], "summary": {}, "scores": []}

    overall = sum(s["overall_score"] for s in scores) / len(scores)

    by_diff: dict[str, list[float]] = {}
    for s in scores:
        by_diff.setdefault(s["difficulty"], []).append(s["overall_score"])

    by_ctx: dict[str, list[float]] = {}
    for s in scores:
        by_ctx.setdefault(s["context_level"], []).append(s["overall_score"])

    by_agent: dict[str, list[float]] = {}
    for s in scores:
        by_agent.setdefault(s["agent"], []).append(s["overall_score"])

    return {
        "run_id": run_id,
        "agents": sorted(set(s["agent"] for s in scores)),
        "summary": {
            "overall": round(overall, 4),
            "by_difficulty": {k: round(sum(v)/len(v), 4) for k, v in sorted(by_diff.items())},
            "by_context_level": {k: round(sum(v)/len(v), 4) for k, v in sorted(by_ctx.items())},
            "by_agent": {k: round(sum(v)/len(v), 4) for k, v in sorted(by_agent.items())},
        },
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CableCar hypothesis discovery benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --qualities perfect --contexts full_vignette\n"
            "  %(prog)s --qualities perfect partial naive --output results.json\n"
            "  %(prog)s --agent llm --provider anthropic --scenarios easy --contexts full_vignette\n"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=["statistical", "llm"],
        default="statistical",
        help="Agent type: 'statistical' (default) or 'llm' (requires API key)",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        choices=VALID_QUALITIES,
        default=list(VALID_QUALITIES),
        help="Quality levels for the statistical agent (default: all)",
    )
    parser.add_argument(
        "--provider",
        choices=VALID_PROVIDERS,
        default="anthropic",
        help="LLM provider (default: anthropic). Only used when --agent=llm.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'claude-sonnet-4-5-20250514'). Uses provider default if omitted.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum tool-use turns for LLM agents (default: 20).",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=list(VALID_CONTEXTS.keys()),
        default=list(VALID_CONTEXTS.keys()),
        help="Context levels to test (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(ALL_SCENARIOS.keys()),
        default=list(ALL_SCENARIOS.keys()),
        help="Scenario difficulty tiers to include (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (optional)",
    )

    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).isoformat()
    specs = [ALL_SCENARIOS[name]() for name in args.scenarios]
    context_levels = [VALID_CONTEXTS[c] for c in args.contexts]

    if args.agent == "llm":
        label, agent = _build_llm_agent(
            provider=args.provider,
            model=args.model,
            max_turns=args.max_turns,
        )
        agents = [(label, agent)]
    else:
        agents = _build_agents(args.qualities)

    print(f"Benchmark run: {run_id}")
    print(f"  Agents: {[label for label, _ in agents]}")
    print(f"  Scenarios: {[s.name for s in specs]}")
    print(f"  Contexts: {[c.value for c in context_levels]}")
    print()

    scores = _run_benchmark(agents, specs, context_levels)
    _print_report(scores)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = _build_output(scores, run_id)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
