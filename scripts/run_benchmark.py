#!/usr/bin/env python
"""Run the hypothesis discovery benchmark suite.

Usage examples::

    # Statistical agent only, one quality/context combination
    uv run python scripts/run_benchmark.py \\
        --agents statistical \\
        --qualities perfect \\
        --contexts full_vignette

    # Full statistical matrix
    uv run python scripts/run_benchmark.py \\
        --agents statistical \\
        --qualities perfect partial naive \\
        --contexts full_vignette domain_hint blind \\
        --output results/benchmark_run.json

    # Include Claude agent (requires ANTHROPIC_API_KEY)
    uv run python scripts/run_benchmark.py \\
        --agents statistical claude \\
        --model claude-sonnet-4-5-20250929 \\
        --output results/benchmark_run.json
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
from cablecar.evaluation.scenarios import ALL_SCENARIOS


VALID_QUALITIES = ("perfect", "partial", "naive")
VALID_CONTEXTS = {
    "full_vignette": ContextLevel.FULL_VIGNETTE,
    "domain_hint": ContextLevel.DOMAIN_HINT,
    "blind": ContextLevel.BLIND,
}


def _build_agents(
    agent_names: list[str],
    qualities: list[str],
    model: str,
) -> list[tuple[str, Any]]:
    """Build (label, agent) pairs from CLI arguments."""
    agents: list[tuple[str, Any]] = []
    for name in agent_names:
        if name == "statistical":
            for quality in qualities:
                label = f"statistical_{quality}"
                agents.append((label, StatisticalAgent(quality=quality)))
        elif name == "claude":
            from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent

            agents.append(("claude", ClaudeDiscoveryAgent(model=model)))
        else:
            print(f"Unknown agent: {name}", file=sys.stderr)
            sys.exit(1)
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
            "  %(prog)s --agents statistical --qualities perfect --contexts full_vignette\n"
            "  %(prog)s --agents statistical --qualities perfect partial naive --output results.json\n"
            "  %(prog)s --agents statistical claude --model claude-sonnet-4-5-20250929\n"
        ),
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["statistical", "claude"],
        default=["statistical"],
        help="Agent types to benchmark (default: statistical)",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        choices=VALID_QUALITIES,
        default=list(VALID_QUALITIES),
        help="Quality levels for the statistical agent (default: all)",
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
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Claude model ID (only used with --agents claude)",
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
    agents = _build_agents(args.agents, args.qualities, args.model)

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
