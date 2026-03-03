"""Subagent-based benchmark runner utilities.

Provides four public functions that Claude Code orchestrates via Bash and Task
subagents to run hypothesis discovery benchmarks:

- :func:`prepare_scenario` — generate synthetic data to disk
- :func:`build_agent_prompt` — build the Task subagent prompt
- :func:`parse_agent_output` — extract a DiscoveryResult from raw agent text
- :func:`score_result` — score a DiscoveryResult against a DGP spec

CLI usage::

    # Generate data
    uv run python -m cablecar.evaluation.benchmark_runner prepare --scenario easy

    # Build prompt
    uv run python -m cablecar.evaluation.benchmark_runner prompt \\
        --scenario easy --data-path /tmp/.../tables --context full_vignette

    # Parse + score
    uv run python -m cablecar.evaluation.benchmark_runner parse-and-score \\
        --scenario easy --context full_vignette --input-file /tmp/agent_output.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from cablecar.evaluation.benchmarks import BenchmarkScore, DiscoveryBenchmark
from cablecar.evaluation.dgp import ContextLevel, DGPSpec
from cablecar.evaluation.discovery_result import DiscoveryResult
from cablecar.evaluation.scenarios import ALL_SCENARIOS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_scenario(spec: DGPSpec, base_dir: Path | None = None) -> Path:
    """Generate synthetic data for a benchmark scenario.

    Writes CSVs to ``{output_dir}/tables/`` and the spec JSON to
    ``{output_dir}/ground_truth/``.

    Parameters
    ----------
    spec:
        DGP specification to generate data from.
    base_dir:
        Parent directory for the scenario output. If ``None``, uses a fresh
        temp directory.

    Returns
    -------
    Path
        Absolute path to the ``tables/`` directory (what ``load_data`` needs).
    """
    from synthetic.dgp_generator import DGPSyntheticGenerator

    if base_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="cablecar_bench_"))
    else:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    generator = DGPSyntheticGenerator(spec)
    generator.generate_and_save(base_dir)

    tables_dir = (base_dir / "tables").resolve()
    return tables_dir


def build_agent_prompt(
    data_path: str | Path,
    spec: DGPSpec,
    context_level: ContextLevel,
) -> str:
    """Build the Task subagent prompt for hypothesis discovery.

    The prompt instructs the subagent to use CableCar slash commands
    (``/load-data``, ``/data-dictionary``, ``/define-cohort``,
    ``/regression``, ``/dag``) to analyze the data and output a structured
    ``discovery_result`` JSON block.

    Ground truth is **never** included in the prompt.

    Parameters
    ----------
    data_path:
        Absolute path to the tables directory.
    spec:
        DGP specification (used for vignette/hint, never ground truth).
    context_level:
        How much clinical context to provide.

    Returns
    -------
    str
        The complete prompt text for a Task subagent.
    """
    data_path = str(Path(data_path).resolve())

    # Section 1: Role
    role_section = (
        "You are a clinical research analyst. Your task is to discover causal "
        "hypotheses from a clinical dataset. You must identify exposures, "
        "outcomes, confounders, and estimate causal effects using rigorous "
        "statistical methods.\n\n"
        "IMPORTANT: You have access to CableCar analysis tools via slash commands. "
        "Use these tools — do NOT attempt to read data files directly or run "
        "Python scripts. The slash commands are your primary interface.\n\n"
        "Do NOT read source code files, test files, or any Python files in this "
        "project. You must discover everything from the data using only the "
        "analysis tools provided."
    )

    # Section 2: Data loading
    data_section = (
        "## Data Loading\n\n"
        "Start by loading the dataset using the `/load-data` command:\n"
        f"- Run `/load-data {data_path}` to load the clinical data.\n"
        "- Run `/data-dictionary` to understand the available tables and columns."
    )

    # Section 3: Context (varies by level)
    if context_level == ContextLevel.FULL_VIGNETTE:
        context_section = (
            "## Clinical Context\n\n"
            f"{spec.vignette}"
        )
    elif context_level == ContextLevel.DOMAIN_HINT:
        context_section = (
            "## Domain Hint\n\n"
            f"{spec.domain_hint}"
        )
    else:  # BLIND
        context_section = (
            "## Context\n\n"
            "No prior information is provided about this dataset. "
            "Explore the data to discover relationships."
        )

    # Section 4: Analysis protocol
    protocol_section = (
        "## Analysis Protocol\n\n"
        "You have these slash commands available:\n"
        "- `/load-data <path>` — Load a clinical dataset\n"
        "- `/data-dictionary` — Explore schema and data structure\n"
        "- `/define-cohort <criteria>` — Define study cohort with inclusion/exclusion criteria\n"
        "- `/table1 <variables>` — Generate baseline characteristics table\n"
        "- `/hypothesis <spec>` — Run hypothesis tests\n"
        "- `/dag <variables>` — Build a causal DAG and identify adjustment sets\n"
        "- `/regression <spec>` — Fit regression models\n"
        "- `/analyze <request>` — General analysis dispatch\n\n"
        "Follow this workflow:\n"
        f"1. **Load data**: `/load-data {data_path}`\n"
        "2. **Profile the data**: `/data-dictionary` to understand table structures, "
        "then `/table1` for descriptive statistics.\n"
        "3. **Define a cohort**: `/define-cohort` with appropriate inclusion/exclusion criteria.\n"
        "4. **Build a causal DAG**: `/dag` to identify the causal structure — exposures, "
        "outcomes, confounders, mediators, and colliders. Get the adjustment set.\n"
        "5. **Regression analysis**: `/regression` to estimate causal effects with "
        "appropriate adjustment sets from the DAG.\n"
        "6. **Interpret results**: Summarize findings, effect sizes, confidence "
        "intervals, and limitations."
    )

    # Section 5: Required output format
    output_section = (
        "## Required Output Format\n\n"
        "After completing your analysis, output your findings in a fenced block "
        "with the tag `discovery_result`. The JSON must match this schema:\n\n"
        "```discovery_result\n"
        "{\n"
        '  "identified_exposure": "<primary exposure variable name>",\n'
        '  "identified_outcome": "<primary outcome variable name>",\n'
        '  "identified_confounders": ["<var1>", "<var2>"],\n'
        '  "identified_mediators": [],\n'
        '  "identified_colliders": [],\n'
        '  "primary_hypothesis": "<text description of primary hypothesis>",\n'
        '  "secondary_hypotheses": [],\n'
        '  "proposed_dag_edges": [["<cause>", "<effect>"], ...],\n'
        '  "proposed_adjustment_set": ["<var1>", "<var2>"],\n'
        '  "methods_used": ["logistic_regression", "descriptive"],\n'
        '  "method_justification": "<why these methods>",\n'
        '  "estimated_effect": 0.0,\n'
        '  "confidence_interval": [0.0, 0.0],\n'
        '  "p_value": 0.0,\n'
        '  "effect_size_metric": "log_odds_ratio",\n'
        '  "missingness_strategy": "",\n'
        '  "missingness_assessment": "",\n'
        '  "interpretation": "<interpretation of findings>",\n'
        '  "limitations": ["<limitation1>"]\n'
        "}\n"
        "```\n\n"
        "Important: You MUST include this fenced block in your final output."
    )

    prompt = "\n\n".join([
        role_section,
        data_section,
        context_section,
        protocol_section,
        output_section,
    ])

    return prompt


def parse_agent_output(text: str) -> DiscoveryResult:
    """Extract a DiscoveryResult from raw subagent output text.

    Parsing strategy:
    1. Primary: look for a ``discovery_result`` fenced block
    2. Fallback: look for the last ``json`` fenced block containing
       ``"identified_exposure"``
    3. Last resort: return a minimal fallback result

    Handles type coercion for dag_edges (list of lists -> list of tuples)
    and confidence_interval (list -> tuple).
    """
    # Strategy 1: discovery_result fence
    pattern_dr = r"```discovery_result\s*\n(.*?)```"
    match = re.search(pattern_dr, text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return _coerce_to_discovery_result(parsed)

    # Strategy 2: last json fence containing identified_exposure
    pattern_json = r"```json\s*\n(.*?)```"
    json_matches = list(re.finditer(pattern_json, text, re.DOTALL))
    for m in reversed(json_matches):
        raw = m.group(1).strip()
        if '"identified_exposure"' in raw:
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return _coerce_to_discovery_result(parsed)

    # Strategy 3: fallback
    return DiscoveryResult(
        identified_exposure="unknown",
        identified_outcome="unknown",
        primary_hypothesis="Agent did not produce structured results",
    )


def score_result(
    spec: DGPSpec,
    result: DiscoveryResult,
    context_level: ContextLevel,
) -> BenchmarkScore:
    """Score a discovery result against a DGP specification.

    Thin wrapper around :class:`DiscoveryBenchmark.run_scenario`.
    """
    benchmark = DiscoveryBenchmark()
    return benchmark.run_scenario(spec, result, context_level)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Attempt to parse a JSON string, returning None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_to_discovery_result(data: dict[str, Any]) -> DiscoveryResult:
    """Coerce parsed JSON dict into a DiscoveryResult.

    Handles common type mismatches from LLM output:
    - dag_edges: [[X, Y], ...] -> [(X, Y), ...]
    - confidence_interval: [lo, hi] -> (lo, hi)
    """
    # Coerce dag_edges: list of lists -> list of tuples
    if "proposed_dag_edges" in data:
        edges = data["proposed_dag_edges"]
        if isinstance(edges, list):
            data["proposed_dag_edges"] = [
                tuple(e) if isinstance(e, (list, tuple)) else e
                for e in edges
            ]

    # Coerce confidence_interval: list -> tuple
    if "confidence_interval" in data:
        ci = data["confidence_interval"]
        if isinstance(ci, list) and len(ci) == 2:
            data["confidence_interval"] = tuple(ci)
        elif ci is not None and not isinstance(ci, tuple):
            data["confidence_interval"] = None

    return DiscoveryResult(**data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VALID_CONTEXTS = {
    "full_vignette": ContextLevel.FULL_VIGNETTE,
    "domain_hint": ContextLevel.DOMAIN_HINT,
    "blind": ContextLevel.BLIND,
}


def _cli_prepare(args: argparse.Namespace) -> None:
    """Handle the ``prepare`` subcommand."""
    spec_factory = ALL_SCENARIOS[args.scenario]
    spec = spec_factory()
    base_dir = Path(args.output_dir) if args.output_dir else None
    tables_path = prepare_scenario(spec, base_dir=base_dir)
    print(str(tables_path))


def _cli_prompt(args: argparse.Namespace) -> None:
    """Handle the ``prompt`` subcommand."""
    spec_factory = ALL_SCENARIOS[args.scenario]
    spec = spec_factory()
    context_level = VALID_CONTEXTS[args.context]
    prompt = build_agent_prompt(args.data_path, spec, context_level)
    print(prompt)


def _cli_parse_and_score(args: argparse.Namespace) -> None:
    """Handle the ``parse-and-score`` subcommand."""
    spec_factory = ALL_SCENARIOS[args.scenario]
    spec = spec_factory()
    context_level = VALID_CONTEXTS[args.context]

    agent_text = Path(args.input_file).read_text()
    result = parse_agent_output(agent_text)
    score = score_result(spec, result, context_level)

    output = {
        "scenario": spec.name,
        "context_level": context_level.value,
        "overall_score": round(score.overall_score, 4),
        "dimensions": {
            d.name: round(d.score, 4) for d in score.dimension_scores
        },
        "feedback": score.feedback,
        "parsed_result": {
            "identified_exposure": result.identified_exposure,
            "identified_outcome": result.identified_outcome,
            "primary_hypothesis": result.primary_hypothesis,
            "estimated_effect": result.estimated_effect,
        },
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    """CLI entrypoint for benchmark runner utilities."""
    parser = argparse.ArgumentParser(
        description="CableCar benchmark runner utilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prep = subparsers.add_parser(
        "prepare",
        help="Generate synthetic data for a scenario.",
    )
    prep.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()),
        required=True,
        help="Scenario to generate.",
    )
    prep.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: temp dir).",
    )
    prep.set_defaults(func=_cli_prepare)

    # prompt
    pmt = subparsers.add_parser(
        "prompt",
        help="Build the subagent prompt for a scenario.",
    )
    pmt.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()),
        required=True,
        help="Scenario to build prompt for.",
    )
    pmt.add_argument(
        "--data-path",
        required=True,
        help="Absolute path to tables directory.",
    )
    pmt.add_argument(
        "--context",
        choices=list(VALID_CONTEXTS.keys()),
        required=True,
        help="Context level for the agent.",
    )
    pmt.set_defaults(func=_cli_prompt)

    # parse-and-score
    pas = subparsers.add_parser(
        "parse-and-score",
        help="Parse agent output and score it.",
    )
    pas.add_argument(
        "--scenario",
        choices=list(ALL_SCENARIOS.keys()),
        required=True,
        help="Scenario the agent was run on.",
    )
    pas.add_argument(
        "--context",
        choices=list(VALID_CONTEXTS.keys()),
        required=True,
        help="Context level the agent was given.",
    )
    pas.add_argument(
        "--input-file",
        required=True,
        help="Path to file containing raw agent output.",
    )
    pas.set_defaults(func=_cli_parse_and_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
