"""Model-agnostic tool-use harness for hypothesis discovery benchmarking.

Provides four components:

1. :func:`get_tool_schemas` — returns the 4 MCP tool schemas matching
   ``cablecar/server/main.py``.
2. :class:`InProcessToolDispatcher` — writes DataFrames to a temp dir,
   creates a :class:`DataServerTools` instance, and routes tool calls.
   The LLM never learns the real filesystem path.
3. :func:`build_tool_use_system_prompt` — constructs the LLM system prompt
   with role, context (varies by :class:`ContextLevel`), workflow guidance,
   and the expected ``discovery_result`` output format.  Ground truth is
   **never** included.
4. :class:`ToolUseAgent` — a :class:`DiscoveryAgent` that drives any
   :class:`LLMAdapter` through the tool-use loop, then parses the output
   into a :class:`DiscoveryResult`.

Integrates with the existing :class:`BenchmarkHarness` with zero changes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from cablecar.evaluation.adapters import (
    AdapterResult,
    LLMAdapter,
    ToolDispatcher,
    ToolSchema,
)
from cablecar.evaluation.agent import AgentContext, DiscoveryAgent
from cablecar.evaluation.benchmark_runner import parse_agent_output
from cablecar.evaluation.dgp import ContextLevel, DGPSpec
from cablecar.evaluation.discovery_result import DiscoveryResult
from cablecar.server.tools import DataServerTools


# ---------------------------------------------------------------------------
# 1. Tool schemas — must match cablecar/server/main.py exactly
# ---------------------------------------------------------------------------


def get_tool_schemas() -> list[ToolSchema]:
    """Return the 4 MCP tool schemas for the data server.

    These schemas define the tool interface that the LLM sees.  They must
    stay in sync with the Tool definitions in ``cablecar/server/main.py``.
    """
    return [
        ToolSchema(
            name="get_schema",
            description=(
                "Get the data schema and data dictionary. Returns table "
                "definitions, column types, and relationships. No patient "
                "data is returned."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        ToolSchema(
            name="load_data",
            description=(
                "Load and validate a clinical dataset. Returns a "
                "privacy-sanitized summary with table counts, validation "
                "status, and data quality metrics. Never returns raw "
                "patient data."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to directory containing data files "
                            "(CSV or Parquet)"
                        ),
                    },
                    "schema": {
                        "type": "string",
                        "description": (
                            "Schema to validate against (e.g., 'clif'). "
                            "Optional."
                        ),
                    },
                },
                "required": ["path"],
            },
        ),
        ToolSchema(
            name="query_cohort",
            description=(
                "Define a study cohort with inclusion/exclusion criteria. "
                "Returns a privacy-sanitized CONSORT flow diagram and "
                "cohort summary. Never returns raw patient data."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for this cohort",
                        "default": "main",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the cohort",
                    },
                    "inclusion": {
                        "type": "array",
                        "description": (
                            "Inclusion criteria. Each: {column, op "
                            "(==,!=,>,<,>=,<=,in,not_in,is_null,not_null), value}"
                        ),
                        "items": {"type": "object"},
                    },
                    "exclusion": {
                        "type": "array",
                        "description": (
                            "Exclusion criteria (same format as inclusion)"
                        ),
                        "items": {"type": "object"},
                    },
                    "index_table": {
                        "type": "string",
                        "description": (
                            "Base table for cohort (default: hospitalization)"
                        ),
                        "default": "hospitalization",
                    },
                },
            },
        ),
        ToolSchema(
            name="execute_analysis",
            description=(
                "Execute a statistical analysis on a defined cohort. "
                "Returns privacy-sanitized results (coefficients, CIs, "
                "p-values, aggregated statistics). Never returns raw "
                "patient data. Supported types: summary_stats, "
                "descriptive, hypothesis, regression, subgroup, "
                "survival, xgboost."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": (
                            "Type: summary_stats, descriptive, hypothesis, "
                            "regression, subgroup, survival, xgboost"
                        ),
                        "enum": [
                            "summary_stats",
                            "descriptive",
                            "hypothesis",
                            "regression",
                            "subgroup",
                            "survival",
                            "xgboost",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": "Analysis-specific parameters",
                    },
                    "cohort_name": {
                        "type": "string",
                        "description": "Which cohort to analyze",
                        "default": "main",
                    },
                },
                "required": ["analysis_type"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# 2. In-process tool dispatcher
# ---------------------------------------------------------------------------


class InProcessToolDispatcher:
    """Routes LLM tool calls to a local :class:`DataServerTools` instance.

    Writes DataFrames to a temporary directory as CSVs, then creates a
    ``DataServerTools`` that reads from that directory.  The LLM can pass
    any ``path`` argument to ``load_data`` — the dispatcher always
    substitutes its own temp directory, so the LLM never learns real
    filesystem paths.

    All outputs pass through :meth:`PrivacyGuard.sanitize_for_llm` via
    ``DataServerTools`` before reaching the adapter.
    """

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables
        self._temp_dir: Path | None = None
        self._tools: DataServerTools | None = None

    def setup(self) -> None:
        """Write CSVs to a temp directory and create the tools instance."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="cablecar_harness_"))
        tables_dir = self._temp_dir / "tables"
        tables_dir.mkdir()

        for table_name, df in self._tables.items():
            df.to_csv(tables_dir / f"{table_name}.csv", index=False)

        self._tools = DataServerTools()

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the appropriate DataServerTools method.

        The ``load_data`` tool always uses the dispatcher's own temp
        directory, regardless of what path the LLM provides.
        """
        if self._tools is None:
            raise RuntimeError(
                "Dispatcher not initialized. Call setup() first."
            )

        if tool_name == "get_schema":
            result = self._tools.get_schema()
        elif tool_name == "load_data":
            # Intercept path — always use our temp dir
            real_path = str(self._temp_dir / "tables")
            result = self._tools.load_data(
                path=real_path,
                schema=arguments.get("schema"),
            )
        elif tool_name == "query_cohort":
            result = self._tools.query_cohort(
                name=arguments.get("name", "main"),
                description=arguments.get("description", ""),
                inclusion=arguments.get("inclusion"),
                exclusion=arguments.get("exclusion"),
                index_table=arguments.get("index_table", "hospitalization"),
            )
        elif tool_name == "execute_analysis":
            result = self._tools.execute_analysis(
                analysis_type=arguments.get("analysis_type", ""),
                params=arguments.get("params", {}),
                cohort_name=arguments.get("cohort_name", "main"),
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}", "sanitized": True}

        return json.dumps(result, indent=2, default=str)

    def cleanup(self) -> None:
        """Remove the temporary directory."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        self._tools = None


# ---------------------------------------------------------------------------
# 3. System prompt builder
# ---------------------------------------------------------------------------


def build_tool_use_system_prompt(
    context: AgentContext,
    spec: DGPSpec,
) -> str:
    """Build the system prompt for a tool-use LLM agent.

    The prompt has four sections:
    1. **Role** — describes the agent's task and references the 4 tools.
    2. **Context** — varies by :class:`ContextLevel`.  Ground truth is
       **never** included.
    3. **Workflow** — step-by-step guidance for using the tools.
    4. **Output format** — the ``discovery_result`` fenced JSON block.

    Parameters
    ----------
    context:
        Agent context with vignette, hint, or blind information.
    spec:
        DGP spec (used for context text only, never ground truth).

    Returns
    -------
    str
        The complete system prompt.
    """
    # Section 1: Role
    role_section = (
        "You are a clinical research analyst. Your task is to discover "
        "causal hypotheses from a clinical dataset.\n\n"
        "You have access to exactly 4 tools:\n"
        "- **get_schema**: Retrieve the data dictionary and table definitions.\n"
        "- **load_data**: Load a clinical dataset and get a summary.\n"
        "- **query_cohort**: Define a study cohort with inclusion/exclusion criteria.\n"
        "- **execute_analysis**: Run statistical analyses (summary_stats, "
        "descriptive, hypothesis, regression, subgroup, survival, xgboost).\n\n"
        "Use these tools to explore the data, identify exposures, outcomes, "
        "and confounders, build a causal DAG, estimate effects, and "
        "interpret your findings.\n\n"
        "IMPORTANT: You only see aggregated, privacy-sanitized results. "
        "You cannot access raw patient data."
    )

    # Section 2: Context (varies by level, never includes ground truth)
    if context.context_level == ContextLevel.FULL_VIGNETTE:
        context_section = (
            "## Clinical Context\n\n"
            f"{spec.vignette}"
        )
    elif context.context_level == ContextLevel.DOMAIN_HINT:
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

    # Section 3: Workflow guidance
    workflow_section = (
        "## Recommended Workflow\n\n"
        "1. **Load data**: Call `load_data` with path 'data' to load the dataset.\n"
        "2. **Explore schema**: Call `get_schema` to understand table structures "
        "and column definitions.\n"
        "3. **Summary statistics**: Use `execute_analysis` with type "
        "'summary_stats' to profile the variables.\n"
        "4. **Define cohort**: Call `query_cohort` with appropriate "
        "inclusion/exclusion criteria.\n"
        "5. **Build a causal DAG**: Based on your understanding of the "
        "variables, identify the causal structure — exposures, outcomes, "
        "confounders, mediators, and colliders. Determine the adjustment set.\n"
        "6. **Regression analysis**: Use `execute_analysis` with type "
        "'regression' to estimate causal effects with your chosen "
        "adjustment set.\n"
        "7. **Interpret results**: Summarize findings, effect sizes, "
        "confidence intervals, and limitations."
    )

    # Section 4: Output format
    output_section = (
        "## Required Output Format\n\n"
        "After completing your analysis, output your findings in a fenced "
        "block with the tag `discovery_result`. The JSON must match this "
        "schema:\n\n"
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
        "IMPORTANT: You MUST include this fenced block in your final response."
    )

    return "\n\n".join([
        role_section,
        context_section,
        workflow_section,
        output_section,
    ])


# ---------------------------------------------------------------------------
# 4. ToolUseAgent
# ---------------------------------------------------------------------------


class ToolUseAgent(DiscoveryAgent):
    """Discovery agent that drives an :class:`LLMAdapter` through tool-use.

    Implements the :class:`DiscoveryAgent` ABC so it plugs directly into
    :class:`BenchmarkHarness` with no changes to the harness.

    Parameters
    ----------
    adapter:
        The LLM adapter (Anthropic, Google, OpenAI) to use.
    max_turns:
        Maximum tool-use round-trips before stopping.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        max_turns: int = 20,
    ) -> None:
        self._adapter = adapter
        self._max_turns = max_turns
        self._spec: DGPSpec | None = None
        self._last_adapter_result: AdapterResult | None = None

    def prepare(self, spec: DGPSpec) -> None:
        """Store the DGP spec for prompt construction.

        Called by :class:`BenchmarkHarness` before :meth:`run`.
        """
        self._spec = spec

    @property
    def last_adapter_result(self) -> AdapterResult | None:
        """The :class:`AdapterResult` from the most recent run, for logging."""
        return self._last_adapter_result

    def run(
        self,
        tables: dict[str, pd.DataFrame],
        context: AgentContext,
    ) -> DiscoveryResult:
        """Run the tool-use loop and parse the result.

        Parameters
        ----------
        tables:
            DataFrames keyed by table name (from DGPSyntheticGenerator).
        context:
            Agent context (vignette, hint, or blind).

        Returns
        -------
        DiscoveryResult
            Parsed from the LLM's final output.

        Raises
        ------
        RuntimeError
            If :meth:`prepare` was not called first.
        """
        if self._spec is None:
            raise RuntimeError(
                "ToolUseAgent requires prepare(spec) to be called "
                "before run(). Use BenchmarkHarness or call prepare() "
                "manually."
            )

        dispatcher = InProcessToolDispatcher(tables)
        try:
            dispatcher.setup()

            system_prompt = build_tool_use_system_prompt(context, self._spec)
            tool_schemas = get_tool_schemas()

            adapter_result = self._adapter.run_tool_loop(
                system_prompt=system_prompt,
                tools=tool_schemas,
                tool_dispatcher=dispatcher,
                max_turns=self._max_turns,
            )

            self._last_adapter_result = adapter_result
            return parse_agent_output(adapter_result.final_text)

        finally:
            dispatcher.cleanup()
