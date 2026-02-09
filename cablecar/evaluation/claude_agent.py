"""Claude-based hypothesis discovery agent.

Uses the Anthropic Python SDK with tool-use to analyze clinical datasets.
The agent mirrors the tools available in :class:`DataServerTools` but
operates on in-memory data (no actual MCP server needed for benchmarks).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from cablecar.evaluation.agent import AgentContext, DiscoveryAgent
from cablecar.evaluation.discovery_result import AnalysisStep, DiscoveryResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (mirrors DataServerTools)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "get_schema",
        "description": (
            "Get schema and data dictionary. Returns table names, column "
            "definitions, data types, and relationships. No patient data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "query_data",
        "description": (
            "Query a specific table with optional column selection and "
            "filtering. Returns summary statistics, NOT raw data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table name to query.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to include (all if omitted).",
                },
                "group_by": {
                    "type": "string",
                    "description": "Column to group by for aggregate stats.",
                },
            },
            "required": ["table"],
        },
    },
    {
        "name": "run_regression",
        "description": (
            "Fit a logistic or linear regression. Returns coefficients, "
            "confidence intervals, p-values, and model fit statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "description": "Outcome variable name.",
                },
                "exposure": {
                    "type": "string",
                    "description": "Primary exposure variable.",
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Adjustment covariates.",
                },
                "model_type": {
                    "type": "string",
                    "enum": ["logistic", "linear"],
                    "description": "Model type (default: logistic).",
                },
            },
            "required": ["outcome", "exposure"],
        },
    },
    {
        "name": "submit_result",
        "description": (
            "Submit the final DiscoveryResult. Call this when you have "
            "completed your analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identified_exposure": {"type": "string"},
                "identified_outcome": {"type": "string"},
                "identified_confounders": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "primary_hypothesis": {"type": "string"},
                "proposed_dag_edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "proposed_adjustment_set": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "methods_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "estimated_effect": {"type": "number"},
                "confidence_interval": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "interpretation": {"type": "string"},
            },
            "required": [
                "identified_exposure",
                "identified_outcome",
                "primary_hypothesis",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical research analyst. You have access to a clinical dataset
organized into tables (similar to an electronic health record). Your task is
to discover and test hypotheses about causal relationships in the data.

**Workflow:**
1. Use `get_schema` to understand the available tables and columns.
2. Use `query_data` to explore distributions and relationships.
3. Use `run_regression` to test hypotheses with appropriate adjustments.
4. Use `submit_result` to report your findings.

**Guidelines:**
- Identify the primary exposure and outcome.
- Consider confounders and build a causal DAG.
- Use the backdoor criterion to determine the adjustment set.
- Report effect estimates with confidence intervals.
- Acknowledge limitations of your analysis.

{context_section}
"""


# ---------------------------------------------------------------------------
# Claude Discovery Agent
# ---------------------------------------------------------------------------


class ClaudeDiscoveryAgent(DiscoveryAgent):
    """Concrete agent that calls the Anthropic API with tool use.

    Parameters
    ----------
    model : str
        Anthropic model ID.
    api_key : str | None
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    max_turns : int
        Maximum agentic tool-use turns before forcing a result.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: str | None = None,
        max_turns: int = 15,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        tables: dict[str, pd.DataFrame],
        context: AgentContext,
    ) -> DiscoveryResult:
        """Run the Claude agent loop and return a DiscoveryResult."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeDiscoveryAgent. "
                "Install it with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)
        system_prompt = self._build_system_prompt(context)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Please analyze the clinical dataset and discover causal hypotheses."},
        ]
        analysis_steps: list[AnalysisStep] = []
        final_result: DiscoveryResult | None = None

        for turn in range(self.max_turns):
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                tools=_TOOLS,
                messages=messages,
            )

            # Collect assistant content
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                # No more tool calls — agent is done
                break

            # Process tool calls
            tool_results = []
            for tool_use in tool_uses:
                step_num = len(analysis_steps) + 1
                tool_name = tool_use.name
                tool_input = tool_use.input

                if tool_name == "submit_result":
                    final_result = self._parse_submit(tool_input, analysis_steps)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps({"status": "submitted"}),
                    })
                else:
                    result = self._dispatch_tool(tool_name, tool_input, tables)
                    analysis_steps.append(AnalysisStep(
                        step_number=step_num,
                        description=f"Called {tool_name}",
                        tool_used=tool_name,
                        result_summary=_truncate(json.dumps(result), 200),
                    ))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result, default=str),
                    })

            messages.append({"role": "user", "content": tool_results})

            if final_result is not None:
                break

        if final_result is None:
            # Try to parse from last text response
            final_result = self._fallback_result(messages, analysis_steps)

        return final_result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: AgentContext) -> str:
        if context.context_level == ContextLevel.FULL_VIGNETTE:
            section = f"**Clinical Context:**\n{context.vignette}"
        elif context.context_level == ContextLevel.DOMAIN_HINT:
            section = f"**Hint:** {context.domain_hint}"
        else:
            section = "You have no prior information about this dataset."
        return _SYSTEM_PROMPT.format(context_section=section)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tables: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        if tool_name == "get_schema":
            return self._tool_get_schema(tables)
        elif tool_name == "query_data":
            return self._tool_query_data(tables, tool_input)
        elif tool_name == "run_regression":
            return self._tool_run_regression(tables, tool_input)
        return {"error": f"Unknown tool: {tool_name}"}

    def _tool_get_schema(self, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for name, df in tables.items():
            schema[name] = {
                "rows": len(df),
                "columns": {
                    col: str(df[col].dtype) for col in df.columns
                },
            }
        return {"tables": schema}

    def _tool_query_data(
        self,
        tables: dict[str, pd.DataFrame],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        table_name = params.get("table", "")
        if table_name not in tables:
            return {"error": f"Table '{table_name}' not found. Available: {list(tables.keys())}"}

        df = tables[table_name]
        columns = params.get("columns")
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        result: dict[str, Any] = {"table": table_name, "rows": len(df), "stats": {}}
        for col in df.columns:
            col_data = df[col]
            stat: dict[str, Any] = {
                "dtype": str(col_data.dtype),
                "missing": int(col_data.isna().sum()),
                "unique": int(col_data.nunique()),
            }
            if pd.api.types.is_numeric_dtype(col_data):
                desc = col_data.describe()
                stat.update({
                    "mean": round(float(desc["mean"]), 3),
                    "std": round(float(desc["std"]), 3),
                    "min": round(float(desc["min"]), 3),
                    "max": round(float(desc["max"]), 3),
                })
            elif col_data.dtype == "object":
                stat["top_values"] = {
                    str(k): int(v)
                    for k, v in col_data.value_counts().head(10).items()
                }

            group_by = params.get("group_by")
            if group_by and group_by in tables[params["table"]].columns and pd.api.types.is_numeric_dtype(col_data):
                group_df = tables[params["table"]]
                if group_by in group_df.columns and col in group_df.columns:
                    grouped = group_df.groupby(group_by)[col].describe()
                    stat["grouped"] = grouped.to_dict()

            result["stats"][col] = stat

        return result

    def _tool_run_regression(
        self,
        tables: dict[str, pd.DataFrame],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        import statsmodels.api as sm

        outcome = params.get("outcome", "")
        exposure = params.get("exposure", "")
        covariates = params.get("covariates", [])
        model_type = params.get("model_type", "logistic")

        # Find the table containing these columns
        target_df = None
        for name, df in tables.items():
            if outcome in df.columns and exposure in df.columns:
                target_df = df
                break

        if target_df is None:
            # Try merging hospitalization + patient
            if "hospitalization" in tables and "patient" in tables:
                target_df = tables["hospitalization"].merge(
                    tables["patient"], on="patient_id", how="left"
                )
            else:
                return {"error": f"Could not find tables containing '{outcome}' and '{exposure}'"}

        # Also merge category tables if covariates reference them
        # (simplified: just use what we have)
        predictors = [exposure] + [c for c in covariates if c in target_df.columns]
        cols = [outcome] + predictors
        model_df = target_df[[c for c in cols if c in target_df.columns]].dropna()

        if len(model_df) < 50:
            return {"error": "Too few observations after dropping missing values"}

        y = model_df[outcome].astype(float)
        X = sm.add_constant(model_df[predictors].astype(float))

        try:
            if model_type == "logistic":
                model = sm.Logit(y, X).fit(disp=0, maxiter=100)
            else:
                model = sm.OLS(y, X).fit()

            coefficients = {}
            for name_param in model.params.index:
                ci = model.conf_int()
                coefficients[name_param] = {
                    "coef": round(float(model.params[name_param]), 4),
                    "ci_low": round(float(ci.loc[name_param, 0]), 4),
                    "ci_high": round(float(ci.loc[name_param, 1]), 4),
                    "p_value": round(float(model.pvalues[name_param]), 6),
                }

            return {
                "model_type": model_type,
                "n": len(model_df),
                "coefficients": coefficients,
            }
        except Exception as e:
            return {"error": f"Model fitting failed: {str(e)}"}

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_submit(
        self,
        tool_input: dict[str, Any],
        analysis_steps: list[AnalysisStep],
    ) -> DiscoveryResult:
        """Parse submit_result tool input into a DiscoveryResult."""
        dag_edges = [tuple(e) for e in tool_input.get("proposed_dag_edges", [])]
        ci = tool_input.get("confidence_interval")
        if ci and len(ci) == 2:
            ci = tuple(ci)
        else:
            ci = None

        return DiscoveryResult(
            identified_exposure=tool_input.get("identified_exposure", ""),
            identified_outcome=tool_input.get("identified_outcome", ""),
            identified_confounders=tool_input.get("identified_confounders", []),
            identified_mediators=tool_input.get("identified_mediators", []),
            identified_colliders=tool_input.get("identified_colliders", []),
            primary_hypothesis=tool_input.get("primary_hypothesis", ""),
            secondary_hypotheses=tool_input.get("secondary_hypotheses", []),
            proposed_dag_edges=dag_edges,
            proposed_adjustment_set=tool_input.get("proposed_adjustment_set", []),
            methods_used=tool_input.get("methods_used", []),
            method_justification=tool_input.get("method_justification", ""),
            estimated_effect=tool_input.get("estimated_effect"),
            confidence_interval=ci,
            p_value=tool_input.get("p_value"),
            effect_size_metric=tool_input.get("effect_size_metric", ""),
            missingness_strategy=tool_input.get("missingness_strategy", ""),
            missingness_assessment=tool_input.get("missingness_assessment", ""),
            interpretation=tool_input.get("interpretation", ""),
            limitations=tool_input.get("limitations", []),
            analysis_steps=analysis_steps,
        )

    def _fallback_result(
        self,
        messages: list[dict[str, Any]],
        analysis_steps: list[AnalysisStep],
    ) -> DiscoveryResult:
        """Construct a minimal DiscoveryResult when submit_result was never called."""
        return DiscoveryResult(
            identified_exposure="unknown",
            identified_outcome="unknown",
            primary_hypothesis="Agent did not submit structured results",
            analysis_steps=analysis_steps,
        )


# Import needed for prompt building
from cablecar.evaluation.dgp import ContextLevel  # noqa: E402


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
