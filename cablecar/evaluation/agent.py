"""Discovery agent protocol, benchmark harness, and statistical baseline agent.

This module defines:
- :class:`DiscoveryAgent` — ABC that any agent must implement.
- :class:`AgentContext` — information envelope passed to agents.
- :class:`BenchmarkHarness` — orchestrates data generation, agent execution,
  and scoring across scenarios and context levels.
- :class:`StatisticalAgent` — deterministic logistic-regression baseline.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from cablecar.evaluation.benchmarks import BenchmarkScore, DiscoveryBenchmark
from cablecar.evaluation.dgp import (
    CLIFMapping,
    ContextLevel,
    DGPSpec,
    VariableRole,
)
from cablecar.evaluation.discovery_result import AnalysisStep, DiscoveryResult
from cablecar.evaluation.scoring import DiscoveryScorer

# Re-export ContextLevel for convenience
__all__ = [
    "AgentContext",
    "BenchmarkHarness",
    "ContextLevel",
    "DiscoveryAgent",
    "StatisticalAgent",
]


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------


class AgentContext(BaseModel):
    """Information envelope provided to a discovery agent."""

    context_level: ContextLevel
    vignette: str = ""
    domain_hint: str = ""
    schema_info: dict[str, Any] = Field(default_factory=dict)
    ground_truth_available: bool = False


# ---------------------------------------------------------------------------
# Discovery agent ABC
# ---------------------------------------------------------------------------


class DiscoveryAgent(ABC):
    """Abstract base class for hypothesis-discovery agents."""

    @abstractmethod
    def run(
        self,
        tables: dict[str, pd.DataFrame],
        context: AgentContext,
    ) -> DiscoveryResult:
        """Analyze *tables* and return structured findings."""
        ...


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """Orchestrate data generation, agent execution, and scoring."""

    def __init__(
        self,
        agent: DiscoveryAgent,
        scorer: DiscoveryScorer | None = None,
    ) -> None:
        self.agent = agent
        self._benchmark = DiscoveryBenchmark()

    def _build_context(
        self,
        spec: DGPSpec,
        context_level: ContextLevel,
    ) -> AgentContext:
        """Construct an :class:`AgentContext` from a spec and context level."""
        schema_info: dict[str, Any] = {}
        for var_name, mapping in spec.schema_mappings.items():
            if isinstance(mapping, CLIFMapping):
                schema_info[var_name] = {
                    "table": mapping.table,
                    "column": mapping.column,
                    "category_column": mapping.category_column,
                    "category_value": mapping.category_value,
                }

        if context_level == ContextLevel.FULL_VIGNETTE:
            return AgentContext(
                context_level=context_level,
                vignette=spec.vignette,
                domain_hint=spec.domain_hint,
                schema_info=schema_info,
            )
        elif context_level == ContextLevel.DOMAIN_HINT:
            return AgentContext(
                context_level=context_level,
                domain_hint=spec.domain_hint,
                schema_info=schema_info,
            )
        else:  # BLIND
            # Only table names and columns — no hints
            blind_schema: dict[str, Any] = {}
            for var_name, info in schema_info.items():
                blind_schema[var_name] = {
                    "table": info["table"],
                    "column": info["column"],
                }
            return AgentContext(
                context_level=context_level,
                schema_info=blind_schema,
            )

    def run_scenario(
        self,
        spec: DGPSpec,
        context_level: ContextLevel,
    ) -> BenchmarkScore:
        """Generate data, run the agent, and score."""
        from synthetic.dgp_generator import DGPSyntheticGenerator

        generator = DGPSyntheticGenerator(spec)
        tables, _ = generator.generate()
        context = self._build_context(spec, context_level)
        result = self.agent.run(tables, context)
        return self._benchmark.run_scenario(spec, result, context_level)

    def run_suite(
        self,
        specs: list[DGPSpec],
        context_levels: list[ContextLevel],
    ) -> list[BenchmarkScore]:
        """Run all combinations of *specs* x *context_levels*."""
        scores: list[BenchmarkScore] = []
        for spec in specs:
            for ctx in context_levels:
                scores.append(self.run_scenario(spec, ctx))
        return scores

    def run_full_benchmark(
        self,
        specs: list[DGPSpec],
        context_levels: list[ContextLevel],
    ) -> dict[str, Any]:
        """Run suite and return summary dict."""
        scores = self.run_suite(specs, context_levels)
        summary = DiscoveryBenchmark.summary(scores)
        summary["scores"] = scores
        return summary


# ---------------------------------------------------------------------------
# Statistical baseline agent
# ---------------------------------------------------------------------------


def _reconstruct_patient_data(
    tables: dict[str, pd.DataFrame],
    spec: DGPSpec,
) -> pd.DataFrame:
    """Join CLIF tables back into a patient-level DataFrame.

    Same logic as the blue team in ``test_red_blue_e2e.py``.
    """
    hosp = tables["hospitalization"].copy()
    patient_df = hosp[["hospitalization_id", "patient_id"]].copy()

    if "patient" in tables:
        patient_df = patient_df.merge(tables["patient"], on="patient_id", how="left")

    for var_name, mapping in spec.schema_mappings.items():
        if not isinstance(mapping, CLIFMapping):
            continue
        if mapping.table == "hospitalization" and mapping.column in hosp.columns:
            if mapping.column not in patient_df.columns:
                patient_df[var_name] = hosp[mapping.column].values
        elif mapping.table == "patient":
            if mapping.column in patient_df.columns and var_name != mapping.column:
                patient_df[var_name] = patient_df[mapping.column]

    for var_name, mapping in spec.schema_mappings.items():
        if not isinstance(mapping, CLIFMapping):
            continue
        if not mapping.category_column or not mapping.category_value:
            continue
        if mapping.table not in tables:
            patient_df[var_name] = np.nan
            continue
        tbl = tables[mapping.table]
        filtered = tbl[tbl[mapping.category_column] == mapping.category_value]
        agg = mapping.aggregation or "mean"
        if len(filtered) > 0:
            grouped = (
                filtered.groupby("hospitalization_id")[mapping.column]
                .agg(agg)
                .reset_index()
                .rename(columns={mapping.column: var_name})
            )
            patient_df = patient_df.merge(grouped, on="hospitalization_id", how="left")
        else:
            patient_df[var_name] = np.nan

    return patient_df


def _fit_logistic(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Fit logistic regression and return exposure coefficient info."""
    import statsmodels.api as sm

    covariates = covariates or []
    predictors = [exposure] + covariates
    available = [c for c in predictors if c in df.columns]
    sub = df[[outcome] + available].dropna()

    if len(sub) < 50 or sub[outcome].nunique() < 2:
        return {"coef": 0.0, "ci_low": -1.0, "ci_high": 1.0, "p_value": 1.0, "converged": False}

    y = sub[outcome].values.astype(float)
    X = sm.add_constant(sub[available].values.astype(float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        except Exception:
            return {"coef": 0.0, "ci_low": -1.0, "ci_high": 1.0, "p_value": 1.0, "converged": False}

    idx = 1
    coef = float(model.params[idx])
    ci = model.conf_int(alpha=0.05)
    ci_low = float(ci[idx, 0])
    ci_high = float(ci[idx, 1])
    p_val = float(model.pvalues[idx])

    return {"coef": coef, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_val, "converged": True}


class StatisticalAgent(DiscoveryAgent):
    """Deterministic logistic-regression baseline agent.

    Serves as the comparison point for LLM agents in the paper.

    Parameters
    ----------
    quality : str
        One of ``"perfect"``, ``"partial"``, or ``"naive"``.
        Controls how much domain knowledge the agent uses.
    """

    def __init__(self, quality: str = "perfect") -> None:
        if quality not in ("perfect", "partial", "naive"):
            raise ValueError(f"quality must be 'perfect', 'partial', or 'naive', got '{quality}'")
        self.quality = quality

    def run(
        self,
        tables: dict[str, pd.DataFrame],
        context: AgentContext,
    ) -> DiscoveryResult:
        """Analyze tables using logistic regression at the configured quality level.

        Requires that ``context.schema_info`` was built from a :class:`DGPSpec`
        so we can infer the spec. For standalone use, the caller provides
        ``ground_truth_spec`` in context metadata.
        """
        # The harness always passes a spec alongside; but the agent protocol
        # only receives tables + context.  StatisticalAgent needs the spec
        # to know ground truth variable names.  We stash it on the instance
        # before calling run().
        spec = getattr(self, "_current_spec", None)
        if spec is None:
            raise RuntimeError(
                "StatisticalAgent requires _current_spec to be set "
                "before calling run(). Use BenchmarkHarness or set it manually."
            )

        patient_df = _reconstruct_patient_data(tables, spec)
        gt = spec.ground_truth

        if self.quality == "perfect":
            return self._perfect(spec, patient_df)
        elif self.quality == "partial":
            return self._partial(spec, patient_df)
        return self._naive(spec, patient_df)

    # ------------------------------------------------------------------
    # Quality levels
    # ------------------------------------------------------------------

    def _perfect(self, spec: DGPSpec, patient_df: pd.DataFrame) -> DiscoveryResult:
        gt = spec.ground_truth
        covariates = list(gt.correct_adjustment_set)
        fit = _fit_logistic(patient_df, gt.primary_outcome, gt.primary_exposure, covariates)
        confounders = [v.name for v in spec.variables if v.role == VariableRole.CONFOUNDER]
        colliders = [v.name for v in spec.variables if v.role == VariableRole.COLLIDER]
        has_miss = any(v.missingness.proportion > 0 for v in spec.variables)

        return DiscoveryResult(
            identified_exposure=gt.primary_exposure,
            identified_outcome=gt.primary_outcome,
            identified_confounders=confounders,
            identified_colliders=colliders,
            primary_hypothesis=f"{gt.primary_exposure} increases risk of {gt.primary_outcome}",
            secondary_hypotheses=[
                f"Age confounds the {gt.primary_exposure}-{gt.primary_outcome} relationship"
            ],
            proposed_dag_edges=list(gt.expected_dag_edges),
            proposed_adjustment_set=covariates,
            methods_used=[
                "logistic_regression",
                "causal DAG / backdoor criterion",
                "descriptive / EDA",
            ],
            method_justification=(
                "Binary outcome requires logistic regression. "
                "Backdoor criterion used to identify adjustment set."
            ),
            estimated_effect=fit["coef"],
            confidence_interval=(fit["ci_low"], fit["ci_high"]),
            p_value=fit["p_value"],
            effect_size_metric="log_odds_ratio",
            missingness_strategy="complete_case" if has_miss else "",
            missingness_assessment=(
                "Assessed missing data pattern; used complete-case analysis"
                if has_miss else ""
            ),
            interpretation=(
                f"{gt.primary_exposure} is associated with {gt.primary_outcome} "
                f"with an estimated log-OR of {fit['coef']:.3f} after adjusting "
                f"for {', '.join(covariates)}."
            ),
            limitations=[
                "Observational study — cannot prove causation",
                "Complete-case analysis may introduce bias if data are MNAR",
            ],
            analysis_steps=[
                AnalysisStep(step_number=1, description="Exploratory data analysis", tool_used="pandas", result_summary="Described distributions"),
                AnalysisStep(step_number=2, description="Identified causal DAG", tool_used="domain knowledge", result_summary="Built DAG from clinical reasoning"),
                AnalysisStep(step_number=3, description="Adjusted logistic regression", tool_used="statsmodels", result_summary=f"log-OR={fit['coef']:.3f}"),
            ],
        )

    def _partial(self, spec: DGPSpec, patient_df: pd.DataFrame) -> DiscoveryResult:
        gt = spec.ground_truth
        partial_adj = list(gt.correct_adjustment_set)[:1]
        fit = _fit_logistic(patient_df, gt.primary_outcome, gt.primary_exposure, partial_adj)

        return DiscoveryResult(
            identified_exposure=gt.primary_exposure,
            identified_outcome=gt.primary_outcome,
            identified_confounders=partial_adj,
            primary_hypothesis=f"{gt.primary_exposure} affects {gt.primary_outcome}",
            proposed_dag_edges=[
                (gt.primary_exposure, gt.primary_outcome),
            ] + [(c, gt.primary_outcome) for c in partial_adj],
            proposed_adjustment_set=partial_adj,
            methods_used=["logistic_regression"],
            method_justification="Used logistic regression for binary outcome.",
            estimated_effect=fit["coef"],
            confidence_interval=(fit["ci_low"], fit["ci_high"]),
            p_value=fit["p_value"],
            effect_size_metric="log_odds_ratio",
            interpretation=(
                f"{gt.primary_exposure} appears associated with {gt.primary_outcome} "
                f"(log-OR={fit['coef']:.3f})."
            ),
            limitations=["May not have adjusted for all confounders"],
            analysis_steps=[
                AnalysisStep(step_number=1, description="Partially adjusted logistic regression", tool_used="statsmodels", result_summary=f"log-OR={fit['coef']:.3f}"),
            ],
        )

    def _naive(self, spec: DGPSpec, patient_df: pd.DataFrame) -> DiscoveryResult:
        gt = spec.ground_truth
        fit = _fit_logistic(patient_df, gt.primary_outcome, gt.primary_exposure)
        colliders = [v.name for v in spec.variables if v.role == VariableRole.COLLIDER]

        return DiscoveryResult(
            identified_exposure=gt.primary_exposure,
            identified_outcome=gt.primary_outcome,
            identified_confounders=[],
            primary_hypothesis=f"There is an association between {gt.primary_exposure} and {gt.primary_outcome}",
            proposed_dag_edges=[
                (gt.primary_exposure, gt.primary_outcome),
            ],
            proposed_adjustment_set=colliders,
            methods_used=["logistic_regression"],
            estimated_effect=fit["coef"],
            confidence_interval=(fit["ci_low"], fit["ci_high"]),
            p_value=fit["p_value"],
            effect_size_metric="log_odds_ratio",
            interpretation=(
                f"Unadjusted analysis shows log-OR={fit['coef']:.3f} "
                f"for {gt.primary_exposure} on {gt.primary_outcome}."
            ),
            analysis_steps=[
                AnalysisStep(step_number=1, description="Unadjusted logistic regression", tool_used="statsmodels", result_summary=f"log-OR={fit['coef']:.3f}"),
            ],
        )
