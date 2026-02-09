"""Discovery result model for hypothesis discovery benchmarking.

Defines the structured output that a hypothesis discovery agent must produce
after analyzing a dataset.  The scorer uses this model to compare the agent's
findings against the DGP ground truth.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisStep(BaseModel):
    """A single step in the discovery agent's analysis trace."""

    step_number: int
    description: str
    tool_used: str = ""
    result_summary: str = ""


class DiscoveryResult(BaseModel):
    """Structured output from a hypothesis discovery agent.

    The discovery agent must populate this model with its findings after
    analyzing a dataset.  Fields cover the full pipeline from variable
    identification through interpretation.
    """

    # ------------------------------------------------------------------
    # Variable identification
    # ------------------------------------------------------------------

    identified_exposure: str = Field(
        ..., description="The variable the agent identifies as the primary exposure."
    )
    identified_outcome: str = Field(
        ..., description="The variable the agent identifies as the primary outcome."
    )
    identified_confounders: list[str] = Field(
        default_factory=list,
        description="Variables identified as confounders.",
    )
    identified_mediators: list[str] = Field(
        default_factory=list,
        description="Variables identified as mediators.",
    )
    identified_colliders: list[str] = Field(
        default_factory=list,
        description="Variables identified as colliders.",
    )

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

    primary_hypothesis: str = Field(
        ..., description="The primary hypothesis tested (text)."
    )
    secondary_hypotheses: list[str] = Field(
        default_factory=list,
        description="Additional hypotheses considered.",
    )

    # ------------------------------------------------------------------
    # DAG
    # ------------------------------------------------------------------

    proposed_dag_edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Proposed causal edges as (cause, effect) tuples.",
    )
    proposed_adjustment_set: list[str] = Field(
        default_factory=list,
        description="Variables the agent proposes to adjust for.",
    )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    methods_used: list[str] = Field(
        default_factory=list,
        description="Statistical methods used (e.g. 'logistic_regression', "
        "'kaplan_meier', 'propensity_score').",
    )
    method_justification: str = Field(
        default="",
        description="Why the agent chose these methods.",
    )

    # ------------------------------------------------------------------
    # Effect estimation
    # ------------------------------------------------------------------

    estimated_effect: float | None = Field(
        default=None,
        description="Point estimate of the causal effect.",
    )
    confidence_interval: tuple[float, float] | None = Field(
        default=None,
        description="95% confidence interval as (lower, upper).",
    )
    p_value: float | None = Field(
        default=None,
        description="P-value for the primary test.",
    )
    effect_size_metric: str = Field(
        default="",
        description="What the effect estimate represents "
        "(e.g. 'odds_ratio', 'risk_difference', 'hazard_ratio').",
    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    missingness_strategy: str = Field(
        default="",
        description="Strategy used to handle missing data "
        "(e.g. 'complete_case', 'multiple_imputation').",
    )
    missingness_assessment: str = Field(
        default="",
        description="Agent's assessment of the missing data pattern.",
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    interpretation: str = Field(
        default="",
        description="Agent's interpretation of the results.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations acknowledged by the agent.",
    )

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    analysis_steps: list[AnalysisStep] = Field(
        default_factory=list,
        description="Ordered trace of the analysis steps taken.",
    )
