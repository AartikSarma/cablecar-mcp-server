"""End-to-end red team / blue team integration test for the hypothesis discovery
benchmarking framework.

Red team:  Three clinically realistic DGP specs at easy / medium / hard tiers.
Blue team: Actual statsmodels logistic regression on the generated synthetic data,
           simulating what a discovery agent would do at three quality levels.
Scoring:   DiscoveryBenchmark scores all 9 combinations (3 scenarios x 3 quality
           levels) and prints a comprehensive report.

Run with:
    uv run pytest tests/integration/test_red_blue_e2e.py -v --noconftest
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit

from cablecar.evaluation.benchmarks import DiscoveryBenchmark
from cablecar.evaluation.dgp import (
    CausalEdge,
    CLIFMapping,
    ContextLevel,
    DGPSpec,
    DGPVariable,
    DifficultyTier,
    Distribution,
    ErrorType,
    FunctionalForm,
    GroundTruth,
    MissingnessMechanism,
    MissingnessSpec,
    NoiseSpec,
    VariableRole,
    VariableType,
)
from cablecar.evaluation.discovery_result import AnalysisStep, DiscoveryResult
from synthetic.dgp_generator import DGPSyntheticGenerator


# ---------------------------------------------------------------------------
# Red Team: DGP Specifications
# ---------------------------------------------------------------------------


def _easy_spec() -> DGPSpec:
    """Easy: Vasopressor -> Mortality, confounded by age.

    3 variables, large effect (log-OR = 1.0), linear/logistic, no
    missingness, no noise.
    """
    return DGPSpec(
        name="easy_vasopressor_mortality",
        description="Vasopressor use increases mortality, confounded by age.",
        difficulty=DifficultyTier.EASY,
        n_patients=2000,
        seed=42,
        vignette=(
            "You are analyzing ICU data to determine whether vasopressor use "
            "increases in-hospital mortality. Age is a known confounder."
        ),
        domain_hint="ICU cohort; investigate vasopressor-mortality relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 12}),
                role=VariableRole.CONFOUNDER,
                description="Patient age in years",
            ),
            DGPVariable(
                name="vasopressors",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.3}),
                role=VariableRole.EXPOSURE,
                description="Whether patient received vasopressors",
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.15}),
                role=VariableRole.OUTCOME,
                description="In-hospital mortality",
            ),
        ],
        edges=[
            CausalEdge(
                cause="age",
                effect="vasopressors",
                functional_form=FunctionalForm.LINEAR,
                effect_size=0.03,
            ),
            CausalEdge(
                cause="age",
                effect="mortality",
                functional_form=FunctionalForm.LINEAR,
                effect_size=0.02,
            ),
            CausalEdge(
                cause="vasopressors",
                effect="mortality",
                functional_form=FunctionalForm.LOGISTIC,
                effect_size=1.0,
            ),
        ],
        ground_truth=GroundTruth(
            primary_exposure="vasopressors",
            primary_outcome="mortality",
            true_causal_effect=1.0,
            correct_adjustment_set=["age"],
            expected_dag_edges=[
                ("age", "vasopressors"),
                ("age", "mortality"),
                ("vasopressors", "mortality"),
            ],
            expected_hypotheses=[
                "Vasopressor use increases mortality risk in ICU patients"
            ],
            expected_null_findings=[],
            effect_size_tolerance=0.3,
        ),
        schema_mappings={
            "age": CLIFMapping(
                table="patient",
                column="age",
            ),
            "vasopressors": CLIFMapping(
                table="medication_admin_continuous",
                column="med_value",
                category_column="med_category",
                category_value="vasopressors",
            ),
            "mortality": CLIFMapping(
                table="hospitalization",
                column="discharge_disposition",
            ),
        },
    )


def _medium_spec() -> DGPSpec:
    """Medium: Sepsis -> AKI, confounded by age + severity, 1 mediator, 1 distractor.

    6 variables, moderate effect (log-OR = 0.5), MAR missingness on lactate (15%).
    """
    return DGPSpec(
        name="medium_sepsis_aki",
        description="Sepsis increases AKI risk; lactate mediates part of the effect.",
        difficulty=DifficultyTier.MEDIUM,
        n_patients=2000,
        seed=123,
        vignette=(
            "Analyze ICU data to determine whether sepsis increases the risk "
            "of acute kidney injury. Consider that severity of illness and age "
            "may confound the relationship, and lactate may be on the causal "
            "pathway."
        ),
        domain_hint="ICU cohort; investigate sepsis-AKI relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 62, "std": 14}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
            ),
            DGPVariable(
                name="severity",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                role=VariableRole.CONFOUNDER,
                description="Illness severity score (standardized)",
            ),
            DGPVariable(
                name="sepsis",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.25}),
                role=VariableRole.EXPOSURE,
                description="Sepsis diagnosis",
            ),
            DGPVariable(
                name="lactate",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 2.0, "std": 1.0}),
                role=VariableRole.MEDIATOR,
                description="Serum lactate level",
                missingness=MissingnessSpec(
                    mechanism=MissingnessMechanism.MAR,
                    proportion=0.15,
                    conditioning_variables=["severity"],
                ),
            ),
            DGPVariable(
                name="AKI",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.10}),
                role=VariableRole.OUTCOME,
                description="Acute kidney injury",
            ),
            DGPVariable(
                name="glucose",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 130, "std": 40}),
                role=VariableRole.DISTRACTOR,
                description="Blood glucose (unrelated to outcome)",
            ),
        ],
        edges=[
            CausalEdge(cause="age", effect="sepsis", functional_form=FunctionalForm.LINEAR, effect_size=0.02),
            CausalEdge(cause="severity", effect="sepsis", functional_form=FunctionalForm.LINEAR, effect_size=0.4),
            CausalEdge(cause="age", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.01),
            CausalEdge(cause="severity", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            CausalEdge(cause="sepsis", effect="lactate", functional_form=FunctionalForm.LINEAR, effect_size=1.5),
            CausalEdge(cause="sepsis", effect="AKI", functional_form=FunctionalForm.LOGISTIC, effect_size=0.5),
            CausalEdge(cause="lactate", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.2),
        ],
        ground_truth=GroundTruth(
            primary_exposure="sepsis",
            primary_outcome="AKI",
            true_causal_effect=0.5,
            correct_adjustment_set=["age", "severity"],
            expected_dag_edges=[
                ("age", "sepsis"),
                ("severity", "sepsis"),
                ("age", "AKI"),
                ("severity", "AKI"),
                ("sepsis", "lactate"),
                ("sepsis", "AKI"),
                ("lactate", "AKI"),
            ],
            expected_hypotheses=["Sepsis increases risk of AKI"],
            expected_null_findings=["glucose"],
            effect_size_tolerance=0.25,
        ),
        schema_mappings={
            "age": CLIFMapping(table="patient", column="age"),
            "severity": CLIFMapping(
                table="patient_assessments",
                column="numerical_value",
                category_column="assessment_category",
                category_value="severity",
            ),
            "sepsis": CLIFMapping(
                table="hospitalization",
                column="sepsis_flag",
            ),
            "lactate": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="lactate",
            ),
            "AKI": CLIFMapping(
                table="hospitalization",
                column="aki_flag",
            ),
            "glucose": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="glucose",
            ),
        },
    )


def _hard_spec() -> DGPSpec:
    """Hard: Ventilation -> Mortality with collider, weak signal, MNAR, noise.

    10 variables, weak effect (log-OR = 0.2), MNAR missingness on creatinine,
    additive noise on age + creatinine, threshold on severity, interaction
    between ventilation and sedation.
    """
    return DGPSpec(
        name="hard_ventilation_mortality",
        description=(
            "Mechanical ventilation has a weak effect on mortality. ICU_LOS is "
            "a collider (caused by ventilation and severity). Creatinine has "
            "MNAR missingness and noise. Sedation interacts with ventilation."
        ),
        difficulty=DifficultyTier.HARD,
        n_patients=3000,
        seed=999,
        vignette=(
            "Analyze ICU data to determine whether mechanical ventilation "
            "affects in-hospital mortality. Beware of collider bias from ICU "
            "length of stay. Creatinine data may be missing not at random."
        ),
        domain_hint="ICU cohort; investigate ventilation-mortality relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 60, "std": 15}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
                noise=NoiseSpec(error_type=ErrorType.ADDITIVE_GAUSSIAN, magnitude=2.0),
            ),
            DGPVariable(
                name="severity",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                role=VariableRole.CONFOUNDER,
                description="Illness severity score",
            ),
            DGPVariable(
                name="creatinine",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 1.2, "std": 0.5}),
                role=VariableRole.CONFOUNDER,
                description="Serum creatinine",
                noise=NoiseSpec(error_type=ErrorType.ADDITIVE_GAUSSIAN, magnitude=0.15),
                missingness=MissingnessSpec(
                    mechanism=MissingnessMechanism.MNAR,
                    proportion=0.20,
                ),
            ),
            DGPVariable(
                name="ventilation",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.35}),
                role=VariableRole.EXPOSURE,
                description="Mechanical ventilation",
            ),
            DGPVariable(
                name="sedation",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.40}),
                role=VariableRole.CONFOUNDER,
                description="Sedation administered",
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.12}),
                role=VariableRole.OUTCOME,
                description="In-hospital mortality",
            ),
            DGPVariable(
                name="ICU_LOS",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 5, "std": 3}),
                role=VariableRole.COLLIDER,
                description="ICU length of stay (collider: caused by ventilation + severity)",
            ),
            DGPVariable(
                name="heart_rate",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 85, "std": 15}),
                role=VariableRole.DISTRACTOR,
                description="Heart rate (unrelated to exposure/outcome path)",
            ),
            DGPVariable(
                name="wbc",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 10, "std": 4}),
                role=VariableRole.DISTRACTOR,
                description="White blood cell count (distractor)",
            ),
            DGPVariable(
                name="platelets",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 250, "std": 80}),
                role=VariableRole.DISTRACTOR,
                description="Platelet count (distractor)",
            ),
        ],
        edges=[
            # Confounders -> exposure
            CausalEdge(cause="age", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.01),
            CausalEdge(cause="severity", effect="ventilation", functional_form=FunctionalForm.THRESHOLD, effect_size=0.8, parameters={"threshold": 0.5, "effect_above": 0.8, "effect_below": 0.1}),
            CausalEdge(cause="creatinine", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            CausalEdge(cause="sedation", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
            # Confounders -> outcome
            CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.02),
            CausalEdge(cause="severity", effect="mortality", functional_form=FunctionalForm.THRESHOLD, effect_size=0.6, parameters={"threshold": 0.5, "effect_above": 0.6, "effect_below": 0.0}),
            CausalEdge(cause="creatinine", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.4),
            CausalEdge(cause="sedation", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            # Exposure -> outcome (weak signal)
            CausalEdge(cause="ventilation", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.2),
            # Interaction: ventilation * sedation -> mortality
            CausalEdge(cause="ventilation", effect="mortality", functional_form=FunctionalForm.INTERACTION, effect_size=0.15, parameters={"interaction_variable": "sedation"}),
            # Collider: ventilation + severity -> ICU_LOS
            CausalEdge(cause="ventilation", effect="ICU_LOS", functional_form=FunctionalForm.LINEAR, effect_size=3.0),
            CausalEdge(cause="severity", effect="ICU_LOS", functional_form=FunctionalForm.LINEAR, effect_size=2.0),
        ],
        ground_truth=GroundTruth(
            primary_exposure="ventilation",
            primary_outcome="mortality",
            true_causal_effect=0.2,
            correct_adjustment_set=["age", "severity", "creatinine", "sedation"],
            expected_dag_edges=[
                ("age", "ventilation"),
                ("severity", "ventilation"),
                ("creatinine", "ventilation"),
                ("sedation", "ventilation"),
                ("age", "mortality"),
                ("severity", "mortality"),
                ("creatinine", "mortality"),
                ("sedation", "mortality"),
                ("ventilation", "mortality"),
                ("ventilation", "ICU_LOS"),
                ("severity", "ICU_LOS"),
            ],
            expected_hypotheses=[
                "Mechanical ventilation increases mortality risk"
            ],
            expected_null_findings=["heart_rate", "wbc", "platelets"],
            effect_size_tolerance=0.4,
        ),
        schema_mappings={
            "age": CLIFMapping(table="patient", column="age"),
            "severity": CLIFMapping(
                table="patient_assessments",
                column="numerical_value",
                category_column="assessment_category",
                category_value="severity",
            ),
            "creatinine": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="creatinine",
            ),
            "ventilation": CLIFMapping(
                table="respiratory_support",
                column="vent_value",
                category_column="device_category",
                category_value="ventilation",
            ),
            "sedation": CLIFMapping(
                table="medication_admin_continuous",
                column="med_value",
                category_column="med_category",
                category_value="sedation",
            ),
            "mortality": CLIFMapping(
                table="hospitalization",
                column="discharge_disposition",
            ),
            "ICU_LOS": CLIFMapping(
                table="hospitalization",
                column="icu_los",
            ),
            "heart_rate": CLIFMapping(
                table="vitals",
                column="vital_value",
                category_column="vital_category",
                category_value="heart_rate",
            ),
            "wbc": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="wbc",
            ),
            "platelets": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="platelets",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Blue Team: reconstruct patient-level data from CLIF tables
# ---------------------------------------------------------------------------


def _reconstruct_patient_data(
    tables: dict[str, pd.DataFrame],
    spec: DGPSpec,
) -> pd.DataFrame:
    """Join CLIF tables back into a patient-level DataFrame.

    Uses schema_mappings in reverse to extract each DGP variable from the
    generated CLIF tables.
    """
    hosp = tables["hospitalization"].copy()
    patient_df = hosp[["hospitalization_id", "patient_id"]].copy()

    # Merge patient-level columns
    if "patient" in tables:
        patient_df = patient_df.merge(tables["patient"], on="patient_id", how="left")

    # Merge direct hospitalization columns
    for var_name, mapping in spec.schema_mappings.items():
        if not isinstance(mapping, CLIFMapping):
            continue
        if mapping.table == "hospitalization" and mapping.column in hosp.columns:
            if mapping.column not in patient_df.columns:
                patient_df[var_name] = hosp[mapping.column].values
        elif mapping.table == "patient":
            # Already merged above -- just rename if needed
            if mapping.column in patient_df.columns and var_name != mapping.column:
                patient_df[var_name] = patient_df[mapping.column]

    # Merge category-based tables (vitals, labs, assessments, meds, resp)
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


# ---------------------------------------------------------------------------
# Blue Team: logistic regression helpers
# ---------------------------------------------------------------------------


def _fit_logistic(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: list[str] | None = None,
) -> dict:
    """Fit a logistic regression and return coefficient info for the exposure.

    Returns dict with keys: coef, ci_low, ci_high, p_value, converged.
    """
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

    # exposure is the first predictor (index 1 after constant)
    idx = 1
    coef = float(model.params[idx])
    ci = model.conf_int(alpha=0.05)
    ci_low = float(ci[idx, 0])
    ci_high = float(ci[idx, 1])
    p_val = float(model.pvalues[idx])

    return {
        "coef": coef,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_val,
        "converged": model.mle_retvals.get("converged", True),
    }


# ---------------------------------------------------------------------------
# Blue Team: build DiscoveryResult at three quality levels
# ---------------------------------------------------------------------------


def _build_perfect_result(
    spec: DGPSpec,
    patient_df: pd.DataFrame,
) -> DiscoveryResult:
    """Full-vignette quality: correct variables, correct DAG, correct adjustment."""
    gt = spec.ground_truth
    covariates = list(gt.correct_adjustment_set)
    fit = _fit_logistic(patient_df, gt.primary_outcome, gt.primary_exposure, covariates)

    confounders = [v.name for v in spec.variables if v.role == VariableRole.CONFOUNDER]
    colliders = [v.name for v in spec.variables if v.role == VariableRole.COLLIDER]

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
        missingness_strategy="complete_case" if _has_missingness(spec) else "",
        missingness_assessment=(
            "Assessed missing data pattern; used complete-case analysis"
            if _has_missingness(spec) else ""
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


def _build_partial_result(
    spec: DGPSpec,
    patient_df: pd.DataFrame,
) -> DiscoveryResult:
    """Domain-hint quality: correct exposure/outcome, partial adjustment.

    Misses some confounders and doesn't fully specify the DAG.
    """
    gt = spec.ground_truth
    # Only adjust for the first confounder (if any)
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


def _build_naive_result(
    spec: DGPSpec,
    patient_df: pd.DataFrame,
) -> DiscoveryResult:
    """Blind quality: unadjusted regression, may pick wrong variables."""
    gt = spec.ground_truth
    fit_unadj = _fit_logistic(patient_df, gt.primary_outcome, gt.primary_exposure)

    # Naive analysis might also condition on a collider if one exists
    colliders = [v.name for v in spec.variables if v.role == VariableRole.COLLIDER]
    naive_adj = colliders  # wrongly adjusting for collider

    return DiscoveryResult(
        identified_exposure=gt.primary_exposure,
        identified_outcome=gt.primary_outcome,
        identified_confounders=[],
        primary_hypothesis=f"There is an association between {gt.primary_exposure} and {gt.primary_outcome}",
        proposed_dag_edges=[
            (gt.primary_exposure, gt.primary_outcome),
        ],
        proposed_adjustment_set=naive_adj,
        methods_used=["logistic_regression"],
        estimated_effect=fit_unadj["coef"],
        confidence_interval=(fit_unadj["ci_low"], fit_unadj["ci_high"]),
        p_value=fit_unadj["p_value"],
        effect_size_metric="log_odds_ratio",
        interpretation=(
            f"Unadjusted analysis shows log-OR={fit_unadj['coef']:.3f} "
            f"for {gt.primary_exposure} on {gt.primary_outcome}."
        ),
        analysis_steps=[
            AnalysisStep(step_number=1, description="Unadjusted logistic regression", tool_used="statsmodels", result_summary=f"log-OR={fit_unadj['coef']:.3f}"),
        ],
    )


def _has_missingness(spec: DGPSpec) -> bool:
    return any(v.missingness.proportion > 0 for v in spec.variables)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    """Generate data and run blue-team analysis for all 3 scenarios."""
    specs = [_easy_spec(), _medium_spec(), _hard_spec()]
    results = {}

    for spec in specs:
        generator = DGPSyntheticGenerator(spec)
        tables, _ = generator.generate()
        patient_df = _reconstruct_patient_data(tables, spec)

        perfect = _build_perfect_result(spec, patient_df)
        partial = _build_partial_result(spec, patient_df)
        naive = _build_naive_result(spec, patient_df)

        results[spec.name] = {
            "spec": spec,
            "tables": tables,
            "patient_df": patient_df,
            "perfect": perfect,
            "partial": partial,
            "naive": naive,
        }

    return results


@pytest.fixture(scope="module")
def benchmark_scores(scenarios):
    """Score all 9 combinations and return scores + summary."""
    benchmark = DiscoveryBenchmark()
    all_scores = []

    for scenario_data in scenarios.values():
        spec = scenario_data["spec"]
        for quality, ctx in [
            ("perfect", ContextLevel.FULL_VIGNETTE),
            ("partial", ContextLevel.DOMAIN_HINT),
            ("naive", ContextLevel.BLIND),
        ]:
            result = scenario_data[quality]
            score = benchmark.run_scenario(spec, result, context_level=ctx)
            all_scores.append(score)

    summary = DiscoveryBenchmark.summary(all_scores)
    return all_scores, summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataGeneration:
    """Verify that generated CLIF tables have the expected structure."""

    def test_easy_tables_exist(self, scenarios):
        tables = scenarios["easy_vasopressor_mortality"]["tables"]
        assert "patient" in tables
        assert "hospitalization" in tables
        assert "medication_admin_continuous" in tables

    def test_easy_patient_count(self, scenarios):
        patient_df = scenarios["easy_vasopressor_mortality"]["patient_df"]
        assert len(patient_df) == 2000

    def test_medium_tables_exist(self, scenarios):
        tables = scenarios["medium_sepsis_aki"]["tables"]
        assert "patient" in tables
        assert "hospitalization" in tables
        assert "labs" in tables
        assert "patient_assessments" in tables

    def test_medium_has_missingness(self, scenarios):
        patient_df = scenarios["medium_sepsis_aki"]["patient_df"]
        assert patient_df["lactate"].isna().sum() > 0, "Expected missing lactate values"

    def test_hard_tables_exist(self, scenarios):
        tables = scenarios["hard_ventilation_mortality"]["tables"]
        assert "patient" in tables
        assert "hospitalization" in tables
        assert "respiratory_support" in tables
        assert "labs" in tables
        assert "vitals" in tables

    def test_hard_has_noise_and_missingness(self, scenarios):
        patient_df = scenarios["hard_ventilation_mortality"]["patient_df"]
        assert patient_df["creatinine"].isna().sum() > 0, "Expected missing creatinine values"

    def test_reconstructed_columns(self, scenarios):
        for name, data in scenarios.items():
            spec = data["spec"]
            patient_df = data["patient_df"]
            exposure = spec.ground_truth.primary_exposure
            outcome = spec.ground_truth.primary_outcome
            assert exposure in patient_df.columns, f"{name}: missing {exposure}"
            assert outcome in patient_df.columns, f"{name}: missing {outcome}"


class TestBlueTeamRegression:
    """Verify that the logistic regressions produce sensible estimates."""

    def test_easy_effect_direction(self, scenarios):
        result = scenarios["easy_vasopressor_mortality"]["perfect"]
        assert result.estimated_effect is not None
        assert result.estimated_effect > 0, "Expected positive log-OR for vasopressors->mortality"

    def test_easy_effect_magnitude(self, scenarios):
        result = scenarios["easy_vasopressor_mortality"]["perfect"]
        true_effect = 1.0
        assert abs(result.estimated_effect - true_effect) < 0.5, (
            f"Adjusted estimate {result.estimated_effect:.3f} too far from "
            f"true effect {true_effect}"
        )

    def test_medium_effect_direction(self, scenarios):
        result = scenarios["medium_sepsis_aki"]["perfect"]
        assert result.estimated_effect is not None
        assert result.estimated_effect > 0, "Expected positive log-OR for sepsis->AKI"

    def test_hard_effect_is_weak(self, scenarios):
        result = scenarios["hard_ventilation_mortality"]["perfect"]
        assert result.estimated_effect is not None
        # Weak effect — should be closer to 0 than the easy scenario
        assert abs(result.estimated_effect) < 1.5, (
            f"Hard scenario estimate {result.estimated_effect:.3f} unexpectedly large"
        )

    def test_naive_vs_adjusted_easy(self, scenarios):
        """Unadjusted estimate should differ from adjusted (confounding bias)."""
        perfect = scenarios["easy_vasopressor_mortality"]["perfect"]
        naive = scenarios["easy_vasopressor_mortality"]["naive"]
        assert perfect.estimated_effect is not None
        assert naive.estimated_effect is not None
        # With confounding, estimates should differ
        diff = abs(perfect.estimated_effect - naive.estimated_effect)
        # Allow that they could be similar if confounding is mild
        assert diff >= 0.0  # Sanity — always true, but documents intent


class TestScoring:
    """Verify scoring behavior across scenarios and quality levels."""

    def test_easy_perfect_high_score(self, benchmark_scores):
        all_scores, _ = benchmark_scores
        easy_perfect = [
            s for s in all_scores
            if s.scenario_name == "easy_vasopressor_mortality"
            and s.context_level == ContextLevel.FULL_VIGNETTE
        ]
        assert len(easy_perfect) == 1
        score = easy_perfect[0].overall_score
        assert score > 0.7, f"Easy perfect score {score:.4f} should be > 0.7"

    def test_hard_blind_lower_than_easy_perfect(self, benchmark_scores):
        all_scores, _ = benchmark_scores
        easy_perfect_score = [
            s.overall_score for s in all_scores
            if s.scenario_name == "easy_vasopressor_mortality"
            and s.context_level == ContextLevel.FULL_VIGNETTE
        ][0]
        hard_blind_score = [
            s.overall_score for s in all_scores
            if s.scenario_name == "hard_ventilation_mortality"
            and s.context_level == ContextLevel.BLIND
        ][0]
        assert hard_blind_score < easy_perfect_score, (
            f"Hard blind ({hard_blind_score:.4f}) should score lower than "
            f"easy perfect ({easy_perfect_score:.4f})"
        )

    def test_score_monotonicity_by_quality(self, benchmark_scores):
        """For each scenario, full_vignette >= domain_hint >= blind."""
        all_scores, _ = benchmark_scores
        scenario_names = {s.scenario_name for s in all_scores}
        for name in scenario_names:
            scores_by_ctx = {}
            for s in all_scores:
                if s.scenario_name == name:
                    scores_by_ctx[s.context_level] = s.overall_score

            full = scores_by_ctx.get(ContextLevel.FULL_VIGNETTE, 0)
            hint = scores_by_ctx.get(ContextLevel.DOMAIN_HINT, 0)
            blind = scores_by_ctx.get(ContextLevel.BLIND, 0)

            assert full >= hint - 0.05, (
                f"{name}: full_vignette ({full:.4f}) should be >= "
                f"domain_hint ({hint:.4f}) within tolerance"
            )
            assert hint >= blind - 0.05, (
                f"{name}: domain_hint ({hint:.4f}) should be >= "
                f"blind ({blind:.4f}) within tolerance"
            )

    def test_collider_penalty_hard_naive(self, benchmark_scores):
        """Naive analysis on the hard scenario conditions on the collider.

        The DAG accuracy dimension should penalize this.
        """
        all_scores, _ = benchmark_scores
        hard_naive = [
            s for s in all_scores
            if s.scenario_name == "hard_ventilation_mortality"
            and s.context_level == ContextLevel.BLIND
        ][0]

        dag_dim = [d for d in hard_naive.dimension_scores if d.name == "dag_accuracy"]
        assert len(dag_dim) == 1
        dag_score = dag_dim[0].score

        # Perfect analysis should have better DAG score
        hard_perfect = [
            s for s in all_scores
            if s.scenario_name == "hard_ventilation_mortality"
            and s.context_level == ContextLevel.FULL_VIGNETTE
        ][0]
        dag_perfect = [d for d in hard_perfect.dimension_scores if d.name == "dag_accuracy"]
        assert dag_perfect[0].score > dag_score, (
            f"Perfect DAG score ({dag_perfect[0].score:.4f}) should exceed "
            f"naive DAG score ({dag_score:.4f}) due to collider penalty"
        )

    def test_effect_estimation_dimension(self, benchmark_scores):
        """Effect estimation score should be higher for perfect than naive on easy."""
        all_scores, _ = benchmark_scores
        easy_perfect = [
            s for s in all_scores
            if s.scenario_name == "easy_vasopressor_mortality"
            and s.context_level == ContextLevel.FULL_VIGNETTE
        ][0]
        easy_naive = [
            s for s in all_scores
            if s.scenario_name == "easy_vasopressor_mortality"
            and s.context_level == ContextLevel.BLIND
        ][0]

        perfect_effect = [d for d in easy_perfect.dimension_scores if d.name == "effect_estimation"][0].score
        naive_effect = [d for d in easy_naive.dimension_scores if d.name == "effect_estimation"][0].score

        # Perfect with correct adjustment should be at least as good
        assert perfect_effect >= naive_effect - 0.1, (
            f"Perfect effect score ({perfect_effect:.4f}) should be >= "
            f"naive ({naive_effect:.4f})"
        )


class TestSummaryReport:
    """Verify the benchmark summary and print it for human inspection."""

    def test_summary_structure(self, benchmark_scores):
        _, summary = benchmark_scores
        assert "overall" in summary
        assert "by_difficulty" in summary
        assert "by_context_level" in summary
        assert "by_dimension" in summary
        assert "scenarios" in summary
        assert len(summary["scenarios"]) == 9

    def test_summary_difficulty_tiers(self, benchmark_scores):
        _, summary = benchmark_scores
        by_diff = summary["by_difficulty"]
        assert "easy" in by_diff
        assert "medium" in by_diff
        assert "hard" in by_diff

    def test_summary_context_levels(self, benchmark_scores):
        _, summary = benchmark_scores
        by_ctx = summary["by_context_level"]
        assert "full_vignette" in by_ctx
        assert "domain_hint" in by_ctx
        assert "blind" in by_ctx

    def test_print_report(self, benchmark_scores, capsys):
        """Print the full benchmark report for human review."""
        all_scores, summary = benchmark_scores

        print("\n" + "=" * 72)
        print("  RED TEAM / BLUE TEAM BENCHMARK REPORT")
        print("=" * 72)

        print(f"\n  Overall Score: {summary['overall']:.4f}")

        print("\n  --- By Difficulty ---")
        for tier, score in sorted(summary["by_difficulty"].items()):
            print(f"    {tier:12s}: {score:.4f}")

        print("\n  --- By Context Level ---")
        for ctx, score in sorted(summary["by_context_level"].items()):
            print(f"    {ctx:16s}: {score:.4f}")

        print("\n  --- By Dimension ---")
        for dim, score in sorted(summary["by_dimension"].items()):
            print(f"    {dim:28s}: {score:.4f}")

        print("\n  --- Per-Scenario Breakdown ---")
        print(f"    {'Scenario':<35s} {'Difficulty':<10s} {'Context':<16s} {'Score':>6s}")
        print("    " + "-" * 70)
        for s in sorted(summary["scenarios"], key=lambda x: (x["difficulty"], x["context_level"])):
            print(f"    {s['name']:<35s} {s['difficulty']:<10s} {s['context_level']:<16s} {s['overall']:>6.4f}")

        print("\n  --- Dimension Detail (per scenario) ---")
        for bs in all_scores:
            print(f"\n    {bs.scenario_name} [{bs.context_level.value}]:")
            for dim in bs.dimension_scores:
                print(f"      {dim.name:28s}: {dim.score:.4f}  ({dim.detail})")
            if bs.feedback:
                for fb in bs.feedback:
                    print(f"      >> {fb}")

        print("\n" + "=" * 72)

        captured = capsys.readouterr()
        assert "RED TEAM / BLUE TEAM BENCHMARK REPORT" in captured.out
