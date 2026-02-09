"""Tests for the hypothesis discovery scoring engine."""

import pytest

from cablecar.evaluation.dgp import (
    CausalEdge,
    CLIFMapping,
    ContextLevel,
    DGPSpec,
    DGPVariable,
    DifficultyTier,
    Distribution,
    FunctionalForm,
    GroundTruth,
    MissingnessSpec,
    MissingnessMechanism,
    VariableRole,
    VariableType,
)
from cablecar.evaluation.discovery_result import AnalysisStep, DiscoveryResult
from cablecar.evaluation.scoring import (
    DIMENSION_WEIGHTS,
    DimensionScore,
    DiscoveryScorer,
    ScoredResult,
)
from cablecar.evaluation.benchmarks import BenchmarkScore, DiscoveryBenchmark


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec() -> DGPSpec:
    return DGPSpec(
        name="scoring_test",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                role=VariableRole.CONFOUNDER,
            ),
            DGPVariable(
                name="severity",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 8, "std": 4}),
                role=VariableRole.CONFOUNDER,
            ),
            DGPVariable(
                name="treatment",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                role=VariableRole.EXPOSURE,
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                role=VariableRole.OUTCOME,
            ),
            DGPVariable(
                name="noise_var",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                role=VariableRole.DISTRACTOR,
            ),
        ],
        edges=[
            CausalEdge(cause="age", effect="treatment", functional_form=FunctionalForm.LOGISTIC, effect_size=0.03),
            CausalEdge(cause="severity", effect="treatment", functional_form=FunctionalForm.LOGISTIC, effect_size=0.1),
            CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.02),
            CausalEdge(cause="severity", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.15),
            CausalEdge(cause="treatment", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.8),
        ],
        ground_truth=GroundTruth(
            primary_exposure="treatment",
            primary_outcome="mortality",
            true_causal_effect=0.8,
            correct_adjustment_set=["age", "severity"],
            expected_dag_edges=[
                ("age", "treatment"), ("severity", "treatment"),
                ("age", "mortality"), ("severity", "mortality"),
                ("treatment", "mortality"),
            ],
            expected_hypotheses=["Treatment increases mortality"],
            expected_null_findings=["noise_var"],
            effect_size_tolerance=0.2,
        ),
        difficulty=DifficultyTier.MEDIUM,
        n_patients=500,
        seed=42,
        schema_mappings={
            "age": CLIFMapping(table="hospitalization", column="age_at_admission"),
            "treatment": CLIFMapping(table="medication_admin_continuous", column="med_dose", category_column="med_category", category_value="norepinephrine"),
            "mortality": CLIFMapping(table="hospitalization", column="hospital_mortality"),
        },
    )


def _make_perfect_result() -> DiscoveryResult:
    """A discovery result that should score highly."""
    return DiscoveryResult(
        identified_exposure="treatment",
        identified_outcome="mortality",
        identified_confounders=["age", "severity"],
        primary_hypothesis="Treatment increases mortality risk after adjusting for age and severity",
        secondary_hypotheses=["Age is independently associated with mortality"],
        proposed_dag_edges=[
            ("age", "treatment"), ("severity", "treatment"),
            ("age", "mortality"), ("severity", "mortality"),
            ("treatment", "mortality"),
        ],
        proposed_adjustment_set=["age", "severity"],
        methods_used=["descriptive", "dag", "logistic_regression"],
        method_justification="Logistic regression chosen because outcome is binary; DAG used to identify confounders",
        estimated_effect=0.75,
        confidence_interval=(0.5, 1.0),
        p_value=0.001,
        effect_size_metric="log_odds_ratio",
        interpretation="After adjusting for age and severity, treatment was associated with increased mortality (log-OR 0.75, 95% CI 0.5-1.0)",
        limitations=["Observational study cannot prove causation", "Possible unmeasured confounding"],
        analysis_steps=[
            AnalysisStep(step_number=1, description="Profiled data", tool_used="load_data", result_summary="500 patients"),
            AnalysisStep(step_number=2, description="Built DAG", tool_used="causal_dag", result_summary="5 edges"),
        ],
    )


def _make_poor_result() -> DiscoveryResult:
    """A discovery result that should score poorly."""
    return DiscoveryResult(
        identified_exposure="noise_var",
        identified_outcome="age",
        primary_hypothesis="Noise variable affects age",
        methods_used=[],
    )


@pytest.fixture
def spec() -> DGPSpec:
    return _make_spec()


@pytest.fixture
def perfect_result() -> DiscoveryResult:
    return _make_perfect_result()


@pytest.fixture
def poor_result() -> DiscoveryResult:
    return _make_poor_result()


@pytest.fixture
def scorer() -> DiscoveryScorer:
    return DiscoveryScorer()


# ---------------------------------------------------------------------------
# Dimension weights
# ---------------------------------------------------------------------------


class TestDimensionWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_eight_dimensions(self):
        assert len(DIMENSION_WEIGHTS) == 8


# ---------------------------------------------------------------------------
# Scoring: perfect result
# ---------------------------------------------------------------------------


class TestPerfectResult:
    def test_overall_score_is_high(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        assert scored.overall_score > 0.7

    def test_variable_identification_high(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        var_dim = next(d for d in scored.dimension_scores if d.name == "variable_identification")
        assert var_dim.score >= 0.9

    def test_dag_accuracy_high(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        dag_dim = next(d for d in scored.dimension_scores if d.name == "dag_accuracy")
        assert dag_dim.score >= 0.8

    def test_effect_estimation_high(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        eff_dim = next(d for d in scored.dimension_scores if d.name == "effect_estimation")
        # 0.75 is within tolerance of 0.2 from 0.8
        assert eff_dim.score >= 0.8

    def test_interpretation_score(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        interp_dim = next(d for d in scored.dimension_scores if d.name == "interpretation_quality")
        assert interp_dim.score >= 0.7


# ---------------------------------------------------------------------------
# Scoring: poor result
# ---------------------------------------------------------------------------


class TestPoorResult:
    def test_overall_score_is_low(self, scorer, spec, poor_result):
        scored = scorer.score(spec, poor_result)
        assert scored.overall_score < 0.3

    def test_variable_identification_low(self, scorer, spec, poor_result):
        scored = scorer.score(spec, poor_result)
        var_dim = next(d for d in scored.dimension_scores if d.name == "variable_identification")
        assert var_dim.score < 0.3

    def test_has_feedback(self, scorer, spec, poor_result):
        scored = scorer.score(spec, poor_result)
        assert len(scored.feedback) > 0


# ---------------------------------------------------------------------------
# Effect estimation edge cases
# ---------------------------------------------------------------------------


class TestEffectEstimation:
    def test_wrong_direction_capped(self, scorer, spec):
        result = _make_perfect_result()
        result.estimated_effect = -0.5  # Wrong direction
        result.confidence_interval = None
        scored = scorer.score(spec, result)
        eff_dim = next(d for d in scored.dimension_scores if d.name == "effect_estimation")
        assert eff_dim.score <= 0.1

    def test_wrong_direction_ci_bonus(self, scorer, spec):
        result = _make_perfect_result()
        result.estimated_effect = -0.1  # Wrong direction
        result.confidence_interval = (-0.5, 1.5)  # But CI contains true value
        scored = scorer.score(spec, result)
        eff_dim = next(d for d in scored.dimension_scores if d.name == "effect_estimation")
        assert eff_dim.score == pytest.approx(0.2)

    def test_exact_match(self, scorer, spec):
        result = _make_perfect_result()
        result.estimated_effect = 0.8
        scored = scorer.score(spec, result)
        eff_dim = next(d for d in scored.dimension_scores if d.name == "effect_estimation")
        assert eff_dim.score >= 0.9

    def test_no_estimate(self, scorer, spec):
        result = _make_perfect_result()
        result.estimated_effect = None
        scored = scorer.score(spec, result)
        eff_dim = next(d for d in scored.dimension_scores if d.name == "effect_estimation")
        assert eff_dim.score == 0.0


# ---------------------------------------------------------------------------
# Confounder handling with colliders
# ---------------------------------------------------------------------------


class TestConfounderHandling:
    def test_correct_adjustment_scores_high(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        conf_dim = next(d for d in scored.dimension_scores if d.name == "confounder_handling")
        assert conf_dim.score >= 0.9

    def test_missing_confounder_lowers_score(self, scorer, spec):
        result = _make_perfect_result()
        result.proposed_adjustment_set = ["age"]  # Missing severity
        result.identified_confounders = ["age"]
        scored = scorer.score(spec, result)
        conf_dim = next(d for d in scored.dimension_scores if d.name == "confounder_handling")
        assert conf_dim.score < 0.8

    def test_conditioning_on_collider_penalized(self):
        spec = DGPSpec(
            name="collider_test",
            variables=[
                DGPVariable(name="x", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.5}), role=VariableRole.EXPOSURE),
                DGPVariable(name="y", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.3}), role=VariableRole.OUTCOME),
                DGPVariable(name="c", variable_type=VariableType.CONTINUOUS, distribution=Distribution(family="normal", params={"mean": 0, "std": 1}), role=VariableRole.COLLIDER),
            ],
            edges=[
                CausalEdge(cause="x", effect="y", functional_form=FunctionalForm.LOGISTIC, effect_size=0.5),
                CausalEdge(cause="x", effect="c", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
                CausalEdge(cause="y", effect="c", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            ],
            ground_truth=GroundTruth(
                primary_exposure="x", primary_outcome="y",
                true_causal_effect=0.5,
                correct_adjustment_set=[],
                expected_dag_edges=[("x", "y"), ("x", "c"), ("y", "c")],
                effect_size_tolerance=0.2,
            ),
            n_patients=100, seed=42, schema_mappings={},
        )

        result = DiscoveryResult(
            identified_exposure="x",
            identified_outcome="y",
            identified_confounders=["c"],
            primary_hypothesis="X affects Y",
            proposed_dag_edges=[("x", "y"), ("x", "c"), ("y", "c")],
            proposed_adjustment_set=["c"],  # Bad! Conditioning on collider
            methods_used=["regression"],
            estimated_effect=0.5,
        )

        scorer = DiscoveryScorer()
        scored = scorer.score(spec, result)
        dag_dim = next(d for d in scored.dimension_scores if d.name == "dag_accuracy")
        # Collider penalty should reduce score
        assert dag_dim.score < 1.0


# ---------------------------------------------------------------------------
# Missingness scoring
# ---------------------------------------------------------------------------


class TestMissingness:
    def test_no_missingness_scores_full(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        miss_dim = next(d for d in scored.dimension_scores if d.name == "missingness_handling")
        assert miss_dim.score == 1.0

    def test_missingness_with_strategy(self, scorer):
        spec = DGPSpec(
            name="miss_test",
            variables=[
                DGPVariable(name="x", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.5}), role=VariableRole.EXPOSURE),
                DGPVariable(
                    name="y", variable_type=VariableType.CONTINUOUS,
                    distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                    role=VariableRole.OUTCOME,
                    missingness=MissingnessSpec(mechanism=MissingnessMechanism.MCAR, proportion=0.2),
                ),
            ],
            edges=[CausalEdge(cause="x", effect="y", functional_form=FunctionalForm.LINEAR, effect_size=0.5)],
            ground_truth=GroundTruth(
                primary_exposure="x", primary_outcome="y",
                true_causal_effect=0.5, expected_dag_edges=[("x", "y")],
                effect_size_tolerance=0.2,
            ),
            n_patients=100, seed=42, schema_mappings={},
        )
        result = DiscoveryResult(
            identified_exposure="x", identified_outcome="y",
            primary_hypothesis="X affects Y",
            methods_used=["regression"],
            estimated_effect=0.5,
            missingness_strategy="multiple_imputation",
            missingness_assessment="20% of Y values are missing, likely MCAR",
        )
        scored = scorer.score(spec, result)
        miss_dim = next(d for d in scored.dimension_scores if d.name == "missingness_handling")
        assert miss_dim.score >= 0.8


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def test_run_scenario(self, spec, perfect_result):
        benchmark = DiscoveryBenchmark()
        score = benchmark.run_scenario(spec, perfect_result)
        assert isinstance(score, BenchmarkScore)
        assert score.scenario_name == "scoring_test"
        assert score.difficulty == DifficultyTier.MEDIUM
        assert 0 <= score.overall_score <= 1

    def test_run_suite(self, spec, perfect_result, poor_result):
        benchmark = DiscoveryBenchmark()
        scores = benchmark.run_suite([spec, spec], [perfect_result, poor_result])
        assert len(scores) == 2
        assert scores[0].overall_score > scores[1].overall_score

    def test_run_suite_mismatched_length(self, spec, perfect_result):
        benchmark = DiscoveryBenchmark()
        with pytest.raises(ValueError, match="same length"):
            benchmark.run_suite([spec], [perfect_result, perfect_result])

    def test_summary(self, spec, perfect_result, poor_result):
        benchmark = DiscoveryBenchmark()
        scores = benchmark.run_suite([spec, spec], [perfect_result, poor_result])
        s = benchmark.summary(scores)
        assert "overall" in s
        assert "by_difficulty" in s
        assert "by_context_level" in s
        assert "by_dimension" in s
        assert len(s["scenarios"]) == 2

    def test_summary_empty(self):
        s = DiscoveryBenchmark.summary([])
        assert s["overall"] == 0.0

    def test_context_level_override(self, spec, perfect_result):
        benchmark = DiscoveryBenchmark()
        score = benchmark.run_scenario(spec, perfect_result, context_level=ContextLevel.BLIND)
        assert score.context_level == ContextLevel.BLIND


# ---------------------------------------------------------------------------
# ScoredResult structure
# ---------------------------------------------------------------------------


class TestScoredResultStructure:
    def test_scored_result_has_all_dimensions(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        dim_names = {d.name for d in scored.dimension_scores}
        assert dim_names == set(DIMENSION_WEIGHTS.keys())

    def test_overall_is_weighted_average(self, scorer, spec, perfect_result):
        scored = scorer.score(spec, perfect_result)
        expected = sum(d.weight * d.score for d in scored.dimension_scores)
        assert abs(scored.overall_score - round(expected, 4)) < 1e-6
