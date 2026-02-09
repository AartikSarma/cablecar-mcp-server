"""Tests for cablecar.evaluation package exports and integration."""

import pytest

from cablecar.evaluation import (
    BenchmarkScore,
    DGPSpec,
    DiscoveryBenchmark,
    DiscoveryResult,
    DiscoveryScorer,
)
from cablecar.evaluation.dgp import (
    CausalEdge,
    CLIFMapping,
    DGPVariable,
    DifficultyTier,
    Distribution,
    FunctionalForm,
    GroundTruth,
    VariableRole,
    VariableType,
)
from cablecar.evaluation.discovery_result import AnalysisStep


class TestPackageExports:
    """Verify the public API from __init__.py works."""

    def test_dgp_spec_importable(self):
        assert DGPSpec is not None

    def test_discovery_result_importable(self):
        assert DiscoveryResult is not None

    def test_scorer_importable(self):
        assert DiscoveryScorer is not None

    def test_benchmark_importable(self):
        assert DiscoveryBenchmark is not None

    def test_benchmark_score_importable(self):
        assert BenchmarkScore is not None


class TestEndToEnd:
    """Basic end-to-end: create spec, create result, score."""

    def test_full_pipeline(self):
        spec = DGPSpec(
            name="e2e_test",
            variables=[
                DGPVariable(
                    name="age",
                    variable_type=VariableType.CONTINUOUS,
                    distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
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
            ],
            edges=[
                CausalEdge(cause="age", effect="treatment", functional_form=FunctionalForm.LOGISTIC, effect_size=0.03),
                CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.02),
                CausalEdge(cause="treatment", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.8),
            ],
            ground_truth=GroundTruth(
                primary_exposure="treatment",
                primary_outcome="mortality",
                true_causal_effect=0.8,
                correct_adjustment_set=["age"],
                expected_dag_edges=[("age", "treatment"), ("age", "mortality"), ("treatment", "mortality")],
                effect_size_tolerance=0.2,
            ),
            difficulty=DifficultyTier.EASY,
            n_patients=100,
            seed=42,
            schema_mappings={},
        )

        result = DiscoveryResult(
            identified_exposure="treatment",
            identified_outcome="mortality",
            identified_confounders=["age"],
            primary_hypothesis="Treatment increases mortality after adjusting for age",
            proposed_dag_edges=[("age", "treatment"), ("age", "mortality"), ("treatment", "mortality")],
            proposed_adjustment_set=["age"],
            methods_used=["descriptive", "logistic_regression", "dag"],
            method_justification="Binary outcome requires logistic regression",
            estimated_effect=0.75,
            confidence_interval=(0.5, 1.0),
            p_value=0.002,
            effect_size_metric="log_odds_ratio",
            interpretation="Treatment is associated with increased mortality after adjusting for age",
            limitations=["Observational design"],
            analysis_steps=[
                AnalysisStep(step_number=1, description="Loaded data", tool_used="load_data", result_summary="100 patients"),
            ],
        )

        scorer = DiscoveryScorer()
        scored = scorer.score(spec, result)
        assert scored.overall_score > 0.5
        assert len(scored.dimension_scores) == 8

        benchmark = DiscoveryBenchmark()
        benchmark_score = benchmark.run_scenario(spec, result)
        assert isinstance(benchmark_score, BenchmarkScore)
        assert benchmark_score.overall_score > 0.5
