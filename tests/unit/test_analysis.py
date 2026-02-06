"""Tests for cablecar.analysis: descriptive, hypothesis, and regression."""
import pandas as pd
import numpy as np
import pytest

from cablecar.analysis.base import AnalysisResult
from cablecar.analysis.descriptive import DescriptiveAnalysis
from cablecar.analysis.hypothesis import HypothesisTest
from cablecar.analysis.regression import RegressionAnalysis


class TestDescriptiveAnalysis:
    def test_numeric_variable(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(variables=["age_at_admission"])
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "descriptive"
        stats = result.results["age_at_admission"]
        assert stats["type"] == "numeric"
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "q1" in stats
        assert "q3" in stats

    def test_categorical_variable(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(variables=["discharge_category"])
        stats = result.results["discharge_category"]
        assert stats["type"] == "categorical"
        assert "categories" in stats
        assert "n_categories" in stats

    def test_stratified(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(
            variables=["age_at_admission"],
            stratify_by="discharge_category",
        )
        assert "overall" in result.results

    def test_missing_variable_warning(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(variables=["nonexistent"])
        assert len(result.warnings) > 0

    def test_missing_table_warning(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(variables=["age"], table="nonexistent_table")
        assert len(result.warnings) > 0

    def test_smd_computation(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(
            variables=["age_at_admission"],
            stratify_by="discharge_category",
        )
        if "smd" in result.results:
            smd_val = result.results["smd"].get("age_at_admission")
            if smd_val is not None:
                assert isinstance(smd_val, float)

    def test_binary_treated_as_categorical(self, mini_cohort):
        analysis = DescriptiveAnalysis(mini_cohort)
        result = analysis.run(variables=["hospital_mortality"])
        stats = result.results["hospital_mortality"]
        assert stats["type"] == "categorical"


class TestHypothesisTest:
    def test_auto_detect_mann_whitney(self, mini_cohort):
        """Numeric variable with 2 groups should auto-select Mann-Whitney."""
        # Create a binary grouping
        analysis = HypothesisTest(mini_cohort)
        result = analysis.run(
            variable="age_at_admission",
            group_variable="hospital_mortality",
            test="auto",
        )
        assert result.analysis_type == "hypothesis_test"
        assert result.results.get("test_name") in ("mann_whitney", "kruskal")
        assert "p_value" in result.results

    def test_explicit_t_test(self, mini_cohort):
        analysis = HypothesisTest(mini_cohort)
        result = analysis.run(
            variable="age_at_admission",
            group_variable="hospital_mortality",
            test="t_test",
        )
        assert result.results.get("test_name") == "t_test"
        assert "p_value" in result.results
        assert "effect_size" in result.results

    def test_chi_square(self, mini_cohort):
        analysis = HypothesisTest(mini_cohort)
        result = analysis.run(
            variable="discharge_category",
            group_variable="hospital_mortality",
            test="chi_square",
        )
        assert "p_value" in result.results

    def test_missing_variable_warning(self, mini_cohort):
        analysis = HypothesisTest(mini_cohort)
        result = analysis.run(
            variable="nonexistent",
            group_variable="hospital_mortality",
        )
        assert len(result.warnings) > 0

    def test_single_group_warning(self, loaded_store):
        """All same value in group var should warn."""
        # Make all patients have mortality=0
        loaded_store.tables["hospitalization"]["hospital_mortality"] = 0
        builder_module = __import__("cablecar.data.cohort", fromlist=["CohortBuilder", "CohortDefinition"])
        builder = builder_module.CohortBuilder(loaded_store)
        cohort = builder.build(builder_module.CohortDefinition(name="all"))
        analysis = HypothesisTest(cohort)
        result = analysis.run(
            variable="age_at_admission",
            group_variable="hospital_mortality",
        )
        assert len(result.warnings) > 0

    def test_batch_correct_bonferroni(self):
        corrected = HypothesisTest.batch_correct([0.01, 0.04, 0.08], method="bonferroni")
        assert len(corrected) == 3
        assert corrected[0] == pytest.approx(0.03, abs=0.001)
        assert corrected[1] == pytest.approx(0.12, abs=0.001)

    def test_batch_correct_fdr(self):
        corrected = HypothesisTest.batch_correct([0.01, 0.04, 0.08], method="fdr")
        assert len(corrected) == 3
        # FDR-corrected p-values should be >= raw but <= 1
        for raw, adj in zip([0.01, 0.04, 0.08], corrected):
            assert adj >= raw
            assert adj <= 1.0

    def test_batch_correct_empty(self):
        assert HypothesisTest.batch_correct([]) == []


class TestRegressionAnalysis:
    def test_logistic_regression(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="logistic",
        )
        assert result.analysis_type == "regression"
        assert "coefficients" in result.results

    def test_linear_regression(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="age_at_admission",
            predictors=["hospital_mortality"],
            model_type="linear",
        )
        assert "coefficients" in result.results
        assert "r_squared" in result.results

    def test_categorical_predictor(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission", "discharge_category"],
            model_type="logistic",
        )
        # Should have dummy-encoded coefficients
        assert "coefficients" in result.results

    def test_missing_column_warning(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["nonexistent_col"],
            model_type="logistic",
        )
        assert len(result.warnings) > 0

    def test_missing_table_warning(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="outcome",
            predictors=["x"],
            table="nonexistent",
        )
        assert len(result.warnings) > 0

    def test_confounders_included(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            confounders=["discharge_category"],
            model_type="logistic",
        )
        assert result.parameters.get("confounders") == ["discharge_category"]

    def test_cox_requires_time_col(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="cox",
        )
        assert len(result.warnings) > 0

    def test_unsupported_model_fallback(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="unknown_model",
        )
        assert len(result.warnings) > 0

    def test_logistic_returns_odds_ratios(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="logistic",
        )
        if result.results.get("coefficients"):
            for var_name, coef_info in result.results["coefficients"].items():
                if var_name != "const":
                    assert "odds_ratio" in coef_info

    def test_result_to_dict(self, mini_cohort):
        analysis = RegressionAnalysis(mini_cohort)
        result = analysis.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="logistic",
        )
        d = result.to_dict()
        assert "analysis_type" in d
        assert "results" in d
        assert "timestamp" in d
