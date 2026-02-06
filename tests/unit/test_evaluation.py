"""Tests for cablecar.evaluation: scenarios, graders, and benchmarks."""
import pytest

from cablecar.evaluation.scenarios import (
    EvaluationScenario, SCENARIOS, get_scenario, list_scenarios,
)
from cablecar.evaluation.graders import OutputGrader, GradeResult
from cablecar.evaluation.benchmarks import ClinicalBenchmark, BenchmarkResult


class TestEvaluationScenario:
    def test_scenario_count(self):
        assert len(SCENARIOS) == 6

    def test_get_scenario(self):
        s = get_scenario("age_mortality")
        assert s is not None
        assert s.name == "age_mortality"

    def test_get_scenario_missing(self):
        assert get_scenario("nonexistent") is None

    def test_list_scenarios(self):
        listing = list_scenarios()
        assert len(listing) == 6
        assert all("name" in s for s in listing)
        assert all("difficulty" in s for s in listing)

    def test_to_dict(self):
        s = SCENARIOS[0]
        d = s.to_dict()
        assert d["name"] == s.name
        assert "expected_findings" in d
        assert "expected_methods" in d

    def test_difficulty_levels(self):
        difficulties = {s.difficulty for s in SCENARIOS}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_domains(self):
        domains = {s.domain for s in SCENARIOS}
        assert "general" in domains
        assert "causal" in domains
        assert "prediction" in domains


class TestOutputGrader:
    def test_grade_with_matching_output(self):
        grader = OutputGrader()
        scenario = get_scenario("age_mortality")
        output = {
            "methods_used": ["descriptive", "logistic_regression"],
            "results": {"effect_estimate": 0.03, "p_value": 0.01},
            "descriptive_stats_provided": True,
            "hypothesis_tested": True,
            "effect_size_reported": True,
            "confidence_intervals": True,
            "assumptions_checked": True,
            "limitations_noted": True,
        }
        result = grader.grade(scenario, output)
        assert isinstance(result, GradeResult)
        assert result.overall_score > 50

    def test_grade_empty_output(self):
        grader = OutputGrader()
        scenario = get_scenario("age_mortality")
        result = grader.grade(scenario, {})
        assert isinstance(result, GradeResult)
        assert result.overall_score >= 0

    def test_grade_causal_scenario(self):
        grader = OutputGrader()
        scenario = get_scenario("confounding_challenge")
        output = {
            "methods_used": ["causal_dag", "unadjusted_regression", "adjusted_regression"],
            "results": {"effect_estimate": 0.5, "p_value": 0.01},
            "confounders_identified": ["severity"],
            "dag_constructed": True,
            "adjusted_analysis": True,
            "descriptive_stats_provided": True,
            "hypothesis_tested": True,
            "effect_size_reported": True,
            "confidence_intervals": True,
            "assumptions_checked": True,
            "limitations_noted": True,
        }
        result = grader.grade(scenario, output)
        assert result.dimensions["causal_reasoning"] > 50

    def test_grade_result_to_dict(self):
        result = GradeResult(
            scenario_name="test",
            overall_score=75.0,
            dimensions={"method_choice": 80.0},
            feedback=["Good job"],
        )
        d = result.to_dict()
        assert d["scenario"] == "test"
        assert d["overall_score"] == 75.0


class TestClinicalBenchmark:
    def test_run_benchmark(self):
        benchmark = ClinicalBenchmark()
        outputs = {
            "age_mortality": {
                "methods_used": ["descriptive"],
                "results": {"effect_estimate": 0.02, "p_value": 0.03},
            },
        }
        result = benchmark.run(outputs, name="test_run")
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_run"
        assert result.n_scenarios == 6
        assert len(result.grades) == 6

    def test_benchmark_summary(self):
        benchmark = ClinicalBenchmark()
        result = benchmark.run({}, name="empty")
        s = result.summary()
        assert "overall_score" in s
        assert "by_domain" in s
        assert "by_difficulty" in s

    def test_list_scenarios(self):
        benchmark = ClinicalBenchmark()
        scenarios = benchmark.list_scenarios()
        assert len(scenarios) == 6

    def test_get_scenario(self):
        benchmark = ClinicalBenchmark()
        s = benchmark.get_scenario("age_mortality")
        assert s is not None
        assert s.name == "age_mortality"

    def test_benchmark_overall_score(self):
        benchmark = ClinicalBenchmark()
        result = benchmark.run({})
        assert isinstance(result.overall_score, float)
        assert 0 <= result.overall_score <= 100
