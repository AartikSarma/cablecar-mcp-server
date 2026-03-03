"""Tests for the subagent-based benchmark runner.

Tests prepare_scenario, build_agent_prompt, parse_agent_output, and
score_result from cablecar.evaluation.benchmark_runner.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cablecar.evaluation.benchmark_runner import (
    build_agent_prompt,
    parse_agent_output,
    prepare_scenario,
    score_result,
)
from cablecar.evaluation.benchmarks import BenchmarkScore
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
    VariableRole,
    VariableType,
)
from cablecar.evaluation.discovery_result import DiscoveryResult


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def easy_spec() -> DGPSpec:
    return DGPSpec(
        name="test_easy",
        description="Test spec for benchmark runner tests",
        difficulty=DifficultyTier.EASY,
        n_patients=200,
        seed=42,
        vignette="Test vignette about vasopressors and mortality.",
        domain_hint="ICU cohort; vasopressor-mortality.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 12}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
            ),
            DGPVariable(
                name="vasopressors",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.3}),
                role=VariableRole.EXPOSURE,
                description="Vasopressor use",
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
            CausalEdge(cause="age", effect="vasopressors", functional_form=FunctionalForm.LINEAR, effect_size=0.03),
            CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.02),
            CausalEdge(cause="vasopressors", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=1.0),
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
            expected_hypotheses=["Vasopressor use increases mortality"],
            effect_size_tolerance=0.3,
        ),
        schema_mappings={
            "age": CLIFMapping(table="patient", column="age"),
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


# ---------------------------------------------------------------------------
# TestPrepareScenario
# ---------------------------------------------------------------------------


class TestPrepareScenario:
    def test_tables_dir_exists(self, easy_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_path = prepare_scenario(easy_spec, base_dir=Path(tmpdir))
            assert tables_path.exists()
            assert tables_path.is_dir()
            assert tables_path.name == "tables"

    def test_contains_expected_csvs(self, easy_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_path = prepare_scenario(easy_spec, base_dir=Path(tmpdir))
            csv_files = {f.name for f in tables_path.glob("*.csv")}
            assert "patient.csv" in csv_files
            assert "hospitalization.csv" in csv_files

    def test_ground_truth_is_separate(self, easy_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_path = prepare_scenario(easy_spec, base_dir=Path(tmpdir))
            base = tables_path.parent
            gt_dir = base / "ground_truth"
            assert gt_dir.exists()
            assert (gt_dir / "dgp_spec.json").exists()
            # Ground truth should NOT be in the tables dir
            assert not (tables_path / "dgp_spec.json").exists()

    def test_returns_absolute_path(self, easy_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_path = prepare_scenario(easy_spec, base_dir=Path(tmpdir))
            assert tables_path.is_absolute()

    def test_uses_temp_dir_when_no_base(self, easy_spec):
        tables_path = prepare_scenario(easy_spec)
        assert tables_path.exists()
        assert "cablecar_bench_" in str(tables_path.parent)


# ---------------------------------------------------------------------------
# TestBuildAgentPrompt
# ---------------------------------------------------------------------------


class TestBuildAgentPrompt:
    def test_full_vignette_includes_vignette(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.FULL_VIGNETTE)
        assert "vasopressors" in prompt
        assert "Test vignette" in prompt

    def test_domain_hint_includes_hint(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.DOMAIN_HINT)
        assert "ICU cohort" in prompt
        assert "Test vignette" not in prompt

    def test_blind_excludes_context(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.BLIND)
        assert "No prior information" in prompt
        assert "Test vignette" not in prompt
        assert "ICU cohort; vasopressor-mortality." not in prompt

    def test_includes_data_path(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.BLIND)
        assert "/tmp/test/tables" in prompt

    def test_includes_load_data_instruction(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.FULL_VIGNETTE)
        assert "/load-data" in prompt

    def test_includes_discovery_result_tag(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.FULL_VIGNETTE)
        assert "discovery_result" in prompt

    def test_excludes_ground_truth(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.FULL_VIGNETTE)
        # Should not contain the true causal effect or adjustment set info
        assert "true_causal_effect" not in prompt
        assert "correct_adjustment_set" not in prompt
        assert "effect_size_tolerance" not in prompt
        assert "1.0" not in prompt or "log-OR" not in prompt  # true effect value


# ---------------------------------------------------------------------------
# TestParseAgentOutput
# ---------------------------------------------------------------------------


class TestParseAgentOutput:
    def test_discovery_result_fence(self):
        text = (
            "Here is my analysis.\n\n"
            "```discovery_result\n"
            '{\n'
            '  "identified_exposure": "vasopressors",\n'
            '  "identified_outcome": "mortality",\n'
            '  "primary_hypothesis": "Vasopressors increase mortality",\n'
            '  "estimated_effect": 0.95\n'
            '}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert isinstance(result, DiscoveryResult)
        assert result.identified_exposure == "vasopressors"
        assert result.identified_outcome == "mortality"
        assert result.estimated_effect == 0.95

    def test_fallback_to_json_fence(self):
        text = (
            "Some analysis text.\n\n"
            "```json\n"
            '{\n'
            '  "identified_exposure": "sepsis",\n'
            '  "identified_outcome": "AKI",\n'
            '  "primary_hypothesis": "Sepsis causes AKI"\n'
            '}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert result.identified_exposure == "sepsis"
        assert result.identified_outcome == "AKI"

    def test_no_json_returns_fallback(self):
        text = "I couldn't complete the analysis. Sorry!"
        result = parse_agent_output(text)
        assert result.identified_exposure == "unknown"
        assert result.identified_outcome == "unknown"

    def test_dag_edges_list_to_tuple_coercion(self):
        text = (
            "```discovery_result\n"
            '{\n'
            '  "identified_exposure": "X",\n'
            '  "identified_outcome": "Y",\n'
            '  "primary_hypothesis": "X causes Y",\n'
            '  "proposed_dag_edges": [["X", "Y"], ["Z", "Y"]]\n'
            '}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert result.proposed_dag_edges == [("X", "Y"), ("Z", "Y")]

    def test_confidence_interval_list_to_tuple_coercion(self):
        text = (
            "```discovery_result\n"
            '{\n'
            '  "identified_exposure": "X",\n'
            '  "identified_outcome": "Y",\n'
            '  "primary_hypothesis": "X causes Y",\n'
            '  "confidence_interval": [0.5, 1.5]\n'
            '}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert result.confidence_interval == (0.5, 1.5)

    def test_prefers_discovery_result_over_json(self):
        text = (
            "```json\n"
            '{"identified_exposure": "wrong", "identified_outcome": "wrong", '
            '"primary_hypothesis": "wrong"}\n'
            "```\n\n"
            "```discovery_result\n"
            '{"identified_exposure": "right", "identified_outcome": "right", '
            '"primary_hypothesis": "right"}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert result.identified_exposure == "right"

    def test_uses_last_json_block_with_exposure(self):
        text = (
            "```json\n"
            '{"some": "other data"}\n'
            "```\n\n"
            "```json\n"
            '{"identified_exposure": "X", "identified_outcome": "Y", '
            '"primary_hypothesis": "test"}\n'
            "```\n"
        )
        result = parse_agent_output(text)
        assert result.identified_exposure == "X"


# ---------------------------------------------------------------------------
# TestScoreResult
# ---------------------------------------------------------------------------


class TestScoreResult:
    def test_perfect_result_scores_high(self, easy_spec):
        perfect_result = DiscoveryResult(
            identified_exposure="vasopressors",
            identified_outcome="mortality",
            identified_confounders=["age"],
            primary_hypothesis="Vasopressor use increases mortality",
            proposed_dag_edges=[
                ("age", "vasopressors"),
                ("age", "mortality"),
                ("vasopressors", "mortality"),
            ],
            proposed_adjustment_set=["age"],
            methods_used=["logistic_regression"],
            method_justification="Binary outcome requires logistic regression.",
            estimated_effect=1.0,
            confidence_interval=(0.7, 1.3),
            p_value=0.001,
            effect_size_metric="log_odds_ratio",
            interpretation="Vasopressors increase mortality after adjusting for age.",
            limitations=["Observational study"],
        )
        score = score_result(easy_spec, perfect_result, ContextLevel.FULL_VIGNETTE)
        assert isinstance(score, BenchmarkScore)
        assert score.overall_score > 0.5

    def test_fallback_result_scores_low(self, easy_spec):
        fallback_result = DiscoveryResult(
            identified_exposure="unknown",
            identified_outcome="unknown",
            primary_hypothesis="Agent did not produce structured results",
        )
        score = score_result(easy_spec, fallback_result, ContextLevel.FULL_VIGNETTE)
        assert isinstance(score, BenchmarkScore)
        assert score.overall_score < 0.3


# ---------------------------------------------------------------------------
# TestSecurityBoundary
# ---------------------------------------------------------------------------


class TestSecurityBoundary:
    """Verify that the agent prompt does not leak ground truth."""

    def test_prompt_forbids_source_reading(self, easy_spec):
        prompt = build_agent_prompt("/tmp/test/tables", easy_spec, ContextLevel.FULL_VIGNETTE)
        assert "Do NOT read source code files" in prompt
        assert "analysis tools provided" in prompt

    def test_prompt_does_not_leak_ground_truth(self, easy_spec):
        for context_level in ContextLevel:
            prompt = build_agent_prompt("/tmp/test/tables", easy_spec, context_level)
            assert "true_causal_effect" not in prompt
            assert "correct_adjustment_set" not in prompt
            assert "expected_dag_edges" not in prompt
