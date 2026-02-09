"""Tests for the LLM agent framework.

Tests StatisticalAgent, BenchmarkHarness, AgentContext, and
ClaudeDiscoveryAgent tool dispatch (mocked API).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cablecar.evaluation.agent import (
    AgentContext,
    BenchmarkHarness,
    DiscoveryAgent,
    StatisticalAgent,
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
from synthetic.dgp_generator import DGPSyntheticGenerator


# ---------------------------------------------------------------------------
# Shared fixture: a simple DGP spec
# ---------------------------------------------------------------------------


@pytest.fixture
def easy_spec() -> DGPSpec:
    return DGPSpec(
        name="test_easy",
        description="Test spec for agent tests",
        difficulty=DifficultyTier.EASY,
        n_patients=500,
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


@pytest.fixture
def easy_tables(easy_spec) -> dict:
    generator = DGPSyntheticGenerator(easy_spec)
    tables, _ = generator.generate()
    return tables


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_full_vignette(self, easy_spec):
        ctx = AgentContext(
            context_level=ContextLevel.FULL_VIGNETTE,
            vignette=easy_spec.vignette,
            domain_hint=easy_spec.domain_hint,
        )
        assert ctx.context_level == ContextLevel.FULL_VIGNETTE
        assert "vasopressors" in ctx.vignette

    def test_blind(self):
        ctx = AgentContext(context_level=ContextLevel.BLIND)
        assert ctx.vignette == ""
        assert ctx.domain_hint == ""

    def test_domain_hint(self, easy_spec):
        ctx = AgentContext(
            context_level=ContextLevel.DOMAIN_HINT,
            domain_hint=easy_spec.domain_hint,
        )
        assert "ICU" in ctx.domain_hint


# ---------------------------------------------------------------------------
# StatisticalAgent
# ---------------------------------------------------------------------------


class TestStatisticalAgent:
    def test_perfect_produces_valid_result(self, easy_spec, easy_tables):
        agent = StatisticalAgent(quality="perfect")
        agent._current_spec = easy_spec
        ctx = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        result = agent.run(easy_tables, ctx)

        assert isinstance(result, DiscoveryResult)
        assert result.identified_exposure == "vasopressors"
        assert result.identified_outcome == "mortality"
        assert result.estimated_effect is not None
        assert result.estimated_effect > 0

    def test_partial_fewer_confounders(self, easy_spec, easy_tables):
        agent = StatisticalAgent(quality="partial")
        agent._current_spec = easy_spec
        ctx = AgentContext(context_level=ContextLevel.DOMAIN_HINT)
        result = agent.run(easy_tables, ctx)

        assert result.identified_exposure == "vasopressors"
        assert len(result.proposed_adjustment_set) <= 1

    def test_naive_no_adjustment(self, easy_spec, easy_tables):
        agent = StatisticalAgent(quality="naive")
        agent._current_spec = easy_spec
        ctx = AgentContext(context_level=ContextLevel.BLIND)
        result = agent.run(easy_tables, ctx)

        assert result.identified_confounders == []

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError):
            StatisticalAgent(quality="invalid")

    def test_no_spec_raises(self, easy_tables):
        agent = StatisticalAgent(quality="perfect")
        ctx = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        with pytest.raises(RuntimeError):
            agent.run(easy_tables, ctx)


# ---------------------------------------------------------------------------
# BenchmarkHarness
# ---------------------------------------------------------------------------


class TestBenchmarkHarness:
    def test_run_scenario(self, easy_spec):
        agent = StatisticalAgent(quality="perfect")
        agent._current_spec = easy_spec
        harness = BenchmarkHarness(agent)

        # Monkey-patch so harness sets spec before calling run
        original_run_scenario = harness.run_scenario

        def patched_run_scenario(spec, ctx):
            agent._current_spec = spec
            return original_run_scenario(spec, ctx)

        harness.run_scenario = patched_run_scenario

        score = harness.run_scenario(easy_spec, ContextLevel.FULL_VIGNETTE)
        assert isinstance(score, BenchmarkScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.scenario_name == "test_easy"

    def test_run_suite(self, easy_spec):
        agent = StatisticalAgent(quality="perfect")
        harness = BenchmarkHarness(agent)

        original_run_scenario = harness.run_scenario

        def patched_run_scenario(spec, ctx):
            agent._current_spec = spec
            return original_run_scenario(spec, ctx)

        harness.run_scenario = patched_run_scenario

        scores = harness.run_suite(
            [easy_spec],
            [ContextLevel.FULL_VIGNETTE, ContextLevel.BLIND],
        )
        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, BenchmarkScore)

    def test_run_full_benchmark(self, easy_spec):
        agent = StatisticalAgent(quality="perfect")
        harness = BenchmarkHarness(agent)

        original_run_scenario = harness.run_scenario

        def patched_run_scenario(spec, ctx):
            agent._current_spec = spec
            return original_run_scenario(spec, ctx)

        harness.run_scenario = patched_run_scenario

        result = harness.run_full_benchmark(
            [easy_spec],
            [ContextLevel.FULL_VIGNETTE],
        )
        assert "overall" in result
        assert "scores" in result


# ---------------------------------------------------------------------------
# ClaudeDiscoveryAgent (mocked API)
# ---------------------------------------------------------------------------


class TestClaudeDiscoveryAgent:
    def test_import(self):
        from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent
        agent = ClaudeDiscoveryAgent(model="claude-sonnet-4-5-20250929")
        assert agent.model == "claude-sonnet-4-5-20250929"
        assert agent.max_turns == 15

    def test_tool_dispatch_get_schema(self, easy_tables):
        from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent
        agent = ClaudeDiscoveryAgent()
        result = agent._dispatch_tool("get_schema", {}, easy_tables)
        assert "tables" in result
        assert "patient" in result["tables"]

    def test_tool_dispatch_query_data(self, easy_tables):
        from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent
        agent = ClaudeDiscoveryAgent()
        result = agent._dispatch_tool(
            "query_data",
            {"table": "patient"},
            easy_tables,
        )
        assert result["table"] == "patient"
        assert result["rows"] == 500

    def test_tool_dispatch_unknown(self, easy_tables):
        from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent
        agent = ClaudeDiscoveryAgent()
        result = agent._dispatch_tool("nonexistent_tool", {}, easy_tables)
        assert "error" in result

    def test_parse_submit(self):
        from cablecar.evaluation.claude_agent import ClaudeDiscoveryAgent
        agent = ClaudeDiscoveryAgent()
        raw = {
            "identified_exposure": "vasopressors",
            "identified_outcome": "mortality",
            "primary_hypothesis": "Vasopressors increase mortality",
            "estimated_effect": 0.95,
            "confidence_interval": [0.5, 1.4],
        }
        result = agent._parse_submit(raw, [])
        assert isinstance(result, DiscoveryResult)
        assert result.identified_exposure == "vasopressors"
        assert result.estimated_effect == 0.95
        assert result.confidence_interval == (0.5, 1.4)
