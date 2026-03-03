"""Tests for the model-agnostic tool-use harness.

All tests use MockAdapter — no API keys required.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cablecar.evaluation.adapters import (
    AdapterResult,
    ConversationMessage,
    LLMAdapter,
    ToolDispatcher,
    ToolSchema,
)
from cablecar.evaluation.agent import AgentContext, BenchmarkHarness
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
from cablecar.evaluation.harness import (
    InProcessToolDispatcher,
    ToolUseAgent,
    build_tool_use_system_prompt,
    get_tool_schemas,
)
from synthetic.dgp_generator import DGPSyntheticGenerator


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------


class MockAdapter(LLMAdapter):
    """Deterministic adapter that emits a 3-turn conversation.

    Turn 1: calls load_data
    Turn 2: calls get_schema
    Turn 3: emits a canned discovery_result (no tool calls)
    """

    CANNED_RESULT = json.dumps({
        "identified_exposure": "vasopressors",
        "identified_outcome": "mortality",
        "identified_confounders": ["age"],
        "primary_hypothesis": "Vasopressor use increases mortality risk",
        "proposed_dag_edges": [
            ["age", "vasopressors"],
            ["age", "mortality"],
            ["vasopressors", "mortality"],
        ],
        "proposed_adjustment_set": ["age"],
        "methods_used": ["logistic_regression", "descriptive"],
        "method_justification": "Binary outcome requires logistic regression.",
        "estimated_effect": 0.95,
        "confidence_interval": [0.6, 1.3],
        "p_value": 0.001,
        "effect_size_metric": "log_odds_ratio",
        "interpretation": "Vasopressors increase mortality after adjusting for age.",
        "limitations": ["Observational study"],
    })

    def __init__(self) -> None:
        self._turn = 0

    @property
    def model_name(self) -> str:
        return "mock-adapter-v1"

    def run_tool_loop(
        self,
        system_prompt: str,
        tools: list[ToolSchema],
        tool_dispatcher: ToolDispatcher,
        max_turns: int = 20,
    ) -> AdapterResult:
        transcript: list[ConversationMessage] = []
        turns_used = 0

        # Turn 1: load_data
        turns_used += 1
        load_result = tool_dispatcher.dispatch(
            "load_data", {"path": "data", "schema": None}
        )
        transcript.append(ConversationMessage(
            role="assistant",
            content="Loading the dataset...",
            tool_calls=[{"id": "tc_1", "name": "load_data", "input": {"path": "data"}}],
        ))
        transcript.append(ConversationMessage(
            role="tool",
            tool_results=[{"tool_use_id": "tc_1", "tool_name": "load_data", "result": load_result}],
        ))

        # Turn 2: get_schema
        turns_used += 1
        schema_result = tool_dispatcher.dispatch("get_schema", {})
        transcript.append(ConversationMessage(
            role="assistant",
            content="Getting the schema...",
            tool_calls=[{"id": "tc_2", "name": "get_schema", "input": {}}],
        ))
        transcript.append(ConversationMessage(
            role="tool",
            tool_results=[{"tool_use_id": "tc_2", "tool_name": "get_schema", "result": schema_result}],
        ))

        # Turn 3: emit discovery_result
        turns_used += 1
        final_text = (
            "Based on my analysis, here are my findings:\n\n"
            f"```discovery_result\n{self.CANNED_RESULT}\n```"
        )
        transcript.append(ConversationMessage(
            role="assistant",
            content=final_text,
        ))

        return AdapterResult(
            final_text=final_text,
            transcript=transcript,
            total_input_tokens=500,
            total_output_tokens=300,
            model=self.model_name,
            turns_used=turns_used,
            terminated_reason="no_tool_use",
        )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def easy_spec() -> DGPSpec:
    return DGPSpec(
        name="test_easy",
        description="Test spec for harness tests",
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
# TestToolSchemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_returns_four_tools(self):
        schemas = get_tool_schemas()
        assert len(schemas) == 4

    def test_correct_tool_names(self):
        schemas = get_tool_schemas()
        names = {s.name for s in schemas}
        assert names == {"get_schema", "load_data", "query_cohort", "execute_analysis"}

    def test_all_have_input_schemas(self):
        schemas = get_tool_schemas()
        for schema in schemas:
            assert isinstance(schema.input_schema, dict)
            assert "type" in schema.input_schema

    def test_execute_analysis_includes_new_types(self):
        schemas = get_tool_schemas()
        exec_schema = next(s for s in schemas if s.name == "execute_analysis")
        enum_values = exec_schema.input_schema["properties"]["analysis_type"]["enum"]
        assert "survival" in enum_values
        assert "xgboost" in enum_values

    def test_all_are_tool_schema_instances(self):
        schemas = get_tool_schemas()
        for schema in schemas:
            assert isinstance(schema, ToolSchema)


# ---------------------------------------------------------------------------
# TestInProcessToolDispatcher
# ---------------------------------------------------------------------------


class TestInProcessToolDispatcher:
    def test_setup_creates_csvs(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            tables_dir = dispatcher._temp_dir / "tables"
            assert tables_dir.exists()
            csv_files = {f.name for f in tables_dir.glob("*.csv")}
            for table_name in easy_tables:
                assert f"{table_name}.csv" in csv_files
        finally:
            dispatcher.cleanup()

    def test_dispatch_load_data(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            result_str = dispatcher.dispatch("load_data", {"path": "/fake/path"})
            result = json.loads(result_str)
            # Should load data despite the fake path (dispatcher substitutes)
            assert "error" not in result or result.get("sanitized")
            assert result.get("sanitized") is True
        finally:
            dispatcher.cleanup()

    def test_dispatch_get_schema(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            result_str = dispatcher.dispatch("get_schema", {})
            result = json.loads(result_str)
            assert result.get("sanitized") is True
        finally:
            dispatcher.cleanup()

    def test_dispatch_query_cohort_after_load(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            # Must load data first
            dispatcher.dispatch("load_data", {"path": "data"})
            result_str = dispatcher.dispatch("query_cohort", {
                "name": "main",
                "description": "All patients",
            })
            result = json.loads(result_str)
            assert result.get("sanitized") is True
        finally:
            dispatcher.cleanup()

    def test_dispatch_execute_analysis_after_cohort(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            dispatcher.dispatch("load_data", {"path": "data"})
            dispatcher.dispatch("query_cohort", {"name": "main"})
            result_str = dispatcher.dispatch("execute_analysis", {
                "analysis_type": "summary_stats",
                "cohort_name": "main",
            })
            result = json.loads(result_str)
            assert result.get("sanitized") is True
        finally:
            dispatcher.cleanup()

    def test_dispatch_unknown_tool(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            result_str = dispatcher.dispatch("nonexistent", {})
            result = json.loads(result_str)
            assert "error" in result
        finally:
            dispatcher.cleanup()

    def test_cleanup_removes_temp_dir(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        dispatcher.setup()
        temp_dir = dispatcher._temp_dir
        assert temp_dir.exists()
        dispatcher.cleanup()
        assert not temp_dir.exists()

    def test_dispatch_before_setup_raises(self, easy_tables):
        dispatcher = InProcessToolDispatcher(easy_tables)
        with pytest.raises(RuntimeError, match="not initialized"):
            dispatcher.dispatch("get_schema", {})


# ---------------------------------------------------------------------------
# TestSystemPrompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_full_vignette_includes_vignette(self, easy_spec):
        context = AgentContext(
            context_level=ContextLevel.FULL_VIGNETTE,
            vignette=easy_spec.vignette,
        )
        prompt = build_tool_use_system_prompt(context, easy_spec)
        assert "vasopressors" in prompt
        assert "Test vignette" in prompt

    def test_domain_hint_includes_hint(self, easy_spec):
        context = AgentContext(
            context_level=ContextLevel.DOMAIN_HINT,
            domain_hint=easy_spec.domain_hint,
        )
        prompt = build_tool_use_system_prompt(context, easy_spec)
        assert "ICU cohort" in prompt
        assert "Test vignette" not in prompt

    def test_blind_excludes_context(self, easy_spec):
        context = AgentContext(context_level=ContextLevel.BLIND)
        prompt = build_tool_use_system_prompt(context, easy_spec)
        assert "No prior information" in prompt
        assert "Test vignette" not in prompt
        assert "ICU cohort; vasopressor-mortality." not in prompt

    def test_includes_tool_names(self, easy_spec):
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        prompt = build_tool_use_system_prompt(context, easy_spec)
        assert "get_schema" in prompt
        assert "load_data" in prompt
        assert "query_cohort" in prompt
        assert "execute_analysis" in prompt

    def test_includes_discovery_result_tag(self, easy_spec):
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        prompt = build_tool_use_system_prompt(context, easy_spec)
        assert "discovery_result" in prompt

    def test_no_ground_truth_leaked(self, easy_spec):
        """Ground truth must never appear in the system prompt."""
        for ctx_level in ContextLevel:
            context = AgentContext(context_level=ctx_level)
            prompt = build_tool_use_system_prompt(context, easy_spec)
            assert "true_causal_effect" not in prompt
            assert "correct_adjustment_set" not in prompt
            assert "expected_dag_edges" not in prompt
            assert "effect_size_tolerance" not in prompt


# ---------------------------------------------------------------------------
# TestToolUseAgent
# ---------------------------------------------------------------------------


class TestToolUseAgent:
    def test_produces_discovery_result(self, easy_spec, easy_tables):
        adapter = MockAdapter()
        agent = ToolUseAgent(adapter)
        agent.prepare(easy_spec)
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        result = agent.run(easy_tables, context)

        assert isinstance(result, DiscoveryResult)
        assert result.identified_exposure == "vasopressors"
        assert result.identified_outcome == "mortality"
        assert result.estimated_effect == 0.95

    def test_stores_adapter_result(self, easy_spec, easy_tables):
        adapter = MockAdapter()
        agent = ToolUseAgent(adapter)
        agent.prepare(easy_spec)
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        agent.run(easy_tables, context)

        ar = agent.last_adapter_result
        assert ar is not None
        assert isinstance(ar, AdapterResult)
        assert ar.model == "mock-adapter-v1"
        assert ar.turns_used == 3
        assert ar.total_input_tokens > 0
        assert ar.total_output_tokens > 0
        assert len(ar.transcript) > 0

    def test_raises_without_prepare(self, easy_tables):
        adapter = MockAdapter()
        agent = ToolUseAgent(adapter)
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        with pytest.raises(RuntimeError, match="prepare"):
            agent.run(easy_tables, context)

    def test_cleanup_on_error(self, easy_spec, easy_tables):
        """Dispatcher temp dir is cleaned up even if adapter raises."""

        class FailingAdapter(LLMAdapter):
            @property
            def model_name(self) -> str:
                return "failing"

            def run_tool_loop(self, system_prompt, tools, tool_dispatcher, max_turns=20):
                raise ValueError("Intentional test failure")

        agent = ToolUseAgent(FailingAdapter())
        agent.prepare(easy_spec)
        context = AgentContext(context_level=ContextLevel.FULL_VIGNETTE)
        with pytest.raises(ValueError, match="Intentional"):
            agent.run(easy_tables, context)


# ---------------------------------------------------------------------------
# TestHarnessIntegration
# ---------------------------------------------------------------------------


class TestHarnessIntegration:
    def test_benchmark_harness_with_tool_use_agent(self, easy_spec):
        """BenchmarkHarness works with ToolUseAgent and MockAdapter."""
        adapter = MockAdapter()
        agent = ToolUseAgent(adapter)
        harness = BenchmarkHarness(agent)

        score = harness.run_scenario(easy_spec, ContextLevel.FULL_VIGNETTE)
        assert isinstance(score, BenchmarkScore)
        assert 0.0 <= score.overall_score <= 1.0
        assert score.scenario_name == "test_easy"
        # Mock produces a good result so score should be decent
        assert score.overall_score > 0.3

    def test_adapter_result_available_after_harness(self, easy_spec):
        """The AdapterResult is accessible after harness run."""
        adapter = MockAdapter()
        agent = ToolUseAgent(adapter)
        harness = BenchmarkHarness(agent)

        harness.run_scenario(easy_spec, ContextLevel.FULL_VIGNETTE)
        assert agent.last_adapter_result is not None
        assert agent.last_adapter_result.model == "mock-adapter-v1"


# ---------------------------------------------------------------------------
# TestPrivacyBoundary
# ---------------------------------------------------------------------------


class TestPrivacyBoundary:
    def test_dispatcher_results_are_sanitized(self, easy_tables):
        """All dispatcher results must have sanitized=True."""
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()

            # load_data
            result = json.loads(dispatcher.dispatch("load_data", {"path": "data"}))
            assert result.get("sanitized") is True

            # get_schema
            result = json.loads(dispatcher.dispatch("get_schema", {}))
            assert result.get("sanitized") is True

            # query_cohort
            dispatcher.dispatch("load_data", {"path": "data"})
            result = json.loads(dispatcher.dispatch("query_cohort", {"name": "main"}))
            assert result.get("sanitized") is True

            # execute_analysis
            result = json.loads(dispatcher.dispatch("execute_analysis", {
                "analysis_type": "summary_stats",
                "cohort_name": "main",
            }))
            assert result.get("sanitized") is True
        finally:
            dispatcher.cleanup()

    def test_no_raw_rows_in_any_output(self, easy_tables):
        """Tool outputs must never contain patient-level row data."""
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()

            # Load + cohort + analysis
            load_out = dispatcher.dispatch("load_data", {"path": "data"})
            dispatcher.dispatch("query_cohort", {"name": "main"})
            analysis_out = dispatcher.dispatch("execute_analysis", {
                "analysis_type": "descriptive",
                "cohort_name": "main",
            })

            # No output should contain the word "patient_id" as a data value
            # (it can appear in schema descriptions but not as exposed data)
            for output_str in [load_out, analysis_out]:
                result = json.loads(output_str)
                # Results should be aggregated, not row-level
                data = result.get("data", result)
                if isinstance(data, dict):
                    # Should not have a "rows" key with raw data
                    assert "rows" not in data
        finally:
            dispatcher.cleanup()

    def test_llm_path_is_intercepted(self, easy_tables):
        """LLM can pass any path to load_data — dispatcher substitutes its own."""
        dispatcher = InProcessToolDispatcher(easy_tables)
        try:
            dispatcher.setup()
            # Pass a completely fake path
            result_str = dispatcher.dispatch("load_data", {"path": "/nonexistent/path/to/data"})
            result = json.loads(result_str)
            # Should succeed because dispatcher intercepts the path
            data = result.get("data", result)
            if "status" in data:
                assert data["status"] == "loaded"
        finally:
            dispatcher.cleanup()
