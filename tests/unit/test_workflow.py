"""Tests for cablecar.workflow: pipeline, state, plan, and cache."""
import pytest

from cablecar.workflow.pipeline import AnalysisPipeline
from cablecar.workflow.state import WorkflowState
from cablecar.workflow.cache import ComputationCache


class TestWorkflowState:
    def test_immutability(self):
        state = WorkflowState(study_name="test")
        with pytest.raises(AttributeError):
            state.study_name = "changed"

    def test_with_update(self):
        state = WorkflowState(study_name="test")
        new_state = state.with_update(data_loaded=True, data_path="/data")
        assert new_state.data_loaded is True
        assert new_state.data_path == "/data"
        assert state.data_loaded is False  # original unchanged

    def test_add_analysis(self):
        state = WorkflowState()
        state2 = state.add_analysis("table1")
        state3 = state2.add_analysis("regression")
        assert state.analyses_completed == ()
        assert state2.analyses_completed == ("table1",)
        assert state3.analyses_completed == ("table1", "regression")

    def test_summary(self):
        state = WorkflowState(study_name="ICU Study", cohort_n=500)
        s = state.summary()
        assert s["study_name"] == "ICU Study"
        assert s["cohort_n"] == 500
        assert "timestamp" in s


class TestAnalysisPipeline:
    def test_add_and_run_step(self):
        pipeline = AnalysisPipeline(name="test")
        pipeline.add_step("load", "Load data", function=lambda: "loaded")
        result = pipeline.run_step("load")
        assert result == "loaded"

    def test_chaining(self):
        pipeline = (
            AnalysisPipeline(name="test")
            .add_step("step1", "First step", lambda: 1)
            .add_step("step2", "Second step", lambda: 2)
        )
        summary = pipeline.summary()
        assert len(summary["steps"]) == 2

    def test_run_all(self):
        results = []
        pipeline = AnalysisPipeline(name="test")
        pipeline.add_step("a", "Step A", function=lambda: results.append("a") or "a")
        pipeline.add_step("b", "Step B", function=lambda: results.append("b") or "b")
        all_results = pipeline.run_all()
        assert "a" in all_results
        assert "b" in all_results

    def test_missing_step_raises(self):
        pipeline = AnalysisPipeline()
        with pytest.raises(KeyError):
            pipeline.run_step("nonexistent")

    def test_step_failure(self):
        pipeline = AnalysisPipeline()
        def fail():
            raise RuntimeError("boom")
        pipeline.add_step("bad", "Fails", function=fail)
        with pytest.raises(RuntimeError):
            pipeline.run_step("bad")
        summary = pipeline.summary()
        assert summary["steps"][0]["status"] == "failed"

    def test_get_result(self):
        pipeline = AnalysisPipeline()
        pipeline.add_step("step", "A step", function=lambda: 42)
        pipeline.run_step("step")
        assert pipeline.get_result("step") == 42
        assert pipeline.get_result("nonexistent") is None

    def test_state_tracks_progress(self):
        pipeline = AnalysisPipeline(name="study")
        pipeline.add_step("load", "Load", function=lambda: True)
        pipeline.run_step("load")
        assert "load" in pipeline.state.analyses_completed

    def test_summary(self):
        pipeline = AnalysisPipeline(name="my_pipeline")
        pipeline.add_step("a", "Step A")
        s = pipeline.summary()
        assert s["name"] == "my_pipeline"
        assert len(s["steps"]) == 1
        assert "state" in s


class TestComputationCache:
    def test_get_set(self):
        cache = ComputationCache()
        cache.set("descriptive", {"var": "age"}, "result")
        assert cache.get("descriptive", {"var": "age"}) == "result"

    def test_get_missing_returns_none(self):
        cache = ComputationCache()
        assert cache.get("descriptive", {"var": "x"}) is None

    def test_has(self):
        cache = ComputationCache()
        cache.set("regression", {"outcome": "y"}, "result")
        assert cache.has("regression", {"outcome": "y"}) is True
        assert cache.has("regression", {"outcome": "z"}) is False

    def test_clear(self):
        cache = ComputationCache()
        cache.set("a", {"x": 1}, "v")
        assert cache.size() == 1
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a", {"x": 1}) is None

    def test_deterministic_keys(self):
        cache = ComputationCache()
        cache.set("t", {"a": 1, "b": 2}, "val")
        # Same params in different order should hit same key
        assert cache.get("t", {"b": 2, "a": 1}) == "val"
