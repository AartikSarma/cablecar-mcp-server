"""Tests for cablecar.analysis.causal: CausalDAG."""
import pytest

from cablecar.analysis.causal import CausalDAG, Variable


class TestCausalDAG:
    def test_add_variable(self):
        dag = CausalDAG("test")
        dag.add_variable("treatment", role="exposure")
        dag.add_variable("mortality", role="outcome")
        assert dag.get_exposure() == "treatment"
        assert dag.get_outcome() == "mortality"

    def test_chaining(self):
        dag = (
            CausalDAG("test")
            .add_variable("X", role="exposure")
            .add_variable("Y", role="outcome")
            .add_edge("X", "Y")
        )
        assert dag.get_exposure() == "X"

    def test_add_edge(self):
        dag = CausalDAG()
        dag.add_variable("A")
        dag.add_variable("B")
        dag.add_edge("A", "B")
        summary = dag.summary()
        assert summary["n_edges"] == 1

    def test_cycle_detection(self):
        dag = CausalDAG()
        dag.add_variable("A")
        dag.add_variable("B")
        dag.add_variable("C")
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")
        with pytest.raises(ValueError, match="cycle"):
            dag.add_edge("C", "A")

    def test_auto_create_variables_on_edge(self):
        dag = CausalDAG()
        dag.add_edge("X", "Y")
        summary = dag.summary()
        assert summary["n_variables"] == 2

    def test_minimal_adjustment_set_simple(self):
        """Classic confounding: severity -> treatment AND severity -> mortality."""
        dag = (
            CausalDAG("test")
            .add_variable("treatment", role="exposure")
            .add_variable("mortality", role="outcome")
            .add_variable("severity", role="confounder")
            .add_edge("severity", "treatment")
            .add_edge("severity", "mortality")
            .add_edge("treatment", "mortality")
        )
        adj = dag.get_minimal_adjustment_set()
        assert "severity" in adj
        assert "treatment" not in adj
        assert "mortality" not in adj

    def test_adjustment_set_excludes_colliders(self):
        dag = (
            CausalDAG("test")
            .add_variable("treatment", role="exposure")
            .add_variable("mortality", role="outcome")
            .add_variable("severity", role="confounder")
            .add_variable("los", role="collider")
            .add_edge("severity", "treatment")
            .add_edge("severity", "mortality")
            .add_edge("treatment", "mortality")
            .add_edge("treatment", "los")
            .add_edge("severity", "los")
        )
        adj = dag.get_minimal_adjustment_set()
        assert "severity" in adj
        assert "los" not in adj

    def test_empty_adjustment_no_exposure(self):
        dag = CausalDAG()
        dag.add_variable("A", role="confounder")
        assert dag.get_minimal_adjustment_set() == set()

    def test_collider_bias_warning(self):
        dag = (
            CausalDAG()
            .add_variable("A", role="exposure")
            .add_variable("B", role="outcome")
            .add_variable("C", role="collider")
            .add_edge("A", "C")
            .add_edge("B", "C")
        )
        warnings = dag.check_for_collider_bias()
        assert len(warnings) > 0
        assert "C" in warnings[0]

    def test_no_collider_warnings_when_clean(self):
        dag = (
            CausalDAG()
            .add_variable("A", role="exposure")
            .add_variable("B", role="outcome")
            .add_edge("A", "B")
        )
        assert dag.check_for_collider_bias() == []

    def test_to_mermaid(self):
        dag = (
            CausalDAG()
            .add_variable("X", role="exposure")
            .add_variable("Y", role="outcome")
            .add_edge("X", "Y")
        )
        mermaid = dag.to_mermaid()
        assert "graph LR" in mermaid
        assert "X" in mermaid
        assert "Y" in mermaid
        assert "-->" in mermaid

    def test_summary(self):
        dag = (
            CausalDAG("my_dag")
            .add_variable("X", role="exposure")
            .add_variable("Y", role="outcome")
            .add_variable("C", role="confounder")
            .add_edge("C", "X")
            .add_edge("C", "Y")
            .add_edge("X", "Y")
        )
        s = dag.summary()
        assert s["name"] == "my_dag"
        assert s["n_variables"] == 3
        assert s["n_edges"] == 3
        assert s["exposure"] == "X"
        assert s["outcome"] == "Y"
        assert "C" in s["confounders"]
        assert s["is_dag"] is True

    def test_get_confounders(self):
        dag = CausalDAG()
        dag.add_variable("A", role="confounder")
        dag.add_variable("B", role="confounder")
        dag.add_variable("C", role="mediator")
        assert set(dag.get_confounders()) == {"A", "B"}

    def test_multiple_confounders_adjustment(self):
        """Multiple confounders should all appear in adjustment set."""
        dag = (
            CausalDAG()
            .add_variable("X", role="exposure")
            .add_variable("Y", role="outcome")
            .add_variable("C1", role="confounder")
            .add_variable("C2", role="confounder")
            .add_edge("C1", "X")
            .add_edge("C1", "Y")
            .add_edge("C2", "X")
            .add_edge("C2", "Y")
            .add_edge("X", "Y")
        )
        adj = dag.get_minimal_adjustment_set()
        assert "C1" in adj
        assert "C2" in adj
