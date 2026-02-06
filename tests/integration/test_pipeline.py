"""Integration test: full research workflow pipeline."""
import pytest

from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition
from cablecar.analysis.descriptive import DescriptiveAnalysis
from cablecar.analysis.hypothesis import HypothesisTest
from cablecar.analysis.regression import RegressionAnalysis
from cablecar.analysis.causal import CausalDAG
from cablecar.privacy.guard import PrivacyGuard
from cablecar.codegen.engine import CodeGenerator
from cablecar.codegen.provenance import AnalysisProvenance
from cablecar.reporting.strobe import STROBEReport
from cablecar.workflow.pipeline import AnalysisPipeline


class TestFullWorkflow:
    """End-to-end: load -> cohort -> table1 -> regression -> subgroup -> export -> report."""

    def test_complete_study_workflow(self, loaded_store, mini_schema):
        # --- Step 1: Cohort ---
        builder = CohortBuilder(loaded_store)
        cohort = builder.build(CohortDefinition(
            name="adults",
            inclusion_criteria=[{"column": "age_at_admission", "op": ">=", "value": 18}],
        ))
        assert cohort.n > 0

        # --- Step 2: Descriptive (Table 1) ---
        desc = DescriptiveAnalysis(cohort)
        table1 = desc.run(
            variables=["age_at_admission", "hospital_mortality", "discharge_category"],
            stratify_by="hospital_mortality",
        )
        assert table1.analysis_type == "descriptive"
        assert "overall" in table1.results

        # --- Step 3: Causal DAG ---
        dag = (
            CausalDAG("mortality_study")
            .add_variable("age_at_admission", role="exposure")
            .add_variable("hospital_mortality", role="outcome")
            .add_variable("discharge_category", role="confounder")
            .add_edge("age_at_admission", "hospital_mortality")
            .add_edge("discharge_category", "hospital_mortality")
        )
        adj_set = dag.get_minimal_adjustment_set()

        # --- Step 4: Regression ---
        reg = RegressionAnalysis(cohort)
        reg_result = reg.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="logistic",
        )
        assert "coefficients" in reg_result.results

        # --- Step 5: Subgroup ---
        elderly = cohort.subgroup(
            "elderly",
            [{"column": "age_at_admission", "op": ">=", "value": 65}],
        )
        young = cohort.subgroup(
            "young",
            [{"column": "age_at_admission", "op": "<", "value": 65}],
        )
        assert elderly.n + young.n == cohort.n

        elderly_desc = DescriptiveAnalysis(elderly).run(variables=["age_at_admission"])
        young_desc = DescriptiveAnalysis(young).run(variables=["age_at_admission"])
        if elderly_desc.results.get("age_at_admission", {}).get("mean") and \
           young_desc.results.get("age_at_admission", {}).get("mean"):
            assert elderly_desc.results["age_at_admission"]["mean"] >= \
                   young_desc.results["age_at_admission"]["mean"]

        # --- Step 6: Privacy ---
        guard = PrivacyGuard()
        safe_result = guard.sanitize_for_llm(reg_result.to_dict(), context="regression")
        assert safe_result["sanitized"] is True

        # --- Step 7: Code Export ---
        prov = AnalysisProvenance(
            study_name="ICU Mortality Study",
            data_source="./data/synthetic",
            schema_name="clif",
            cohort_definition={
                "inclusion": [{"column": "age_at_admission", "op": ">=", "value": 18}],
            },
        )
        prov.add_step("table1", "Descriptive analysis")
        prov.add_step("regression_analysis", "Logistic regression: mortality ~ age")

        gen = CodeGenerator()
        python_code = gen.generate("python", "regression", prov,
                                   outcome="hospital_mortality",
                                   predictors=["age_at_admission"],
                                   model_type="logistic")
        r_code = gen.generate("r", "regression", prov,
                              outcome="hospital_mortality",
                              predictors=["age_at_admission"],
                              model_type="logistic")
        assert "import" in python_code
        assert "library" in r_code

        # --- Step 8: STROBE Report ---
        report = STROBEReport()
        report.auto_populate(prov, cohort_summary={"n": cohort.n})
        completion = report.get_completion()
        assert completion["drafted"] > 0

    def test_pipeline_orchestration(self, loaded_store):
        """Test the AnalysisPipeline class for step management."""
        cohort_result = {}

        def load_step():
            return {"tables": list(loaded_store.tables.keys())}

        def cohort_step():
            builder = CohortBuilder(loaded_store)
            cohort = builder.build(CohortDefinition(name="all"))
            cohort_result["cohort"] = cohort
            return {"n": cohort.n}

        def analysis_step():
            cohort = cohort_result["cohort"]
            return DescriptiveAnalysis(cohort).run(variables=["age_at_admission"])

        pipeline = (
            AnalysisPipeline(name="test_study")
            .add_step("load", "Load data", load_step)
            .add_step("cohort", "Define cohort", cohort_step)
            .add_step("analyze", "Run descriptive", analysis_step)
        )

        results = pipeline.run_all()
        assert "load" in results
        assert "cohort" in results
        assert "analyze" in results
        assert pipeline.state.analyses_completed == ("load", "cohort", "analyze")

    def test_subgroup_efficiency(self, mini_cohort):
        """Subgroups should filter without reloading data."""
        initial_tables = set(mini_cohort.tables.keys())
        sub = mini_cohort.subgroup("sub", [{"column": "age_at_admission", "op": ">", "value": 60}])
        # Subgroup should have the same set of tables
        assert set(sub.tables.keys()) == initial_tables
        # Subgroup should be smaller
        assert sub.n <= mini_cohort.n
