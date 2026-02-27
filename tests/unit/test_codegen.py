"""Tests for cablecar.codegen: engine and provenance."""
import pytest

from cablecar.codegen.engine import CodeGenerator
from cablecar.codegen.provenance import AnalysisProvenance, AnalysisStep


class TestAnalysisProvenance:
    def test_creation(self):
        prov = AnalysisProvenance(study_name="test", data_source="./data")
        assert prov.study_name == "test"
        assert prov.steps == []
        assert prov.tables_used == []
        assert prov.data_format == "csv"

    def test_add_step(self):
        prov = AnalysisProvenance(study_name="test")
        prov.add_step("load", "Loaded data", {"path": "./data"})
        assert len(prov.steps) == 1
        assert prov.steps[0].step_name == "load"
        assert prov.steps[0].parameters == {"path": "./data"}

    def test_add_step_with_analysis_type(self):
        prov = AnalysisProvenance(study_name="test")
        prov.add_step(
            "regression_analysis",
            "Logistic regression: mortality ~ age",
            parameters={"outcome": "mortality", "predictors": ["age"]},
            analysis_type="regression",
            result_summary={"model_type": "logistic", "n": 500},
        )
        step = prov.steps[0]
        assert step.analysis_type == "regression"
        assert step.result_summary["model_type"] == "logistic"

    def test_to_dict_backward_compatible(self):
        """to_dict() must still have the keys STROBE/TRIPOD expect."""
        prov = AnalysisProvenance(study_name="test", schema_name="clif")
        prov.add_step("cohort", "Defined cohort")
        d = prov.to_dict()
        assert d["study_name"] == "test"
        assert d["schema_name"] == "clif"
        assert len(d["steps"]) == 1
        assert "timestamp" in d["steps"][0]
        assert "name" in d["steps"][0]
        assert "description" in d["steps"][0]
        assert "parameters" in d["steps"][0]
        # New keys present but additive
        assert "analysis_type" in d["steps"][0]
        assert "tables_used" in d

    def test_to_scaffold_context(self):
        prov = AnalysisProvenance(
            study_name="Test Study",
            data_source="./data",
            schema_name="clif",
            data_format="parquet",
            tables_used=["patient", "hospitalization"],
        )
        ctx = prov.to_scaffold_context()
        assert ctx["read_fn"] == "read_parquet"
        assert ctx["file_ext"] == "parquet"
        assert ctx["tables"] == ["patient", "hospitalization"]

    def test_to_scaffold_context_infers_tables(self):
        """When tables_used is empty, infer from schema."""
        prov = AnalysisProvenance(schema_name="clif")
        ctx = prov.to_scaffold_context()
        assert "patient" in ctx["tables"]
        assert "hospitalization" in ctx["tables"]
        assert len(ctx["tables"]) == 8

    def test_to_scaffold_context_csv_default(self):
        prov = AnalysisProvenance(data_source="./data")
        ctx = prov.to_scaffold_context()
        assert ctx["read_fn"] == "read_csv"
        assert ctx["file_ext"] == "csv"

    def test_to_llm_context_contains_study_info(self):
        prov = AnalysisProvenance(
            study_name="ICU Mortality",
            data_source="./data",
            schema_name="clif",
            cohort_definition={"inclusion": [{"column": "age", "op": ">=", "value": 18}]},
        )
        prov.add_step("table1", "Descriptive stats", analysis_type="descriptive")
        md = prov.to_llm_context()
        assert "ICU Mortality" in md
        assert "age" in md
        assert ">=" in md
        assert "18" in md
        assert "descriptive" in md

    def test_to_llm_context_with_parameters(self):
        prov = AnalysisProvenance(study_name="test")
        prov.add_step(
            "regression", "Logistic regression",
            parameters={"outcome": "mortality", "predictors": ["age", "sofa"]},
            analysis_type="regression",
        )
        md = prov.to_llm_context()
        assert "outcome: mortality" in md
        assert "regression" in md

    def test_infer_tables_clif(self):
        prov = AnalysisProvenance(schema_name="clif")
        tables = prov._infer_tables()
        assert "patient" in tables
        assert "hospitalization" in tables
        assert len(tables) == 8

    def test_infer_tables_unknown_schema(self):
        prov = AnalysisProvenance(schema_name="custom")
        tables = prov._infer_tables()
        assert tables == ["hospitalization"]


class TestCodeGenerator:
    @pytest.fixture
    def generator(self):
        return CodeGenerator()

    @pytest.fixture
    def provenance(self):
        return AnalysisProvenance(
            study_name="ICU Mortality Study",
            data_source="./data/synthetic",
            schema_name="clif",
            cohort_definition={
                "inclusion": [{"column": "age", "op": ">=", "value": 18}],
            },
        )

    def test_generate_scaffold_python(self, generator, provenance):
        provenance.add_step(
            "table1", "Baseline characteristics",
            analysis_type="descriptive",
            parameters={"variables": ["age", "sex"]},
        )
        code = generator.generate_scaffold("python", provenance)
        assert "import pandas" in code
        assert "ICU Mortality Study" in code
        assert "argparse" in code
        assert "def load_data" in code
        assert "def build_cohort" in code
        assert "def table1" in code
        assert "scipy" in code  # descriptive triggers scipy import

    def test_generate_scaffold_r(self, generator, provenance):
        provenance.add_step(
            "table1", "Baseline characteristics",
            analysis_type="descriptive",
            parameters={"variables": ["age", "sex"]},
        )
        code = generator.generate_scaffold("r", provenance)
        assert "library(tidyverse)" in code
        assert "library(broom)" in code
        assert "ICU Mortality Study" in code
        assert "load_data <- function" in code
        assert "build_cohort <- function" in code

    def test_generate_scaffold_regression_imports(self, generator, provenance):
        provenance.add_step(
            "regression_analysis", "Logistic regression",
            analysis_type="regression",
            parameters={"outcome": "mortality", "predictors": ["age"]},
        )
        code = generator.generate_scaffold("python", provenance)
        assert "statsmodels" in code

    def test_generate_scaffold_survival_imports(self, generator, provenance):
        provenance.add_step(
            "survival_analysis", "KM curves",
            analysis_type="survival",
            parameters={"time_col": "los", "event_col": "mortality"},
        )
        python_code = generator.generate_scaffold("python", provenance)
        assert "lifelines" in python_code
        assert "KaplanMeierFitter" in python_code

        # Reset provenance for R (steps already added)
        r_prov = AnalysisProvenance(
            study_name="ICU Mortality Study",
            data_source="./data/synthetic",
            schema_name="clif",
        )
        r_prov.add_step("survival_analysis", "KM curves", analysis_type="survival")
        r_code = generator.generate_scaffold("r", r_prov)
        assert "library(survival)" in r_code
        assert "library(survminer)" in r_code

    def test_generate_backward_compat(self, generator, provenance):
        """generate() still works and delegates to generate_scaffold()."""
        code = generator.generate(
            "python", "regression", provenance,
            outcome="mortality", predictors=["age"],
        )
        assert "import" in code
        assert "ICU Mortality Study" in code

    def test_generate_backward_compat_r(self, generator, provenance):
        code = generator.generate(
            "r", "regression", provenance,
            outcome="mortality", predictors=["age"],
        )
        assert "library(tidyverse)" in code
        assert "ICU Mortality Study" in code

    def test_unsupported_language_raises(self, generator, provenance):
        with pytest.raises(ValueError):
            generator.generate_scaffold("julia", provenance)

    def test_unsupported_language_via_generate(self, generator, provenance):
        with pytest.raises(ValueError):
            generator.generate("julia", "table1", provenance)

    def test_python_cohort_filter(self, generator, provenance):
        provenance.add_step("table1", "Stats", analysis_type="descriptive")
        code = generator.generate_scaffold("python", provenance)
        assert "age" in code
        assert ">= 18" in code

    def test_r_cohort_filter(self, generator, provenance):
        provenance.add_step("table1", "Stats", analysis_type="descriptive")
        code = generator.generate_scaffold("r", provenance)
        assert "filter" in code

    def test_list_templates(self, generator):
        templates = generator.list_templates()
        assert isinstance(templates, list)
        assert len(templates) >= 2  # scaffold.py.j2 and scaffold.R.j2

    def test_scaffold_loads_only_used_tables(self, generator):
        prov = AnalysisProvenance(
            study_name="Focused Study",
            data_source="./data",
            schema_name="clif",
            tables_used=["hospitalization", "vitals"],
        )
        prov.add_step("table1", "Stats", analysis_type="descriptive")
        code = generator.generate_scaffold("python", prov)
        assert '"hospitalization"' in code
        assert '"vitals"' in code
        assert '"medication_admin_continuous"' not in code

    def test_scaffold_parquet_format(self, generator):
        prov = AnalysisProvenance(
            study_name="Parquet Study",
            data_source="./data",
            schema_name="clif",
            data_format="parquet",
            tables_used=["hospitalization"],
        )
        prov.add_step("table1", "Stats", analysis_type="descriptive")
        code = generator.generate_scaffold("python", prov)
        assert "read_parquet" in code
        assert ".parquet" in code

    def test_scaffold_prediction_imports(self, generator, provenance):
        provenance.add_step("predict", "ML model", analysis_type="prediction")
        code = generator.generate_scaffold("python", provenance)
        assert "sklearn" in code

    def test_scaffold_stub_functions(self, generator, provenance):
        """Each analysis step gets a stub function with TODO marker."""
        provenance.add_step("table1", "Descriptive", analysis_type="descriptive")
        provenance.add_step("regression_analysis", "Logistic", analysis_type="regression")
        code = generator.generate_scaffold("python", provenance)
        assert "def table1(" in code
        assert "def regression_analysis(" in code
        assert "NotImplementedError" in code

    def test_generate_adds_missing_step(self, generator, provenance):
        """generate() auto-adds a step if analysis_type not in provenance."""
        assert len(provenance.steps) == 0
        generator.generate("python", "regression", provenance, outcome="mortality")
        assert len(provenance.steps) == 1
        assert provenance.steps[0].analysis_type == "regression"

    def test_generate_does_not_duplicate_step(self, generator, provenance):
        """generate() does not add a step if analysis_type already exists."""
        provenance.add_step("reg", "Regression", analysis_type="regression")
        generator.generate("python", "regression", provenance, outcome="mortality")
        regression_steps = [s for s in provenance.steps if s.analysis_type == "regression"]
        assert len(regression_steps) == 1
