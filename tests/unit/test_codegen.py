"""Tests for cablecar.codegen: engine and provenance."""
import pytest

from cablecar.codegen.engine import CodeGenerator
from cablecar.codegen.provenance import AnalysisProvenance, AnalysisStep


class TestAnalysisProvenance:
    def test_creation(self):
        prov = AnalysisProvenance(study_name="test", data_source="./data")
        assert prov.study_name == "test"
        assert prov.steps == []

    def test_add_step(self):
        prov = AnalysisProvenance(study_name="test")
        prov.add_step("load", "Loaded data", {"path": "./data"})
        assert len(prov.steps) == 1
        assert prov.steps[0].step_name == "load"
        assert prov.steps[0].parameters == {"path": "./data"}

    def test_to_dict(self):
        prov = AnalysisProvenance(study_name="test", schema_name="clif")
        prov.add_step("cohort", "Defined cohort")
        d = prov.to_dict()
        assert d["study_name"] == "test"
        assert d["schema_name"] == "clif"
        assert len(d["steps"]) == 1
        assert "timestamp" in d["steps"][0]


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

    def test_generate_python_fallback(self, generator, provenance):
        code = generator.generate("python", "table1", provenance, variables=["age", "sex"])
        assert "import pandas" in code
        assert "ICU Mortality Study" in code

    def test_generate_r_fallback(self, generator, provenance):
        code = generator.generate("r", "table1", provenance, variables=["age", "sex"])
        assert "library(tidyverse)" in code
        assert "ICU Mortality Study" in code

    def test_generate_python_regression(self, generator, provenance):
        code = generator.generate(
            "python", "regression", provenance,
            outcome="mortality", predictors=["age", "sofa"], model_type="logistic",
        )
        assert "statsmodels" in code
        assert "Logit" in code

    def test_generate_r_regression(self, generator, provenance):
        code = generator.generate(
            "r", "regression", provenance,
            outcome="mortality", predictors=["age", "sofa"], model_type="logistic",
        )
        assert "glm" in code
        assert "binomial" in code

    def test_generate_python_survival(self, generator, provenance):
        code = generator.generate(
            "python", "survival", provenance,
            time_col="los", event_col="mortality", group_col="treatment",
        )
        assert "lifelines" in code
        assert "KaplanMeierFitter" in code

    def test_generate_r_survival(self, generator, provenance):
        code = generator.generate(
            "r", "survival", provenance,
            time_col="los", event_col="mortality", group_col="treatment",
        )
        assert "library(survival)" in code
        assert "survfit" in code

    def test_unsupported_language_raises(self, generator, provenance):
        with pytest.raises(ValueError):
            generator.generate("julia", "table1", provenance)

    def test_python_cohort_filter(self, generator, provenance):
        code = generator.generate("python", "table1", provenance)
        assert "cohort = cohort" in code  # filter applied

    def test_r_cohort_filter(self, generator, provenance):
        code = generator.generate("r", "table1", provenance)
        assert "filter" in code

    def test_list_templates(self, generator):
        templates = generator.list_templates()
        assert isinstance(templates, list)
