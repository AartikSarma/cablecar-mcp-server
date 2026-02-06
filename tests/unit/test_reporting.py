"""Tests for cablecar.reporting: STROBE and TRIPOD reports."""
import pytest

from cablecar.reporting.strobe import STROBEReport, STROBE_CHECKLIST
from cablecar.reporting.tripod import TRIPODReport, TRIPOD_CHECKLIST
from cablecar.codegen.provenance import AnalysisProvenance


class TestSTROBEReport:
    def test_creation(self):
        report = STROBEReport()
        completion = report.get_completion()
        assert completion["total"] == 22
        assert completion["completed"] == 0

    def test_set_item(self):
        report = STROBEReport()
        report.set_item(1, "This is a cohort study", status="complete")
        completion = report.get_completion()
        assert completion["completed"] == 1

    def test_set_invalid_item(self):
        report = STROBEReport()
        report.set_item(999, "Nonexistent")
        completion = report.get_completion()
        assert completion["completed"] == 0

    def test_auto_populate(self):
        prov = AnalysisProvenance(
            study_name="test",
            data_source="./data",
            schema_name="clif",
            cohort_definition={
                "inclusion": [{"column": "age", "op": ">=", "value": 18}],
            },
        )
        prov.add_step("regression_analysis", "Logistic regression")
        report = STROBEReport()
        report.auto_populate(prov, cohort_summary={"n": 500})
        completion = report.get_completion()
        assert completion["drafted"] > 0

    def test_to_markdown(self):
        report = STROBEReport()
        report.set_item(1, "Cohort study", status="complete")
        md = report.to_markdown()
        assert "# STROBE Checklist" in md
        assert "Item 1" in md
        assert "Cohort study" in md

    def test_completion_percentage(self):
        report = STROBEReport()
        report.set_item(1, "Done", status="complete")
        report.set_item(2, "Drafted", status="draft")
        completion = report.get_completion()
        # 1 complete + 0.5 * 1 draft = 1.5 / 22 * 100
        expected = round(1.5 / 22 * 100, 1)
        assert completion["pct_complete"] == expected

    def test_checklist_has_22_items(self):
        assert len(STROBE_CHECKLIST) == 22


class TestTRIPODReport:
    def test_creation(self):
        report = TRIPODReport()
        completion = report.get_completion()
        assert completion["total"] == len(TRIPOD_CHECKLIST)
        assert completion["completed"] == 0

    def test_set_item(self):
        report = TRIPODReport()
        report.set_item("1", "Prediction model for ICU mortality", status="complete")
        completion = report.get_completion()
        assert completion["completed"] == 1

    def test_auto_populate_with_model_results(self):
        prov = AnalysisProvenance(
            study_name="test",
            data_source="./data",
            schema_name="clif",
        )
        model_results = {
            "performance": {"auroc": 0.82, "auprc": 0.45},
            "feature_importance": {"age": 0.3, "sofa": 0.5, "lactate": 0.2},
        }
        report = TRIPODReport()
        report.auto_populate(prov, model_results=model_results)
        completion = report.get_completion()
        assert completion["drafted"] > 0

    def test_to_markdown(self):
        report = TRIPODReport()
        report.set_item("1", "Prediction model study")
        md = report.to_markdown()
        assert "# TRIPOD+AI Checklist" in md

    def test_ai_items_present(self):
        ai_items = [item for item in TRIPOD_CHECKLIST if item.number.startswith("AI")]
        assert len(ai_items) == 4
