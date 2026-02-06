"""Integration test: privacy boundary ensures no PHI leaks."""
import pandas as pd
import numpy as np
import pytest

from cablecar.privacy.guard import PrivacyGuard
from cablecar.privacy.phi_detector import PHIDetector
from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition
from cablecar.analysis.descriptive import DescriptiveAnalysis
from cablecar.analysis.regression import RegressionAnalysis


@pytest.fixture
def phi_laden_store():
    """A DataStore with known PHI planted in the data."""
    rng = np.random.default_rng(99)
    n = 30

    hosp_df = pd.DataFrame({
        "hospitalization_id": [f"H{i:04d}" for i in range(n)],
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "patient_name": [f"John Doe {i}" for i in range(n)],
        "ssn": [f"{100+i}-{45+i%10}-{6789+i}" for i in range(n)],
        "email": [f"patient{i}@hospital.org" for i in range(n)],
        "age_at_admission": rng.normal(65, 15, n).round(1),
        "hospital_mortality": rng.binomial(1, 0.2, n),
        "discharge_category": rng.choice(["Home", "SNF"], n),
    })

    store = DataStore()
    store.tables = {"hospitalization": hosp_df}
    return store


class TestPrivacyBoundary:
    def test_guard_never_returns_raw_rows(self, privacy_guard, mini_dataframes):
        """sanitize_for_llm should return summary, never raw patient rows."""
        result = privacy_guard.sanitize_for_llm(mini_dataframes["patient"])
        data = result["data"]
        # Should have shape info
        assert "shape" in data
        # Should NOT have any raw row data
        assert "columns" in data
        for col_info in data["columns"]:
            assert "values" not in col_info

    def test_phi_redacted_in_string(self, privacy_guard):
        phi_string = "Patient SSN: 123-45-6789, email: john@hospital.org"
        result = privacy_guard.sanitize_for_llm(phi_string)
        output = str(result["data"])
        assert "123-45-6789" not in output
        assert "john@hospital.org" not in output

    def test_phi_in_analysis_results(self, phi_laden_store):
        """Analysis results should not contain PHI even if data has it."""
        builder = CohortBuilder(phi_laden_store)
        cohort = builder.build(CohortDefinition(name="all"))

        analysis = DescriptiveAnalysis(cohort)
        result = analysis.run(variables=["age_at_admission", "hospital_mortality"])

        guard = PrivacyGuard()
        safe = guard.sanitize_for_llm(result.to_dict(), context="descriptive")

        # Convert entire result to string and check for PHI
        output_str = str(safe)
        detector = PHIDetector()
        assert not detector.contains_phi(output_str), \
            f"PHI found in sanitized output: {detector.scan_text(output_str)}"

    def test_small_cells_suppressed_in_analysis(self):
        """Groups with fewer than min_cell_size should be suppressed."""
        guard = PrivacyGuard()
        # Simulate a count table with small cells
        data = {
            "group_a": {"count": 100, "mean_age": 65.2},
            "group_b": {"count": 5, "mean_age": 70.1},  # should be suppressed
            "group_c": {"count": 15, "mean_age": 62.3},
        }
        safe = guard.sanitize_for_llm(data)
        assert safe["data"]["group_b"]["count"] == "<suppressed>"
        assert safe["data"]["group_a"]["count"] == 100

    def test_nested_phi_redaction(self, privacy_guard):
        """PHI should be redacted even in deeply nested structures."""
        data = {
            "results": {
                "patients": {
                    "note": "Contact john.doe@hospital.org for follow-up",
                    "id": "P001",
                }
            }
        }
        safe = privacy_guard.sanitize_for_llm(data)
        output_str = str(safe["data"])
        assert "john.doe@hospital.org" not in output_str

    def test_regression_results_safe(self, phi_laden_store):
        """Regression results should only contain coefficients, not patient data."""
        builder = CohortBuilder(phi_laden_store)
        cohort = builder.build(CohortDefinition(name="all"))

        reg = RegressionAnalysis(cohort)
        result = reg.run(
            outcome="hospital_mortality",
            predictors=["age_at_admission"],
            model_type="logistic",
        )

        guard = PrivacyGuard()
        safe = guard.sanitize_for_llm(result.to_dict())

        output_str = str(safe)
        detector = PHIDetector()
        assert not detector.contains_phi(output_str)

    def test_cohort_summary_safe(self, mini_cohort):
        """Cohort summary should not expose raw data."""
        summary = mini_cohort.summary()
        summary_str = str(summary)
        # Summary should contain counts and table names, not patient records
        assert "n_subjects" in summary
        assert "tables" in summary
        detector = PHIDetector()
        assert not detector.contains_phi(summary_str)
