"""Shared fixtures for CableCar v2 test suite."""
import pytest
import pandas as pd
import numpy as np

from cablecar.schema.base import SchemaDefinition, TableSpec, ColumnSpec
from cablecar.privacy.guard import PrivacyGuard
from cablecar.privacy.policy import PrivacyPolicy
from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition, Cohort


@pytest.fixture
def mini_schema() -> SchemaDefinition:
    """A minimal schema for testing with 3 tables."""
    return SchemaDefinition(
        name="test_schema",
        version="1.0.0",
        description="Minimal test schema",
        tables={
            "patient": TableSpec(
                name="patient",
                columns=[
                    ColumnSpec(name="patient_id", dtype="str", required=True),
                    ColumnSpec(name="age", dtype="float", required=True),
                    ColumnSpec(name="sex", dtype="str", required=True),
                    ColumnSpec(name="race", dtype="str", required=False),
                ],
                primary_key=["patient_id"],
                description="Patient demographics",
            ),
            "hospitalization": TableSpec(
                name="hospitalization",
                columns=[
                    ColumnSpec(name="hospitalization_id", dtype="str", required=True),
                    ColumnSpec(name="patient_id", dtype="str", required=True),
                    ColumnSpec(name="age_at_admission", dtype="float", required=True),
                    ColumnSpec(name="hospital_mortality", dtype="int", required=False),
                    ColumnSpec(name="discharge_category", dtype="str", required=False),
                ],
                primary_key=["hospitalization_id"],
                foreign_keys={"patient_id": "patient.patient_id"},
                description="Hospital encounters",
            ),
            "vitals": TableSpec(
                name="vitals",
                columns=[
                    ColumnSpec(name="hospitalization_id", dtype="str", required=True),
                    ColumnSpec(name="recorded_dttm", dtype="datetime", required=True),
                    ColumnSpec(name="vital_name", dtype="str", required=True),
                    ColumnSpec(name="vital_value", dtype="float", required=True),
                ],
                primary_key=[],
                foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
                description="Vital signs",
            ),
        },
    )


@pytest.fixture
def mini_dataframes() -> dict[str, pd.DataFrame]:
    """Small synthetic DataFrames matching the mini_schema."""
    rng = np.random.default_rng(42)
    n_patients = 50

    patient_ids = [f"P{i:04d}" for i in range(n_patients)]
    hosp_ids = [f"H{i:04d}" for i in range(n_patients)]

    patient_df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": rng.normal(65, 15, n_patients).round(1),
        "sex": rng.choice(["M", "F"], n_patients),
        "race": rng.choice(["White", "Black", "Asian", "Hispanic", "Other"], n_patients),
    })

    ages = patient_df["age"].values
    mortality_prob = 1 / (1 + np.exp(-0.03 * (ages - 70)))
    mortality = rng.binomial(1, mortality_prob)

    hosp_df = pd.DataFrame({
        "hospitalization_id": hosp_ids,
        "patient_id": patient_ids,
        "age_at_admission": ages,
        "hospital_mortality": mortality,
        "discharge_category": rng.choice(["Home", "SNF", "Rehab", "Hospice"], n_patients),
    })

    n_vitals = 200
    vitals_df = pd.DataFrame({
        "hospitalization_id": rng.choice(hosp_ids, n_vitals),
        "recorded_dttm": pd.date_range("2024-01-01", periods=n_vitals, freq="h"),
        "vital_name": rng.choice(["heart_rate", "sbp", "dbp", "spo2", "temp"], n_vitals),
        "vital_value": rng.normal(100, 20, n_vitals).round(1),
    })

    return {
        "patient": patient_df,
        "hospitalization": hosp_df,
        "vitals": vitals_df,
    }


@pytest.fixture
def loaded_store(mini_dataframes, mini_schema) -> DataStore:
    """A DataStore loaded with mini data and schema."""
    store = DataStore()
    store.tables = mini_dataframes
    store._schema = mini_schema
    return store


@pytest.fixture
def mini_cohort(loaded_store) -> Cohort:
    """A Cohort built from the mini data with an age filter."""
    builder = CohortBuilder(loaded_store)
    definition = CohortDefinition(
        name="adults",
        description="Adult patients (age >= 18)",
        inclusion_criteria=[
            {"column": "age_at_admission", "op": ">=", "value": 18},
        ],
    )
    return builder.build(definition)


@pytest.fixture
def privacy_guard() -> PrivacyGuard:
    """A PrivacyGuard with default policy."""
    return PrivacyGuard()


@pytest.fixture
def strict_policy() -> PrivacyPolicy:
    """A strict privacy policy with higher thresholds."""
    return PrivacyPolicy(
        min_cell_size=20,
        k_anonymity=10,
        redact_phi=True,
    )
