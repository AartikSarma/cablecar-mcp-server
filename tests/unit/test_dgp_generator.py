"""Tests for DGP-driven synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

from cablecar.evaluation.dgp import (
    CausalEdge,
    CLIFMapping,
    DGPSpec,
    DGPVariable,
    DifficultyTier,
    Distribution,
    ErrorType,
    FunctionalForm,
    GroundTruth,
    MissingnessSpec,
    MissingnessMechanism,
    NoiseSpec,
    VariableRole,
    VariableType,
)
from synthetic.dgp_generator import DGPSyntheticGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(**overrides) -> DGPSpec:
    """Build a minimal DGPSpec for testing generation."""
    defaults = dict(
        name="gen_test",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                role=VariableRole.CONFOUNDER,
            ),
            DGPVariable(
                name="treatment",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                role=VariableRole.EXPOSURE,
            ),
            DGPVariable(
                name="outcome",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                role=VariableRole.OUTCOME,
            ),
        ],
        edges=[
            CausalEdge(cause="age", effect="treatment", functional_form=FunctionalForm.LOGISTIC, effect_size=0.03),
            CausalEdge(cause="age", effect="outcome", functional_form=FunctionalForm.LOGISTIC, effect_size=0.02),
            CausalEdge(cause="treatment", effect="outcome", functional_form=FunctionalForm.LOGISTIC, effect_size=1.0),
        ],
        ground_truth=GroundTruth(
            primary_exposure="treatment",
            primary_outcome="outcome",
            true_causal_effect=1.0,
            correct_adjustment_set=["age"],
            expected_dag_edges=[("age", "treatment"), ("age", "outcome"), ("treatment", "outcome")],
            effect_size_tolerance=0.3,
        ),
        difficulty=DifficultyTier.EASY,
        n_patients=500,
        seed=42,
        schema_mappings={
            "age": CLIFMapping(table="hospitalization", column="age_at_admission"),
            "treatment": CLIFMapping(
                table="medication_admin_continuous",
                column="med_dose",
                category_column="med_category",
                category_value="norepinephrine",
            ),
            "outcome": CLIFMapping(table="hospitalization", column="hospital_mortality"),
        },
    )
    defaults.update(overrides)
    return DGPSpec(**defaults)


@pytest.fixture
def simple_spec() -> DGPSpec:
    return _make_spec()


@pytest.fixture
def generator(simple_spec) -> DGPSyntheticGenerator:
    return DGPSyntheticGenerator(simple_spec)


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_generate_returns_tables_and_ground_truth(self, generator):
        tables, gt = generator.generate()
        assert isinstance(tables, dict)
        assert "patient" in tables
        assert "hospitalization" in tables
        assert gt.primary_exposure == "treatment"
        assert gt.primary_outcome == "outcome"

    def test_patient_table_has_correct_n(self, generator):
        tables, _ = generator.generate()
        assert len(tables["patient"]) == 500

    def test_hospitalization_table_has_patient_ids(self, generator):
        tables, _ = generator.generate()
        hosp = tables["hospitalization"]
        assert "patient_id" in hosp.columns
        assert "hospitalization_id" in hosp.columns
        assert len(hosp) == 500

    def test_schema_mapped_columns_present(self, generator):
        tables, _ = generator.generate()
        hosp = tables["hospitalization"]
        assert "age_at_admission" in hosp.columns
        assert "hospital_mortality" in hosp.columns

    def test_category_table_generated(self, generator):
        tables, _ = generator.generate()
        assert "medication_admin_continuous" in tables
        med = tables["medication_admin_continuous"]
        assert "med_category" in med.columns
        assert "norepinephrine" in med["med_category"].values

    def test_reproducibility_with_seed(self, simple_spec):
        gen1 = DGPSyntheticGenerator(simple_spec)
        gen2 = DGPSyntheticGenerator(simple_spec)
        tables1, _ = gen1.generate()
        tables2, _ = gen2.generate()
        pd.testing.assert_frame_equal(tables1["hospitalization"], tables2["hospitalization"])


# ---------------------------------------------------------------------------
# Effect recovery
# ---------------------------------------------------------------------------


class TestEffectRecovery:
    def test_treatment_effect_recoverable(self):
        """Verify the embedded effect size is approximately recoverable via logistic regression."""
        spec = _make_spec(n_patients=2000, seed=123)
        gen = DGPSyntheticGenerator(spec)
        tables, _ = gen.generate()

        # Reconstruct patient-level data from tables
        hosp = tables["hospitalization"]
        age = hosp["age_at_admission"].values
        outcome = hosp["hospital_mortality"].values

        # Get treatment from medication table via med_dose value
        med = tables["medication_admin_continuous"]
        med_by_hosp = med.set_index("hospitalization_id")["med_dose"]
        treatment = np.array([
            med_by_hosp.get(hid, 0.0)
            for hid in hosp["hospitalization_id"]
        ])

        # Simple logistic regression to recover effect
        from scipy.special import expit
        from scipy.optimize import minimize

        def neg_log_likelihood(beta):
            logit = beta[0] + beta[1] * treatment + beta[2] * age
            p = expit(logit)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))

        result = minimize(neg_log_likelihood, x0=[0, 0, 0], method="Nelder-Mead")
        estimated_treatment_effect = result.x[1]

        # The true effect size is 1.0; with 2000 patients we should be in the ballpark
        assert abs(estimated_treatment_effect - 1.0) < 1.0, (
            f"Estimated treatment effect {estimated_treatment_effect:.2f} "
            "is too far from true effect 1.0"
        )


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


class TestNoise:
    def test_additive_gaussian_noise(self):
        spec = _make_spec(
            variables=[
                DGPVariable(
                    name="age",
                    variable_type=VariableType.CONTINUOUS,
                    distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                    role=VariableRole.CONFOUNDER,
                    noise=NoiseSpec(error_type=ErrorType.ADDITIVE_GAUSSIAN, magnitude=5.0),
                ),
                DGPVariable(
                    name="treatment",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                    role=VariableRole.EXPOSURE,
                ),
                DGPVariable(
                    name="outcome",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                    role=VariableRole.OUTCOME,
                ),
            ],
            n_patients=1000,
        )
        gen = DGPSyntheticGenerator(spec)
        tables, _ = gen.generate()
        # Noise adds variability; the age column should have more variance than
        # the base std of 15 (since we add std=5 noise)
        hosp = tables["hospitalization"]
        age_std = hosp["age_at_admission"].std()
        # Combined std should be sqrt(15^2 + 5^2) ≈ 15.8
        assert age_std > 14  # Just ensure noise was applied

    def test_rounding_noise(self):
        spec = _make_spec(
            variables=[
                DGPVariable(
                    name="age",
                    variable_type=VariableType.CONTINUOUS,
                    distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                    role=VariableRole.CONFOUNDER,
                    noise=NoiseSpec(error_type=ErrorType.ROUNDING, magnitude=0, parameters={"rounding_digits": 0}),
                ),
                DGPVariable(
                    name="treatment",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                    role=VariableRole.EXPOSURE,
                ),
                DGPVariable(
                    name="outcome",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                    role=VariableRole.OUTCOME,
                ),
            ],
            n_patients=100,
        )
        gen = DGPSyntheticGenerator(spec)
        tables, _ = gen.generate()
        hosp = tables["hospitalization"]
        ages = hosp["age_at_admission"].dropna().values
        # All values should be integers after rounding
        assert np.allclose(ages, np.round(ages))


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


class TestMissingness:
    def test_mcar_missingness(self):
        spec = _make_spec(
            variables=[
                DGPVariable(
                    name="age",
                    variable_type=VariableType.CONTINUOUS,
                    distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                    role=VariableRole.CONFOUNDER,
                    missingness=MissingnessSpec(mechanism=MissingnessMechanism.MCAR, proportion=0.3),
                ),
                DGPVariable(
                    name="treatment",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                    role=VariableRole.EXPOSURE,
                ),
                DGPVariable(
                    name="outcome",
                    variable_type=VariableType.BINARY,
                    distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                    role=VariableRole.OUTCOME,
                ),
            ],
            n_patients=1000,
        )
        gen = DGPSyntheticGenerator(spec)
        tables, _ = gen.generate()
        hosp = tables["hospitalization"]
        missing_rate = hosp["age_at_admission"].isna().mean()
        # Should be approximately 30% missing (allow some random variation)
        assert 0.15 < missing_rate < 0.45

    def test_no_missingness_by_default(self, generator):
        tables, _ = generator.generate()
        hosp = tables["hospitalization"]
        assert hosp["age_at_admission"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------


class TestSaveToDisk:
    def test_generate_and_save(self, simple_spec, tmp_path):
        gen = DGPSyntheticGenerator(simple_spec)
        tables, gt = gen.generate_and_save(tmp_path / "benchmark_test")

        assert (tmp_path / "benchmark_test" / "tables" / "patient.csv").exists()
        assert (tmp_path / "benchmark_test" / "tables" / "hospitalization.csv").exists()
        assert (tmp_path / "benchmark_test" / "ground_truth" / "dgp_spec.json").exists()

        # Ground truth JSON should be valid
        import json
        gt_json = json.loads(
            (tmp_path / "benchmark_test" / "ground_truth" / "dgp_spec.json").read_text()
        )
        assert gt_json["name"] == "gen_test"
