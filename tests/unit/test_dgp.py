"""Tests for DGP specification models and validation."""

import json

import pytest

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
    MissingnessSpec,
    MissingnessMechanism,
    NoiseSpec,
    ErrorType,
    VariableRole,
    VariableType,
    validate_dgp_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_spec(**overrides) -> DGPSpec:
    """Build a minimal valid DGPSpec (easy tier, 3 variables)."""
    defaults = dict(
        name="test_scenario",
        description="Test scenario for unit tests",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
            ),
            DGPVariable(
                name="vasopressors",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.4}),
                role=VariableRole.EXPOSURE,
                description="Vasopressor use",
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                role=VariableRole.OUTCOME,
                description="In-hospital mortality",
            ),
        ],
        edges=[
            CausalEdge(cause="age", effect="vasopressors", functional_form=FunctionalForm.LOGISTIC, effect_size=0.03),
            CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.02),
            CausalEdge(cause="vasopressors", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.8),
        ],
        ground_truth=GroundTruth(
            primary_exposure="vasopressors",
            primary_outcome="mortality",
            true_causal_effect=0.8,
            correct_adjustment_set=["age"],
            expected_dag_edges=[("age", "vasopressors"), ("age", "mortality"), ("vasopressors", "mortality")],
            expected_hypotheses=["Vasopressor use increases mortality risk"],
            effect_size_tolerance=0.2,
        ),
        difficulty=DifficultyTier.EASY,
        n_patients=100,
        seed=42,
        schema_mappings={
            "age": CLIFMapping(table="hospitalization", column="age_at_admission"),
            "vasopressors": CLIFMapping(
                table="medication_admin_continuous",
                column="med_dose",
                category_column="med_category",
                category_value="norepinephrine",
            ),
            "mortality": CLIFMapping(table="hospitalization", column="hospital_mortality"),
        },
    )
    defaults.update(overrides)
    return DGPSpec(**defaults)


@pytest.fixture
def simple_spec() -> DGPSpec:
    return _make_simple_spec()


# ---------------------------------------------------------------------------
# DGPSpec construction & serialization
# ---------------------------------------------------------------------------


class TestDGPSpecConstruction:
    def test_simple_spec_builds(self, simple_spec):
        assert simple_spec.name == "test_scenario"
        assert len(simple_spec.variables) == 3
        assert len(simple_spec.edges) == 3

    def test_variable_names(self, simple_spec):
        assert simple_spec.variable_names == ["age", "vasopressors", "mortality"]

    def test_get_variable(self, simple_spec):
        age = simple_spec.get_variable("age")
        assert age is not None
        assert age.role == VariableRole.CONFOUNDER

    def test_get_variable_missing(self, simple_spec):
        assert simple_spec.get_variable("nonexistent") is None

    def test_get_variables_by_role(self, simple_spec):
        confounders = simple_spec.get_variables_by_role(VariableRole.CONFOUNDER)
        assert len(confounders) == 1
        assert confounders[0].name == "age"

    def test_json_roundtrip(self, simple_spec):
        json_str = simple_spec.model_dump_json()
        parsed = json.loads(json_str)
        restored = DGPSpec.model_validate(parsed)
        assert restored.name == simple_spec.name
        assert len(restored.variables) == len(simple_spec.variables)
        assert len(restored.edges) == len(simple_spec.edges)

    def test_model_dump_json_indented(self, simple_spec):
        json_str = simple_spec.model_dump_json(indent=2)
        assert '"name": "test_scenario"' in json_str


# ---------------------------------------------------------------------------
# Bridge to CausalDAG
# ---------------------------------------------------------------------------


class TestCausalDAGBridge:
    def test_to_causal_dag(self, simple_spec):
        dag = simple_spec.to_causal_dag()
        assert dag.name == "test_scenario"
        assert dag.get_exposure() == "vasopressors"
        assert dag.get_outcome() == "mortality"
        assert "age" in dag.get_confounders()

    def test_dag_adjustment_set(self, simple_spec):
        dag = simple_spec.to_causal_dag()
        adj_set = dag.get_minimal_adjustment_set()
        assert "age" in adj_set

    def test_dag_is_acyclic(self, simple_spec):
        dag = simple_spec.to_causal_dag()
        summary = dag.summary()
        assert summary["is_dag"] is True


# ---------------------------------------------------------------------------
# Reverse mapping
# ---------------------------------------------------------------------------


class TestReverseMapping:
    def test_reverse_mapping_simple_column(self, simple_spec):
        rev = simple_spec.get_reverse_mapping()
        assert rev["hospitalization.age_at_admission"] == "age"

    def test_reverse_mapping_category(self, simple_spec):
        rev = simple_spec.get_reverse_mapping()
        assert rev["medication_admin_continuous.norepinephrine"] == "vasopressors"

    def test_reverse_mapping_outcome(self, simple_spec):
        rev = simple_spec.get_reverse_mapping()
        assert rev["hospitalization.hospital_mortality"] == "mortality"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_spec_has_no_errors(self, simple_spec):
        # The spec was already validated on construction; explicitly call too
        errors = validate_dgp_spec(simple_spec)
        assert errors == []

    def test_edge_references_nonexistent_variable(self):
        with pytest.raises(ValueError, match="not a defined variable"):
            _make_simple_spec(
                edges=[
                    CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
                    CausalEdge(cause="nonexistent", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
                    CausalEdge(cause="vasopressors", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.8),
                ],
            )

    def test_missing_exposure(self):
        with pytest.raises(ValueError, match="exactly 1 exposure"):
            _make_simple_spec(
                variables=[
                    DGPVariable(
                        name="age",
                        variable_type=VariableType.CONTINUOUS,
                        distribution=Distribution(family="normal", params={"mean": 65, "std": 15}),
                        role=VariableRole.CONFOUNDER,
                    ),
                    DGPVariable(
                        name="mortality",
                        variable_type=VariableType.BINARY,
                        distribution=Distribution(family="bernoulli", params={"p": 0.2}),
                        role=VariableRole.OUTCOME,
                    ),
                ],
                edges=[
                    CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
                ],
                ground_truth=GroundTruth(
                    primary_exposure="age",
                    primary_outcome="mortality",
                    true_causal_effect=0.5,
                    correct_adjustment_set=[],
                    expected_dag_edges=[("age", "mortality")],
                    effect_size_tolerance=0.2,
                ),
                schema_mappings={},
            )

    def test_multiple_outcomes(self):
        with pytest.raises(ValueError, match="exactly 1 outcome"):
            _make_simple_spec(
                variables=[
                    DGPVariable(name="x", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.5}), role=VariableRole.EXPOSURE),
                    DGPVariable(name="y1", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.3}), role=VariableRole.OUTCOME),
                    DGPVariable(name="y2", variable_type=VariableType.BINARY, distribution=Distribution(family="bernoulli", params={"p": 0.3}), role=VariableRole.OUTCOME),
                ],
                edges=[
                    CausalEdge(cause="x", effect="y1", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
                    CausalEdge(cause="x", effect="y2", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
                ],
                ground_truth=GroundTruth(
                    primary_exposure="x", primary_outcome="y1",
                    true_causal_effect=0.5, effect_size_tolerance=0.2,
                    expected_dag_edges=[("x", "y1"), ("x", "y2")],
                ),
                schema_mappings={},
            )

    def test_cyclic_graph_rejected(self):
        with pytest.raises(ValueError, match="cycle"):
            _make_simple_spec(
                variables=[
                    DGPVariable(name="a", variable_type=VariableType.CONTINUOUS, distribution=Distribution(family="normal", params={"mean": 0, "std": 1}), role=VariableRole.EXPOSURE),
                    DGPVariable(name="b", variable_type=VariableType.CONTINUOUS, distribution=Distribution(family="normal", params={"mean": 0, "std": 1}), role=VariableRole.OUTCOME),
                ],
                edges=[
                    CausalEdge(cause="a", effect="b", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
                    CausalEdge(cause="b", effect="a", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
                ],
                ground_truth=GroundTruth(
                    primary_exposure="a", primary_outcome="b",
                    true_causal_effect=0.5, effect_size_tolerance=0.2,
                    expected_dag_edges=[("a", "b"), ("b", "a")],
                ),
                schema_mappings={},
            )

    def test_invalid_clif_table(self):
        with pytest.raises(ValueError, match="unknown table"):
            _make_simple_spec(
                schema_mappings={
                    "age": CLIFMapping(table="nonexistent_table", column="age"),
                    "vasopressors": CLIFMapping(table="medication_admin_continuous", column="med_dose", category_column="med_category", category_value="norepinephrine"),
                    "mortality": CLIFMapping(table="hospitalization", column="hospital_mortality"),
                },
            )

    def test_zero_effect_for_non_distractor_rejected(self):
        with pytest.raises(ValueError, match="zero effect size"):
            _make_simple_spec(
                edges=[
                    CausalEdge(cause="age", effect="vasopressors", functional_form=FunctionalForm.LINEAR, effect_size=0.0),
                    CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
                    CausalEdge(cause="vasopressors", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.8),
                ],
            )

    def test_ground_truth_references_invalid_variable(self):
        with pytest.raises(ValueError, match="not a defined variable"):
            _make_simple_spec(
                ground_truth=GroundTruth(
                    primary_exposure="vasopressors",
                    primary_outcome="nonexistent",
                    true_causal_effect=0.8,
                    correct_adjustment_set=["age"],
                    expected_dag_edges=[("age", "vasopressors"), ("vasopressors", "mortality")],
                    effect_size_tolerance=0.2,
                ),
            )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_difficulty_values(self):
        assert DifficultyTier.EASY.value == "easy"
        assert DifficultyTier.MEDIUM.value == "medium"
        assert DifficultyTier.HARD.value == "hard"

    def test_context_level_values(self):
        assert ContextLevel.FULL_VIGNETTE.value == "full_vignette"
        assert ContextLevel.DOMAIN_HINT.value == "domain_hint"
        assert ContextLevel.BLIND.value == "blind"

    def test_variable_roles(self):
        assert VariableRole.DISTRACTOR.value == "distractor"
        assert VariableRole.INSTRUMENT.value == "instrument"

    def test_functional_forms(self):
        assert FunctionalForm.THRESHOLD.value == "threshold"
        assert FunctionalForm.INTERACTION.value == "interaction"
