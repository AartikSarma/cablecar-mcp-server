"""Data Generating Process (DGP) specification schema for hypothesis discovery benchmarking.

Defines Pydantic models for specifying a complete data generating process with known
ground truth, enabling rigorous evaluation of LLM agents' ability to discover and
test hypotheses from clinical data.

The schema is designed to be schema-agnostic at its core -- variables have semantic
names and schema mappings are pluggable (CLIF, OMOP, generic CSV).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, model_validator

from cablecar.analysis.causal import CausalDAG


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VariableType(str, Enum):
    """Statistical type of a DGP variable."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    COUNT = "count"
    TIME_TO_EVENT = "time_to_event"


class VariableRole(str, Enum):
    """Causal role of a variable within the DGP."""

    EXPOSURE = "exposure"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENT = "instrument"
    DISTRACTOR = "distractor"


class FunctionalForm(str, Enum):
    """Functional form of a causal edge."""

    LINEAR = "linear"
    LOGISTIC = "logistic"
    THRESHOLD = "threshold"
    QUADRATIC = "quadratic"
    INTERACTION = "interaction"


class ErrorType(str, Enum):
    """Type of measurement error."""

    NONE = "none"
    ADDITIVE_GAUSSIAN = "additive_gaussian"
    MULTIPLICATIVE = "multiplicative"
    MISCLASSIFICATION = "misclassification"
    ROUNDING = "rounding"


class MissingnessMechanism(str, Enum):
    """Missing data mechanism."""

    NONE = "none"
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


class DifficultyTier(str, Enum):
    """Benchmark difficulty tier.

    Guidelines:
    - easy: 3-5 variables, 1 confounder, no mediators/colliders, complete data,
      large effects (Cohen's d > 0.5), linear relationships.
    - medium: 5-8 variables, 2-3 confounders, 1 mediator, MAR missingness (10-20%),
      moderate effects (d 0.2-0.5), 1-2 distractor variables.
    - hard: 8-12 variables, colliders present, MNAR missingness, measurement error,
      weak signals (d < 0.2), nonlinear relationships, 3+ distractors.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContextLevel(str, Enum):
    """How much context the discovery agent receives.

    - full_vignette: Clinical narrative + research question from the DGP spec.
    - domain_hint: One-sentence domain hint (e.g. "This dataset comes from an
      ICU cohort; investigate AKI outcomes").
    - blind: Only the data tables with no context.
    """

    FULL_VIGNETTE = "full_vignette"
    DOMAIN_HINT = "domain_hint"
    BLIND = "blind"


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------


class Distribution(BaseModel):
    """Probability distribution specification."""

    family: str = Field(
        ...,
        description="Distribution family (e.g. 'normal', 'bernoulli', 'poisson', "
        "'uniform', 'exponential', 'categorical').",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Distribution parameters (e.g. {'mean': 0, 'std': 1} for normal).",
    )


class NoiseSpec(BaseModel):
    """Measurement error specification for a variable."""

    error_type: ErrorType = ErrorType.NONE
    magnitude: float = Field(
        default=0.0,
        ge=0.0,
        description="Scale of measurement error.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional noise parameters (e.g. {'rounding_digits': 0} "
        "for rounding error, {'flip_probability': 0.05} for misclassification).",
    )


class MissingnessSpec(BaseModel):
    """Missing data specification for a variable."""

    mechanism: MissingnessMechanism = MissingnessMechanism.NONE
    proportion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of values that are missing.",
    )
    conditioning_variables: list[str] = Field(
        default_factory=list,
        description="Variables that determine missingness (for MAR/MNAR).",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional missingness parameters.",
    )


# ---------------------------------------------------------------------------
# Schema mappings (pluggable per data format)
# ---------------------------------------------------------------------------


class SchemaMapping(BaseModel):
    """Abstract base for mapping a DGP variable to a concrete data format.

    Subclasses define how a semantic variable is stored in a specific schema
    (CLIF, OMOP, generic CSV, etc.).
    """

    mapping_type: str = Field(
        ..., description="Discriminator for the mapping type."
    )


class CLIFMapping(SchemaMapping):
    """Maps a DGP variable to a column in a CLIF table.

    For category-based tables (vitals, labs, etc.), ``category_column`` and
    ``category_value`` identify the relevant rows, and ``value_column``
    holds the numeric value.
    """

    mapping_type: str = Field(default="clif", frozen=True)
    table: str = Field(..., description="CLIF table name (e.g. 'vitals', 'labs').")
    column: str = Field(
        ...,
        description="Column containing the value. For category tables this is the "
        "value column (e.g. 'vital_value').",
    )
    category_column: str | None = Field(
        default=None,
        description="Column used to filter rows by category (e.g. 'vital_category').",
    )
    category_value: str | None = Field(
        default=None,
        description="Value in category_column that identifies this variable "
        "(e.g. 'heart_rate').",
    )
    aggregation: str | None = Field(
        default=None,
        description="Aggregation to apply for time-varying data (e.g. 'mean', "
        "'max', 'first', 'last').",
    )


# ---------------------------------------------------------------------------
# Core DGP building blocks
# ---------------------------------------------------------------------------


class DGPVariable(BaseModel):
    """A variable in the data generating process."""

    name: str = Field(..., description="Semantic variable name.")
    variable_type: VariableType
    distribution: Distribution
    role: VariableRole
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    missingness: MissingnessSpec = Field(default_factory=MissingnessSpec)
    description: str = ""
    categories: list[str] | None = Field(
        default=None,
        description="Category labels for categorical variables.",
    )


class CausalEdge(BaseModel):
    """A directed causal edge in the DGP's DAG."""

    cause: str = Field(..., description="Name of the cause variable.")
    effect: str = Field(..., description="Name of the effect variable.")
    functional_form: FunctionalForm = FunctionalForm.LINEAR
    effect_size: float = Field(
        ..., description="Magnitude of the causal effect."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the functional form. E.g. "
        "{'threshold': 2.0, 'effect_above': 1.5, 'effect_below': 0.0} for "
        "threshold, {'quad_coeff': 0.1} for quadratic, "
        "{'interaction_variable': 'X'} for interaction.",
    )


class GroundTruth(BaseModel):
    """Ground truth for scoring the discovery agent's output."""

    primary_exposure: str
    primary_outcome: str
    true_causal_effect: float = Field(
        ..., description="True average causal effect of exposure on outcome."
    )
    correct_adjustment_set: list[str] = Field(
        default_factory=list,
        description="Correct set of variables to adjust for.",
    )
    expected_dag_edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Expected edges in the true causal DAG.",
    )
    expected_hypotheses: list[str] = Field(
        default_factory=list,
        description="Text descriptions of expected hypotheses.",
    )
    expected_null_findings: list[str] = Field(
        default_factory=list,
        description="Variables/relationships that should be null (distractors).",
    )
    effect_size_tolerance: float = Field(
        default=0.1,
        gt=0,
        description="Acceptable absolute deviation from true effect.",
    )
    additional_valid_findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Secondary findings that earn bonus credit.",
    )


# ---------------------------------------------------------------------------
# Top-level DGP Spec
# ---------------------------------------------------------------------------


class DGPSpec(BaseModel):
    """Complete specification of a data generating process.

    This is the top-level container that fully describes a synthetic dataset:
    the variables, their causal relationships, the ground truth, difficulty,
    and how variables map to concrete data tables.
    """

    name: str
    description: str = ""
    variables: list[DGPVariable]
    edges: list[CausalEdge]
    ground_truth: GroundTruth
    difficulty: DifficultyTier = DifficultyTier.MEDIUM
    context_level: ContextLevel = ContextLevel.FULL_VIGNETTE
    vignette: str = Field(
        default="",
        description="Clinical narrative / research question for full_vignette context.",
    )
    domain_hint: str = Field(
        default="",
        description="One-sentence domain hint for domain_hint context level.",
    )
    n_patients: int = Field(default=500, gt=0)
    seed: int = Field(default=42)
    schema_mappings: dict[str, CLIFMapping] = Field(
        default_factory=dict,
        description="Mapping from DGP variable name to concrete schema location.",
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def variable_names(self) -> list[str]:
        """Return all variable names."""
        return [v.name for v in self.variables]

    def get_variable(self, name: str) -> DGPVariable | None:
        """Look up a variable by name."""
        for v in self.variables:
            if v.name == name:
                return v
        return None

    def get_variables_by_role(self, role: VariableRole) -> list[DGPVariable]:
        """Return all variables with the given role."""
        return [v for v in self.variables if v.role == role]

    # ------------------------------------------------------------------
    # Bridge to CausalDAG
    # ------------------------------------------------------------------

    def to_causal_dag(self) -> CausalDAG:
        """Build a :class:`CausalDAG` from the DGP specification.

        Maps DGP variable roles to CausalDAG roles and adds all edges.
        """
        dag = CausalDAG(name=self.name)

        for var in self.variables:
            dag.add_variable(
                name=var.name,
                role=var.role.value,
                description=var.description,
            )

        for edge in self.edges:
            dag.add_edge(edge.cause, edge.effect)

        return dag

    # ------------------------------------------------------------------
    # Reverse mapping (data column -> DGP variable)
    # ------------------------------------------------------------------

    def get_reverse_mapping(self) -> dict[str, str]:
        """Return a mapping from data column identifiers to DGP variable names.

        For simple columns this is ``table.column -> var_name``.
        For category-based columns (e.g. vitals) the key is
        ``table.category_value`` so the scorer can translate the agent's
        data-level names back to semantic DGP names.
        """
        reverse: dict[str, str] = {}
        for var_name, mapping in self.schema_mappings.items():
            if isinstance(mapping, CLIFMapping):
                if mapping.category_value:
                    key = f"{mapping.table}.{mapping.category_value}"
                else:
                    key = f"{mapping.table}.{mapping.column}"
                reverse[key] = var_name
        return reverse

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_spec(self) -> DGPSpec:
        """Run all internal consistency checks."""
        errors = validate_dgp_spec(self)
        if errors:
            raise ValueError(
                "DGP spec validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return self


# ---------------------------------------------------------------------------
# Standalone validation function
# ---------------------------------------------------------------------------


def validate_dgp_spec(spec: DGPSpec) -> list[str]:
    """Validate a DGPSpec for internal consistency.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []
    var_names = set(spec.variable_names)

    # 1. All edge endpoints reference existing variables
    for edge in spec.edges:
        if edge.cause not in var_names:
            errors.append(f"Edge cause '{edge.cause}' is not a defined variable.")
        if edge.effect not in var_names:
            errors.append(f"Edge effect '{edge.effect}' is not a defined variable.")

    # 2. Exactly 1 exposure, exactly 1 outcome
    exposures = spec.get_variables_by_role(VariableRole.EXPOSURE)
    outcomes = spec.get_variables_by_role(VariableRole.OUTCOME)

    if len(exposures) != 1:
        errors.append(
            f"Expected exactly 1 exposure variable, found {len(exposures)}."
        )
    if len(outcomes) != 1:
        errors.append(
            f"Expected exactly 1 outcome variable, found {len(outcomes)}."
        )

    # 3. Ground truth references valid variables
    if spec.ground_truth.primary_exposure not in var_names:
        errors.append(
            f"Ground truth primary_exposure '{spec.ground_truth.primary_exposure}' "
            "is not a defined variable."
        )
    if spec.ground_truth.primary_outcome not in var_names:
        errors.append(
            f"Ground truth primary_outcome '{spec.ground_truth.primary_outcome}' "
            "is not a defined variable."
        )
    for adj_var in spec.ground_truth.correct_adjustment_set:
        if adj_var not in var_names:
            errors.append(
                f"Adjustment set variable '{adj_var}' is not a defined variable."
            )
    for src, dst in spec.ground_truth.expected_dag_edges:
        if src not in var_names:
            errors.append(
                f"Expected DAG edge source '{src}' is not a defined variable."
            )
        if dst not in var_names:
            errors.append(
                f"Expected DAG edge target '{dst}' is not a defined variable."
            )

    # 4. DAG is acyclic (only check if edges are structurally valid)
    edge_vars_valid = all(
        e.cause in var_names and e.effect in var_names for e in spec.edges
    )
    if edge_vars_valid and spec.edges:
        graph = nx.DiGraph()
        graph.add_nodes_from(var_names)
        for edge in spec.edges:
            graph.add_edge(edge.cause, edge.effect)
        if not nx.is_directed_acyclic_graph(graph):
            errors.append("The causal graph contains a cycle.")

    # 5. Schema mappings reference valid variables
    for var_name in spec.schema_mappings:
        if var_name not in var_names:
            errors.append(
                f"Schema mapping for '{var_name}' does not match any defined variable."
            )

    # 6. Schema mappings reference valid CLIF tables (if CLIF)
    _valid_clif_tables = {
        "patient",
        "hospitalization",
        "adt",
        "vitals",
        "labs",
        "respiratory_support",
        "medication_admin_continuous",
        "patient_assessments",
    }
    for var_name, mapping in spec.schema_mappings.items():
        if isinstance(mapping, CLIFMapping):
            if mapping.table not in _valid_clif_tables:
                errors.append(
                    f"CLIF mapping for '{var_name}' references unknown table "
                    f"'{mapping.table}'. Valid tables: {sorted(_valid_clif_tables)}."
                )

    # 7. Missingness proportions in [0, 1] (enforced by Pydantic, but double-check)
    for var in spec.variables:
        if not 0.0 <= var.missingness.proportion <= 1.0:
            errors.append(
                f"Variable '{var.name}' has missingness proportion "
                f"{var.missingness.proportion} outside [0, 1]."
            )

    # 8. Effect sizes are non-zero for non-distractor edges
    distractor_names = {
        v.name for v in spec.variables if v.role == VariableRole.DISTRACTOR
    }
    for edge in spec.edges:
        if edge.cause not in distractor_names and edge.effect not in distractor_names:
            if edge.effect_size == 0.0:
                errors.append(
                    f"Edge {edge.cause} -> {edge.effect} has zero effect size "
                    "but neither variable is a distractor."
                )

    return errors
