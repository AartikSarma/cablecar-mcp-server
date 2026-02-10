"""Standard DGP scenarios for benchmarking.

Provides easy / medium / hard DGP specifications that define clinically
realistic causal structures at increasing complexity. Used by both
integration tests and the benchmark runner script.
"""

from __future__ import annotations

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
    MissingnessMechanism,
    MissingnessSpec,
    NoiseSpec,
    VariableRole,
    VariableType,
)


def easy_spec() -> DGPSpec:
    """Easy: Vasopressor -> Mortality, confounded by age.

    3 variables, large effect (log-OR = 1.0), linear/logistic, no
    missingness, no noise.
    """
    return DGPSpec(
        name="easy_vasopressor_mortality",
        description="Vasopressor use increases mortality, confounded by age.",
        difficulty=DifficultyTier.EASY,
        n_patients=2000,
        seed=42,
        vignette=(
            "You are analyzing ICU data to determine whether vasopressor use "
            "increases in-hospital mortality. Age is a known confounder."
        ),
        domain_hint="ICU cohort; investigate vasopressor-mortality relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 65, "std": 12}),
                role=VariableRole.CONFOUNDER,
                description="Patient age in years",
            ),
            DGPVariable(
                name="vasopressors",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.3}),
                role=VariableRole.EXPOSURE,
                description="Whether patient received vasopressors",
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.15}),
                role=VariableRole.OUTCOME,
                description="In-hospital mortality",
            ),
        ],
        edges=[
            CausalEdge(
                cause="age",
                effect="vasopressors",
                functional_form=FunctionalForm.LINEAR,
                effect_size=0.03,
            ),
            CausalEdge(
                cause="age",
                effect="mortality",
                functional_form=FunctionalForm.LINEAR,
                effect_size=0.02,
            ),
            CausalEdge(
                cause="vasopressors",
                effect="mortality",
                functional_form=FunctionalForm.LOGISTIC,
                effect_size=1.0,
            ),
        ],
        ground_truth=GroundTruth(
            primary_exposure="vasopressors",
            primary_outcome="mortality",
            true_causal_effect=1.0,
            correct_adjustment_set=["age"],
            expected_dag_edges=[
                ("age", "vasopressors"),
                ("age", "mortality"),
                ("vasopressors", "mortality"),
            ],
            expected_hypotheses=[
                "Vasopressor use increases mortality risk in ICU patients"
            ],
            expected_null_findings=[],
            effect_size_tolerance=0.3,
        ),
        schema_mappings={
            "age": CLIFMapping(
                table="patient",
                column="age",
            ),
            "vasopressors": CLIFMapping(
                table="medication_admin_continuous",
                column="med_value",
                category_column="med_category",
                category_value="vasopressors",
            ),
            "mortality": CLIFMapping(
                table="hospitalization",
                column="discharge_disposition",
            ),
        },
    )


def medium_spec() -> DGPSpec:
    """Medium: Sepsis -> AKI, confounded by age + severity, 1 mediator, 1 distractor.

    6 variables, moderate effect (log-OR = 0.5), MAR missingness on lactate (15%).
    """
    return DGPSpec(
        name="medium_sepsis_aki",
        description="Sepsis increases AKI risk; lactate mediates part of the effect.",
        difficulty=DifficultyTier.MEDIUM,
        n_patients=2000,
        seed=123,
        vignette=(
            "Analyze ICU data to determine whether sepsis increases the risk "
            "of acute kidney injury. Consider that severity of illness and age "
            "may confound the relationship, and lactate may be on the causal "
            "pathway."
        ),
        domain_hint="ICU cohort; investigate sepsis-AKI relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 62, "std": 14}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
            ),
            DGPVariable(
                name="severity",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                role=VariableRole.CONFOUNDER,
                description="Illness severity score (standardized)",
            ),
            DGPVariable(
                name="sepsis",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.25}),
                role=VariableRole.EXPOSURE,
                description="Sepsis diagnosis",
            ),
            DGPVariable(
                name="lactate",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 2.0, "std": 1.0}),
                role=VariableRole.MEDIATOR,
                description="Serum lactate level",
                missingness=MissingnessSpec(
                    mechanism=MissingnessMechanism.MAR,
                    proportion=0.15,
                    conditioning_variables=["severity"],
                ),
            ),
            DGPVariable(
                name="AKI",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.10}),
                role=VariableRole.OUTCOME,
                description="Acute kidney injury",
            ),
            DGPVariable(
                name="glucose",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 130, "std": 40}),
                role=VariableRole.DISTRACTOR,
                description="Blood glucose (unrelated to outcome)",
            ),
        ],
        edges=[
            CausalEdge(cause="age", effect="sepsis", functional_form=FunctionalForm.LINEAR, effect_size=0.02),
            CausalEdge(cause="severity", effect="sepsis", functional_form=FunctionalForm.LINEAR, effect_size=0.4),
            CausalEdge(cause="age", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.01),
            CausalEdge(cause="severity", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            CausalEdge(cause="sepsis", effect="lactate", functional_form=FunctionalForm.LINEAR, effect_size=1.5),
            CausalEdge(cause="sepsis", effect="AKI", functional_form=FunctionalForm.LOGISTIC, effect_size=0.5),
            CausalEdge(cause="lactate", effect="AKI", functional_form=FunctionalForm.LINEAR, effect_size=0.2),
        ],
        ground_truth=GroundTruth(
            primary_exposure="sepsis",
            primary_outcome="AKI",
            true_causal_effect=0.5,
            correct_adjustment_set=["age", "severity"],
            expected_dag_edges=[
                ("age", "sepsis"),
                ("severity", "sepsis"),
                ("age", "AKI"),
                ("severity", "AKI"),
                ("sepsis", "lactate"),
                ("sepsis", "AKI"),
                ("lactate", "AKI"),
            ],
            expected_hypotheses=["Sepsis increases risk of AKI"],
            expected_null_findings=["glucose"],
            effect_size_tolerance=0.25,
        ),
        schema_mappings={
            "age": CLIFMapping(table="patient", column="age"),
            "severity": CLIFMapping(
                table="patient_assessments",
                column="numerical_value",
                category_column="assessment_category",
                category_value="severity",
            ),
            "sepsis": CLIFMapping(
                table="hospitalization",
                column="sepsis_flag",
            ),
            "lactate": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="lactate",
            ),
            "AKI": CLIFMapping(
                table="hospitalization",
                column="aki_flag",
            ),
            "glucose": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="glucose",
            ),
        },
    )


def hard_spec() -> DGPSpec:
    """Hard: Ventilation -> Mortality with collider, weak signal, MNAR, noise.

    10 variables, weak effect (log-OR = 0.2), MNAR missingness on creatinine,
    additive noise on age + creatinine, threshold on severity, interaction
    between ventilation and sedation.
    """
    return DGPSpec(
        name="hard_ventilation_mortality",
        description=(
            "Mechanical ventilation has a weak effect on mortality. ICU_LOS is "
            "a collider (caused by ventilation and severity). Creatinine has "
            "MNAR missingness and noise. Sedation interacts with ventilation."
        ),
        difficulty=DifficultyTier.HARD,
        n_patients=3000,
        seed=999,
        vignette=(
            "Analyze ICU data to determine whether mechanical ventilation "
            "affects in-hospital mortality. Beware of collider bias from ICU "
            "length of stay. Creatinine data may be missing not at random."
        ),
        domain_hint="ICU cohort; investigate ventilation-mortality relationship.",
        variables=[
            DGPVariable(
                name="age",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 60, "std": 15}),
                role=VariableRole.CONFOUNDER,
                description="Patient age",
                noise=NoiseSpec(error_type=ErrorType.ADDITIVE_GAUSSIAN, magnitude=2.0),
            ),
            DGPVariable(
                name="severity",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 0, "std": 1}),
                role=VariableRole.CONFOUNDER,
                description="Illness severity score",
            ),
            DGPVariable(
                name="creatinine",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 1.2, "std": 0.5}),
                role=VariableRole.CONFOUNDER,
                description="Serum creatinine",
                noise=NoiseSpec(error_type=ErrorType.ADDITIVE_GAUSSIAN, magnitude=0.15),
                missingness=MissingnessSpec(
                    mechanism=MissingnessMechanism.MNAR,
                    proportion=0.20,
                ),
            ),
            DGPVariable(
                name="ventilation",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.35}),
                role=VariableRole.EXPOSURE,
                description="Mechanical ventilation",
            ),
            DGPVariable(
                name="sedation",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.40}),
                role=VariableRole.CONFOUNDER,
                description="Sedation administered",
            ),
            DGPVariable(
                name="mortality",
                variable_type=VariableType.BINARY,
                distribution=Distribution(family="bernoulli", params={"p": 0.12}),
                role=VariableRole.OUTCOME,
                description="In-hospital mortality",
            ),
            DGPVariable(
                name="ICU_LOS",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 5, "std": 3}),
                role=VariableRole.COLLIDER,
                description="ICU length of stay (collider: caused by ventilation + severity)",
            ),
            DGPVariable(
                name="heart_rate",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 85, "std": 15}),
                role=VariableRole.DISTRACTOR,
                description="Heart rate (unrelated to exposure/outcome path)",
            ),
            DGPVariable(
                name="wbc",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 10, "std": 4}),
                role=VariableRole.DISTRACTOR,
                description="White blood cell count (distractor)",
            ),
            DGPVariable(
                name="platelets",
                variable_type=VariableType.CONTINUOUS,
                distribution=Distribution(family="normal", params={"mean": 250, "std": 80}),
                role=VariableRole.DISTRACTOR,
                description="Platelet count (distractor)",
            ),
        ],
        edges=[
            # Confounders -> exposure
            CausalEdge(cause="age", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.01),
            CausalEdge(cause="severity", effect="ventilation", functional_form=FunctionalForm.THRESHOLD, effect_size=0.8, parameters={"threshold": 0.5, "effect_above": 0.8, "effect_below": 0.1}),
            CausalEdge(cause="creatinine", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            CausalEdge(cause="sedation", effect="ventilation", functional_form=FunctionalForm.LINEAR, effect_size=0.5),
            # Confounders -> outcome
            CausalEdge(cause="age", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.02),
            CausalEdge(cause="severity", effect="mortality", functional_form=FunctionalForm.THRESHOLD, effect_size=0.6, parameters={"threshold": 0.5, "effect_above": 0.6, "effect_below": 0.0}),
            CausalEdge(cause="creatinine", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.4),
            CausalEdge(cause="sedation", effect="mortality", functional_form=FunctionalForm.LINEAR, effect_size=0.3),
            # Exposure -> outcome (weak signal)
            CausalEdge(cause="ventilation", effect="mortality", functional_form=FunctionalForm.LOGISTIC, effect_size=0.2),
            # Interaction: ventilation * sedation -> mortality
            CausalEdge(cause="ventilation", effect="mortality", functional_form=FunctionalForm.INTERACTION, effect_size=0.15, parameters={"interaction_variable": "sedation"}),
            # Collider: ventilation + severity -> ICU_LOS
            CausalEdge(cause="ventilation", effect="ICU_LOS", functional_form=FunctionalForm.LINEAR, effect_size=3.0),
            CausalEdge(cause="severity", effect="ICU_LOS", functional_form=FunctionalForm.LINEAR, effect_size=2.0),
        ],
        ground_truth=GroundTruth(
            primary_exposure="ventilation",
            primary_outcome="mortality",
            true_causal_effect=0.2,
            correct_adjustment_set=["age", "severity", "creatinine", "sedation"],
            expected_dag_edges=[
                ("age", "ventilation"),
                ("severity", "ventilation"),
                ("creatinine", "ventilation"),
                ("sedation", "ventilation"),
                ("age", "mortality"),
                ("severity", "mortality"),
                ("creatinine", "mortality"),
                ("sedation", "mortality"),
                ("ventilation", "mortality"),
                ("ventilation", "ICU_LOS"),
                ("severity", "ICU_LOS"),
            ],
            expected_hypotheses=[
                "Mechanical ventilation increases mortality risk"
            ],
            expected_null_findings=["heart_rate", "wbc", "platelets"],
            effect_size_tolerance=0.4,
        ),
        schema_mappings={
            "age": CLIFMapping(table="patient", column="age"),
            "severity": CLIFMapping(
                table="patient_assessments",
                column="numerical_value",
                category_column="assessment_category",
                category_value="severity",
            ),
            "creatinine": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="creatinine",
            ),
            "ventilation": CLIFMapping(
                table="respiratory_support",
                column="vent_value",
                category_column="device_category",
                category_value="ventilation",
            ),
            "sedation": CLIFMapping(
                table="medication_admin_continuous",
                column="med_value",
                category_column="med_category",
                category_value="sedation",
            ),
            "mortality": CLIFMapping(
                table="hospitalization",
                column="discharge_disposition",
            ),
            "ICU_LOS": CLIFMapping(
                table="hospitalization",
                column="icu_los",
            ),
            "heart_rate": CLIFMapping(
                table="vitals",
                column="vital_value",
                category_column="vital_category",
                category_value="heart_rate",
            ),
            "wbc": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="wbc",
            ),
            "platelets": CLIFMapping(
                table="labs",
                column="lab_value",
                category_column="lab_category",
                category_value="platelets",
            ),
        },
    )


ALL_SCENARIOS = {
    "easy": easy_spec,
    "medium": medium_spec,
    "hard": hard_spec,
}
