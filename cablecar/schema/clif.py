"""CLIF v2.1.0 schema definition.

The Common Longitudinal ICU Format (CLIF) is a standardized data model for
representing clinical data from intensive care units across multiple institutions.
"""

from __future__ import annotations

from cablecar.schema.base import ColumnSpec, SchemaDefinition, TableSpec


def get_clif_schema() -> SchemaDefinition:
    """Return the complete CLIF v2.1.0 schema definition."""

    patient = TableSpec(
        name="patient",
        columns=[
            ColumnSpec(
                name="patient_id",
                dtype="str",
                required=True,
                description="Unique patient identifier",
                is_phi=True,
            ),
            ColumnSpec(
                name="race_category",
                dtype="str",
                required=False,
                description="Patient race category",
            ),
            ColumnSpec(
                name="ethnicity_category",
                dtype="str",
                required=False,
                description="Patient ethnicity category",
            ),
            ColumnSpec(
                name="sex_category",
                dtype="str",
                required=False,
                description="Patient sex category",
            ),
        ],
        primary_key=["patient_id"],
        description="Patient demographics table",
    )

    hospitalization = TableSpec(
        name="hospitalization",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Unique hospitalization identifier",
            ),
            ColumnSpec(
                name="patient_id",
                dtype="str",
                required=True,
                description="Reference to patient",
                is_phi=True,
            ),
            ColumnSpec(
                name="admission_dttm",
                dtype="str",
                required=True,
                description="Admission datetime",
            ),
            ColumnSpec(
                name="discharge_dttm",
                dtype="str",
                required=True,
                description="Discharge datetime",
            ),
            ColumnSpec(
                name="age_at_admission",
                dtype="float",
                required=False,
                description="Patient age at time of admission",
                unit="years",
            ),
            ColumnSpec(
                name="admission_type_name",
                dtype="str",
                required=False,
                description="Type of admission (e.g., emergency, elective)",
            ),
        ],
        primary_key=["hospitalization_id"],
        foreign_keys={"patient_id": "patient.patient_id"},
        description="Hospitalization encounters table",
    )

    adt = TableSpec(
        name="adt",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="in_dttm",
                dtype="str",
                required=True,
                description="Transfer-in datetime",
            ),
            ColumnSpec(
                name="out_dttm",
                dtype="str",
                required=True,
                description="Transfer-out datetime",
            ),
            ColumnSpec(
                name="location_category",
                dtype="str",
                required=True,
                description="Location category for the ADT event",
                allowed_values=["ICU", "Ward", "ED", "OR", "Other"],
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Admission, discharge, and transfer events table",
    )

    vitals = TableSpec(
        name="vitals",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="recorded_dttm",
                dtype="str",
                required=True,
                description="Datetime the vital sign was recorded",
            ),
            ColumnSpec(
                name="vital_category",
                dtype="str",
                required=True,
                description="Category of vital sign measurement",
                allowed_values=[
                    "heart_rate",
                    "sbp",
                    "dbp",
                    "resp_rate",
                    "spo2",
                    "temperature",
                    "map",
                ],
            ),
            ColumnSpec(
                name="vital_value",
                dtype="float",
                required=True,
                description="Numeric value of the vital sign measurement",
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Vital signs measurements table",
    )

    labs = TableSpec(
        name="labs",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="lab_order_dttm",
                dtype="str",
                required=True,
                description="Datetime the lab was ordered",
            ),
            ColumnSpec(
                name="lab_collect_dttm",
                dtype="str",
                required=True,
                description="Datetime the lab specimen was collected",
            ),
            ColumnSpec(
                name="lab_result_dttm",
                dtype="str",
                required=True,
                description="Datetime the lab result was available",
            ),
            ColumnSpec(
                name="lab_category",
                dtype="str",
                required=True,
                description="Category of lab test",
                allowed_values=[
                    "lactate",
                    "creatinine",
                    "bilirubin",
                    "wbc",
                    "hemoglobin",
                    "platelets",
                    "sodium",
                    "potassium",
                    "glucose",
                    "bun",
                    "pco2",
                    "po2",
                    "ph",
                    "bicarbonate",
                    "fio2",
                ],
            ),
            ColumnSpec(
                name="lab_value",
                dtype="float",
                required=True,
                description="Numeric value of the lab result",
            ),
            ColumnSpec(
                name="lab_value_text",
                dtype="str",
                required=False,
                description="Text representation of lab result (for non-numeric values)",
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Laboratory test results table",
    )

    respiratory_support = TableSpec(
        name="respiratory_support",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="recorded_dttm",
                dtype="str",
                required=True,
                description="Datetime the respiratory support was recorded",
            ),
            ColumnSpec(
                name="device_category",
                dtype="str",
                required=True,
                description="Category of respiratory support device",
                allowed_values=[
                    "vent",
                    "hfnc",
                    "nippv",
                    "nasal_cannula",
                    "room_air",
                ],
            ),
            ColumnSpec(
                name="mode_category",
                dtype="str",
                required=False,
                description="Ventilation mode category",
            ),
            ColumnSpec(
                name="fio2_set",
                dtype="float",
                required=False,
                description="Fraction of inspired oxygen setting",
                unit="fraction",
            ),
            ColumnSpec(
                name="peep_set",
                dtype="float",
                required=False,
                description="Positive end-expiratory pressure setting",
                unit="cmH2O",
            ),
            ColumnSpec(
                name="tidal_volume_set",
                dtype="float",
                required=False,
                description="Tidal volume setting",
                unit="mL",
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Respiratory support and ventilation settings table",
    )

    medication_admin_continuous = TableSpec(
        name="medication_admin_continuous",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="admin_dttm",
                dtype="str",
                required=True,
                description="Datetime of medication administration",
            ),
            ColumnSpec(
                name="med_category",
                dtype="str",
                required=True,
                description="Category of continuous medication",
                allowed_values=[
                    "norepinephrine",
                    "vasopressin",
                    "epinephrine",
                    "phenylephrine",
                    "dopamine",
                    "dobutamine",
                    "milrinone",
                    "propofol",
                    "fentanyl",
                    "midazolam",
                    "dexmedetomidine",
                    "insulin",
                ],
            ),
            ColumnSpec(
                name="med_dose",
                dtype="float",
                required=True,
                description="Medication dose",
            ),
            ColumnSpec(
                name="med_dose_unit",
                dtype="str",
                required=True,
                description="Unit of medication dose",
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Continuous medication administration table",
    )

    patient_assessments = TableSpec(
        name="patient_assessments",
        columns=[
            ColumnSpec(
                name="hospitalization_id",
                dtype="str",
                required=True,
                description="Reference to hospitalization",
            ),
            ColumnSpec(
                name="recorded_dttm",
                dtype="str",
                required=True,
                description="Datetime the assessment was recorded",
            ),
            ColumnSpec(
                name="assessment_category",
                dtype="str",
                required=True,
                description="Category of clinical assessment",
                allowed_values=[
                    "gcs_total",
                    "gcs_eye",
                    "gcs_verbal",
                    "gcs_motor",
                    "rass",
                    "cam_icu",
                ],
            ),
            ColumnSpec(
                name="assessment_value",
                dtype="float",
                required=True,
                description="Numeric value of the assessment",
            ),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        description="Patient clinical assessments table (GCS, RASS, CAM-ICU)",
    )

    return SchemaDefinition(
        name="clif",
        version="2.1.0",
        tables={
            "patient": patient,
            "hospitalization": hospitalization,
            "adt": adt,
            "vitals": vitals,
            "labs": labs,
            "respiratory_support": respiratory_support,
            "medication_admin_continuous": medication_admin_continuous,
            "patient_assessments": patient_assessments,
        },
        description="Common Longitudinal ICU Format (CLIF) v2.1.0 - "
        "A standardized data model for representing clinical data from "
        "intensive care units across multiple institutions.",
    )
