"""Schema validation for clinical datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cablecar.schema.base import SchemaDefinition

if TYPE_CHECKING:
    import pandas as pd


# Mapping from schema dtype strings to expected pandas/numpy dtype kinds.
# numpy dtype.kind reference:
#   'f' = floating-point, 'i' = signed integer, 'u' = unsigned integer,
#   'O' = object (usually strings), 'U' = Unicode string,
#   'S' = byte string, 'b' = boolean, 'M' = datetime64
_DTYPE_KIND_MAP: dict[str, set[str]] = {
    "str": {"O", "U", "S"},
    "float": {"f", "i", "u"},
    "int": {"i", "u"},
    "bool": {"b"},
    "datetime": {"M"},
}


@dataclass
class TableValidationReport:
    """Validation report for a single table."""

    table_name: str
    row_count: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    dtype_mismatches: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Overall validation result for a dataset against a schema."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    table_reports: dict[str, TableValidationReport] = field(default_factory=dict)


class SchemaValidator:
    """Validates clinical datasets against schema definitions.

    Performs structural validation including table presence, column presence,
    null checks for required columns, and basic dtype verification.
    """

    def validate(
        self,
        tables: dict[str, pd.DataFrame],
        schema: SchemaDefinition,
    ) -> ValidationResult:
        """Validate a set of tables against a schema definition.

        Args:
            tables: Dictionary mapping table names to pandas DataFrames.
            schema: The schema definition to validate against.

        Returns:
            A ValidationResult containing any errors, warnings, and
            per-table validation reports.
        """
        result = ValidationResult()

        # Check for required tables
        for table_name in schema.get_required_tables():
            if table_name not in tables:
                result.errors.append(
                    f"Required table '{table_name}' is missing from the dataset."
                )
                result.is_valid = False

        # Check for optional tables that are missing (informational)
        for table_name in schema.get_optional_tables():
            if table_name not in tables:
                result.warnings.append(
                    f"Optional table '{table_name}' is not present in the dataset."
                )

        # Warn about tables not in the schema
        schema_table_names = set(schema.tables.keys())
        for table_name in tables:
            if table_name not in schema_table_names:
                result.warnings.append(
                    f"Table '{table_name}' is present in the data but not "
                    f"defined in the '{schema.name}' schema."
                )

        # Validate each table that exists in both data and schema
        for table_name, table_spec in schema.tables.items():
            if table_name not in tables:
                continue

            df = tables[table_name]
            report = self._validate_table(df, table_name, table_spec, schema)
            result.table_reports[table_name] = report

            # Propagate table-level issues to the overall result
            if report.missing_columns:
                # Only flag as error if the missing columns are required
                required_missing = [
                    col
                    for col in report.missing_columns
                    if table_spec.get_column(col) is not None
                    and table_spec.get_column(col).required
                ]
                if required_missing:
                    result.errors.append(
                        f"Table '{table_name}' is missing required columns: "
                        f"{required_missing}"
                    )
                    result.is_valid = False

                optional_missing = [
                    col
                    for col in report.missing_columns
                    if col not in required_missing
                ]
                if optional_missing:
                    result.warnings.append(
                        f"Table '{table_name}' is missing optional columns: "
                        f"{optional_missing}"
                    )

            if report.extra_columns:
                result.warnings.append(
                    f"Table '{table_name}' has extra columns not in schema: "
                    f"{report.extra_columns}"
                )

            if report.dtype_mismatches:
                for mismatch in report.dtype_mismatches:
                    result.warnings.append(
                        f"Table '{table_name}': {mismatch}"
                    )

            # Check for nulls in required columns
            for col_name, null_count in report.null_counts.items():
                if null_count > 0:
                    col_spec = table_spec.get_column(col_name)
                    if col_spec is not None and col_spec.required:
                        result.errors.append(
                            f"Table '{table_name}', column '{col_name}': "
                            f"{null_count} null values in required column."
                        )
                        result.is_valid = False

        return result

    def _validate_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_spec: "TableSpec",
        schema: SchemaDefinition,
    ) -> TableValidationReport:
        """Validate a single DataFrame against its table specification.

        Args:
            df: The DataFrame to validate.
            table_name: Name of the table being validated.
            table_spec: The table's schema specification.
            schema: The parent schema definition (for context).

        Returns:
            A TableValidationReport with detailed findings.
        """
        from cablecar.schema.base import TableSpec  # noqa: F811

        report = TableValidationReport(
            table_name=table_name,
            row_count=len(df),
        )

        schema_columns = set(table_spec.column_names())
        data_columns = set(df.columns.tolist())

        # Missing columns
        report.missing_columns = sorted(schema_columns - data_columns)

        # Extra columns
        report.extra_columns = sorted(data_columns - schema_columns)

        # Check columns that exist in both schema and data
        common_columns = schema_columns & data_columns
        for col_name in sorted(common_columns):
            col_spec = table_spec.get_column(col_name)
            if col_spec is None:
                continue

            # Null count check
            null_count = int(df[col_name].isnull().sum())
            if null_count > 0:
                report.null_counts[col_name] = null_count

            # Basic dtype check
            actual_kind = df[col_name].dtype.kind
            expected_kinds = _DTYPE_KIND_MAP.get(col_spec.dtype)
            if expected_kinds is not None and actual_kind not in expected_kinds:
                report.dtype_mismatches.append(
                    f"Column '{col_name}' expected dtype '{col_spec.dtype}' "
                    f"(kinds {expected_kinds}) but found dtype "
                    f"'{df[col_name].dtype}' (kind '{actual_kind}')."
                )

        return report
