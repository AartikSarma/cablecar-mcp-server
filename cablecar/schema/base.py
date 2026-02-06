"""Core schema types for CableCar data definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnSpec(BaseModel):
    """Specification for a single column in a clinical data table."""

    name: str
    dtype: str
    required: bool = True
    description: str = ""
    allowed_values: list[str] | None = None
    unit: str | None = None
    is_phi: bool = False


class TableSpec(BaseModel):
    """Specification for a clinical data table."""

    name: str
    columns: list[ColumnSpec]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    def get_column(self, name: str) -> ColumnSpec | None:
        """Return a column spec by name, or None if not found."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def required_columns(self) -> list[str]:
        """Return names of all required columns."""
        return [col.name for col in self.columns if col.required]

    def column_names(self) -> list[str]:
        """Return all column names."""
        return [col.name for col in self.columns]


class SchemaDefinition(BaseModel):
    """Complete schema definition containing multiple table specifications."""

    name: str
    version: str
    tables: dict[str, TableSpec]
    description: str = ""

    def get_table(self, name: str) -> TableSpec | None:
        """Return a table spec by name, or None if not found."""
        return self.tables.get(name)

    def get_phi_columns(self) -> dict[str, list[str]]:
        """Return a mapping of table name to list of PHI column names.

        Only includes tables that have at least one PHI column.
        """
        phi: dict[str, list[str]] = {}
        for table_name, table_spec in self.tables.items():
            phi_cols = [col.name for col in table_spec.columns if col.is_phi]
            if phi_cols:
                phi[table_name] = phi_cols
        return phi

    def get_required_tables(self) -> list[str]:
        """Return names of tables that have at least one required column.

        Tables with required columns are considered required tables since
        the schema expects them to be present for a valid dataset.
        """
        return [
            name
            for name, spec in self.tables.items()
            if any(col.required for col in spec.columns)
        ]

    def get_optional_tables(self) -> list[str]:
        """Return names of tables where no columns are required.

        These tables are optional and may or may not be present in a dataset.
        """
        return [
            name
            for name, spec in self.tables.items()
            if not any(col.required for col in spec.columns)
        ]
