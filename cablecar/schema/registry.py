"""Schema registry for managing and discovering clinical data schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cablecar.schema.base import SchemaDefinition

if TYPE_CHECKING:
    import pandas as pd


class SchemaRegistry:
    """Registry for managing clinical data schema definitions.

    The registry holds named schemas and supports automatic schema inference
    by matching loaded data against registered schema definitions.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, SchemaDefinition] = {}

    def register(self, schema: SchemaDefinition) -> None:
        """Register a schema definition.

        Args:
            schema: The schema definition to register. The schema's name
                is used as the registry key.
        """
        self._schemas[schema.name] = schema

    def get(self, name: str) -> SchemaDefinition:
        """Retrieve a registered schema by name.

        Args:
            name: The name of the schema to retrieve.

        Returns:
            The matching schema definition.

        Raises:
            KeyError: If no schema with the given name is registered.
        """
        if name not in self._schemas:
            available = ", ".join(sorted(self._schemas.keys())) or "(none)"
            raise KeyError(
                f"Schema '{name}' not found. Available schemas: {available}"
            )
        return self._schemas[name]

    def list_schemas(self) -> list[str]:
        """Return a list of all registered schema names.

        Returns:
            Sorted list of registered schema names.
        """
        return sorted(self._schemas.keys())

    def infer_schema(self, tables: dict[str, pd.DataFrame]) -> str | None:
        """Try to match loaded data against registered schemas.

        Compares the table names and column names in the provided data
        against each registered schema. Returns the name of the schema
        with the best match, or None if no schema matches well enough.

        A schema is considered a match if at least 50% of its defined
        tables are present in the data and the average column overlap
        across matched tables is at least 50%.

        Args:
            tables: Dictionary mapping table names to DataFrames.

        Returns:
            The name of the best matching schema, or None if no schema
            matches the data sufficiently.
        """
        if not tables:
            return None

        best_schema: str | None = None
        best_score: float = 0.0

        input_table_names = set(tables.keys())

        for schema_name, schema_def in self._schemas.items():
            schema_table_names = set(schema_def.tables.keys())

            # Find tables that exist in both the data and the schema
            matched_tables = input_table_names & schema_table_names
            if not matched_tables:
                continue

            # Calculate table overlap ratio
            table_overlap = len(matched_tables) / len(schema_table_names)
            if table_overlap < 0.5:
                continue

            # Calculate average column overlap for matched tables
            column_scores: list[float] = []
            for table_name in matched_tables:
                schema_columns = set(
                    schema_def.tables[table_name].column_names()
                )
                data_columns = set(tables[table_name].columns.tolist())
                if schema_columns:
                    overlap = len(schema_columns & data_columns) / len(
                        schema_columns
                    )
                    column_scores.append(overlap)

            avg_column_overlap = (
                sum(column_scores) / len(column_scores)
                if column_scores
                else 0.0
            )
            if avg_column_overlap < 0.5:
                continue

            # Combined score: weight table overlap and column overlap equally
            combined_score = (table_overlap + avg_column_overlap) / 2.0

            if combined_score > best_score:
                best_score = combined_score
                best_schema = schema_name

        return best_schema

    @classmethod
    def default(cls) -> SchemaRegistry:
        """Create a registry pre-loaded with built-in schemas.

        Returns:
            A SchemaRegistry instance containing the CLIF schema and any
            other built-in schemas.
        """
        from cablecar.schema.clif import get_clif_schema

        registry = cls()
        registry.register(get_clif_schema())
        return registry
