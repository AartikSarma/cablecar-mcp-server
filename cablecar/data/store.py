"""In-memory data store with schema awareness and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cablecar.data.loader import DataLoader


class DataStore:
    """Central store for loaded clinical data tables.

    Wraps :class:`DataLoader` and provides schema-aware access, simple
    querying, and a lightweight cache.
    """

    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {}
        self._schema: Any = None  # SchemaDefinition | None
        self._cache: dict[str, Any] = {}
        self._loader = DataLoader()

    # ------------------------------------------------------------------
    # Schema property
    # ------------------------------------------------------------------

    @property
    def schema(self):
        """Return the associated schema (if any)."""
        return self._schema

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path, schema_name: str | None = None) -> dict:
        """Load data from *path* and optionally bind a schema.

        Returns a summary dict ``{"tables": {name: summary, ...}}``.
        """
        path = Path(path)
        self.tables = self._loader.load_directory(path)

        if schema_name:
            from cablecar.schema.registry import SchemaRegistry

            registry = SchemaRegistry.default()
            self._schema = registry.get(schema_name)
        elif self._schema is None:
            # Try auto-detection
            try:
                from cablecar.schema.registry import SchemaRegistry

                registry = SchemaRegistry.default()
                inferred = registry.infer_schema(self.tables)
                if inferred:
                    self._schema = registry.get(inferred)
            except Exception:
                pass

        return {"tables": self.get_summary()}

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def get_table(self, name: str) -> pd.DataFrame:
        """Return a table by name.

        Raises:
            KeyError: If the table is not loaded.
        """
        if name not in self.tables:
            raise KeyError(
                f"Table '{name}' not found. "
                f"Available: {sorted(self.tables.keys())}"
            )
        return self.tables[name]

    def list_tables(self) -> list[str]:
        """Return the names of all loaded tables."""
        return sorted(self.tables.keys())

    def get_summary(self) -> dict[str, dict]:
        """Return per-table summaries."""
        return {
            name: self._loader.get_table_summary(df, name)
            for name, df in self.tables.items()
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        table_name: str,
        columns: list[str] | None = None,
        conditions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Simple column-selection and equality-filter query.

        Raises:
            KeyError: If the table or requested columns are missing.
        """
        df = self.get_table(table_name)

        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Columns not found in '{table_name}': {missing}"
                )
            df = df[columns]

        if conditions:
            for col, val in conditions.items():
                if col not in df.columns:
                    raise KeyError(
                        f"Condition column '{col}' not in '{table_name}'"
                    )
                df = df[df[col] == val]

        return df

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def cache_get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def cache_clear(self) -> None:
        self._cache.clear()
