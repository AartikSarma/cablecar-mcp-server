"""Cohort definition, building, and manipulation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CohortDefinition(BaseModel):
    """Declarative cohort specification."""

    name: str
    description: str = ""
    index_table: str = "hospitalization"
    inclusion_criteria: list[dict[str, Any]] = Field(default_factory=list)
    exclusion_criteria: list[dict[str, Any]] = Field(default_factory=list)


class FlowStep(BaseModel):
    """One step in the CONSORT flow diagram."""

    step_name: str
    n_before: int
    n_after: int


# ---------------------------------------------------------------------------
# Operator dispatch
# ---------------------------------------------------------------------------

_OPERATORS: dict[str, Any] = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    "in": lambda s, v: s.isin(v),
    "not_in": lambda s, v: ~s.isin(v),
    "not_null": lambda s, v: s.notna(),
}


def _apply_criterion(
    df: pd.DataFrame,
    criterion: dict[str, Any],
) -> pd.DataFrame:
    """Filter *df* according to a single criterion dict.

    Raises:
        KeyError: If the column is missing.
        ValueError: If the operator is unsupported.
    """
    col = criterion["column"]
    op = criterion["op"]
    value = criterion.get("value")

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {sorted(df.columns.tolist())}")

    op_fn = _OPERATORS.get(op)
    if op_fn is None:
        raise ValueError(
            f"Unsupported operator '{op}'. "
            f"Supported: {sorted(_OPERATORS.keys())}"
        )

    mask = op_fn(df[col], value)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# CohortBuilder
# ---------------------------------------------------------------------------


class CohortBuilder:
    """Build a :class:`Cohort` from a :class:`CohortDefinition` and a DataStore."""

    def __init__(self, store) -> None:
        self.store = store

    def build(self, definition: CohortDefinition) -> Cohort:
        """Apply criteria to the index table and filter all tables."""
        index_df = self.store.get_table(definition.index_table).copy()
        flow_diagram: list[FlowStep] = []

        # Inclusion criteria
        for criterion in definition.inclusion_criteria:
            n_before = len(index_df)
            index_df = _apply_criterion(index_df, criterion)
            flow_diagram.append(
                FlowStep(
                    step_name=f"Include: {criterion['column']} {criterion['op']} {criterion.get('value', '')}",
                    n_before=n_before,
                    n_after=len(index_df),
                )
            )

        # Exclusion criteria (inverted)
        for criterion in definition.exclusion_criteria:
            n_before = len(index_df)
            kept = _apply_criterion(index_df, criterion)
            index_df = index_df[~index_df.index.isin(kept.index)].copy()
            flow_diagram.append(
                FlowStep(
                    step_name=f"Exclude: {criterion['column']} {criterion['op']} {criterion.get('value', '')}",
                    n_before=n_before,
                    n_after=len(index_df),
                )
            )

        # Filter all tables to matching hospitalization_ids
        hosp_ids = set(index_df["hospitalization_id"]) if "hospitalization_id" in index_df.columns else set()
        filtered_tables: dict[str, pd.DataFrame] = {}
        for table_name in self.store.list_tables():
            tbl = self.store.get_table(table_name)
            if table_name == definition.index_table:
                filtered_tables[table_name] = index_df
            elif "hospitalization_id" in tbl.columns and hosp_ids:
                filtered_tables[table_name] = tbl[tbl["hospitalization_id"].isin(hosp_ids)].copy()
            elif "patient_id" in tbl.columns and "patient_id" in index_df.columns:
                patient_ids = set(index_df["patient_id"])
                filtered_tables[table_name] = tbl[tbl["patient_id"].isin(patient_ids)].copy()
            else:
                filtered_tables[table_name] = tbl.copy()

        return Cohort(
            name=definition.name,
            definition=definition,
            index_df=index_df,
            tables_dict=filtered_tables,
            flow_diagram=flow_diagram,
        )


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------


class Cohort:
    """An immutable view of a filtered dataset."""

    def __init__(
        self,
        name: str,
        definition: CohortDefinition,
        index_df: pd.DataFrame,
        tables_dict: dict[str, pd.DataFrame],
        flow_diagram: list[FlowStep],
    ) -> None:
        self._name = name
        self._definition = definition
        self._index_df = index_df
        self._tables = tables_dict
        self._flow_diagram = list(flow_diagram)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def n(self) -> int:
        return len(self._index_df)

    @property
    def index(self) -> pd.DataFrame:
        return self._index_df

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return self._tables

    @property
    def flow_diagram(self) -> list[FlowStep]:
        return self._flow_diagram

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def get_table(self, name: str) -> pd.DataFrame:
        """Return a filtered table by name.

        Raises:
            KeyError: If the table is not in this cohort.
        """
        if name not in self._tables:
            raise KeyError(
                f"Table '{name}' not in cohort. "
                f"Available: {sorted(self._tables.keys())}"
            )
        return self._tables[name]

    def get_variable(self, table_name: str, column: str) -> pd.Series:
        """Return a single column from a cohort table.

        Raises:
            KeyError: If the table or column is missing.
        """
        df = self.get_table(table_name)
        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' not in table '{table_name}'. "
                f"Available: {sorted(df.columns.tolist())}"
            )
        return df[column]

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge_tables(self, table1: str, table2: str) -> pd.DataFrame:
        """Merge two tables on auto-detected shared columns."""
        df1 = self.get_table(table1)
        df2 = self.get_table(table2)
        shared = list(set(df1.columns) & set(df2.columns))
        if not shared:
            raise ValueError(
                f"Tables '{table1}' and '{table2}' share no columns."
            )
        return pd.merge(df1, df2, on=shared, how="inner")

    # ------------------------------------------------------------------
    # Subgroup
    # ------------------------------------------------------------------

    def subgroup(
        self,
        name: str,
        criteria: list[dict[str, Any]],
    ) -> Cohort:
        """Create a sub-cohort by filtering in-memory (no store reload)."""
        new_index = self._index_df.copy()
        new_flow = list(self._flow_diagram)

        for criterion in criteria:
            n_before = len(new_index)
            new_index = _apply_criterion(new_index, criterion)
            new_flow.append(
                FlowStep(
                    step_name=f"Subgroup: {criterion['column']} {criterion['op']} {criterion.get('value', '')}",
                    n_before=n_before,
                    n_after=len(new_index),
                )
            )

        # Filter tables
        hosp_ids = set(new_index["hospitalization_id"]) if "hospitalization_id" in new_index.columns else set()
        new_tables: dict[str, pd.DataFrame] = {}
        for tbl_name, tbl_df in self._tables.items():
            if tbl_name == self._definition.index_table:
                new_tables[tbl_name] = new_index
            elif "hospitalization_id" in tbl_df.columns and hosp_ids:
                new_tables[tbl_name] = tbl_df[tbl_df["hospitalization_id"].isin(hosp_ids)].copy()
            elif "patient_id" in tbl_df.columns and "patient_id" in new_index.columns:
                patient_ids = set(new_index["patient_id"])
                new_tables[tbl_name] = tbl_df[tbl_df["patient_id"].isin(patient_ids)].copy()
            else:
                new_tables[tbl_name] = tbl_df.copy()

        return Cohort(
            name=name,
            definition=self._definition,
            index_df=new_index,
            tables_dict=new_tables,
            flow_diagram=new_flow,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of this cohort."""
        flow_list = [
            {"step": s.step_name, "n_before": s.n_before, "n_after": s.n_after}
            for s in self._flow_diagram
        ]
        tables_summary = {
            name: {"rows": len(df), "columns": list(df.columns)}
            for name, df in self._tables.items()
        }
        return {
            "name": self._name,
            "n_subjects": self.n,
            "flow": flow_list,
            "tables": tables_summary,
        }
