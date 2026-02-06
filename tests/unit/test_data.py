"""Tests for cablecar.data: loader, store, and cohort."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from cablecar.data.loader import DataLoader
from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition, Cohort, FlowStep


class TestDataLoader:
    def test_detect_csv(self):
        loader = DataLoader()
        assert loader.detect_format("test.csv") == "csv"

    def test_detect_parquet(self):
        loader = DataLoader()
        assert loader.detect_format("test.parquet") == "parquet"
        assert loader.detect_format("test.pq") == "parquet"

    def test_detect_unknown_raises(self):
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.detect_format("test.xlsx")

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loader = DataLoader()
        result = loader.load_file(path)
        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_load_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)

        loader = DataLoader()
        result = loader.load_file(path)
        assert len(result) == 2

    def test_load_directory(self, tmp_path):
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "table1.csv", index=False)
        pd.DataFrame({"y": [2]}).to_parquet(tmp_path / "table2.parquet", index=False)

        loader = DataLoader()
        tables = loader.load_directory(tmp_path)
        assert "table1" in tables
        assert "table2" in tables

    def test_load_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.csv")

    def test_load_directory_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/dir")

    def test_get_table_summary(self):
        loader = DataLoader()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        summary = loader.get_table_summary(df, "test_table")
        assert summary["name"] == "test_table"
        assert summary["rows"] == 3
        assert "a" in summary["columns"]
        assert "memory_mb" in summary


class TestDataStore:
    def test_load_directory(self, tmp_path):
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / "t.csv", index=False)
        store = DataStore()
        result = store.load(tmp_path)
        assert "t" in store.tables
        assert "tables" in result

    def test_get_table(self, loaded_store):
        df = loaded_store.get_table("patient")
        assert len(df) == 50

    def test_get_table_missing_raises(self, loaded_store):
        with pytest.raises(KeyError):
            loaded_store.get_table("nonexistent")

    def test_list_tables(self, loaded_store):
        tables = loaded_store.list_tables()
        assert "patient" in tables
        assert "hospitalization" in tables

    def test_get_summary(self, loaded_store):
        summary = loaded_store.get_summary()
        assert "patient" in summary
        assert summary["patient"]["rows"] == 50

    def test_query_basic(self, loaded_store):
        result = loaded_store.query("patient", columns=["patient_id", "age"])
        assert list(result.columns) == ["patient_id", "age"]

    def test_query_with_conditions(self, loaded_store):
        result = loaded_store.query("patient", conditions={"sex": "M"})
        assert all(result["sex"] == "M")

    def test_query_missing_column_raises(self, loaded_store):
        with pytest.raises(KeyError):
            loaded_store.query("patient", columns=["nonexistent"])

    def test_cache(self, loaded_store):
        loaded_store.cache_set("key1", "value1")
        assert loaded_store.cache_get("key1") == "value1"
        assert loaded_store.cache_get("missing") is None

        loaded_store.cache_clear()
        assert loaded_store.cache_get("key1") is None

    def test_schema_property(self, loaded_store, mini_schema):
        assert loaded_store.schema is not None
        assert loaded_store.schema.name == mini_schema.name


class TestCohortDefinition:
    def test_defaults(self):
        defn = CohortDefinition(name="test")
        assert defn.index_table == "hospitalization"
        assert defn.inclusion_criteria == []
        assert defn.exclusion_criteria == []


class TestCohortBuilder:
    def test_build_no_criteria(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(name="all")
        cohort = builder.build(defn)
        assert cohort.n == 50

    def test_build_with_inclusion(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(
            name="adults",
            inclusion_criteria=[{"column": "age_at_admission", "op": ">=", "value": 50}],
        )
        cohort = builder.build(defn)
        assert cohort.n <= 50
        assert all(cohort.index["age_at_admission"] >= 50)

    def test_build_with_exclusion(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(
            name="no_hospice",
            exclusion_criteria=[{"column": "discharge_category", "op": "==", "value": "Hospice"}],
        )
        cohort = builder.build(defn)
        assert "Hospice" not in cohort.index["discharge_category"].values

    def test_missing_column_raises(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(
            name="bad",
            inclusion_criteria=[{"column": "nonexistent", "op": "==", "value": 1}],
        )
        with pytest.raises(KeyError):
            builder.build(defn)

    def test_unsupported_operator_raises(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(
            name="bad",
            inclusion_criteria=[{"column": "age_at_admission", "op": "like", "value": 50}],
        )
        with pytest.raises(ValueError):
            builder.build(defn)

    def test_all_operators(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        ops = [
            ({"column": "age_at_admission", "op": "==", "value": 65.0}, None),
            ({"column": "age_at_admission", "op": "!=", "value": 0}, None),
            ({"column": "age_at_admission", "op": ">", "value": 30}, None),
            ({"column": "age_at_admission", "op": "<", "value": 100}, None),
            ({"column": "age_at_admission", "op": "<=", "value": 100}, None),
            ({"column": "discharge_category", "op": "in", "value": ["Home", "SNF"]}, None),
            ({"column": "discharge_category", "op": "not_in", "value": ["Hospice"]}, None),
            ({"column": "hospital_mortality", "op": "not_null"}, None),
        ]
        for criterion, _ in ops:
            defn = CohortDefinition(name="test", inclusion_criteria=[criterion])
            cohort = builder.build(defn)
            assert isinstance(cohort, Cohort)

    def test_flow_diagram(self, loaded_store):
        builder = CohortBuilder(loaded_store)
        defn = CohortDefinition(
            name="filtered",
            inclusion_criteria=[
                {"column": "age_at_admission", "op": ">=", "value": 18},
                {"column": "age_at_admission", "op": "<=", "value": 90},
            ],
        )
        cohort = builder.build(defn)
        assert len(cohort.flow_diagram) == 2
        for step in cohort.flow_diagram:
            assert isinstance(step, FlowStep)
            assert step.n_before >= step.n_after


class TestCohort:
    def test_properties(self, mini_cohort):
        assert mini_cohort.n > 0
        assert isinstance(mini_cohort.index, pd.DataFrame)
        assert isinstance(mini_cohort.tables, dict)

    def test_get_table(self, mini_cohort):
        hosp = mini_cohort.get_table("hospitalization")
        assert len(hosp) == mini_cohort.n

    def test_get_table_missing_raises(self, mini_cohort):
        with pytest.raises(KeyError):
            mini_cohort.get_table("nonexistent")

    def test_get_variable(self, mini_cohort):
        ages = mini_cohort.get_variable("hospitalization", "age_at_admission")
        assert isinstance(ages, pd.Series)
        assert all(ages >= 18)

    def test_get_variable_missing_raises(self, mini_cohort):
        with pytest.raises(KeyError):
            mini_cohort.get_variable("hospitalization", "nonexistent_col")

    def test_subgroup(self, mini_cohort):
        subgroup = mini_cohort.subgroup(
            "elderly",
            [{"column": "age_at_admission", "op": ">=", "value": 70}],
        )
        assert isinstance(subgroup, Cohort)
        assert subgroup.n <= mini_cohort.n
        assert all(subgroup.index["age_at_admission"] >= 70)
        # Flow diagram grows
        assert len(subgroup.flow_diagram) > len(mini_cohort.flow_diagram)

    def test_subgroup_no_data_reload(self, mini_cohort):
        """Subgroup should filter existing data, not reload from store."""
        original_n = mini_cohort.n
        subgroup = mini_cohort.subgroup(
            "test",
            [{"column": "age_at_admission", "op": ">", "value": 50}],
        )
        # Original cohort should be unchanged
        assert mini_cohort.n == original_n

    def test_merge_tables(self, mini_cohort):
        # Patient and hospitalization share patient_id
        if "patient" in mini_cohort.tables:
            merged = mini_cohort.merge_tables("hospitalization", "patient")
            assert "age" in merged.columns or "age_at_admission" in merged.columns

    def test_summary(self, mini_cohort):
        summary = mini_cohort.summary()
        assert "name" in summary
        assert "n_subjects" in summary
        assert summary["n_subjects"] == mini_cohort.n
        assert "flow" in summary
        assert "tables" in summary
