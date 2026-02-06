"""Tests for cablecar.schema: base types, validator, and registry."""
import pandas as pd
import pytest

from cablecar.schema.base import SchemaDefinition, TableSpec, ColumnSpec
from cablecar.schema.validator import SchemaValidator, ValidationResult
from cablecar.schema.registry import SchemaRegistry


class TestColumnSpec:
    def test_defaults(self):
        col = ColumnSpec(name="x", dtype="str")
        assert col.required is True
        assert col.is_phi is False
        assert col.allowed_values is None

    def test_phi_column(self):
        col = ColumnSpec(name="ssn", dtype="str", is_phi=True)
        assert col.is_phi is True


class TestTableSpec:
    def test_get_column(self):
        spec = TableSpec(
            name="t",
            columns=[
                ColumnSpec(name="a", dtype="str"),
                ColumnSpec(name="b", dtype="int"),
            ],
        )
        assert spec.get_column("a").dtype == "str"
        assert spec.get_column("missing") is None

    def test_required_columns(self):
        spec = TableSpec(
            name="t",
            columns=[
                ColumnSpec(name="a", dtype="str", required=True),
                ColumnSpec(name="b", dtype="int", required=False),
            ],
        )
        assert spec.required_columns() == ["a"]

    def test_column_names(self):
        spec = TableSpec(
            name="t",
            columns=[
                ColumnSpec(name="x", dtype="str"),
                ColumnSpec(name="y", dtype="int"),
            ],
        )
        assert spec.column_names() == ["x", "y"]


class TestSchemaDefinition:
    def test_get_table(self, mini_schema):
        assert mini_schema.get_table("patient") is not None
        assert mini_schema.get_table("nonexistent") is None

    def test_get_phi_columns(self):
        schema = SchemaDefinition(
            name="test",
            version="1.0",
            tables={
                "t": TableSpec(
                    name="t",
                    columns=[
                        ColumnSpec(name="ssn", dtype="str", is_phi=True),
                        ColumnSpec(name="age", dtype="float"),
                    ],
                ),
            },
        )
        phi = schema.get_phi_columns()
        assert "t" in phi
        assert "ssn" in phi["t"]

    def test_required_vs_optional_tables(self, mini_schema):
        required = mini_schema.get_required_tables()
        assert "patient" in required
        assert "hospitalization" in required


class TestSchemaValidator:
    def test_valid_data(self, mini_schema, mini_dataframes):
        validator = SchemaValidator()
        result = validator.validate(mini_dataframes, mini_schema)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_table(self, mini_schema, mini_dataframes):
        tables = {k: v for k, v in mini_dataframes.items() if k != "patient"}
        validator = SchemaValidator()
        result = validator.validate(tables, mini_schema)
        assert result.is_valid is False
        assert any("patient" in e for e in result.errors)

    def test_missing_required_column(self, mini_schema):
        df = pd.DataFrame({"patient_id": ["P1"], "age": [50.0]})
        # Missing "sex" column which is required
        validator = SchemaValidator()
        result = validator.validate({"patient": df}, mini_schema)
        assert result.is_valid is False
        assert any("sex" in e for e in result.errors)

    def test_extra_columns_warning(self, mini_schema, mini_dataframes):
        tables = dict(mini_dataframes)
        tables["patient"] = tables["patient"].copy()
        tables["patient"]["extra_col"] = "hello"
        validator = SchemaValidator()
        result = validator.validate(tables, mini_schema)
        assert any("extra_col" in w for w in result.warnings)

    def test_null_in_required_column(self, mini_schema):
        df = pd.DataFrame({
            "patient_id": ["P1", None],
            "age": [50.0, 60.0],
            "sex": ["M", "F"],
        })
        validator = SchemaValidator()
        result = validator.validate({"patient": df}, mini_schema)
        assert result.is_valid is False
        assert any("null" in e.lower() for e in result.errors)

    def test_unknown_table_warning(self, mini_schema, mini_dataframes):
        tables = dict(mini_dataframes)
        tables["unknown_table"] = pd.DataFrame({"x": [1]})
        validator = SchemaValidator()
        result = validator.validate(tables, mini_schema)
        assert any("unknown_table" in w for w in result.warnings)


class TestSchemaRegistry:
    def test_register_and_get(self, mini_schema):
        registry = SchemaRegistry()
        registry.register(mini_schema)
        retrieved = registry.get("test_schema")
        assert retrieved.name == "test_schema"

    def test_get_missing_raises(self):
        registry = SchemaRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_schemas(self, mini_schema):
        registry = SchemaRegistry()
        registry.register(mini_schema)
        assert "test_schema" in registry.list_schemas()

    def test_default_has_clif(self):
        registry = SchemaRegistry.default()
        assert "clif" in registry.list_schemas()

    def test_infer_schema(self, mini_schema, mini_dataframes):
        registry = SchemaRegistry()
        registry.register(mini_schema)
        inferred = registry.infer_schema(mini_dataframes)
        assert inferred == "test_schema"

    def test_infer_returns_none_for_empty(self):
        registry = SchemaRegistry()
        assert registry.infer_schema({}) is None

    def test_infer_returns_none_no_match(self, mini_schema):
        registry = SchemaRegistry()
        registry.register(mini_schema)
        tables = {"random_table": pd.DataFrame({"x": [1]})}
        assert registry.infer_schema(tables) is None
