"""CableCar Data Server Tools.

These 4 tools are the ONLY interface to raw patient data.
ALL outputs pass through PrivacyGuard.sanitize_for_llm() before returning.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cablecar.data.store import DataStore
from cablecar.data.cohort import CohortBuilder, CohortDefinition, Cohort
from cablecar.privacy.guard import PrivacyGuard
from cablecar.privacy.audit import AuditLog
from cablecar.schema.registry import SchemaRegistry


class DataServerTools:
    """The 4-tool data server. Every method returns privacy-sanitized output."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        privacy_guard: PrivacyGuard | None = None,
        audit_log: AuditLog | None = None,
    ):
        self.store = DataStore()
        self.privacy = privacy_guard or PrivacyGuard()
        self.audit = audit_log or AuditLog()
        self.registry = SchemaRegistry.default()
        self._cohorts: dict[str, Cohort] = {}

        if data_path:
            self.get_schema()  # Initialize schema info

    def get_schema(self) -> dict:
        """Tool 1: Get schema and data dictionary. No patient data returned.

        Returns schema information including table names, column definitions,
        data types, descriptions, and relationships.
        """
        try:
            result = {
                "available_schemas": self.registry.list_schemas(),
                "tables": {},
            }

            # If data is loaded, include actual table info
            if self.store.list_tables():
                schema = self.store.schema
                summary = self.store.get_summary()
                result["loaded_data"] = summary

                if schema:
                    result["schema_name"] = schema.name
                    result["schema_version"] = schema.version
                    for table_name, table_spec in schema.tables.items():
                        result["tables"][table_name] = {
                            "description": table_spec.description,
                            "columns": [
                                {
                                    "name": col.name,
                                    "dtype": col.dtype,
                                    "required": col.required,
                                    "description": col.description,
                                    "is_phi": col.is_phi,
                                }
                                for col in table_spec.columns
                            ],
                            "primary_key": table_spec.primary_key,
                            "foreign_keys": table_spec.foreign_keys,
                        }
            else:
                # Return schema definitions without loaded data
                for schema_name in self.registry.list_schemas():
                    schema = self.registry.get(schema_name)
                    for table_name, table_spec in schema.tables.items():
                        result["tables"][table_name] = {
                            "description": table_spec.description,
                            "columns": [
                                {
                                    "name": col.name,
                                    "dtype": col.dtype,
                                    "required": col.required,
                                    "description": col.description,
                                }
                                for col in table_spec.columns
                            ],
                        }

            self.audit.log_tool_call("get_schema", "Retrieved schema information")

            # Schema info contains no patient data, but sanitize anyway for consistency
            return self.privacy.sanitize_for_llm(result, context="get_schema")

        except Exception as e:
            return {"error": str(e), "sanitized": True}

    def load_data(self, path: str, schema: str | None = None) -> dict:
        """Tool 2: Load and validate clinical dataset. Returns sanitized summary only.

        Args:
            path: Path to data directory containing CSV/Parquet files
            schema: Optional schema name to validate against (e.g., "clif")

        Returns:
            Sanitized summary with table counts, validation status, data quality info.
            NEVER returns raw patient data.
        """
        try:
            summary = self.store.load(path, schema_name=schema)

            # Get PHI columns from schema for privacy guard
            if self.store.schema:
                phi_cols = self.store.schema.get_phi_columns()
                self.privacy.phi_columns = phi_cols

            # Build detailed summary with validation
            result = {
                "status": "loaded",
                "path": str(path),
                "tables_loaded": self.store.list_tables(),
                "summary": summary,
            }

            # Validate against schema if specified
            if schema or self.store.schema:
                from cablecar.schema.validator import SchemaValidator
                schema_def = self.store.schema
                if schema_def:
                    validator = SchemaValidator()
                    validation = validator.validate(
                        {name: self.store.get_table(name) for name in self.store.list_tables()},
                        schema_def,
                    )
                    result["validation"] = {
                        "is_valid": validation.is_valid,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                    }

            self.audit.log_tool_call(
                "load_data",
                f"Loaded {len(self.store.list_tables())} tables from {path}",
                data_accessed=[str(path)],
            )

            return self.privacy.sanitize_for_llm(result, context="load_data")

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "sanitized": True}

    def query_cohort(
        self,
        name: str = "main",
        description: str = "",
        inclusion: list[dict] | None = None,
        exclusion: list[dict] | None = None,
        index_table: str = "hospitalization",
    ) -> dict:
        """Tool 3: Define and query a cohort. Returns sanitized flow diagram and summary.

        Args:
            name: Cohort name for reference
            description: Human-readable description
            inclusion: List of inclusion criteria, e.g. [{"column": "age_at_admission", "op": ">=", "value": 18}]
            exclusion: List of exclusion criteria (same format)
            index_table: Base table (default "hospitalization")

        Returns:
            Sanitized CONSORT flow diagram and cohort summary.
            NEVER returns raw patient data.
        """
        try:
            if not self.store.list_tables():
                return {"error": "No data loaded. Use load_data first.", "sanitized": True}

            definition = CohortDefinition(
                name=name,
                description=description,
                inclusion_criteria=inclusion or [],
                exclusion_criteria=exclusion or [],
                index_table=index_table,
            )

            builder = CohortBuilder(self.store)
            cohort = builder.build(definition)

            # Store cohort for later use
            self._cohorts[name] = cohort

            # Build sanitized result
            result = cohort.summary()

            privacy_actions = []
            self.audit.log_tool_call(
                "query_cohort",
                f"Built cohort '{name}' with {cohort.n} subjects",
                privacy_actions=privacy_actions,
            )

            return self.privacy.sanitize_for_llm(result, context="query_cohort")

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "sanitized": True}

    def execute_analysis(
        self,
        analysis_type: str,
        params: dict | None = None,
        cohort_name: str = "main",
    ) -> dict:
        """Tool 4: Execute analysis on a cohort. Returns sanitized results only.

        Args:
            analysis_type: Type of analysis - "descriptive", "hypothesis", "regression",
                          "prediction", "survival", "subgroup", "summary_stats"
            params: Analysis-specific parameters
            cohort_name: Which cohort to analyze (default "main")

        Returns:
            Sanitized analysis results (coefficients, CIs, p-values, aggregated tables).
            NEVER returns raw patient data.
        """
        try:
            params = params or {}

            if cohort_name not in self._cohorts:
                return {
                    "error": f"Cohort '{cohort_name}' not found. Available: {list(self._cohorts.keys())}",
                    "sanitized": True,
                }

            cohort = self._cohorts[cohort_name]

            # Dispatch to appropriate analysis
            if analysis_type == "summary_stats":
                result = self._summary_stats(cohort, params)
            elif analysis_type == "descriptive":
                result = self._descriptive_analysis(cohort, params)
            elif analysis_type == "hypothesis":
                result = self._hypothesis_test(cohort, params)
            elif analysis_type == "regression":
                result = self._regression_analysis(cohort, params)
            elif analysis_type == "subgroup":
                result = self._subgroup_analysis(cohort, params)
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}. Supported: summary_stats, descriptive, hypothesis, regression, subgroup"}

            self.audit.log_tool_call(
                "execute_analysis",
                f"Executed {analysis_type} on cohort '{cohort_name}'",
            )

            return self.privacy.sanitize_for_llm(result, context=f"execute_analysis:{analysis_type}")

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "sanitized": True}

    def _summary_stats(self, cohort: Cohort, params: dict) -> dict:
        """Compute summary statistics for specified variables."""
        table_name = params.get("table", "hospitalization")
        columns = params.get("columns")

        df = cohort.get_table(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not in cohort"}

        if columns:
            df = df[[c for c in columns if c in df.columns]]

        result = {"table": table_name, "n": len(df), "variables": {}}

        for col in df.columns:
            col_data = df[col]
            stats = {
                "dtype": str(col_data.dtype),
                "missing_count": int(col_data.isna().sum()),
                "missing_pct": round(col_data.isna().mean() * 100, 1),
                "n_unique": int(col_data.nunique()),
            }

            if pd.api.types.is_numeric_dtype(col_data):
                desc = col_data.describe()
                stats.update({
                    "mean": round(float(desc["mean"]), 2),
                    "std": round(float(desc["std"]), 2),
                    "median": round(float(desc["50%"]), 2),
                    "q1": round(float(desc["25%"]), 2),
                    "q3": round(float(desc["75%"]), 2),
                    "min": round(float(desc["min"]), 2),
                    "max": round(float(desc["max"]), 2),
                })
            elif col_data.dtype == "object":
                value_counts = col_data.value_counts()
                stats["top_values"] = {
                    str(k): int(v) for k, v in value_counts.head(20).items()
                }

            result["variables"][col] = stats

        return result

    def _descriptive_analysis(self, cohort: Cohort, params: dict) -> dict:
        """Generate Table 1 / descriptive statistics."""
        stratify_by = params.get("stratify_by")
        variables = params.get("variables")
        table_name = params.get("table", "hospitalization")

        df = cohort.get_table(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not in cohort"}

        if variables:
            cols_to_use = [c for c in variables if c in df.columns]
            if stratify_by and stratify_by in df.columns:
                cols_to_use.append(stratify_by)
            df = df[list(set(cols_to_use))]

        result = {"analysis": "descriptive", "n": len(df), "table": table_name}

        if stratify_by and stratify_by in df.columns:
            result["stratified_by"] = stratify_by
            result["strata"] = {}

            for group_val, group_df in df.groupby(stratify_by):
                group_stats = {}
                for col in group_df.columns:
                    if col == stratify_by:
                        continue
                    if pd.api.types.is_numeric_dtype(group_df[col]):
                        desc = group_df[col].dropna().describe()
                        group_stats[col] = {
                            "n": int(desc["count"]),
                            "mean": round(float(desc["mean"]), 2),
                            "std": round(float(desc["std"]), 2),
                            "median": round(float(desc["50%"]), 2),
                        }
                    else:
                        vc = group_df[col].value_counts()
                        group_stats[col] = {str(k): int(v) for k, v in vc.items()}

                result["strata"][str(group_val)] = {
                    "n": len(group_df),
                    "statistics": group_stats,
                }
        else:
            result["statistics"] = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].dropna().describe()
                    result["statistics"][col] = {
                        "n": int(desc["count"]),
                        "mean": round(float(desc["mean"]), 2),
                        "std": round(float(desc["std"]), 2),
                        "median": round(float(desc["50%"]), 2),
                    }
                else:
                    vc = df[col].value_counts()
                    result["statistics"][col] = {str(k): int(v) for k, v in vc.items()}

        return result

    def _hypothesis_test(self, cohort: Cohort, params: dict) -> dict:
        """Perform hypothesis testing."""
        from scipy import stats as scipy_stats

        test_type = params.get("test", "mann_whitney")
        variable = params.get("variable")
        group_var = params.get("group_variable")
        table_name = params.get("table", "hospitalization")

        if not variable or not group_var:
            return {"error": "Must specify 'variable' and 'group_variable'"}

        df = cohort.get_table(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not in cohort"}

        groups = df.groupby(group_var)[variable].apply(lambda x: x.dropna().values)

        if len(groups) < 2:
            return {"error": f"Need at least 2 groups in '{group_var}', found {len(groups)}"}

        group_names = list(groups.index)
        group_data = [groups[g] for g in group_names]

        result = {
            "analysis": "hypothesis_test",
            "test": test_type,
            "variable": variable,
            "group_variable": group_var,
            "groups": {str(g): {"n": len(d), "mean": round(float(d.mean()), 3), "std": round(float(d.std()), 3)} for g, d in zip(group_names, group_data)},
        }

        if test_type == "mann_whitney" and len(group_data) == 2:
            stat, pval = scipy_stats.mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
            result["statistic"] = round(float(stat), 4)
            result["p_value"] = round(float(pval), 6)
            result["method"] = "Mann-Whitney U test"
        elif test_type == "t_test" and len(group_data) == 2:
            stat, pval = scipy_stats.ttest_ind(group_data[0], group_data[1])
            result["statistic"] = round(float(stat), 4)
            result["p_value"] = round(float(pval), 6)
            result["method"] = "Independent samples t-test"
        elif test_type == "kruskal":
            stat, pval = scipy_stats.kruskal(*group_data)
            result["statistic"] = round(float(stat), 4)
            result["p_value"] = round(float(pval), 6)
            result["method"] = "Kruskal-Wallis H test"
        elif test_type == "chi_square":
            contingency = pd.crosstab(df[group_var], df[variable])
            stat, pval, dof, expected = scipy_stats.chi2_contingency(contingency)
            result["statistic"] = round(float(stat), 4)
            result["p_value"] = round(float(pval), 6)
            result["degrees_of_freedom"] = int(dof)
            result["method"] = "Chi-square test of independence"
        else:
            result["error"] = f"Unsupported test: {test_type} for {len(group_data)} groups"

        return result

    def _regression_analysis(self, cohort: Cohort, params: dict) -> dict:
        """Perform regression analysis."""
        import statsmodels.api as sm

        model_type = params.get("model", "logistic")
        outcome = params.get("outcome")
        predictors = params.get("predictors", [])
        table_name = params.get("table", "hospitalization")

        if not outcome or not predictors:
            return {"error": "Must specify 'outcome' and 'predictors'"}

        df = cohort.get_table(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not in cohort"}

        # Prepare data
        cols_needed = [outcome] + predictors
        model_df = df[cols_needed].dropna()

        y = model_df[outcome]
        X = model_df[predictors]

        # Handle categorical predictors
        X = pd.get_dummies(X, drop_first=True, dtype=float)
        X = sm.add_constant(X)

        result = {
            "analysis": "regression",
            "model_type": model_type,
            "outcome": outcome,
            "predictors": predictors,
            "n": len(model_df),
            "n_excluded_missing": len(df) - len(model_df),
        }

        try:
            if model_type == "logistic":
                model = sm.Logit(y, X).fit(disp=0)
                result["coefficients"] = {}
                for name in model.params.index:
                    coef = float(model.params[name])
                    result["coefficients"][name] = {
                        "coefficient": round(coef, 4),
                        "odds_ratio": round(float(np.exp(coef)), 4) if name != "const" else None,
                        "std_error": round(float(model.bse[name]), 4),
                        "z_value": round(float(model.tvalues[name]), 4),
                        "p_value": round(float(model.pvalues[name]), 6),
                        "ci_lower": round(float(model.conf_int().loc[name, 0]), 4),
                        "ci_upper": round(float(model.conf_int().loc[name, 1]), 4),
                    }
                result["model_fit"] = {
                    "pseudo_r_squared": round(float(model.prsquared), 4),
                    "aic": round(float(model.aic), 2),
                    "bic": round(float(model.bic), 2),
                    "log_likelihood": round(float(model.llf), 2),
                }
            elif model_type == "linear":
                model = sm.OLS(y, X).fit()
                result["coefficients"] = {}
                for name in model.params.index:
                    result["coefficients"][name] = {
                        "coefficient": round(float(model.params[name]), 4),
                        "std_error": round(float(model.bse[name]), 4),
                        "t_value": round(float(model.tvalues[name]), 4),
                        "p_value": round(float(model.pvalues[name]), 6),
                        "ci_lower": round(float(model.conf_int().loc[name, 0]), 4),
                        "ci_upper": round(float(model.conf_int().loc[name, 1]), 4),
                    }
                result["model_fit"] = {
                    "r_squared": round(float(model.rsquared), 4),
                    "adjusted_r_squared": round(float(model.rsquared_adj), 4),
                    "f_statistic": round(float(model.fvalue), 4),
                    "f_p_value": round(float(model.f_pvalue), 6),
                    "aic": round(float(model.aic), 2),
                    "bic": round(float(model.bic), 2),
                }
            else:
                result["error"] = f"Unsupported model type: {model_type}. Supported: linear, logistic"
        except Exception as e:
            result["error"] = f"Model fitting failed: {str(e)}"

        return result

    def _subgroup_analysis(self, cohort: Cohort, params: dict) -> dict:
        """Perform subgroup analysis efficiently using Cohort.subgroup()."""
        subgroup_var = params.get("subgroup_variable")
        analysis_type = params.get("analysis_type", "descriptive")
        analysis_params = params.get("analysis_params", {})
        table_name = params.get("table", "hospitalization")

        if not subgroup_var:
            return {"error": "Must specify 'subgroup_variable'"}

        df = cohort.get_table(table_name)
        if df is None:
            return {"error": f"Table '{table_name}' not in cohort"}

        if subgroup_var not in df.columns:
            return {"error": f"Variable '{subgroup_var}' not in table '{table_name}'"}

        unique_vals = df[subgroup_var].dropna().unique()

        result = {
            "analysis": "subgroup",
            "subgroup_variable": subgroup_var,
            "n_subgroups": len(unique_vals),
            "subgroups": {},
        }

        for val in unique_vals:
            # Use Cohort.subgroup() - efficient, no data reload
            sub = cohort.subgroup(
                name=f"{cohort.name}_{subgroup_var}_{val}",
                criteria=[{"column": subgroup_var, "op": "==", "value": val}],
            )

            # Run analysis on subgroup
            if analysis_type == "descriptive":
                sub_result = self._descriptive_analysis(sub, analysis_params)
            elif analysis_type == "regression":
                sub_result = self._regression_analysis(sub, analysis_params)
            else:
                sub_result = {"n": sub.n}

            result["subgroups"][str(val)] = {
                "n": sub.n,
                "result": sub_result,
            }

        return result

    def get_cohort(self, name: str = "main") -> Cohort | None:
        """Get a stored cohort by name (internal use only)."""
        return self._cohorts.get(name)
