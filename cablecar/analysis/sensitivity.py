"""Sensitivity analyses to test robustness of primary results."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class SensitivityAnalysis(BaseAnalysis):
    """Run sensitivity analyses alongside a base analysis.

    Sensitivity types:
    - ``"complete_case"``: Restrict to rows with no missing values in the
      analysis columns.
    - ``"missing_indicator"``: Add binary missing-indicator columns for
      variables with missing data, then impute missing numeric values
      with the column median.
    - ``"exclude_outliers"``: Remove rows where any numeric analysis
      variable falls outside mean +/- 3 * SD.
    - ``"alternate_definition"``: Re-run with a modified variable
      definition (caller supplies a ``transform_fn`` mapping
      DataFrame -> DataFrame).

    Each run produces a pair of results -- the *base* result and the
    *sensitivity* result -- so the researcher can compare them.
    """

    _SUPPORTED_TYPES = {
        "complete_case",
        "missing_indicator",
        "exclude_outliers",
        "alternate_definition",
    }

    def run(
        self,
        base_analysis_fn: Callable,
        sensitivity_type: str,
        table: str = "hospitalization",
        columns: list[str] | None = None,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> AnalysisResult:
        """Run a sensitivity analysis.

        Parameters
        ----------
        base_analysis_fn:
            A callable that takes a :class:`~cablecar.data.cohort.Cohort`
            and returns an :class:`AnalysisResult`.  This is run on both
            the original and the modified cohort.
        sensitivity_type:
            One of ``"complete_case"``, ``"missing_indicator"``,
            ``"exclude_outliers"``, ``"alternate_definition"``.
        table:
            Cohort table to modify for the sensitivity analysis.
        columns:
            Columns relevant to the analysis (used to scope missing-data
            operations and outlier detection).  If ``None``, all columns
            in *table* are considered.
        transform_fn:
            Required when *sensitivity_type* is ``"alternate_definition"``.
            A callable that transforms the table DataFrame.
        **kwargs:
            Additional keyword arguments (reserved for future use).

        Returns
        -------
        AnalysisResult
            Contains ``base_result`` and ``sensitivity_result`` for
            side-by-side comparison.
        """
        self._warnings = []

        if sensitivity_type not in self._SUPPORTED_TYPES:
            self._warn(
                f"Unknown sensitivity_type '{sensitivity_type}'. "
                f"Supported: {sorted(self._SUPPORTED_TYPES)}."
            )
            return AnalysisResult(
                analysis_type="sensitivity",
                parameters={"sensitivity_type": sensitivity_type},
                results={},
                warnings=self._warnings,
            )

        # Run the base analysis on the original cohort
        try:
            base_result = base_analysis_fn(self.cohort)
            if isinstance(base_result, AnalysisResult):
                base_dict = base_result.to_dict()
            else:
                base_dict = base_result
        except Exception as exc:
            self._warn(f"Base analysis failed: {exc}")
            base_dict = {"error": str(exc)}

        # Build modified cohort
        try:
            modified_cohort = self._build_modified_cohort(
                sensitivity_type=sensitivity_type,
                table=table,
                columns=columns,
                transform_fn=transform_fn,
            )
        except Exception as exc:
            self._warn(f"Failed to build modified cohort: {exc}")
            return AnalysisResult(
                analysis_type="sensitivity",
                parameters={"sensitivity_type": sensitivity_type, "table": table},
                results={"base_result": base_dict, "sensitivity_result": {"error": str(exc)}},
                warnings=self._warnings,
            )

        # Run the sensitivity analysis on the modified cohort
        try:
            sens_result = base_analysis_fn(modified_cohort)
            if isinstance(sens_result, AnalysisResult):
                sens_dict = sens_result.to_dict()
            else:
                sens_dict = sens_result
        except Exception as exc:
            self._warn(f"Sensitivity analysis failed: {exc}")
            sens_dict = {"error": str(exc)}

        # Comparison metadata
        comparison = self._compare_results(base_dict, sens_dict)

        return AnalysisResult(
            analysis_type="sensitivity",
            parameters={
                "sensitivity_type": sensitivity_type,
                "table": table,
                "columns": columns,
            },
            results={
                "base_result": base_dict,
                "sensitivity_result": sens_dict,
                "comparison": comparison,
            },
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Cohort modification
    # ------------------------------------------------------------------

    def _build_modified_cohort(
        self,
        sensitivity_type: str,
        table: str,
        columns: list[str] | None,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None,
    ) -> Any:
        """Build a modified cohort for the specified sensitivity type."""
        from cablecar.data.cohort import Cohort, CohortDefinition

        try:
            original_df = self.cohort.get_table(table)
        except KeyError:
            raise KeyError(f"Table '{table}' not found in cohort.")

        target_cols = columns if columns else list(original_df.columns)
        modified_df: pd.DataFrame

        if sensitivity_type == "complete_case":
            modified_df = self._complete_case(original_df, target_cols)

        elif sensitivity_type == "missing_indicator":
            modified_df = self._missing_indicator(original_df, target_cols)

        elif sensitivity_type == "exclude_outliers":
            modified_df = self._exclude_outliers(original_df, target_cols)

        elif sensitivity_type == "alternate_definition":
            if transform_fn is None:
                raise ValueError(
                    "'alternate_definition' sensitivity requires a transform_fn."
                )
            modified_df = transform_fn(original_df.copy())

        else:
            modified_df = original_df.copy()

        n_original = len(original_df)
        n_modified = len(modified_df)
        self._warn(
            f"Sensitivity '{sensitivity_type}': {n_original} -> {n_modified} rows "
            f"({n_original - n_modified} removed)."
        )

        # Build a new Cohort with the modified table
        new_data = {k: v.copy() for k, v in self.cohort.tables.items()}
        new_data[table] = modified_df

        return Cohort(
            name=f"{self.cohort._name}_sensitivity_{sensitivity_type}",
            definition=self.cohort._definition,
            data=new_data,
            flow=list(self.cohort.flow_diagram),
            store=self.cohort._store,
        )

    # ------------------------------------------------------------------
    # Modification strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _complete_case(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Keep only rows with no missing values in the specified columns."""
        valid_cols = [c for c in columns if c in df.columns]
        return df.dropna(subset=valid_cols).copy()

    @staticmethod
    def _missing_indicator(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Add binary missing-indicator columns and impute with medians.

        For each column with missing values, a new column
        ``<col>_missing`` is added (1 = was missing, 0 = was present).
        Numeric columns are then imputed with the column median.
        """
        result = df.copy()
        valid_cols = [c for c in columns if c in result.columns]

        for col in valid_cols:
            n_miss = result[col].isna().sum()
            if n_miss == 0:
                continue

            # Add indicator
            result[f"{col}_missing"] = result[col].isna().astype(int)

            # Impute numeric with median
            if pd.api.types.is_numeric_dtype(result[col]):
                median_val = result[col].median()
                result[col] = result[col].fillna(median_val)

        return result

    @staticmethod
    def _exclude_outliers(
        df: pd.DataFrame, columns: list[str], n_sd: float = 3.0,
    ) -> pd.DataFrame:
        """Remove rows where any numeric variable is outside mean +/- n_sd * SD."""
        result = df.copy()
        valid_cols = [c for c in columns if c in result.columns]
        mask = pd.Series(True, index=result.index)

        for col in valid_cols:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            col_data = result[col].dropna()
            if len(col_data) == 0:
                continue

            mean = col_data.mean()
            std = col_data.std(ddof=1)
            if std == 0:
                continue

            lower = mean - n_sd * std
            upper = mean + n_sd * std

            col_mask = result[col].isna() | ((result[col] >= lower) & (result[col] <= upper))
            mask = mask & col_mask

        return result.loc[mask].copy()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_results(
        base: dict[str, Any], sensitivity: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a brief comparison between base and sensitivity results.

        Extracts top-level numeric metrics from both results and
        computes absolute and relative differences.
        """
        comparison: dict[str, Any] = {}

        if "error" in base or "error" in sensitivity:
            comparison["note"] = "One or both analyses had errors; comparison limited."
            return comparison

        # Try to find common numeric fields in results
        base_results = base.get("results", {})
        sens_results = sensitivity.get("results", {})

        # Look for coefficients or aggregate metrics
        for key in ["coefficients", "aggregate_metrics"]:
            if key in base_results and key in sens_results:
                base_vals = base_results[key]
                sens_vals = sens_results[key]
                diffs: dict[str, Any] = {}

                if isinstance(base_vals, dict) and isinstance(sens_vals, dict):
                    for k in base_vals:
                        if k in sens_vals:
                            b_val = base_vals[k]
                            s_val = sens_vals[k]

                            # Extract p_value for comparison if available
                            if isinstance(b_val, dict) and isinstance(s_val, dict):
                                for metric in ["p_value", "coefficient", "odds_ratio", "mean"]:
                                    bm = b_val.get(metric)
                                    sm = s_val.get(metric)
                                    if (
                                        isinstance(bm, (int, float))
                                        and isinstance(sm, (int, float))
                                    ):
                                        diffs[f"{k}_{metric}_diff"] = round(
                                            float(sm - bm), 6
                                        )
                            elif isinstance(b_val, (int, float)) and isinstance(s_val, (int, float)):
                                diffs[k] = round(float(s_val - b_val), 6)

                if diffs:
                    comparison[key] = diffs

        # Compare sample sizes
        base_n = base.get("parameters", {}).get("n_observations")
        sens_n = sensitivity.get("parameters", {}).get("n_observations")
        if isinstance(base_n, (int, float)) and isinstance(sens_n, (int, float)):
            comparison["n_observations"] = {
                "base": base_n,
                "sensitivity": sens_n,
                "difference": sens_n - base_n,
            }

        if not comparison:
            comparison["note"] = "Automated comparison not available; review results manually."

        return comparison
