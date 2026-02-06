"""Descriptive statistics and Table 1 generation for clinical research."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class DescriptiveAnalysis(BaseAnalysis):
    """Compute descriptive statistics suitable for a publication-ready Table 1.

    For each requested variable the analysis produces:
    - Numeric variables: n, missing count/%, mean, std, median, IQR.
    - Categorical variables: n, missing count/%, counts and percentages per
      category.

    When *stratify_by* is provided, statistics are computed per group and
    overall, and standardised mean differences (SMD) are reported for
    numeric variables between the first two groups.
    """

    def run(
        self,
        variables: list[str],
        stratify_by: str | None = None,
        table: str = "hospitalization",
        **kwargs: Any,
    ) -> AnalysisResult:
        """Run descriptive analysis.

        Parameters
        ----------
        variables:
            Column names to describe.
        stratify_by:
            Optional column name to stratify by. Each unique value becomes a
            group.
        table:
            Name of the cohort table to pull data from.

        Returns
        -------
        AnalysisResult
            Results dict keyed by variable name, each containing summary
            statistics.
        """
        self._warnings = []

        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="descriptive",
                parameters={"variables": variables, "stratify_by": stratify_by, "table": table},
                results={},
                warnings=self._warnings,
            )

        # Validate requested variables exist
        available = set(df.columns)
        valid_vars: list[str] = []
        for v in variables:
            if v not in available:
                self._warn(f"Variable '{v}' not found in table '{table}'; skipping.")
            else:
                valid_vars.append(v)

        if stratify_by and stratify_by not in available:
            self._warn(
                f"Stratification variable '{stratify_by}' not found; "
                "falling back to unstratified analysis."
            )
            stratify_by = None

        results: dict[str, Any] = {}

        if stratify_by is None:
            # Unstratified
            for var in valid_vars:
                results[var] = self._describe_variable(df[var])
        else:
            groups = df[stratify_by].dropna().unique().tolist()
            groups.sort(key=str)

            # Overall
            overall: dict[str, Any] = {}
            for var in valid_vars:
                overall[var] = self._describe_variable(df[var])
            results["overall"] = overall

            # Per group
            for group in groups:
                group_df = df.loc[df[stratify_by] == group]
                group_results: dict[str, Any] = {}
                for var in valid_vars:
                    if var == stratify_by:
                        continue
                    group_results[var] = self._describe_variable(group_df[var])
                results[str(group)] = group_results

            # SMD between first two groups (if applicable)
            if len(groups) >= 2:
                g1 = df.loc[df[stratify_by] == groups[0]]
                g2 = df.loc[df[stratify_by] == groups[1]]
                smd_results: dict[str, float | None] = {}
                for var in valid_vars:
                    if var == stratify_by:
                        continue
                    smd = self._compute_smd(g1[var], g2[var])
                    smd_results[var] = smd
                results["smd"] = smd_results
                results["smd_groups"] = [str(groups[0]), str(groups[1])]

        return AnalysisResult(
            analysis_type="descriptive",
            parameters={
                "variables": valid_vars,
                "stratify_by": stratify_by,
                "table": table,
            },
            results=results,
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _describe_variable(self, series: pd.Series) -> dict[str, Any]:
        """Produce a summary dict for a single variable.

        Automatically detects whether the variable is numeric or categorical.
        """
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_pct = round(n_missing / n_total * 100, 1) if n_total > 0 else 0.0
        non_null = series.dropna()

        info: dict[str, Any] = {
            "n": n_total,
            "missing_count": n_missing,
            "missing_pct": missing_pct,
        }

        if self._is_numeric(series):
            info["type"] = "numeric"
            if len(non_null) == 0:
                info.update(
                    {"mean": None, "std": None, "median": None, "q1": None, "q3": None, "iqr": None}
                )
            else:
                desc = non_null.describe()
                info["mean"] = round(float(desc["mean"]), 4)
                info["std"] = round(float(desc["std"]), 4)
                info["median"] = round(float(desc["50%"]), 4)
                info["q1"] = round(float(desc["25%"]), 4)
                info["q3"] = round(float(desc["75%"]), 4)
                info["iqr"] = round(float(desc["75%"] - desc["25%"]), 4)
        else:
            info["type"] = "categorical"
            counts = non_null.value_counts()
            total_valid = len(non_null)
            categories: dict[str, dict] = {}
            for cat, count in counts.items():
                pct = round(count / total_valid * 100, 1) if total_valid > 0 else 0.0
                categories[str(cat)] = {"count": int(count), "pct": pct}
            info["categories"] = categories
            info["n_categories"] = len(categories)

        return info

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        """Determine whether a Series should be treated as numeric."""
        if pd.api.types.is_numeric_dtype(series):
            # Treat binary int columns (0/1 only) with few unique values as
            # categorical to match clinical convention.
            if pd.api.types.is_integer_dtype(series):
                unique = series.dropna().unique()
                if len(unique) <= 2 and set(unique).issubset({0, 1}):
                    return False
            return True
        return False

    @staticmethod
    def _compute_smd(a: pd.Series, b: pd.Series) -> float | None:
        """Compute the standardised mean difference between two groups.

        For numeric variables:
            SMD = (mean_a - mean_b) / sqrt((sd_a^2 + sd_b^2) / 2)

        Returns None for categorical or constant variables.
        """
        if not pd.api.types.is_numeric_dtype(a) or not pd.api.types.is_numeric_dtype(b):
            return None

        a_clean = a.dropna()
        b_clean = b.dropna()

        if len(a_clean) == 0 or len(b_clean) == 0:
            return None

        mean_a = float(a_clean.mean())
        mean_b = float(b_clean.mean())
        sd_a = float(a_clean.std(ddof=1))
        sd_b = float(b_clean.std(ddof=1))

        denominator = math.sqrt((sd_a**2 + sd_b**2) / 2)

        if denominator == 0:
            return None

        return round((mean_a - mean_b) / denominator, 4)
