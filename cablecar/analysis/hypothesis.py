"""Hypothesis testing for clinical research with multiple-comparison corrections."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class HypothesisTest(BaseAnalysis):
    """Run hypothesis tests on cohort data with automatic test selection.

    Supports both explicit test selection and automatic detection based on
    the data types and number of groups.

    Auto-detection rules:
    - 2 groups, numeric outcome  -> Mann-Whitney U
    - 2 groups, categorical      -> Chi-square (or Fisher's exact if any
      expected cell < 5)
    - 3+ groups, numeric outcome -> Kruskal-Wallis H

    Multiple-testing corrections: Bonferroni, Benjamini-Hochberg (FDR).
    """

    _SUPPORTED_TESTS = {
        "t_test",
        "mann_whitney",
        "chi_square",
        "fisher_exact",
        "kruskal",
        "anova",
        "auto",
    }

    def run(
        self,
        variable: str,
        group_variable: str,
        test: str = "auto",
        table: str = "hospitalization",
        correction: str | None = "bonferroni",
        **kwargs: Any,
    ) -> AnalysisResult:
        """Execute a hypothesis test.

        Parameters
        ----------
        variable:
            The outcome / dependent variable column name.
        group_variable:
            The grouping / independent variable column name.
        test:
            Test to run. ``"auto"`` selects the most appropriate test.
        table:
            Name of the cohort table.
        correction:
            Multiple-testing correction method.  ``"bonferroni"`` or
            ``"fdr"`` (Benjamini-Hochberg).  ``None`` to skip.

        Returns
        -------
        AnalysisResult
        """
        self._warnings = []

        if test not in self._SUPPORTED_TESTS:
            self._warn(
                f"Unknown test '{test}'. Supported: {sorted(self._SUPPORTED_TESTS)}. "
                "Falling back to 'auto'."
            )
            test = "auto"

        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="hypothesis_test",
                parameters={"variable": variable, "group_variable": group_variable, "test": test},
                results={},
                warnings=self._warnings,
            )

        # Validate columns
        for col_name, col_label in [(variable, "variable"), (group_variable, "group_variable")]:
            if col_name not in df.columns:
                self._warn(f"{col_label} '{col_name}' not found in table '{table}'.")
                return AnalysisResult(
                    analysis_type="hypothesis_test",
                    parameters={"variable": variable, "group_variable": group_variable, "test": test},
                    results={},
                    warnings=self._warnings,
                )

        # Drop rows with missing values in either column
        working = df[[variable, group_variable]].dropna()
        if len(working) == 0:
            self._warn("No non-missing observations available for testing.")
            return AnalysisResult(
                analysis_type="hypothesis_test",
                parameters={"variable": variable, "group_variable": group_variable, "test": test},
                results={},
                warnings=self._warnings,
            )

        groups = working[group_variable].unique().tolist()
        n_groups = len(groups)

        if n_groups < 2:
            self._warn(f"Only {n_groups} group(s) found; need at least 2 for testing.")
            return AnalysisResult(
                analysis_type="hypothesis_test",
                parameters={"variable": variable, "group_variable": group_variable, "test": test},
                results={"n_groups": n_groups},
                warnings=self._warnings,
            )

        is_numeric = pd.api.types.is_numeric_dtype(working[variable])

        # Auto-detect test
        if test == "auto":
            test = self._auto_select_test(working, variable, group_variable, is_numeric, n_groups)

        # Run test
        result = self._run_test(test, working, variable, group_variable, groups, is_numeric)

        # Multiple-testing correction (applied when caller runs several
        # tests in sequence -- here we store the raw p and note the method)
        if correction is not None and result.get("p_value") is not None:
            result["correction_method"] = correction
            result["p_value_raw"] = result["p_value"]
            # Correction is meaningful when batching; for a single test
            # Bonferroni with n_tests=1 is identity.
            n_tests = kwargs.get("n_tests", 1)
            if correction == "bonferroni" and n_tests > 1:
                result["p_value_corrected"] = min(result["p_value"] * n_tests, 1.0)
            elif correction == "fdr" and n_tests > 1:
                # Single-test FDR is the same value; true FDR needs a batch.
                result["p_value_corrected"] = result["p_value"]
                self._warn(
                    "FDR correction is most meaningful when applied across a "
                    "batch of tests. Consider using batch_correct() after "
                    "collecting all p-values."
                )
            else:
                result["p_value_corrected"] = result["p_value"]

        return AnalysisResult(
            analysis_type="hypothesis_test",
            parameters={
                "variable": variable,
                "group_variable": group_variable,
                "test": test,
                "table": table,
                "correction": correction,
            },
            results=result,
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def _auto_select_test(
        self,
        df: pd.DataFrame,
        variable: str,
        group_variable: str,
        is_numeric: bool,
        n_groups: int,
    ) -> str:
        """Select the most appropriate test based on data characteristics."""
        if is_numeric:
            if n_groups == 2:
                return "mann_whitney"
            else:
                return "kruskal"
        else:
            if n_groups == 2:
                # Check expected cell counts for Fisher's exact
                contingency = pd.crosstab(df[variable], df[group_variable])
                chi2, _, _, expected = sp_stats.chi2_contingency(contingency)
                if (expected < 5).any():
                    if contingency.shape == (2, 2):
                        return "fisher_exact"
                    else:
                        self._warn(
                            "Some expected cell counts < 5 but table is larger than 2x2; "
                            "Fisher's exact not available. Using chi-square with caution."
                        )
                return "chi_square"
            else:
                return "chi_square"

    # ------------------------------------------------------------------
    # Test runners
    # ------------------------------------------------------------------

    def _run_test(
        self,
        test: str,
        df: pd.DataFrame,
        variable: str,
        group_variable: str,
        groups: list,
        is_numeric: bool,
    ) -> dict[str, Any]:
        """Dispatch to the appropriate test runner."""
        group_data = [df.loc[df[group_variable] == g, variable] for g in groups]
        group_sizes = {str(g): len(gd) for g, gd in zip(groups, group_data)}

        result: dict[str, Any] = {
            "test_name": test,
            "groups": [str(g) for g in groups],
            "group_sizes": group_sizes,
        }

        try:
            if test == "t_test":
                result.update(self._t_test(group_data, groups))
            elif test == "mann_whitney":
                result.update(self._mann_whitney(group_data, groups))
            elif test == "chi_square":
                result.update(self._chi_square(df, variable, group_variable))
            elif test == "fisher_exact":
                result.update(self._fisher_exact(df, variable, group_variable))
            elif test == "kruskal":
                result.update(self._kruskal(group_data, groups))
            elif test == "anova":
                result.update(self._anova(group_data, groups))
            else:
                self._warn(f"Test '{test}' not implemented.")
        except Exception as exc:
            self._warn(f"Test '{test}' failed: {exc}")
            result["error"] = str(exc)

        return result

    def _t_test(self, group_data: list[pd.Series], groups: list) -> dict[str, Any]:
        """Independent-samples t-test (2 groups)."""
        if len(group_data) != 2:
            self._warn("t-test requires exactly 2 groups.")
            return {}
        a, b = group_data[0].astype(float), group_data[1].astype(float)
        stat, p = sp_stats.ttest_ind(a, b, equal_var=False)
        effect = self._cohens_d(a, b)
        ci_low, ci_high = self._mean_diff_ci(a, b)
        return {
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "effect_size": {"cohens_d": effect},
            "mean_difference_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

    def _mann_whitney(self, group_data: list[pd.Series], groups: list) -> dict[str, Any]:
        """Mann-Whitney U test (2 groups, non-parametric)."""
        if len(group_data) != 2:
            self._warn("Mann-Whitney requires exactly 2 groups.")
            return {}
        a, b = group_data[0].astype(float), group_data[1].astype(float)
        stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial correlation as effect size
        n1, n2 = len(a), len(b)
        r = 1 - (2 * stat) / (n1 * n2) if (n1 * n2) > 0 else None
        return {
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "effect_size": {
                "rank_biserial_r": round(float(r), 4) if r is not None else None,
            },
        }

    def _chi_square(
        self, df: pd.DataFrame, variable: str, group_variable: str,
    ) -> dict[str, Any]:
        """Chi-square test of independence."""
        contingency = pd.crosstab(df[variable], df[group_variable])
        chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)

        # Cramer's V
        n = contingency.sum().sum()
        k = min(contingency.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * k)) if (n * k) > 0 else None

        if (expected < 5).any():
            self._warn(
                "Some expected cell counts < 5. Chi-square results may be unreliable. "
                "Consider Fisher's exact test for 2x2 tables."
            )

        return {
            "statistic": round(float(chi2), 4),
            "p_value": float(p),
            "degrees_of_freedom": int(dof),
            "effect_size": {
                "cramers_v": round(float(cramers_v), 4) if cramers_v is not None else None,
            },
            "contingency_shape": list(contingency.shape),
        }

    def _fisher_exact(
        self, df: pd.DataFrame, variable: str, group_variable: str,
    ) -> dict[str, Any]:
        """Fisher's exact test (2x2 tables only)."""
        contingency = pd.crosstab(df[variable], df[group_variable])
        if contingency.shape != (2, 2):
            self._warn(
                f"Fisher's exact test requires a 2x2 table; got {contingency.shape}. "
                "Falling back to chi-square."
            )
            return self._chi_square(df, variable, group_variable)

        odds_ratio, p = sp_stats.fisher_exact(contingency)
        return {
            "statistic": round(float(odds_ratio), 4),
            "p_value": float(p),
            "effect_size": {"odds_ratio": round(float(odds_ratio), 4)},
        }

    def _kruskal(self, group_data: list[pd.Series], groups: list) -> dict[str, Any]:
        """Kruskal-Wallis H test (3+ groups, non-parametric)."""
        numeric_groups = [g.astype(float) for g in group_data]
        stat, p = sp_stats.kruskal(*numeric_groups)
        # Epsilon-squared as effect size
        n = sum(len(g) for g in numeric_groups)
        k = len(numeric_groups)
        epsilon_sq = (stat - k + 1) / (n - k) if (n - k) > 0 else None
        return {
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "effect_size": {
                "epsilon_squared": round(float(epsilon_sq), 4)
                if epsilon_sq is not None
                else None,
            },
        }

    def _anova(self, group_data: list[pd.Series], groups: list) -> dict[str, Any]:
        """One-way ANOVA."""
        numeric_groups = [g.astype(float) for g in group_data]
        stat, p = sp_stats.f_oneway(*numeric_groups)

        # Eta-squared
        grand_mean = np.concatenate([g.values for g in numeric_groups]).mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in numeric_groups
        )
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in numeric_groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else None

        return {
            "statistic": round(float(stat), 4),
            "p_value": float(p),
            "effect_size": {
                "eta_squared": round(float(eta_sq), 4) if eta_sq is not None else None,
            },
        }

    # ------------------------------------------------------------------
    # Effect-size helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_d(a: pd.Series, b: pd.Series) -> float | None:
        """Compute Cohen's d for two independent samples."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return None
        mean_diff = float(a.mean() - b.mean())
        pooled_std = math.sqrt(
            ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return None
        return round(mean_diff / pooled_std, 4)

    @staticmethod
    def _mean_diff_ci(
        a: pd.Series, b: pd.Series, alpha: float = 0.05,
    ) -> tuple[float, float]:
        """95% CI for the difference in means (Welch's method)."""
        n1, n2 = len(a), len(b)
        mean_diff = float(a.mean() - b.mean())
        se = math.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)

        # Welch-Satterthwaite degrees of freedom
        v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        dof = num / den if den > 0 else max(n1, n2) - 1

        t_crit = sp_stats.t.ppf(1 - alpha / 2, dof)
        return (mean_diff - t_crit * se, mean_diff + t_crit * se)

    # ------------------------------------------------------------------
    # Batch correction utility (class method)
    # ------------------------------------------------------------------

    @staticmethod
    def batch_correct(
        p_values: list[float], method: str = "bonferroni",
    ) -> list[float]:
        """Apply multiple-testing correction to a list of p-values.

        Parameters
        ----------
        p_values:
            Raw p-values.
        method:
            ``"bonferroni"`` or ``"fdr"`` (Benjamini-Hochberg).

        Returns
        -------
        list[float]
            Corrected p-values in the same order as input.
        """
        n = len(p_values)
        if n == 0:
            return []

        if method == "bonferroni":
            return [min(p * n, 1.0) for p in p_values]
        elif method == "fdr":
            # Benjamini-Hochberg procedure
            indexed = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected = [0.0] * n
            prev = 1.0
            for rank_idx in range(n - 1, -1, -1):
                orig_idx, p = indexed[rank_idx]
                rank = rank_idx + 1  # 1-based rank
                corrected_p = min(p * n / rank, prev)
                corrected_p = min(corrected_p, 1.0)
                corrected[orig_idx] = corrected_p
                prev = corrected_p
            return corrected
        else:
            return list(p_values)
