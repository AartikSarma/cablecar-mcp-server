"""Efficient subgroup analysis using cohort.subgroup() without data reload."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class SubgroupAnalysis(BaseAnalysis):
    """Run an analysis across subgroups of a cohort.

    For each unique value of the *subgroup_variable*, creates a subgroup
    cohort (via :meth:`Cohort.subgroup` -- no data reload) and runs the
    specified analysis on it.  Reports per-subgroup results and an
    interaction p-value testing whether the effect differs across
    subgroups.

    Accepts either a callable ``analysis_fn`` or an ``analysis_type``
    string that dispatches to a built-in analysis class (currently
    ``"regression"``).
    """

    def run(
        self,
        subgroup_variable: str,
        analysis_fn: Callable | None = None,
        analysis_type: str | None = None,
        table: str = "hospitalization",
        **analysis_kwargs: Any,
    ) -> AnalysisResult:
        """Run subgroup analysis.

        Parameters
        ----------
        subgroup_variable:
            Column name whose unique values define the subgroups.
        analysis_fn:
            A callable that takes a :class:`~cablecar.data.cohort.Cohort`
            and returns an :class:`AnalysisResult`.  Mutually exclusive
            with *analysis_type*.
        analysis_type:
            String shorthand for a built-in analysis class.  Currently
            supported: ``"regression"`` (dispatches to
            :class:`RegressionAnalysis`).
        table:
            Cohort table containing *subgroup_variable*.
        **analysis_kwargs:
            Additional keyword arguments forwarded to the analysis
            function or class.

        Returns
        -------
        AnalysisResult
            Contains ``subgroup_results`` (per-group) and an
            ``interaction_test`` comparing effects across subgroups.
        """
        self._warnings = []

        # Resolve analysis function
        fn = self._resolve_analysis_fn(analysis_fn, analysis_type, **analysis_kwargs)
        if fn is None:
            return AnalysisResult(
                analysis_type="subgroup",
                parameters={"subgroup_variable": subgroup_variable},
                results={},
                warnings=self._warnings,
            )

        # Get subgroup values
        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="subgroup",
                parameters={"subgroup_variable": subgroup_variable},
                results={},
                warnings=self._warnings,
            )

        if subgroup_variable not in df.columns:
            self._warn(f"Subgroup variable '{subgroup_variable}' not found in table '{table}'.")
            return AnalysisResult(
                analysis_type="subgroup",
                parameters={"subgroup_variable": subgroup_variable},
                results={},
                warnings=self._warnings,
            )

        unique_values = sorted(df[subgroup_variable].dropna().unique(), key=str)
        if len(unique_values) < 2:
            self._warn(
                f"Subgroup variable '{subgroup_variable}' has fewer than 2 unique values; "
                "cannot perform subgroup analysis."
            )
            return AnalysisResult(
                analysis_type="subgroup",
                parameters={"subgroup_variable": subgroup_variable},
                results={"n_subgroups": len(unique_values)},
                warnings=self._warnings,
            )

        # Run analysis per subgroup
        subgroup_results: dict[str, Any] = {}
        for value in unique_values:
            label = str(value)
            try:
                sub_cohort = self.cohort.subgroup(
                    name=f"subgroup_{subgroup_variable}_{label}",
                    criteria=[{"column": subgroup_variable, "op": "==", "value": value}],
                )
                result = fn(sub_cohort)
                if isinstance(result, AnalysisResult):
                    subgroup_results[label] = result.to_dict()
                else:
                    subgroup_results[label] = result
            except Exception as exc:
                self._warn(f"Analysis failed for subgroup '{label}': {exc}")
                subgroup_results[label] = {"error": str(exc)}

        # Interaction test
        interaction = self._interaction_test(
            df, subgroup_variable, unique_values, analysis_type, **analysis_kwargs,
        )

        return AnalysisResult(
            analysis_type="subgroup",
            parameters={
                "subgroup_variable": subgroup_variable,
                "n_subgroups": len(unique_values),
                "subgroup_values": [str(v) for v in unique_values],
                "table": table,
            },
            results={
                "subgroup_results": subgroup_results,
                "interaction_test": interaction,
            },
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_analysis_fn(
        self,
        analysis_fn: Callable | None,
        analysis_type: str | None,
        **analysis_kwargs: Any,
    ) -> Callable | None:
        """Resolve a callable analysis function from the provided arguments."""
        if analysis_fn is not None:
            return analysis_fn

        if analysis_type == "regression":
            def _regression_fn(cohort: Any) -> AnalysisResult:
                from cablecar.analysis.regression import RegressionAnalysis

                return RegressionAnalysis(cohort).run(**analysis_kwargs)

            return _regression_fn

        if analysis_type is not None:
            self._warn(
                f"Unknown analysis_type '{analysis_type}'. "
                "Currently supported: 'regression'. "
                "Alternatively, pass a callable via analysis_fn."
            )
            return None

        self._warn("Either analysis_fn or analysis_type must be provided.")
        return None

    def _interaction_test(
        self,
        df: pd.DataFrame,
        subgroup_variable: str,
        unique_values: list,
        analysis_type: str | None,
        **analysis_kwargs: Any,
    ) -> dict[str, Any]:
        """Test for interaction between the subgroup variable and the exposure.

        For regression-based analyses, fits a model with an interaction
        term and reports its p-value.  For other analysis types, returns
        a descriptive note.
        """
        if analysis_type != "regression":
            return {
                "note": (
                    "Formal interaction test requires regression-based analysis. "
                    "Compare effect estimates and confidence intervals qualitatively."
                ),
            }

        outcome = analysis_kwargs.get("outcome")
        predictors = analysis_kwargs.get("predictors", [])

        if not outcome or not predictors:
            return {"note": "Cannot compute interaction: outcome or predictors not specified."}

        try:
            import statsmodels.api as sm

            # Build interaction model
            working = df[[outcome, subgroup_variable] + predictors].dropna().copy()

            # Encode subgroup variable as dummy
            working = pd.get_dummies(
                working, columns=[subgroup_variable], drop_first=True, dtype=float,
            )

            # Identify the subgroup dummy columns
            subgroup_dummies = [
                c for c in working.columns
                if c.startswith(f"{subgroup_variable}_")
            ]

            # Create interaction terms between subgroup dummies and each predictor
            interaction_cols: list[str] = []
            for sd in subgroup_dummies:
                for pred in predictors:
                    if pred in working.columns and pd.api.types.is_numeric_dtype(working[pred]):
                        int_col = f"{sd}_x_{pred}"
                        working[int_col] = working[sd] * working[pred]
                        interaction_cols.append(int_col)

            if not interaction_cols:
                return {"note": "No numeric predictors for interaction term construction."}

            # Encode any remaining categorical predictors
            cat_remaining = [
                c for c in predictors
                if c in working.columns
                and (pd.api.types.is_object_dtype(working[c])
                     or pd.api.types.is_categorical_dtype(working[c]))
            ]
            if cat_remaining:
                working = pd.get_dummies(
                    working, columns=cat_remaining, drop_first=True, dtype=float,
                )

            # All predictor columns in the full model
            all_pred_cols = [
                c for c in working.columns if c != outcome
            ]

            y = working[outcome].astype(float)
            X_full = sm.add_constant(working[all_pred_cols].astype(float))

            # Reduced model (without interaction terms)
            reduced_pred_cols = [c for c in all_pred_cols if c not in interaction_cols]
            X_reduced = sm.add_constant(working[reduced_pred_cols].astype(float))

            # Fit both models
            unique_outcome = y.unique()
            if len(unique_outcome) == 2:
                full_model = sm.Logit(y, X_full).fit(disp=0, maxiter=100)
                reduced_model = sm.Logit(y, X_reduced).fit(disp=0, maxiter=100)
                # Likelihood ratio test
                lr_stat = -2 * (reduced_model.llf - full_model.llf)
            else:
                full_model = sm.OLS(y, X_full).fit()
                reduced_model = sm.OLS(y, X_reduced).fit()
                # F-test via SSR comparison
                lr_stat = (
                    (reduced_model.ssr - full_model.ssr) / len(interaction_cols)
                ) / (full_model.ssr / full_model.df_resid) if full_model.df_resid > 0 else 0

            from scipy import stats as sp_stats

            dof = len(interaction_cols)
            p_value = float(1 - sp_stats.chi2.cdf(lr_stat, dof))

            return {
                "test": "likelihood_ratio" if len(unique_outcome) == 2 else "f_test_approx",
                "statistic": round(float(lr_stat), 4),
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "interaction_significant": p_value < 0.05,
                "interaction_terms": interaction_cols,
            }

        except Exception as exc:
            self._warn(f"Interaction test failed: {exc}")
            return {"error": str(exc)}
