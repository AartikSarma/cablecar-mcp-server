"""Regression analysis (linear, logistic, Cox PH) for clinical research."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class RegressionAnalysis(BaseAnalysis):
    """Fit regression models to cohort data.

    Supported model types:
    - ``"linear"``: OLS via statsmodels.
    - ``"logistic"``: Logistic regression via statsmodels.
    - ``"cox"``: Cox proportional-hazards via lifelines.

    Automatically expands categorical predictors using dummy encoding and
    reports model-appropriate diagnostics (VIF, residual normality, etc.).
    """

    _SUPPORTED_MODELS = {"linear", "logistic", "cox"}

    def run(
        self,
        outcome: str,
        predictors: list[str],
        model_type: str = "logistic",
        table: str = "hospitalization",
        confounders: list[str] | None = None,
        time_col: str | None = None,
        **kwargs: Any,
    ) -> AnalysisResult:
        """Fit a regression model.

        Parameters
        ----------
        outcome:
            Column name for the outcome / dependent variable.
        predictors:
            List of predictor column names.
        model_type:
            One of ``"linear"``, ``"logistic"``, ``"cox"``.
        table:
            Cohort table to use.
        confounders:
            Optional additional confounder columns to include in the model.
        time_col:
            Required for Cox models -- the time-to-event column.

        Returns
        -------
        AnalysisResult
        """
        self._warnings = []

        if model_type not in self._SUPPORTED_MODELS:
            self._warn(
                f"Unknown model_type '{model_type}'. Supported: {sorted(self._SUPPORTED_MODELS)}. "
                "Falling back to 'logistic'."
            )
            model_type = "logistic"

        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="regression",
                parameters={"outcome": outcome, "predictors": predictors, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        all_predictors = list(predictors)
        if confounders:
            all_predictors.extend([c for c in confounders if c not in all_predictors])

        # Validate columns
        required_cols = [outcome] + all_predictors
        if model_type == "cox" and time_col:
            required_cols.append(time_col)
        elif model_type == "cox" and time_col is None:
            self._warn("Cox model requires 'time_col'. Cannot proceed.")
            return AnalysisResult(
                analysis_type="regression",
                parameters={"outcome": outcome, "predictors": predictors, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self._warn(f"Columns not found in table: {missing_cols}")
            return AnalysisResult(
                analysis_type="regression",
                parameters={"outcome": outcome, "predictors": predictors, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        # Prepare design matrix
        working = df[required_cols].dropna().copy()
        if len(working) == 0:
            self._warn("No complete cases available for regression.")
            return AnalysisResult(
                analysis_type="regression",
                parameters={"outcome": outcome, "predictors": predictors, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        # Dummy-encode categorical predictors
        cat_cols = [
            c for c in all_predictors
            if pd.api.types.is_object_dtype(working[c])
            or pd.api.types.is_categorical_dtype(working[c])
        ]
        if cat_cols:
            working = pd.get_dummies(working, columns=cat_cols, drop_first=True, dtype=float)

        # Reconstruct predictor list after dummification
        encoded_predictors = [c for c in working.columns if c != outcome and c != time_col]

        if model_type == "linear":
            result = self._fit_linear(working, outcome, encoded_predictors)
        elif model_type == "logistic":
            result = self._fit_logistic(working, outcome, encoded_predictors)
        elif model_type == "cox":
            result = self._fit_cox(working, outcome, encoded_predictors, time_col)
        else:
            result = {}

        # Diagnostics: VIF for multicollinearity
        diagnostics: dict[str, Any] = {}
        if len(encoded_predictors) >= 3:
            diagnostics["vif"] = self._compute_vif(working, encoded_predictors)

        return AnalysisResult(
            analysis_type="regression",
            parameters={
                "outcome": outcome,
                "predictors": predictors,
                "confounders": confounders,
                "model_type": model_type,
                "table": table,
                "n_observations": len(working),
                "n_predictors": len(encoded_predictors),
            },
            results=result,
            diagnostics=diagnostics,
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Model fitters
    # ------------------------------------------------------------------

    def _fit_linear(
        self, df: pd.DataFrame, outcome: str, predictors: list[str],
    ) -> dict[str, Any]:
        """Fit OLS linear regression via statsmodels."""
        try:
            import statsmodels.api as sm
        except ImportError:
            self._warn("statsmodels is required for linear regression.")
            return {"error": "statsmodels not installed"}

        X = sm.add_constant(df[predictors].astype(float))
        y = df[outcome].astype(float)

        try:
            model = sm.OLS(y, X).fit()
        except Exception as exc:
            self._warn(f"OLS fitting failed: {exc}")
            return {"error": str(exc)}

        coef_table = self._extract_coef_table(model, transform="none")

        result: dict[str, Any] = {
            "coefficients": coef_table,
            "r_squared": round(float(model.rsquared), 4),
            "r_squared_adj": round(float(model.rsquared_adj), 4),
            "f_statistic": round(float(model.fvalue), 4) if model.fvalue is not None else None,
            "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else None,
            "aic": round(float(model.aic), 2),
            "bic": round(float(model.bic), 2),
            "n_observations": int(model.nobs),
        }

        # Residual normality test
        try:
            from scipy import stats as sp_stats

            resid = model.resid
            stat, p_norm = sp_stats.shapiro(resid[:5000])  # cap sample size
            result["residual_normality"] = {
                "shapiro_statistic": round(float(stat), 4),
                "shapiro_p_value": float(p_norm),
                "normal": p_norm > 0.05,
            }
            if p_norm <= 0.05:
                self._warn(
                    "Residuals deviate significantly from normality "
                    "(Shapiro-Wilk p <= 0.05). Consider robust standard errors "
                    "or a different model."
                )
        except Exception:
            pass

        return result

    def _fit_logistic(
        self, df: pd.DataFrame, outcome: str, predictors: list[str],
    ) -> dict[str, Any]:
        """Fit logistic regression via statsmodels."""
        try:
            import statsmodels.api as sm
        except ImportError:
            self._warn("statsmodels is required for logistic regression.")
            return {"error": "statsmodels not installed"}

        X = sm.add_constant(df[predictors].astype(float))
        y = df[outcome].astype(float)

        # Check that outcome is binary
        unique_vals = y.unique()
        if len(unique_vals) > 2:
            self._warn(
                f"Logistic regression expects a binary outcome but found "
                f"{len(unique_vals)} unique values. Results may be unreliable."
            )

        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        except Exception as exc:
            self._warn(f"Logistic regression fitting failed: {exc}")
            return {"error": str(exc)}

        coef_table = self._extract_coef_table(model, transform="or")

        result: dict[str, Any] = {
            "coefficients": coef_table,
            "pseudo_r_squared": round(float(model.prsquared), 4),
            "aic": round(float(model.aic), 2),
            "bic": round(float(model.bic), 2),
            "log_likelihood": round(float(model.llf), 2),
            "n_observations": int(model.nobs),
            "converged": bool(model.mle_retvals.get("converged", False)),
        }

        if not result["converged"]:
            self._warn("Logistic regression did not converge. Consider simplifying the model.")

        return result

    def _fit_cox(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: list[str],
        time_col: str | None,
    ) -> dict[str, Any]:
        """Fit Cox proportional-hazards model via lifelines."""
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            self._warn("lifelines is required for Cox regression.")
            return {"error": "lifelines not installed"}

        if time_col is None:
            self._warn("Cox model requires 'time_col'.")
            return {"error": "time_col is required for Cox models"}

        cox_df = df[[time_col, outcome] + predictors].copy().astype(float)

        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col=time_col, event_col=outcome)
        except Exception as exc:
            self._warn(f"Cox PH fitting failed: {exc}")
            return {"error": str(exc)}

        summary = cph.summary
        coef_table: dict[str, dict[str, Any]] = {}
        for var_name in summary.index:
            row = summary.loc[var_name]
            coef_table[str(var_name)] = {
                "coefficient": round(float(row["coef"]), 4),
                "hazard_ratio": round(float(row["exp(coef)"]), 4),
                "se": round(float(row["se(coef)"]), 4),
                "z": round(float(row["z"]), 4),
                "p_value": float(row["p"]),
                "ci_lower_95": round(float(row["exp(coef) lower 95%"]), 4),
                "ci_upper_95": round(float(row["exp(coef) upper 95%"]), 4),
            }

        result: dict[str, Any] = {
            "coefficients": coef_table,
            "concordance_index": round(float(cph.concordance_index_), 4),
            "log_likelihood": round(float(cph.log_likelihood_), 2),
            "n_observations": int(cph.summary.shape[0]),
            "n_events": int(cox_df[outcome].sum()),
        }

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_coef_table(
        model: Any, transform: str = "none",
    ) -> dict[str, dict[str, Any]]:
        """Extract a coefficient table from a statsmodels result.

        Parameters
        ----------
        model:
            A fitted statsmodels result.
        transform:
            ``"none"`` for raw coefficients, ``"or"`` for odds ratios.
        """
        table: dict[str, dict[str, Any]] = {}
        conf = model.conf_int()

        for i, name in enumerate(model.params.index):
            coef = float(model.params.iloc[i])
            se = float(model.bse.iloc[i])
            p = float(model.pvalues.iloc[i])
            ci_low = float(conf.iloc[i, 0])
            ci_high = float(conf.iloc[i, 1])

            entry: dict[str, Any] = {
                "coefficient": round(coef, 4),
                "se": round(se, 4),
                "p_value": p,
                "ci_lower_95": round(ci_low, 4),
                "ci_upper_95": round(ci_high, 4),
            }

            if transform == "or":
                entry["odds_ratio"] = round(np.exp(coef), 4)
                entry["or_ci_lower_95"] = round(np.exp(ci_low), 4)
                entry["or_ci_upper_95"] = round(np.exp(ci_high), 4)

            table[str(name)] = entry

        return table

    def _compute_vif(
        self, df: pd.DataFrame, predictors: list[str],
    ) -> dict[str, float]:
        """Compute Variance Inflation Factor for each predictor."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import statsmodels.api as sm
        except ImportError:
            self._warn("statsmodels required for VIF calculation.")
            return {}

        X = df[predictors].astype(float)
        X = sm.add_constant(X)

        vif_data: dict[str, float] = {}
        for i, col in enumerate(X.columns):
            if col == "const":
                continue
            try:
                vif_val = variance_inflation_factor(X.values, i)
                vif_data[col] = round(float(vif_val), 2)
                if vif_val > 10:
                    self._warn(
                        f"High multicollinearity: VIF for '{col}' = {vif_val:.1f} (>10)."
                    )
                elif vif_val > 5:
                    self._warn(
                        f"Moderate multicollinearity: VIF for '{col}' = {vif_val:.1f} (>5)."
                    )
            except Exception:
                vif_data[col] = float("nan")

        return vif_data
