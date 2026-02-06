"""Machine learning prediction models with cross-validation and calibration."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class PredictionModel(BaseAnalysis):
    """Build and evaluate classification models for clinical prediction.

    Supports logistic regression, random forest, and gradient boosting.
    Reports AUROC, AUPRC, accuracy, sensitivity, specificity, PPV, NPV
    via stratified k-fold cross-validation.  Also provides feature
    importances and decile-based calibration metrics.
    """

    _SUPPORTED_MODELS = {"logistic", "random_forest", "gradient_boosting"}

    def run(
        self,
        outcome: str,
        features: list[str],
        model_type: str = "random_forest",
        table: str = "hospitalization",
        cv_folds: int = 5,
        **kwargs: Any,
    ) -> AnalysisResult:
        """Fit and cross-validate a prediction model.

        Parameters
        ----------
        outcome:
            Binary outcome column name.
        features:
            Feature column names.
        model_type:
            One of ``"logistic"``, ``"random_forest"``, ``"gradient_boosting"``.
        table:
            Cohort table to use.
        cv_folds:
            Number of stratified cross-validation folds.

        Returns
        -------
        AnalysisResult
        """
        self._warnings = []

        if model_type not in self._SUPPORTED_MODELS:
            self._warn(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {sorted(self._SUPPORTED_MODELS)}. "
                "Falling back to 'random_forest'."
            )
            model_type = "random_forest"

        try:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import (
                roc_auc_score,
                average_precision_score,
                accuracy_score,
                confusion_matrix,
            )
        except ImportError:
            self._warn("scikit-learn is required for prediction models.")
            return AnalysisResult(
                analysis_type="prediction",
                parameters={"outcome": outcome, "features": features, "model_type": model_type},
                results={"error": "scikit-learn not installed"},
                warnings=self._warnings,
            )

        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="prediction",
                parameters={"outcome": outcome, "features": features, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        # Validate columns
        required = [outcome] + features
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            self._warn(f"Columns not found: {missing_cols}")
            return AnalysisResult(
                analysis_type="prediction",
                parameters={"outcome": outcome, "features": features, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        working = df[required].dropna().copy()
        if len(working) == 0:
            self._warn("No complete cases available.")
            return AnalysisResult(
                analysis_type="prediction",
                parameters={"outcome": outcome, "features": features, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        # Dummy-encode categoricals
        cat_cols = [
            c for c in features
            if pd.api.types.is_object_dtype(working[c])
            or pd.api.types.is_categorical_dtype(working[c])
        ]
        if cat_cols:
            working = pd.get_dummies(working, columns=cat_cols, drop_first=True, dtype=float)

        encoded_features = [c for c in working.columns if c != outcome]
        X = working[encoded_features].values.astype(float)
        y = working[outcome].values.astype(float)

        # Check outcome is binary
        unique_y = np.unique(y[~np.isnan(y)])
        if len(unique_y) != 2:
            self._warn(
                f"Prediction model expects binary outcome; found {len(unique_y)} unique values."
            )

        # Get model
        estimator = self._get_estimator(model_type)

        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_metrics: list[dict[str, Any]] = []
        all_y_true: list[np.ndarray] = []
        all_y_prob: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                estimator.fit(X_train, y_train)
                y_prob = estimator.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            except Exception as exc:
                self._warn(f"Fold {fold_idx + 1} failed: {exc}")
                continue

            all_y_true.append(y_test)
            all_y_prob.append(y_prob)

            # Confusion matrix -> sensitivity, specificity, PPV, NPV
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            fold_metrics.append({
                "fold": fold_idx + 1,
                "auroc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "auprc": round(float(average_precision_score(y_test, y_prob)), 4),
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "sensitivity": round(float(sensitivity), 4),
                "specificity": round(float(specificity), 4),
                "ppv": round(float(ppv), 4),
                "npv": round(float(npv), 4),
                "n_train": len(y_train),
                "n_test": len(y_test),
            })

        if not fold_metrics:
            self._warn("All cross-validation folds failed.")
            return AnalysisResult(
                analysis_type="prediction",
                parameters={"outcome": outcome, "features": features, "model_type": model_type},
                results={},
                warnings=self._warnings,
            )

        # Aggregate metrics
        metric_names = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "ppv", "npv"]
        aggregate: dict[str, dict[str, float]] = {}
        for m in metric_names:
            values = [f[m] for f in fold_metrics]
            aggregate[m] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
            }

        # Feature importance (refit on full data)
        try:
            estimator.fit(X, y)
            importance = self._get_feature_importance(estimator, encoded_features, model_type)
        except Exception as exc:
            self._warn(f"Feature importance computation failed: {exc}")
            importance = {}

        # Calibration (decile-based Hosmer-Lemeshow style)
        calibration = {}
        if all_y_true and all_y_prob:
            combined_true = np.concatenate(all_y_true)
            combined_prob = np.concatenate(all_y_prob)
            calibration = self._compute_calibration(combined_true, combined_prob)

        return AnalysisResult(
            analysis_type="prediction",
            parameters={
                "outcome": outcome,
                "features": features,
                "encoded_features": encoded_features,
                "model_type": model_type,
                "table": table,
                "cv_folds": cv_folds,
                "n_observations": len(working),
            },
            results={
                "fold_metrics": fold_metrics,
                "aggregate_metrics": aggregate,
                "feature_importance": importance,
                "calibration": calibration,
            },
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Estimator factory
    # ------------------------------------------------------------------

    @staticmethod
    def _get_estimator(model_type: str) -> Any:
        """Return a scikit-learn estimator for the given model type."""
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=42,
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @staticmethod
    def _get_feature_importance(
        estimator: Any, feature_names: list[str], model_type: str,
    ) -> dict[str, float]:
        """Extract feature importances from a fitted estimator."""
        importance: dict[str, float] = {}

        if model_type == "logistic":
            coefs = estimator.coef_[0]
            for name, coef in zip(feature_names, coefs):
                importance[name] = round(float(abs(coef)), 4)
        elif hasattr(estimator, "feature_importances_"):
            for name, imp in zip(feature_names, estimator.feature_importances_):
                importance[name] = round(float(imp), 4)

        # Sort by importance descending
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        return importance

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
    ) -> dict[str, Any]:
        """Compute decile-based calibration statistics.

        Returns observed vs predicted event rates per decile, plus a
        Hosmer-Lemeshow-style chi-squared statistic.
        """
        # Sort into deciles by predicted probability
        order = np.argsort(y_prob)
        y_true_sorted = y_true[order]
        y_prob_sorted = y_prob[order]

        bin_size = len(y_true_sorted) // n_bins
        if bin_size == 0:
            return {"error": "Too few observations for calibration"}

        deciles: list[dict[str, Any]] = []
        hl_chi2 = 0.0

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(y_true_sorted)
            bin_true = y_true_sorted[start:end]
            bin_prob = y_prob_sorted[start:end]

            observed = float(bin_true.mean())
            predicted = float(bin_prob.mean())
            n = len(bin_true)

            deciles.append({
                "decile": i + 1,
                "n": n,
                "observed_rate": round(observed, 4),
                "predicted_rate": round(predicted, 4),
            })

            # Hosmer-Lemeshow contribution
            expected_events = predicted * n
            expected_non = (1 - predicted) * n
            obs_events = bin_true.sum()
            obs_non = n - obs_events

            if expected_events > 0:
                hl_chi2 += (obs_events - expected_events) ** 2 / expected_events
            if expected_non > 0:
                hl_chi2 += (obs_non - expected_non) ** 2 / expected_non

        # p-value with (n_bins - 2) degrees of freedom
        from scipy import stats as sp_stats

        dof = max(n_bins - 2, 1)
        hl_p = float(1 - sp_stats.chi2.cdf(hl_chi2, dof))

        return {
            "deciles": deciles,
            "hosmer_lemeshow_chi2": round(float(hl_chi2), 4),
            "hosmer_lemeshow_p": hl_p,
            "hosmer_lemeshow_dof": dof,
            "well_calibrated": hl_p > 0.05,
        }
