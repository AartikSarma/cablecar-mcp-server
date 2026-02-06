"""Survival analysis with Kaplan-Meier estimation and log-rank testing."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cablecar.analysis.base import BaseAnalysis, AnalysisResult


class SurvivalAnalysis(BaseAnalysis):
    """Kaplan-Meier survival analysis with optional group comparisons.

    Uses the ``lifelines`` library for Kaplan-Meier fitting and log-rank
    testing.  Reports survival probabilities at key clinical timepoints
    and median survival per group.
    """

    # Standard timepoints to report survival probabilities at
    _DEFAULT_TIMEPOINTS = [1, 7, 14, 28, 30, 60, 90]

    def run(
        self,
        time_col: str,
        event_col: str,
        group_col: str | None = None,
        table: str = "hospitalization",
        timepoints: list[int | float] | None = None,
        **kwargs: Any,
    ) -> AnalysisResult:
        """Run Kaplan-Meier survival analysis.

        Parameters
        ----------
        time_col:
            Column name for the time-to-event variable.
        event_col:
            Column name for the event indicator (1 = event, 0 = censored).
        group_col:
            Optional column to stratify survival curves by.
        table:
            Cohort table to use.
        timepoints:
            Custom timepoints at which to report survival probabilities.
            Defaults to days [1, 7, 14, 28, 30, 60, 90].

        Returns
        -------
        AnalysisResult
        """
        self._warnings = []

        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            self._warn("lifelines is required for survival analysis.")
            return AnalysisResult(
                analysis_type="survival",
                parameters={"time_col": time_col, "event_col": event_col},
                results={"error": "lifelines not installed"},
                warnings=self._warnings,
            )

        try:
            df = self.cohort.get_table(table)
        except KeyError:
            self._warn(f"Table '{table}' not found in cohort.")
            return AnalysisResult(
                analysis_type="survival",
                parameters={"time_col": time_col, "event_col": event_col},
                results={},
                warnings=self._warnings,
            )

        # Validate columns
        required = [time_col, event_col]
        if group_col:
            required.append(group_col)
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            self._warn(f"Columns not found: {missing_cols}")
            return AnalysisResult(
                analysis_type="survival",
                parameters={"time_col": time_col, "event_col": event_col},
                results={},
                warnings=self._warnings,
            )

        working = df[required].dropna().copy()
        if len(working) == 0:
            self._warn("No complete cases available for survival analysis.")
            return AnalysisResult(
                analysis_type="survival",
                parameters={"time_col": time_col, "event_col": event_col},
                results={},
                warnings=self._warnings,
            )

        report_timepoints = timepoints if timepoints is not None else self._DEFAULT_TIMEPOINTS
        # Filter timepoints to within observed range
        max_time = float(working[time_col].max())
        report_timepoints = [t for t in report_timepoints if t <= max_time]
        if not report_timepoints:
            report_timepoints = [max_time]
            self._warn(
                f"All default timepoints exceed maximum observed time ({max_time}). "
                f"Reporting at max time only."
            )

        T = working[time_col].astype(float)
        E = working[event_col].astype(float)

        results: dict[str, Any] = {}

        if group_col is None:
            # Overall (ungrouped)
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, label="overall")

            results["overall"] = self._extract_km_results(
                kmf, report_timepoints, "overall",
            )
            results["n_subjects"] = len(working)
            results["n_events"] = int(E.sum())
        else:
            # Per-group
            groups = sorted(working[group_col].unique(), key=str)
            group_results: dict[str, Any] = {}
            group_fitters: dict[str, KaplanMeierFitter] = {}

            for g in groups:
                mask = working[group_col] == g
                T_g = T[mask]
                E_g = E[mask]

                if len(T_g) == 0:
                    self._warn(f"Group '{g}' has no observations; skipping.")
                    continue

                kmf = KaplanMeierFitter()
                kmf.fit(T_g, event_observed=E_g, label=str(g))
                group_fitters[str(g)] = kmf

                group_results[str(g)] = self._extract_km_results(
                    kmf, report_timepoints, str(g),
                )
                group_results[str(g)]["n_subjects"] = int(mask.sum())
                group_results[str(g)]["n_events"] = int(E_g.sum())

            results["groups"] = group_results

            # Log-rank test (pairwise for 2 groups, overall for 3+)
            if len(group_fitters) >= 2:
                group_names = list(group_fitters.keys())

                if len(group_names) == 2:
                    g1, g2 = group_names
                    mask1 = working[group_col].astype(str) == g1
                    mask2 = working[group_col].astype(str) == g2

                    try:
                        lr = logrank_test(
                            T[mask1], T[mask2], E[mask1], E[mask2],
                        )
                        results["log_rank_test"] = {
                            "test_statistic": round(float(lr.test_statistic), 4),
                            "p_value": float(lr.p_value),
                            "groups_compared": [g1, g2],
                        }
                    except Exception as exc:
                        self._warn(f"Log-rank test failed: {exc}")
                else:
                    # Pairwise log-rank for 3+ groups
                    pairwise_results: list[dict[str, Any]] = []
                    for i in range(len(group_names)):
                        for j in range(i + 1, len(group_names)):
                            g_i, g_j = group_names[i], group_names[j]
                            mask_i = working[group_col].astype(str) == g_i
                            mask_j = working[group_col].astype(str) == g_j
                            try:
                                lr = logrank_test(
                                    T[mask_i], T[mask_j], E[mask_i], E[mask_j],
                                )
                                pairwise_results.append({
                                    "groups": [g_i, g_j],
                                    "test_statistic": round(float(lr.test_statistic), 4),
                                    "p_value": float(lr.p_value),
                                })
                            except Exception as exc:
                                self._warn(
                                    f"Log-rank test between '{g_i}' and '{g_j}' failed: {exc}"
                                )
                    results["log_rank_tests_pairwise"] = pairwise_results

            results["n_subjects"] = len(working)
            results["n_events"] = int(E.sum())

        return AnalysisResult(
            analysis_type="survival",
            parameters={
                "time_col": time_col,
                "event_col": event_col,
                "group_col": group_col,
                "table": table,
                "timepoints": report_timepoints,
            },
            results=results,
            warnings=self._warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_km_results(
        kmf: Any,
        timepoints: list[int | float],
        label: str,
    ) -> dict[str, Any]:
        """Extract key results from a fitted KaplanMeierFitter."""
        # Median survival
        median = kmf.median_survival_time_
        median_val: float | str
        if np.isinf(median):
            median_val = "not reached"
        else:
            median_val = round(float(median), 2)

        # Survival probabilities at requested timepoints
        survival_at: dict[str, float | None] = {}
        for t in timepoints:
            try:
                # Use the survival function; find nearest time <= t
                sf = kmf.survival_function_at_times([t])
                prob = float(sf.iloc[0])
                survival_at[str(t)] = round(prob, 4)
            except Exception:
                survival_at[str(t)] = None

        # Confidence intervals at timepoints
        ci_at: dict[str, dict[str, float | None]] = {}
        try:
            ci = kmf.confidence_interval_survival_function_
            for t in timepoints:
                # Find nearest time in the CI index
                if len(ci) > 0:
                    idx = ci.index.get_indexer([t], method="ffill")
                    if idx[0] >= 0:
                        row = ci.iloc[idx[0]]
                        ci_at[str(t)] = {
                            "lower": round(float(row.iloc[0]), 4),
                            "upper": round(float(row.iloc[1]), 4),
                        }
                    else:
                        ci_at[str(t)] = {"lower": None, "upper": None}
        except Exception:
            pass

        return {
            "label": label,
            "median_survival_time": median_val,
            "survival_probabilities": survival_at,
            "confidence_intervals": ci_at if ci_at else None,
        }
