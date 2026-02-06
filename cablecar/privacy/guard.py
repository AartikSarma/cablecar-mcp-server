from __future__ import annotations

from typing import Any

import pandas as pd

from cablecar.privacy.policy import PrivacyPolicy
from cablecar.privacy.phi_detector import PHIDetector


class PrivacyGuard:
    """Core privacy boundary that sanitizes ALL output before it reaches an LLM.

    Every analysis result, DataFrame summary, or free-text string must pass
    through :meth:`sanitize_for_llm` before being returned to the caller.
    The guard enforces cell suppression, PHI redaction, and audit logging.
    """

    def __init__(
        self,
        policy: PrivacyPolicy | None = None,
        phi_columns: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the guard.

        Parameters
        ----------
        policy:
            A :class:`PrivacyPolicy` instance.  Defaults are used when
            ``None``.
        phi_columns:
            Mapping of ``table_name`` -> list of column names known to
            contain PHI.  Used by :meth:`sanitize_dataframe_summary` to
            apply targeted redaction.
        """
        self.policy = policy or PrivacyPolicy()
        self.phi_columns = phi_columns or {}
        self._detector = PHIDetector()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sanitize_for_llm(self, data: Any, context: str = "") -> dict:
        """Sanitize *data* so it is safe to send to an LLM.

        This is the **critical privacy boundary**.  It:
        - Never returns raw patient-level data.
        - Applies cell suppression to all counts.
        - Redacts any detected PHI.
        - Adds an audit note describing what was sanitized.
        - Always returns a ``dict`` with ``"sanitized": True``.

        Parameters
        ----------
        data:
            The analysis output to sanitize.  Supported types are ``dict``,
            ``pd.DataFrame``, and ``str``.
        context:
            Optional label describing the source (e.g. ``"table1"``).

        Returns
        -------
        dict
            ``{"sanitized": True, "privacy_notes": [...], "data": <safe>}``
        """
        privacy_notes: list[str] = []

        if isinstance(data, pd.DataFrame):
            safe_data = self.sanitize_dataframe_summary(
                data, name=context, _notes=privacy_notes,
            )
        elif isinstance(data, dict):
            safe_data = self._sanitize_dict(data, privacy_notes)
        elif isinstance(data, str):
            safe_data = self._sanitize_string(data, privacy_notes)
        else:
            # Fallback: convert to string and sanitize.
            safe_data = self._sanitize_string(str(data), privacy_notes)

        return {
            "sanitized": True,
            "context": context,
            "privacy_notes": privacy_notes,
            "data": safe_data,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def suppress_small_cells(
        self,
        counts: dict | pd.Series,
        label: str = "",
    ) -> dict:
        """Replace counts below ``min_cell_size`` with the suppress marker.

        Parameters
        ----------
        counts:
            A mapping or Series of category -> count.
        label:
            Optional label used in privacy notes.

        Returns
        -------
        dict
            The suppressed counts as a plain ``dict``.
        """
        if isinstance(counts, pd.Series):
            counts = counts.to_dict()

        suppressed: dict[str, Any] = {}
        suppressed_keys: list[str] = []
        for key, value in counts.items():
            if isinstance(value, (int, float)) and value < self.policy.min_cell_size:
                suppressed[key] = self.policy.suppress_marker
                suppressed_keys.append(str(key))
            else:
                suppressed[key] = value

        return suppressed

    def sanitize_dataframe_summary(
        self,
        df: pd.DataFrame,
        name: str = "",
        _notes: list[str] | None = None,
    ) -> dict:
        """Convert a DataFrame into a safe summary dict.

        The output contains shape, column metadata, summary statistics (with
        cell suppression), and missing-data counts.  **Raw rows are never
        included.**

        Parameters
        ----------
        df:
            The DataFrame to summarize.
        name:
            Optional label for audit / notes.
        """
        notes = _notes if _notes is not None else []
        notes.append(f"DataFrame '{name}' summarized; raw rows excluded.")

        n_rows, n_cols = df.shape

        # --- Column metadata ------------------------------------------------
        col_info: list[dict] = []
        for col in df.columns:
            info: dict[str, Any] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 1),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                stats: dict[str, Any] = {
                    "count": int(desc.get("count", 0)),
                    "mean": round(float(desc.get("mean", 0)), 2)
                    if self.policy.round_means
                    else float(desc.get("mean", 0)),
                    "std": round(float(desc.get("std", 0)), 2),
                    "median": float(desc.get("50%", 0)),
                    "q1": float(desc.get("25%", 0)),
                    "q3": float(desc.get("75%", 0)),
                }
                # Suppress min/max for small groups if configured.
                if (
                    self.policy.suppress_extreme_percentiles
                    and int(desc.get("count", 0)) < self.policy.min_cell_size
                ):
                    stats["min"] = self.policy.suppress_marker
                    stats["max"] = self.policy.suppress_marker
                    notes.append(
                        f"Column '{col}': min/max suppressed (n < {self.policy.min_cell_size})."
                    )
                else:
                    stats["min"] = float(desc.get("min", 0))
                    stats["max"] = float(desc.get("max", 0))
                info["stats"] = stats

            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                n_unique = df[col].nunique()
                info["n_unique"] = n_unique
                if n_unique <= self.policy.max_unique_categories:
                    raw_counts = df[col].value_counts()
                    safe_counts = self.suppress_small_cells(
                        raw_counts, label=col,
                    )
                    suppressed_count = sum(
                        1 for v in safe_counts.values()
                        if v == self.policy.suppress_marker
                    )
                    if suppressed_count:
                        notes.append(
                            f"Column '{col}': {suppressed_count} category "
                            f"count(s) suppressed (n < {self.policy.min_cell_size})."
                        )
                    info["value_counts"] = safe_counts
                else:
                    info["value_counts"] = (
                        f"High cardinality ({n_unique} unique values); counts omitted."
                    )
                    notes.append(
                        f"Column '{col}': high cardinality ({n_unique}); counts omitted."
                    )

            col_info.append(info)

        # --- PHI scan on column names ----------------------------------------
        if self.policy.redact_phi:
            for ci in col_info:
                if self._detector.contains_phi(ci["name"]):
                    ci["name"] = self._detector.redact_text(ci["name"])
                    notes.append(f"PHI detected in column name; redacted.")

        return {
            "name": name,
            "shape": {"rows": n_rows, "columns": n_cols},
            "columns": col_info,
        }

    def sanitize_analysis_result(self, result: dict) -> dict:
        """Sanitize an analysis result dict (e.g. regression output).

        Most numeric results (coefficients, p-values, CIs) pass through, but
        any embedded PHI strings are redacted, and counts are checked for
        suppression.
        """
        notes: list[str] = []
        safe = self._sanitize_dict(result, notes)
        return {
            "sanitized": True,
            "privacy_notes": notes,
            "data": safe,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_dict(self, d: dict, notes: list[str]) -> dict:
        """Recursively sanitize a dict."""
        safe: dict[str, Any] = {}
        for key, value in d.items():
            safe_key = self._maybe_redact_string(str(key), notes)
            if isinstance(value, dict):
                safe[safe_key] = self._sanitize_dict(value, notes)
            elif isinstance(value, list):
                safe[safe_key] = self._sanitize_list(value, notes)
            elif isinstance(value, pd.DataFrame):
                safe[safe_key] = self.sanitize_dataframe_summary(
                    value, name=safe_key, _notes=notes,
                )
            elif isinstance(value, str):
                safe[safe_key] = self._maybe_redact_string(value, notes)
            elif isinstance(value, (int, float)):
                safe[safe_key] = self._check_and_suppress(value)
            else:
                safe[safe_key] = value
        return safe

    def _sanitize_list(self, items: list, notes: list[str]) -> list:
        """Recursively sanitize a list."""
        safe: list[Any] = []
        for item in items:
            if isinstance(item, dict):
                safe.append(self._sanitize_dict(item, notes))
            elif isinstance(item, str):
                safe.append(self._maybe_redact_string(item, notes))
            elif isinstance(item, (int, float)):
                safe.append(self._check_and_suppress(item))
            else:
                safe.append(item)
        return safe

    def _sanitize_string(self, text: str, notes: list[str]) -> str:
        """Redact PHI from a string if policy requires it."""
        if not self.policy.redact_phi:
            return text
        if self._detector.contains_phi(text):
            notes.append("PHI detected and redacted from text output.")
            return self._detector.redact_text(text)
        return text

    def _maybe_redact_string(self, text: str, notes: list[str]) -> str:
        """Conditionally redact PHI from a string, appending a note if so."""
        if self.policy.redact_phi and self._detector.contains_phi(text):
            notes.append("PHI detected and redacted from text output.")
            return self._detector.redact_text(text)
        return text

    def _check_and_suppress(self, value: Any) -> Any:
        """Check if a numeric value represents a count needing suppression.

        Heuristic: integer values (or floats equal to their int cast) below
        ``min_cell_size`` are treated as potentially-identifiable counts and
        are suppressed.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if 0 < value < self.policy.min_cell_size:
                return self.policy.suppress_marker
        elif isinstance(value, float) and value == int(value):
            int_val = int(value)
            if 0 < int_val < self.policy.min_cell_size:
                return self.policy.suppress_marker
        return value
