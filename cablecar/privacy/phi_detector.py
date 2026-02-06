from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class PHIMatch:
    """Represents a single detected PHI occurrence."""
    phi_type: str
    value: str
    start: int
    end: int


# Built-in regex patterns for common PHI types.
_BUILTIN_PATTERNS: dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "MRN_labeled": r"\bMRN[:\s]*\d{6,10}\b",
    "MRN_coded": r"\b[A-Z]{1,3}\d{6,10}\b",
    "PHONE": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "DOB": (
        r"\b(?:DOB|Date of Birth|Birth Date)"
        r"[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    ),
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


class PHIDetector:
    """Detect and redact Protected Health Information (PHI) in text and data.

    Uses a set of built-in regular-expression patterns to identify common PHI
    types (SSN, MRN, phone numbers, email addresses, dates of birth, IP
    addresses).  Additional patterns can be supplied at construction time.
    """

    def __init__(self, extra_patterns: dict[str, str] | None = None) -> None:
        """Initialize the detector.

        Parameters
        ----------
        extra_patterns:
            Optional mapping of ``phi_type`` -> regex pattern string to extend
            the built-in set.
        """
        self._patterns: dict[str, re.Pattern] = {}
        all_patterns = dict(_BUILTIN_PATTERNS)
        if extra_patterns:
            all_patterns.update(extra_patterns)
        for name, pattern in all_patterns.items():
            self._patterns[name] = re.compile(pattern, re.IGNORECASE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_text(self, text: str) -> list[PHIMatch]:
        """Scan *text* and return all PHI matches found.

        Returns a list of :class:`PHIMatch` instances sorted by their start
        position in the string.
        """
        matches: list[PHIMatch] = []
        for phi_type, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                matches.append(
                    PHIMatch(
                        phi_type=phi_type,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        matches.sort(key=lambda m: m.start)
        return matches

    def scan_dataframe(
        self,
        df: pd.DataFrame,
        phi_columns: list[str] | None = None,
    ) -> list[PHIMatch]:
        """Scan a DataFrame for PHI in column names and cell values.

        Parameters
        ----------
        df:
            The DataFrame to scan.
        phi_columns:
            If provided, only these columns are scanned for cell-level PHI.
            Column *names* are always scanned regardless of this parameter.

        Returns
        -------
        list[PHIMatch]
            All detected PHI occurrences.  ``start`` and ``end`` offsets refer
            to positions within the string representation of the value that
            matched.
        """
        matches: list[PHIMatch] = []

        # Scan column names.
        for col in df.columns:
            matches.extend(self.scan_text(str(col)))

        # Scan cell values.
        columns_to_scan = phi_columns if phi_columns is not None else list(df.columns)
        for col in columns_to_scan:
            if col not in df.columns:
                continue
            for value in df[col].dropna().unique():
                text = str(value)
                matches.extend(self.scan_text(text))

        return matches

    def redact_text(self, text: str) -> str:
        """Return a copy of *text* with all detected PHI replaced.

        Each occurrence is replaced by ``[REDACTED-{type}]`` where *type* is
        the PHI category (e.g. ``SSN``, ``EMAIL``).
        """
        # Process matches in reverse order so that earlier indices stay valid
        # after replacement.
        matches = self.scan_text(text)
        # Deduplicate overlapping matches by keeping the longest span for each
        # position.
        result = text
        for match in reversed(matches):
            replacement = f"[REDACTED-{match.phi_type}]"
            result = result[: match.start] + replacement + result[match.end:]
        return result

    def contains_phi(self, text: str) -> bool:
        """Quick check: does *text* contain any detectable PHI?"""
        for pattern in self._patterns.values():
            if pattern.search(text):
                return True
        return False
