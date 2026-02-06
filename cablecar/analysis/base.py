"""Base classes for all CableCar analyses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cablecar.data.cohort import Cohort


@dataclass
class AnalysisResult:
    """Standard result container for all analyses."""

    analysis_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parameters: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }


class BaseAnalysis(ABC):
    """Abstract base class for analyses. Every analysis takes a Cohort."""

    def __init__(self, cohort: Cohort):
        self.cohort = cohort
        self._warnings: list[str] = []

    @abstractmethod
    def run(self, **kwargs) -> AnalysisResult:
        """Execute the analysis and return results."""
        ...

    def _warn(self, msg: str):
        self._warnings.append(msg)
