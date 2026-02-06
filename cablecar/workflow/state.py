"""Immutable workflow state snapshots."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class WorkflowState:
    """Immutable snapshot of the research workflow state."""
    study_name: str = ""
    study_description: str = ""
    data_loaded: bool = False
    data_path: str = ""
    schema_name: str = ""
    cohort_defined: bool = False
    cohort_name: str = ""
    cohort_n: int = 0
    analyses_completed: tuple[str, ...] = ()
    current_step: str = "not_started"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: tuple[tuple[str, Any], ...] = ()

    def with_update(self, **kwargs) -> WorkflowState:
        """Return a new state with specified fields updated."""
        current = {
            "study_name": self.study_name,
            "study_description": self.study_description,
            "data_loaded": self.data_loaded,
            "data_path": self.data_path,
            "schema_name": self.schema_name,
            "cohort_defined": self.cohort_defined,
            "cohort_name": self.cohort_name,
            "cohort_n": self.cohort_n,
            "analyses_completed": self.analyses_completed,
            "current_step": self.current_step,
            "metadata": self.metadata,
        }
        current.update(kwargs)
        current["timestamp"] = datetime.now(timezone.utc).isoformat()
        return WorkflowState(**current)

    def add_analysis(self, analysis_name: str) -> WorkflowState:
        return self.with_update(analyses_completed=self.analyses_completed + (analysis_name,))

    def summary(self) -> dict:
        return {
            "study_name": self.study_name,
            "current_step": self.current_step,
            "data_loaded": self.data_loaded,
            "cohort_defined": self.cohort_defined,
            "cohort_n": self.cohort_n,
            "analyses_completed": list(self.analyses_completed),
            "timestamp": self.timestamp,
        }
