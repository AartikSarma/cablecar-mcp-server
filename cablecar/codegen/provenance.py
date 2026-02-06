"""Analysis provenance tracking for reproducible code generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class AnalysisStep:
    """A single step in the analysis provenance chain."""
    step_name: str
    description: str
    parameters: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class AnalysisProvenance:
    """Complete provenance chain for reproducible analysis."""
    study_name: str = ""
    data_source: str = ""
    schema_name: str = ""
    cohort_definition: dict = field(default_factory=dict)
    steps: list[AnalysisStep] = field(default_factory=list)

    def add_step(self, name: str, description: str, parameters: dict | None = None):
        self.steps.append(AnalysisStep(
            step_name=name,
            description=description,
            parameters=parameters or {},
        ))

    def to_dict(self) -> dict:
        return {
            "study_name": self.study_name,
            "data_source": self.data_source,
            "schema_name": self.schema_name,
            "cohort_definition": self.cohort_definition,
            "steps": [
                {"name": s.step_name, "description": s.description,
                 "parameters": s.parameters, "timestamp": s.timestamp}
                for s in self.steps
            ],
        }
