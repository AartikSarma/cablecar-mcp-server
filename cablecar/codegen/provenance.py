"""Analysis provenance tracking for reproducible code generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AnalysisStep:
    """A single step in the analysis provenance chain."""
    step_name: str
    description: str
    parameters: dict = field(default_factory=dict)
    analysis_type: str = ""
    result_summary: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AnalysisProvenance:
    """Complete provenance chain for reproducible analysis."""
    study_name: str = ""
    data_source: str = ""
    schema_name: str = ""
    data_format: str = "csv"
    cohort_definition: dict = field(default_factory=dict)
    tables_used: list[str] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)

    def add_step(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
        analysis_type: str = "",
        result_summary: dict | None = None,
    ):
        self.steps.append(AnalysisStep(
            step_name=name,
            description=description,
            parameters=parameters or {},
            analysis_type=analysis_type,
            result_summary=result_summary or {},
        ))

    def to_dict(self) -> dict:
        """Backward-compatible dict representation.

        Used by STROBEReport.auto_populate() and TRIPODReport.auto_populate().
        Original keys are preserved; new keys are additive.
        """
        return {
            "study_name": self.study_name,
            "data_source": self.data_source,
            "schema_name": self.schema_name,
            "data_format": self.data_format,
            "cohort_definition": self.cohort_definition,
            "tables_used": self.tables_used,
            "steps": [
                {
                    "name": s.step_name,
                    "description": s.description,
                    "parameters": s.parameters,
                    "analysis_type": s.analysis_type,
                    "result_summary": s.result_summary,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
        }

    def to_scaffold_context(self) -> dict:
        """Return context dict for Jinja2 scaffold templates."""
        tables = self.tables_used if self.tables_used else self._infer_tables()
        read_fn = "read_parquet" if self.data_format == "parquet" else "read_csv"
        file_ext = "parquet" if self.data_format == "parquet" else "csv"

        return {
            "study_name": self.study_name,
            "data_source": self.data_source,
            "schema_name": self.schema_name,
            "data_format": self.data_format,
            "read_fn": read_fn,
            "file_ext": file_ext,
            "tables": tables,
            "cohort": self.cohort_definition,
            "steps": [
                {
                    "name": s.step_name,
                    "analysis_type": s.analysis_type,
                    "description": s.description,
                    "parameters": s.parameters,
                }
                for s in self.steps
            ],
        }

    def to_llm_context(self) -> str:
        """Return Markdown-formatted study summary for Claude."""
        lines = [
            f"## Study: {self.study_name}",
            f"- **Data source**: `{self.data_source}`",
            f"- **Schema**: {self.schema_name}",
            f"- **Data format**: {self.data_format}",
            "",
        ]

        tables = self.tables_used if self.tables_used else self._infer_tables()
        if tables:
            lines.append(f"### Tables used: {', '.join(tables)}")
            lines.append("")

        if self.cohort_definition:
            lines.append("### Cohort definition")
            for crit in self.cohort_definition.get("inclusion", []):
                lines.append(
                    f"- Include: {crit.get('column')} {crit.get('op')} {crit.get('value')}"
                )
            for crit in self.cohort_definition.get("exclusion", []):
                lines.append(
                    f"- Exclude: {crit.get('column')} {crit.get('op')} {crit.get('value')}"
                )
            lines.append("")

        if self.steps:
            lines.append("### Analysis steps performed")
            for i, s in enumerate(self.steps, 1):
                lines.append(f"{i}. **{s.step_name}** ({s.analysis_type or 'unspecified'})")
                lines.append(f"   {s.description}")
                if s.parameters:
                    for k, v in s.parameters.items():
                        lines.append(f"   - {k}: {v}")
                if s.result_summary:
                    lines.append(
                        f"   - Result keys: {', '.join(s.result_summary.keys())}"
                    )
            lines.append("")

        return "\n".join(lines)

    def _infer_tables(self) -> list[str]:
        """Infer table list from schema name."""
        if self.schema_name == "clif":
            return [
                "patient", "hospitalization", "adt", "vitals", "labs",
                "respiratory_support", "medication_admin_continuous",
                "patient_assessments",
            ]
        return ["hospitalization"]
