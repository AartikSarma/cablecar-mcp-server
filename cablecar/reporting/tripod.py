"""TRIPOD+AI-compliant prediction model reporting."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TRIPODItem:
    number: str
    section: str
    item: str
    content: str = ""
    status: str = "not_started"

TRIPOD_CHECKLIST = [
    TRIPODItem("1", "Title", "Identify the study as developing/validating a prediction model"),
    TRIPODItem("2", "Abstract", "Provide a structured summary"),
    TRIPODItem("3a", "Introduction", "Explain the medical context and rationale"),
    TRIPODItem("3b", "Introduction", "Specify the objectives"),
    TRIPODItem("4a", "Methods - Source of data", "Describe the study design and data source"),
    TRIPODItem("4b", "Methods - Source of data", "Specify dates of study period"),
    TRIPODItem("5a", "Methods - Participants", "Specify key elements of the study setting"),
    TRIPODItem("5b", "Methods - Participants", "Describe eligibility criteria"),
    TRIPODItem("6a", "Methods - Outcome", "Define the outcome and timing"),
    TRIPODItem("7a", "Methods - Predictors", "Define all candidate predictors"),
    TRIPODItem("8", "Methods - Sample size", "Explain how the study size was arrived at"),
    TRIPODItem("9", "Methods - Missing data", "Describe how missing data were handled"),
    TRIPODItem("10a", "Methods - Analysis", "Describe how predictors were handled"),
    TRIPODItem("10b", "Methods - Analysis", "Specify type of model and method for predictor selection"),
    TRIPODItem("10d", "Methods - Analysis", "Specify all measures used to assess model performance"),
    TRIPODItem("11", "Methods - Analysis", "Describe how risk groups were created"),
    TRIPODItem("13a", "Results - Participants", "Describe flow of participants through the study"),
    TRIPODItem("13b", "Results - Participants", "Describe participant characteristics"),
    TRIPODItem("14a", "Results - Model", "Report number of participants and outcome events"),
    TRIPODItem("15a", "Results - Performance", "Report discrimination and calibration"),
    TRIPODItem("16", "Results - Performance", "Report results of any model updating"),
    TRIPODItem("18", "Discussion", "Discuss limitations"),
    TRIPODItem("19a", "Discussion", "Give an overall interpretation of results"),
    TRIPODItem("20", "Other", "Provide supplementary information on availability of data and code"),
    # AI-specific items
    TRIPODItem("AI-1", "AI Methods", "Describe AI model architecture and hyperparameters"),
    TRIPODItem("AI-2", "AI Methods", "Describe data preprocessing and feature engineering"),
    TRIPODItem("AI-3", "AI Results", "Report model explainability (feature importance, SHAP)"),
    TRIPODItem("AI-4", "AI Ethics", "Describe fairness assessment across demographic groups"),
]

class TRIPODReport:
    """Generate TRIPOD+AI-compliant prediction model reports."""

    def __init__(self):
        self._items = {item.number: TRIPODItem(
            number=item.number, section=item.section, item=item.item
        ) for item in TRIPOD_CHECKLIST}

    def set_item(self, number: str, content: str, status: str = "draft"):
        if number in self._items:
            self._items[number].content = content
            self._items[number].status = status

    def auto_populate(self, provenance: Any, model_results: dict | None = None) -> None:
        from cablecar.codegen.provenance import AnalysisProvenance
        if isinstance(provenance, AnalysisProvenance):
            self.set_item("4a", f"Retrospective study using {provenance.schema_name} data from {provenance.data_source}")

        if model_results:
            perf = model_results.get("performance", {})
            if perf:
                self.set_item("15a", f"AUROC: {perf.get('auroc', 'N/A')}, AUPRC: {perf.get('auprc', 'N/A')}")
            features = model_results.get("feature_importance", {})
            if features:
                top = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                self.set_item("AI-3", "Top features: " + ", ".join(f"{k} ({v:.3f})" for k, v in top))

    def get_completion(self) -> dict:
        total = len(self._items)
        completed = sum(1 for item in self._items.values() if item.status == "complete")
        drafted = sum(1 for item in self._items.values() if item.status == "draft")
        return {
            "total": total, "completed": completed, "drafted": drafted,
            "not_started": total - completed - drafted,
            "pct_complete": round((completed + drafted * 0.5) / total * 100, 1),
        }

    def to_markdown(self) -> str:
        lines = ["# TRIPOD+AI Checklist", ""]
        current_section = ""
        for num in sorted(self._items.keys(), key=lambda x: (not x.startswith("AI"), x)):
            item = self._items[num]
            if item.section != current_section:
                current_section = item.section
                lines.append(f"\n## {current_section}")
            status_icon = {"complete": "✓", "draft": "~", "not_started": "○"}.get(item.status, "?")
            lines.append(f"\n### Item {num} [{status_icon}]")
            lines.append(f"**{item.item}**")
            if item.content:
                lines.append(f"\n{item.content}")
            else:
                lines.append("\n*Not yet completed*")

        completion = self.get_completion()
        lines.append(f"\n---\n**Completion: {completion['pct_complete']}%**")
        return "\n".join(lines)
