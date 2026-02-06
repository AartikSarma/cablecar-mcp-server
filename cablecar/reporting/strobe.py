"""STROBE-compliant observational study reporting (22 items)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class STROBEItem:
    number: int
    section: str
    item: str
    content: str = ""
    status: str = "not_started"  # not_started, draft, complete

STROBE_CHECKLIST = [
    STROBEItem(1, "Title and Abstract", "Indicate the study's design with a commonly used term; provide an informative summary"),
    STROBEItem(2, "Introduction - Background", "Explain the scientific background and rationale"),
    STROBEItem(3, "Introduction - Objectives", "State specific objectives, including any prespecified hypotheses"),
    STROBEItem(4, "Methods - Study design", "Present key elements of study design early in the paper"),
    STROBEItem(5, "Methods - Setting", "Describe the setting, locations, and relevant dates"),
    STROBEItem(6, "Methods - Participants", "Give eligibility criteria, sources, methods of selection and follow-up"),
    STROBEItem(7, "Methods - Variables", "Define all outcomes, exposures, predictors, confounders, and effect modifiers"),
    STROBEItem(8, "Methods - Data sources", "Give sources of data and details of methods of assessment"),
    STROBEItem(9, "Methods - Bias", "Describe any efforts to address potential sources of bias"),
    STROBEItem(10, "Methods - Study size", "Explain how the study size was arrived at"),
    STROBEItem(11, "Methods - Quantitative variables", "Explain how quantitative variables were handled in the analyses"),
    STROBEItem(12, "Methods - Statistical methods", "Describe all statistical methods"),
    STROBEItem(13, "Results - Participants", "Report numbers at each stage of study, use flow diagram"),
    STROBEItem(14, "Results - Descriptive", "Give characteristics of study participants"),
    STROBEItem(15, "Results - Outcome data", "Report numbers of outcome events or summary measures"),
    STROBEItem(16, "Results - Main results", "Give unadjusted and adjusted estimates with CIs and p-values"),
    STROBEItem(17, "Results - Other analyses", "Report other analyses done (subgroup, sensitivity)"),
    STROBEItem(18, "Discussion - Key results", "Summarise key results with reference to study objectives"),
    STROBEItem(19, "Discussion - Limitations", "Discuss limitations, including sources of potential bias"),
    STROBEItem(20, "Discussion - Interpretation", "Give a cautious overall interpretation of results"),
    STROBEItem(21, "Discussion - Generalisability", "Discuss the generalisability of the study results"),
    STROBEItem(22, "Other - Funding", "Give the source of funding and the role of the funders"),
]

class STROBEReport:
    """Generate STROBE-compliant observational study reports."""

    def __init__(self):
        self._items = {item.number: STROBEItem(
            number=item.number, section=item.section, item=item.item
        ) for item in STROBE_CHECKLIST}

    def set_item(self, number: int, content: str, status: str = "draft"):
        if number in self._items:
            self._items[number].content = content
            self._items[number].status = status

    def auto_populate(self, provenance: Any, cohort_summary: dict | None = None,
                      analysis_results: dict | None = None) -> None:
        """Auto-populate items from provenance chain and results."""
        from cablecar.codegen.provenance import AnalysisProvenance
        if isinstance(provenance, AnalysisProvenance):
            self.set_item(4, f"Retrospective cohort study using {provenance.schema_name} data")
            self.set_item(8, f"Data source: {provenance.data_source}")

            if provenance.cohort_definition:
                inclusion = provenance.cohort_definition.get("inclusion", [])
                exclusion = provenance.cohort_definition.get("exclusion", [])
                criteria_text = "Inclusion: " + "; ".join(
                    f"{c.get('column')} {c.get('op')} {c.get('value')}" for c in inclusion
                )
                if exclusion:
                    criteria_text += "\nExclusion: " + "; ".join(
                        f"{c.get('column')} {c.get('op')} {c.get('value')}" for c in exclusion
                    )
                self.set_item(6, criteria_text)

            methods = [s.description for s in provenance.steps if "analysis" in s.step_name.lower() or "regression" in s.step_name.lower()]
            if methods:
                self.set_item(12, "; ".join(methods))

        if cohort_summary:
            n = cohort_summary.get("n", "N/A")
            self.set_item(13, f"Study included {n} participants. See CONSORT flow diagram.")

        if analysis_results:
            self.set_item(16, str(analysis_results))

    def get_completion(self) -> dict:
        total = len(self._items)
        completed = sum(1 for item in self._items.values() if item.status == "complete")
        drafted = sum(1 for item in self._items.values() if item.status == "draft")
        return {
            "total": total,
            "completed": completed,
            "drafted": drafted,
            "not_started": total - completed - drafted,
            "pct_complete": round((completed + drafted * 0.5) / total * 100, 1),
        }

    def to_markdown(self) -> str:
        lines = ["# STROBE Checklist", ""]
        current_section = ""
        for num in sorted(self._items.keys()):
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
        lines.append(f"\n---\n**Completion: {completion['pct_complete']}%** ({completion['completed']} complete, {completion['drafted']} drafted, {completion['not_started']} remaining)")
        return "\n".join(lines)
