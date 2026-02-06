"""Study planning with PICO framework and causal reasoning."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class StudyPlan:
    """Research study plan using PICO framework."""
    title: str = ""
    # PICO
    population: str = ""
    intervention_exposure: str = ""
    comparator: str = ""
    outcome: str = ""
    # Study design
    study_type: str = ""  # "cohort", "case-control", "cross-sectional"
    hypothesis: str = ""
    # Causal framework
    dag_variables: list[dict] = field(default_factory=list)
    dag_edges: list[tuple[str, str]] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    # Analysis plan
    primary_analysis: str = ""
    secondary_analyses: list[str] = field(default_factory=list)
    sensitivity_analyses: list[str] = field(default_factory=list)
    subgroup_analyses: list[str] = field(default_factory=list)
    # Sample size
    expected_n: int = 0
    power_analysis: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "pico": {
                "population": self.population,
                "intervention_exposure": self.intervention_exposure,
                "comparator": self.comparator,
                "outcome": self.outcome,
            },
            "study_type": self.study_type,
            "hypothesis": self.hypothesis,
            "causal_framework": {
                "variables": self.dag_variables,
                "edges": self.dag_edges,
                "confounders": self.confounders,
            },
            "analysis_plan": {
                "primary": self.primary_analysis,
                "secondary": self.secondary_analyses,
                "sensitivity": self.sensitivity_analyses,
                "subgroups": self.subgroup_analyses,
            },
        }

    def summary(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.population:
            lines.append(f"**Population**: {self.population}")
        if self.intervention_exposure:
            lines.append(f"**Exposure**: {self.intervention_exposure}")
        if self.comparator:
            lines.append(f"**Comparator**: {self.comparator}")
        if self.outcome:
            lines.append(f"**Outcome**: {self.outcome}")
        if self.hypothesis:
            lines.append(f"\n**Hypothesis**: {self.hypothesis}")
        if self.primary_analysis:
            lines.append(f"**Primary Analysis**: {self.primary_analysis}")
        return "\n".join(lines)
