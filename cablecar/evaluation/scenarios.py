"""Evaluation scenarios with known ground-truth effects for testing AI analysis capabilities."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class EvaluationScenario:
    """A clinical research scenario with known ground truth for evaluation."""
    name: str
    description: str
    research_question: str
    expected_findings: dict = field(default_factory=dict)
    expected_methods: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = "general"  # general, causal, prediction, survival

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "research_question": self.research_question,
            "expected_findings": self.expected_findings,
            "expected_methods": self.expected_methods,
            "difficulty": self.difficulty,
            "domain": self.domain,
        }

# Pre-defined scenarios based on synthetic data ground truths
SCENARIOS = [
    EvaluationScenario(
        name="age_mortality",
        description="Investigate the association between age and mortality in ICU patients",
        research_question="Is older age associated with higher mortality in ICU patients?",
        expected_findings={
            "direction": "positive",
            "significant": True,
            "method": "logistic_regression",
            "ground_truth": "Mortality probability increases linearly with age (embedded in synthetic data generator)",
            "expected_or_range": (1.01, 1.05),
        },
        expected_methods=["descriptive", "logistic_regression", "kaplan_meier"],
        difficulty="easy",
        domain="general",
    ),
    EvaluationScenario(
        name="severity_vasopressors",
        description="Examine the relationship between illness severity and vasopressor use",
        research_question="Do more severely ill patients receive more vasopressors?",
        expected_findings={
            "direction": "positive",
            "significant": True,
            "ground_truth": "Severe patients have 90% probability of vasopressors vs 30% mild",
        },
        expected_methods=["descriptive", "chi_square", "logistic_regression"],
        difficulty="easy",
        domain="general",
    ),
    EvaluationScenario(
        name="ventilation_los",
        description="Assess the relationship between mechanical ventilation and length of stay",
        research_question="Is mechanical ventilation associated with longer ICU length of stay?",
        expected_findings={
            "direction": "positive",
            "significant": True,
            "confounders": ["age", "severity"],
            "ground_truth": "Severe patients get vent AND have longer LOS (confounded)",
        },
        expected_methods=["descriptive", "mann_whitney", "linear_regression"],
        difficulty="medium",
        domain="causal",
    ),
    EvaluationScenario(
        name="lactate_mortality_prediction",
        description="Build a prediction model for mortality using early lab values",
        research_question="Can early lactate and creatinine levels predict in-hospital mortality?",
        expected_findings={
            "direction": "positive",
            "significant": True,
            "expected_auroc_range": (0.55, 0.75),
            "ground_truth": "Severe patients have elevated lactate/creatinine AND higher mortality",
        },
        expected_methods=["prediction_model", "cross_validation", "feature_importance"],
        difficulty="medium",
        domain="prediction",
    ),
    EvaluationScenario(
        name="confounding_challenge",
        description="Identify and handle confounding in the vasopressor-mortality relationship",
        research_question="Is vasopressor use associated with mortality? What role does confounding play?",
        expected_findings={
            "unadjusted_direction": "positive",
            "adjusted_direction": "attenuated_or_reversed",
            "key_confounder": "severity",
            "ground_truth": "Vasopressors given to sickest patients (confounding by indication). Unadjusted association is misleading.",
        },
        expected_methods=["causal_dag", "unadjusted_regression", "adjusted_regression"],
        difficulty="hard",
        domain="causal",
    ),
    EvaluationScenario(
        name="subgroup_age_effect",
        description="Examine whether the effect of severity on mortality differs by age group",
        research_question="Does the mortality impact of high SOFA scores differ between younger and older ICU patients?",
        expected_findings={
            "interaction_expected": True,
            "ground_truth": "Both age and severity independently affect mortality; interaction may be present",
        },
        expected_methods=["subgroup_analysis", "interaction_test", "forest_plot"],
        difficulty="hard",
        domain="general",
    ),
]

def get_scenario(name: str) -> EvaluationScenario | None:
    for s in SCENARIOS:
        if s.name == name:
            return s
    return None

def list_scenarios() -> list[dict]:
    return [{"name": s.name, "difficulty": s.difficulty, "domain": s.domain, "question": s.research_question} for s in SCENARIOS]
