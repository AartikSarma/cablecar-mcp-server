"""Grade AI analysis output quality against expected results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from cablecar.evaluation.scenarios import EvaluationScenario

@dataclass
class GradeResult:
    """Result of grading an AI analysis output."""
    scenario_name: str
    overall_score: float = 0.0  # 0-100
    dimensions: dict[str, float] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "overall_score": self.overall_score,
            "dimensions": self.dimensions,
            "feedback": self.feedback,
        }

class OutputGrader:
    """Grade analysis outputs against expected results."""

    def grade(self, scenario: EvaluationScenario, output: dict) -> GradeResult:
        """Grade an analysis output against a scenario's expected findings."""
        result = GradeResult(scenario_name=scenario.name)

        scores = {}

        # 1. Method appropriateness (25%)
        scores["method_choice"] = self._grade_methods(scenario, output)

        # 2. Statistical correctness (25%)
        scores["statistical_correctness"] = self._grade_statistics(scenario, output)

        # 3. Causal reasoning (25%)
        scores["causal_reasoning"] = self._grade_causal(scenario, output)

        # 4. Completeness (25%)
        scores["completeness"] = self._grade_completeness(scenario, output)

        result.dimensions = scores
        result.overall_score = sum(scores.values()) / len(scores)
        result.feedback = self._generate_feedback(scenario, output, scores)

        return result

    def _grade_methods(self, scenario: EvaluationScenario, output: dict) -> float:
        """Grade whether appropriate statistical methods were used."""
        score = 0.0
        used_methods = output.get("methods_used", [])
        expected = scenario.expected_methods

        if not expected:
            return 50.0

        matched = sum(1 for m in expected if any(m in u for u in used_methods))
        score = (matched / len(expected)) * 100

        return min(100, score)

    def _grade_statistics(self, scenario: EvaluationScenario, output: dict) -> float:
        """Grade statistical correctness of results."""
        score = 50.0  # Base score
        findings = scenario.expected_findings
        results = output.get("results", {})

        # Check direction of effect
        expected_dir = findings.get("direction")
        actual_effect = results.get("effect_estimate", results.get("coefficient", 0))

        if expected_dir == "positive" and actual_effect > 0:
            score += 25
        elif expected_dir == "negative" and actual_effect < 0:
            score += 25
        elif expected_dir == "positive" and actual_effect <= 0:
            score -= 25

        # Check significance
        expected_sig = findings.get("significant")
        actual_pval = results.get("p_value", 1.0)

        if expected_sig is True and actual_pval < 0.05:
            score += 25
        elif expected_sig is False and actual_pval >= 0.05:
            score += 25

        return max(0, min(100, score))

    def _grade_causal(self, scenario: EvaluationScenario, output: dict) -> float:
        """Grade causal reasoning quality."""
        if scenario.domain != "causal":
            return 50.0  # Neutral for non-causal scenarios

        score = 0.0

        # Did they identify confounders?
        if output.get("confounders_identified"):
            score += 40
            expected_confounders = scenario.expected_findings.get("confounders", [])
            identified = output.get("confounders_identified", [])
            if any(c in identified for c in expected_confounders):
                score += 30

        # Did they build a DAG?
        if output.get("dag_constructed"):
            score += 15

        # Did they adjust for confounders?
        if output.get("adjusted_analysis"):
            score += 15

        return min(100, score)

    def _grade_completeness(self, scenario: EvaluationScenario, output: dict) -> float:
        """Grade completeness of the analysis."""
        checks = [
            output.get("descriptive_stats_provided", False),
            output.get("hypothesis_tested", False),
            output.get("effect_size_reported", False),
            output.get("confidence_intervals", False),
            output.get("assumptions_checked", False),
            output.get("limitations_noted", False),
        ]

        return (sum(checks) / len(checks)) * 100

    def _generate_feedback(self, scenario: EvaluationScenario, output: dict, scores: dict) -> list[str]:
        """Generate actionable feedback."""
        feedback = []

        if scores.get("method_choice", 0) < 50:
            feedback.append(f"Consider using: {', '.join(scenario.expected_methods)}")

        if scores.get("causal_reasoning", 0) < 50 and scenario.domain == "causal":
            feedback.append("Build a causal DAG to identify confounders before running regression")

        if scores.get("completeness", 0) < 50:
            feedback.append("Include effect sizes, confidence intervals, and assumption checks")

        if scores.get("statistical_correctness", 0) >= 75:
            feedback.append("Statistical analysis is well-executed")

        return feedback
