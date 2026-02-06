"""Clinical research capability benchmarks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from cablecar.evaluation.scenarios import SCENARIOS, EvaluationScenario
from cablecar.evaluation.graders import OutputGrader, GradeResult

@dataclass
class BenchmarkResult:
    """Results from running a full benchmark suite."""
    name: str
    n_scenarios: int = 0
    grades: list[GradeResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.grades:
            return 0.0
        return sum(g.overall_score for g in self.grades) / len(self.grades)

    @property
    def by_domain(self) -> dict[str, float]:
        domain_scores: dict[str, list[float]] = {}
        for grade, scenario in zip(self.grades, SCENARIOS[:len(self.grades)]):
            domain_scores.setdefault(scenario.domain, []).append(grade.overall_score)
        return {domain: sum(scores)/len(scores) for domain, scores in domain_scores.items()}

    @property
    def by_difficulty(self) -> dict[str, float]:
        diff_scores: dict[str, list[float]] = {}
        for grade, scenario in zip(self.grades, SCENARIOS[:len(self.grades)]):
            diff_scores.setdefault(scenario.difficulty, []).append(grade.overall_score)
        return {diff: sum(scores)/len(scores) for diff, scores in diff_scores.items()}

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_scenarios": self.n_scenarios,
            "overall_score": round(self.overall_score, 1),
            "by_domain": {k: round(v, 1) for k, v in self.by_domain.items()},
            "by_difficulty": {k: round(v, 1) for k, v in self.by_difficulty.items()},
            "scenario_scores": [
                {"scenario": g.scenario_name, "score": round(g.overall_score, 1)}
                for g in self.grades
            ],
        }

class ClinicalBenchmark:
    """Run benchmark evaluation suites."""

    def __init__(self):
        self._grader = OutputGrader()
        self._scenarios = list(SCENARIOS)

    def run(self, outputs: dict[str, dict], name: str = "benchmark") -> BenchmarkResult:
        """Run benchmark against provided outputs.

        Args:
            outputs: Dict mapping scenario name to analysis output dict
            name: Name for this benchmark run
        """
        result = BenchmarkResult(name=name, n_scenarios=len(self._scenarios))

        for scenario in self._scenarios:
            output = outputs.get(scenario.name, {})
            grade = self._grader.grade(scenario, output)
            result.grades.append(grade)

        return result

    def list_scenarios(self) -> list[dict]:
        return [s.to_dict() for s in self._scenarios]

    def get_scenario(self, name: str) -> EvaluationScenario | None:
        for s in self._scenarios:
            if s.name == name:
                return s
        return None
