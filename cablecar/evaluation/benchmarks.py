"""Hypothesis discovery benchmark runner.

Orchestrates scoring of discovery results against DGP specs and aggregates
results across scenarios, difficulty tiers, and context levels.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cablecar.evaluation.dgp import ContextLevel, DGPSpec, DifficultyTier
from cablecar.evaluation.discovery_result import DiscoveryResult
from cablecar.evaluation.scoring import DimensionScore, DiscoveryScorer, ScoredResult


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


class BenchmarkScore(BaseModel):
    """Score for a single benchmark scenario."""

    scenario_name: str
    difficulty: DifficultyTier
    context_level: ContextLevel
    dimension_scores: list[DimensionScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class DiscoveryBenchmark:
    """Run hypothesis discovery benchmarks.

    Wraps :class:`DiscoveryScorer` and adds aggregation by difficulty,
    context level, and scoring dimension.
    """

    def __init__(self) -> None:
        self._scorer = DiscoveryScorer()

    def run_scenario(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        context_level: ContextLevel | None = None,
    ) -> BenchmarkScore:
        """Score a single scenario.

        Parameters
        ----------
        spec:
            DGP specification containing ground truth.
        result:
            Discovery agent's output.
        context_level:
            Override context level (defaults to ``spec.context_level``).
        """
        ctx = context_level if context_level is not None else spec.context_level
        scored = self._scorer.score(spec, result)

        return BenchmarkScore(
            scenario_name=spec.name,
            difficulty=spec.difficulty,
            context_level=ctx,
            dimension_scores=scored.dimension_scores,
            overall_score=scored.overall_score,
            feedback=scored.feedback,
            metadata=scored.metadata,
        )

    def run_suite(
        self,
        specs: list[DGPSpec],
        results: list[DiscoveryResult],
        context_level: ContextLevel | None = None,
    ) -> list[BenchmarkScore]:
        """Score multiple scenarios.

        ``specs`` and ``results`` must be parallel lists.
        """
        if len(specs) != len(results):
            raise ValueError(
                f"specs ({len(specs)}) and results ({len(results)}) "
                "must have the same length."
            )
        return [
            self.run_scenario(spec, result, context_level)
            for spec, result in zip(specs, results)
        ]

    @staticmethod
    def summary(scores: list[BenchmarkScore]) -> dict[str, Any]:
        """Aggregate benchmark scores.

        Returns a dict with:
        - ``overall``: mean overall score
        - ``by_difficulty``: mean score per difficulty tier
        - ``by_context_level``: mean score per context level
        - ``by_dimension``: mean score per scoring dimension
        - ``scenarios``: per-scenario breakdown
        """
        if not scores:
            return {"overall": 0.0, "by_difficulty": {}, "by_context_level": {}, "by_dimension": {}, "scenarios": []}

        overall = sum(s.overall_score for s in scores) / len(scores)

        # By difficulty
        by_diff: dict[str, list[float]] = {}
        for s in scores:
            by_diff.setdefault(s.difficulty.value, []).append(s.overall_score)
        by_difficulty = {k: sum(v) / len(v) for k, v in by_diff.items()}

        # By context level
        by_ctx: dict[str, list[float]] = {}
        for s in scores:
            by_ctx.setdefault(s.context_level.value, []).append(s.overall_score)
        by_context_level = {k: sum(v) / len(v) for k, v in by_ctx.items()}

        # By dimension
        by_dim: dict[str, list[float]] = {}
        for s in scores:
            for dim in s.dimension_scores:
                by_dim.setdefault(dim.name, []).append(dim.score)
        by_dimension = {k: sum(v) / len(v) for k, v in by_dim.items()}

        scenarios = [
            {
                "name": s.scenario_name,
                "difficulty": s.difficulty.value,
                "context_level": s.context_level.value,
                "overall": round(s.overall_score, 4),
            }
            for s in scores
        ]

        return {
            "overall": round(overall, 4),
            "by_difficulty": {k: round(v, 4) for k, v in by_difficulty.items()},
            "by_context_level": {k: round(v, 4) for k, v in by_context_level.items()},
            "by_dimension": {k: round(v, 4) for k, v in by_dimension.items()},
            "scenarios": scenarios,
        }
