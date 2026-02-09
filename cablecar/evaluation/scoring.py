"""Eight-dimension scoring engine for hypothesis discovery benchmarking.

Compares a :class:`DiscoveryResult` against the :class:`GroundTruth` embedded
in a :class:`DGPSpec` to produce a :class:`ScoredResult` with per-dimension
scores and an overall weighted score.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cablecar.evaluation.dgp import DGPSpec, VariableRole
from cablecar.evaluation.discovery_result import DiscoveryResult


# ---------------------------------------------------------------------------
# Score containers
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    name: str
    weight: float
    score: float = Field(ge=0.0, le=1.0)
    detail: str = ""


class ScoredResult(BaseModel):
    """Complete scoring result for one scenario."""

    scenario_name: str
    dimension_scores: list[DimensionScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


# Dimension weights (must sum to 1.0)
DIMENSION_WEIGHTS: dict[str, float] = {
    "variable_identification": 0.15,
    "hypothesis_quality": 0.10,
    "dag_accuracy": 0.20,
    "method_selection": 0.10,
    "effect_estimation": 0.15,
    "confounder_handling": 0.15,
    "missingness_handling": 0.05,
    "interpretation_quality": 0.10,
}


class DiscoveryScorer:
    """Score a discovery result against a DGP spec's ground truth.

    Eight scoring dimensions, each producing a 0--1 score.  The overall
    score is the weighted average.
    """

    def score(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
    ) -> ScoredResult:
        """Score *result* against the ground truth in *spec*."""
        gt = spec.ground_truth
        reverse_map = spec.get_reverse_mapping()
        feedback: list[str] = []

        dimensions: list[DimensionScore] = []
        for dim_name, weight in DIMENSION_WEIGHTS.items():
            method = getattr(self, f"_score_{dim_name}")
            score_val, detail = method(spec, result, reverse_map)
            dimensions.append(
                DimensionScore(
                    name=dim_name,
                    weight=weight,
                    score=score_val,
                    detail=detail,
                )
            )
            if score_val < 0.5:
                feedback.append(f"Low score on {dim_name}: {detail}")

        overall = sum(d.weight * d.score for d in dimensions)

        return ScoredResult(
            scenario_name=spec.name,
            dimension_scores=dimensions,
            overall_score=round(overall, 4),
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    # Helper: resolve agent variable name to DGP name
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(name: str, reverse_map: dict[str, str]) -> str:
        """Try to translate a data-level name to a DGP semantic name."""
        return reverse_map.get(name, name)

    def _resolve_set(
        self, names: list[str], reverse_map: dict[str, str]
    ) -> set[str]:
        return {self._resolve(n, reverse_map) for n in names}

    # ------------------------------------------------------------------
    # 1. Variable identification (0.15)
    # ------------------------------------------------------------------

    def _score_variable_identification(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        gt = spec.ground_truth
        score = 0.0
        parts: list[str] = []

        # Exposure match (40%)
        resolved_exposure = self._resolve(result.identified_exposure, reverse_map)
        if resolved_exposure == gt.primary_exposure:
            score += 0.4
            parts.append("exposure correct")
        else:
            parts.append(
                f"exposure wrong (got '{resolved_exposure}', "
                f"expected '{gt.primary_exposure}')"
            )

        # Outcome match (40%)
        resolved_outcome = self._resolve(result.identified_outcome, reverse_map)
        if resolved_outcome == gt.primary_outcome:
            score += 0.4
            parts.append("outcome correct")
        else:
            parts.append(
                f"outcome wrong (got '{resolved_outcome}', "
                f"expected '{gt.primary_outcome}')"
            )

        # Confounder identification (20%)
        true_confounders = {
            v.name for v in spec.variables if v.role == VariableRole.CONFOUNDER
        }
        if true_confounders:
            identified = self._resolve_set(
                result.identified_confounders, reverse_map
            )
            overlap = len(identified & true_confounders)
            confounder_recall = overlap / len(true_confounders)
            score += 0.2 * confounder_recall
            parts.append(
                f"confounders {overlap}/{len(true_confounders)} found"
            )
        else:
            score += 0.2
            parts.append("no confounders to find")

        return round(min(score, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 2. Hypothesis quality (0.10)
    # ------------------------------------------------------------------

    def _score_hypothesis_quality(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        score = 0.0
        parts: list[str] = []

        # Has a hypothesis at all? (30%)
        if result.primary_hypothesis:
            score += 0.3
            parts.append("hypothesis stated")
        else:
            parts.append("no hypothesis")
            return 0.0, "; ".join(parts)

        # Does the hypothesis mention exposure AND outcome? (40%)
        hyp_lower = result.primary_hypothesis.lower()
        gt = spec.ground_truth
        exposure_mentioned = (
            gt.primary_exposure.lower() in hyp_lower
            or self._resolve(result.identified_exposure, reverse_map).lower()
            in hyp_lower
            or result.identified_exposure.lower() in hyp_lower
        )
        outcome_mentioned = (
            gt.primary_outcome.lower() in hyp_lower
            or self._resolve(result.identified_outcome, reverse_map).lower()
            in hyp_lower
            or result.identified_outcome.lower() in hyp_lower
        )
        if exposure_mentioned and outcome_mentioned:
            score += 0.4
            parts.append("mentions exposure and outcome")
        elif exposure_mentioned or outcome_mentioned:
            score += 0.2
            parts.append("mentions only one of exposure/outcome")

        # Secondary hypotheses (30%)
        if result.secondary_hypotheses:
            score += min(0.3, 0.1 * len(result.secondary_hypotheses))
            parts.append(
                f"{len(result.secondary_hypotheses)} secondary hypotheses"
            )

        return round(min(score, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 3. DAG accuracy (0.20)
    # ------------------------------------------------------------------

    def _score_dag_accuracy(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        gt = spec.ground_truth
        parts: list[str] = []

        if not result.proposed_dag_edges and not gt.expected_dag_edges:
            return 1.0, "no DAG expected or proposed"

        if not result.proposed_dag_edges:
            parts.append("no DAG proposed")
            return 0.0, "; ".join(parts)

        # Translate proposed edges to semantic names
        proposed = {
            (self._resolve(c, reverse_map), self._resolve(e, reverse_map))
            for c, e in result.proposed_dag_edges
        }
        true_edges = set(map(tuple, gt.expected_dag_edges))

        # Edge F1 (40%)
        if true_edges:
            tp = len(proposed & true_edges)
            precision = tp / len(proposed) if proposed else 0
            recall = tp / len(true_edges)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
        else:
            f1 = 1.0 if not proposed else 0.0
        edge_f1_score = 0.4 * f1
        parts.append(f"edge F1={f1:.2f}")

        # Adjustment set Jaccard (40%)
        proposed_adj = self._resolve_set(
            result.proposed_adjustment_set, reverse_map
        )
        true_adj = set(gt.correct_adjustment_set)
        if true_adj or proposed_adj:
            intersection = len(proposed_adj & true_adj)
            union = len(proposed_adj | true_adj)
            jaccard = intersection / union if union > 0 else 0.0
        else:
            jaccard = 1.0
        adj_score = 0.4 * jaccard
        parts.append(f"adj Jaccard={jaccard:.2f}")

        # Collider avoidance (20%)
        true_colliders = {
            v.name for v in spec.variables if v.role == VariableRole.COLLIDER
        }
        conditioned_colliders = proposed_adj & true_colliders
        if true_colliders:
            collider_penalty = len(conditioned_colliders) / len(true_colliders)
            collider_score = 0.2 * (1.0 - collider_penalty)
        else:
            collider_score = 0.2
        parts.append(
            f"collider penalty={len(conditioned_colliders)} conditioned"
        )

        total = edge_f1_score + adj_score + collider_score
        return round(min(total, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 4. Method selection (0.10)
    # ------------------------------------------------------------------

    def _score_method_selection(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        score = 0.0
        parts: list[str] = []

        if not result.methods_used:
            return 0.0, "no methods reported"

        # Reward for having a justification (20%)
        if result.method_justification:
            score += 0.2
            parts.append("justification provided")

        # Reward for using regression/modeling (30%)
        regression_keywords = {
            "regression",
            "logistic",
            "cox",
            "propensity",
            "matching",
            "iv",
            "instrumental",
        }
        if any(
            kw in m.lower()
            for m in result.methods_used
            for kw in regression_keywords
        ):
            score += 0.3
            parts.append("modeling method used")

        # Reward for DAG-based reasoning if there are confounders (30%)
        has_confounders = any(
            v.role == VariableRole.CONFOUNDER for v in spec.variables
        )
        dag_keywords = {"dag", "causal", "backdoor", "adjustment"}
        used_dag = any(
            kw in m.lower()
            for m in result.methods_used
            for kw in dag_keywords
        )
        if has_confounders and used_dag:
            score += 0.3
            parts.append("causal reasoning method used")
        elif not has_confounders:
            score += 0.3
            parts.append("no confounders (DAG not required)")

        # Reward for descriptive analysis (20%)
        descriptive_keywords = {
            "descriptive",
            "summary",
            "exploratory",
            "eda",
            "table_1",
        }
        if any(
            kw in m.lower()
            for m in result.methods_used
            for kw in descriptive_keywords
        ):
            score += 0.2
            parts.append("descriptive analysis used")

        return round(min(score, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 5. Effect estimation (0.15)
    # ------------------------------------------------------------------

    def _score_effect_estimation(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        gt = spec.ground_truth
        parts: list[str] = []

        if result.estimated_effect is None:
            return 0.0, "no effect estimate"

        true_effect = gt.true_causal_effect
        estimated = result.estimated_effect
        tolerance = gt.effect_size_tolerance

        # Wrong direction caps score at 0.1
        if (true_effect > 0 and estimated < 0) or (
            true_effect < 0 and estimated > 0
        ):
            parts.append(
                f"wrong direction (est={estimated:.3f}, true={true_effect:.3f})"
            )
            score = 0.1
            # Bonus if CI contains true value
            if result.confidence_interval is not None:
                lo, hi = result.confidence_interval
                if lo <= true_effect <= hi:
                    score = 0.2
                    parts.append("but CI contains true value")
            return round(score, 4), "; ".join(parts)

        # Relative error scoring
        abs_error = abs(estimated - true_effect)
        if abs_error <= tolerance:
            accuracy_score = 1.0
        elif abs_error <= 3 * tolerance:
            accuracy_score = 1.0 - (abs_error - tolerance) / (2 * tolerance)
        else:
            accuracy_score = 0.0
        parts.append(
            f"est={estimated:.3f}, true={true_effect:.3f}, "
            f"err={abs_error:.3f}, tol={tolerance:.3f}"
        )

        # CI bonus (+0.1 if CI contains true value, capped at 1.0)
        ci_bonus = 0.0
        if result.confidence_interval is not None:
            lo, hi = result.confidence_interval
            if lo <= true_effect <= hi:
                ci_bonus = 0.1
                parts.append("CI contains true value")

        score = min(accuracy_score + ci_bonus, 1.0)
        return round(score, 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 6. Confounder handling (0.15)
    # ------------------------------------------------------------------

    def _score_confounder_handling(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        gt = spec.ground_truth
        parts: list[str] = []

        true_adj = set(gt.correct_adjustment_set)
        if not true_adj:
            parts.append("no adjustment needed")
            return 1.0, "; ".join(parts)

        proposed_adj = self._resolve_set(
            result.proposed_adjustment_set, reverse_map
        )
        identified_conf = self._resolve_set(
            result.identified_confounders, reverse_map
        )

        # Correct variables in adjustment set (50%)
        if true_adj:
            recall = len(proposed_adj & true_adj) / len(true_adj)
        else:
            recall = 1.0
        adj_score = 0.5 * recall
        parts.append(f"adj recall={recall:.2f}")

        # Not including colliders (25%)
        true_colliders = {
            v.name for v in spec.variables if v.role == VariableRole.COLLIDER
        }
        conditioned_colliders = proposed_adj & true_colliders
        if true_colliders:
            avoid_score = 0.25 * (
                1.0 - len(conditioned_colliders) / len(true_colliders)
            )
        else:
            avoid_score = 0.25
        parts.append(f"colliders conditioned={len(conditioned_colliders)}")

        # Identified confounders (25%)
        conf_recall = (
            len(identified_conf & true_adj) / len(true_adj) if true_adj else 1.0
        )
        conf_score = 0.25 * conf_recall
        parts.append(f"confounder recall={conf_recall:.2f}")

        score = adj_score + avoid_score + conf_score
        return round(min(score, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 7. Missingness handling (0.05)
    # ------------------------------------------------------------------

    def _score_missingness_handling(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        has_missingness = any(
            v.missingness.proportion > 0 for v in spec.variables
        )
        parts: list[str] = []

        if not has_missingness:
            parts.append("no missingness in data")
            return 1.0, "; ".join(parts)

        score = 0.0

        # Acknowledged missingness (40%)
        if result.missingness_assessment:
            score += 0.4
            parts.append("missingness assessed")

        # Had a strategy (40%)
        if result.missingness_strategy:
            score += 0.4
            parts.append(f"strategy: {result.missingness_strategy}")

        # Strategy is reasonable (20%)
        reasonable_strategies = {
            "multiple_imputation",
            "inverse_probability",
            "complete_case",
            "sensitivity_analysis",
            "pattern_mixture",
        }
        if result.missingness_strategy and any(
            s in result.missingness_strategy.lower()
            for s in reasonable_strategies
        ):
            score += 0.2
            parts.append("strategy recognized as reasonable")

        return round(min(score, 1.0), 4), "; ".join(parts)

    # ------------------------------------------------------------------
    # 8. Interpretation quality (0.10)
    # ------------------------------------------------------------------

    def _score_interpretation_quality(
        self,
        spec: DGPSpec,
        result: DiscoveryResult,
        reverse_map: dict[str, str],
    ) -> tuple[float, str]:
        score = 0.0
        parts: list[str] = []

        # Has interpretation (30%)
        if result.interpretation:
            score += 0.3
            parts.append("interpretation provided")
        else:
            return 0.0, "no interpretation"

        # Mentions limitations (30%)
        if result.limitations:
            score += 0.3
            parts.append(f"{len(result.limitations)} limitations noted")

        # Interpretation mentions the key variables (20%)
        gt = spec.ground_truth
        interp_lower = result.interpretation.lower()
        mentions_exposure = (
            gt.primary_exposure.lower() in interp_lower
            or result.identified_exposure.lower() in interp_lower
        )
        mentions_outcome = (
            gt.primary_outcome.lower() in interp_lower
            or result.identified_outcome.lower() in interp_lower
        )
        if mentions_exposure and mentions_outcome:
            score += 0.2
            parts.append("mentions key variables")
        elif mentions_exposure or mentions_outcome:
            score += 0.1
            parts.append("mentions one key variable")

        # Has analysis steps trace (20%)
        if result.analysis_steps:
            score += 0.2
            parts.append(f"{len(result.analysis_steps)} analysis steps traced")

        return round(min(score, 1.0), 4), "; ".join(parts)
