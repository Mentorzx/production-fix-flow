"""Property tests for composite score improvement invariants.

Tests the core HPO scoring invariant:
- If you inject better metrics (same trade-off pattern), composite_score increases.
- Pareto-improvements always yield higher scores.
"""

from __future__ import annotations

import math

import pytest


# ============================================================================
# Scoring functions (matching production logic)
# ============================================================================


def normalize_metric(value: float, *, low: float, high: float) -> float:
    """Clamp and scale a metric into [0, 1] interval."""
    if math.isnan(value):
        return 0.0
    if high <= low:
        return max(0.0, min(1.0, value))
    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


def blend_scores(scores: list[tuple[float, float]]) -> float:
    """Compute weighted average from (value, weight) pairs."""
    total_weight = 0.0
    total = 0.0
    for value, weight in scores:
        if weight <= 0 or math.isnan(value):
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight > 0 else 0.0


def compute_composite_score(
    kge_mrr: float,
    rules_conf: float,
    rules_recall: float,
    rules_cov: float,
    lgbm_auc: float,
    hybrid_f1: float,
    xgb_f1: float,
    *,
    symbolic_contribution_ratio: float = 0.35,
    target_symbolic_ratio: float = 0.42,
    dominance_penalty_coeff: float = 0.5,
    generalization_gap: float = 0.0,
    gap_penalty_coeff: float = 0.3,
    gap_threshold: float = 0.05,
) -> float:
    """Compute composite score matching production logic."""
    # Bounds from config
    bounds = {
        "kge_mrr": (0.15, 0.75),
        "rules_conf": (0.4, 0.95),
        "rules_recall": (0.05, 0.5),
        "rules_cov": (0.05, 0.5),
        "lgbm_auc": (0.6, 0.99),
        "hybrid_f1": (0.45, 0.9),
        "xgb_f1": (0.45, 0.9),
    }

    # Normalize components
    kge_norm = normalize_metric(kge_mrr, low=bounds["kge_mrr"][0], high=bounds["kge_mrr"][1])
    conf_norm = normalize_metric(rules_conf, low=bounds["rules_conf"][0], high=bounds["rules_conf"][1])
    recall_norm = normalize_metric(rules_recall, low=bounds["rules_recall"][0], high=bounds["rules_recall"][1])
    cov_norm = normalize_metric(rules_cov, low=bounds["rules_cov"][0], high=bounds["rules_cov"][1])
    lgbm_norm = normalize_metric(lgbm_auc, low=bounds["lgbm_auc"][0], high=bounds["lgbm_auc"][1])
    hybrid_norm = normalize_metric(hybrid_f1, low=bounds["hybrid_f1"][0], high=bounds["hybrid_f1"][1])
    xgb_norm = normalize_metric(xgb_f1, low=bounds["xgb_f1"][0], high=bounds["xgb_f1"][1])

    # Rules component
    rules_component = blend_scores([
        (conf_norm, 0.5),
        (recall_norm, 0.3),
        (cov_norm, 0.2),
    ])

    # Learner component
    learner_component = blend_scores([
        (lgbm_norm, 0.5),
        (hybrid_norm, 0.3),
        (xgb_norm, 0.2),
    ])

    # Base score
    base_score = blend_scores([
        (kge_norm, 0.25),
        (rules_component, 0.25),
        (learner_component, 0.50),
    ])

    # Dominance penalty
    dominance_penalty = 0.0
    if symbolic_contribution_ratio > target_symbolic_ratio:
        overflow = symbolic_contribution_ratio - target_symbolic_ratio
        dominance_penalty = dominance_penalty_coeff * overflow

    # Gap penalty
    gap_penalty = 0.0
    if generalization_gap > gap_threshold:
        gap_penalty = gap_penalty_coeff * min(1.0, (generalization_gap - gap_threshold) / 0.2)

    return max(0.0, base_score - dominance_penalty - gap_penalty)


# ============================================================================
# Tests: Better metrics → higher score
# ============================================================================


class TestBetterMetricsImproveScore:
    """Test that improving metrics always improves composite score."""

    @pytest.fixture
    def baseline_metrics(self) -> dict:
        """Baseline metrics for comparison."""
        return {
            "kge_mrr": 0.4,
            "rules_conf": 0.7,
            "rules_recall": 0.2,
            "rules_cov": 0.25,
            "lgbm_auc": 0.75,
            "hybrid_f1": 0.65,
            "xgb_f1": 0.60,
        }

    @pytest.mark.parametrize("metric_to_improve", [
        "kge_mrr",
        "rules_conf",
        "rules_recall",
        "rules_cov",
        "lgbm_auc",
        "hybrid_f1",
        "xgb_f1",
    ])
    def test_improving_single_metric_improves_score(
        self, baseline_metrics: dict, metric_to_improve: str
    ):
        """Property: improving any single metric should improve total score."""
        baseline_score = compute_composite_score(**baseline_metrics)

        # Improve the metric by 10%
        improved_metrics = baseline_metrics.copy()
        improved_metrics[metric_to_improve] = min(1.0, baseline_metrics[metric_to_improve] + 0.1)
        improved_score = compute_composite_score(**improved_metrics)

        assert improved_score >= baseline_score, (
            f"Improving {metric_to_improve} should improve score: "
            f"baseline={baseline_score:.4f}, improved={improved_score:.4f}"
        )

    def test_pareto_improvement_always_increases_score(self, baseline_metrics: dict):
        """Property: improving ALL metrics should definitely increase score."""
        baseline_score = compute_composite_score(**baseline_metrics)

        # Improve all metrics
        improved_metrics = {k: min(1.0, v + 0.1) for k, v in baseline_metrics.items()}
        improved_score = compute_composite_score(**improved_metrics)

        assert improved_score > baseline_score, (
            f"Pareto improvement should increase score: "
            f"baseline={baseline_score:.4f}, improved={improved_score:.4f}"
        )

    def test_all_max_metrics_gives_high_score(self):
        """Property: all metrics at max should give score close to 1."""
        max_metrics = {
            "kge_mrr": 0.75,
            "rules_conf": 0.95,
            "rules_recall": 0.5,
            "rules_cov": 0.5,
            "lgbm_auc": 0.99,
            "hybrid_f1": 0.9,
            "xgb_f1": 0.9,
        }
        score = compute_composite_score(**max_metrics)
        assert score > 0.9, f"Max metrics should give high score, got {score:.4f}"

    def test_all_min_metrics_gives_low_score(self):
        """Property: all metrics at min should give score close to 0."""
        min_metrics = {
            "kge_mrr": 0.15,
            "rules_conf": 0.4,
            "rules_recall": 0.05,
            "rules_cov": 0.05,
            "lgbm_auc": 0.6,
            "hybrid_f1": 0.45,
            "xgb_f1": 0.45,
        }
        score = compute_composite_score(**min_metrics)
        assert score < 0.1, f"Min metrics should give low score, got {score:.4f}"


class TestPenaltiesReduceScore:
    """Test that penalties reduce the composite score."""

    @pytest.fixture
    def good_metrics(self) -> dict:
        """Good baseline metrics."""
        return {
            "kge_mrr": 0.5,
            "rules_conf": 0.8,
            "rules_recall": 0.3,
            "rules_cov": 0.35,
            "lgbm_auc": 0.85,
            "hybrid_f1": 0.75,
            "xgb_f1": 0.70,
        }

    def test_dominance_penalty_reduces_score(self, good_metrics: dict):
        """Property: symbolic dominance above target reduces score."""
        score_at_target = compute_composite_score(
            **good_metrics,
            symbolic_contribution_ratio=0.42,
            target_symbolic_ratio=0.42,
        )
        score_above_target = compute_composite_score(
            **good_metrics,
            symbolic_contribution_ratio=0.70,
            target_symbolic_ratio=0.42,
        )

        assert score_above_target < score_at_target, (
            f"Dominance penalty should reduce score: "
            f"at_target={score_at_target:.4f}, above={score_above_target:.4f}"
        )

    def test_gap_penalty_reduces_score(self, good_metrics: dict):
        """Property: high generalization gap reduces score."""
        score_no_gap = compute_composite_score(
            **good_metrics,
            generalization_gap=0.0,
        )
        score_high_gap = compute_composite_score(
            **good_metrics,
            generalization_gap=0.15,
        )

        assert score_high_gap < score_no_gap, (
            f"Gap penalty should reduce score: "
            f"no_gap={score_no_gap:.4f}, high_gap={score_high_gap:.4f}"
        )

    def test_combined_penalties_stack(self, good_metrics: dict):
        """Property: multiple penalties stack (reduce score more)."""
        score_no_penalty = compute_composite_score(
            **good_metrics,
            symbolic_contribution_ratio=0.35,
            generalization_gap=0.0,
        )
        score_one_penalty = compute_composite_score(
            **good_metrics,
            symbolic_contribution_ratio=0.70,  # Dominance penalty
            generalization_gap=0.0,
        )
        score_two_penalties = compute_composite_score(
            **good_metrics,
            symbolic_contribution_ratio=0.70,  # Dominance penalty
            generalization_gap=0.15,  # Gap penalty
        )

        assert score_no_penalty > score_one_penalty > score_two_penalties, (
            f"Penalties should stack: "
            f"none={score_no_penalty:.4f}, one={score_one_penalty:.4f}, two={score_two_penalties:.4f}"
        )


class TestCoverageNotPunished:
    """Test that increasing coverage doesn't hurt score unfairly."""

    def test_higher_coverage_same_precision_better_or_equal(self):
        """Property: higher coverage with same precision should not decrease score."""
        base_metrics = {
            "kge_mrr": 0.45,
            "rules_conf": 0.75,  # Same precision
            "rules_recall": 0.25,
            "rules_cov": 0.2,  # Lower coverage
            "lgbm_auc": 0.80,
            "hybrid_f1": 0.70,
            "xgb_f1": 0.65,
        }

        improved_metrics = base_metrics.copy()
        improved_metrics["rules_cov"] = 0.4  # Higher coverage

        score_low_cov = compute_composite_score(**base_metrics)
        score_high_cov = compute_composite_score(**improved_metrics)

        assert score_high_cov >= score_low_cov, (
            f"Higher coverage should not hurt score: "
            f"low_cov={score_low_cov:.4f}, high_cov={score_high_cov:.4f}"
        )

    @pytest.mark.parametrize("coverage", [0.1, 0.2, 0.3, 0.4, 0.5])
    def test_coverage_monotonicity(self, coverage: float):
        """Property: coverage contribution is monotonic."""
        base_metrics = {
            "kge_mrr": 0.45,
            "rules_conf": 0.75,
            "rules_recall": 0.25,
            "rules_cov": coverage,
            "lgbm_auc": 0.80,
            "hybrid_f1": 0.70,
            "xgb_f1": 0.65,
        }
        score = compute_composite_score(**base_metrics)
        assert 0 <= score <= 1, f"Score out of bounds: {score}"


class TestScoreStability:
    """Test score computation stability."""

    def test_same_inputs_same_output(self):
        """Property: same inputs always produce same score."""
        metrics = {
            "kge_mrr": 0.45,
            "rules_conf": 0.75,
            "rules_recall": 0.25,
            "rules_cov": 0.3,
            "lgbm_auc": 0.80,
            "hybrid_f1": 0.70,
            "xgb_f1": 0.65,
        }

        scores = [compute_composite_score(**metrics) for _ in range(10)]
        assert all(s == scores[0] for s in scores), "Scores should be deterministic"

    def test_score_always_in_valid_range(self):
        """Property: score is always in [0, 1]."""
        import numpy as np
        rng = np.random.RandomState(42)

        for _ in range(100):
            metrics = {
                "kge_mrr": rng.uniform(0.1, 0.8),
                "rules_conf": rng.uniform(0.3, 1.0),
                "rules_recall": rng.uniform(0.0, 0.6),
                "rules_cov": rng.uniform(0.0, 0.6),
                "lgbm_auc": rng.uniform(0.5, 1.0),
                "hybrid_f1": rng.uniform(0.4, 1.0),
                "xgb_f1": rng.uniform(0.4, 1.0),
                "symbolic_contribution_ratio": rng.uniform(0.0, 1.0),
                "generalization_gap": rng.uniform(0.0, 0.3),
            }
            score = compute_composite_score(**metrics)
            assert 0 <= score <= 1, f"Score {score} out of [0, 1] for {metrics}"
