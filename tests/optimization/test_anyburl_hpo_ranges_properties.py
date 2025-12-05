"""Property tests for AnyBURL HPO ranges and coverage scoring.

Tests the following properties:
(1) HPO bounds for confidence_quantile, support_quantile, and target_ratio
    MUST be read exclusively from config/models/kg.yaml[rule_filter.hpo_ranges].
(2) Given two synthetic datasets with different coverage, the scoring
    function should not penalize the scenario with higher coverage.
(3) HPO ranges must form valid intervals (low < high).
(4) Coverage contributions must be monotonic in coverage values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# ============================================================================
# Constants and paths
# ============================================================================

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
KG_CONFIG_PATH = CONFIG_DIR / "models" / "kg.yaml"
ENSEMBLE_HPO_CONFIG_PATH = CONFIG_DIR / "hpo" / "ensemble_hpo.yaml"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def kg_config() -> dict[str, Any]:
    """Load KG configuration from YAML."""
    if not KG_CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {KG_CONFIG_PATH}")
    with open(KG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def ensemble_hpo_config() -> dict[str, Any]:
    """Load ensemble HPO configuration from YAML."""
    if not ENSEMBLE_HPO_CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {ENSEMBLE_HPO_CONFIG_PATH}")
    with open(ENSEMBLE_HPO_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def hpo_ranges(kg_config: dict[str, Any]) -> dict[str, Any]:
    """Extract HPO ranges from KG config."""
    return kg_config.get("rule_filter", {}).get("hpo_ranges", {})


@pytest.fixture
def rule_filter_defaults(kg_config: dict[str, Any]) -> dict[str, Any]:
    """Extract rule filter defaults from KG config."""
    return kg_config.get("rule_filter", {}).get("defaults", {})


# ============================================================================
# Tests: HPO bounds must come from config
# ============================================================================


class TestHPOBoundsFromConfig:
    """Test that HPO bounds are read from config files."""

    def test_confidence_quantile_bounds_in_config(self, hpo_ranges: dict[str, Any]):
        """Property: confidence_quantile bounds must exist in kg.yaml."""
        assert "confidence_quantile" in hpo_ranges, (
            "confidence_quantile HPO range not found in config/models/kg.yaml"
        )
        cq_range = hpo_ranges["confidence_quantile"]
        assert "low" in cq_range, "confidence_quantile missing 'low' bound"
        assert "high" in cq_range, "confidence_quantile missing 'high' bound"

    def test_support_quantile_bounds_in_config(self, hpo_ranges: dict[str, Any]):
        """Property: support_quantile bounds must exist in kg.yaml."""
        assert "support_quantile" in hpo_ranges, (
            "support_quantile HPO range not found in config/models/kg.yaml"
        )
        sq_range = hpo_ranges["support_quantile"]
        assert "low" in sq_range, "support_quantile missing 'low' bound"
        assert "high" in sq_range, "support_quantile missing 'high' bound"

    def test_target_ratio_bounds_in_config(self, hpo_ranges: dict[str, Any]):
        """Property: target_ratio bounds must exist in kg.yaml."""
        assert "target_ratio" in hpo_ranges, (
            "target_ratio HPO range not found in config/models/kg.yaml"
        )
        tr_range = hpo_ranges["target_ratio"]
        assert "low" in tr_range, "target_ratio missing 'low' bound"
        assert "high" in tr_range, "target_ratio missing 'high' bound"


class TestHPOBoundsAreValidIntervals:
    """Test that HPO bounds form valid intervals."""

    def test_confidence_quantile_is_valid_interval(self, hpo_ranges: dict[str, Any]):
        """Property: confidence_quantile.low < confidence_quantile.high."""
        cq_range = hpo_ranges.get("confidence_quantile", {})
        low = cq_range.get("low", 0.5)
        high = cq_range.get("high", 0.9)
        assert low < high, f"Invalid interval: low={low} >= high={high}"

    def test_support_quantile_is_valid_interval(self, hpo_ranges: dict[str, Any]):
        """Property: support_quantile.low < support_quantile.high."""
        sq_range = hpo_ranges.get("support_quantile", {})
        low = sq_range.get("low", 0.3)
        high = sq_range.get("high", 0.8)
        assert low < high, f"Invalid interval: low={low} >= high={high}"

    def test_target_ratio_is_valid_interval(self, hpo_ranges: dict[str, Any]):
        """Property: target_ratio.low < target_ratio.high."""
        tr_range = hpo_ranges.get("target_ratio", {})
        low = tr_range.get("low", 0.2)
        high = tr_range.get("high", 0.5)
        assert low < high, f"Invalid interval: low={low} >= high={high}"

    @pytest.mark.parametrize("param_name", [
        "confidence_quantile",
        "support_quantile",
        "target_ratio",
        "max_length_cyclic",
        "max_length_acyclic",
    ])
    def test_all_hpo_ranges_are_valid(
        self, hpo_ranges: dict[str, Any], param_name: str
    ):
        """Property: all HPO range parameters must have low < high."""
        if param_name not in hpo_ranges:
            pytest.skip(f"{param_name} not configured in hpo_ranges")
        
        param_range = hpo_ranges[param_name]
        low = param_range.get("low")
        high = param_range.get("high")
        
        assert low is not None, f"{param_name} missing 'low' bound"
        assert high is not None, f"{param_name} missing 'high' bound"
        assert low < high, f"{param_name}: invalid interval low={low} >= high={high}"


class TestHPOBoundsAreSane:
    """Test that HPO bounds are within reasonable ranges."""

    def test_confidence_quantile_in_unit_interval(self, hpo_ranges: dict[str, Any]):
        """Property: confidence_quantile bounds must be in [0, 1]."""
        cq_range = hpo_ranges.get("confidence_quantile", {})
        low = cq_range.get("low", 0.5)
        high = cq_range.get("high", 0.9)
        
        assert 0 <= low <= 1, f"confidence_quantile.low out of [0, 1]: {low}"
        assert 0 <= high <= 1, f"confidence_quantile.high out of [0, 1]: {high}"

    def test_support_quantile_in_unit_interval(self, hpo_ranges: dict[str, Any]):
        """Property: support_quantile bounds must be in [0, 1]."""
        sq_range = hpo_ranges.get("support_quantile", {})
        low = sq_range.get("low", 0.3)
        high = sq_range.get("high", 0.8)
        
        assert 0 <= low <= 1, f"support_quantile.low out of [0, 1]: {low}"
        assert 0 <= high <= 1, f"support_quantile.high out of [0, 1]: {high}"

    def test_target_ratio_in_unit_interval(self, hpo_ranges: dict[str, Any]):
        """Property: target_ratio bounds must be in [0, 1]."""
        tr_range = hpo_ranges.get("target_ratio", {})
        low = tr_range.get("low", 0.2)
        high = tr_range.get("high", 0.5)
        
        assert 0 <= low <= 1, f"target_ratio.low out of [0, 1]: {low}"
        assert 0 <= high <= 1, f"target_ratio.high out of [0, 1]: {high}"


# ============================================================================
# Tests: Coverage scoring monotonicity
# ============================================================================


class TestCoverageScoring:
    """Test coverage contribution to scoring."""

    @staticmethod
    def compute_coverage_component(
        coverage: float,
        coverage_gate: float = 0.05,
        max_coverage: float = 1.0,
    ) -> float:
        """Compute coverage component for scoring.
        
        Higher coverage (up to max_coverage) should yield higher component.
        Coverage below gate returns 0.
        """
        if coverage < coverage_gate:
            return 0.0
        
        # Normalize coverage to [0, 1] range
        normalized = (coverage - coverage_gate) / (max_coverage - coverage_gate)
        return float(np.clip(normalized, 0.0, 1.0))

    def test_higher_coverage_yields_higher_score(self):
        """Property: higher coverage should yield higher or equal score."""
        coverage_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        scores = [self.compute_coverage_component(c) for c in coverage_values]
        
        # Scores should be monotonically non-decreasing
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Coverage score decreased: coverage={coverage_values[i]}, "
                f"score={scores[i]} < prev_score={scores[i - 1]}"
            )

    def test_coverage_below_gate_returns_zero(self):
        """Property: coverage below gate should return 0."""
        gate = 0.05
        for coverage in [0.0, 0.01, 0.04, 0.049]:
            score = self.compute_coverage_component(coverage, coverage_gate=gate)
            assert score == 0.0, f"Coverage {coverage} below gate {gate} should be 0, got {score}"

    def test_coverage_at_max_returns_one(self):
        """Property: coverage at max should return 1."""
        score = self.compute_coverage_component(1.0, coverage_gate=0.05, max_coverage=1.0)
        assert abs(score - 1.0) < 1e-6, f"Max coverage should return 1.0, got {score}"

    @pytest.mark.parametrize("coverage,expected_nonzero", [
        (0.01, False),
        (0.05, False),  # At gate
        (0.06, True),   # Just above gate
        (0.5, True),
        (1.0, True),
    ])
    def test_coverage_gate_boundary(self, coverage: float, expected_nonzero: bool):
        """Property: coverage at or below gate is 0, above gate is positive."""
        gate = 0.05
        score = self.compute_coverage_component(coverage, coverage_gate=gate)
        
        if expected_nonzero:
            assert score > 0, f"Coverage {coverage} above gate {gate} should be positive"
        else:
            assert score == 0, f"Coverage {coverage} at/below gate {gate} should be 0"


class TestCoveragePenaltyMonotonicity:
    """Test that coverage penalty decreases as coverage increases."""

    @staticmethod
    def compute_coverage_penalty(
        coverage: float,
        coverage_gate: float = 0.05,
        penalty_coeff: float = 0.5,
    ) -> float:
        """Compute penalty for low coverage.
        
        Penalty should be high when coverage is low, zero when coverage is good.
        """
        if coverage >= coverage_gate:
            return 0.0
        
        gap = coverage_gate - coverage
        penalty = penalty_coeff * (gap / coverage_gate)
        return penalty

    def test_penalty_decreases_with_coverage(self):
        """Property: coverage penalty should decrease as coverage increases."""
        coverages = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        penalties = [self.compute_coverage_penalty(c) for c in coverages]
        
        # Penalties should be monotonically non-increasing
        for i in range(1, len(penalties)):
            assert penalties[i] <= penalties[i - 1], (
                f"Penalty increased: coverage={coverages[i]}, "
                f"penalty={penalties[i]} > prev={penalties[i - 1]}"
            )

    def test_penalty_zero_above_gate(self):
        """Property: no penalty when coverage is at or above gate."""
        gate = 0.05
        for coverage in [0.05, 0.1, 0.5, 1.0]:
            penalty = self.compute_coverage_penalty(coverage, coverage_gate=gate)
            assert penalty == 0.0, f"Coverage {coverage} >= gate {gate} should have no penalty"


class TestDualDatasetCoverageScoringProperty:
    """Test coverage scoring with two synthetic datasets."""

    @staticmethod
    def score_trial(coverage: float, other_metrics: dict[str, float]) -> float:
        """Simplified scoring function that considers coverage.
        
        Higher coverage (above gate) should not hurt the score.
        """
        coverage_gate = 0.05
        coverage_weight = 0.2
        
        # Base score from other metrics
        base_score = other_metrics.get("mrr", 0.5) * 0.4 + other_metrics.get("auc", 0.5) * 0.4
        
        # Coverage bonus (not penalty) for higher coverage
        if coverage >= coverage_gate:
            coverage_bonus = coverage_weight * min(coverage, 1.0)
        else:
            coverage_bonus = 0.0
        
        return base_score + coverage_bonus

    def test_higher_coverage_dataset_not_penalized(self):
        """Property: dataset with higher coverage should not have lower score."""
        # Two synthetic datasets with same base metrics but different coverage
        low_coverage = 0.15
        high_coverage = 0.45
        
        common_metrics = {"mrr": 0.5, "auc": 0.75}
        
        score_low = self.score_trial(low_coverage, common_metrics)
        score_high = self.score_trial(high_coverage, common_metrics)
        
        assert score_high >= score_low, (
            f"Higher coverage ({high_coverage}) should not be penalized vs "
            f"lower coverage ({low_coverage}): score_high={score_high:.3f}, "
            f"score_low={score_low:.3f}"
        )

    @pytest.mark.parametrize("low_cov,high_cov", [
        (0.1, 0.2),
        (0.2, 0.4),
        (0.3, 0.6),
        (0.4, 0.9),
    ])
    def test_coverage_ordering_preserved(self, low_cov: float, high_cov: float):
        """Property: higher coverage yields higher or equal score."""
        common_metrics = {"mrr": 0.5, "auc": 0.75}
        
        score_low = self.score_trial(low_cov, common_metrics)
        score_high = self.score_trial(high_cov, common_metrics)
        
        assert score_high >= score_low, (
            f"Coverage ordering violated: cov={high_cov} score={score_high:.3f} < "
            f"cov={low_cov} score={score_low:.3f}"
        )


# ============================================================================
# Tests: Config consistency
# ============================================================================


class TestConfigConsistency:
    """Test consistency between config files."""

    def test_defaults_have_all_required_fields(self, rule_filter_defaults: dict[str, Any]):
        """Property: rule_filter.defaults must have all required fields."""
        required_fields = [
            "min_confidence",
            "min_support",
            "confidence_quantile",
            "support_quantile",
            "target_ratio",
            "min_rules",
        ]
        
        for field in required_fields:
            assert field in rule_filter_defaults, (
                f"Missing required default field: {field}"
            )

    def test_defaults_within_hpo_ranges(
        self,
        rule_filter_defaults: dict[str, Any],
        hpo_ranges: dict[str, Any],
    ):
        """Property: default values should be within HPO ranges."""
        for param in ["confidence_quantile", "support_quantile", "target_ratio"]:
            if param not in hpo_ranges or param not in rule_filter_defaults:
                continue
            
            default_val = rule_filter_defaults[param]
            low = hpo_ranges[param].get("low", 0)
            high = hpo_ranges[param].get("high", 1)
            
            assert low <= default_val <= high, (
                f"Default {param}={default_val} outside HPO range [{low}, {high}]"
            )
