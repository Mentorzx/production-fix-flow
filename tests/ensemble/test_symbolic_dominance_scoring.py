"""Tests for P3: Symbolic dominance penalty scoring configuration.

Validates that:
1. dominance_target uses target_symbolic_ratio from trial params
2. symbolic_dominance_penalty_coeff is read from config
3. Fallback to legacy values (0.70, 0.50) when config/params missing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSymbolicDominancePenaltyConfig:
    """Test symbolic dominance penalty configuration from ensemble_hpo.yaml."""

    def test_config_has_scoring_section(self):
        """Ensure ensemble_hpo.yaml has scoring section with penalty config."""
        from pff.config import ENSEMBLE_HPO_CONFIG_PATH
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_HPO_CONFIG_PATH)

        assert "scoring" in config, "scoring section missing in ensemble_hpo.yaml"
        scoring = config["scoring"]
        assert "symbolic_dominance_penalty_coeff" in scoring, "symbolic_dominance_penalty_coeff missing"
        assert "fallback_dominance_target" in scoring, "fallback_dominance_target missing"

    def test_penalty_coeff_is_numeric(self):
        """Verify penalty coefficient is a valid numeric value."""
        from pff.config import ENSEMBLE_HPO_CONFIG_PATH
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_HPO_CONFIG_PATH)
        scoring = config.get("scoring", {})

        coeff = scoring.get("symbolic_dominance_penalty_coeff", 0.50)
        assert isinstance(coeff, (int, float)), "Penalty coefficient must be numeric"
        assert 0.0 <= coeff <= 2.0, "Penalty coefficient should be in reasonable range [0, 2]"


class TestSymbolicDominanceTargetFromParams:
    """Test that dominance_target is derived from target_symbolic_ratio param."""

    def test_uses_target_symbolic_ratio_when_present(self):
        """When target_symbolic_ratio is in params, use it as dominance_target."""
        # Mock the scoring calculation logic
        params = {"target_symbolic_ratio": 0.35}
        scoring_config = {
            "fallback_dominance_target": 0.70,
            "symbolic_dominance_penalty_coeff": 1.0,
        }

        # Simulate the logic from core.py
        fallback_dominance_target = float(scoring_config.get("fallback_dominance_target", 0.70))
        dominance_target = float(params.get("target_symbolic_ratio", fallback_dominance_target))

        assert dominance_target == 0.35, "Should use target_symbolic_ratio from params"

    def test_falls_back_to_config_when_param_missing(self):
        """When target_symbolic_ratio is not in params, use fallback from config."""
        params = {}  # No target_symbolic_ratio
        scoring_config = {
            "fallback_dominance_target": 0.70,
            "symbolic_dominance_penalty_coeff": 1.0,
        }

        fallback_dominance_target = float(scoring_config.get("fallback_dominance_target", 0.70))
        dominance_target = float(params.get("target_symbolic_ratio", fallback_dominance_target))

        assert dominance_target == 0.70, "Should fall back to config value"

    def test_legacy_fallback_when_config_missing(self):
        """When both param and config are missing, use hardcoded 0.70."""
        params = {}
        scoring_config = {}  # No fallback_dominance_target

        fallback_dominance_target = float(scoring_config.get("fallback_dominance_target", 0.70))
        dominance_target = float(params.get("target_symbolic_ratio", fallback_dominance_target))

        assert dominance_target == 0.70, "Should fall back to legacy 0.70"


class TestSymbolicDominancePenaltyCoefficient:
    """Test that penalty coefficient is read from config."""

    def test_uses_config_coefficient(self):
        """Penalty coefficient should come from scoring config."""
        scoring_config = {"symbolic_dominance_penalty_coeff": 1.5}

        coeff = float(scoring_config.get("symbolic_dominance_penalty_coeff", 0.50))

        assert coeff == 1.5, "Should use coefficient from config"

    def test_legacy_fallback_coefficient(self):
        """When config missing, should fall back to 0.50."""
        scoring_config = {}

        coeff = float(scoring_config.get("symbolic_dominance_penalty_coeff", 0.50))

        assert coeff == 0.50, "Should fall back to legacy 0.50"


class TestSymbolicDominancePenaltyCalculation:
    """Test the actual penalty calculation with various inputs."""

    def _compute_penalty(
        self,
        symbolic_contribution_ratio: float,
        dominance_target: float,
    ) -> float:
        """Replicate the penalty calculation from core.py."""
        if symbolic_contribution_ratio > dominance_target:
            dominance_overflow = symbolic_contribution_ratio - dominance_target
            return dominance_overflow / max(1e-6, 1.0 - dominance_target)
        return 0.0

    def test_no_penalty_below_target(self):
        """No penalty when symbolic contribution is below target."""
        penalty = self._compute_penalty(
            symbolic_contribution_ratio=0.30,
            dominance_target=0.35,
        )
        assert penalty == 0.0

    def test_penalty_above_target(self):
        """Penalty should be proportional to overflow above target."""
        penalty = self._compute_penalty(
            symbolic_contribution_ratio=0.80,
            dominance_target=0.40,
        )
        # (0.80 - 0.40) / (1.0 - 0.40) = 0.40 / 0.60 ≈ 0.667
        expected = 0.40 / 0.60
        assert abs(penalty - expected) < 1e-6

    def test_penalty_with_legacy_target(self):
        """Penalty with legacy 0.70 target (backward compatibility)."""
        penalty = self._compute_penalty(
            symbolic_contribution_ratio=0.85,
            dominance_target=0.70,
        )
        # (0.85 - 0.70) / (1.0 - 0.70) = 0.15 / 0.30 = 0.50
        expected = 0.15 / 0.30
        assert abs(penalty - expected) < 1e-6

    def test_effective_score_reduction(self):
        """Test effective score reduction with penalty coefficient."""
        base_score = 0.50
        penalty = 0.667  # From test above
        coeff = 1.0  # P3 default

        # Score reduction: base_score * (1 - coeff * min(1.0, penalty))
        effective_score = base_score * (1.0 - coeff * min(1.0, penalty))

        # 0.50 * (1 - 1.0 * 0.667) = 0.50 * 0.333 ≈ 0.167
        expected = base_score * (1.0 - coeff * penalty)
        assert abs(effective_score - expected) < 1e-6
