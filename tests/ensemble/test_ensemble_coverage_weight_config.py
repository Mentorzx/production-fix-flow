"""Tests for P2.2 - Configurable coverage_weight in ensemble score.

Verifies that the coverage weight in rules component is configurable via
config/models/ensemble.yaml and properly clamped to [0.15, 0.40].

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pff.config import ENSEMBLE_CONFIG_PATH
from scripts.optimization.trials.config_loader import clear_config_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestCoverageWeightConfig:
    """Test coverage_weight configuration in ensemble.yaml."""

    def test_config_has_coverage_weight(self):
        """Verify ensemble.yaml contains coverage_weight in balancing.rules."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        assert "balancing" in config, "balancing section should exist"
        balancing = config["balancing"]
        
        assert "rules" in balancing, "balancing.rules section should exist"
        rules_config = balancing["rules"]
        
        assert "coverage_weight" in rules_config, "coverage_weight should be in config"

    def test_default_coverage_weight_is_0_2(self):
        """Verify default coverage_weight is 0.2 (matches previous behavior)."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        coverage_weight = config.get("balancing", {}).get("rules", {}).get("coverage_weight", 0.2)
        
        assert coverage_weight == 0.2, f"Default coverage_weight should be 0.2, got {coverage_weight}"


class TestCoverageWeightInBounds:
    """Test that bounds.py uses coverage_weight from config."""

    def test_get_rules_coverage_weight_reads_config(self):
        """Verify get_rules_coverage_weight reads from config."""
        from scripts.optimization.trials.bounds import get_rules_coverage_weight

        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {
                "rules": {
                    "coverage_weight": 0.3,
                }
            }
        }

        weight = get_rules_coverage_weight(mock_fm)

        assert weight == 0.3

    def test_get_rules_coverage_weight_clamps_low(self):
        """Verify coverage_weight is clamped to minimum 0.15."""
        from scripts.optimization.trials.bounds import get_rules_coverage_weight

        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {
                "rules": {
                    "coverage_weight": 0.05,  # Below minimum
                }
            }
        }

        weight = get_rules_coverage_weight(mock_fm)

        assert weight == 0.15, f"Weight should be clamped to 0.15, got {weight}"

    def test_get_rules_coverage_weight_clamps_high(self):
        """Verify coverage_weight is clamped to maximum 0.40."""
        from scripts.optimization.trials.bounds import get_rules_coverage_weight

        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {
                "rules": {
                    "coverage_weight": 0.8,  # Above maximum
                }
            }
        }

        weight = get_rules_coverage_weight(mock_fm)

        assert weight == 0.40, f"Weight should be clamped to 0.40, got {weight}"

    def test_get_rules_coverage_weight_default_on_missing(self):
        """Verify default 0.2 is used when config is missing."""
        from scripts.optimization.trials.bounds import get_rules_coverage_weight

        mock_fm = MagicMock()
        mock_fm.read.return_value = {}  # Empty config

        weight = get_rules_coverage_weight(mock_fm)

        assert weight == 0.2, f"Default weight should be 0.2, got {weight}"

    def test_get_rules_coverage_weight_default_on_exception(self):
        """Verify default 0.2 is used when config read fails."""
        from scripts.optimization.trials.bounds import get_rules_coverage_weight

        mock_fm = MagicMock()
        mock_fm.read.side_effect = Exception("Config read failed")

        weight = get_rules_coverage_weight(mock_fm)

        assert weight == 0.2, f"Default weight should be 0.2 on error, got {weight}"


class TestBlendScoresWithCoverageWeight:
    """Test that blend_scores properly uses weights."""

    def test_blend_scores_basic(self):
        """Test blend_scores computes weighted average correctly."""
        from scripts.optimization.trials.bounds import blend_scores

        # Simple case: equal weights
        result = blend_scores([
            (0.8, 0.5),
            (0.6, 0.3),
            (0.4, 0.2),
        ])

        # Expected: (0.8*0.5 + 0.6*0.3 + 0.4*0.2) / (0.5+0.3+0.2) = 0.66
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2) / 1.0
        assert abs(result - expected) < 0.001

    def test_blend_scores_ignores_zero_weight(self):
        """Test blend_scores ignores components with zero weight."""
        from scripts.optimization.trials.bounds import blend_scores

        result = blend_scores([
            (0.8, 0.5),
            (0.0, 0.0),  # Should be ignored
            (0.6, 0.5),
        ])

        expected = (0.8 * 0.5 + 0.6 * 0.5) / 1.0
        assert abs(result - expected) < 0.001


class TestRuleComponentWeights:
    """Tests for config-driven rule component weights (confidence/recall/coverage)."""

    def test_rule_component_weights_scale_with_coverage(self):
        """Verify confidence/recall weights are scaled when coverage changes."""
        from scripts.optimization.trials.bounds import get_rule_component_weights

        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {
                "rules": {
                    "confidence_weight": 0.5,
                    "recall_weight": 0.3,
                    "coverage_weight": 0.25,
                }
            }
        }

        conf_w, recall_w, coverage_w = get_rule_component_weights(mock_fm)

        assert coverage_w == 0.25
        # Remaining mass = 0.75; ratio 0.5:0.3 ⇒ normalized to 0.46875 and 0.28125
        assert abs(conf_w + recall_w + coverage_w - 1.0) < 1e-6
        assert conf_w > recall_w

    def test_rule_component_weights_handles_zero_conf_recall(self):
        """Verify degenerate config splits remaining mass evenly."""
        from scripts.optimization.trials.bounds import get_rule_component_weights

        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {
                "rules": {
                    "confidence_weight": 0.0,
                    "recall_weight": 0.0,
                    "coverage_weight": 0.2,
                }
            }
        }

        conf_w, recall_w, coverage_w = get_rule_component_weights(mock_fm)

        assert coverage_w == 0.2
        assert abs(conf_w - recall_w) < 1e-6
        assert abs(conf_w + recall_w + coverage_w - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
