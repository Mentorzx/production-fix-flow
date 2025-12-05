"""Tests for P1 Implementation Items from Consensus Plan.

P1.1 - Stronger LightGBM regularization (config-driven)
P1.2 - AnyBURL rule filter HPO ranges (config-driven)
P1.3 - Conservative expansion of AnyBURL rule length
P1.4 - Adaptive expert weighting (config-driven, default OFF)

These tests verify configuration changes and new functionality without
requiring real KG data or full pipeline execution.

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pff.config import (
    ENSEMBLE_CONFIG_PATH,
    ROTATE_CONFIG_PATH,
    RULE_FILTER_CONFIG_PATH,
    RULE_FILTER_HPO_CONFIG_PATH,
)


class TestP1_1_LightGBMRegularization:
    """P1.1: Test stronger LightGBM regularization params in config."""

    def test_rotate_yaml_has_regularization_params(self):
        """Verify rotate.yaml contains stronger regularization params."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)

        lgb_params = config.get("lightgbm", {}).get("params", {})

        # P1.1 - Verify stronger regularization params exist
        assert "reg_alpha" in lgb_params, "reg_alpha (L1) should be in config"
        assert "reg_lambda" in lgb_params, "reg_lambda (L2) should be in config"
        assert "min_data_in_leaf" in lgb_params, "min_data_in_leaf should be in config"
        assert "max_bin" in lgb_params, "max_bin should be in config"

        # Verify values are reasonable for regularization
        assert lgb_params["reg_alpha"] >= 0.1, "reg_alpha should be >= 0.1 for regularization"
        assert lgb_params["reg_lambda"] >= 1.0, "reg_lambda should be >= 1.0 for regularization"
        assert lgb_params["min_data_in_leaf"] >= 20, "min_data_in_leaf should be >= 20"
        assert lgb_params["max_bin"] <= 255, "max_bin should be <= 255"

    def test_num_leaves_configured(self):
        """Verify num_leaves is configured appropriately."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)

        lgb_params = config.get("lightgbm", {}).get("params", {})
        num_leaves = lgb_params.get("num_leaves", 31)

        # P1.1 - num_leaves should be reasonable (31 is production default)
        # Provides good balance between model capacity and regularization
        assert num_leaves <= 63, f"num_leaves should be <= 63, got {num_leaves}"
        assert num_leaves >= 15, f"num_leaves should be >= 15 for sufficient capacity, got {num_leaves}"


class TestP1_2_RuleFilterHPORanges:
    """P1.2: Test HPO ranges in rule_filter.yaml."""

    def test_rule_filter_yaml_has_hpo_ranges(self):
        """Verify rule_filter.yaml contains hpo_ranges section."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)

        rule_filter_cfg = config.get("rule_filter", {})
        assert "hpo_ranges" in rule_filter_cfg, "hpo_ranges section should exist in rule_filter.hpo_ranges"

        hpo_ranges = rule_filter_cfg.get("hpo_ranges", {})

        # Verify expected range keys exist
        expected_keys = [
            "confidence_quantile",
            "support_quantile",
            "target_ratio",
            "max_length_cyclic",
            "max_length_acyclic",
        ]
        for key in expected_keys:
            assert key in hpo_ranges, f"{key} should be in hpo_ranges"
            assert "low" in hpo_ranges[key], f"{key} should have 'low' bound"
            assert "high" in hpo_ranges[key], f"{key} should have 'high' bound"

    def test_hpo_ranges_have_valid_bounds(self):
        """Verify HPO ranges have low < high."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})

        for key, bounds in hpo_ranges.items():
            low = bounds.get("low", 0)
            high = bounds.get("high", 0)
            assert low < high, f"{key}: low ({low}) should be < high ({high})"


class TestP1_3_RuleLengthExpansion:
    """P1.3: Test conservative expansion of AnyBURL rule lengths."""

    def test_cyclic_range_expanded(self):
        """Verify max_length_cyclic range is conservatively expanded."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})

        cyclic_range = hpo_ranges.get("max_length_cyclic", {})
        low = cyclic_range.get("low", 1)
        high = cyclic_range.get("high", 4)

        # P1.3 - Conservative expansion: low should be >= 3
        assert low >= 3, f"max_length_cyclic low should be >= 3, got {low}"
        assert high >= 4, f"max_length_cyclic high should be >= 4, got {high}"

    def test_acyclic_range_expanded(self):
        """Verify max_length_acyclic range is conservatively expanded."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})

        acyclic_range = hpo_ranges.get("max_length_acyclic", {})
        low = acyclic_range.get("low", 1)
        high = acyclic_range.get("high", 4)

        # P1.3 - Conservative expansion: high should be >= 5 for acyclic
        assert low >= 3, f"max_length_acyclic low should be >= 3, got {low}"
        assert high >= 5, f"max_length_acyclic high should be >= 5, got {high}"


class TestP1_4_AdaptiveWeighting:
    """P1.4: Test adaptive expert weighting config and implementation."""

    def test_ensemble_yaml_has_adaptive_weighting(self):
        """Verify ensemble.yaml contains adaptive_weighting section."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        assert "adaptive_weighting" in config, "adaptive_weighting section should exist"

        aw_config = config["adaptive_weighting"]

        # Verify expected keys
        assert "enabled" in aw_config, "enabled flag should exist"
        assert "weight_clip_min" in aw_config, "weight_clip_min should exist"
        assert "weight_clip_max" in aw_config, "weight_clip_max should exist"
        assert "log_weights" in aw_config, "log_weights should exist"

    def test_adaptive_weighting_default_off(self):
        """Verify adaptive weighting is OFF by default for backward compatibility."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        aw_config = config.get("adaptive_weighting", {})
        enabled = aw_config.get("enabled", True)

        assert enabled is False, "adaptive_weighting should be disabled by default"

    def test_adaptive_weighting_has_strategies(self):
        """Verify adaptive weighting has strategy definitions."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)

        aw_config = config.get("adaptive_weighting", {})
        strategies = aw_config.get("strategies", {})

        # Verify expected strategies exist
        expected_strategies = ["balanced", "neural_dominant", "symbolic_dominant"]
        for strategy in expected_strategies:
            assert strategy in strategies, f"{strategy} strategy should exist"
            weights = strategies[strategy]
            assert "neural" in weights, f"{strategy} should have neural weight"
            assert "symbolic" in weights, f"{strategy} should have symbolic weight"
            assert "hybrid" in weights, f"{strategy} should have hybrid weight"

    @pytest.fixture
    def mock_file_manager(self):
        """Create a mock FileManager that returns test config."""
        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {"neural": 0.2, "rules": 0.2, "lightgbm": 0.6},
            "adaptive_weighting": {
                "enabled": True,
                "weight_clip_min": 0.5,
                "weight_clip_max": 2.0,
                "log_weights": False,  # Disable logging in tests
                "strategies": {
                    "balanced": {"neural": 0.35, "symbolic": 0.35, "hybrid": 0.30},
                    "neural_dominant": {"neural": 0.5, "symbolic": 0.2, "hybrid": 0.3},
                    "symbolic_dominant": {"neural": 0.2, "symbolic": 0.5, "hybrid": 0.3},
                },
            },
        }
        return mock_fm

    def test_compute_adaptive_weights_disabled(self, mock_file_manager):
        """Verify static weights returned when adaptive weighting is disabled."""
        mock_file_manager.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {"neural": 0.2, "rules": 0.2, "lightgbm": 0.6},
            "adaptive_weighting": {"enabled": False},
        }

        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            # Create trainer with mocked paths (won't actually load models)
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                weights = trainer.compute_adaptive_weights(
                    rule_violations=3,
                    symbolic_coverage=0.4,
                    oov_ratio=0.2,
                )

                # When disabled, should return static weights
                assert "neural" in weights
                assert "symbolic" in weights
                assert "hybrid" in weights
                assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1

    def test_compute_adaptive_weights_high_violations(self, mock_file_manager):
        """Verify symbolic weight boosted when rule violations are high."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                # Get weights with high violations (>5)
                weights_high = trainer.compute_adaptive_weights(
                    rule_violations=10,
                    symbolic_coverage=0.4,
                    oov_ratio=0.2,
                )

                # Get weights with no violations
                weights_zero = trainer.compute_adaptive_weights(
                    rule_violations=0,
                    symbolic_coverage=0.4,
                    oov_ratio=0.2,
                )

                # High violations should boost symbolic weight
                assert weights_high["symbolic"] > weights_zero["symbolic"], \
                    "Symbolic weight should be higher with more violations"

    def test_compute_adaptive_weights_low_coverage(self, mock_file_manager):
        """Verify neural weight is boosted when symbolic coverage is very low (<0.3).

        The canonical logic in oov_solution_config.py applies a 1.2x multiplier
        to neural weight when symbolic_coverage < 0.3. This test verifies that
        adjustment is applied correctly within the same strategy.

        Note: Strategy selection also depends on coverage, so we compare
        scenarios within the same strategy ('base') to isolate the coverage effect.
        """
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                # Get weights with very low coverage (<0.3, triggers neural boost)
                weights_very_low = trainer.compute_adaptive_weights(
                    rule_violations=2,
                    symbolic_coverage=0.1,  # < 0.3 triggers neural *= 1.2
                    oov_ratio=0.2,
                )

                # Get weights with moderate coverage (>=0.3, but <0.5 to stay in 'base' strategy)
                weights_moderate = trainer.compute_adaptive_weights(
                    rule_violations=2,
                    symbolic_coverage=0.4,  # >= 0.3 no boost, < 0.5 keeps 'base' strategy
                    oov_ratio=0.2,
                )

                # Very low coverage should have neural boost applied
                # (both use 'base' strategy, but 0.1 < 0.3 triggers neural *= 1.2)
                assert weights_very_low["neural"] > weights_moderate["neural"], \
                    f"Neural weight should be higher with very low coverage (0.1): " \
                    f"got {weights_very_low['neural']:.3f} vs {weights_moderate['neural']:.3f}"

    def test_compute_adaptive_weights_normalized(self, mock_file_manager):
        """Verify adaptive weights always sum to 1."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer

            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager,
                )

                # Test various scenarios
                test_cases = [
                    (0, 0.1, 0.0),
                    (10, 0.5, 0.3),
                    (3, 0.8, 0.7),
                    (0, 0.2, 0.9),
                ]

                for violations, coverage, oov in test_cases:
                    weights = trainer.compute_adaptive_weights(
                        rule_violations=violations,
                        symbolic_coverage=coverage,
                        oov_ratio=oov,
                    )
                    total = sum(weights.values())
                    assert abs(total - 1.0) < 0.001, \
                        f"Weights should sum to 1, got {total} for case ({violations}, {coverage}, {oov})"


class TestP1_CoreOptimizationRanges:
    """Test that core.py reads HPO ranges from config."""

    def test_hpo_ranges_structure(self):
        """Verify HPO ranges have correct structure for core.py consumption."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(RULE_FILTER_HPO_CONFIG_PATH)
        hpo_ranges = config.get("rule_filter", {}).get("hpo_ranges", {})

        # Verify the structure matches what core.py expects
        cyclic = hpo_ranges.get("max_length_cyclic", {})
        acyclic = hpo_ranges.get("max_length_acyclic", {})

        # core.py uses int() on these values
        assert isinstance(cyclic.get("low"), (int, float)), "cyclic low should be numeric"
        assert isinstance(cyclic.get("high"), (int, float)), "cyclic high should be numeric"
        assert isinstance(acyclic.get("low"), (int, float)), "acyclic low should be numeric"
        assert isinstance(acyclic.get("high"), (int, float)), "acyclic high should be numeric"

        # Values should be valid for trial.suggest_int
        assert int(cyclic["low"]) >= 1, "cyclic low should be >= 1"
        assert int(acyclic["low"]) >= 1, "acyclic low should be >= 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
