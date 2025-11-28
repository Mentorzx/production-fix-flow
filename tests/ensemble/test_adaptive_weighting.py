"""Tests for Adaptive Expert Weighting.

P1.4: Verify that AdvancedEnsembleTrainer correctly implements adaptive
expert weighting with:
- Static weights when disabled (backward compatible)
- Delegation to OOVAwareEnsembleManager when enabled
- Proper clipping and normalization
- Debug-level logging (per AGENTS.md)

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pff.config import ENSEMBLE_CONFIG_PATH


class TestAdaptiveWeightingConfig:
    """Test adaptive_weighting config section."""

    def test_config_has_adaptive_weighting_section(self):
        """Verify ensemble.yaml contains adaptive_weighting section."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)
        
        assert "adaptive_weighting" in config, "adaptive_weighting section should exist"
        
        aw_config = config["adaptive_weighting"]
        
        # Required fields
        assert "enabled" in aw_config, "enabled flag required"
        assert "weight_clip_min" in aw_config, "weight_clip_min required"
        assert "weight_clip_max" in aw_config, "weight_clip_max required"
        assert "log_weights" in aw_config, "log_weights required"

    def test_adaptive_weighting_default_off(self):
        """Verify adaptive weighting is OFF by default for backward compatibility."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)
        
        aw_config = config.get("adaptive_weighting", {})
        enabled = aw_config.get("enabled", True)
        
        assert enabled is False, "adaptive_weighting.enabled should be False by default"

    def test_log_weights_default_false(self):
        """Verify log_weights is False by default (AGENTS.md: avoid noisy logs)."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)
        
        aw_config = config.get("adaptive_weighting", {})
        log_weights = aw_config.get("log_weights", True)
        
        assert log_weights is False, "log_weights should be False by default"

    def test_clipping_bounds_valid(self):
        """Verify clipping bounds are reasonable."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ENSEMBLE_CONFIG_PATH)
        
        aw_config = config.get("adaptive_weighting", {})
        clip_min = aw_config.get("weight_clip_min", 0.5)
        clip_max = aw_config.get("weight_clip_max", 2.0)
        
        assert 0 < clip_min < 1, "weight_clip_min should be in (0, 1)"
        assert clip_max > 1, "weight_clip_max should be > 1"
        assert clip_min < clip_max, "clip_min should be < clip_max"


class TestAdaptiveWeightingDisabled:
    """Test adaptive weighting when disabled (backward compatible mode)."""

    @pytest.fixture
    def mock_file_manager_disabled(self):
        """Create a mock FileManager with adaptive_weighting disabled."""
        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {
                "neural": 0.2,
                "rules": 0.2,
                "lightgbm": 0.6,
            },
            "adaptive_weighting": {
                "enabled": False,
                "weight_clip_min": 0.5,
                "weight_clip_max": 2.0,
                "log_weights": False,
            },
        }
        return mock_fm

    def test_disabled_returns_static_weights(self, mock_file_manager_disabled):
        """Verify static weights returned when adaptive weighting is disabled."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_disabled,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_disabled,
                )
                
                weights = trainer.compute_adaptive_weights(
                    rule_violations=10,
                    symbolic_coverage=0.1,
                    oov_ratio=0.9,
                )
                
                # Should return config static weights regardless of input
                assert weights["neural"] == 0.2
                assert weights["symbolic"] == 0.2
                assert weights["hybrid"] == 0.6

    def test_disabled_ignores_extreme_inputs(self, mock_file_manager_disabled):
        """Verify extreme inputs don't affect static weights when disabled."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_disabled,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_disabled,
                )
                
                # Various extreme scenarios should all return same static weights
                test_cases = [
                    (0, 0.0, 0.0),
                    (100, 1.0, 1.0),
                    (50, 0.5, 0.5),
                ]
                
                for violations, coverage, oov in test_cases:
                    weights = trainer.compute_adaptive_weights(
                        rule_violations=violations,
                        symbolic_coverage=coverage,
                        oov_ratio=oov,
                    )
                    assert weights["neural"] == 0.2
                    assert weights["symbolic"] == 0.2
                    assert weights["hybrid"] == 0.6


class TestAdaptiveWeightingEnabled:
    """Test adaptive weighting when enabled."""

    @pytest.fixture
    def mock_file_manager_enabled(self):
        """Create a mock FileManager with adaptive_weighting enabled."""
        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {
                "neural": 0.2,
                "rules": 0.2,
                "lightgbm": 0.6,
            },
            "adaptive_weighting": {
                "enabled": True,
                "weight_clip_min": 0.5,
                "weight_clip_max": 2.0,
                "log_weights": False,
                "strategies": {
                    "balanced": {"neural": 0.35, "symbolic": 0.35, "hybrid": 0.30},
                    "neural_dominant": {"neural": 0.5, "symbolic": 0.2, "hybrid": 0.3},
                    "symbolic_dominant": {"neural": 0.2, "symbolic": 0.5, "hybrid": 0.3},
                },
            },
        }
        return mock_fm

    def test_enabled_delegates_to_oov_manager(self, mock_file_manager_enabled):
        """Verify enabled mode delegates to OOVAwareEnsembleManager."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_enabled,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_enabled,
                )
                
                # Mock the oov_manager
                mock_oov_manager = Mock()
                mock_oov_manager.compute_adaptive_expert_weights.return_value = {
                    "neural": 0.4,
                    "symbolic": 0.3,
                    "hybrid": 0.3,
                }
                mock_oov_manager._assess_data_quality.return_value = "good"
                trainer.oov_manager = mock_oov_manager
                
                weights = trainer.compute_adaptive_weights(
                    rule_violations=5,
                    symbolic_coverage=0.4,
                    oov_ratio=0.3,
                )
                
                # Verify delegation happened
                assert mock_oov_manager.compute_adaptive_expert_weights.called

    def test_enabled_applies_clipping(self, mock_file_manager_enabled):
        """Verify clipping is applied to extreme weights."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_enabled,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_enabled,
                )
                
                # Mock oov_manager to return extreme weights
                mock_oov_manager = Mock()
                mock_oov_manager.compute_adaptive_expert_weights.return_value = {
                    "neural": 0.9,  # Very high
                    "symbolic": 0.05,  # Very low
                    "hybrid": 0.05,  # Very low
                }
                mock_oov_manager._assess_data_quality.return_value = "fair"
                trainer.oov_manager = mock_oov_manager
                
                weights = trainer.compute_adaptive_weights(
                    rule_violations=0,
                    symbolic_coverage=0.1,
                    oov_ratio=0.8,
                )
                
                # All weights should be within clipped bounds after normalization
                # With clip_min=0.5 and clip_max=2.0, effective bounds per weight are ~16.5% to ~66%
                min_weight = trainer.weight_clip_min * 0.33
                max_weight = trainer.weight_clip_max * 0.33
                
                for key, value in weights.items():
                    # After clipping and normalization, values should be bounded
                    assert value >= 0.1, f"{key} weight too low after clipping"
                    assert value <= 0.8, f"{key} weight too high after clipping"

    def test_enabled_normalizes_to_one(self, mock_file_manager_enabled):
        """Verify weights always sum to 1 after normalization."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_enabled,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_enabled,
                )
                
                # Mock with various weight distributions
                test_weights = [
                    {"neural": 0.5, "symbolic": 0.3, "hybrid": 0.2},
                    {"neural": 0.1, "symbolic": 0.1, "hybrid": 0.8},
                    {"neural": 0.33, "symbolic": 0.33, "hybrid": 0.34},
                ]
                
                for raw_weights in test_weights:
                    mock_oov_manager = Mock()
                    mock_oov_manager.compute_adaptive_expert_weights.return_value = raw_weights
                    mock_oov_manager._assess_data_quality.return_value = "good"
                    trainer.oov_manager = mock_oov_manager
                    
                    weights = trainer.compute_adaptive_weights(
                        rule_violations=3,
                        symbolic_coverage=0.5,
                        oov_ratio=0.2,
                    )
                    
                    total = sum(weights.values())
                    assert abs(total - 1.0) < 0.001, f"Weights should sum to 1, got {total}"


class TestAdaptiveWeightingLogging:
    """Test logging behavior for adaptive weighting."""

    @pytest.fixture
    def mock_file_manager_with_logging(self):
        """Create a mock FileManager with log_weights enabled."""
        mock_fm = MagicMock()
        mock_fm.read.return_value = {
            "balancing": {"symbolic_dominance_threshold": 0.85},
            "ensemble_weights": {"neural": 0.2, "rules": 0.2, "lightgbm": 0.6},
            "adaptive_weighting": {
                "enabled": True,
                "weight_clip_min": 0.5,
                "weight_clip_max": 2.0,
                "log_weights": True,  # Logging enabled
                "strategies": {
                    "balanced": {"neural": 0.35, "symbolic": 0.35, "hybrid": 0.30},
                },
            },
        }
        return mock_fm

    @patch("pff.validators.ensembles.advanced_trainer.logger")
    def test_logging_uses_debug_level(self, mock_logger, mock_file_manager_with_logging):
        """Verify weight logging uses debug level per AGENTS.md."""
        with patch(
            "pff.validators.ensembles.advanced_trainer.FileManager",
            return_value=mock_file_manager_with_logging,
        ):
            from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer
            
            with patch.object(AdvancedEnsembleTrainer, "_resolve_lightgbm_path"):
                trainer = AdvancedEnsembleTrainer(
                    neural_model_path="/fake/path",
                    rules_path="/fake/rules.tsv",
                    lightgbm_model_path="/fake/lgb.bin",
                    file_manager=mock_file_manager_with_logging,
                )
                
                mock_oov_manager = Mock()
                mock_oov_manager.compute_adaptive_expert_weights.return_value = {
                    "neural": 0.35, "symbolic": 0.35, "hybrid": 0.30
                }
                mock_oov_manager._assess_data_quality.return_value = "good"
                trainer.oov_manager = mock_oov_manager
                
                # Reset mock to ignore __init__ calls
                mock_logger.reset_mock()
                
                trainer.compute_adaptive_weights(
                    rule_violations=3,
                    symbolic_coverage=0.4,
                    oov_ratio=0.2,
                )
                
                # Should use debug, NOT info for weight logging
                assert mock_logger.debug.called, "Should log with debug level"
                # Check that the debug call contains "Adaptive weights"
                debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
                assert any("Adaptive weights" in str(c) for c in debug_calls), \
                    f"Debug should log adaptive weights, got: {debug_calls}"


class TestOOVManagerIntegration:
    """Test integration with OOVAwareEnsembleManager."""

    def test_oov_manager_compute_adaptive_expert_weights_exists(self):
        """Verify OOVAwareEnsembleManager has the compute_adaptive_expert_weights method."""
        from pff.validators.ensembles.oov_solution_config import OOVAwareEnsembleManager
        
        manager = OOVAwareEnsembleManager()
        assert hasattr(manager, "compute_adaptive_expert_weights")
        assert callable(manager.compute_adaptive_expert_weights)

    def test_oov_manager_assess_data_quality_exists(self):
        """Verify OOVAwareEnsembleManager has _assess_data_quality method."""
        from pff.validators.ensembles.oov_solution_config import OOVAwareEnsembleManager
        
        manager = OOVAwareEnsembleManager()
        assert hasattr(manager, "_assess_data_quality")
        assert callable(manager._assess_data_quality)

    def test_compute_adaptive_expert_weights_signature(self):
        """Verify compute_adaptive_expert_weights accepts expected parameters."""
        from pff.validators.ensembles.oov_solution_config import OOVAwareEnsembleManager
        
        manager = OOVAwareEnsembleManager()
        
        # Should accept input_quality dict, rule_violations int, symbolic_coverage float
        input_quality = {
            "oov_ratio": 0.3,
            "recommended_strategy": "base",
            "data_quality": "good",
        }
        
        result = manager.compute_adaptive_expert_weights(
            input_quality=input_quality,
            rule_violations=5,
            symbolic_coverage=0.4,
        )
        
        # Should return dict with expert weights
        assert isinstance(result, dict)
        assert "neural" in result or "symbolic" in result or "hybrid" in result

    def test_assess_data_quality_returns_valid_quality(self):
        """Verify _assess_data_quality returns valid quality strings."""
        from pff.validators.ensembles.oov_solution_config import OOVAwareEnsembleManager
        
        manager = OOVAwareEnsembleManager()
        
        valid_qualities = {"excellent", "good", "fair", "poor"}
        
        test_cases = [
            (0.1, 0.7),  # Low OOV, high complexity
            (0.5, 0.4),  # Medium OOV, medium complexity
            (0.9, 0.1),  # High OOV, low complexity
        ]
        
        for oov_ratio, complexity in test_cases:
            quality = manager._assess_data_quality(oov_ratio, complexity)
            assert quality in valid_qualities, f"Invalid quality: {quality}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
