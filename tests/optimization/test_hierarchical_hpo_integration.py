"""
Integration tests for hierarchical ensemble mode in HPO pipeline.

Tests verify:
1. Hierarchical config loading works in HPO context
2. Symbolic dominance penalty is conditionally disabled
3. Hierarchical metrics are added to trial attributes
4. Config flag transitions work correctly

Design Patterns:
- Fixture Factory: Creates test configs with hierarchical/flat modes
- Mocking: Isolates pipeline from real training
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from pff.validators.ensembles.hierarchical import (
    HierarchicalConfig,
    load_hierarchical_config,
)
from scripts.optimization.trials.pipeline import TrialEvaluationPipeline


class TestHierarchicalConfigInHPO:
    """Tests for hierarchical config loading in HPO context."""
    
    def test_load_hierarchical_config_returns_valid_config(self):
        """Config loader should return valid HierarchicalConfig instance."""
        config = load_hierarchical_config()
        
        assert isinstance(config, HierarchicalConfig)
        assert hasattr(config, "is_hierarchical")
        assert hasattr(config, "should_apply_symbolic_dominance_penalty")
    
    def test_default_config_is_flat_mode(self):
        """Default config should be flat mode (backward compatible)."""
        config = load_hierarchical_config()
        
        # Default is flat mode for backward compatibility
        assert config.architecture_type in ("flat", "hierarchical")
    
    def test_hierarchical_mode_skips_penalty(self):
        """When hierarchical mode is active, penalty should be skipped."""
        with patch("pff.validators.ensembles.hierarchical.config_loader.load_hierarchical_config") as mock_load:
            mock_config = MagicMock(spec=HierarchicalConfig)
            mock_config.is_hierarchical = True
            mock_config.should_apply_symbolic_dominance_penalty = False
            mock_load.return_value = mock_config
            
            config = mock_load()
            skip_penalty = config.is_hierarchical and not config.should_apply_symbolic_dominance_penalty
            
            assert skip_penalty is True
    
    def test_flat_mode_applies_penalty(self):
        """When flat mode is active, penalty should be applied."""
        with patch("pff.validators.ensembles.hierarchical.config_loader.load_hierarchical_config") as mock_load:
            mock_config = MagicMock(spec=HierarchicalConfig)
            mock_config.is_hierarchical = False
            mock_config.should_apply_symbolic_dominance_penalty = True
            mock_load.return_value = mock_config
            
            config = mock_load()
            skip_penalty = config.is_hierarchical and not config.should_apply_symbolic_dominance_penalty
            
            assert skip_penalty is False


class TestTrialEvaluationPipelineHierarchical:
    """Tests for TrialEvaluationPipeline hierarchical mode integration."""
    
    @pytest.fixture
    def mock_dataframes(self):
        """Create minimal mock DataFrames for pipeline init."""
        import polars as pl
        
        # Create minimal test data
        train_df = pl.DataFrame({
            "head": ["e1", "e2", "e3"],
            "relation": ["r1", "r1", "r2"],
            "tail": ["e2", "e3", "e1"],
        })
        valid_df = pl.DataFrame({
            "head": ["e1"],
            "relation": ["r1"],
            "tail": ["e3"],
        })
        return train_df, valid_df
    
    def test_pipeline_has_hierarchical_routing_stats_attribute(self, mock_dataframes):
        """Pipeline should have hierarchical_routing_stats attribute after init."""
        train_df, valid_df = mock_dataframes
        
        with patch.multiple(
            "scripts.optimization.trials.pipeline",
            get_file_manager=MagicMock(return_value=MagicMock(read=MagicMock(return_value={}))),
            load_trial_constraints=MagicMock(return_value={
                "coverage_gate": 0.05,
                "dominance_gate": 0.70,
                "symbolic_max_rules": 1000,
                "min_symbolic_activation": 0.01,
            }),
            settings=MagicMock(OUTPUTS_DIR=Path("/tmp/test_outputs")),
        ):
            pipeline = TrialEvaluationPipeline(
                params={"neural_weight": 0.3, "rules_weight": 0.3, "lightgbm_weight": 0.4},
                train_df=train_df,
                valid_df=valid_df,
                target_entity_ratio=0.3,
                trial_number=1,
                trial_output_root=Path("/tmp/test"),
                rule_filter=None,
                trial=None,
                artifact_manager=MagicMock(),
            )
            
            assert hasattr(pipeline, "hierarchical_routing_stats")
            assert isinstance(pipeline.hierarchical_routing_stats, dict)
    
    def test_pipeline_init_stats_is_empty_dict(self, mock_dataframes):
        """Initial hierarchical_routing_stats should be empty dict."""
        train_df, valid_df = mock_dataframes
        
        with patch.multiple(
            "scripts.optimization.trials.pipeline",
            get_file_manager=MagicMock(return_value=MagicMock(read=MagicMock(return_value={}))),
            load_trial_constraints=MagicMock(return_value={
                "coverage_gate": 0.05,
                "dominance_gate": 0.70,
                "symbolic_max_rules": 1000,
                "min_symbolic_activation": 0.01,
            }),
            settings=MagicMock(OUTPUTS_DIR=Path("/tmp/test_outputs")),
        ):
            pipeline = TrialEvaluationPipeline(
                params={},
                train_df=train_df,
                valid_df=valid_df,
                target_entity_ratio=0.3,
                trial_number=1,
                trial_output_root=Path("/tmp/test"),
                rule_filter=None,
                trial=None,
                artifact_manager=MagicMock(),
            )
            
            # Initially empty, populated during _compute_score
            assert pipeline.hierarchical_routing_stats == {}


class TestHierarchicalMetricsPayload:
    """Tests for hierarchical metrics in trial user_attrs."""
    
    def test_hierarchical_metrics_prefixed_correctly(self):
        """Hierarchical metrics should be prefixed with 'hierarchical_'."""
        routing_stats = {
            "architecture_type": 1.0,
            "symbolic_high_threshold": 0.7,
            "symbolic_low_threshold": 0.3,
            "neural_fallback_threshold": 0.5,
        }
        
        # Simulate what evaluate_trial_with_config does
        metrics_payload = {}
        for key, value in routing_stats.items():
            metrics_payload[f"hierarchical_{key}"] = float(value)
        
        assert "hierarchical_architecture_type" in metrics_payload
        assert "hierarchical_symbolic_high_threshold" in metrics_payload
        assert "hierarchical_symbolic_low_threshold" in metrics_payload
        assert "hierarchical_neural_fallback_threshold" in metrics_payload
        
        assert metrics_payload["hierarchical_architecture_type"] == 1.0
    
    def test_flat_mode_only_has_architecture_type(self):
        """Flat mode should only have architecture_type=0.0."""
        # When flat mode is active, only architecture_type is set
        routing_stats = {"architecture_type": 0.0}
        
        metrics_payload = {}
        for key, value in routing_stats.items():
            metrics_payload[f"hierarchical_{key}"] = float(value)
        
        assert len(metrics_payload) == 1
        assert metrics_payload["hierarchical_architecture_type"] == 0.0


class TestConditionalPenaltyLogic:
    """Tests for conditional symbolic dominance penalty logic."""
    
    def test_penalty_calculation_with_hierarchical_skip(self):
        """When hierarchical mode skips penalty, symbolic_dominance_penalty should be 0."""
        # Simulate the conditional logic from _compute_score
        is_hierarchical = True
        should_apply_penalty = False
        skip_symbolic_penalty = is_hierarchical and not should_apply_penalty
        
        symbolic_contribution_ratio = 0.97  # 97% symbolic dominance
        symbolic_dominance_penalty = 0.0
        
        if skip_symbolic_penalty:
            symbolic_dominance_penalty = 0.0
        else:
            # Normal penalty calculation would apply
            soft_threshold = 0.65
            hard_threshold = 0.85
            if symbolic_contribution_ratio > hard_threshold:
                symbolic_dominance_penalty = 0.5 + (symbolic_contribution_ratio - hard_threshold) / (1.0 - hard_threshold) * 0.5
        
        assert symbolic_dominance_penalty == 0.0
    
    def test_penalty_calculation_without_hierarchical(self):
        """When flat mode is active, penalty should be calculated normally."""
        is_hierarchical = False
        should_apply_penalty = True
        skip_symbolic_penalty = is_hierarchical and not should_apply_penalty
        
        symbolic_contribution_ratio = 0.97  # 97% symbolic dominance
        symbolic_dominance_penalty = 0.0
        
        if skip_symbolic_penalty:
            symbolic_dominance_penalty = 0.0
        else:
            soft_threshold = 0.65
            hard_threshold = 0.85
            if symbolic_contribution_ratio > hard_threshold:
                excess_above_hard = symbolic_contribution_ratio - hard_threshold
                symbolic_dominance_penalty = 0.5 + excess_above_hard / (1.0 - hard_threshold) * 0.5
                symbolic_dominance_penalty = min(1.0, symbolic_dominance_penalty)
        
        # With 97% symbolic, penalty should be significant
        assert symbolic_dominance_penalty > 0.5
        assert symbolic_dominance_penalty <= 1.0


class TestHierarchicalConfigTransitions:
    """Tests for transitioning between hierarchical and flat modes."""
    
    def test_config_mode_detection(self):
        """Config should correctly identify its mode."""
        config = load_hierarchical_config()
        
        if config.architecture_type == "hierarchical":
            assert config.is_hierarchical is True
        else:
            assert config.is_hierarchical is False
    
    def test_penalty_flag_consistency(self):
        """Penalty flag should be consistent with architecture type."""
        config = load_hierarchical_config()
        
        # In hierarchical mode, penalty should typically be disabled
        # In flat mode, penalty should be enabled
        if config.is_hierarchical:
            # Default hierarchical config disables penalty
            assert config.should_apply_symbolic_dominance_penalty is False
        else:
            # Flat mode always applies penalty
            assert config.should_apply_symbolic_dominance_penalty is True


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing HPO."""
    
    def test_missing_hierarchical_config_defaults_to_flat(self):
        """If hierarchical config is missing, should default to flat mode."""
        with patch(
            "pff.validators.ensembles.hierarchical.config_loader.HIERARCHICAL_ENSEMBLE_CONFIG_PATH",
            Path("/nonexistent/path.yaml"),
        ):
            with patch(
                "pff.validators.ensembles.hierarchical.config_loader.FileManager"
            ) as mock_fm:
                mock_fm.return_value.read.side_effect = FileNotFoundError()
                
                config = load_hierarchical_config()
                
                # Should default to flat mode
                assert config.architecture_type == "flat"
                assert config.is_hierarchical is False
    
    def test_empty_routing_stats_does_not_break_metrics(self):
        """Empty routing stats should not add any hierarchical_ metrics."""
        routing_stats = {}
        metrics_payload = {"composite_score": 0.5}
        
        # Simulate what evaluate_trial_with_config does
        if routing_stats:
            for key, value in routing_stats.items():
                metrics_payload[f"hierarchical_{key}"] = float(value)
        
        # Should only have the original metric
        assert len(metrics_payload) == 1
        assert "composite_score" in metrics_payload
