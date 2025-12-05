"""Tests for LightGBM regularization parameters from config.

P1.1: Verify that LightGBM trainer reads and uses regularization params
from config/models/rotate.yaml (num_leaves, reg_alpha, reg_lambda, min_data_in_leaf, max_bin).

This test mocks lightgbm.train to verify the params passed match the config values.

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pff.config import ROTATE_CONFIG_PATH


class TestLightGBMRegularizationParams:
    """P1.1: Test that LightGBM trainer uses regularization params from config."""

    @pytest.fixture
    def mock_rotate_manager(self):
        """Create a mock RotatE manager with embeddings."""
        manager = Mock()
        model = Mock()
        model.num_entities = 100
        model.num_relations = 10
        model.embedding_dim = 64
        
        # Mock complex embeddings
        entity_real = np.random.randn(100, 32).astype(np.float32)
        entity_imag = np.random.randn(100, 32).astype(np.float32)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = np.random.randn(10, 32).astype(np.float32)
        
        manager.model = model
        manager.entity_to_idx = {f"e{i}": i for i in range(100)}
        manager.idx_to_entity = {i: f"e{i}" for i in range(100)}
        manager.relation_to_idx = {f"r{i}": i for i in range(10)}
        manager.idx_to_relation = {i: f"r{i}" for i in range(10)}
        manager.node_embeddings = None
        
        return manager

    def test_config_has_regularization_params(self):
        """Verify rotate.yaml contains the expected regularization params."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        
        lgb_params = config.get("lightgbm", {}).get("params", {})
        
        # P1.1: Required regularization params
        assert "num_leaves" in lgb_params, "num_leaves should be in config"
        assert "reg_alpha" in lgb_params, "reg_alpha (L1) should be in config"
        assert "reg_lambda" in lgb_params, "reg_lambda (L2) should be in config"
        assert "min_data_in_leaf" in lgb_params, "min_data_in_leaf should be in config"
        assert "max_bin" in lgb_params, "max_bin should be in config"

    def test_regularization_values_are_strong(self):
        """Verify regularization values are set for strong regularization."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        lgb_params = config.get("lightgbm", {}).get("params", {})
        
        # P1.1: Strong regularization requirements
        # num_leaves: 31 is the production value - provides good balance
        # between model complexity and regularization
        assert lgb_params.get("num_leaves", 31) <= 63, "num_leaves should be <= 63 for regularization"
        assert lgb_params.get("reg_alpha", 0) >= 0.1, "reg_alpha should be >= 0.1"
        assert lgb_params.get("reg_lambda", 0) >= 1.0, "reg_lambda should be >= 1.0"
        assert lgb_params.get("min_data_in_leaf", 5) >= 20, "min_data_in_leaf should be >= 20"
        assert lgb_params.get("max_bin", 255) <= 255, "max_bin should be <= 255"

    @patch("pff.validators.rotate.lightgbm_trainer.lgb")
    def test_trainer_passes_config_params_to_lgb_train(self, mock_lgb, mock_rotate_manager):
        """Verify that _train_lightgbm passes config params to lgb.train."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        from pff.utils.file_manager import FileManager
        
        # Load expected config values
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        lgb_config = config.get("lightgbm", {})
        expected_params = lgb_config.get("params", {})
        
        # Setup mock LightGBM
        mock_dataset = Mock()
        mock_lgb.Dataset.return_value = mock_dataset
        mock_booster = Mock()
        mock_lgb.train.return_value = mock_booster
        
        # Create trainer
        trainer = RotatELightGBMTrainer(mock_rotate_manager)
        
        # Create small test data
        X_train = np.random.randn(50, 64).astype(np.float32)
        y_train = np.random.randint(0, 2, 50).astype(np.int32)
        X_val = np.random.randn(20, 64).astype(np.float32)
        y_val = np.random.randint(0, 2, 20).astype(np.int32)
        
        # Call _train_lightgbm
        trainer._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Verify lgb.train was called
        assert mock_lgb.train.called, "lgb.train should have been called"
        
        # Get the params passed to lgb.train
        call_args = mock_lgb.train.call_args
        actual_params = call_args[1].get("params", call_args[0][0] if call_args[0] else {})
        
        # Verify regularization params are present and match config
        assert actual_params.get("num_leaves") == expected_params.get("num_leaves"), \
            f"num_leaves mismatch: {actual_params.get('num_leaves')} != {expected_params.get('num_leaves')}"
        assert actual_params.get("reg_alpha") == expected_params.get("reg_alpha"), \
            f"reg_alpha mismatch: {actual_params.get('reg_alpha')} != {expected_params.get('reg_alpha')}"
        assert actual_params.get("reg_lambda") == expected_params.get("reg_lambda"), \
            f"reg_lambda mismatch: {actual_params.get('reg_lambda')} != {expected_params.get('reg_lambda')}"
        assert actual_params.get("min_data_in_leaf") == expected_params.get("min_data_in_leaf"), \
            f"min_data_in_leaf mismatch"
        assert actual_params.get("max_bin") == expected_params.get("max_bin"), \
            f"max_bin mismatch"

    @patch("pff.validators.rotate.lightgbm_trainer.lgb")
    def test_trainer_uses_cpu_fallback_gracefully(self, mock_lgb, mock_rotate_manager):
        """Verify trainer falls back to CPU when CUDA unavailable."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        # Setup mock to simulate CUDA failure
        mock_dataset = Mock()
        mock_lgb.Dataset.return_value = mock_dataset
        mock_booster = Mock()
        
        # First call (CUDA test) fails, second call (actual training) succeeds
        mock_lgb.train.side_effect = [
            Exception("CUDA not available"),  # CUDA test fails
            mock_booster,  # Actual training succeeds on CPU
        ]
        
        trainer = RotatELightGBMTrainer(mock_rotate_manager)
        
        X_train = np.random.randn(50, 64).astype(np.float32)
        y_train = np.random.randint(0, 2, 50).astype(np.int32)
        X_val = np.random.randn(20, 64).astype(np.float32)
        y_val = np.random.randint(0, 2, 20).astype(np.int32)
        
        # Should not raise, should fall back to CPU
        with patch("torch.cuda.is_available", return_value=True):
            result = trainer._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Verify train was called twice (test + actual)
        assert mock_lgb.train.call_count == 2


class TestLightGBMConfigIntegration:
    """Integration tests for LightGBM config loading."""

    def test_lightgbm_section_complete(self):
        """Verify lightgbm config section has all required fields."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        
        lgb_config = config.get("lightgbm", {})
        
        # Required sections
        assert "params" in lgb_config, "lightgbm.params section required"
        assert "training" in lgb_config, "lightgbm.training section required"
        
        # Required training params
        training = lgb_config.get("training", {})
        assert "num_boost_round" in training, "num_boost_round required"
        assert "early_stopping_rounds" in training, "early_stopping_rounds required"

    def test_params_are_numeric(self):
        """Verify all numeric params have correct types."""
        from pff.utils.file_manager import FileManager
        
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        lgb_params = config.get("lightgbm", {}).get("params", {})
        
        numeric_params = [
            "num_leaves", "learning_rate", "feature_fraction", 
            "bagging_fraction", "reg_alpha", "reg_lambda",
            "min_data_in_leaf", "max_bin"
        ]
        
        for param in numeric_params:
            if param in lgb_params:
                value = lgb_params[param]
                assert isinstance(value, (int, float)), \
                    f"{param} should be numeric, got {type(value)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
