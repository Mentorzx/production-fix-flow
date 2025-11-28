"""Tests for RotatELightGBMTrainer parameter passing from config.

Task 1: Verify that RotatELightGBMTrainer._train_lightgbm reads lightgbm.params
from config/models/rotate.yaml and passes them to lightgbm.train or lgb.train.

This test patches lightgbm.train, captures the parameters received, and validates
that num_leaves, reg_alpha, reg_lambda, min_data_in_leaf, max_bin come from
config/models/rotate.yaml or DEFAULT_LGB_PARAMS when absent.

Author: PFF Team
Date: 2025-11-27
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch, call

import numpy as np
import pytest
import torch

from pff.config import ROTATE_CONFIG_PATH


class TestRotatELightGBMTrainerParamPassing:
    """Test that RotatELightGBMTrainer passes config params to lgb.train."""

    @pytest.fixture
    def mock_rotate_manager(self):
        """Create a minimal mock RotatE manager."""
        manager = Mock()
        model = Mock()
        model.num_entities = 50
        model.num_relations = 5
        model.embedding_dim = 32

        # Mock complex embeddings (real + imaginary parts)
        entity_real = torch.randn(50, 16)
        entity_imag = torch.randn(50, 16)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(5, 16)

        manager.model = model
        manager.config = {"model": {"embedding_dim": 32}}
        manager.entity_to_idx = {f"e{i}": i for i in range(50)}
        manager.idx_to_entity = {i: f"e{i}" for i in range(50)}
        manager.relation_to_idx = {f"r{i}": i for i in range(5)}
        manager.idx_to_relation = {i: f"r{i}" for i in range(5)}
        manager.node_embeddings = None

        return manager

    @pytest.fixture
    def small_test_data(self):
        """Create small synthetic train/val data."""
        np.random.seed(42)
        X_train = np.random.randn(40, 32).astype(np.float32)
        y_train = np.random.randint(0, 2, 40).astype(np.int32)
        X_val = np.random.randn(10, 32).astype(np.float32)
        y_val = np.random.randint(0, 2, 10).astype(np.int32)
        return X_train, y_train, X_val, y_val

    def test_lgb_train_receives_config_params(self, mock_rotate_manager, small_test_data):
        """Verify lgb.train receives params from rotate.yaml config."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        from pff.utils.file_manager import FileManager

        # Load expected config values
        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        lgb_config = config.get("lightgbm", {})
        expected_params = lgb_config.get("params", {})

        X_train, y_train, X_val, y_val = small_test_data

        with patch("pff.validators.rotate.lightgbm_trainer.lgb") as mock_lgb:
            # Setup mock
            mock_dataset = Mock()
            mock_lgb.Dataset.return_value = mock_dataset
            mock_booster = Mock()
            mock_lgb.train.return_value = mock_booster

            trainer = RotatELightGBMTrainer(mock_rotate_manager)
            trainer._train_lightgbm(X_train, y_train, X_val, y_val)

            # Verify lgb.train was called
            assert mock_lgb.train.called, "lgb.train should have been called"

            # Get the actual params passed to lgb.train
            call_args = mock_lgb.train.call_args
            # lgb.train(params, train_data, ...) - first positional arg is params
            actual_params = call_args[0][0] if call_args[0] else call_args[1].get("params", {})

            # Verify regularization params match config
            assert actual_params.get("num_leaves") == expected_params.get("num_leaves"), \
                f"num_leaves: expected {expected_params.get('num_leaves')}, got {actual_params.get('num_leaves')}"

            assert actual_params.get("reg_alpha") == expected_params.get("reg_alpha"), \
                f"reg_alpha: expected {expected_params.get('reg_alpha')}, got {actual_params.get('reg_alpha')}"

            assert actual_params.get("reg_lambda") == expected_params.get("reg_lambda"), \
                f"reg_lambda: expected {expected_params.get('reg_lambda')}, got {actual_params.get('reg_lambda')}"

            assert actual_params.get("min_data_in_leaf") == expected_params.get("min_data_in_leaf"), \
                f"min_data_in_leaf: expected {expected_params.get('min_data_in_leaf')}, got {actual_params.get('min_data_in_leaf')}"

            assert actual_params.get("max_bin") == expected_params.get("max_bin"), \
                f"max_bin: expected {expected_params.get('max_bin')}, got {actual_params.get('max_bin')}"

    def test_lgb_train_receives_all_core_params(self, mock_rotate_manager, small_test_data):
        """Verify all core LightGBM params are passed."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        expected_params = config.get("lightgbm", {}).get("params", {})

        X_train, y_train, X_val, y_val = small_test_data

        with patch("pff.validators.rotate.lightgbm_trainer.lgb") as mock_lgb:
            mock_lgb.Dataset.return_value = Mock()
            mock_lgb.train.return_value = Mock()

            trainer = RotatELightGBMTrainer(mock_rotate_manager)
            trainer._train_lightgbm(X_train, y_train, X_val, y_val)

            actual_params = mock_lgb.train.call_args[0][0]

            # Core params that should come from config
            core_params = ["objective", "metric", "boosting_type", "learning_rate", 
                          "feature_fraction", "bagging_fraction"]

            for param in core_params:
                if param in expected_params:
                    assert actual_params.get(param) == expected_params.get(param), \
                        f"{param}: expected {expected_params.get(param)}, got {actual_params.get(param)}"

    def test_uses_default_params_when_config_missing(self, mock_rotate_manager, small_test_data):
        """Verify defaults are used when config section is missing."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer

        X_train, y_train, X_val, y_val = small_test_data

        # Mock FileManager to return empty lightgbm config
        with patch("pff.validators.rotate.lightgbm_trainer.lgb") as mock_lgb, \
             patch("pff.validators.rotate.lightgbm_trainer.FileManager") as mock_fm_class:

            # FileManager instance returns empty lightgbm config
            mock_fm_instance = Mock()
            mock_fm_instance.read.return_value = {"lightgbm": {}}
            mock_fm_class.return_value = mock_fm_instance

            mock_lgb.Dataset.return_value = Mock()
            mock_lgb.train.return_value = Mock()

            trainer = RotatELightGBMTrainer(mock_rotate_manager)
            trainer._train_lightgbm(X_train, y_train, X_val, y_val)

            actual_params = mock_lgb.train.call_args[0][0]

            # Should have defaults
            assert actual_params.get("objective") == "binary"
            assert actual_params.get("num_leaves") == 31  # Default
            assert actual_params.get("learning_rate") == 0.05  # Default

    def test_training_config_num_boost_round(self, mock_rotate_manager, small_test_data):
        """Verify num_boost_round comes from config."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        expected_rounds = config.get("lightgbm", {}).get("training", {}).get("num_boost_round", 100)

        X_train, y_train, X_val, y_val = small_test_data

        with patch("pff.validators.rotate.lightgbm_trainer.lgb") as mock_lgb:
            mock_lgb.Dataset.return_value = Mock()
            mock_lgb.train.return_value = Mock()

            trainer = RotatELightGBMTrainer(mock_rotate_manager)
            trainer._train_lightgbm(X_train, y_train, X_val, y_val)

            # num_boost_round is passed as a keyword argument
            call_kwargs = mock_lgb.train.call_args[1]
            actual_rounds = call_kwargs.get("num_boost_round", 
                                            mock_lgb.train.call_args[0][2] if len(mock_lgb.train.call_args[0]) > 2 else None)

            assert actual_rounds == expected_rounds, \
                f"num_boost_round: expected {expected_rounds}, got {actual_rounds}"


class TestRotatELightGBMTrainerConfigIntegration:
    """Integration tests for config loading in RotatELightGBMTrainer."""

    def test_config_lightgbm_section_exists(self):
        """Verify rotate.yaml has lightgbm section."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)

        assert "lightgbm" in config, "lightgbm section should exist in rotate.yaml"
        assert "params" in config["lightgbm"], "lightgbm.params should exist"
        assert "training" in config["lightgbm"], "lightgbm.training should exist"

    def test_config_has_p1_regularization_params(self):
        """Verify P1.1 regularization params are in config."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        params = config.get("lightgbm", {}).get("params", {})

        # P1.1 required params
        p1_params = ["num_leaves", "reg_alpha", "reg_lambda", "min_data_in_leaf", "max_bin"]

        for param in p1_params:
            assert param in params, f"P1.1 param '{param}' should be in config"

    def test_regularization_values_appropriate(self):
        """Verify regularization values are set appropriately."""
        from pff.utils.file_manager import FileManager

        fm = FileManager()
        config = fm.read(ROTATE_CONFIG_PATH)
        params = config.get("lightgbm", {}).get("params", {})

        # P1.1: Strong regularization constraints
        assert params.get("num_leaves", 31) <= 31, "num_leaves should be reasonable"
        assert params.get("reg_alpha", 0) >= 0, "reg_alpha should be non-negative"
        assert params.get("reg_lambda", 0) >= 0, "reg_lambda should be non-negative"
        assert params.get("min_data_in_leaf", 5) >= 1, "min_data_in_leaf should be >= 1"
        assert 0 < params.get("max_bin", 255) <= 255, "max_bin should be in (0, 255]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
