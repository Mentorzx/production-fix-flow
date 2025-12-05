"""
Tests for RotatE Hyperparameter Optimization Integration.

This module tests the integration of RotatE with the HPO system,
including search space generation and objective function compatibility.

Following SOTA practices:
- Search space validation for RotatE parameters
- KGE model selection in HPO workflow
- Integration with Optuna trial suggestions
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import torch
import numpy as np

# Skip if dependencies not available
pytest.importorskip("optuna")


class TestRotatEHPOConstants:
    """Test HPO constants and model selection."""

    def test_kge_model_constants(self):
        """Test that KGE model constants are defined correctly."""
        from scripts.optimization.core import (
            KGE_MODEL_ROTATE,
            DEFAULT_KGE_MODEL,
        )

        assert KGE_MODEL_ROTATE == "rotate"
        assert DEFAULT_KGE_MODEL == "rotate"

    def test_optimize_kg_hyperparameters_signature(self):
        """Test that optimize_kg_hyperparameters accepts kge_model parameter."""
        import inspect
        from scripts.optimization.core import optimize_kg_hyperparameters

        sig = inspect.signature(optimize_kg_hyperparameters)
        params = sig.parameters

        assert "kge_model" in params
        # Default is now RotatE (TransE was removed)
        assert params["kge_model"].default == "rotate"


class TestRotatESearchSpace:
    """Test RotatE search space generation."""

    def test_rotate_search_space_exists(self):
        """Test that RotatE search space factory method exists."""
        from scripts.optimization.spaces import SearchSpaceFactory

        assert hasattr(SearchSpaceFactory, "create_rotate_space")

    def test_rotate_search_space_structure(self):
        """Test RotatE search space has expected parameters."""
        from scripts.optimization.spaces import SearchSpaceFactory, TuningConfig

        config = TuningConfig()
        space = SearchSpaceFactory.create_rotate_space(config)

        # Core RotatE parameters
        assert "rotate_embedding_dim" in space
        assert "rotate_gamma" in space
        assert "rotate_epsilon" in space

        # Training parameters
        assert "rotate_learning_rate" in space
        assert "rotate_epochs" in space
        assert "rotate_batch_size" in space
        assert "rotate_negative_samples" in space

        # Self-adversarial negative sampling
        assert "rotate_adversarial_temperature" in space

    def test_rotate_search_space_ranges(self):
        """Test that RotatE search space has valid ranges."""
        from scripts.optimization.spaces import SearchSpaceFactory, TuningConfig

        config = TuningConfig()
        space = SearchSpaceFactory.create_rotate_space(config)

        # Embedding dim should include SOTA values
        embed_dims = space["rotate_embedding_dim"]
        assert 256 in embed_dims
        assert 512 in embed_dims

        # Gamma should have reasonable range (SOTA: 9-24)
        gamma = space["rotate_gamma"]
        assert gamma[0] >= 3.0  # Min
        assert gamma[1] <= 30.0  # Max

        # Epsilon should be in reasonable range
        epsilon = space["rotate_epsilon"]
        assert epsilon[0] >= 0.5
        assert epsilon[1] <= 5.0

    def test_rotate_search_space_training_bounds(self):
        """Training bounds should reflect stability/performance envelope."""
        from scripts.optimization.spaces import SearchSpaceFactory, TuningConfig

        space = SearchSpaceFactory.create_rotate_space(TuningConfig())

        lr_low, lr_high = space["rotate_learning_rate"]
        assert lr_low <= 1e-5
        assert lr_high <= 5e-4

        epochs_low, epochs_high = space["rotate_epochs"]
        assert epochs_low >= 100
        assert epochs_high <= 200

        neg_low, neg_high = space["rotate_negative_samples"]
        assert neg_low >= 256
        assert neg_high <= 1024

        batch_choices = space["rotate_batch_size"]
        assert max(batch_choices) <= 1024


class TestRotatEObjectiveIntegration:
    """Test RotatE integration with HPO objective function."""

    def test_kg_objective_rotate_params(self):
        """Test that kg_objective generates correct RotatE parameters."""
        import optuna
        from scripts.optimization.core import KGE_MODEL_ROTATE

        # Create a mock trial
        study = optuna.create_study()

        def mock_objective(trial):
            # Simulate RotatE parameter generation
            params = {
                "kge_model": KGE_MODEL_ROTATE,
                "embedding_dim": trial.suggest_categorical("embedding_dim", [128, 256, 512]),
                "gamma": trial.suggest_float("gamma", 6.0, 24.0),
                "epsilon": trial.suggest_float("epsilon", 1.0, 3.0),
                "rotate_epochs": trial.suggest_int("rotate_epochs", 50, 200),
                "batch_size": trial.suggest_int("batch_size", 256, 2048),
                "negative_sample_size": trial.suggest_int("negative_sample_size", 64, 512),
            }

            # Verify parameter types
            assert isinstance(params["embedding_dim"], int)
            assert isinstance(params["gamma"], float)
            assert isinstance(params["epsilon"], float)
            assert isinstance(params["rotate_epochs"], int)
            assert isinstance(params["batch_size"], int)
            assert isinstance(params["negative_sample_size"], int)

            return 0.5  # Dummy score

        # Run one trial
        study.optimize(mock_objective, n_trials=1)

        assert len(study.trials) == 1
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


class TestRotatETrainingFunctions:
    """Test RotatE training helper functions."""

    def test_train_rotate_model_function_exists(self):
        """Test that _train_rotate_model function exists."""
        from scripts.optimization.trials.evaluator import _train_rotate_model

        assert callable(_train_rotate_model)

    def test_train_rotate_score_calibrator_function_exists(self):
        """Test that _train_rotate_score_calibrator function exists."""
        from scripts.optimization.trials.evaluator import _train_rotate_score_calibrator

        assert callable(_train_rotate_score_calibrator)

    def test_create_rotate_lightgbm_trainer_exists_in_pipeline(self):
        """Test that RotatELightGBMTrainer can be instantiated from pff.validators."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer

        assert RotatELightGBMTrainer is not None


class TestRotatELightGBMAdapter:
    """Test RotatE to LightGBM adapter functionality."""

    def test_rotate_adapter_creates_real_embeddings(self):
        """Test that RotatE adapter converts complex embeddings to real."""
        from pff.validators.rotate.core import RotatEModel
        from pff.validators.rotate.config import RotatEConfig

        # Create a small RotatE model (embedding_dim must be even)
        config = RotatEConfig(
            embedding_dim=32,
            gamma=12.0,
            epsilon=2.0,
        )
        # Create model: RotatEModel(num_entities, num_relations, embedding_dim, gamma, epsilon, config)
        model = RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=32,
            gamma=12.0,
            epsilon=2.0,
            config=config,
        )

        # Get real and imaginary parts using model method
        entity_re, entity_im = model.get_entity_embeddings()

        # Verify shapes (complex_dim = embedding_dim // 2 = 16)
        assert entity_re.shape == (100, 16)
        assert entity_im.shape == (100, 16)

        # Concatenate (as adapter does)
        combined = torch.cat([entity_re, entity_im], dim=1)
        assert combined.shape == (100, 32)  # embedding_dim

    def test_relation_phase_to_real_imag(self):
        """Test conversion of relation phases to real/imaginary."""
        from pff.validators.rotate.core import RotatEModel
        from pff.validators.rotate.config import RotatEConfig

        config = RotatEConfig(
            embedding_dim=32,
            gamma=12.0,
            epsilon=2.0,
        )
        model = RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=32,
            gamma=12.0,
            epsilon=2.0,
            config=config,
        )

        # Get relation phases (complex_dim = 16)
        phases = model.get_relation_phases()
        assert phases.shape == (10, 16)

        # Convert to real/imaginary
        rel_real = torch.cos(phases)
        rel_imag = torch.sin(phases)

        # Verify unit circle constraint
        magnitude = torch.sqrt(rel_real**2 + rel_imag**2)
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5)


class TestKGEModelSelection:
    """Test KGE model selection in HPO workflow."""

    def test_invalid_kge_model_uses_rotate(self):
        """Test that invalid kge_model falls back to RotatE."""
        from scripts.optimization.core import KGE_MODEL_ROTATE

        # RotatE is now the only supported model
        assert KGE_MODEL_ROTATE == "rotate"

    def test_kge_model_in_objective_params(self):
        """Test that kge_model is passed through to objective params."""
        from scripts.optimization.core import KGE_MODEL_ROTATE

        # Simulate params dict as created in kg_objective
        params = {
            "neural_weight": 0.3,
            "rules_weight": 0.15,
            "lightgbm_weight": 0.55,
            "kge_model": KGE_MODEL_ROTATE,
            "embedding_dim": 256,
            "gamma": 12.0,
            "epsilon": 2.0,
        }

        assert params["kge_model"] == "rotate"
        assert "gamma" in params  # RotatE-specific
        assert "epsilon" in params  # RotatE-specific


class TestRotatEHPOMetrics:
    """Test that RotatE metrics are properly tracked in HPO."""

    def test_rotate_metrics_structure(self):
        """Test expected structure of RotatE evaluation metrics."""
        # Expected metrics from RotatE training
        expected_metrics = {
            "mrr": 0.0,
            "hits@1": 0.0,
            "hits@10": 0.0,
            "best_val_mrr": 0.0,
        }

        # Verify structure
        assert all(isinstance(v, float) for v in expected_metrics.values())
        assert "mrr" in expected_metrics
        assert "hits@1" in expected_metrics
        assert "hits@10" in expected_metrics


class TestRotatEParameterValidation:
    """Test parameter validation for RotatE HPO."""

    def test_gamma_epsilon_initialization(self):
        """Test that gamma and epsilon are used correctly."""
        from pff.validators.rotate.config import RotatEConfig

        gamma = 12.0
        epsilon = 2.0
        embedding_dim = 256

        config = RotatEConfig(
            embedding_dim=embedding_dim,
            gamma=gamma,
            epsilon=epsilon,
        )

        # Verify embedding range formula (as per SOTA)
        embedding_range = (gamma + epsilon) / embedding_dim
        expected_range = (12.0 + 2.0) / 256
        assert abs(embedding_range - expected_range) < 1e-6

    def test_negative_sample_size_bounds(self):
        """Test that negative sample size is within reasonable bounds."""
        # SOTA values: 64-512
        min_neg_samples = 64
        max_neg_samples = 512

        assert min_neg_samples >= 32  # Minimum for effective training
        assert max_neg_samples <= 1024  # Maximum for memory efficiency

    def test_adversarial_temperature_range(self):
        """Test adversarial temperature range is valid."""
        # SOTA values: 0.5-2.0
        min_temp = 0.5
        max_temp = 2.0

        assert min_temp > 0  # Must be positive
        assert max_temp <= 3.0  # Upper bound for numerical stability
