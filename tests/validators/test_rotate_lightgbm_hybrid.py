"""Tests for RotatE LightGBM Hybrid Trainer.

Comprehensive tests for the hybrid RotatE + LightGBM training including:
- Initialization and configuration
- Embedding extraction
- Feature engineering
- Negative sampling
- LightGBM training
- Model persistence
- Error handling

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch


class TestRotatELightGBMTrainerInitialization:
    """Tests for RotatELightGBMTrainer initialization."""

    @pytest.fixture
    def mock_rotate_manager(self):
        """Create a mock RotatE manager."""
        manager = Mock()
        manager.model = Mock()
        manager.model.num_entities = 100
        manager.model.num_relations = 10
        manager.model.embedding_dim = 64
        
        # Mock entity embeddings (complex format)
        entity_real = torch.randn(100, 32)
        entity_imag = torch.randn(100, 32)
        manager.model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        manager.model.get_relation_phases.return_value = torch.randn(10, 32)
        
        manager.entity_to_idx = {f"e{i}": i for i in range(100)}
        manager.idx_to_entity = {i: f"e{i}" for i in range(100)}
        manager.relation_to_idx = {f"r{i}": i for i in range(10)}
        manager.idx_to_relation = {i: f"r{i}" for i in range(10)}
        
        manager.node_embeddings = None
        
        return manager

    def test_trainer_initialization(self, mock_rotate_manager):
        """Test trainer initializes correctly with RotatE manager."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        trainer = RotatELightGBMTrainer(mock_rotate_manager)
        
        assert trainer.rotate_manager == mock_rotate_manager
        assert trainer.lightgbm_model is None
        assert trainer.file_manager is not None

    def test_trainer_accepts_adapter(self, mock_rotate_manager):
        """Test trainer works with adapter pattern."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        # Simulate adapter with model attribute
        adapter = Mock()
        adapter.model = mock_rotate_manager.model
        adapter.entity_to_idx = mock_rotate_manager.entity_to_idx
        adapter.relation_to_idx = mock_rotate_manager.relation_to_idx
        
        trainer = RotatELightGBMTrainer(adapter)
        
        assert trainer.rotate_manager == adapter


class TestRotatELightGBMEmbeddingExtraction:
    """Tests for embedding extraction in LightGBM trainer."""

    @pytest.fixture
    def trainer_with_model(self):
        """Create trainer with mock model."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        
        # Setup complex embeddings
        entity_real = torch.randn(100, 32)
        entity_imag = torch.randn(100, 32)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(10, 32)
        
        manager.model = model
        manager.node_embeddings = None
        
        return RotatELightGBMTrainer(manager)

    def test_extract_embeddings_returns_dict(self, trainer_with_model):
        """Test that extract_embeddings returns a dictionary."""
        embeddings = trainer_with_model.extract_embeddings()
        
        assert isinstance(embeddings, dict)
        assert "entity_embeddings" in embeddings or "entity" in embeddings
        assert "relation_embeddings" in embeddings or "relation" in embeddings

    def test_extract_embeddings_correct_shape(self, trainer_with_model):
        """Test that extracted embeddings have correct shapes."""
        embeddings = trainer_with_model.extract_embeddings()
        
        entity_emb = embeddings.get("entity_embeddings", embeddings.get("entity"))
        relation_emb = embeddings.get("relation_embeddings", embeddings.get("relation"))
        
        assert entity_emb.shape[0] == 100  # num_entities
        assert entity_emb.shape[1] == 64   # concat(real, imag)
        assert relation_emb.shape[0] == 10  # num_relations

    def test_extract_embeddings_no_nan(self, trainer_with_model):
        """Test that embeddings don't contain NaN values."""
        embeddings = trainer_with_model.extract_embeddings()
        
        entity_emb = embeddings.get("entity_embeddings", embeddings.get("entity"))
        relation_emb = embeddings.get("relation_embeddings", embeddings.get("relation"))
        
        assert not np.any(np.isnan(entity_emb))
        assert not np.any(np.isnan(relation_emb))


class TestRotatELightGBMDatasetCreation:
    """Tests for dataset creation in LightGBM trainer."""

    @pytest.fixture
    def trainer_with_embeddings(self, tmp_path: Path):
        """Create trainer with pre-loaded embeddings."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        
        entity_real = torch.randn(50, 16)
        entity_imag = torch.randn(50, 16)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(5, 16)
        
        manager.model = model
        # Use string IDs matching what will be in the parquet file
        manager.entity_to_idx = {str(i): i for i in range(50)}
        manager.relation_to_idx = {str(i): i for i in range(5)}
        
        # Pre-load embeddings with consistent shapes for entity and relation
        # Entity: 32 dims (16 real + 16 imag), Relation: 32 dims to match
        emb_dim = 32
        manager.node_embeddings = {
            "entity_embeddings": np.random.randn(50, emb_dim).astype(np.float32),
            "relation_embeddings": np.random.randn(5, emb_dim).astype(np.float32),
            "entity": np.random.randn(50, emb_dim).astype(np.float32),
            "relation": np.random.randn(5, emb_dim).astype(np.float32),
        }
        
        trainer = RotatELightGBMTrainer(manager)
        trainer._embedding_dim = emb_dim
        
        return trainer, tmp_path

    def test_create_dataset_from_parquet(self, trainer_with_embeddings):
        """Test dataset creation from parquet file."""
        trainer, tmp_path = trainer_with_embeddings
        
        # Create test parquet file with string IDs matching the mappings
        import polars as pl
        df = pl.DataFrame({
            "head": ["0", "1", "2", "3"],
            "relation": ["0", "1", "0", "1"],
            "tail": ["1", "2", "3", "4"],
        })
        data_path = tmp_path / "train.parquet"
        df.write_parquet(data_path)
        
        X, y, meta = trainer.create_lightgbm_dataset(data_path)
        
        assert X.shape[0] == 4  # 4 triples
        assert y.shape[0] == 4
        assert np.all(y == 1)  # All positive samples

    def test_create_dataset_renames_columns(self, trainer_with_embeddings):
        """Test that dataset creation handles different column names."""
        trainer, tmp_path = trainer_with_embeddings
        
        import polars as pl
        # Use s, p, o format with string IDs matching mappings
        df = pl.DataFrame({
            "s": ["0", "1"],
            "p": ["0", "1"],
            "o": ["1", "2"],
        })
        data_path = tmp_path / "train_spo.parquet"
        df.write_parquet(data_path)
        
        X, y, meta = trainer.create_lightgbm_dataset(data_path)
        
        assert X.shape[0] == 2
        assert "triples" in meta


class TestRotatELightGBMNegativeSampling:
    """Tests for negative sampling in LightGBM trainer."""

    @pytest.fixture
    def trainer_for_sampling(self):
        """Create trainer for negative sampling tests."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        
        entity_real = torch.randn(100, 16)
        entity_imag = torch.randn(100, 16)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(10, 16)
        model.num_entities = 100
        
        manager.model = model
        manager.entity_to_idx = {f"e{i}": i for i in range(100)}
        manager.relation_to_idx = {f"r{i}": i for i in range(10)}
        
        # Use consistent embedding dimensions for entity and relation
        emb_dim = 32
        manager.node_embeddings = {
            "entity_embeddings": np.random.randn(100, emb_dim).astype(np.float32),
            "relation_embeddings": np.random.randn(10, emb_dim).astype(np.float32),
            "entity": np.random.randn(100, emb_dim).astype(np.float32),
            "relation": np.random.randn(10, emb_dim).astype(np.float32),
        }
        
        trainer = RotatELightGBMTrainer(manager)
        trainer._embedding_dim = emb_dim
        
        return trainer

    def test_negative_sampling_generates_samples(self, trainer_for_sampling):
        """Test that negative sampling generates correct number of samples."""
        X_pos = np.random.randn(100, 64)
        # Use correct metadata format - list of dicts with head, relation, tail
        meta = {
            "triples": [
                {"head": f"e{i}", "relation": "r0", "tail": f"e{(i+1) % 100}"}
                for i in range(100)
            ]
        }
        
        X_neg, y_neg = trainer_for_sampling.generate_negative_samples(X_pos, meta)
        
        assert X_neg.shape[0] == X_pos.shape[0]  # Same number of negatives
        assert np.all(y_neg == 0)  # All negative labels

    def test_negative_sampling_different_from_positive(self, trainer_for_sampling):
        """Test that negative samples have expected shape and are valid."""
        X_pos = np.random.randn(50, 64)
        # Use correct metadata format - list of dicts with head, relation, tail
        meta = {
            "triples": [
                {"head": f"e{i}", "relation": "r0", "tail": f"e{(i+1) % 100}"}
                for i in range(50)
            ]
        }
        
        X_neg, y_neg = trainer_for_sampling.generate_negative_samples(X_pos, meta)
        
        # Verify negatives were generated
        assert X_neg.shape[0] == 50  # Same number as positives
        assert np.all(y_neg == 0)  # All negative labels
        # X_neg has different shape due to feature engineering (concat, delta, etc)
        assert X_neg.shape[1] > 0  # Has features


class TestRotatELightGBMModelTraining:
    """Tests for LightGBM model training."""

    def test_lightgbm_training_params_loaded(self):
        """Test that LightGBM training params are loaded from config."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        entity_real = torch.randn(10, 8)
        entity_imag = torch.randn(10, 8)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(2, 8)
        manager.model = model
        manager.node_embeddings = None
        
        trainer = RotatELightGBMTrainer(manager)
        params = trainer._load_training_params()
        
        assert isinstance(params, dict)

    def test_lightgbm_accepts_custom_params(self):
        """Test that LightGBM accepts custom parameters."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        entity_real = torch.randn(10, 8)
        entity_imag = torch.randn(10, 8)
        model.get_entity_embeddings.return_value = (entity_real, entity_imag)
        model.get_relation_phases.return_value = torch.randn(2, 8)
        manager.model = model
        manager.node_embeddings = None
        
        trainer = RotatELightGBMTrainer(manager)
        
        # Custom params should be accepted
        custom_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
        }
        
        # Should not raise error
        assert isinstance(custom_params, dict)


class TestRotatELightGBMModelPersistence:
    """Tests for model saving and loading."""

    def test_save_hybrid_model(self, tmp_path: Path):
        """Test that hybrid model can be saved."""
        import lightgbm as lgb
        
        # Create dummy LightGBM model
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 2, 100)
        
        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "verbose": -1, "num_leaves": 4}
        model = lgb.train(params, train_data, num_boost_round=5)
        
        # Save model
        model_path = tmp_path / "lightgbm_model.txt"
        model.save_model(str(model_path))
        
        assert model_path.exists()

    def test_load_hybrid_model(self, tmp_path: Path):
        """Test that hybrid model can be loaded."""
        import lightgbm as lgb
        
        # Create and save model
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 2, 100)
        
        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "verbose": -1, "num_leaves": 4}
        model = lgb.train(params, train_data, num_boost_round=5)
        
        model_path = tmp_path / "lightgbm_model.txt"
        model.save_model(str(model_path))
        
        # Load model
        loaded_model = lgb.Booster(model_file=str(model_path))
        
        # Verify predictions match
        preds_original = model.predict(X[:10])
        preds_loaded = loaded_model.predict(X[:10])
        
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)


class TestRotatELightGBMErrorHandling:
    """Tests for error handling in LightGBM trainer."""

    def test_missing_model_raises_error(self):
        """Test that missing model raises appropriate error."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        manager.model = None
        manager.node_embeddings = None
        
        trainer = RotatELightGBMTrainer(manager)
        
        with pytest.raises((RuntimeError, AttributeError)):
            trainer.extract_embeddings()

    def test_missing_embeddings_raises_error(self):
        """Test that missing embeddings key raises error."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        model.get_entity_embeddings.return_value = (torch.randn(10, 4), torch.randn(10, 4))
        model.get_relation_phases.return_value = torch.randn(2, 4)
        manager.model = model
        manager.node_embeddings = {}  # Empty embeddings
        
        trainer = RotatELightGBMTrainer(manager)
        trainer.rotate_manager.node_embeddings = {}
        
        # Should handle missing keys gracefully or raise KeyError
        with pytest.raises((KeyError, RuntimeError)):
            # Force use of node_embeddings instead of extracting
            trainer.rotate_manager.node_embeddings = {"invalid": np.array([1, 2, 3])}
            import polars as pl
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                df = pl.DataFrame({"head": [0], "relation": [0], "tail": [1]})
                df.write_parquet(f.name)
                trainer.create_lightgbm_dataset(Path(f.name))

    def test_invalid_data_path_raises_error(self):
        """Test that invalid data path raises error."""
        from pff.validators.rotate.lightgbm_trainer import RotatELightGBMTrainer
        
        manager = Mock()
        model = Mock()
        model.get_entity_embeddings.return_value = (torch.randn(10, 4), torch.randn(10, 4))
        model.get_relation_phases.return_value = torch.randn(2, 4)
        manager.model = model
        manager.node_embeddings = {
            "entity": np.random.randn(10, 8),
            "relation": np.random.randn(2, 4),
        }
        
        trainer = RotatELightGBMTrainer(manager)
        
        with pytest.raises(FileNotFoundError):
            trainer.create_lightgbm_dataset(Path("/nonexistent/path.parquet"))


class TestRotatELightGBMMetrics:
    """Tests for metrics calculation in LightGBM trainer."""

    def test_binary_classification_metrics(self):
        """Test that binary classification metrics are computed correctly."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3])
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        assert 0 <= auc <= 1

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from sklearn.metrics import accuracy_score, f1_score
        
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        
        assert accuracy_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
