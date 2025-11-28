"""Tests for RotatE Trainer and Adapter Components.

Tests for Phase 3 (Training Pipeline) and Phase 5 (Ensemble Adapter) components.

Author: PFF Team
Date: 2025-11-25
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import torch


# ============================================================================
# Test: RotatETrainerConfig
# ============================================================================

class TestRotatETrainerConfig:
    """Tests for RotatETrainerConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from pff.validators.rotate.trainer import RotatETrainerConfig
        
        config = RotatETrainerConfig()
        
        assert config.num_epochs == 100
        assert config.batch_size == 1024
        assert config.gamma == 12.0
        assert config.adversarial_temperature == 1.0
        assert config.use_self_adversarial is True
        assert config.gradient_clip_val == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from pff.validators.rotate.trainer import RotatETrainerConfig
        
        config = RotatETrainerConfig(
            num_epochs=200,
            batch_size=512,
            gamma=24.0,
            adversarial_temperature=0.5,
            use_self_adversarial=False,
        )
        
        assert config.num_epochs == 200
        assert config.batch_size == 512
        assert config.gamma == 24.0
        assert config.adversarial_temperature == 0.5
        assert config.use_self_adversarial is False


# ============================================================================
# Test: RotatETrainer
# ============================================================================

class TestRotatETrainer:
    """Tests for RotatETrainer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock RotatE model."""
        model = Mock()
        model.num_entities = 100
        model.num_relations = 10
        model.embedding_dim = 64
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        model.train.return_value = None
        model.eval.return_value = None
        model.to.return_value = model
        model.state_dict.return_value = {}
        model.score_triples_batch.return_value = torch.randn(32)
        model.regularization_loss.return_value = torch.tensor(0.0)
        return model
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1000)
        dataset.__getitem__ = Mock(return_value={
            "positive": torch.tensor([[0, 0, 1]]),
            "neg_heads": torch.tensor([[2, 3, 4]]),
            "neg_tails": torch.tensor([[5, 6, 7]]),
        })
        return dataset
    
    def test_trainer_initialization(self, mock_model):
        """Test trainer initializes correctly."""
        from pff.validators.rotate.trainer import RotatETrainer, RotatETrainerConfig
        
        config = RotatETrainerConfig(num_epochs=10)
        
        with patch.object(mock_model, 'to', return_value=mock_model):
            trainer = RotatETrainer(mock_model, config)
        
        assert trainer.model == mock_model
        assert trainer.config == config
        assert trainer.global_step == 0
    
    def test_trainer_device_resolution(self, mock_model):
        """Test device resolution."""
        from pff.validators.rotate.trainer import RotatETrainer, RotatETrainerConfig
        
        config = RotatETrainerConfig(device="cpu")
        
        with patch.object(mock_model, 'to', return_value=mock_model):
            trainer = RotatETrainer(mock_model, config)
        
        assert trainer.device == torch.device("cpu")
    
    def test_trainer_setup(self, mock_model, mock_dataset):
        """Test trainer setup creates optimizer and scheduler."""
        from pff.validators.rotate.trainer import RotatETrainer, RotatETrainerConfig
        
        config = RotatETrainerConfig(num_epochs=10)
        
        with patch.object(mock_model, 'to', return_value=mock_model):
            trainer = RotatETrainer(mock_model, config)
            trainer.setup(mock_dataset)
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None


# ============================================================================
# Test: RotatEEnsembleAdapter
# ============================================================================

class TestRotatEEnsembleAdapter:
    """Tests for RotatEEnsembleAdapter class."""
    
    @pytest.fixture
    def mock_scorer_service(self):
        """Create mock scorer service."""
        service = Mock()
        service.entity_to_idx = {f"e_{i}": i for i in range(100)}
        service.idx_to_entity = {i: f"e_{i}" for i in range(100)}
        service.relation_to_idx = {f"r_{i}": i for i in range(10)}
        service.idx_to_relation = {i: f"r_{i}" for i in range(10)}
        service.model = Mock()
        service.model.embedding_dim = 64
        service.score_triple.return_value = 5.0
        service.score_triple_batch.return_value = np.array([5.0, 6.0, 7.0])
        service.score_to_probability.return_value = 0.7
        service.get_combined_entity_embeddings.return_value = np.random.randn(100, 128)
        service.get_combined_relation_embeddings.return_value = np.random.randn(10, 128)
        return service
    
    @patch("pff.validators.rotate.adapter.KGConfig")
    @patch("pff.validators.rotate.adapter.RotatEScorerService")
    def test_adapter_initialization(self, mock_service_class, mock_kg_config_class):
        """Test adapter initializes correctly."""
        from pff.validators.rotate.adapter import RotatEEnsembleAdapter
        
        mock_service = Mock()
        mock_service.entity_to_idx = {"e1": 0}
        mock_service.idx_to_entity = {0: "e1"}
        mock_service.relation_to_idx = {"r1": 0}
        mock_service.idx_to_relation = {0: "r1"}
        mock_service.model = Mock()
        mock_service.model.embedding_dim = 64
        mock_service_class.return_value = mock_service
        
        adapter = RotatEEnsembleAdapter(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        assert adapter.entity_to_idx == {"e1": 0}
        assert adapter.relation_to_idx == {"r1": 0}
    
    @patch("pff.validators.rotate.adapter.KGConfig")
    @patch("pff.validators.rotate.adapter.RotatEScorerService")
    def test_adapter_score_triple(self, mock_service_class, mock_kg_config_class):
        """Test adapter score_triple method."""
        from pff.validators.rotate.adapter import RotatEEnsembleAdapter
        
        mock_service = Mock()
        mock_service.entity_to_idx = {"e1": 0, "e2": 1}
        mock_service.idx_to_entity = {0: "e1", 1: "e2"}
        mock_service.relation_to_idx = {"r1": 0}
        mock_service.idx_to_relation = {0: "r1"}
        mock_service.model = Mock()
        mock_service.model.embedding_dim = 64
        mock_service.score_triple.return_value = 5.0
        mock_service_class.return_value = mock_service
        
        adapter = RotatEEnsembleAdapter(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        score = adapter.score_triple("e1", "r1", "e2")
        
        assert score == 5.0
        mock_service.score_triple.assert_called_once_with("e1", "r1", "e2")
    
    @patch("pff.validators.rotate.adapter.KGConfig")
    @patch("pff.validators.rotate.adapter.RotatEScorerService")
    def test_adapter_get_entity_embedding(self, mock_service_class, mock_kg_config_class):
        """Test adapter get_entity_embedding method."""
        from pff.validators.rotate.adapter import RotatEEnsembleAdapter
        
        mock_service = Mock()
        mock_service.entity_to_idx = {"e1": 0}
        mock_service.idx_to_entity = {0: "e1"}
        mock_service.relation_to_idx = {}
        mock_service.idx_to_relation = {}
        mock_service.model = Mock()
        mock_service.model.embedding_dim = 64
        
        embeddings = np.random.randn(1, 128).astype(np.float32)
        mock_service.get_combined_entity_embeddings.return_value = embeddings
        mock_service_class.return_value = mock_service
        
        adapter = RotatEEnsembleAdapter(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        emb = adapter.get_entity_embedding("e1")
        
        assert emb.shape == (128,)
        np.testing.assert_array_equal(emb, embeddings[0])
    
    @patch("pff.validators.rotate.adapter.KGConfig")
    @patch("pff.validators.rotate.adapter.RotatEScorerService")
    def test_adapter_get_all_embeddings(self, mock_service_class, mock_kg_config_class):
        """Test adapter get_all_entity_embeddings method."""
        from pff.validators.rotate.adapter import RotatEEnsembleAdapter
        
        mock_service = Mock()
        mock_service.entity_to_idx = {f"e_{i}": i for i in range(100)}
        mock_service.idx_to_entity = {i: f"e_{i}" for i in range(100)}
        mock_service.relation_to_idx = {f"r_{i}": i for i in range(10)}
        mock_service.idx_to_relation = {i: f"r_{i}" for i in range(10)}
        mock_service.model = Mock()
        mock_service.model.embedding_dim = 64
        
        entity_emb = np.random.randn(100, 128).astype(np.float32)
        relation_emb = np.random.randn(10, 128).astype(np.float32)
        mock_service.get_combined_entity_embeddings.return_value = entity_emb
        mock_service.get_combined_relation_embeddings.return_value = relation_emb
        mock_service_class.return_value = mock_service
        
        adapter = RotatEEnsembleAdapter(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        all_ent = adapter.get_all_entity_embeddings()
        all_rel = adapter.get_all_relation_embeddings()
        
        assert all_ent.shape == (100, 128)
        assert all_rel.shape == (10, 128)


# ============================================================================
# Test: RotatEEmbeddingAdapter
# ============================================================================

class TestRotatEEmbeddingAdapter:
    """Tests for RotatEEmbeddingAdapter class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock RotatE model."""
        model = Mock()
        model.num_entities = 100
        model.num_relations = 10
        model.embedding_dim = 64
        
        # Mock embedding extraction
        ent_real = torch.randn(100, 64)
        ent_imag = torch.randn(100, 64)
        phases = torch.randn(10, 64)
        
        model.get_entity_embeddings.return_value = (ent_real, ent_imag)
        model.get_relation_phases.return_value = phases
        
        return model
    
    def test_embedding_adapter_initialization(self, mock_model):
        """Test embedding adapter initializes correctly."""
        from pff.validators.rotate.adapter import RotatEEmbeddingAdapter
        
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        adapter = RotatEEmbeddingAdapter(mock_model, entity2idx, relation2idx)
        
        assert adapter.entity_embeddings.shape == (100, 128)  # 2 * 64
        assert adapter.relation_embeddings.shape == (10, 128)  # 2 * 64
        assert adapter.embedding_dim == 128
    
    def test_embedding_adapter_entity_embedding(self, mock_model):
        """Test embedding adapter get_entity_embedding."""
        from pff.validators.rotate.adapter import RotatEEmbeddingAdapter
        
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        adapter = RotatEEmbeddingAdapter(mock_model, entity2idx, relation2idx)
        
        emb = adapter.get_entity_embedding(0)
        
        assert emb.shape == (128,)
    
    def test_embedding_adapter_relation_embedding(self, mock_model):
        """Test embedding adapter get_relation_embedding."""
        from pff.validators.rotate.adapter import RotatEEmbeddingAdapter
        
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        adapter = RotatEEmbeddingAdapter(mock_model, entity2idx, relation2idx)
        
        emb = adapter.get_relation_embedding(0)
        
        assert emb.shape == (128,)
    
    def test_embedding_adapter_invalid_index(self, mock_model):
        """Test embedding adapter handles invalid indices."""
        from pff.validators.rotate.adapter import RotatEEmbeddingAdapter
        
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        adapter = RotatEEmbeddingAdapter(mock_model, entity2idx, relation2idx)
        
        # Invalid entity index
        emb = adapter.get_entity_embedding(999)
        assert emb.shape == (128,)
        np.testing.assert_array_equal(emb, np.zeros(128))
        
        # Invalid relation index
        rel_emb = adapter.get_relation_embedding(999)
        assert rel_emb.shape == (128,)
        np.testing.assert_array_equal(rel_emb, np.zeros(128))


# ============================================================================
# Test: Integration
# ============================================================================

class TestTrainerAdapterIntegration:
    """Integration tests for trainer and adapter components."""
    
    def test_imports_work(self):
        """Test all components can be imported."""
        from pff.validators.rotate.trainer import RotatETrainer, RotatETrainerConfig
        from pff.validators.rotate.adapter import RotatEEnsembleAdapter, RotatEEmbeddingAdapter
        
        assert RotatETrainer is not None
        assert RotatETrainerConfig is not None
        assert RotatEEnsembleAdapter is not None
        assert RotatEEmbeddingAdapter is not None
    
    def test_trainer_inherits_base_trainer(self):
        """Test RotatETrainer inherits from BaseTrainer."""
        from pff.validators.rotate.trainer import RotatETrainer
        from pff.utils.ml.base_trainer import BaseTrainer
        
        assert issubclass(RotatETrainer, BaseTrainer)
    
    def test_trainer_config_inherits_base_config(self):
        """Test RotatETrainerConfig inherits from TrainerConfig."""
        from pff.validators.rotate.trainer import RotatETrainerConfig
        from pff.utils.ml.base_trainer import TrainerConfig
        
        assert issubclass(RotatETrainerConfig, TrainerConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
