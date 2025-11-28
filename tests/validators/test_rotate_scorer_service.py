"""Tests for RotatE Scorer Service.

Comprehensive tests for the RotatEScorerService including:
- Initialization and configuration
- Model loading and checkpointing
- Triple scoring
- Batch scoring
- Score calibration
- Embedding extraction
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


class TestRotatEScorerServiceInitialization:
    """Tests for RotatEScorerService initialization."""

    @pytest.fixture
    def mock_kg_config(self):
        """Create a mock KG configuration."""
        config = Mock()
        config.num_entities = 100
        config.num_relations = 10
        return config

    @pytest.fixture
    def mock_config_file(self, tmp_path: Path) -> Path:
        """Create a mock RotatE config YAML file."""
        config_path = tmp_path / "rotate_config.yaml"
        config_content = """
model:
  embedding_dim: 64
  gamma: 9.0
  epsilon: 2.0
outputs:
  dir: outputs/rotate
training:
  epochs: 10
  batch_size: 256
"""
        config_path.write_text(config_content)
        return config_path

    @patch("pff.validators.rotate.rotate_service.FileManager")
    def test_service_initialization(self, mock_fm, mock_kg_config, tmp_path: Path):
        """Test service initializes correctly without loading model."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        # Setup mock
        mock_fm_instance = Mock()
        mock_fm_instance.read.return_value = {
            "model": {"embedding_dim": 64, "gamma": 9.0, "epsilon": 2.0},
            "outputs": {"dir": str(tmp_path / "rotate")},
        }
        mock_fm.return_value = mock_fm_instance
        
        config_path = tmp_path / "config.yaml"
        config_path.touch()
        
        with patch.object(RotatEScorerService, "_load_mappings"):
            with patch.object(RotatEScorerService, "_load_model"):
                with patch.object(RotatEScorerService, "_load_calibrator"):
                    service = RotatEScorerService(
                        mock_kg_config, config_path, load_best_model=False
                    )
        
        assert service.model is None
        assert service.kg_config == mock_kg_config

    @patch("pff.validators.rotate.rotate_service.FileManager")
    def test_service_with_config_values(self, mock_fm, mock_kg_config, tmp_path: Path):
        """Test service uses config values correctly."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        mock_fm_instance = Mock()
        mock_fm_instance.read.return_value = {
            "model": {"embedding_dim": 128, "gamma": 12.0, "epsilon": 3.0},
            "outputs": {"dir": str(tmp_path / "rotate")},
        }
        mock_fm.return_value = mock_fm_instance
        
        config_path = tmp_path / "config.yaml"
        config_path.touch()
        
        with patch.object(RotatEScorerService, "_load_mappings"):
            with patch.object(RotatEScorerService, "_load_model"):
                with patch.object(RotatEScorerService, "_load_calibrator"):
                    service = RotatEScorerService(
                        mock_kg_config, config_path, load_best_model=False
                    )
        
        assert service.rotate_config.embedding_dim == 128
        assert service.rotate_config.gamma == 12.0


class TestRotatEScorerServiceScoring:
    """Tests for triple scoring functionality."""

    @pytest.fixture
    def service_with_model(self):
        """Create service with mock model."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Setup mock model
            service.model = Mock()
            service.model.score_triple.return_value = 5.0
            service.model.score_triples_batch.return_value = np.array([5.0, 6.0, 7.0])
            
            # Setup mappings
            service.entity_to_idx = {"e1": 0, "e2": 1, "e3": 2}
            service.relation_to_idx = {"r1": 0, "r2": 1}
            
            # Setup config
            service.rotate_config = Mock()
            service.rotate_config.gamma = 9.0
            
            service.calibrator = None
            
            return service

    def test_score_triple_valid_entities(self, service_with_model):
        """Test scoring valid triple returns score."""
        score = service_with_model.score_triple("e1", "r1", "e2")
        
        assert score == 5.0
        service_with_model.model.score_triple.assert_called_once_with(0, 0, 1)

    def test_score_triple_unknown_entity(self, service_with_model):
        """Test scoring unknown entity returns gamma."""
        score = service_with_model.score_triple("unknown", "r1", "e2")
        
        assert score == service_with_model.rotate_config.gamma

    def test_score_triple_unknown_relation(self, service_with_model):
        """Test scoring unknown relation returns gamma."""
        score = service_with_model.score_triple("e1", "unknown_rel", "e2")
        
        assert score == service_with_model.rotate_config.gamma

    def test_score_triple_no_model_raises_error(self, service_with_model):
        """Test scoring without model raises ValueError."""
        service_with_model.model = None
        
        with pytest.raises(ValueError, match="nao carregado"):
            service_with_model.score_triple("e1", "r1", "e2")

    def test_score_triple_batch(self, service_with_model):
        """Test batch scoring returns array of scores."""
        triples = [("e1", "r1", "e2"), ("e2", "r1", "e3"), ("e1", "r2", "e3")]
        
        scores = service_with_model.score_triple_batch(triples)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3

    def test_score_triple_batch_with_unknown(self, service_with_model):
        """Test batch scoring handles unknown entities."""
        # First triple is valid (e1, r1, e2), second has unknown entity
        triples = [("e1", "r1", "e2"), ("unknown", "r1", "e2")]
        
        # Mock should return only scores for valid triples
        # The service will only score the first triple (valid)
        service_with_model.model.score_triples_batch.return_value = np.array([5.0])
        
        scores = service_with_model.score_triple_batch(triples)
        
        # First should have the real score, second should get gamma (unknown)
        assert scores[0] == 5.0  # From mock
        assert scores[1] == service_with_model.rotate_config.gamma  # Default for unknown


class TestRotatEScorerServiceCalibration:
    """Tests for score calibration functionality."""

    @pytest.fixture
    def service_with_calibrator(self):
        """Create service with mock calibrator."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            service.model = Mock()
            service.rotate_config = Mock()
            service.rotate_config.gamma = 9.0
            
            service.calibrator = Mock()
            service.calibrator.transform.return_value = np.array([0.75])
            
            return service

    def test_score_to_probability_with_calibrator(self, service_with_calibrator):
        """Test probability conversion with calibrator."""
        prob = service_with_calibrator.score_to_probability(5.0)
        
        assert prob == 0.75
        service_with_calibrator.calibrator.transform.assert_called_once()

    def test_score_to_probability_without_calibrator(self):
        """Test probability conversion without calibrator (sigmoid fallback)."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.calibrator = None
            service.rotate_config = Mock()
            service.rotate_config.gamma = 9.0
            
            prob = service.score_to_probability(4.5)
            
            assert 0 <= prob <= 1

    def test_score_to_probability_edge_cases(self):
        """Test probability conversion with edge case scores."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.calibrator = None
            service.rotate_config = Mock()
            service.rotate_config.gamma = 9.0
            
            # Very low score (good prediction)
            prob_low = service.score_to_probability(0.0)
            
            # Score equal to gamma
            prob_gamma = service.score_to_probability(9.0)
            
            # High score (bad prediction)
            prob_high = service.score_to_probability(18.0)
            
            assert prob_low > prob_gamma > prob_high


class TestRotatEScorerServiceEmbeddings:
    """Tests for embedding extraction functionality."""

    @pytest.fixture
    def service_with_embeddings(self):
        """Create service with mock embeddings."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Create mock model with embeddings
            service.model = Mock()
            
            entity_real = torch.randn(100, 32)
            entity_imag = torch.randn(100, 32)
            relation_phases = torch.randn(10, 32)
            
            service.model.get_entity_embeddings.return_value = (entity_real, entity_imag)
            service.model.get_relation_phases.return_value = relation_phases
            
            return service

    def test_get_entity_embeddings(self, service_with_embeddings):
        """Test getting entity embeddings as tuple."""
        real, imag = service_with_embeddings.get_entity_embeddings()
        
        assert isinstance(real, np.ndarray)
        assert isinstance(imag, np.ndarray)
        assert real.shape == (100, 32)
        assert imag.shape == (100, 32)

    def test_get_relation_embeddings(self, service_with_embeddings):
        """Test getting relation phase embeddings."""
        phases = service_with_embeddings.get_relation_embeddings()
        
        assert isinstance(phases, np.ndarray)
        assert phases.shape == (10, 32)

    def test_get_combined_entity_embeddings(self, service_with_embeddings):
        """Test getting concatenated entity embeddings."""
        combined = service_with_embeddings.get_combined_entity_embeddings()
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (100, 64)  # real + imag

    def test_get_combined_relation_embeddings(self, service_with_embeddings):
        """Test getting combined relation embeddings (cos + sin)."""
        combined = service_with_embeddings.get_combined_relation_embeddings()
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (10, 64)  # cos + sin

    def test_get_embeddings_no_model_raises_error(self):
        """Test getting embeddings without model raises error."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.model = None
            
            with pytest.raises(ValueError, match="nao carregado"):
                service.get_entity_embeddings()


class TestRotatEScorerServiceMappings:
    """Tests for mapping loading functionality."""

    def test_load_mappings_sets_mappings(self, tmp_path: Path):
        """Test that _load_mappings sets entity and relation mappings."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        import polars as pl
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.config_data = {"outputs": {"dir": str(tmp_path)}}
            service.file_manager = Mock()
            service.entity_to_idx = {}
            service.idx_to_entity = {}
            service.relation_to_idx = {}
            service.idx_to_relation = {}
            
            # Create mock mapping files
            maps_dir = tmp_path / "maps"
            maps_dir.mkdir(parents=True)
            
            entity_df = pl.DataFrame({"id": [0, 1], "label": ["e1", "e2"]})
            entity_df.write_parquet(maps_dir / "rotate_entity_map_raw.parquet")
            
            relation_df = pl.DataFrame({"id": [0], "label": ["r1"]})
            relation_df.write_parquet(maps_dir / "rotate_relation_map_raw.parquet")
            
            # Call _load_mappings
            service._load_mappings()
            
            # Verify mappings were loaded
            assert len(service.entity_to_idx) == 2
            assert len(service.relation_to_idx) == 1
            assert service.entity_to_idx["e1"] == 0
            assert service.relation_to_idx["r1"] == 0


class TestRotatEScorerServiceModelLoading:
    """Tests for model loading functionality."""

    def test_load_model_from_checkpoint(self, tmp_path: Path):
        """Test loading model from checkpoint file."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        from pff.validators.rotate.core import RotatEModel
        
        # Create checkpoint directory
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        
        # Create a real model and save checkpoint
        model = RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
            gamma=9.0,
        )
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_entities": 100,
            "num_relations": 10,
        }
        torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        
        # Create service and test loading
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.config_data = {"outputs": {"dir": str(tmp_path)}}
            service.entity_to_idx = {f"e{i}": i for i in range(100)}
            service.relation_to_idx = {f"r{i}": i for i in range(10)}
            service.rotate_config = Mock()
            service.rotate_config.embedding_dim = 64
            service.rotate_config.gamma = 9.0
            service.rotate_config.epsilon = 2.0
            service._device = torch.device("cpu")
            service.model = None
            
            service._load_model()
            
            assert service.model is not None
            assert isinstance(service.model, RotatEModel)

    def test_load_model_missing_checkpoint_warns(self, tmp_path: Path):
        """Test that missing checkpoint logs warning."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.config_data = {"outputs": {"dir": str(tmp_path)}}
            service.entity_to_idx = {}
            service.relation_to_idx = {}
            service.model = None
            
            with patch("pff.validators.rotate.rotate_service.logger") as mock_logger:
                service._load_model()
                
                assert service.model is None
                mock_logger.warning.assert_called()


class TestRotatEScorerServiceIntegration:
    """Integration tests for RotatEScorerService."""

    def test_full_scoring_pipeline(self):
        """Test complete scoring pipeline with mock model."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Setup complete service
            service.model = Mock()
            service.model.score_triple.return_value = 3.0
            service.entity_to_idx = {"user:1": 0, "user:2": 1, "device:1": 2}
            service.relation_to_idx = {"uses": 0, "owns": 1}
            service.rotate_config = Mock()
            service.rotate_config.gamma = 9.0
            service.calibrator = None
            
            # Test scoring
            score = service.score_triple("user:1", "uses", "device:1")
            prob = service.score_to_probability(score)
            
            assert score == 3.0
            assert 0 < prob < 1  # Should be above 0.5 for low score

    def test_embedding_extraction_pipeline(self):
        """Test complete embedding extraction pipeline."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Setup model with real tensors
            service.model = Mock()
            entity_real = torch.randn(50, 32)
            entity_imag = torch.randn(50, 32)
            relation_phases = torch.randn(5, 32)
            
            service.model.get_entity_embeddings.return_value = (entity_real, entity_imag)
            service.model.get_relation_phases.return_value = relation_phases
            
            # Extract embeddings
            entity_combined = service.get_combined_entity_embeddings()
            relation_combined = service.get_combined_relation_embeddings()
            
            # Verify shapes for LightGBM compatibility
            assert entity_combined.shape == (50, 64)
            assert relation_combined.shape == (5, 64)
            
            # Verify no NaN
            assert not np.any(np.isnan(entity_combined))
            assert not np.any(np.isnan(relation_combined))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
