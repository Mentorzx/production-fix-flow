"""Tests for RotatE Ensemble Integration.

Phase 5 validation: Ensemble adapter components for RotatE integration
with sklearn-compatible interfaces and hybrid scoring.

Author: PFF Team
Date: 2025
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Tuple
from pathlib import Path


# ============================================================================
# Test: RotatEScorerService
# ============================================================================

class TestRotatEScorerService:
    """Test suite for RotatEScorerService scoring functionality."""
    
    @pytest.fixture
    def mock_kg_config(self) -> Mock:
        """Create mock KG config."""
        config = Mock()
        config.train_path = Path("/tmp/test_train.txt")
        return config
    
    @pytest.fixture
    def mock_file_manager(self) -> Mock:
        """Create mock FileManager."""
        fm = Mock()
        fm.read.return_value = {
            "model": {
                "embedding_dim": 256,
                "gamma": 12.0,
                "epsilon": 2.0,
            },
            "outputs": {
                "dir": "/tmp/rotate_outputs"
            }
        }
        return fm
    
    @patch("pff.validators.rotate.rotate_service.FileManager")
    @patch("pff.validators.rotate.mapping_utils.load_mappings")
    @patch("torch.load")
    def test_service_initialization(
        self, mock_torch_load, mock_load_mappings, mock_fm_class
    ):
        """Test RotatEScorerService initializes correctly."""
        # Setup mocks
        mock_load_mappings.return_value = (
            {f"e_{i}": i for i in range(100)},  # entity_to_idx
            {i: f"e_{i}" for i in range(100)},  # idx_to_entity
            {f"r_{i}": i for i in range(10)},   # relation_to_idx
            {i: f"r_{i}" for i in range(10)},   # idx_to_relation
        )
        mock_fm_class.return_value.read.return_value = {
            "model": {"embedding_dim": 256, "gamma": 12.0, "epsilon": 2.0},
            "outputs": {"dir": "/tmp/outputs"}
        }
        
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        mock_kg_config = Mock()
        mock_kg_config.train_path = Path("/tmp/train.txt")
        
        # Use patch.object for Path.exists to allow selective path behavior
        with patch.object(Path, 'exists', return_value=True):
            service = RotatEScorerService(
                kg_config=mock_kg_config,
                rotate_config_path=Path("/tmp/rotate.yaml"),
                load_best_model=False  # Don't load model for init test
            )
        
        assert len(service.entity_to_idx) == 100
        assert len(service.relation_to_idx) == 10
    
    def test_score_to_probability_sigmoid(self):
        """Test converting scores to probabilities using sigmoid."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            service.calibrator = None
            service.rotate_config = Mock()
            service.rotate_config.gamma = 12.0
            
            # Test with known scores
            # Lower score = more likely valid in RotatE
            score_low = 2.0  # Very good score
            score_mid = 6.0  # Medium score
            score_high = 11.0  # Bad score
            
            prob_low = service.score_to_probability(score_low)
            prob_mid = service.score_to_probability(score_mid)
            prob_high = service.score_to_probability(score_high)
            
            # Lower score should give higher probability
            assert prob_low > prob_mid > prob_high
            assert 0.0 <= prob_low <= 1.0
            assert 0.0 <= prob_mid <= 1.0
            assert 0.0 <= prob_high <= 1.0
    
    def test_get_combined_entity_embeddings(self):
        """Test combined entity embeddings extraction."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Mock model
            mock_model = Mock()
            mock_real = np.random.randn(100, 64).astype(np.float32)
            mock_imag = np.random.randn(100, 64).astype(np.float32)
            
            mock_model.get_entity_embeddings.return_value = (
                Mock(cpu=lambda: Mock(numpy=lambda: mock_real)),
                Mock(cpu=lambda: Mock(numpy=lambda: mock_imag))
            )
            service.model = mock_model
            
            combined = service.get_combined_entity_embeddings()
            
            assert combined.shape == (100, 128)  # dim * 2
    
    def test_get_combined_relation_embeddings(self):
        """Test combined relation embeddings extraction."""
        from pff.validators.rotate.rotate_service import RotatEScorerService
        
        with patch.object(RotatEScorerService, "__init__", lambda x, *args, **kwargs: None):
            service = RotatEScorerService.__new__(RotatEScorerService)
            
            # Mock model
            mock_model = Mock()
            mock_phases = np.random.uniform(-np.pi, np.pi, (10, 64)).astype(np.float32)
            
            mock_model.get_relation_phases.return_value = Mock(
                cpu=lambda: Mock(numpy=lambda: mock_phases)
            )
            service.model = mock_model
            
            combined = service.get_combined_relation_embeddings()
            
            # cos + sin components
            assert combined.shape == (10, 128)  # dim * 2


# ============================================================================
# Test: RotatEWrapper
# ============================================================================

class TestRotatEWrapper:
    """Test suite for RotatEWrapper sklearn compatibility."""
    
    @pytest.fixture
    def mock_scorer_service(self) -> Mock:
        """Create mock scorer service."""
        service = Mock()
        service.score_triple.return_value = 5.0
        service.score_to_probability.return_value = 0.7
        return service
    
    @patch("pff.validators.rotate.wrappers.KGConfig")
    @patch("pff.validators.rotate.wrappers.RotatEScorerService")
    def test_wrapper_fit(self, mock_service_class, mock_kg_config_class):
        """Test RotatEWrapper fit method."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Create dummy data
        X = [[("e1", "r1", "e2")]]
        y = np.array([1])
        
        # Fit should return self
        result = wrapper.fit(X, y)
        
        assert result is wrapper
        assert wrapper.scorer_service_ is not None
    
    @patch("pff.validators.rotate.wrappers.KGConfig")
    @patch("pff.validators.rotate.wrappers.RotatEScorerService")
    def test_wrapper_predict(self, mock_service_class, mock_kg_config_class):
        """Test RotatEWrapper predict method."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        
        mock_service = Mock()
        mock_service.score_triple.return_value = 5.0
        mock_service.score_to_probability.return_value = 0.7
        mock_service_class.return_value = mock_service
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Fit first
        wrapper.fit([[("e1", "r1", "e2")]], np.array([1]))
        
        # Predict
        X_test = [[("e1", "r1", "e2")]]
        
        # Need to mock concurrency manager
        wrapper.concurrency_manager = Mock()
        wrapper.concurrency_manager.execute_sync.return_value = [(0, 0.7)]
        
        predictions = wrapper.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1
        assert predictions[0] == 1  # 0.7 > 0.5
    
    @patch("pff.validators.rotate.wrappers.KGConfig")
    @patch("pff.validators.rotate.wrappers.RotatEScorerService")
    def test_wrapper_predict_proba(self, mock_service_class, mock_kg_config_class):
        """Test RotatEWrapper predict_proba method."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Fit first
        wrapper.fit([[("e1", "r1", "e2")]], np.array([1]))
        
        # Mock concurrency manager
        wrapper.concurrency_manager = Mock()
        wrapper.concurrency_manager.execute_sync.return_value = [(0, 0.7), (1, 0.3)]
        
        # Predict probabilities
        X_test = [[("e1", "r1", "e2")], [("e3", "r2", "e4")]]
        proba = wrapper.predict_proba(X_test)
        
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (2, 2)  # Binary classification
        assert proba[0, 1] == 0.7
        assert proba[1, 1] == 0.3
    
    def test_wrapper_sklearn_compatible_interface(self):
        """Test wrapper has sklearn-compatible interface."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Check required methods exist
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'predict_proba')
        assert callable(wrapper.fit)
        assert callable(wrapper.predict)
        assert callable(wrapper.predict_proba)


# ============================================================================
# Test: RotatEHybridWrapper
# ============================================================================

class TestRotatEHybridWrapper:
    """Test suite for RotatEHybridWrapper hybrid scoring."""
    
    @pytest.fixture
    def mock_lgb_model(self) -> Mock:
        """Create mock LightGBM model."""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.4, 0.6], [0.5, 0.5], [0.1, 0.9]])
        return model
    
    @pytest.fixture
    def mock_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create mock embeddings."""
        entity_emb = np.random.randn(100, 128).astype(np.float32)
        relation_emb = np.random.randn(10, 128).astype(np.float32)
        return entity_emb, relation_emb
    
    @pytest.fixture
    def mock_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Create mock mappings."""
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        return entity2idx, relation2idx
    
    def test_hybrid_initialization(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test RotatEHybridWrapper initializes correctly."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        assert hybrid.lightgbm_model == mock_lgb_model
        assert hybrid.entity_to_idx == entity2idx
        assert hybrid.relation_to_idx == relation2idx
        assert hybrid.entity_embeddings is entity_emb
        assert hybrid.relation_embeddings is relation_emb
    
    def test_hybrid_fit(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test hybrid fit method (no-op)."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [[("e_0", "r_0", "e_1")]]
        y = np.array([1])
        
        result = hybrid.fit(X, y)
        
        assert result is hybrid
    
    def test_hybrid_predict_proba(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test hybrid predict_proba with LightGBM."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        # Configure mock to return expected probabilities
        mock_lgb_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [[("e_0", "r_0", "e_1")]]
        proba = hybrid.predict_proba(X)
        
        assert isinstance(proba, np.ndarray)
        assert proba.shape == (1, 2)
        assert proba[0, 1] == 0.6
    
    def test_hybrid_predict(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test hybrid predict method."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        mock_lgb_model.predict_proba.return_value = np.array([
            [0.4, 0.6],  # Class 1
            [0.7, 0.3],  # Class 0
        ])
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [[("e_0", "r_0", "e_1")], [("e_2", "r_1", "e_3")]]
        predictions = hybrid.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        assert predictions[0] == 1  # 0.6 > 0.5
        assert predictions[1] == 0  # 0.3 < 0.5
    
    def test_hybrid_feature_extraction(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test feature extraction from triples."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [[("e_0", "r_0", "e_1")]]
        features = hybrid._extract_features(X)
        
        # Features should be concatenation of h, r, t embeddings
        expected_dim = 128 * 3  # entity + relation + entity
        assert features.shape == (1, expected_dim)
    
    def test_hybrid_unknown_entity_handling(self, mock_lgb_model, mock_embeddings, mock_mappings):
        """Test handling of unknown entities uses mean embeddings."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity2idx, relation2idx = mock_mappings
        entity_emb, relation_emb = mock_embeddings
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb_model,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        # Use unknown entity
        X = [[("unknown_entity", "r_0", "e_1")]]
        features = hybrid._extract_features(X)
        
        # Should still extract features using mean embedding
        expected_dim = 128 * 3
        assert features.shape == (1, expected_dim)


# ============================================================================
# Test: Ensemble Integration
# ============================================================================

class TestEnsembleIntegration:
    """Test suite for RotatE ensemble integration patterns."""
    
    def test_hybrid_as_base_model(self):
        """Test RotatEHybridWrapper can serve as base model."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        # Create mock components
        mock_lgb = Mock()
        mock_lgb.predict_proba.return_value = np.array([[0.4, 0.6]] * 10)
        
        entity_emb = np.random.randn(100, 128).astype(np.float32)
        relation_emb = np.random.randn(10, 128).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        # Use as estimator in ensemble
        X = [[("e_0", "r_0", "e_1")] for _ in range(10)]
        y = np.random.randint(0, 2, 10)
        
        hybrid.fit(X, y)
        proba = hybrid.predict_proba(X)
        
        assert proba.shape == (10, 2)
    
    @patch("pff.validators.rotate.wrappers.KGConfig")
    @patch("pff.validators.rotate.wrappers.RotatEScorerService")
    def test_wrapper_serialization(self, mock_service_class, mock_kg_config_class):
        """Test wrapper can be serialized (pickle)."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        import pickle
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Serialize
        serialized = pickle.dumps(wrapper)
        
        # Deserialize
        restored = pickle.loads(serialized)
        
        assert restored.kg_config_path == wrapper.kg_config_path
        assert restored.rotate_config_path == wrapper.rotate_config_path


# ============================================================================
# Test: Feature Extraction for LightGBM
# ============================================================================

class TestFeatureExtractionForLightGBM:
    """Test suite for RotatE feature extraction for LightGBM hybrid."""
    
    def test_extract_triple_features(self):
        """Test extracting features from triple for LightGBM."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        # Setup
        mock_lgb = Mock()
        entity_emb = np.random.randn(100, 64).astype(np.float32)
        relation_emb = np.random.randn(10, 64).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        # Extract features
        X = [[("e_0", "r_0", "e_1"), ("e_2", "r_1", "e_3")]]  # 2 triples per sample
        features = hybrid._extract_features(X)
        
        # Should average features across triples
        assert features.shape == (1, 64 * 3)  # averaged h+r+t
    
    def test_multiple_samples_features(self):
        """Test feature extraction for multiple samples."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        mock_lgb = Mock()
        entity_emb = np.random.randn(100, 64).astype(np.float32)
        relation_emb = np.random.randn(10, 64).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [
            [("e_0", "r_0", "e_1")],
            [("e_2", "r_1", "e_3")],
            [("e_4", "r_2", "e_5")],
        ]
        features = hybrid._extract_features(X)
        
        assert features.shape == (3, 64 * 3)


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling in ensemble components."""
    
    def test_wrapper_not_fitted_error(self):
        """Test that scorer_service_ is None before fitting."""
        from pff.validators.rotate.wrappers import RotatEWrapper
        
        wrapper = RotatEWrapper(
            kg_config_path="/tmp/kg.yaml",
            rotate_config_path="/tmp/rotate.yaml"
        )
        
        # Verify the wrapper's scorer_service_ is None before fitting
        assert wrapper.scorer_service_ is None
    
    def test_empty_sample_handling(self):
        """Test handling of empty samples."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        mock_lgb = Mock()
        mock_lgb.predict_proba.return_value = np.array([[0.5, 0.5]])
        
        entity_emb = np.random.randn(100, 64).astype(np.float32)
        relation_emb = np.random.randn(10, 64).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        # Empty sample (no triples)
        X = [[]]
        features = hybrid._extract_features(X)
        
        # Should return zeros for empty sample
        assert features.shape == (1, 64 * 3)
        np.testing.assert_array_equal(features[0], np.zeros(64 * 3))
    
    def test_model_not_loaded_fallback(self):
        """Test fallback when LightGBM model is None."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        entity_emb = np.random.randn(100, 64).astype(np.float32)
        relation_emb = np.random.randn(10, 64).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=None,  # No model
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        X = [[("e_0", "r_0", "e_1")]]
        proba = hybrid.predict_proba(X)
        
        # Should return 0.5 probability when no model
        np.testing.assert_array_equal(proba, np.full((1, 2), 0.5))


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Test suite for ensemble configuration options."""
    
    def test_mean_embedding_computation(self):
        """Test mean embedding is computed for unknown entity fallback."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        mock_lgb = Mock()
        entity_emb = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)
        relation_emb = np.array([
            [0.1, 0.2, 0.3],
        ], dtype=np.float32)
        entity2idx = {"e_0": 0, "e_1": 1}
        relation2idx = {"r_0": 0}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        # Mean entity embedding should be [2.5, 3.5, 4.5]
        expected_mean = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(
            hybrid.mean_entity_embedding_, expected_mean
        )
    
    def test_embedding_dimension_storage(self):
        """Test embedding dimension is stored correctly."""
        from pff.validators.rotate.wrappers import RotatEHybridWrapper
        
        mock_lgb = Mock()
        entity_emb = np.random.randn(100, 128).astype(np.float32)
        relation_emb = np.random.randn(10, 128).astype(np.float32)
        entity2idx = {f"e_{i}": i for i in range(100)}
        relation2idx = {f"r_{i}": i for i in range(10)}
        
        hybrid = RotatEHybridWrapper(
            lightgbm_model=mock_lgb,
            entity_to_idx=entity2idx,
            relation_to_idx=relation2idx,
            entity_embeddings=entity_emb,
            relation_embeddings=relation_emb,
        )
        
        assert hybrid._embedding_dim == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
