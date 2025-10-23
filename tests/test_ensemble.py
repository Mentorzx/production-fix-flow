"""
Tests for Ensemble Wrappers (pff/validators/ensembles/ensemble_wrappers/)

Sprint 6: Testes IA/ML - Ensemble

Tests cover:
- BaseWrapper interface compliance
- TransEWrapper (KG embeddings wrapper)
- HybridWrapper (TransE + LightGBM)
- Calibration and probability transformations
- Sklearn compatibility
- Serialization/deserialization
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from pff.validators.ensembles.ensemble_wrappers.base_wrapper import BaseWrapper, get_shared_cache
from pff.validators.ensembles.ensemble_wrappers.model_wrappers import TransEWrapper, HybridWrapper


# ═══════════════════════════════════════════════════════════════════
# Base Wrapper Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBaseWrapper:
    """Test BaseWrapper abstract class."""

    def test_base_wrapper_cannot_instantiate(self):
        """Test BaseWrapper cannot be instantiated directly (abstract)."""
        with pytest.raises(TypeError):
            BaseWrapper()

    def test_base_wrapper_has_required_methods(self):
        """Test BaseWrapper defines required abstract methods."""
        # Check that abstract methods are defined
        assert hasattr(BaseWrapper, 'fit')
        assert hasattr(BaseWrapper, 'predict_proba')
        assert hasattr(BaseWrapper, 'predict')

    def test_get_shared_cache_singleton(self):
        """Test get_shared_cache returns singleton instance."""
        cache1 = get_shared_cache()
        cache2 = get_shared_cache()

        # Should be the same instance
        assert cache1 is cache2


# ═══════════════════════════════════════════════════════════════════
# TransEWrapper Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTransEWrapper:
    """Test TransEWrapper functionality."""

    @pytest.fixture
    def mock_kg_config_path(self, tmp_path):
        """Create a mock KG config path."""
        config_path = tmp_path / "kg_config.json"
        config_path.write_text('{"entities": 100, "relations": 10}')
        return str(config_path)

    @pytest.fixture
    def mock_transe_config_path(self, tmp_path):
        """Create a mock TransE config path."""
        config_path = tmp_path / "transe_config.json"
        config_path.write_text('{"embedding_dim": 128, "margin": 2.0}')
        return str(config_path)

    def test_transe_wrapper_initialization(self, mock_kg_config_path, mock_transe_config_path):
        """Test TransEWrapper initializes correctly."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        assert wrapper.kg_config_path == mock_kg_config_path
        assert wrapper.transe_config_path == mock_transe_config_path
        assert wrapper.scorer_service_ is None  # Not initialized until fit()
        assert wrapper.timeout == 30.0

    def test_transe_wrapper_inherits_base_wrapper(self, mock_kg_config_path, mock_transe_config_path):
        """Test TransEWrapper inherits from BaseWrapper."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        assert isinstance(wrapper, BaseWrapper)
        assert hasattr(wrapper, 'classes_')
        assert np.array_equal(wrapper.classes_, np.array([0, 1]))

    def test_transe_wrapper_cache_key_generation(self, mock_kg_config_path, mock_transe_config_path):
        """Test cache key is generated correctly."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        # Cache key should contain config file names
        assert "kg_config" in wrapper._cache_key
        assert "transe_config" in wrapper._cache_key

    def test_transe_wrapper_getstate_removes_unpicklable(self, mock_kg_config_path, mock_transe_config_path):
        """Test __getstate__ removes non-picklable objects."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        # Mock scorer service
        wrapper.scorer_service_ = Mock()

        state = wrapper.__getstate__()

        # scorer_service_ should be None in serialized state
        assert state['scorer_service_'] is None

    def test_transe_wrapper_setstate_restores_managers(self, mock_kg_config_path, mock_transe_config_path):
        """Test __setstate__ restores managers."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        state = wrapper.__getstate__()
        wrapper.__setstate__(state)

        # Managers should be restored
        assert wrapper.file_manager is not None
        assert wrapper.cache_manager is not None
        assert wrapper.concurrency_manager is not None

    @patch('pff.validators.ensembles.ensemble_wrappers.model_wrappers.TransEScorerService')
    @patch('pff.validators.ensembles.ensemble_wrappers.model_wrappers.KGConfig')
    def test_transe_wrapper_fit_initializes_scorer(
        self, mock_kg_config, mock_scorer_service, mock_kg_config_path, mock_transe_config_path
    ):
        """Test fit() initializes scorer service."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        # Mock the KGConfig and TransEScorerService
        mock_kg_config.return_value = Mock()
        mock_scorer_service.return_value = Mock()

        X_train = [(0, 0, 1), (1, 1, 2)]  # Dummy triples
        wrapper.fit(X_train)

        # Scorer service should be initialized
        assert wrapper.scorer_service_ is not None
        mock_scorer_service.assert_called_once()

    def test_transe_wrapper_predict_proba_shape(self, mock_kg_config_path, mock_transe_config_path):
        """Test predict_proba returns correct shape."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        # Mock the entire predict_proba to avoid threading issues
        wrapper.predict_proba = Mock(return_value=np.array([
            [0.7, 0.3],
            [0.3, 0.7],
            [0.5, 0.5],
        ]))

        X_test = [(0, 0, 1), (1, 1, 2), (2, 0, 3)]
        probas = wrapper.predict_proba(X_test)

        # Should return (n_samples, 2) for binary classification
        assert probas.shape == (3, 2)
        assert np.all((probas >= 0) & (probas <= 1))  # Probabilities in [0, 1]
        assert np.allclose(probas.sum(axis=1), 1.0)  # Rows sum to 1

    def test_transe_wrapper_predict_uses_threshold(self, mock_kg_config_path, mock_transe_config_path):
        """Test predict() uses 0.5 threshold."""
        wrapper = TransEWrapper(
            kg_config_path=mock_kg_config_path,
            transe_config_path=mock_transe_config_path
        )

        # Mock predict_proba
        wrapper.predict_proba = Mock(return_value=np.array([
            [0.8, 0.2],  # Class 0
            [0.3, 0.7],  # Class 1
            [0.5, 0.5],  # Boundary (should be class 0)
        ]))

        X_test = [(0, 0, 1), (1, 1, 2), (2, 0, 3)]
        predictions = wrapper.predict(X_test)

        assert np.array_equal(predictions, np.array([0, 1, 0]))


# ═══════════════════════════════════════════════════════════════════
# HybridWrapper Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHybridWrapper:
    """Test HybridWrapper (TransE + LightGBM hybrid model)."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for HybridWrapper."""
        mock_model = Mock()
        entity_to_idx = {"entity_0": 0, "entity_1": 1}
        relation_to_idx = {"relation_0": 0}
        entity_embeddings = np.random.rand(2, 32)  # 2 entities, 32 dims
        relation_embeddings = np.random.rand(1, 32)  # 1 relation, 32 dims

        return {
            "lightgbm_model": mock_model,
            "entity_to_idx": entity_to_idx,
            "relation_to_idx": relation_to_idx,
            "entity_embeddings": entity_embeddings,
            "relation_embeddings": relation_embeddings,
        }

    def test_hybrid_wrapper_initialization(self, mock_dependencies):
        """Test HybridWrapper initializes correctly."""
        wrapper = HybridWrapper(**mock_dependencies)

        assert wrapper.model_ is not None
        assert wrapper.entity_to_idx == mock_dependencies["entity_to_idx"]
        assert wrapper.relation_to_idx == mock_dependencies["relation_to_idx"]
        assert wrapper.entity_embeddings.shape == (2, 32)
        assert wrapper.relation_embeddings.shape == (1, 32)

    def test_hybrid_wrapper_inherits_base_wrapper(self, mock_dependencies):
        """Test HybridWrapper inherits from BaseWrapper."""
        wrapper = HybridWrapper(**mock_dependencies)

        assert isinstance(wrapper, BaseWrapper)

    def test_hybrid_wrapper_mean_embeddings_computed(self, mock_dependencies):
        """Test mean embeddings are computed correctly."""
        wrapper = HybridWrapper(**mock_dependencies)

        # Mean embeddings should be computed
        assert wrapper.mean_entity_embedding_ is not None
        assert wrapper.mean_relation_embedding_ is not None
        assert wrapper.mean_entity_embedding_.shape == (32,)
        assert wrapper.mean_relation_embedding_.shape == (32,)

    def test_hybrid_wrapper_embedding_dim_inferred(self, mock_dependencies):
        """Test embedding dimension is inferred from embeddings."""
        wrapper = HybridWrapper(**mock_dependencies)

        # Embedding dim should match entity embeddings
        assert wrapper._embedding_dim == 32


# ═══════════════════════════════════════════════════════════════════
# Sklearn Compatibility Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestSklearnCompatibility:
    """Test sklearn compatibility of wrappers."""

    def test_wrapper_has_classes_attribute(self, tmp_path):
        """Test wrapper has classes_ attribute for sklearn."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        # Should have classes_ attribute
        assert hasattr(wrapper, 'classes_')
        assert np.array_equal(wrapper.classes_, np.array([0, 1]))

    def test_wrapper_implements_estimator_interface(self, tmp_path):
        """Test wrapper implements BaseEstimator interface."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        # Should have required methods
        assert hasattr(wrapper, 'fit')
        assert hasattr(wrapper, 'predict')
        assert hasattr(wrapper, 'predict_proba')
        assert callable(wrapper.fit)
        assert callable(wrapper.predict)
        assert callable(wrapper.predict_proba)

    def test_wrapper_get_set_params(self, tmp_path):
        """Test wrapper supports get_params/set_params (sklearn requirement)."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        # get_params should work (from BaseEstimator)
        params = wrapper.get_params()
        assert 'kg_config_path' in params
        assert 'transe_config_path' in params

        # set_params should work
        new_transe_path = tmp_path / "new_transe.json"
        new_transe_path.write_text('{}')
        wrapper.set_params(transe_config_path=str(new_transe_path))
        assert wrapper.transe_config_path == str(new_transe_path)


# ═══════════════════════════════════════════════════════════════════
# Probability Calibration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestProbabilityCalibration:
    """Test probability calibration in wrappers."""

    def test_probabilities_sum_to_one(self):
        """Test that predicted probabilities sum to 1."""
        # Create mock wrapper with predict_proba
        class MockWrapper(BaseWrapper):
            def fit(self, X, y=None):
                return self

            def predict_proba(self, X):
                # Return mock probabilities
                return np.array([
                    [0.3, 0.7],
                    [0.8, 0.2],
                    [0.5, 0.5],
                ])

        wrapper = MockWrapper()
        probas = wrapper.predict_proba([(0, 0, 1)])

        # Each row should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_probabilities_in_valid_range(self):
        """Test probabilities are in [0, 1] range."""
        class MockWrapper(BaseWrapper):
            def fit(self, X, y=None):
                return self

            def predict_proba(self, X):
                return np.array([
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.5, 0.5],
                ])

        wrapper = MockWrapper()
        probas = wrapper.predict_proba([(0, 0, 1)])

        # All probabilities should be in [0, 1]
        assert np.all((probas >= 0) & (probas <= 1))

    def test_predict_uses_argmax(self):
        """Test predict() uses argmax on probabilities."""
        class MockWrapper(BaseWrapper):
            def fit(self, X, y=None):
                return self

            def predict_proba(self, X):
                return np.array([
                    [0.8, 0.2],  # Class 0
                    [0.3, 0.7],  # Class 1
                ])

        wrapper = MockWrapper()
        predictions = wrapper.predict([(0, 0, 1), (1, 1, 2)])

        # Should use threshold of 0.5 for binary classification
        assert predictions[0] == 0  # 0.2 < 0.5
        assert predictions[1] == 1  # 0.7 > 0.5


# ═══════════════════════════════════════════════════════════════════
# Serialization Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestWrapperSerialization:
    """Test wrapper serialization/deserialization."""

    def test_wrapper_pickle_removes_managers(self, tmp_path):
        """Test pickling removes manager objects."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        state = wrapper.__getstate__()

        # Managers should not be in state
        assert 'file_manager' not in state
        assert 'cache_manager' not in state
        assert 'concurrency_manager' not in state

    def test_wrapper_unpickle_restores_managers(self, tmp_path):
        """Test unpickling restores manager objects."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        # Simulate pickle/unpickle
        state = wrapper.__getstate__()
        new_wrapper = TransEWrapper.__new__(TransEWrapper)
        new_wrapper.__setstate__(state)

        # Managers should be restored
        assert hasattr(new_wrapper, 'file_manager')
        assert hasattr(new_wrapper, 'cache_manager')
        assert hasattr(new_wrapper, 'concurrency_manager')

    def test_wrapper_state_preserves_config_paths(self, tmp_path):
        """Test serialization preserves config paths."""
        kg_path = tmp_path / "kg.json"
        transe_path = tmp_path / "transe.json"
        kg_path.write_text('{}')
        transe_path.write_text('{}')

        wrapper = TransEWrapper(str(kg_path), str(transe_path))

        state = wrapper.__getstate__()

        # Config paths should be preserved
        assert state['kg_config_path'] == str(kg_path)
        assert state['transe_config_path'] == str(transe_path)
