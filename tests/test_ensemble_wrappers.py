"""
Testes baseline para ensemble_wrappers.py antes da refatoração Sprint 4.

Objetivo: Garantir que a refatoração NÃO quebre a compatibilidade.

Foco em:
1. Contract Testing - API pública não muda
2. Determinismo - mesma entrada → mesma saída
3. Response Format - estrutura consistente
4. sklearn compatibility - BaseEstimator, ClassifierMixin, TransformerMixin

Arquivo dependente identificado:
- pff/validators/ensembles/baseline_comparison.py
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch


# ════════════════════════════════════════════════════════════════════════
# CONTRACT TESTING - API Pública
# ════════════════════════════════════════════════════════════════════════


class TestEnsembleWrappersContract:
    """Testa que a API pública não muda (contract testing)."""

    def test_module_exists_and_importable(self):
        """Verifica que o módulo pode ser importado."""
        from pff.validators.ensembles import ensemble_wrappers

        assert ensemble_wrappers is not None

    def test_all_classes_exist(self):
        """Verifica que todas as classes públicas existem."""
        from pff.validators.ensembles.ensemble_wrappers import (
            BaseWrapper,
            TransEWrapper,
            HybridWrapper,
            ProbaTransformer,
            SymbolicFeatureExtractor,
        )

        assert BaseWrapper is not None
        assert TransEWrapper is not None
        assert HybridWrapper is not None
        assert ProbaTransformer is not None
        assert SymbolicFeatureExtractor is not None

    def test_base_wrapper_has_required_methods(self):
        """Verifica que BaseWrapper tem métodos abstratos."""
        from pff.validators.ensembles.ensemble_wrappers import BaseWrapper

        # BaseWrapper é abstrata, não pode ser instanciada diretamente
        with pytest.raises(TypeError):
            BaseWrapper()

    def test_transe_wrapper_signature(self):
        """Verifica assinatura do construtor TransEWrapper."""
        from pff.validators.ensembles.ensemble_wrappers import TransEWrapper
        import inspect

        sig = inspect.signature(TransEWrapper.__init__)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "kg_config_path" in params
        assert "transe_config_path" in params

    def test_hybrid_wrapper_signature(self):
        """Verifica assinatura do construtor HybridWrapper."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper
        import inspect

        sig = inspect.signature(HybridWrapper.__init__)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "lightgbm_model" in params
        assert "entity_to_idx" in params
        assert "relation_to_idx" in params
        assert "entity_embeddings" in params
        assert "relation_embeddings" in params

    def test_proba_transformer_signature(self):
        """Verifica assinatura do construtor ProbaTransformer."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer
        import inspect

        sig = inspect.signature(ProbaTransformer.__init__)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "model" in params

    def test_symbolic_feature_extractor_signature(self):
        """Verifica assinatura do construtor SymbolicFeatureExtractor."""
        from pff.validators.ensembles.ensemble_wrappers import (
            SymbolicFeatureExtractor,
        )
        import inspect

        sig = inspect.signature(SymbolicFeatureExtractor.__init__)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "rules_path" in params


# ════════════════════════════════════════════════════════════════════════
# SKLEARN COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════


class TestSklearnCompatibility:
    """Testa compatibilidade com sklearn BaseEstimator."""

    def test_hybrid_wrapper_is_base_estimator(self):
        """HybridWrapper deve herdar de BaseEstimator."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper
        from sklearn.base import BaseEstimator

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((10, 100)),
            relation_embeddings=np.zeros((5, 100)),
        )

        assert isinstance(wrapper, BaseEstimator)

    def test_hybrid_wrapper_is_classifier(self):
        """HybridWrapper deve herdar de ClassifierMixin."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper
        from sklearn.base import ClassifierMixin

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((10, 100)),
            relation_embeddings=np.zeros((5, 100)),
        )

        assert isinstance(wrapper, ClassifierMixin)

    def test_proba_transformer_is_transformer(self):
        """ProbaTransformer deve herdar de TransformerMixin."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer
        from sklearn.base import TransformerMixin

        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.8, 0.2]]))
        transformer = ProbaTransformer(model=mock_model)

        assert isinstance(transformer, TransformerMixin)

    def test_hybrid_wrapper_has_classes_attribute(self):
        """Wrapper deve ter atributo classes_ (sklearn requirement)."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((10, 100)),
            relation_embeddings=np.zeros((5, 100)),
        )

        assert hasattr(wrapper, "classes_")
        assert np.array_equal(wrapper.classes_, np.array([0, 1]))

    def test_hybrid_wrapper_fit_returns_self(self):
        """fit() deve retornar self (sklearn requirement)."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((10, 100)),
            relation_embeddings=np.zeros((5, 100)),
        )

        result = wrapper.fit(X=[], y=None)
        assert result is wrapper

    def test_proba_transformer_fit_returns_self(self):
        """fit() deve retornar self (sklearn requirement)."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer

        mock_model = Mock()
        transformer = ProbaTransformer(model=mock_model)

        result = transformer.fit(X=[], y=None)
        assert result is transformer


# ════════════════════════════════════════════════════════════════════════
# DETERMINISMO - Mesma entrada → Mesma saída
# ════════════════════════════════════════════════════════════════════════


class TestEnsembleDeterminism:
    """Testa que mesma entrada produz mesma saída."""

    def test_hybrid_wrapper_predict_proba_deterministic(self):
        """predict_proba deve retornar mesmos valores para mesma entrada."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        # Mock LightGBM model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.7, 0.3, 0.9]))
        mock_model.num_feature = Mock(return_value=300)  # Mock for feature count

        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={"entity1": 0, "entity2": 1},
            relation_to_idx={"rel1": 0},
            entity_embeddings=np.random.rand(2, 100),
            relation_embeddings=np.random.rand(1, 100),
        )
        wrapper.fit(X=[], y=None)

        # Mesma entrada
        X_test = [("entity1", "rel1", "entity2")]

        result1 = wrapper.predict_proba(X_test)
        result2 = wrapper.predict_proba(X_test)

        assert np.array_equal(result1, result2), "predict_proba deve ser determinístico"

    def test_proba_transformer_transform_deterministic(self):
        """transform() deve retornar mesmos valores para mesma entrada."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer

        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.8, 0.2]]))

        transformer = ProbaTransformer(model=mock_model)
        transformer.fit(X=[], y=None)

        X_test = ["sample1", "sample2"]

        result1 = transformer.transform(X_test)
        result2 = transformer.transform(X_test)

        assert np.array_equal(result1, result2), "transform deve ser determinístico"


# ════════════════════════════════════════════════════════════════════════
# RESPONSE FORMAT - Estrutura consistente
# ════════════════════════════════════════════════════════════════════════


class TestResponseFormat:
    """Testa que respostas têm formato correto."""

    def test_predict_proba_returns_2d_array(self):
        """predict_proba deve retornar array 2D com shape (n_samples, 2)."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.7, 0.3]))
        mock_model.num_feature = Mock(return_value=300)  # Mock for feature count

        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={"e1": 0},
            relation_to_idx={"r1": 0},
            entity_embeddings=np.zeros((1, 100)),
            relation_embeddings=np.zeros((1, 100)),
        )
        wrapper.fit(X=[], y=None)

        result = wrapper.predict_proba([("e1", "r1", "e1"), ("e1", "r1", "e1")])

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2  # [prob_class_0, prob_class_1]

    def test_predict_returns_1d_array(self):
        """predict deve retornar array 1D com valores 0 ou 1."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.7, 0.3]))
        mock_model.num_feature = Mock(return_value=300)  # Mock for feature count

        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={"e1": 0},
            relation_to_idx={"r1": 0},
            entity_embeddings=np.zeros((1, 100)),
            relation_embeddings=np.zeros((1, 100)),
        )
        wrapper.fit(X=[], y=None)

        result = wrapper.predict([("e1", "r1", "e1"), ("e1", "r1", "e1")])

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert all(val in [0, 1] for val in result)

    def test_proba_transformer_output_shape(self):
        """transform deve retornar array com shape (n_samples, 1)."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer

        mock_model = Mock()
        mock_model.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
        )

        transformer = ProbaTransformer(model=mock_model)
        transformer.fit(X=[], y=None)

        result = transformer.transform(["s1", "s2", "s3"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)  # (n_samples, 1)

    def test_get_feature_names_out_returns_list(self):
        """get_feature_names_out deve retornar lista de strings."""
        from pff.validators.ensembles.ensemble_wrappers import ProbaTransformer

        mock_model = Mock()
        mock_model.__class__.__name__ = "MockModel"
        transformer = ProbaTransformer(model=mock_model)

        result = transformer.get_feature_names_out()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)


# ════════════════════════════════════════════════════════════════════════
# SERIALIZATION - Pickle compatibility
# ════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Testa que wrappers podem ser serializados/deserializados."""

    def test_hybrid_wrapper_getstate(self):
        """__getstate__ deve remover objetos não-picklable."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((1, 100)),
            relation_embeddings=np.zeros((1, 100)),
        )

        state = wrapper.__getstate__()

        # Deve remover objetos não-picklable
        assert "concurrency_manager" not in state
        assert "cache_manager" not in state
        assert "file_manager" not in state

    def test_hybrid_wrapper_setstate(self):
        """__setstate__ deve restaurar objetos não-picklable."""
        from pff.validators.ensembles.ensemble_wrappers import HybridWrapper

        mock_model = Mock()
        wrapper = HybridWrapper(
            lightgbm_model=mock_model,
            entity_to_idx={},
            relation_to_idx={},
            entity_embeddings=np.zeros((1, 100)),
            relation_embeddings=np.zeros((1, 100)),
        )

        # Simula pickle/unpickle
        state = wrapper.__getstate__()
        new_wrapper = HybridWrapper.__new__(HybridWrapper)
        new_wrapper.__setstate__(state)

        # Deve restaurar objetos
        assert hasattr(new_wrapper, "file_manager")
        assert hasattr(new_wrapper, "cache_manager")
        assert hasattr(new_wrapper, "concurrency_manager")


# ════════════════════════════════════════════════════════════════════════
# INTEGRATION - Dependent files
# ════════════════════════════════════════════════════════════════════════


class TestDependentFiles:
    """Testa que arquivos dependentes podem importar corretamente."""

    def test_baseline_comparison_can_import(self):
        """baseline_comparison.py deve conseguir importar."""
        try:
            from pff.validators.ensembles import baseline_comparison

            assert baseline_comparison is not None
        except ImportError as e:
            pytest.fail(f"baseline_comparison.py falhou ao importar: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
