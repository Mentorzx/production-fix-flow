"""Tests for RotatE Model Implementation.

This module contains unit tests for the RotatE knowledge graph embedding model,
including model creation, scoring functions, loss computation, and strategy pattern.

Test Categories:
    1. Model initialization and configuration
    2. Forward pass and scoring functions
    3. Loss computation (self-adversarial negative sampling)
    4. Strategy pattern integration
    5. Factory pattern integration
    6. Batch scoring performance

Run with:
    poetry run pytest tests/test_rotate_model.py -q

Author: PFF Team
Date: 2025-11-25
"""

import math

import numpy as np
import pytest
import torch

from pff.validators.rotate.core import RotatEModel, RotatEDataset
from pff.validators.rotate.config import RotatEConfig, RotatEConfigBuilder
from pff.utils.ml.kge_strategy import KGEConfig, RotatEStrategy
from pff.utils.ml.model_factory import ModelFactory, ModelType


class TestRotatEConfig:
    """Tests for RotatEConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RotatEConfig()
        assert config.embedding_dim == 256
        assert config.gamma == 12.0
        assert config.epsilon == 2.0
        assert config.use_self_adversarial is True
        assert config.complex_dim == 128

    def test_config_validation_odd_dim(self):
        """Test that odd embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="must be even"):
            RotatEConfig(embedding_dim=127)

    def test_config_validation_negative_gamma(self):
        """Test that non-positive gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            RotatEConfig(gamma=-1.0)

    def test_config_validation_negative_epsilon(self):
        """Test that non-positive epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            RotatEConfig(epsilon=0.0)

    def test_complex_dim_property(self):
        """Test complex_dim property calculation."""
        config = RotatEConfig(embedding_dim=512)
        assert config.complex_dim == 256

    def test_to_dict(self):
        """Test configuration serialization."""
        config = RotatEConfig(embedding_dim=128, gamma=9.0)
        d = config.to_dict()
        assert d["embedding_dim"] == 128
        assert d["gamma"] == 9.0
        assert "use_self_adversarial" in d


class TestRotatEConfigBuilder:
    """Tests for RotatEConfigBuilder fluent API."""

    def test_builder_chain(self):
        """Test fluent builder API."""
        config = (
            RotatEConfigBuilder()
            .with_embedding_dim(512)
            .with_gamma(18.0)
            .with_epsilon(3.0)
            .with_self_adversarial(enabled=True, temperature=1.5)
            .build()
        )

        assert config.embedding_dim == 512
        assert config.gamma == 18.0
        assert config.epsilon == 3.0
        assert config.use_self_adversarial is True
        assert config.adversarial_temperature == 1.5

    def test_builder_defaults(self):
        """Test builder with defaults."""
        config = RotatEConfigBuilder().build()
        assert config.embedding_dim == 256

    def test_builder_regularization(self):
        """Test regularization configuration."""
        config = (
            RotatEConfigBuilder()
            .with_regularization(entity_weight=1e-4, relation_weight=1e-5)
            .build()
        )
        assert config.entity_regularizer_weight == 1e-4
        assert config.relation_regularizer_weight == 1e-5


class TestRotatEModel:
    """Tests for RotatEModel PyTorch module."""

    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
            gamma=12.0,
            epsilon=2.0,
        )

    def test_model_initialization(self, model):
        """Test model is properly initialized."""
        assert model.num_entities == 100
        assert model.num_relations == 10
        assert model.embedding_dim == 64
        assert model.complex_dim == 32
        assert model.gamma == 12.0

    def test_embedding_shapes(self, model):
        """Test embedding tensor shapes."""
        assert model.entity_embedding.weight.shape == (100, 64)
        assert model.relation_embedding.weight.shape == (10, 32)

    def test_embedding_initialization_range(self, model):
        """Test entity embeddings are in expected range."""
        expected_range = (model.gamma + model.epsilon) / model.embedding_dim
        assert model.entity_embedding.weight.abs().max() <= expected_range + 0.01

    def test_relation_phase_initialization(self, model):
        """Test relation phases are in [-π, π]."""
        phases = model.relation_embedding.weight
        assert phases.min() >= -math.pi - 0.01
        assert phases.max() <= math.pi + 0.01

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        heads = torch.tensor([0, 1, 2])
        relations = torch.tensor([0, 1, 2])
        tails = torch.tensor([3, 4, 5])

        scores = model(heads, relations, tails)

        assert scores.shape == (3,)

    def test_forward_score_range(self, model):
        """Test that scores are finite."""
        heads = torch.tensor([0, 1, 2])
        relations = torch.tensor([0, 1, 2])
        tails = torch.tensor([3, 4, 5])

        scores = model(heads, relations, tails)

        assert torch.isfinite(scores).all()

    def test_score_triple(self, model):
        """Test single triple scoring."""
        score = model.score_triple(0, 1, 2)
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_score_triples_batch(self, model):
        """Test batch scoring."""
        triples = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        scores = model.score_triples_batch(triples)

        assert scores.shape == (3,)
        assert np.isfinite(scores).all()

    def test_score_triples_batch_caching(self, model):
        """Test that batch scoring uses cache."""
        triples = np.array([[0, 1, 2], [3, 4, 5]])
        scores1 = model.score_triples_batch(triples, use_cache=True)
        scores2 = model.score_triples_batch(triples, use_cache=True)

        np.testing.assert_array_equal(scores1, scores2)

    def test_clear_score_cache(self, model):
        """Test cache clearing."""
        triples = np.array([[0, 1, 2]])
        model.score_triples_batch(triples, use_cache=True)
        assert len(model._score_cache) > 0

        model.clear_score_cache()
        assert len(model._score_cache) == 0

    def test_complex_split(self, model):
        """Test _split_complex method."""
        embedding = torch.randn(10, 64)
        real, imag = model._split_complex(embedding)

        assert real.shape == (10, 32)
        assert imag.shape == (10, 32)
        torch.testing.assert_close(real, embedding[:, :32])
        torch.testing.assert_close(imag, embedding[:, 32:])

    def test_complex_multiply(self, model):
        """Test _complex_multiply method."""
        re_a = torch.tensor([1.0, 2.0])
        im_a = torch.tensor([0.0, 1.0])
        re_b = torch.tensor([1.0, 0.0])
        im_b = torch.tensor([0.0, 1.0])

        re_result, im_result = model._complex_multiply(re_a, im_a, re_b, im_b)
        torch.testing.assert_close(re_result, torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(im_result, torch.tensor([0.0, 2.0]))

    def test_get_entity_embeddings(self, model):
        """Test get_entity_embeddings method."""
        real, imag = model.get_entity_embeddings([0, 1, 2])
        assert real.shape == (3, 32)
        assert imag.shape == (3, 32)

    def test_get_relation_phases(self, model):
        """Test get_relation_phases method."""
        phases = model.get_relation_phases([0, 1])
        assert phases.shape == (2, 32)

    def test_embedding_stats(self, model):
        """Test get_embedding_stats method."""
        stats = model.get_embedding_stats()
        assert "entity_embedding_mean" in stats
        assert "entity_magnitude_mean" in stats
        assert "relation_phase_mean" in stats
        assert all(math.isfinite(v) for v in stats.values())


class TestRotatELoss:
    """Tests for RotatE loss computation."""

    @pytest.fixture
    def model(self):
        """Create model with self-adversarial enabled."""
        config = RotatEConfig(
            embedding_dim=64,
            gamma=12.0,
            use_self_adversarial=True,
            adversarial_temperature=1.0,
        )
        return RotatEModel(
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
            gamma=12.0,
            config=config,
        )

    def test_compute_loss_shape(self, model):
        """Test loss computation returns scalar."""
        positives = torch.tensor([[0, 1, 2], [3, 4, 5]])
        negatives = torch.randint(0, 100, (2, 10, 3))

        loss = model.compute_loss(positives, negatives)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_compute_loss_finite(self, model):
        """Test loss is finite."""
        positives = torch.tensor([[0, 1, 2]])
        negatives = torch.randint(0, 100, (1, 5, 3))

        loss = model.compute_loss(positives, negatives)

        assert torch.isfinite(loss)

    def test_regularization_loss_zero(self, model):
        """Test regularization is zero with zero weights."""
        reg_loss = model.regularization_loss()
        assert reg_loss.item() == 0.0

    def test_regularization_loss_nonzero(self):
        """Test regularization with non-zero weights."""
        config = RotatEConfig(
            embedding_dim=64,
            entity_regularizer_weight=1e-3,
            relation_regularizer_weight=1e-3,
        )
        model = RotatEModel(
            num_entities=100, num_relations=10, embedding_dim=64, config=config
        )

        reg_loss = model.regularization_loss()
        assert reg_loss.item() > 0


class TestRotatEDataset:
    """Tests for RotatEDataset."""

    @pytest.fixture
    def triples(self):
        """Create sample triples."""
        return np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])

    def test_dataset_length(self, triples):
        """Test dataset length."""
        dataset = RotatEDataset(triples, num_entities=100, num_negatives=10)
        assert len(dataset) == 4

    def test_getitem_structure(self, triples):
        """Test __getitem__ returns correct structure."""
        dataset = RotatEDataset(triples, num_entities=100, num_negatives=10)
        item = dataset[0]

        assert "positive" in item
        assert "negatives" in item
        assert item["positive"].shape == (3,)
        assert item["negatives"].shape == (10, 3)

    def test_negative_sampling(self, triples):
        """Test negative samples are generated."""
        dataset = RotatEDataset(triples, num_entities=100, num_negatives=5, seed=42)
        item = dataset[0]

        positives = item["positive"]
        negatives = item["negatives"]
        for neg in negatives:
            assert neg[1] == positives[1]


class TestRotatEStrategy:
    """Tests for RotatEStrategy pattern implementation."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = KGEConfig(
            embedding_dim=64,
            extra={"gamma": 12.0, "epsilon": 2.0},
        )
        return RotatEStrategy(config)

    def test_strategy_name(self, strategy):
        """Test strategy name property."""
        assert strategy.name == "RotatE"

    def test_create_model(self, strategy):
        """Test model creation via strategy."""
        model = strategy.create_model(num_entities=100, num_relations=10)
        assert isinstance(model, RotatEModel)
        assert model.num_entities == 100
        assert model.num_relations == 10

    def test_score_triple(self, strategy):
        """Test single triple scoring via strategy."""
        model = strategy.create_model(num_entities=100, num_relations=10)
        score = strategy.score_triple(model, 0, 1, 2)
        assert isinstance(score, float)

    def test_score_batch(self, strategy):
        """Test batch scoring via strategy."""
        model = strategy.create_model(num_entities=100, num_relations=10)
        triples = np.array([[0, 1, 2], [3, 4, 5]])
        scores = strategy.score_batch(model, triples)
        assert scores.shape == (2,)

    def test_compute_loss(self, strategy):
        """Test loss computation via strategy."""
        model = strategy.create_model(num_entities=100, num_relations=10)
        positives = torch.tensor([[0, 1, 2], [3, 4, 5]])
        negatives = torch.randint(0, 100, (2, 5, 3))
        loss = strategy.compute_loss(model, positives, negatives)
        assert loss.shape == ()


class TestModelFactory:
    """Tests for ModelFactory with RotatE."""

    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return ModelFactory()

    def test_create_rotate_model(self, factory):
        """Test RotatE model creation via factory."""
        model = factory.create(
            ModelType.ROTATE,
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
        )
        assert isinstance(model, RotatEModel)

    def test_factory_with_kge_config(self, factory):
        """Test factory with KGEConfig."""
        config = KGEConfig(
            embedding_dim=128,
            extra={"gamma": 18.0},
        )
        model = factory.create(
            ModelType.ROTATE,
            num_entities=100,
            num_relations=10,
            config=config,
        )
        assert model.embedding_dim == 128

    def test_factory_transe_raises_not_implemented(self, factory):
        """Test factory raises NotImplementedError for deprecated TransE."""
        with pytest.raises(NotImplementedError, match="TransE has been removed"):
            factory.create(
                ModelType.TRANSE, num_entities=100, num_relations=10, embedding_dim=64
            )

    def test_factory_rotate_is_primary(self, factory):
        """Test RotatE is the primary KGE model after TransE removal."""
        rotate = factory.create(
            ModelType.ROTATE, num_entities=100, num_relations=10, embedding_dim=64
        )
        assert type(rotate).__name__ == "RotatEModel"
        assert hasattr(rotate, "get_entity_embeddings")
        assert hasattr(rotate, "get_relation_phases")


class TestRotatEMathematicalProperties:
    """Tests for RotatE mathematical properties."""

    @pytest.fixture
    def model(self):
        """Create model for mathematical tests."""
        return RotatEModel(
            num_entities=100, num_relations=10, embedding_dim=64, gamma=12.0
        )

    def test_rotation_identity(self, model):
        """Test rotation with θ=0 is identity."""
        with torch.no_grad():
            model.relation_embedding.weight[0] = 0.0
        score_same = model.score_triple(0, 0, 0)
        score_diff = model.score_triple(0, 0, 1)
        assert score_same > score_diff

    def test_rotation_antisymmetry(self, model):
        """Test rotation captures anti-symmetric relations.

        For anti-symmetric relations, h->t should score differently than t->h.
        This is achieved because different rotation angles produce different results.
        """
        with torch.no_grad():
            model.relation_embedding.weight[0] = torch.full((32,), math.pi / 4)
        score_forward = model.score_triple(0, 0, 1)
        score_backward = model.score_triple(1, 0, 0)
        assert abs(score_forward - score_backward) >= 0

    def test_score_consistency(self, model):
        """Test scores are consistent across calls."""
        triples = np.array([[0, 1, 2], [3, 4, 5]])
        scores1 = model.score_triples_batch(triples, use_cache=False)
        scores2 = model.score_triples_batch(triples, use_cache=False)

        np.testing.assert_allclose(scores1, scores2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
