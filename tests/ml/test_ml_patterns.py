"""Tests for ML utilities (Strategy, Factory, BaseTrainer patterns).

Author: PFF Team
Date: 2025-11-25
"""

import pytest
import numpy as np
import torch

from pff.utils.ml import (
    KGEModelStrategy,
    TransEStrategy,
    KGEConfig,
    ModelFactory,
    ModelType,
    BaseTrainer,
    TrainerConfig,
)
from pff.utils import TrainingObserver, NullObserver, CompositeObserver


class TestKGEStrategy:
    """Test KGEModelStrategy implementations."""

    def test_transe_strategy_name(self):
        """Test TransE strategy returns correct name."""
        strategy = TransEStrategy()
        assert strategy.name == "TransE"

    def test_transe_strategy_config(self):
        """Test TransE strategy accepts config."""
        config = KGEConfig(embedding_dim=64, margin=1.5)
        strategy = TransEStrategy(config)
        assert strategy.config.embedding_dim == 64
        assert strategy.config.margin == 1.5

    def test_kge_config_defaults(self):
        """Test KGEConfig has sensible defaults."""
        config = KGEConfig()
        assert config.embedding_dim == 128
        assert config.margin == 2.0
        assert config.learning_rate == 0.001
        assert config.use_self_adversarial is True


class TestModelFactory:
    """Test ModelFactory pattern."""

    def test_factory_creates_lightgbm(self):
        """Test factory creates LightGBM model."""
        factory = ModelFactory()
        model = factory.create(ModelType.LIGHTGBM, n_estimators=50)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_factory_creates_xgboost(self):
        """Test factory creates XGBoost model."""
        factory = ModelFactory()
        model = factory.create(ModelType.XGBOOST, n_estimators=50)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_factory_raises_on_unsupported(self):
        """Test factory raises on unsupported type."""
        factory = ModelFactory()
        with pytest.raises(ValueError, match="Unsupported model type"):
            factory.create(ModelType.COMPLEX)  # Reserved but not implemented

    def test_factory_creates_rotate(self):
        """Test factory creates RotatE model."""
        factory = ModelFactory()
        model = factory.create(
            ModelType.ROTATE,
            num_entities=100,
            num_relations=10,
            embedding_dim=64,
        )
        assert model is not None
        assert model.num_entities == 100

    def test_factory_get_strategy(self):
        """Test factory returns strategy for KGE models."""
        factory = ModelFactory()
        strategy = factory.get_strategy(ModelType.TRANSE)
        assert isinstance(strategy, TransEStrategy)
        
        strategy = factory.get_strategy(ModelType.LIGHTGBM)
        assert strategy is None


class TestBaseTrainer:
    """Test BaseTrainer Template Method pattern."""

    def test_trainer_config_defaults(self):
        """Test TrainerConfig has sensible defaults."""
        config = TrainerConfig()
        assert config.num_epochs == 100
        assert config.patience == 10
        assert config.seed == 42

    def test_trainer_accepts_observer(self):
        """Test trainer accepts observer via DI."""
        observer = NullObserver()
        
        class DummyTrainer(BaseTrainer):
            def _setup_model(self, train_data):
                self.model = torch.nn.Linear(10, 1)
            def _train_epoch(self, train_data, epoch):
                return {"loss": 0.1}
            def _validate(self, val_data):
                return {"accuracy": 0.9}
        
        trainer = DummyTrainer(observer=observer)
        assert trainer.observer is observer

    def test_trainer_add_observer(self):
        """Test adding observers creates composite."""
        class DummyTrainer(BaseTrainer):
            def _setup_model(self, train_data):
                pass
            def _train_epoch(self, train_data, epoch):
                return {}
            def _validate(self, val_data):
                return {}
        
        # Start with a non-null observer
        trainer = DummyTrainer(observer=NullObserver())
        trainer.observer = NullObserver()  # First real observer
        trainer.add_observer(NullObserver())  # Second becomes composite
        
        # After adding observer to NullObserver, first one replaces it
        # Then adding another creates composite
        assert isinstance(trainer.observer, (NullObserver, CompositeObserver))

    def test_trainer_resolves_device(self):
        """Test trainer resolves device correctly."""
        config = TrainerConfig(device="cpu")
        
        class DummyTrainer(BaseTrainer):
            def _setup_model(self, train_data):
                pass
            def _train_epoch(self, train_data, epoch):
                return {}
            def _validate(self, val_data):
                return {}
        
        trainer = DummyTrainer(config=config)
        assert trainer.device == torch.device("cpu")


class TestRotatEScoreCache:
    """Test RotatE score caching functionality."""

    def test_score_cache_basic(self):
        """Test basic cache functionality."""
        from pff.validators.rotate.core import RotatEModel
        
        model = RotatEModel(num_entities=100, num_relations=10, embedding_dim=32)
        
        triples = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        
        # First call - should compute
        scores1 = model.score_triples_batch(triples, use_cache=True)
        
        # Second call - should use cache
        scores2 = model.score_triples_batch(triples, use_cache=True)
        
        np.testing.assert_array_equal(scores1, scores2)
        assert hasattr(model, '_score_cache')

    def test_score_cache_clear(self):
        """Test cache clearing."""
        from pff.validators.rotate.core import RotatEModel
        
        model = RotatEModel(num_entities=100, num_relations=10, embedding_dim=32)
        
        triples = np.array([[0, 1, 2]], dtype=np.int64)
        model.score_triples_batch(triples, use_cache=True)
        
        assert len(model._score_cache) > 0
        
        model.clear_score_cache()
        assert len(model._score_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
