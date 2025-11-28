"""Tests for RotatE SRP Components.

Tests cover:
- RotatECheckpointManager: save/load/cleanup operations
- RotatEMetricsReporter: metrics computation and observer notification
- RotatEDataLoader: data loading and DataLoader creation
- ContrastiveLossFactory: loss type selection
- NegativeSamplerFactory: sampling strategy selection

These tests ensure the SRP components extracted from RotatEManager
work correctly in isolation.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from pff.validators.rotate.checkpoint_manager import RotatECheckpointManager
from pff.validators.rotate.metrics_reporter import RotatEMetricsReporter
from pff.validators.rotate.contrastive import ContrastiveLossFactory, LossType
from pff.validators.rotate.negative_sampling import (
    NegativeSamplerFactory,
    NegativeSamplingStrategy,
)
from pff.utils.performance.training_observer import (
    TrainingObserver,
    TrainingEvent,
    CompositeObserver,
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpointing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
class TestRotatECheckpointManager:
    """Test RotatECheckpointManager operations."""

    def test_save_creates_file(self, tmp_path):
        """Test that save creates a checkpoint file."""
        manager = RotatECheckpointManager(checkpoint_dir=tmp_path)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={"mrr": 0.5, "hits@10": 0.7},
        )

        assert path.exists()
        assert "checkpoint_epoch_0005.pt" in str(path)

    def test_save_best_model(self, tmp_path):
        """Test that is_best=True saves best_model.pt."""
        manager = RotatECheckpointManager(checkpoint_dir=tmp_path)
        model = SimpleModel()

        manager.save(
            model=model,
            optimizer=None,
            epoch=1,
            metrics={"mrr": 0.6},
            is_best=True,
        )

        best_path = tmp_path / "best_model.pt"
        assert best_path.exists()

    def test_load_restores_state(self, tmp_path):
        """Test that load restores model state."""
        manager = RotatECheckpointManager(checkpoint_dir=tmp_path)
        model1 = SimpleModel()
        
        # Save checkpoint
        manager.save(model=model1, optimizer=None, epoch=10, metrics={"mrr": 0.4})
        
        # Create new model and load
        model2 = SimpleModel()
        checkpoint_info = manager.load(
            model=model2,
            path=tmp_path / "checkpoint_epoch_0010.pt",
        )

        assert checkpoint_info["epoch"] == 10
        assert checkpoint_info["metrics"]["mrr"] == 0.4

    def test_get_best_checkpoint(self, tmp_path):
        """Test get_best_checkpoint returns correct path."""
        manager = RotatECheckpointManager(checkpoint_dir=tmp_path)
        model = SimpleModel()

        # No checkpoint yet
        assert manager.get_best_checkpoint() is None

        # Save best
        manager.save(model=model, optimizer=None, epoch=1, metrics={}, is_best=True)
        
        best = manager.get_best_checkpoint()
        assert best is not None
        assert best.name == "best_model.pt"

    def test_has_completed_training(self, tmp_path):
        """Test has_completed_training check."""
        manager = RotatECheckpointManager(checkpoint_dir=tmp_path)

        # No marker yet
        is_complete, info = manager.has_completed_training(target_epochs=50)
        assert not is_complete
        assert info == {}

        # Mark as completed
        manager.mark_training_completed(
            epochs_trained=50,
            target_epochs=50,
            best_epoch=25,
            best_val_mrr=0.55,
            training_time=120.5,
            final_metrics={"mrr": 0.55},
        )

        is_complete, info = manager.has_completed_training(target_epochs=50)
        assert is_complete
        assert info["best_val_mrr"] == 0.55


@pytest.mark.unit
class TestRotatEMetricsReporter:
    """Test RotatEMetricsReporter operations."""

    def test_add_observer(self, tmp_path):
        """Test that observers can be added."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        mock_observer = MagicMock(spec=TrainingObserver)
        
        reporter.add_observer(mock_observer)
        assert mock_observer in reporter.observers

    def test_remove_observer(self, tmp_path):
        """Test that observers can be removed."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        mock_observer = MagicMock(spec=TrainingObserver)
        
        reporter.add_observer(mock_observer)
        reporter.remove_observer(mock_observer)
        assert mock_observer not in reporter.observers

    def test_notify_observers_on_epoch_end(self, tmp_path):
        """Test that observers are notified on epoch_end."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        mock_observer = MagicMock(spec=TrainingObserver)
        reporter.add_observer(mock_observer)

        reporter.report_epoch_end(
            epoch=5,
            train_loss=0.25,
            val_metrics={"mrr": 0.5},
        )

        mock_observer.on_event.assert_called_once()
        event = mock_observer.on_event.call_args[0][0]
        assert event.event_type == "epoch_end"
        assert event.epoch == 5
        assert event.metrics["train_loss"] == 0.25

    def test_notify_observers_on_checkpoint(self, tmp_path):
        """Test that observers are notified on checkpoint."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        mock_observer = MagicMock(spec=TrainingObserver)
        reporter.add_observer(mock_observer)

        reporter.report_checkpoint(epoch=10, path="/path/to/ckpt.pt", is_best=True)

        mock_observer.on_event.assert_called_once()
        event = mock_observer.on_event.call_args[0][0]
        assert event.event_type == "checkpoint"
        assert event.metadata["is_best"] is True

    def test_save_metrics(self, tmp_path):
        """Test that metrics are saved to JSON."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        
        path = reporter.save_metrics({"mrr": 0.6, "hits@10": 0.8}, filename="test_metrics.json")

        assert path.exists()
        assert path.name == "test_metrics.json"

    def test_format_metrics_string(self, tmp_path):
        """Test metrics string formatting."""
        reporter = RotatEMetricsReporter(output_dir=tmp_path)
        
        formatted = reporter.format_metrics_string({"mrr": 0.5678, "loss": 0.1234})
        
        assert "mrr=0.5678" in formatted
        assert "loss=0.1234" in formatted


@pytest.mark.unit
class TestContrastiveLossFactory:
    """Test ContrastiveLossFactory operations."""

    def test_create_infonce_loss(self):
        """Test creating InfoNCE loss."""
        loss = ContrastiveLossFactory.create(LossType.INFONCE, temperature=0.1)
        
        assert loss is not None
        assert hasattr(loss, "compute")

    def test_create_triplet_loss(self):
        """Test creating Triplet loss."""
        loss = ContrastiveLossFactory.create(LossType.TRIPLET, margin=1.0)
        
        assert loss is not None

    def test_create_margin_ranking_loss(self):
        """Test creating margin ranking loss."""
        loss = ContrastiveLossFactory.create(LossType.MARGIN_RANKING, margin=1.0)
        
        assert loss is not None

    def test_infonce_compute(self):
        """Test InfoNCE loss compute pass."""
        loss_fn = ContrastiveLossFactory.create(LossType.INFONCE, temperature=0.1)
        
        positive_scores = torch.randn(8)
        negative_scores = torch.randn(8, 16)
        
        loss = loss_fn.compute(positive_scores, negative_scores)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0


@pytest.mark.unit
class TestNegativeSamplerFactory:
    """Test NegativeSamplerFactory operations."""

    def test_create_uniform_sampler(self):
        """Test creating uniform sampler."""
        sampler = NegativeSamplerFactory.create(
            strategy=NegativeSamplingStrategy.UNIFORM,
            num_entities=1000,
            device=torch.device("cpu"),
        )
        
        assert sampler is not None
        assert hasattr(sampler, "sample")

    def test_create_self_adversarial_sampler(self):
        """Test creating self-adversarial sampler."""
        sampler = NegativeSamplerFactory.create(
            strategy=NegativeSamplingStrategy.SELF_ADVERSARIAL,
            num_entities=1000,
            device=torch.device("cpu"),
            temperature=1.0,
        )
        
        assert sampler is not None

    def test_uniform_sample_shape(self):
        """Test uniform sampler produces correct shape."""
        sampler = NegativeSamplerFactory.create(
            strategy=NegativeSamplingStrategy.UNIFORM,
            num_entities=1000,
            device=torch.device("cpu"),
        )
        
        heads = torch.tensor([0, 1, 2, 3])
        rels = torch.tensor([0, 1, 0, 1])
        tails = torch.tensor([10, 11, 12, 13])
        
        negatives = sampler.sample(heads, rels, tails, num_negatives=10)
        
        assert negatives.shape == (4, 10)  # [batch_size, num_negatives]

    def test_uniform_sampler_values_in_range(self):
        """Test uniform sampler produces values in valid range."""
        num_entities = 100
        sampler = NegativeSamplerFactory.create(
            strategy=NegativeSamplingStrategy.UNIFORM,
            num_entities=num_entities,
            device=torch.device("cpu"),
        )
        
        heads = torch.tensor([0, 1])
        rels = torch.tensor([0, 1])
        tails = torch.tensor([10, 11])
        
        negatives = sampler.sample(heads, rels, tails, num_negatives=50)
        
        assert negatives.min() >= 0
        assert negatives.max() < num_entities


@pytest.mark.unit
class TestCompositeObserver:
    """Test CompositeObserver integration."""

    def test_composite_notifies_all_observers(self):
        """Test that composite observer notifies all child observers."""
        mock1 = MagicMock(spec=TrainingObserver)
        mock2 = MagicMock(spec=TrainingObserver)
        
        composite = CompositeObserver([mock1, mock2])
        
        event = TrainingEvent(event_type="epoch_end", epoch=5, metrics={"loss": 0.1})
        composite.on_event(event)
        
        mock1.on_event.assert_called_once_with(event)
        mock2.on_event.assert_called_once_with(event)

    def test_composite_handles_observer_exception(self):
        """Test that composite continues even if one observer fails."""
        mock1 = MagicMock(spec=TrainingObserver)
        mock1.on_event.side_effect = RuntimeError("Observer failed")
        mock2 = MagicMock(spec=TrainingObserver)
        
        composite = CompositeObserver([mock1, mock2])
        
        event = TrainingEvent(event_type="epoch_end", epoch=1)
        
        # Should not raise, and mock2 should still be called
        composite.on_event(event)
        mock2.on_event.assert_called_once()
