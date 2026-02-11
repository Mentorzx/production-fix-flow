"""Tests for binary metric computation in DSLFM/PC HPO evaluator."""

import builtins
from types import SimpleNamespace

import numpy as np
import torch

from pff.domain.learning.ml.training_observer import TrainingEvent
from pff.infrastructure.hpo.trials.evaluator import (
    BinaryMetricsObserver,
    _compute_binary_metrics,
)


class DummyModel(torch.nn.Module):
    """Minimal scoring model with base_model namespace lacking scoring methods."""

    def __init__(self) -> None:
        super().__init__()
        self.num_entities = 3
        self.config = SimpleNamespace(num_entities=3)
        # base_model exists but has no score_triples_batch, exercising fallback
        self.base_model = SimpleNamespace()
        # Register a parameter so parameters() yields a device
        self.register_parameter("weight", torch.nn.Parameter(torch.tensor(0.0)))

    def score_triples_batch(self, triples: torch.Tensor) -> torch.Tensor:
        """Simple separable score: higher when head==tail."""
        return (triples[:, 0] == triples[:, 2]).float()


class DummyManager:
    """Manager exposing the model attribute expected by the evaluator."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model


def test_binary_metrics_are_computed_when_base_model_lacks_scoring() -> None:
    """Metrics should be produced even if base_model has no score_triples_batch."""
    model = DummyModel()
    manager = DummyManager(model)
    # Two positives with head==tail, one harder case (head!=tail)
    val_triples = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 1]], dtype=np.int64)

    metrics = _compute_binary_metrics(manager, val_triples, num_negatives=2, seed=0)

    assert metrics, "Binary metrics must not be empty"
    # Ensure core metrics are present and within valid ranges
    assert 0.0 <= metrics.get("auc", 0.0) <= 1.0
    assert 0.0 <= metrics.get("pr_auc", 0.0) <= 1.0
    assert "precision" in metrics and "recall" in metrics


def test_binary_metrics_observer_updates_event_metrics() -> None:
    """The observer should calculate and inject binary metrics into the event metrics dict."""
    model = DummyModel()
    manager = DummyManager(model)
    val_triples = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int64)

    observer = BinaryMetricsObserver(manager, val_triples, params={"binary_metrics_enabled": True})

    metrics = {"loss": 0.5, "mrr": 0.2}
    event = TrainingEvent(event_type="epoch_end", epoch=1, metrics=metrics)

    observer.on_event(event)

    assert "mcc" in metrics, "MCC should be injected into the metrics dictionary"
    assert "auc" in metrics, "AUC should be injected into the metrics dictionary"
    assert metrics["loss"] == 0.5, "Original metrics should be preserved"


def test_binary_metrics_fallback_when_accel_import_fails(monkeypatch) -> None:
    """Fallback to sklearn metrics when accelerated imports raise non-ImportError."""
    model = DummyModel()
    manager = DummyManager(model)
    val_triples = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 1]], dtype=np.int64)

    original_import = builtins.__import__

    def _broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pff_rust":
            raise RuntimeError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    metrics = _compute_binary_metrics(manager, val_triples, num_negatives=2, seed=0)

    assert metrics, "Fallback metrics should be computed when accel import fails"
