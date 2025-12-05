from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from pff.validators.ensembles.advanced_trainer import AdvancedEnsembleTrainer


class _DummyEnsemble:
    def __init__(self, prob: float = 0.8) -> None:
        self.prob = prob

    def predict(self, X):  # noqa: N803
        labels = np.zeros(len(X), dtype=int)
        labels[::2] = 1
        return labels

    def predict_proba(self, X):  # noqa: N803
        probs = np.full((len(X), 2), 0.0, dtype=float)
        probs[:, 1] = self.prob
        probs[:, 0] = 1.0 - self.prob
        return probs


@dataclass
class _MockHierarchicalConfig:
    """Mock hierarchical config for testing."""
    is_hierarchical: bool = False


def test_evaluate_includes_calibration_metrics() -> None:
    trainer = AdvancedEnsembleTrainer.__new__(AdvancedEnsembleTrainer)
    trainer.ensemble_model = _DummyEnsemble(prob=0.75)
    trainer.hierarchical_config = _MockHierarchicalConfig(is_hierarchical=False)

    X = np.array([[0], [1], [2], [3]])
    y = np.array([1, 1, 0, 1])

    metrics = trainer.evaluate(X, y, prefix="test")

    assert "test_ece" in metrics
    assert "test_entropy" in metrics
    assert metrics["test_ece"] >= 0.0
    assert metrics["test_entropy"] > 0.0
