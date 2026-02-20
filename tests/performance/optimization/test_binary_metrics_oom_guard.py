"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_binary_metrics_oom_guard.py

"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pff.infrastructure.hpo.trials import evaluator


class _DummyScoringModel(torch.nn.Module):
    def __init__(self) -> None:
        """Execute init."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 1, bias=False)

    def score_triples_batch(self, triples: torch.Tensor) -> torch.Tensor:
        """Execute score triples batch.



        Args:

            triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        values = triples.to(torch.float32)
        return self.linear(values).squeeze(-1)


def test_binary_metrics_batches_scores(monkeypatch) -> None:
    """Execute test binary metrics batches scores.



    Args:

        monkeypatch: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    model = _DummyScoringModel()
    model.num_entities = 10
    manager = SimpleNamespace(model=model)
    val_triples = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64)
    calls: list[int] = []

    def _counting_scores(triples: torch.Tensor) -> torch.Tensor:
        """Execute counting scores.



        Args:

            triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        calls.append(int(triples.shape[0]))
        values = triples.to(torch.float32)
        return model.linear(values).squeeze(-1)

    model.score_triples_batch = _counting_scores  # type: ignore[method-assign]

    def _fake_config(file_manager=None):  # noqa: ARG001
        return {
            "binary_metrics": {
                "enabled": True,
                "num_negatives": 2,
                "max_samples": 3,
                "batch_size": 1,
                "device": "cpu",
                "cuda_free_ratio_min": 0.0,
            }
        }

    monkeypatch.setattr(
        "pff.infrastructure.hpo.config_loader.load_optimization_config",
        _fake_config,
    )

    metrics = evaluator._compute_binary_metrics(
        manager,
        val_triples,
        num_negatives=2,
        seed=123,
        params={"binary_metrics_batch_size": 1},
    )

    assert calls
    assert all(size == 1 for size in calls)
    assert isinstance(metrics, dict)
