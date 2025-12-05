from __future__ import annotations

import numpy as np
import pytest
import torch

from pff.validators.rotate.manager import RotatEManager


class _DummyModel:
    def __init__(self, num_entities: int, score_batches: list[torch.Tensor]) -> None:
        self.num_entities = num_entities
        self._score_batches = score_batches
        self._call_index = 0

    def eval(self) -> "_DummyModel":
        return self

    def forward(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        scores = self._score_batches[self._call_index]
        self._call_index += 1
        return scores.reshape(-1)


def test_validate_computes_hits3_and_mean_rank() -> None:
    """RotatEManager._validate deve retornar hits@3 e mean_rank."""
    # Two validation triples with deterministic ranking
    val_triples = np.array(
        [
            [0, 0, 1],  # True tail index = 1 (best)
            [1, 0, 2],  # True tail index = 2 (rank 3)
        ],
        dtype=np.int64,
    )

    score_grid = torch.tensor(
        [
            [0.10, 0.90, 0.20, 0.30],  # rank = 1
            [0.90, 0.80, 0.70, 0.60],  # rank = 3 (two scores above true)
        ],
        dtype=torch.float32,
    )

    manager = object.__new__(RotatEManager)
    manager.model = _DummyModel(num_entities=4, score_batches=[score_grid])
    manager.device = torch.device("cpu")
    manager.current_epoch = 0

    metrics = manager._validate(val_triples)

    assert metrics["hits@1"] == pytest.approx(0.5)
    assert metrics["hits@3"] == pytest.approx(1.0)
    assert metrics["hits@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics["mean_rank"] == pytest.approx(2.0)
