"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/performance/optimization/test_binary_metrics_overlap.py

"""

from __future__ import annotations

import numpy as np
import torch

from pff.infrastructure.hpo.trials.evaluator import _compute_binary_metrics


class _DummyModel(torch.nn.Module):
    def __init__(self, num_entities: int) -> None:
        """Execute init.



        Args:

            num_entities: Input value used by this callable.

        """

        super().__init__()
        self.num_entities = num_entities
        self.param = torch.nn.Parameter(torch.zeros(1))

    def score_triples_batch(self, triples: torch.Tensor) -> torch.Tensor:
        """Execute score triples batch.



        Args:

            triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return torch.zeros((triples.shape[0],), device=triples.device)


class _DummyManager:
    def __init__(self, num_entities: int, filter_arrays: dict[tuple[int, int], np.ndarray]):
        """Execute init.



        Args:

            num_entities: Input value used by this callable.

            filter_arrays: Input value used by this callable.

        """

        self.model = _DummyModel(num_entities)
        self._filter_arrays = filter_arrays


def test_binary_metrics_filters_known_positives(monkeypatch, tmp_path) -> None:
    """Execute test binary metrics filters known positives.



    Args:

        monkeypatch: Input value used by this callable.

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    dump_dir = tmp_path / "binary_metrics_dump"

    monkeypatch.setenv("PFF_BINARY_METRICS_DUMP_DIR", str(dump_dir))

    val_triples = np.array(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 2, 5],
        ],
        dtype=np.int64,
    )
    filter_arrays = {
        (0, 0): np.array([1], dtype=np.int64),
        (2, 1): np.array([3], dtype=np.int64),
        (4, 2): np.array([5], dtype=np.int64),
    }

    manager = _DummyManager(num_entities=1000, filter_arrays=filter_arrays)

    _compute_binary_metrics(
        manager,
        val_triples,
        num_negatives=5,
        seed=123,
        params={},
    )

    meta_files = sorted(dump_dir.glob("*.meta.json"))
    assert meta_files, "Expected binary metrics dump files to be created"

    base = meta_files[-1].with_suffix("")
    pos = np.load(base.with_suffix(".pos_triples.npy"))
    neg = np.load(base.with_suffix(".neg_triples.npy"))

    pos_view = {tuple(row) for row in pos.tolist()}
    overlap = sum(1 for row in neg.tolist() if tuple(row) in pos_view)

    assert overlap == 0
