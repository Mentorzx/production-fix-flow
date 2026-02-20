"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kgc_vectorized_ops.py

"""

from __future__ import annotations

import numpy as np
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    """Represent MockPersistencePort."""

    def save_checkpoint(self, data, filename):
        """Execute save checkpoint.



        Args:

            data: Input value used by this callable.

            filename: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pass

    def load_checkpoint(self, filename, map_location=None):
        """Execute load checkpoint.



        Args:

            filename: Input value used by this callable.

            map_location: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return None


def test_vectorized_mask_equivalence() -> None:
    """Regression test: Ensures vectorized packing matches exhaustive filtering."""
    device = torch.device("cpu")
    model_config = DSLFMKGCConfig(num_entities=100, num_relations=10)
    train_config = KGCTrainingConfig(epochs=1, batch_size=4)

    manager = DSLFMKGCManager(
        model_config,
        train_config,
        persistence_port=MockPersistencePort(),
        device=device,
    )

    # Setup: 3 triples, where (0,0) has two tails [1, 2]
    train_triples = np.array([[0, 0, 1], [0, 0, 2], [5, 1, 6]], dtype=np.int64)
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    # Query: head=0, rel=0, true_tail=1. Candidates=[0,1,2,3]
    h = torch.tensor([0])
    r = torch.tensor([0])
    t = torch.tensor([1])
    candidates = torch.arange(4)
    scores = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)

    # Execute vectorized mask
    masked = manager._mask_known_tails(scores.clone(), h, r, candidates, t)

    # Expectations:
    # tail 2 is a known positive for (0,0) -> MUST be -inf
    # tail 1 is the CURRENT true tail -> MUST NOT be -inf (to allow rank calculation)
    # tail 0 and 3 are not known positives -> MUST be original score
    assert masked[0, 2] == float("-inf")
    assert masked[0, 1] == 20.0
    assert masked[0, 0] == 10.0
    assert masked[0, 3] == 40.0


def test_inbatch_positive_mask_equivalence() -> None:
    """Regression test: Ensures packed lookup matches exhaustive in-batch mask."""
    device = torch.device("cpu")
    model_config = DSLFMKGCConfig(num_entities=100, num_relations=10)
    train_config = KGCTrainingConfig(epochs=1, batch_size=2)
    manager = DSLFMKGCManager(
        model_config,
        train_config,
        persistence_port=MockPersistencePort(),
        device=device,
    )

    # Triple (0,0,1) exists in KG
    train_triples = np.array([[0, 0, 1]], dtype=np.int64)
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    # Batch where triple 0 is (0,0,x) and triple 1 is (x,x,1)
    # If triple 1's tail (1) is a valid tail for triple 0's (h,r) (0,0), it should be masked.
    h = torch.tensor([0, 9])
    r = torch.tensor([0, 8])
    t = torch.tensor([5, 1])

    mask = manager._build_inbatch_known_positive_mask(h, r, t)

    # mask[row, col] -> row 0 (0,0), col 1 (tail 1). (0,0,1) is known.
    assert mask[0, 1]
    assert not mask[0, 0]
    assert not mask[1, 0]
    assert not mask[1, 1]
