"""Tests for Domain/Range Aware Negative Sampling in DSLFM.

SOTA: Domain/Range Aware Negative Sampling (2024)
Verifies that negatives are sampled from valid domain/range when provided.
"""

from __future__ import annotations

import numpy as np
import torch

from pff.domain.learning.dslfm.backbone import DSLFMDataset


class TestDomainRangeAwareNegativeSampling:
    """Test suite for Domain/Range Aware Negative Sampling."""

    def test_backward_compatibility_without_domain_range(self) -> None:
        """Dataset should work without domain/range (backward compatible)."""
        triples = np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64)
        dataset = DSLFMDataset(triples, num_entities=10, num_negatives=5)

        batch = dataset[0]
        assert "positive" in batch
        assert "negatives" in batch
        assert batch["negatives"].shape == (5, 3)

    def test_domain_range_aware_sampling_tail_corruption(self) -> None:
        """Corrupted tails should be sampled from relation's range when provided."""
        # Relation 0 has range {10, 11, 12}
        triples = np.array([[0, 0, 10], [1, 0, 11], [2, 0, 12]], dtype=np.int64)
        domain = {0: np.array([0, 1, 2])}
        range_ = {0: np.array([10, 11, 12])}

        dataset = DSLFMDataset(
            triples,
            num_entities=100,
            num_negatives=20,
            seed=42,
            relation_domain=domain,
            relation_range=range_,
        )

        batch = dataset[0]  # Triple [0, 0, 10]
        negatives = batch["negatives"]

        # All negatives should have relation 0
        assert torch.all(negatives[:, 1] == 0)

        # Check corrupted tails are in range {10, 11, 12}
        # and corrupted heads are in domain {0, 1, 2}
        for neg in negatives:
            h, r, t = neg.tolist()
            # Either head was corrupted (and is in domain) or tail was corrupted (and is in range)
            head_corrupted = h != 0  # Original head was 0
            tail_corrupted = t != 10  # Original tail was 10

            if head_corrupted:
                assert h in [0, 1, 2], f"Corrupted head {h} not in domain"
            if tail_corrupted:
                assert t in [10, 11, 12], f"Corrupted tail {t} not in range"

    def test_fallback_to_uniform_when_domain_empty(self) -> None:
        """Should fall back to uniform sampling when domain/range not provided for relation."""
        triples = np.array([[0, 5, 1]], dtype=np.int64)  # Relation 5 not in domain dict
        domain = {0: np.array([0, 1, 2])}  # Only relation 0 has domain
        range_ = {0: np.array([10, 11, 12])}

        dataset = DSLFMDataset(
            triples,
            num_entities=100,
            num_negatives=10,
            seed=42,
            relation_domain=domain,
            relation_range=range_,
        )

        batch = dataset[0]
        negatives = batch["negatives"]

        # Should not crash and negatives should be valid
        assert negatives.shape == (10, 3)
        assert torch.all(negatives[:, 1] == 5)  # Relation unchanged

    def test_small_domain_range_uses_available_entities(self) -> None:
        """With small domain/range, sampling should use available entities."""
        # Relation 0: domain={5}, range={6} (only 1 entity each - edge case)
        triples = np.array([[5, 0, 6]], dtype=np.int64)
        domain = {0: np.array([5])}
        range_ = {0: np.array([6])}

        dataset = DSLFMDataset(
            triples,
            num_entities=100,
            num_negatives=10,
            seed=42,
            relation_domain=domain,
            relation_range=range_,
        )

        batch = dataset[0]
        negatives = batch["negatives"]

        # With only 1 entity in domain/range, should fall back to uniform
        # (our implementation requires len > 1 to use domain/range)
        assert negatives.shape == (10, 3)

    def test_negatives_never_equal_positive(self) -> None:
        """Critical: Negatives must NEVER be identical to positive triple.

        This was a bug that caused learning stagnation - when negatives could
        be identical to positives, the model couldn't learn to distinguish them.
        """
        # Create triples where domain/range could sample the positive
        triples = np.array([[5, 0, 10], [5, 0, 11], [6, 0, 10]], dtype=np.int64)
        domain = {0: np.array([5, 6])}  # Head 5 is in domain
        range_ = {0: np.array([10, 11])}  # Tail 10 is in range

        dataset = DSLFMDataset(
            triples,
            num_entities=100,
            num_negatives=100,  # High count to increase chance of collision
            seed=42,
            relation_domain=domain,
            relation_range=range_,
        )

        # Test multiple samples to ensure robustness
        for idx in range(len(triples)):
            batch = dataset[idx]
            positive = batch["positive"]
            negatives = batch["negatives"]

            pos_h, pos_r, pos_t = positive.tolist()

            for neg in negatives:
                neg_h, neg_r, neg_t = neg.tolist()
                # Negative should NEVER be identical to positive
                identical = neg_h == pos_h and neg_r == pos_r and neg_t == pos_t
                assert (
                    not identical
                ), f"Negative {neg.tolist()} is identical to positive {positive.tolist()}"
