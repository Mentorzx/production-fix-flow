"""Tests for DSLFM-KGC global negative sampling.

Validates that global negative tail sampling never produces the positive tail,
which prevents false negatives and removes avoidable resampling overhead.
"""

from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


def test_global_negative_tail_sampling_excludes_positive_tail() -> None:
    """Global negative tails must never equal the positive tail for each row."""
    torch.manual_seed(7)
    num_entities = 50
    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
        num_global_negatives=5,
    )
    model = DSLFMKGCModel(config)
    tails = torch.tensor([0, 1, 9, 5], dtype=torch.long)
    neg_tail_ids = model._sample_global_negative_tail_ids(  # type: ignore[attr-defined]
        tails,
        num_entities=num_entities,
        num_negatives=7,
    )

    assert neg_tail_ids.shape == (len(tails), 7)
    assert torch.all(neg_tail_ids >= 0)
    assert torch.all(neg_tail_ids < num_entities)
    assert torch.all(neg_tail_ids.ne(tails.unsqueeze(1)))


def test_global_negative_tail_sampling_rejects_invalid_num_entities() -> None:
    """Sampling must fail fast when the entity universe is too small."""
    config = DSLFMKGCConfig(
        num_entities=1,
        num_relations=1,
        entity_dim=4,
        feature_dim=4,
        max_communities=2,
    )
    model = DSLFMKGCModel(config)
    tails = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError, match="num_entities must be > 1"):
        model._sample_global_negative_tail_ids(  # type: ignore[attr-defined]
            tails,
            num_entities=config.num_entities,
            num_negatives=1,
        )
