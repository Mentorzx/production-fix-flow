"""Test that validation cache is refreshed correctly.

This test verifies that bug C (stale validation cache) is fixed.
The default config should have refresh_cache_on_val=True.
"""

from __future__ import annotations

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import KGCTrainingConfig


def test_validation_uses_current_embeddings_default() -> None:
    """Default config should refresh cache on validation."""
    config = KGCTrainingConfig()
    assert (
        config.refresh_cache_on_val is True
    ), "Default must be True to prevent stale embeddings during validation"


def test_cache_cleared_on_refresh() -> None:
    """Verify that cache is actually invalidated when refresh_cache=True."""
    torch.manual_seed(42)

    model_config = DSLFMKGCConfig(
        num_entities=6,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
    )
    KGCTrainingConfig(
        epochs=1,
        batch_size=2,
        effective_batch_size=2,
        checkpoint_dir=torch.device("cpu"),  # type: ignore
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        refresh_cache_on_val=True,
    )

    # We only test the model, not the full manager training loop
    from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCModel

    model = DSLFMKGCModel(model_config)
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

    # First evaluation with cache
    model.evaluate(triples, batch_size=2, refresh_cache=True)
    cached_features_1 = model._all_entity_features.detach().clone()  # type: ignore[union-attr]
    cached_communities_1 = model._all_entity_communities.detach().clone()  # type: ignore[union-attr]

    # Modify weights
    with torch.no_grad():
        model.entity_embedding.weight.normal_(mean=1.0, std=0.5)

    # Second evaluation with refresh - should see different metrics
    model.evaluate(triples, batch_size=2, refresh_cache=True)
    cached_features_2 = model._all_entity_features.detach()  # type: ignore[union-attr]
    cached_communities_2 = model._all_entity_communities.detach()  # type: ignore[union-attr]

    assert not torch.allclose(cached_features_1, cached_features_2)
    assert not torch.allclose(cached_communities_1, cached_communities_2)


def test_refresh_cache_flag_is_configurable() -> None:
    """Verify that refresh_cache_on_val can be set to False if needed."""
    config_true = KGCTrainingConfig(refresh_cache_on_val=True)
    config_false = KGCTrainingConfig(refresh_cache_on_val=False)

    assert config_true.refresh_cache_on_val is True
    assert config_false.refresh_cache_on_val is False
