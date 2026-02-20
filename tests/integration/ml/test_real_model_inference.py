"""Integration tests for DSLFM Model using real inference (no mocks)."""

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


@pytest.mark.integration
def test_real_model_evaluation_flow():
    """
    Test the full evaluation flow with a real initialized model.
    This ensures that shapes, broadcasting, and metric calculations
    work correctly with actual tensors, not just mocks.
    """
    # 1. Setup minimal config
    config = DSLFMKGCConfig(
        num_entities=50,
        num_relations=5,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
        use_bert_relations=False,
        stochastic_latents=False,
    )

    model = DSLFMKGCModel(config=config)
    model.eval()

    # 2. Create synthetic test triples
    # (h, r, t)
    triples = torch.tensor(
        [[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 5]], dtype=torch.long
    )

    # 3. Running evaluation
    # This calls model.forward(), model.decoder.score_all_tails(), etc.
    with torch.no_grad():
        metrics = model.evaluate(
            eval_triples=triples,
            batch_size=5,
            filter_fn=None,
        )

    # 4. Verification
    # Check keys exist
    assert "mrr" in metrics
    assert "hits@1" in metrics
    assert "hits@10" in metrics

    # Check value ranges
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["hits@1"] <= 1.0
    assert 0.0 <= metrics["hits@10"] <= 1.0

    # Check consistency
    assert metrics["hits@1"] <= metrics["hits@10"]
    assert metrics["hits@1"] <= metrics["mrr"]
