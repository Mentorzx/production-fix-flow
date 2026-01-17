import torch
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCModel, DSLFMKGCConfig


def test_eval_sanity_recompute():
    """Recompute filtered MRR/Hits@K for a small sample and verify against model output."""

    # Setup toy KG
    num_entities = 100
    num_relations = 5
    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=16,
        feature_dim=16,
        hidden_dim=32,
    )
    model = DSLFMKGCModel(config)
    model.eval()

    # Create mock scores to control ranking
    # Query: (0, 0, ?) -> True tail: 10
    # Positives to filter: {10, 20, 30}
    # We want 10 to be ranked. 20 and 30 should be filtered.

    # We set scores such that:
    # 20 (score 0.9) > 10 (score 0.8) > 30 (score 0.7) > 40 (score 0.6) ...
    # Raw Rank of 10: 2 (behind 20)
    # Filtered Rank of 10: 1 (20 is filtered, 30 is filtered)

    def mock_score_chunked(heads, relations, batch_size=None):
        scores = torch.zeros(len(heads), num_entities)
        # Set scores for the single query
        scores[0, 20] = 0.9
        scores[0, 10] = 0.8
        scores[0, 30] = 0.7
        scores[0, 40] = 0.6
        return scores

    model.score_all_tails_chunked = mock_score_chunked

    # Filter function masking 20 and 30
    def mock_filter(scores, heads, relations, tails):
        masked = scores.clone()
        # tails contains true tail (10).
        # We must filter OTHER positives (20, 30).
        # In real code, filter_dict provides this.
        masked[0, 20] = float("-inf")
        masked[0, 30] = float("-inf")
        return masked

    eval_triples = torch.tensor([[0, 0, 10]], dtype=torch.long)

    # Run evaluation
    metrics = model.evaluate(
        eval_triples, batch_size=1, refresh_cache=False, filter_fn=mock_filter
    )

    print(f"\n[TEST] Metrics: {metrics}")

    # Verification
    # Rank should be 1.0 because 20 is filtered.
    assert metrics["mrr"] == 1.0, f"Expected MRR 1.0, got {metrics['mrr']}"
    assert metrics["hits@1"] == 1.0

    # Now test scenario where target is NOT top 1
    # 40 (score 0.6) > 50 (score 0.5) ...
    # Let's make target 50 (score 0.5). 40 is negative (not filtered).
    # Filter: {20, 30} (positives)
    # Scores: 20(0.9), 10(0.8), 30(0.7), 40(0.6), 50(0.5)
    # Filtered Scores: 10(0.8), 40(0.6), 50(0.5) ... (20,30 removed)
    # Wait, if target is 50.
    # Sorted: 10, 40, 50.
    # Rank of 50 is 3.

    # Update mock for second case
    eval_triples_2 = torch.tensor([[0, 0, 50]], dtype=torch.long)

    # We need to update mock filter to NOT filter 10 (assume 10 is negative now? No, filter is static set of positives)
    # Assume 10, 20, 30 are positives. Target is 50 (which must be a positive too for eval).
    # So 10, 20, 30 are "other" positives.

    def mock_filter_2(scores, heads, relations, tails):
        masked = scores.clone()
        masked[0, 10] = float("-inf")
        masked[0, 20] = float("-inf")
        masked[0, 30] = float("-inf")
        return masked

    metrics_2 = model.evaluate(
        eval_triples_2, batch_size=1, refresh_cache=False, filter_fn=mock_filter_2
    )

    print(f"[TEST] Metrics 2 (Target 50): {metrics_2}")

    # Expected Rank:
    # Scores: 40 (0.6) > 50 (0.5).
    # Rank 2.
    assert metrics_2["mrr"] == 0.5, f"Expected MRR 0.5 (Rank 2), got {metrics_2['mrr']}"
    assert metrics_2["hits@1"] == 0.0
    assert metrics_2["hits@3"] == 1.0


if __name__ == "__main__":
    test_eval_sanity_recompute()
