import numpy as np
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


def test_toy_kg_perfect_eval():
    """Verify that a perfect model gets MRR=1.0 in filtered evaluation."""

    num_entities = 5
    num_relations = 1

    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=8,
        feature_dim=8,
        hidden_dim=8,
    )
    model = DSLFMKGCModel(config)

    def mock_score_all_tails(z_head, f_head, relations, all_z, all_f):
        return torch.tensor([[0.1, 0.8, 0.9, 0.5, 0.0]])

    model.decoder.score_all_tails = mock_score_all_tails

    def mock_filter(scores, heads, relations, candidates, true_tails, correction_only):
        masked = scores.clone()
        if not correction_only:
            # Mask entity 2 (which is better than true tail 1 in this mock)
            # Find index of entity 2 in candidates
            idx = (candidates == 2).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                masked[0, idx[0]] = float("-inf")
        return masked if not correction_only else torch.zeros(len(heads), dtype=torch.int32)

    eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

    metrics = model.evaluate(eval_triples, batch_size=1, refresh_cache=True, filter_fn=mock_filter)

    print(f"\n[TEST] Perfect Toy Metrics: {metrics}")
    assert metrics["mrr"] == 1.0
    assert metrics["hits@1"] == 1.0


def test_random_baseline_sanity():
    """Verify that random scores produce expected random baseline metrics.

    This test runs actual model evaluation with random scores to verify
    that the ranking mechanism produces statistically expected metrics
    for a random baseline (MRR ~ 1/N, Hits@k ~ k/N).
    """

    N = 1000
    num_queries = 2000

    torch.manual_seed(123)
    np.random.seed(123)

    config = DSLFMKGCConfig(
        num_entities=N, num_relations=1, entity_dim=8, feature_dim=8, hidden_dim=8
    )
    model = DSLFMKGCModel(config)
    model.eval()

    # Store original decoder methods
    original_score_all_tails = model.decoder.score_all_tails
    original_forward = model.decoder.forward

    # Mock both forward and score_all_tails to return random scores
    def mock_random_forward(z_head, z_tail, f_head, f_tail, relations):
        """Random pairwise scores for true triples."""
        batch_size = z_head.shape[0]
        return torch.rand(batch_size)

    def mock_random_score_all_tails(z_head, f_head, relations, all_z, all_f):
        """Random scores for all-tails evaluation.

        Returns [batch_size, num_tails] tensor where num_tails is the chunk size.
        """
        batch_size = z_head.shape[0]
        num_tails = all_z.shape[0]  # This is the chunk size, not total entities
        return torch.rand(batch_size, num_tails)

    model.decoder.forward = mock_random_forward
    model.decoder.score_all_tails = mock_random_score_all_tails

    # Create evaluation triples: (head=0, rel=0, tail=random)
    eval_triples = torch.zeros((num_queries, 3), dtype=torch.long)
    eval_triples[:, 2] = torch.randint(0, N, (num_queries,))

    # Run evaluation with refresh_cache to ensure latents are computed
    metrics = model.evaluate(eval_triples, batch_size=50, refresh_cache=True)

    print(f"\n[TEST] Random Baseline (N={N}): {metrics}")

    # For random scores:
    # - Expected MRR ~ 1/N * sum(1/k for k in 1..N) ~ ln(N)/N ~ 0.007 for N=1000
    # - Expected Hits@10 ~ 10/N = 0.01 for N=1000
    # We use wider bounds due to statistical variance
    assert 0.0 < metrics["mrr"] < 0.02, f"Random MRR {metrics['mrr']} out of expected range ~0.007"
    assert 0.0 < metrics["hits@10"] < 0.03, (
        f"Random Hits@10 {metrics['hits@10']} out of expected range ~0.01"
    )

    # Restore original methods
    model.decoder.score_all_tails = original_score_all_tails
    model.decoder.forward = original_forward


if __name__ == "__main__":
    test_toy_kg_perfect_eval()
    test_random_baseline_sanity()
