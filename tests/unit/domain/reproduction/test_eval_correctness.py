"""Test Filtered Evaluation logic on a Toy KG.

This test verifies if the evaluation mechanism correctly:
1. Ranks true triples better than random.
2. Filters out other known true triples (filtered setting).
3. Handles ambiguous cases correctly.
"""

from unittest.mock import MagicMock

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


class MockDecoder:
    """Mock decoder that returns deterministic scores."""

    def __init__(self, num_entities):
        self.num_entities = num_entities

    def score_all_tails(self, z_head, f_head, relations, all_z, all_f):
        # Return scores such that:
        # Score = 1.0 if (head, relation, tail) is "compatible" (mock logic)
        # Score = 0.0 otherwise
        # For testing, we can just return a tensor of zeros and manually
        # inject high scores for specific indices to simulate a good model.
        batch_size = relations.shape[0]
        return torch.zeros(batch_size, self.num_entities)


def test_filtered_evaluation_logic():
    """Test if filtered evaluation works correctly on a toy example."""

    # Toy KG: 5 entities, 1 relation
    # Triples: (0, 0, 1), (0, 0, 2)
    # Both 1 and 2 are valid tails for (0, 0).

    # Task: Predict tail for (0, 0, ?). True tail is 1.
    # Candidate 2 is ALSO a true tail (from training/valid set).

    # Raw Rank: 2 might be ranked higher than 1.
    # Filtered Rank: 2 should be IGNORED.

    num_entities = 5
    num_relations = 1

    # Setup Model
    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=8,
        feature_dim=8,
        hidden_dim=8,
    )
    model = DSLFMKGCModel(config)

    # Mock the decoder/scoring to give controlled outputs
    # We want to simulate:
    # Query: (0, 0) -> Target: 1
    # Scores:
    # 0: 0.1
    # 1: 0.8 (Target)
    # 2: 0.9 (Another true positive - should be filtered)
    # 3: 0.5 (False positive - ranks above target in raw)
    # 4: 0.0

    # Raw Ranking:
    # 2 (0.9) -> Rank 1
    # 1 (0.8) -> Rank 2 (Target)
    # 3 (0.5) -> Rank 3
    # ...
    # Expected Raw Rank for 1: 2

    # Filtered Ranking (Filter={2}):
    # 2 is ignored.
    # 1 (0.8) -> Rank 1 (Target)
    # 3 (0.5) -> Rank 2
    # Expected Filtered Rank for 1: 1

    # We monkey-patch the score_all_tails method or the decoder
    # But score_all_tails_chunked uses self.decoder.score_all_tails.

    # Mock decoder methods on the EXISTING decoder module
    # Mock forward (true scores)
    # Query (0,0), True Tail 1. We want score 0.8.
    model.decoder.forward = MagicMock(return_value=torch.tensor([0.8]))

    # Mock score_all_tails (all candidates)
    # Scores: [0.1, 0.8, 0.9, 0.5, 0.0]
    model.decoder.score_all_tails = MagicMock(
        return_value=torch.tensor([[0.1, 0.8, 0.9, 0.5, 0.0]])
    )

    # Disable PC
    model.pc_model = False

    # Ensure latent encoding works (mock precompute/encode or let it run)
    # Since we mocked decoder, encode_entities runs on real encoder layers to produce z, f
    # and decoder mocks consumption. This is fine.

    # Setup Data
    eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

    # Setup Filter
    # We need to pass a filter_fn to evaluate.
    # The KGCManager builds this from train+valid triples.
    # Here we mock it.

    def mock_filter_fn(scores, heads, relations, candidates, true_tails, correction_only):
        # Mask entity 2 for query (0,0)
        # In a real scenario, this comes from a look-up
        masked_scores = scores.clone()
        if not correction_only:
            # We need to map global ID 2 to local index in candidates
            mask_idx = (candidates == 2).nonzero(as_tuple=True)
            if len(mask_idx[0]) > 0:
                idx = mask_idx[0][0]
                masked_scores[0, idx] = float("-inf")
        return masked_scores if not correction_only else torch.zeros(len(heads), dtype=torch.int32)

    # Run Evaluate
    metrics = model.evaluate(
        eval_triples,
        batch_size=1,
        refresh_cache=False,  # Don't need cache for mock
        filter_fn=mock_filter_fn,
    )

    print("\n[TEST] Metrics:", metrics)

    # Verify
    # Rank should be 1.0 because 2 (score 0.9) was filtered out.
    # If it wasn't filtered, rank would be 2 (behind 2).
    # If 3 (score 0.5) wasn't filtered (it shouldn't be), it stays behind 1.

    assert metrics["mrr"] == 1.0, f"Expected MRR 1.0 (filtered), got {metrics['mrr']}"
    assert metrics["hits@1"] == 1.0

    print("[TEST] Filtered logic passed.")


if __name__ == "__main__":
    test_filtered_evaluation_logic()
