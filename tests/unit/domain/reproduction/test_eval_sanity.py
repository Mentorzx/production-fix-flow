from unittest.mock import MagicMock

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


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

    # Mock decoder
    # Model uses self.decoder.score_all_tails and self.decoder.forward

    # Common scores for all calls
    # 20=0.9, 10=0.8, 30=0.7, 40=0.6, others=0.0
    common_scores = torch.zeros(1, num_entities)  # batch=1
    common_scores[0, 20] = 0.9
    common_scores[0, 10] = 0.8
    common_scores[0, 30] = 0.7
    common_scores[0, 40] = 0.6

    # Patch the decoder on the specific instance
    model.decoder.score_all_tails = MagicMock(return_value=common_scores)

    # Patch forward: First call for target 10 (0.8), Second call for target 50 (0.0)
    model.decoder.forward = MagicMock(side_effect=[torch.tensor([0.8]), torch.tensor([0.0])])

    # Ensure PC is disabled to simplify logic
    model.pc_model = False

    def mock_filter(scores, heads, relations, candidates, tails):
        masked = scores.clone()
        # Mock logic assuming batch=1 and we want to mask 20 and 30
        mask_20 = (candidates == 20).nonzero(as_tuple=True)
        if len(mask_20[0]) > 0:
            masked[0, mask_20[0][0]] = float("-inf")

        mask_30 = (candidates == 30).nonzero(as_tuple=True)
        if len(mask_30[0]) > 0:
            masked[0, mask_30[0][0]] = float("-inf")

        return masked

    eval_triples = torch.tensor([[0, 0, 10]], dtype=torch.long)

    metrics = model.evaluate(eval_triples, batch_size=1, refresh_cache=False, filter_fn=mock_filter)

    assert metrics["mrr"] == 1.0, f"Expected MRR 1.0, got {metrics['mrr']}"
    assert metrics["hits@1"] == 1.0

    eval_triples_2 = torch.tensor([[0, 0, 50]], dtype=torch.long)

    def mock_filter_2(scores, heads, relations, candidates, tails):
        masked = scores.clone()
        for idx in [10, 20, 30]:
            mask_idx = (candidates == idx).nonzero(as_tuple=True)
            if len(mask_idx[0]) > 0:
                masked[0, mask_idx[0][0]] = float("-inf")
        return masked

    metrics_2 = model.evaluate(
        eval_triples_2, batch_size=1, refresh_cache=False, filter_fn=mock_filter_2
    )

    assert metrics_2["mrr"] == 0.5, f"Expected MRR 0.5 (Rank 2), got {metrics_2['mrr']}"
    assert metrics_2["hits@1"] == 0.0
    assert metrics_2["hits@3"] == 1.0


def test_eval_filter_small_entities_batch_safe():
    """Ensure filtered evaluation handles batch_size > num_entities without crashing."""
    num_entities = 5
    num_relations = 2
    batch_size = 10

    config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        entity_dim=8,
        feature_dim=8,
        hidden_dim=16,
    )
    model = DSLFMKGCModel(config)
    model.eval()
    model.pc_model = False

    def _mock_score_all_tails(*_, **kwargs):
        rels = kwargs.get("relations")
        return torch.zeros((int(rels.shape[0]), num_entities))

    def _mock_forward(*_, **kwargs):
        rels = kwargs.get("relations")
        return torch.zeros((int(rels.shape[0]),))

    model.decoder.score_all_tails = MagicMock(side_effect=_mock_score_all_tails)
    model.decoder.forward = MagicMock(side_effect=_mock_forward)

    called = {"filter": False}

    def filter_fn(scores, heads, relations, candidates, tails):
        called["filter"] = True
        assert tails.shape[0] == heads.shape[0]
        # Verify access to tails works
        _ = tails[heads.shape[0] - 1].item()
        return scores

    eval_triples = torch.tensor(
        [[i % num_entities, 0, i % num_entities] for i in range(batch_size)],
        dtype=torch.long,
    )

    metrics = model.evaluate(
        eval_triples,
        batch_size=batch_size,
        refresh_cache=False,
        filter_fn=filter_fn,
    )

    assert called["filter"] is True
    assert "mrr" in metrics


if __name__ == "__main__":
    test_eval_sanity_recompute()
