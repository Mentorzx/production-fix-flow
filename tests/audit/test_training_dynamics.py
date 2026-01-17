import torch
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCModel, DSLFMKGCConfig


def test_cache_refresh_on_validation():
    """Verify that entity cache is refreshed (values change) when weights update."""

    config = DSLFMKGCConfig(
        num_entities=100,
        num_relations=2,
        entity_dim=16,
        feature_dim=16,
        hidden_dim=32,
    )
    model = DSLFMKGCModel(config)

    # 1. Precompute initially
    model.precompute_entity_latents()
    initial_features = model._all_entity_features.clone()

    # 2. Update weights (simulate training step)
    model.entity_embedding.weight.data += torch.randn_like(
        model.entity_embedding.weight.data
    )

    # 3. Trigger evaluation (should refresh cache)
    # Mocking data
    eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

    # Run evaluate with refresh_cache=True (default)
    model.evaluate(eval_triples, batch_size=1, refresh_cache=True)

    refreshed_features = model._all_entity_features

    # Check if changed
    diff = (refreshed_features - initial_features).abs().sum().item()
    print(f"\n[TEST] Cache Diff after update: {diff}")

    assert diff > 0, "Entity cache was not refreshed!"


def test_evaluate_mode_dropout():
    """Verify that evaluate() runs in eval mode (no dropout)."""

    config = DSLFMKGCConfig(
        num_entities=100,
        num_relations=2,
        entity_dim=16,
        feature_dim=16,
        hidden_dim=32,
    )
    model = DSLFMKGCModel(config)

    # Helper to check mode
    def check_mode(scores, heads, rels, tails):
        assert not model.training, "Model should be in eval mode during scoring"
        return scores

    # Run evaluate
    eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
    model.evaluate(eval_triples, batch_size=1, filter_fn=check_mode)

    print("[PASS] Evaluate runs in eval mode.")


if __name__ == "__main__":
    test_cache_refresh_on_validation()
    test_evaluate_mode_dropout()
