import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


def test_scoring_consistency_bilinear_and_dot_product():
    """Verify decoder.score_all_tails matches decoder.forward() for both scoring modes.

    Tests that the pairwise scoring (forward) and all-tails scoring (score_all_tails)
    produce consistent results for the same head-relation-tail combinations.
    """

    test_configs = [(64, "Bilinear"), (256, "Dot Product")]

    for feat_dim, mode in test_configs:
        print(f"\n[TEST] Testing consistency for {mode} mode (dim={feat_dim})...")
        config = DSLFMKGCConfig(
            num_entities=100,
            num_relations=5,
            entity_dim=feat_dim,
            feature_dim=feat_dim,
            hidden_dim=feat_dim * 2,
        )
        model = DSLFMKGCModel(config)
        model.eval()

        # Verify mode assumption
        if mode == "Bilinear":
            assert model.decoder.use_bilinear, "Expected bilinear mode"
        else:
            assert not model.decoder.use_bilinear, "Expected dot product mode"

        # Mock data
        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 0])
        tails = torch.tensor([2, 3])  # True tails

        with torch.no_grad():
            # 1. Get latents for heads and tails
            head_latents = model.encode_entities(heads)
            tail_latents = model.encode_entities(tails)

            # 2. Forward scores (pairwise: specific head-tail pairs)
            scores_forward = model.decoder.forward(
                z_head=head_latents["communities"],
                z_tail=tail_latents["communities"],
                f_head=head_latents["features"],
                f_tail=tail_latents["features"],
                relations=relations,
            )  # [2]

            # 3. Get all entity latents for score_all_tails
            all_entity_ids = torch.arange(config.num_entities)
            all_latents = model.encode_entities(all_entity_ids)

            # 4. Score all tails
            scores_all = model.decoder.score_all_tails(
                z_head=head_latents["communities"],
                f_head=head_latents["features"],
                relations=relations,
                all_z=all_latents["communities"],
                all_f=all_latents["features"],
            )  # [2, 100]

        # Extract scores for true tails from scores_all
        # Batch 0: tail 2
        score_0_all_tails = scores_all[0, 2]
        # Batch 1: tail 3
        score_1_all_tails = scores_all[1, 3]

        print(f"  Forward Score 0: {scores_forward[0].item():.6f}")
        print(f"  All-tails Score 0: {score_0_all_tails.item():.6f}")

        # Using atol=1e-5 for float precision tolerance
        assert torch.isclose(
            scores_forward[0], score_0_all_tails, atol=1e-5
        ), f"Mismatch in {mode}: {scores_forward[0]} vs {score_0_all_tails}"
        assert torch.isclose(
            scores_forward[1], score_1_all_tails, atol=1e-5
        ), f"Mismatch in {mode}: {scores_forward[1]} vs {score_1_all_tails}"
        print(f"  [PASS] {mode} consistency verified.")


if __name__ == "__main__":
    test_scoring_consistency_bilinear_and_dot_product()
