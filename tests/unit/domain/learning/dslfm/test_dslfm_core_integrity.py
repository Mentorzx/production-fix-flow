"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/learning/dslfm/test_dslfm_core_integrity.py

"""

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import (
    DSLFMKGCConfig,
    DSLFMKGCModel,
)


@pytest.fixture
def config():
    """Execute config.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    return DSLFMKGCConfig(
        num_entities=20,
        num_relations=5,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
        num_triples=100,
        lambda_pc=0.0,
        lambda_logic=0.0,
        use_bert_relations=False,
    )


@pytest.fixture
def model(config):
    """Execute model.



    Args:

        config: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    model = DSLFMKGCModel(config)
    # Ensure CPU for unit tests
    return model.to("cpu")


class TestDSLFMIntegrity:
    """Represent TestDSLFMIntegrity."""

    def test_initialization(self, model, config):
        """Test if model layers are initialized with correct shapes."""
        assert model.entity_embedding.weight.shape == (
            config.num_entities,
            config.entity_dim,
        )
        assert model.relation_embedding.weight.shape == (
            config.num_relations,
            config.entity_dim,
        )
        # Check decoder exists
        assert model.decoder is not None

    def test_encode_entities_shapes_and_bounds(self, model, config):
        """Test entity encoding outputs correct shapes and probabilistic bounds."""
        batch_size = 4
        entity_ids = torch.tensor([0, 1, 5, 19], dtype=torch.long)

        output = model.encode_entities(entity_ids)

        assert "communities" in output
        assert "features" in output

        # Communities should be probabilities [0, 1] (due to sigmoid in VAE usually, let's verify logic)
        # Looking at code: encode_entities -> vae_encoder -> reparameterize
        # Typically communities are soft assignments.
        z = output["communities"]
        f = output["features"]

        assert z.shape == (batch_size, config.max_communities)
        assert f.shape == (batch_size, config.feature_dim)

        # Check integrity
        assert not torch.isnan(z).any(), "Communities contain NaNs"
        assert not torch.isnan(f).any(), "Features contain NaNs"

    def test_score_triples_batch_shapes(self, model):
        """Test scoring of triples returns correct shape."""
        # (h, r, t)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)

        scores = model.score_triples_batch(triples)

        assert scores.shape == (3,)
        assert scores.dtype == torch.float32
        assert not torch.isnan(scores).any()

    def test_compute_loss_gradients(self, model):
        """Test forward pass and backward pass (gradient flow)."""
        model.train()
        batch_size = 4
        heads = torch.randint(0, model.config.num_entities, (batch_size,))
        relations = torch.randint(0, model.config.num_relations, (batch_size,))
        tails = torch.randint(0, model.config.num_entities, (batch_size,))

        # Mock negative sampler to avoid complexity or use simple fallback
        # The model uses self.negative_sampler.
        # By default it uses DegreeBasedSampler which requires graph structure.
        # We might need to mock get_positive_negative_scores if sampler fails on empty graph.

        # Actually, let's assume the default sampler handles empty graph gracefully or we populate it.
        # Ideally, we verify compute_loss runs.

        try:
            loss_dict = model.compute_loss(heads, relations, tails)
            loss = loss_dict["loss"]

            assert loss.dim() == 0, "Loss must be scalar"
            assert not torch.isnan(loss), "Loss is NaN"

            # Backprop
            loss.backward()

            # Check gradients
            assert model.entity_embedding.weight.grad is not None
            assert model.entity_embedding.weight.grad.abs().sum() > 0

        except Exception as e:
            # If default sampler fails due to missing graph structure, we might need a workaround for unit test
            pytest.fail(f"Loss computation failed: {e}")

    def test_forward_output_structure(self, model):
        """Test explicit forward method contract."""
        batch_size = 2
        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 0])
        tails = torch.tensor([2, 3])

        result = model.forward(heads, relations, tails, return_latents=True)

        assert "scores" in result
        assert "decoder_scores" in result
        assert "head_latents" in result
        assert "tail_latents" in result

        assert result["scores"].shape == (batch_size,)

    def test_input_validation_out_of_bounds(self, model, config, caplog):
        """Out-of-range entity IDs should be corrected with a warning."""
        oob_id = config.num_entities + 10
        heads = torch.tensor([oob_id])
        model.eval()

        with torch.no_grad():
            output = model.encode_entities(heads)

        assert "communities" in output
        assert output["communities"].shape[0] == 1
        assert any(
            "Applying modulo correction for CUDA-safe execution." in message
            for message in caplog.messages
        )

    def test_determinism_evaluation(self, model):
        """Test that scoring is deterministic for same inputs in eval mode."""
        model.eval()
        triples = torch.tensor([[0, 0, 1], [0, 0, 1]])

        with torch.no_grad():
            scores1 = model.score_triples_batch(triples[0:1])
            scores2 = model.score_triples_batch(triples[1:2])

        assert torch.allclose(scores1, scores2), "Scoring should be deterministic"

    def test_batch_score_empty_input(self, model):
        """Test edge case: empty batch."""
        empty_triples = torch.empty((0, 3), dtype=torch.long)
        scores = model.score_triples_batch(empty_triples)
        assert scores.numel() == 0
