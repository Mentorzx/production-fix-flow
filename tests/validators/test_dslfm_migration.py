from __future__ import annotations

import torch

from pff.domain.learning.ml.kge_strategy import DSLFMStrategy, KGEConfig
from pff.domain.learning.dslfm.core import DSLFMModel


def test_lambda_zero_matches_base_kge_loss() -> None:
    """Test that lambda=0 produces same loss as base KGE model."""
    model = DSLFMModel(num_entities=8, num_relations=3, embedding_dim=16)
    pos = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    neg = torch.randint(0, 8, (pos.size(0), 2, 3), dtype=torch.long)

    base_loss = model.compute_kge_loss(pos, neg)

    strategy = DSLFMStrategy(
        KGEConfig(
            embedding_dim=16,
            extra={"lambda_logic": 0.0, "lambda_pc": 0.0},
        )
    )
    loss = strategy.compute_loss(model, pos, neg)

    torch.testing.assert_close(loss, base_loss, rtol=1e-5, atol=1e-6)


# Legacy AdvancedEnsembleTrainer migration tests - skipped (DSLFM+PC only)
# The migration settings were for transitioning from legacy ensembles to DSLFM+PC
# Now that migration is complete, these tests are no longer needed
