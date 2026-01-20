"""Gradient flow tests for DSLFM + PC components."""

from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


@pytest.fixture(autouse=True)
def _disable_cuda(monkeypatch) -> None:
    """Evita warnings de CUDA em ambientes CPU-only."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=False)
    monkeypatch.setattr(torch.cuda, "_getDeviceCount", lambda *_, **__: 0, raising=False)


class DummyPC(torch.nn.Module):
    """Simple PC stub with a trainable weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, attr_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Simple log-prob proportional to mean prob of class 1 times a weight
        probs_class1 = attr_probs[..., 1]
        return probs_class1.mean(dim=-1) * self.weight

    def log_prob(self, attr_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs_class1 = attr_probs[..., 1]
        # Reduce over community dimension to match score matrix shape (batch, batch)
        probs_reduced = probs_class1.mean(dim=-1)
        return torch.log(probs_reduced + 1e-6) * self.weight


def test_pc_params_receive_gradients() -> None:
    """PC parameters must accumulate gradient when lambda_pc > 0."""
    torch.manual_seed(0)
    config = DSLFMKGCConfig(
        num_entities=4,
        num_relations=1,
        entity_dim=8,
        feature_dim=8,
        max_communities=2,
        hidden_dim=8,
        lambda_pc=0.5,
        pc_pruning_threshold=0.01,
    )
    model = DSLFMKGCModel(config)
    model.pc_model = DummyPC()  # type: ignore[assignment]

    triples = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 0, 3],
            [3, 0, 0],
        ],
        dtype=torch.long,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    losses = model.compute_loss(
        heads=triples[:, 0],
        relations=triples[:, 1],
        tails=triples[:, 2],
        use_inbatch_negatives=True,
        entity_temperature=0.5,
        regularization_scale=1.0,
    )
    losses["loss"].backward()
    pc_grad = model.pc_model.weight.grad  # type: ignore[attr-defined]

    assert pc_grad is not None
    assert torch.isfinite(pc_grad)
    assert abs(pc_grad.item()) > 0.0


def test_pc_params_do_not_receive_gradients_when_disabled() -> None:
    """Com lambda_pc=0, PC não deve acumular gradientes."""
    torch.manual_seed(1)
    config = DSLFMKGCConfig(
        num_entities=4,
        num_relations=1,
        entity_dim=8,
        feature_dim=8,
        max_communities=2,
        hidden_dim=8,
        lambda_pc=0.0,
        pc_pruning_threshold=0.01,
    )
    model = DSLFMKGCModel(config)
    model.pc_model = DummyPC()  # type: ignore[assignment]

    triples = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 0, 3],
            [3, 0, 0],
        ],
        dtype=torch.long,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    losses = model.compute_loss(
        heads=triples[:, 0],
        relations=triples[:, 1],
        tails=triples[:, 2],
        use_inbatch_negatives=True,
        entity_temperature=0.5,
        regularization_scale=1.0,
    )
    losses["loss"].backward()
    pc_grad = model.pc_model.weight.grad  # type: ignore[attr-defined]

    assert pc_grad is None or torch.allclose(pc_grad, torch.zeros_like(pc_grad))


def test_decoder_embeddings_change_after_train_step() -> None:
    """Parâmetros do decoder devem ser atualizados após backprop."""
    torch.manual_seed(2)
    config = DSLFMKGCConfig(
        num_entities=4,
        num_relations=1,
        entity_dim=8,
        feature_dim=8,
        max_communities=2,
        hidden_dim=8,
        lambda_pc=0.0,
        pc_pruning_threshold=0.01,
    )
    model = DSLFMKGCModel(config)

    triples = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 2],
            [2, 0, 3],
            [3, 0, 0],
        ],
        dtype=torch.long,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "Model should expose trainable parameters"
    before = params[0].detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    losses = model.compute_loss(
        heads=triples[:, 0],
        relations=triples[:, 1],
        tails=triples[:, 2],
        use_inbatch_negatives=True,
        entity_temperature=0.5,
        regularization_scale=1.0,
    )
    losses["loss"].backward()
    optimizer.step()

    after = params[0].detach().clone()
    assert not torch.allclose(before, after)
