"""Smoke tests for DSLFM learning and scoring stability on tiny graphs.

These tests focus on bug-hunting scenarios:
- Loss should decrease over a few optimization steps.
- Positive triples should separate from negatives after short training.
- PC rerank path should return finite metrics (no NaN/inf) even when active.
"""

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


def _tiny_config(lambda_pc: float = 0.0) -> DSLFMKGCConfig:
    """Create a minimal configuration for quick CPU smoke tests."""
    return DSLFMKGCConfig(
        num_entities=6,
        num_relations=3,
        entity_dim=16,
        feature_dim=16,
        max_communities=4,
        hidden_dim=8,
        temperature=0.5,
        kl_weight=0.01,
        sparsity_weight=0.001,
        sampler_type="uniform",
        sampler_temperature=1.0,
        lambda_logic=0.0,
        lambda_pc=lambda_pc,
        pc_pruning_threshold=0.01,
        pc_grow_noise=0.01,
        pc_rebuild_every=2,
        pc_max_depth=2,
    )


def test_loss_decreases_over_steps() -> None:
    """Loss should go down after a few optimization steps on a tiny graph."""
    torch.manual_seed(7)
    config = _tiny_config(lambda_pc=0.0)
    model = DSLFMKGCModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    triples = torch.tensor(
        [
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 3],
            [3, 0, 4],
            [4, 1, 5],
            [5, 2, 0],
        ],
        dtype=torch.long,
    )

    model.train()
    initial = None
    last_loss = None
    for step in range(10):
        optimizer.zero_grad()
        losses = model.compute_loss(
            heads=triples[:, 0],
            relations=triples[:, 1],
            tails=triples[:, 2],
            use_inbatch_negatives=True,
            entity_temperature=0.5,
            regularization_scale=0.5,
        )
        loss = losses["loss"]
        loss.backward()
        optimizer.step()
        if initial is None:
            initial = loss.item()
        last_loss = loss.item()

    assert initial is not None and last_loss is not None
    assert last_loss < initial * 0.75


def test_positive_scores_separate_from_negatives() -> None:
    """Short training should increase the gap between positive and negative scores."""
    torch.manual_seed(11)
    config = _tiny_config(lambda_pc=0.0)
    model = DSLFMKGCModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    positives = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 2, 5],
            [1, 0, 2],
        ],
        dtype=torch.long,
    )
    negatives = torch.tensor(
        [
            [0, 0, 3],
            [2, 1, 0],
            [4, 2, 1],
            [1, 0, 5],
        ],
        dtype=torch.long,
    )

    def _score_gap() -> float:
        with torch.no_grad():
            pos = model.score_triples_batch(positives)
            neg = model.score_triples_batch(negatives)
            return float((pos.mean() - neg.mean()).item())

    gap_before = _score_gap()
    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        losses = model.compute_loss(
            heads=positives[:, 0],
            relations=positives[:, 1],
            tails=positives[:, 2],
            use_inbatch_negatives=True,
            entity_temperature=0.5,
            regularization_scale=0.5,
        )
        losses["loss"].backward()
        optimizer.step()

    gap_after = _score_gap()
    assert gap_after > gap_before + 0.1


def test_pc_rerank_returns_finite_metrics() -> None:
    """PC rerank path should not produce NaN/inf metrics on tiny input."""
    torch.manual_seed(13)
    config = _tiny_config(lambda_pc=0.2)
    model = DSLFMKGCModel(config)
    # Ensure PC exists for the rerank path
    assert model.pc_model is not None

    triples = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 2, 5],
        ],
        dtype=torch.long,
    )

    metrics = model.evaluate(
        triples,
        batch_size=3,
        refresh_cache=True,
        rerank_top_k=2,
    )

    expected_keys = {
        "mrr",
        "hits@1",
        "hits@3",
        "hits@10",
        "ap@10",
    }
    assert expected_keys.issubset(metrics.keys())
    numeric_values = [v for k, v in metrics.items() if k != "evaluation_mode"]
    assert all(torch.isfinite(torch.tensor(numeric_values)))
    assert 0.0 <= metrics["mrr"] <= 1.0


def test_compute_loss_outputs_are_finite() -> None:
    """compute_loss deve retornar termos finitos e chaves esperadas."""
    torch.manual_seed(21)
    config = _tiny_config(lambda_pc=0.1)
    model = DSLFMKGCModel(config)

    triples = torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 2, 5],
        ],
        dtype=torch.long,
    )

    losses = model.compute_loss(
        heads=triples[:, 0],
        relations=triples[:, 1],
        tails=triples[:, 2],
        use_inbatch_negatives=True,
        entity_temperature=0.5,
        regularization_scale=0.5,
    )

    expected_keys = {"loss", "pc_penalty", "sparsity_loss", "kl_gaussian", "kl_ibp"}
    assert expected_keys.issubset(losses.keys())
    assert all(torch.isfinite(val) for val in losses.values())


def test_evaluate_without_pc_matches_decoder_ranking(monkeypatch) -> None:
    """Com lambda_pc=0, evaluate deve refletir ranking do decoder sem influência do PC."""
    torch.manual_seed(23)
    config = _tiny_config(lambda_pc=0.0)
    model = DSLFMKGCModel(config)
    # Stub decoder scores to a deterministic matrix
    decoder_scores = torch.tensor([[2.5, 0.5, 1.0]])
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    # ALSO stub forward to consistency!
    # True tail is 1, so forward should return 0.5
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([0.5]))

    triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
    metrics = model.evaluate(
        triples,
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=3,
    )

    assert metrics["mrr"] == pytest.approx(1.0 / 3.0, rel=1e-6)  # tail at rank 3


def test_evaluate_no_rerank_matches_decoder_for_batch(monkeypatch) -> None:
    """Sem rerank_top_k, avaliação deve seguir scores do decoder para todo o batch."""
    torch.manual_seed(29)
    config = _tiny_config(lambda_pc=0.0)
    model = DSLFMKGCModel(config)
    decoder_scores = torch.tensor([[2.0, 1.0, 0.5], [0.2, 3.0, 1.0]])
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    # Stub forward:
    # Batch 0: tail=1, score=1.0
    # Batch 1: tail=0, score=0.2
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([1.0, 0.2]))

    triples = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.long)
    metrics = model.evaluate(
        triples,
        batch_size=2,
        refresh_cache=False,
        rerank_top_k=None,
    )

    # Rank estimado pelo decoder (score maior = melhor): tail1 rank=2, tail0 rank=3
    expected_mrr = (1.0 / 2.0 + 1.0 / 3.0) / 2.0
    assert metrics["mrr"] == pytest.approx(expected_mrr, rel=1e-6)
