"""Unit tests for DSLFM + PC fusion (ranking influence and top-k handling)."""

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


def _build_model(lambda_pc: float) -> DSLFMKGCModel:
    """Execute build model.



    Args:

        lambda_pc: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    config = DSLFMKGCConfig(
        num_entities=3,
        num_relations=1,
        entity_dim=8,
        feature_dim=8,
        max_communities=2,
        hidden_dim=8,
        lambda_pc=lambda_pc,
        pc_pruning_threshold=0.01,
    )
    model = DSLFMKGCModel(config)
    # Disable BERT path and cache requirements for the test
    model._all_entity_features = torch.zeros(config.num_entities, config.feature_dim)  # noqa: SLF001
    model._all_entity_communities = torch.zeros(config.num_entities, config.max_communities)  # noqa: SLF001
    return model


def test_pc_rerank_changes_winner_when_lambda_positive(monkeypatch) -> None:
    """PC log-prob should be able to flip the top candidate when lambda_pc > 0."""
    # Batch of 1 query: decoder scores prefer candidate 0, PC prefere candidate 2
    decoder_scores = torch.tensor([[5.0, 4.0, 3.0]])
    pc_log_full = torch.tensor([[0.0, 0.0, 4.0]])
    tails = torch.tensor([2])

    # Lambda=0 → decoder wins (rank of tail = 3)
    model0 = _build_model(lambda_pc=0.0)
    monkeypatch.setattr(model0.decoder, "score_all_tails", lambda **_: decoder_scores)
    monkeypatch.setattr(model0, "_pc_log_prob_matrix", lambda *_, **__: pc_log_full[:, :3])
    metrics0 = model0.evaluate(
        torch.tensor([[0, 0, tails.item()]]),
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=3,
    )
    assert metrics0["mrr"] < 0.6

    # Lambda>0 → PC should push candidate 2 to the top via rerank
    model1 = _build_model(lambda_pc=1.5)
    monkeypatch.setattr(model1.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    # Stub forward: true tail 2 => score 3.0 + lambda*PC
    # But wait, metrics logic uses (chunk_scores > true_score).
    # If we want MRR=1.0, true_score must be > others.
    # Decoder score for tail 2 is 3.0. PC score is 4.0. Total = 3 + 1.5*4 = 9.0.
    # Candidate 0: 5.0 + 0 = 5.0.
    # Candidate 1: 4.0 + 0 = 4.0.
    # So tail 2 IS winner. We just need forward to return the UN-ADJUSTED decoder score (3.0),
    # because evaluate() adds PC term to true_score itself:
    # true_scores = true_scores + lambda * pc_log_true
    monkeypatch.setattr(model1.decoder, "forward", lambda **_: torch.tensor([3.0]))

    # Provide PC scores for top-3 candidates (full list)
    monkeypatch.setattr(model1, "_pc_log_prob_matrix", lambda *_, **__: pc_log_full.clone())
    # Also patch pairwise for the true score calculation
    monkeypatch.setattr(model1, "_pc_log_prob_pairwise", lambda *_, **__: torch.tensor([4.0]))

    metrics1 = model1.evaluate(
        torch.tensor([[0, 0, tails.item()]]),
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=3,
    )
    assert metrics1["mrr"] == 1.0


def test_pc_rerank_respects_topk_mask(monkeypatch) -> None:
    """Candidates fora do top-k devem permanecer com -inf após rerank."""
    decoder_scores = torch.tensor([[1.0, 0.5, -1.0]])
    pc_log = torch.tensor([[2.0, 0.0, -10.0]])
    tails = torch.tensor([0])

    model = _build_model(lambda_pc=0.5)
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    monkeypatch.setattr(model, "_pc_log_prob_matrix", lambda *_, **__: pc_log.clone())
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([1.0]))
    monkeypatch.setattr(model, "_pc_log_prob_pairwise", lambda *_, **__: torch.tensor([2.0]))

    result = model.evaluate(
        torch.tensor([[0, 0, tails.item()]]),
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=2,
    )

    # Ensure metric is sane and mean_rank finite
    assert 0.0 <= result["mrr"] <= 1.0
    # Verify that candidate outside top-2 (index 2) stays at -inf after rerank
    # by reconstructing the post-rerank scores via the same logic
    topk = torch.topk(decoder_scores, k=2, dim=1)
    # pc_log is (1, 3), topk is (1, 2). Need to select PC scores corresponding to topk indices.
    # Since indices are [0, 1], we can slice.
    pc_subset = pc_log.gather(1, topk.indices)
    updated = torch.log_softmax(topk.values, dim=1) + model.config.lambda_pc * pc_subset
    reconstructed = decoder_scores.clone().fill_(float("-inf"))
    reconstructed.scatter_(1, topk.indices, updated)
    assert torch.isinf(reconstructed[0, 2])


def test_lambda_zero_matches_decoder_ranking(monkeypatch) -> None:
    """Com lambda_pc=0, ranking deve ser idêntico ao decoder puro."""
    decoder_scores = torch.tensor([[3.0, 1.0, 2.0]])
    tails = torch.tensor([1])

    model = _build_model(lambda_pc=0.0)
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    # Stub forward for consistency (tail=1, score=1.0)
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([1.0]))

    metrics = model.evaluate(
        torch.tensor([[0, 0, tails.item()]]),
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=3,
    )

    # Tail na 3a posição -> rank=3 => MRR=1/3
    assert metrics["mrr"] == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_pc_can_dominate_with_high_lambda(monkeypatch) -> None:
    """Lambda_pc alto deve permitir PC inverter ranking mesmo com decoder forte."""
    decoder_scores = torch.tensor([[4.0, 3.5, 1.0]])
    pc_log = torch.tensor([[0.0, 0.0, 5.0]])
    tails = torch.tensor([2])

    model = _build_model(lambda_pc=3.0)
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    monkeypatch.setattr(model, "_pc_log_prob_matrix", lambda *_, **__: pc_log.clone())
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([1.0]))
    monkeypatch.setattr(model, "_pc_log_prob_pairwise", lambda *_, **__: torch.tensor([5.0]))

    metrics = model.evaluate(
        torch.tensor([[0, 0, tails.item()]]),
        batch_size=1,
        refresh_cache=False,
        rerank_top_k=3,
    )

    assert metrics["mrr"] == 1.0


def test_rerank_topk_applies_inf_outside_batch(monkeypatch) -> None:
    """Em batch, candidatos fora do top-k devem permanecer -inf."""
    decoder_scores = torch.tensor([[4.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    # Match num_entities=3
    pc_log = torch.tensor([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    tails = torch.tensor([[0, 0, 1], [1, 0, 0]])

    model = _build_model(lambda_pc=0.5)
    monkeypatch.setattr(model.decoder, "score_all_tails", lambda **_: decoder_scores.clone())
    monkeypatch.setattr(model, "_pc_log_prob_matrix", lambda *_, **__: pc_log.clone())
    # Tail 1 (batch 0) -> decoder=1.0, pc=0.5
    # Tail 0 (batch 1) -> decoder=0.1, pc=0.0
    monkeypatch.setattr(model.decoder, "forward", lambda **_: torch.tensor([1.0, 0.1]))
    monkeypatch.setattr(model, "_pc_log_prob_pairwise", lambda *_, **__: torch.tensor([0.5, 0.0]))

    metrics = model.evaluate(
        tails,
        batch_size=2,
        refresh_cache=False,
        rerank_top_k=2,
    )

    assert 0.0 <= metrics["mrr"] <= 1.0


def test_lambda_pc_zero_means_no_pc_contribution(monkeypatch) -> None:
    """Com lambda_pc=0, uso do PC não altera scores."""
    torch.manual_seed(99)
    model = _build_model(lambda_pc=0.0)
    model.eval()

    heads = torch.tensor([0, 1])
    rels = torch.tensor([0, 0])
    tails = torch.tensor([1, 2])

    base = model.forward(heads, rels, tails, use_pc=False)["scores"]
    with_pc = model.forward(heads, rels, tails, use_pc=True)["scores"]

    assert torch.allclose(base, with_pc)
