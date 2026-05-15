"""Tests for Triton ranking path in DSLFM metrics reporter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pff.domain.learning.dslfm.metrics_reporter import DSLFMMetricsReporter


class _DummyRankModel(torch.nn.Module):
    def __init__(self, num_entities: int) -> None:
        super().__init__()
        self.num_entities = num_entities

    def forward(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        target = torch.remainder(heads + rels, self.num_entities)
        return -torch.abs(tails - target).float()


def test_metrics_reporter_uses_triton_ranker_on_cuda(
    tmp_path: pytest.TempPathFactory, monkeypatch
) -> None:
    from pff.shared.acceleration import triton_kernels

    called = {"ranker": False}

    def _fake_is_triton_available() -> bool:
        return True

    def _fake_triton_ranker(scores: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        called["ranker"] = True
        true_scores = scores.gather(1, tails.unsqueeze(1))
        return (scores > true_scores).sum(dim=1) + 1

    monkeypatch.setattr(triton_kernels, "is_triton_available", _fake_is_triton_available)
    monkeypatch.setattr(triton_kernels, "compute_ranks_from_scores_triton", _fake_triton_ranker)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_entities = 32768 if device.type == "cuda" else 1024
    model = _DummyRankModel(num_entities=num_entities).to(device)
    reporter = DSLFMMetricsReporter(output_dir=tmp_path)

    triples = np.array(
        [[h, r, (h + r) % num_entities] for h, r in [(1, 2), (3, 4), (7, 1), (0, 5)]],
        dtype=np.int64,
    )
    metrics = reporter.compute_link_prediction_metrics(
        model=model,
        triples=triples,
        device=device,
    )

    assert called["ranker"] is (device.type == "cuda")
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["hits@1"] == pytest.approx(1.0)
    assert metrics["hits@10"] == pytest.approx(1.0)
