from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.dslfm.kgc_manager import (
    DSLFMKGCManager,
    KGCTrainingConfig,
)
from pff.domain.learning.ml.kge_strategy import DSLFMStrategy, KGEConfig

pytestmark = pytest.mark.filterwarnings("ignore:.*cudaGetDeviceCount.*:UserWarning")

warnings.filterwarnings(
    "ignore",
    message=".*cudaGetDeviceCount.*",
    category=UserWarning,
)


def test_attribute_calibration() -> None:
    config = DSLFMKGCConfig(num_entities=32, num_relations=8, entity_dim=16)
    model = DSLFMKGCModel(config)
    triples = torch.tensor([[0, 0, 1], [2, 3, 4]], dtype=torch.long)

    output = model(heads=triples[:, 0], relations=triples[:, 1], tails=triples[:, 2])
    attr_probs = output["attr_probs"]

    assert torch.all(attr_probs >= 0.0)
    assert torch.all(attr_probs <= 1.0)
    summed = torch.sum(attr_probs, dim=-1)
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


def test_gradient_flow_dslfm_pc(synthetic_kg_triples: torch.Tensor) -> None:
    config = KGEConfig(
        embedding_dim=32,
        extra={
            "lambda_pc": 0.5,
            "lambda_logic": 0.1,
            "pc_rebuild_every": 1,
        },
    )
    strategy = DSLFMStrategy(config)
    model = strategy.create_model(
        num_entities=64,
        num_relations=12,
        device=torch.device("cpu"),
    )

    negatives = torch.randint(0, 64, (synthetic_kg_triples.size(0), 2, 3), dtype=torch.long)
    model.zero_grad(set_to_none=True)

    loss = strategy.compute_loss(model, synthetic_kg_triples, negatives)
    loss.backward()

    base_grad = model.base_model.entity_embedding.weight.grad
    npc_grads = [p.grad for p in strategy.npc.parameters()] if strategy.npc is not None else []

    assert base_grad is not None
    assert any(g is not None for g in npc_grads)


def test_logic_penalty_uses_t_norms() -> None:
    config = KGEConfig(
        embedding_dim=16,
        extra={
            "lambda_logic": 0.5,
            "t_norm": "lukasiewicz",
            "lambda_pc": 0.0,
        },
    )
    strategy = DSLFMStrategy(config)
    model = strategy.create_model(num_entities=16, num_relations=6, device=torch.device("cpu"))

    triples = torch.tensor([[1, 2, 3], [0, 1, 4]], dtype=torch.long)
    negatives = torch.randint(0, 16, (2, 1, 3), dtype=torch.long)

    loss = strategy.compute_loss(model, triples, negatives)
    loss.backward()

    assert loss.item() > 0
    assert model.base_model.entity_embedding.weight.grad is not None


def test_compile_preserves_evaluate(monkeypatch) -> None:
    """Ensure compiled models keep evaluate available (compile mocked)."""

    def _fake_compile(module, **_kwargs):
        return module

    monkeypatch.setattr("torch.compile", _fake_compile, raising=True)
    mock_persistence = MagicMock()
    mock_persistence.save_checkpoint = MagicMock()
    mock_persistence.load_checkpoint = MagicMock(return_value=None)
    manager = DSLFMKGCManager(
        model_config=DSLFMKGCConfig(num_entities=8, num_relations=3),
        training_config=KGCTrainingConfig(use_compile=True),
        persistence_port=mock_persistence,
        device=torch.device("cpu"),
    )
    assert hasattr(manager.model, "evaluate")


def test_vectorized_mask_known_tails(monkeypatch) -> None:
    """Test vectorized _mask_known_tails produces correct masking."""

    def _fake_compile(module, **_kwargs):
        return module

    monkeypatch.setattr("torch.compile", _fake_compile, raising=True)
    mock_persistence = MagicMock()
    mock_persistence.save_checkpoint = MagicMock()
    mock_persistence.load_checkpoint = MagicMock(return_value=None)

    manager = DSLFMKGCManager(
        model_config=DSLFMKGCConfig(num_entities=16, num_relations=4),
        training_config=KGCTrainingConfig(use_compile=False),
        persistence_port=mock_persistence,
        device=torch.device("cpu"),
    )

    # Use official API to build filters
    train_triples = np.array(
        [[0, 1, 2], [0, 1, 5], [0, 1, 8], [1, 2, 3], [1, 2, 6]], dtype=np.int64
    )
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    scores = torch.zeros((4, 10), dtype=torch.float32)
    h = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    r = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    candidates = torch.arange(0, 10, dtype=torch.long)
    t = torch.tensor([2, 5, 3, 7], dtype=torch.long)

    result = manager._mask_known_tails(scores, h, r, candidates, t)

    # In filtered eval, true tail (t) is EXCLUDED from masking
    assert result[0, 5].item() == float("-inf")
    assert result[0, 8].item() == float("-inf")
    assert result[0, 2].item() == 0.0  # true tail

    assert result[1, 2].item() == float("-inf")
    assert result[1, 8].item() == float("-inf")
    assert result[1, 5].item() == 0.0  # true tail

    assert result[2, 6].item() == float("-inf")
    assert result[2, 3].item() == 0.0  # true tail

    assert result[3, 6].item() == float("-inf")
    assert result[3, 3].item() == float("-inf")


def test_vectorized_build_inbatch_known_positive_mask(monkeypatch) -> None:
    """Test vectorized _build_inbatch_known_positive_mask produces correct mask."""

    def _fake_compile(module, **_kwargs):
        return module

    monkeypatch.setattr("torch.compile", _fake_compile, raising=True)
    mock_persistence = MagicMock()
    mock_persistence.save_checkpoint = MagicMock()
    mock_persistence.load_checkpoint = MagicMock(return_value=None)

    manager = DSLFMKGCManager(
        model_config=DSLFMKGCConfig(num_entities=16, num_relations=4),
        training_config=KGCTrainingConfig(use_compile=False),
        persistence_port=mock_persistence,
        device=torch.device("cpu"),
    )

    # Use official API to build filters
    train_triples = np.array([[0, 1, 2], [0, 1, 5]], dtype=np.int64)
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    h = torch.tensor([0, 0], dtype=torch.long)
    r = torch.tensor([1, 1], dtype=torch.long)
    t = torch.tensor([2, 5], dtype=torch.long)

    mask = manager._build_inbatch_known_positive_mask(h, r, t)

    assert mask.shape == (2, 2)
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is True
    assert mask[1, 1].item() is True
