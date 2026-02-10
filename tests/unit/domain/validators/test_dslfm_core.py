from __future__ import annotations

import warnings
from typing import cast
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


def test_mask_known_tails_handles_noncontiguous_candidates(monkeypatch) -> None:
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

    train_triples = np.array(
        [[0, 1, 2], [0, 1, 5], [0, 1, 8], [1, 2, 3]],
        dtype=np.int64,
    )
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    scores = torch.zeros((2, 3), dtype=torch.float32)
    h = torch.tensor([0, 0], dtype=torch.long)
    r = torch.tensor([1, 1], dtype=torch.long)
    t = torch.tensor([2, 5], dtype=torch.long)
    candidates = torch.tensor([[8, 2, 6], [5, 8, 7]], dtype=torch.long)

    result = manager._mask_known_tails(scores, h, r, candidates, t)

    assert result[0, 0].item() == float("-inf")
    assert result[0, 1].item() == 0.0
    assert result[1, 1].item() == float("-inf")
    assert result[1, 0].item() == 0.0


def test_train_dslfm_kgc_applies_config_defaults(monkeypatch) -> None:
    from pff.domain.learning.dslfm import kgc_manager as kgc_manager_module
    from pff.domain.learning.dslfm import dslfm_kgc as dslfm_kgc_module

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(self, model_config, training_config, persistence_port, **_kwargs):
            captured["model_config"] = model_config
            captured["training_config"] = training_config

        def train(self, train_triples, valid_triples, **_kwargs):
            return {"best_val_mrr": 0.0}

    class DummyPersistence:
        def save_checkpoint(self, *_args, **_kwargs):
            pass

        def load_checkpoint(self, *_args, **_kwargs):
            return None

    config_payload = {
        "kgc": {
            "model": {
                "entity_dim": 33,
                "feature_dim": 44,
                "max_communities": 12,
                "hidden_dim": 64,
                "ibp_alpha": 2.5,
                "temperature": 0.12,
                "stochastic_latents": True,
                "encoder_dropout_p": 0.15,
                "kl_weight": 0.02,
                "free_bits": 0.2,
                "sparsity_weight": 0.003,
                "use_checkpointing": True,
                "sampler_type": "uniform",
                "sampler_temperature": 3.0,
                "learnable_temperature": False,
                "contrastive_temperature": 0.03,
                "negative_sample_size": 128,
                "num_global_negatives": 12,
                "cache_global_negatives": True,
                "global_negatives_refresh_steps": 7,
                "logvar_clip_min": -8.0,
                "logvar_clip_max": 6.0,
                "community_weight": 0.5,
                "feature_weight": 0.7,
            },
            "training": {
                "epochs": 12,
                "batch_size": 16,
                "effective_batch_size": 64,
                "learning_rate": 0.001,
                "warmup_steps": 10,
                "kl_warmup_epochs": 4,
                "min_kl_weight": 0.1,
                "max_kl_weight": 0.2,
                "temperature_anneal": 0.95,
                "min_temperature": 0.2,
                "validate_every": 2,
                "early_stopping_patience": 3,
                "min_delta": 0.001,
                "train_heartbeat_interval_s": 30.0,
                "score_all_tails_chunk_size": 1234,
                "mixed_precision": False,
                "use_compile": False,
                "optimizer_fused": False,
                "optimizer_foreach": True,
                "num_workers": 0,
                "pin_memory": False,
                "dataloader_prefetch_factor": 2,
                "dataloader_persistent_workers": False,
                "max_grad_norm": 1.1,
                "cuda_cache_flush_steps": 5,
                "cuda_cache_flush": {
                    "enabled": False,
                    "free_ratio_low": 0.2,
                    "free_ratio_high": 0.5,
                },
                "use_faiss_eval": True,
                "faiss_candidate_k": 256,
                "allow_tf32": False,
                "matmul_precision": "medium",
                "regularization_warmup_epochs": 3,
                "regularization_start_scale": 0.2,
                "rerank_top_k": 32,
                "triton_min_entities": 2048,
                "adaptive_batch_size": True,
                "min_batch_size": 8,
                "max_batch_size": 32,
                "oom_backoff_factor": 0.6,
                "batch_growth_factor": 1.1,
                "target_gpu_mem_util": 0.6,
                "max_oom_retries": 2,
                "num_workers_heuristic": {
                    "min_workers": 1,
                    "max_workers": 2,
                    "vram_threshold_gb": 8,
                },
            },
        },
        "logic": {
            "lambda_logic": 0.04,
            "t_norm": "lukasiewicz",
            "smoothing_epsilon": 0.005,
        },
        "pc": {
            "lambda_pc": 0.02,
            "pruning_threshold": 0.3,
            "grow_noise": 0.02,
            "rebuild_every": 11,
            "max_circuit_depth": 3,
        },
        "compile": {
            "mode": "reduce-overhead",
            "dynamic": True,
            "fullgraph": False,
        },
    }

    monkeypatch.setattr(kgc_manager_module, "DSLFMKGCManager", DummyManager)
    monkeypatch.setattr(
        dslfm_kgc_module,
        "load_dslfm_kgc_settings",
        lambda *_args, **_kwargs: config_payload,
    )

    train_triples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
    valid_triples = np.array([[0, 0, 1]], dtype=np.int64)

    kgc_manager_module.train_dslfm_kgc(
        train_triples=train_triples,
        valid_triples=valid_triples,
        num_entities=3,
        num_relations=1,
        persistence_port=DummyPersistence(),
        relation_names=["r0"],
        use_bert=False,
    )

    model_config = cast(DSLFMKGCConfig, captured["model_config"])
    training_config = cast(KGCTrainingConfig, captured["training_config"])

    assert model_config.temperature == 0.12
    assert model_config.lambda_pc == 0.02
    assert model_config.lambda_logic == 0.04
    assert model_config.sampler_type == "uniform"
    assert model_config.negative_sample_size == 128
    assert training_config.epochs == 12
    assert training_config.batch_size == 16
    assert training_config.learning_rate == 0.001
    assert training_config.use_faiss_eval is True
    assert training_config.faiss_candidate_k == 256
    assert training_config.rerank_top_k == 32
    assert training_config.cuda_cache_flush_enabled is False


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


def test_faiss_candidate_scoring_pc_pairwise() -> None:
    config = DSLFMKGCConfig(
        num_entities=8,
        num_relations=3,
        entity_dim=8,
        feature_dim=8,
        hidden_dim=16,
        max_communities=4,
        lambda_pc=0.1,
    )
    model = DSLFMKGCModel(config)
    model.eval()
    model.precompute_entity_latents(batch_size=4)

    class DummyIndex:
        def search(self, feat_np, k):
            import numpy as np

            batch = feat_np.shape[0]
            indices = np.tile(np.arange(k), (batch, 1))
            scores = np.zeros((batch, k), dtype=np.float32)
            return scores, indices

    model._faiss_index = DummyIndex()
    model._faiss_index_key = (0, 0)

    heads = torch.tensor([0, 1], dtype=torch.long)
    relations = torch.tensor([0, 1], dtype=torch.long)
    latents = model.encode_entities(heads)

    scores, cand_idx = model._score_faiss_candidates(
        latents["communities"],
        latents["features"],
        relations,
        3,
    )

    assert scores.shape == (2, 3)
    assert cand_idx.shape == (2, 3)
    assert torch.isfinite(scores).all()
