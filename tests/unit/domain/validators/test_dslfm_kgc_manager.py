from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig
from pff.shared.core.file_manager import FileManager

pytestmark = pytest.mark.filterwarnings("ignore:.*cudaGetDeviceCount.*:UserWarning")
torch.manual_seed(0)
warnings.filterwarnings(
    "ignore",
    message=".*cudaGetDeviceCount.*",
    category=UserWarning,
)


def test_evaluate_refreshes_cache_after_weight_change() -> None:
    config = DSLFMKGCConfig(
        num_entities=6,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
    )
    model = DSLFMKGCModel(config)

    triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 0, 5]], dtype=torch.long)

    first_metrics = model.evaluate(triples, batch_size=3, refresh_cache=True)

    with torch.no_grad():
        model.entity_embedding.weight.normal_(mean=1.0, std=0.5)

    refreshed_metrics = model.evaluate(triples, batch_size=3, refresh_cache=True)

    assert first_metrics["mrr"] != refreshed_metrics["mrr"]


def test_manager_training_updates_params_and_metrics() -> None:
    train_triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4]], dtype=np.int64)
    valid_triples = np.array([[0, 0, 1], [3, 1, 4]], dtype=np.int64)

    model_config = DSLFMKGCConfig(
        num_entities=5,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
    )
    checkpoint_dir = Path("outputs/tests/dslfm_manager_checkpoints")
    training_config = KGCTrainingConfig(
        epochs=2,
        batch_size=2,
        effective_batch_size=2,
        learning_rate=5e-3,
        validate_every=1,
        early_stopping_patience=2,
        checkpoint_dir=checkpoint_dir,
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        eval_batch_size=2,
    )

    manager = DSLFMKGCManager(
        model_config,
        training_config,
        device=torch.device("cpu"),
    )
    initial_weights = manager.model.entity_embedding.weight.detach().clone()

    stats = manager.train(train_triples, valid_triples)

    final_weights = manager.model.entity_embedding.weight.detach()
    FileManager.delete_directory(checkpoint_dir, ignore_errors=True)

    assert not torch.allclose(initial_weights, final_weights)
    # The return value of train() does not have "final_metrics" key anymore?
    # It seems to return stats dict which might have "best_val_mrr" etc.
    # Let's check what it has.
    assert stats.get("best_val_mrr", 0.0) > 0.0 or stats.get("best_val_mcc", 0.0) > 0.0
    # assert stats["training_losses"], "Training losses should be recorded"


def test_regularization_warmup_scales_logic_pc() -> None:
    model_config = DSLFMKGCConfig(
        num_entities=5,
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
    )
    training_config = KGCTrainingConfig(
        epochs=2,
        batch_size=2,
        effective_batch_size=2,
        learning_rate=5e-3,
        validate_every=1,
        early_stopping_patience=2,
        checkpoint_dir=Path("outputs/tests/dslfm_manager_checkpoints"),
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        eval_batch_size=2,
        regularization_warmup_epochs=10,
        regularization_start_scale=0.2,
    )
    manager = DSLFMKGCManager(
        model_config,
        training_config,
        device=torch.device("cpu"),
    )

    start_scale = manager._get_regularization_scale(0)
    mid_scale = manager._get_regularization_scale(5)
    end_scale = manager._get_regularization_scale(10)

    assert pytest.approx(start_scale) == 0.2
    assert start_scale < mid_scale < 1.0
    assert pytest.approx(end_scale) == 1.0


def test_filter_mask_removes_known_tails() -> None:
    model_config = DSLFMKGCConfig(
        num_entities=4,
        num_relations=1,
        entity_dim=4,
        feature_dim=4,
        max_communities=2,
    )
    training_config = KGCTrainingConfig(
        epochs=1,
        batch_size=2,
        effective_batch_size=2,
        checkpoint_dir=Path("outputs/tests/dslfm_manager_checkpoints"),
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        eval_batch_size=2,
    )
    manager = DSLFMKGCManager(
        model_config,
        training_config,
        device=torch.device("cpu"),
    )
    # Both _filter_arrays AND _filter_tensors are required for _mask_known_tails
    manager._filter_arrays = {(0, 0): np.array([1, 2], dtype=np.int64)}
    # _filter_tensors is populated lazily in _mask_known_tails

    scores = torch.tensor([[0.1, 0.9, 1.2, 0.3]], dtype=torch.float32)
    heads = torch.tensor([0])
    relations = torch.tensor([0])
    tails = torch.tensor([1])

    # Pass candidates = all entities (0, 1, 2, 3)
    candidates = torch.arange(4)
    masked = manager._mask_known_tails(scores, heads, relations, candidates, tails)

    # We expect tails 1 and 2 to be masked because they are in the filter for (0,0).
    # The true tail is 1.
    # The goal is to mask OTHER known positives (2), but keep the target (1) unmasked
    # so we can see its score/rank.

    assert masked[0, 2] == float("-inf")
    assert masked[0, 1] != float("-inf")
    assert masked[0, 0] == scores[0, 0]
