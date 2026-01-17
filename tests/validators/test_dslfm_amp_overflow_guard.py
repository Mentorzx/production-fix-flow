from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class _NoOpGradScaler:
    def unscale_(
        self, optimizer: torch.optim.Optimizer
    ) -> None:  # noqa: D401 - external API shape
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def update(self) -> None:
        return None


def _set_any_grad_nonfinite(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, float("nan"))
            return
    raise RuntimeError("No trainable parameter found to poison gradients")


def test_optimizer_step_skips_on_nonfinite_grad_when_scaler_present() -> None:
    config = DSLFMKGCConfig(
        num_entities=32,
        num_relations=4,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
    )
    train_cfg = KGCTrainingConfig(
        epochs=1,
        batch_size=8,
        effective_batch_size=8,
        mixed_precision=False,
        max_grad_norm=1.0,
    )
    manager = DSLFMKGCManager(config, train_cfg, device=torch.device("cpu"))
    manager.scaler = _NoOpGradScaler()
    _set_any_grad_nonfinite(manager.model)

    manager._optimizer_step()

    assert manager.global_step == 0


def test_optimizer_step_raises_on_nonfinite_grad_without_scaler() -> None:
    config = DSLFMKGCConfig(
        num_entities=32,
        num_relations=4,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
    )
    train_cfg = KGCTrainingConfig(
        epochs=1,
        batch_size=8,
        effective_batch_size=8,
        mixed_precision=False,
        max_grad_norm=1.0,
    )
    manager = DSLFMKGCManager(config, train_cfg, device=torch.device("cpu"))
    _set_any_grad_nonfinite(manager.model)

    with pytest.raises(RuntimeError, match="Non-finite gradient norm detected"):
        manager._optimizer_step()
