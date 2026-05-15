"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_dslfm_amp_overflow_guard.py

"""

from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class _NoOpGradScaler:
    def __init__(self):
        """Execute init."""

        self._scale = 1.0

    def get_scale(self) -> float:
        """Execute get scale.



        Returns:

            Return value produced by the callable.

        """

        return self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Execute unscale.



        Args:

            optimizer: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Execute step.



        Args:

            optimizer: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._should_decrease = True

    def update(self) -> None:
        """Execute update.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if getattr(self, "_should_decrease", False):
            self._scale *= 0.5
            self._should_decrease = False


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def _set_any_grad_nonfinite(model: torch.nn.Module) -> None:
    """Execute set any grad nonfinite.



    Args:

        model: Input value used by this callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, float("nan"))
            return
    raise RuntimeError("No trainable parameter found to poison gradients")


def test_optimizer_step_skips_on_nonfinite_grad_when_scaler_present() -> None:
    """Execute test optimizer step skips on nonfinite grad when scaler present.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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

    class MockPersistence:
        """Represent MockPersistence."""

        def save_checkpoint(self, data, filename):
            """Execute save checkpoint.



            Args:

                data: Input value used by this callable.

                filename: Input value used by this callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            pass

        def load_checkpoint(self, filename, map_location=None):
            """Execute load checkpoint.



            Args:

                filename: Input value used by this callable.

                map_location: Optional input value.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            return None

    manager = DSLFMKGCManager(
        config,
        train_cfg,
        persistence_port=MockPersistence(),
        device=torch.device("cpu"),
    )
    manager.scaler = _NoOpGradScaler()
    manager.scheduler = _CountingScheduler()
    _set_any_grad_nonfinite(manager.model)

    manager._optimizer_step()

    assert manager.global_step == 1
    assert manager.scheduler.step_calls == 0


def test_optimizer_step_raises_on_nonfinite_grad_without_scaler() -> None:
    """Execute test optimizer step raises on nonfinite grad without scaler.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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

    class MockPersistence:
        """Represent MockPersistence."""

        def save_checkpoint(self, data, filename):
            """Execute save checkpoint.



            Args:

                data: Input value used by this callable.

                filename: Input value used by this callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            pass

        def load_checkpoint(self, filename, map_location=None):
            """Execute load checkpoint.



            Args:

                filename: Input value used by this callable.

                map_location: Optional input value.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            return None

    manager = DSLFMKGCManager(
        config,
        train_cfg,
        persistence_port=MockPersistence(),
        device=torch.device("cpu"),
    )
    _set_any_grad_nonfinite(manager.model)

    with pytest.raises(RuntimeError, match="Non-finite gradient norm detected"):
        manager._optimizer_step()
