"""Base Trainer Template Method Pattern Implementation.

Provides an abstract base trainer that defines the training algorithm skeleton,
with customizable steps for concrete trainer implementations.

Design Patterns Applied:
    - **Template Method:** Defines training flow with abstract hooks for subclasses.
    - **Strategy Pattern:** Uses KGEModelStrategy for model-specific operations.
    - **Observer Pattern:** Integrates with TrainingObserver for event notification.
    - **Dependency Injection:** Accepts FileManager, observers, and strategies.

Example:
    >>> from pff.domain.learning.ml import BaseTrainer, TrainerConfig
    >>> class MyTrainer(BaseTrainer):
    ...     def _train_epoch(self, dataloader, epoch):
    ...         pass
    >>> trainer = MyTrainer(config)
    >>> trainer.train(train_data, val_data, num_epochs=100)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pff.domain.learning.ml.training_observer import (
    CompositeObserver,
    NullObserver,
    TrainingObserver,
)
from pff.shared import FileManager, logger
from pff.shared.core.config import settings


@dataclass
class TrainerConfig:
    """Configuration for trainers.

    Attributes:
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        patience: Early stopping patience.
        validate_every: Validation frequency (epochs).
        checkpoint_dir: Directory for saving checkpoints.
        seed: Random seed for reproducibility.
        device: Target device ('cuda', 'cpu', 'auto').
        use_amp: Enable automatic mixed precision.
        extra: Additional trainer-specific parameters.
    """

    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.001
    patience: int = 10
    validate_every: int = 5
    checkpoint_dir: Path = field(default_factory=lambda: settings.OUTPUTS_DIR / "checkpoints")
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Abstract base trainer implementing Template Method pattern.

    Design Pattern: Template Method
        - Defines the skeleton of the training algorithm.
        - Subclasses override specific steps without changing structure.
        - Common functionality (checkpointing, early stopping) is reused.

    The training flow is:
        1. setup() - Initialize model, optimizer, scheduler
        2. for each epoch:
            a. _on_epoch_start()
            b. _train_epoch() [ABSTRACT]
            c. _validate() [ABSTRACT] (if validation data provided)
            d. _on_epoch_end()
            e. _maybe_checkpoint()
            f. _check_early_stopping()
        3. teardown()

    Attributes:
        config: Trainer configuration.
        model: The model being trained.
        optimizer: Training optimizer.
        scheduler: Learning rate scheduler.
        observer: Training event observer.
        file_manager: FileManager for I/O operations.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        observer: TrainingObserver | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize base trainer.

        Args:
            config: Trainer configuration. Uses defaults if None.
            observer: Training observer for events. Uses NullObserver if None.
            file_manager: FileManager instance. Creates new if None.
        """
        self.config = config or TrainerConfig()
        self.observer = observer or NullObserver()
        self.file_manager = file_manager or FileManager()

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

        self.device = self._resolve_device()
        self.current_epoch = 0
        self.best_score = -float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, Any]] = []

        self._set_seeds()

    def _resolve_device(self) -> torch.device:
        """Resolve target device from config.

        Returns:
            Resolved torch device.
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def train(
        self,
        train_data: Any,
        val_data: Any | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, Any]:
        """Execute the training loop (Template Method).

        This is the main entry point that orchestrates the training flow.
        Subclasses customize behavior by overriding the abstract methods.

        Args:
            train_data: Training dataset or dataloader.
            val_data: Optional validation data.
            num_epochs: Override config num_epochs if provided.

        Returns:
            Dictionary with training statistics and final metrics.
        """
        epochs = num_epochs or self.config.num_epochs

        self.setup(train_data, val_data)

        stats = {
            "epochs_trained": 0,
            "best_epoch": 0,
            "best_score": self.best_score,
            "training_time": 0.0,
            "final_metrics": {},
        }

        logger.info(f"Iniciando treinamento: {epochs} épocas")
        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch

                self._on_epoch_start(epoch)

                train_metrics = self._train_epoch(train_data, epoch)

                val_metrics = {}
                if val_data is not None and epoch % self.config.validate_every == 0:
                    val_metrics = self._validate(val_data)

                self._on_epoch_end(epoch, train_metrics, val_metrics)

                all_metrics = {**train_metrics, **val_metrics}
                self.training_history.append({"epoch": epoch, **all_metrics})

                if val_metrics:
                    primary_metric = self._get_primary_metric(val_metrics)
                    if primary_metric > self.best_score:
                        self.best_score = primary_metric
                        self.patience_counter = 0
                        stats["best_epoch"] = epoch
                        stats["best_score"] = self.best_score
                        self._save_checkpoint(is_best=True)
                    else:
                        self.patience_counter += 1

                    if self._check_early_stopping():
                        logger.info(f"Parada antecipada na época {epoch}")
                        break

                stats["epochs_trained"] = epoch + 1

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._save_checkpoint(is_best=False, suffix="interrupted")

        finally:
            self.teardown()

        stats["training_time"] = time.time() - start_time
        stats["final_metrics"] = self.training_history[-1] if self.training_history else {}

        logger.success(
            f"Treinamento concluído em {stats['training_time']:.1f}s "
            f"({stats['epochs_trained']} épocas)"
        )

        return stats

    def setup(self, train_data: Any, val_data: Any | None = None) -> None:
        """Setup training components.

        Override to customize model/optimizer/scheduler initialization.

        Args:
            train_data: Training data for setup decisions.
            val_data: Validation data for setup decisions.
        """
        self.file_manager.ensure_dir(self.config.checkpoint_dir)
        self._setup_model(train_data)
        self._setup_optimizer()
        self._setup_scheduler()

    def teardown(self) -> None:
        """Cleanup after training.

        Override to add custom cleanup logic.
        """
        pass

    @abstractmethod
    def _setup_model(self, train_data: Any) -> None:
        """Initialize the model.

        Args:
            train_data: Training data for model configuration.
        """
        pass

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer.

        Default implementation uses AdamW. Override to customize.
        """
        if self.model is None:
            raise RuntimeError("Model must be set up before optimizer")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def _setup_scheduler(self) -> None:
        """Initialize the learning rate scheduler.

        Default implementation uses ReduceLROnPlateau. Override to customize.
        """
        if self.optimizer is None:
            return

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

    @abstractmethod
    def _train_epoch(self, train_data: Any, epoch: int) -> dict[str, float]:
        """Train one epoch.

        Args:
            train_data: Training data.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        pass

    @abstractmethod
    def _validate(self, val_data: Any) -> dict[str, float]:
        """Validate the model.

        Args:
            val_data: Validation data.

        Returns:
            Dictionary of validation metrics.
        """
        pass

    def _get_primary_metric(self, metrics: dict[str, float]) -> float:
        """Get the primary metric for early stopping/checkpointing.

        Override to customize which metric drives early stopping.

        Args:
            metrics: Dictionary of metrics.

        Returns:
            Primary metric value.
        """
        for key in ["mrr", "auc", "f1", "accuracy"]:
            if key in metrics:
                return metrics[key]
        return list(metrics.values())[0] if metrics else 0.0

    def _on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number.
        """
        self.observer.on_epoch_start(epoch)

    def _on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics for this epoch.
            val_metrics: Validation metrics for this epoch.
        """
        all_metrics = {**train_metrics, **val_metrics}
        self.observer.on_epoch_end(epoch, all_metrics)
        self._step_scheduler(val_metrics)

    def _step_scheduler(self, val_metrics: dict[str, float]) -> None:
        """Step scheduler with standard PyTorch call pattern.

        Args:
            val_metrics: Validation metrics; used to drive ReduceLROnPlateau.
        """
        if self.scheduler is None or not val_metrics:
            return

        primary = self._get_primary_metric(val_metrics)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(primary)
        else:
            self.scheduler.step()

    def _check_early_stopping(self) -> bool:
        """Check if early stopping should trigger.

        Returns:
            True if training should stop.
        """
        return self.patience_counter >= self.config.patience

    def _save_checkpoint(self, is_best: bool = False, suffix: str = "") -> None:
        """Save a model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
            suffix: Optional suffix for checkpoint filename.
        """
        if self.model is None:
            return

        filename = "best_model.pt" if is_best else f"checkpoint_{self.current_epoch}.pt"
        if suffix:
            filename = f"{suffix}_{filename}"

        path = self.config.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (self.optimizer.state_dict() if self.optimizer else None),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
            "best_score": self.best_score,
            "config": self.config.__dict__,
        }

        try:
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            self.file_manager.save(buffer.getvalue(), path)
        except Exception as exc:
            logger.error(f"Failed to save checkpoint to {path}: {exc}")
            return

        if is_best:
            logger.info(f"Melhor modelo salvo: {path}")

    def load_checkpoint(self, path: Path) -> bool:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            True if checkpoint was loaded successfully.
        """
        if not self.file_manager.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_score = checkpoint.get("best_score", -float("inf"))

        logger.info(f"Checkpoint carregado: {path}")
        return True

    def add_observer(self, observer: TrainingObserver) -> None:
        """Add an observer to the trainer.

        If current observer is a CompositeObserver, adds to it.
        Otherwise, wraps both in a new CompositeObserver.

        Args:
            observer: Observer to add.
        """
        if isinstance(self.observer, CompositeObserver):
            self.observer.add_observer(observer)
        elif isinstance(self.observer, NullObserver):
            self.observer = observer
        else:
            self.observer = CompositeObserver([self.observer, observer])
