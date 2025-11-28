"""Training Observer Pattern Implementation.

Provides a decoupled mechanism for observing training events (epochs, metrics,
checkpoints) without coupling the trainer to specific logging/persistence backends.

Design Patterns Applied:
    - **Observer Pattern:** Trainers notify observers of events; observers react
      independently (MLflow, TensorBoard, console, database).
    - **Strategy Pattern:** Different observer implementations handle events
      in their own way.

Example:
    >>> from pff.utils.performance.training_observer import (
    ...     TrainingObserver, ConsoleObserver, CompositeObserver
    ... )
    >>> observer = CompositeObserver([ConsoleObserver(), MLflowObserver()])
    >>> trainer.add_observer(observer)
    >>> trainer.train(...)  # Observers are notified automatically

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pff.utils import logger


@dataclass
class TrainingEvent:
    """Represents a training event with associated metadata.

    Attributes:
        event_type: Type of event (e.g., 'epoch_end', 'batch_end', 'checkpoint').
        epoch: Current epoch number.
        step: Current step/batch number.
        metrics: Dictionary of metric name to value.
        metadata: Additional event-specific data.
    """

    event_type: str
    epoch: int = 0
    step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingObserver(ABC):
    """Abstract base class for training observers.

    Observers are notified of training events and can react independently.
    This decouples the training loop from logging, persistence, and monitoring.
    """

    @abstractmethod
    def on_event(self, event: TrainingEvent) -> None:
        """Handle a training event.

        Args:
            event: The training event to process.
        """
        pass

    def on_epoch_start(self, epoch: int, metadata: dict[str, Any] | None = None) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: The epoch number starting.
            metadata: Optional additional information.
        """
        self.on_event(TrainingEvent(
            event_type="epoch_start",
            epoch=epoch,
            metadata=metadata or {},
        ))

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: The epoch number that ended.
            metrics: Dictionary of metric values for this epoch.
            metadata: Optional additional information.
        """
        self.on_event(TrainingEvent(
            event_type="epoch_end",
            epoch=epoch,
            metrics=metrics,
            metadata=metadata or {},
        ))

    def on_batch_end(
        self,
        epoch: int,
        step: int,
        loss: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called at the end of each batch.

        Args:
            epoch: Current epoch number.
            step: Current batch/step number.
            loss: Loss value for this batch.
            metadata: Optional additional information.
        """
        self.on_event(TrainingEvent(
            event_type="batch_end",
            epoch=epoch,
            step=step,
            metrics={"loss": loss},
            metadata=metadata or {},
        ))

    def on_checkpoint(
        self,
        epoch: int,
        path: str,
        is_best: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when a checkpoint is saved.

        Args:
            epoch: Epoch at which checkpoint was saved.
            path: Path to the saved checkpoint.
            is_best: Whether this is the best model so far.
            metadata: Optional additional information.
        """
        self.on_event(TrainingEvent(
            event_type="checkpoint",
            epoch=epoch,
            metadata={"path": path, "is_best": is_best, **(metadata or {})},
        ))

    def on_training_end(
        self,
        final_metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called when training completes.

        Args:
            final_metrics: Final evaluation metrics.
            metadata: Optional additional information.
        """
        self.on_event(TrainingEvent(
            event_type="training_end",
            metrics=final_metrics,
            metadata=metadata or {},
        ))


class ConsoleObserver(TrainingObserver):
    """Observer that logs training events to console using the PFF logger.

    Follows the AGENTS.md logging contract:
        - Info/success messages in PT-BR
        - Warnings/errors in EN
    """

    def __init__(self, verbose: bool = True, log_every_n_batches: int = 100) -> None:
        """Initialize console observer.

        Args:
            verbose: Whether to log detailed information.
            log_every_n_batches: Log batch events every N batches.
        """
        self.verbose = verbose
        self.log_every_n_batches = log_every_n_batches

    def on_event(self, event: TrainingEvent) -> None:
        """Log event to console."""
        if event.event_type == "epoch_start":
            logger.info(f"Iniciando época {event.epoch}")

        elif event.event_type == "epoch_end":
            metrics_str = " | ".join(
                f"{k}={v:.4f}" for k, v in event.metrics.items()
            )
            logger.info(f"Época {event.epoch} concluída: {metrics_str}")

        elif event.event_type == "batch_end":
            if self.verbose and event.step % self.log_every_n_batches == 0:
                loss = event.metrics.get("loss", 0.0)
                # Batch-level details are debug (too granular for info)
                logger.debug(f"Epoch {event.epoch} | Batch {event.step} | Loss: {loss:.4f}")

        elif event.event_type == "checkpoint":
            path = event.metadata.get("path", "unknown")
            is_best = event.metadata.get("is_best", False)
            if is_best:
                logger.success(f"Melhor modelo salvo em {path}")
            else:
                logger.info(f"Checkpoint salvo em {path}")

        elif event.event_type == "training_end":
            metrics_str = " | ".join(
                f"{k}={v:.4f}" for k, v in event.metrics.items()
            )
            logger.success(f"Treinamento concluido: {metrics_str}")


class MLflowObserver(TrainingObserver):
    """Observer that logs metrics to MLflow.

    Requires an active MLflow run. If no run is active, events are silently
    ignored to avoid breaking the training loop.
    """

    def on_event(self, event: TrainingEvent) -> None:
        """Log event to MLflow if available."""
        try:
            import mlflow

            if not mlflow.active_run():
                return

            if event.event_type == "epoch_end":
                for name, value in event.metrics.items():
                    mlflow.log_metric(name, value, step=event.epoch)

            elif event.event_type == "batch_end":
                loss = event.metrics.get("loss")
                if loss is not None:
                    mlflow.log_metric("batch_loss", loss, step=event.step)

            elif event.event_type == "checkpoint":
                is_best = event.metadata.get("is_best", False)
                if is_best:
                    path = event.metadata.get("path")
                    if path:
                        mlflow.log_artifact(path)

            elif event.event_type == "training_end":
                for name, value in event.metrics.items():
                    mlflow.log_metric(f"final_{name}", value)

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


class CompositeObserver(TrainingObserver):
    """Composite observer that delegates to multiple observers.

    Implements the Composite pattern to allow treating a group of observers
    as a single observer.

    Example:
        >>> composite = CompositeObserver([
        ...     ConsoleObserver(),
        ...     MLflowObserver(),
        ... ])
        >>> composite.on_epoch_end(epoch=1, metrics={"loss": 0.5})
    """

    def __init__(self, observers: list[TrainingObserver] | None = None) -> None:
        """Initialize composite observer.

        Args:
            observers: List of observers to delegate to.
        """
        self.observers: list[TrainingObserver] = observers or []

    def add_observer(self, observer: TrainingObserver) -> None:
        """Add an observer to the composite.

        Args:
            observer: Observer to add.
        """
        self.observers.append(observer)

    def remove_observer(self, observer: TrainingObserver) -> None:
        """Remove an observer from the composite.

        Args:
            observer: Observer to remove.
        """
        self.observers.remove(observer)

    def on_event(self, event: TrainingEvent) -> None:
        """Delegate event to all registered observers.

        Args:
            event: The training event to broadcast.
        """
        for observer in self.observers:
            try:
                observer.on_event(event)
            except Exception as e:
                logger.warning(f"Observer {type(observer).__name__} failed: {e}")


class NullObserver(TrainingObserver):
    """Null object pattern implementation - does nothing.

    Useful as a default when no observers are configured, avoiding
    None checks throughout the codebase.
    """

    def on_event(self, event: TrainingEvent) -> None:
        """Do nothing."""
        pass


def create_default_observer(
    enable_console: bool = True,
    enable_mlflow: bool = True,
    verbose: bool = True,
) -> TrainingObserver:
    """Factory function to create a default observer configuration.

    Args:
        enable_console: Whether to enable console logging.
        enable_mlflow: Whether to enable MLflow logging.
        verbose: Whether to enable verbose output.

    Returns:
        A configured TrainingObserver (composite or null).
    """
    observers: list[TrainingObserver] = []

    if enable_console:
        observers.append(ConsoleObserver(verbose=verbose))

    if enable_mlflow:
        observers.append(MLflowObserver())

    if not observers:
        return NullObserver()

    if len(observers) == 1:
        return observers[0]

    return CompositeObserver(observers)
