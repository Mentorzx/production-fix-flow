"""Training Observer Pattern Implementation.

Provides a decoupled mechanism for observing training events (epochs, metrics,
checkpoints) without coupling the trainer to specific logging/persistence backends.

Design Patterns Applied:
    - **Observer Pattern:** Trainers notify observers of events; observers react
      independently (MLflow, TensorBoard, console, database).
    - **Strategy Pattern:** Different observer implementations handle events
      in their own way.

Example:
    >>> from pff.domain.learning.ml.training_observer import (
    ...     TrainingObserver, ConsoleObserver, CompositeObserver
    ... )
    >>> observer = CompositeObserver([ConsoleObserver(), MLflowObserver()])
    >>> trainer.add_observer(observer)
    >>> trainer.train(...)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pff.shared import logger
from pff.shared.observer import CompositeObserver as SharedCompositeObserver

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from optuna.trial import Trial as OptunaTrial
else:
    OptunaTrial = Any


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

    def on_training_start(self, config: Any, metadata: dict[str, Any] | None = None) -> None:
        """Called at the start of training.

        Args:
            config: Training configuration.
            metadata: Optional additional information.
        """
        self.on_event(
            TrainingEvent(
                event_type="training_start",
                metadata={"config": config, **(metadata or {})},
            )
        )

    def on_epoch_start(self, epoch: int, metadata: dict[str, Any] | None = None) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: The epoch number starting.
            metadata: Optional additional information.
        """
        self.on_event(
            TrainingEvent(
                event_type="epoch_start",
                epoch=epoch,
                metadata=metadata or {},
            )
        )

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
        self.on_event(
            TrainingEvent(
                event_type="epoch_end",
                epoch=epoch,
                metrics=metrics,
                metadata=metadata or {},
            )
        )

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
        self.on_event(
            TrainingEvent(
                event_type="batch_end",
                epoch=epoch,
                step=step,
                metrics={"loss": loss},
                metadata=metadata or {},
            )
        )

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
        self.on_event(
            TrainingEvent(
                event_type="checkpoint",
                epoch=epoch,
                metadata={"path": path, "is_best": is_best, **(metadata or {})},
            )
        )

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
        self.on_event(
            TrainingEvent(
                event_type="training_end",
                metrics=final_metrics,
                metadata=metadata or {},
            )
        )


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
        if event.event_type in {"training_start", "epoch_start"}:
            return
        if event.event_type == "epoch_end":
            self._log_epoch_end(event)
            return
        if event.event_type == "batch_end":
            self._log_batch_end(event)
            return
        if event.event_type == "checkpoint":
            self._log_checkpoint(event)
            return
        if event.event_type == "training_end":
            self._log_training_end(event)

    def _log_epoch_end(self, event: TrainingEvent) -> None:
        """Execute log epoch end.



        Args:

            event: Input value used by this callable.

        """

        if not event.metrics:
            return
        if not self._has_eval_metrics(event.metrics):
            return
        loss = event.metrics.get("loss", 0.0)
        mrr = event.metrics.get("mrr", event.metrics.get("best_mrr", 0.0))
        h1 = event.metrics.get("hits@1", event.metrics.get("hits1", 0.0))
        h3 = event.metrics.get("hits@3", event.metrics.get("hits3", 0.0))
        h10 = event.metrics.get("hits@10", event.metrics.get("hits10", 0.0))
        ap10 = event.metrics.get("ap@10", event.metrics.get("ap10", 0.0))
        mcc = event.metrics.get("mcc", 0.0)

        logger.info(
            f"epoch={event.epoch} etapa=evaluation loss={loss:.4f}\n"
            f"mrr={mrr:.4f} mcc={mcc:.4f}\n"
            f"hits@1={h1:.4f} hits@3={h3:.4f}\n"
            f"hits@10={h10:.4f} ap@10={ap10:.4f}\n"
        )

    def _log_batch_end(self, event: TrainingEvent) -> None:
        """Execute log batch end.



        Args:

            event: Input value used by this callable.

        """

        if not self.verbose or event.step % self.log_every_n_batches != 0:
            return
        loss = event.metrics.get("loss", 0.0)
        logger.debug(f"Epoch {event.epoch} | Batch {event.step} | Loss: {loss:.4f}")

    def _log_checkpoint(self, event: TrainingEvent) -> None:
        """Execute log checkpoint.



        Args:

            event: Input value used by this callable.

        """

        path = event.metadata.get("path", "unknown")
        is_best = event.metadata.get("is_best", False)
        if is_best:
            logger.success(f"Melhor modelo salvo em {path}")
            return
        logger.info(f"Checkpoint salvo em {path}")

    def _log_training_end(self, event: TrainingEvent) -> None:
        """Execute log training end.



        Args:

            event: Input value used by this callable.

        """

        mrr = event.metrics.get("best_val_mrr", 0.0)
        epochs = event.metrics.get("epochs_trained", 0)
        logger.success(f"Resumo do treinamento: épocas={epochs}, melhor MRR={mrr:.4f}")

    @staticmethod
    def _has_eval_metrics(metrics: dict[str, float]) -> bool:
        return any(k in metrics for k in ["mrr", "hits@1", "hits1", "mcc", "ap10"])


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
                self._log_epoch_end(event, mlflow)
                return
            if event.event_type == "batch_end":
                self._log_batch_end(event, mlflow)
                return
            if event.event_type == "checkpoint":
                self._log_checkpoint(event, mlflow)
                return
            if event.event_type == "training_end":
                self._log_training_end(event, mlflow)

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    @staticmethod
    def _log_epoch_end(event: TrainingEvent, mlflow: Any) -> None:
        """Execute log epoch end.



        Args:

            event: Input value used by this callable.

            mlflow: Input value used by this callable.

        """

        for name, value in event.metrics.items():
            mlflow.log_metric(name, value, step=event.epoch)

    @staticmethod
    def _log_batch_end(event: TrainingEvent, mlflow: Any) -> None:
        """Execute log batch end.



        Args:

            event: Input value used by this callable.

            mlflow: Input value used by this callable.

        """

        loss = event.metrics.get("loss")
        if loss is not None:
            mlflow.log_metric("batch_loss", loss, step=event.step)

    @staticmethod
    def _log_checkpoint(event: TrainingEvent, mlflow: Any) -> None:
        """Execute log checkpoint.



        Args:

            event: Input value used by this callable.

            mlflow: Input value used by this callable.

        """

        is_best = event.metadata.get("is_best", False)
        if not is_best:
            return
        path = event.metadata.get("path")
        if path:
            mlflow.log_artifact(path)

    @staticmethod
    def _log_training_end(event: TrainingEvent, mlflow: Any) -> None:
        """Execute log training end.



        Args:

            event: Input value used by this callable.

            mlflow: Input value used by this callable.

        """

        for name, value in event.metrics.items():
            mlflow.log_metric(f"final_{name}", value)


class CompositeObserver(SharedCompositeObserver, TrainingObserver):
    """Composite observer that delegates to multiple observers."""


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


class OptunaTrialObserver(TrainingObserver):
    """Observer that reports intermediate values to Optuna for pruning."""

    def __init__(
        self,
        trial: OptunaTrial,
        metric_name: str = "mrr",
        maximize: bool = True,
    ) -> None:
        """Initialize Optuna trial observer.

        Args:
            trial: Optuna trial object used for reporting/pruning.
            metric_name: Metric name to report (default: mrr).
            maximize: Whether higher is better for the metric.
        """
        self.trial = trial
        self.metric_name = metric_name
        self.maximize = maximize

    def on_event(self, event: TrainingEvent) -> None:
        """Report intermediate value to Optuna trial.

        Args:
            event: Training event to process.

        Raises:
            optuna.TrialPruned: When the pruner decides to stop the trial.
        """
        if event.event_type != "epoch_end":
            return

        metric_value = event.metrics.get(self.metric_name)
        if metric_value is None:
            metric_value = event.metrics.get("val_mrr")
        if metric_value is None:
            metric_value = event.metrics.get("loss")
            if metric_value is not None and self.maximize:
                metric_value = -metric_value

        if metric_value is None:
            return

        if optuna is None:
            logger.debug("Optuna not installed; skipping trial.report")
            return

        try:
            self.trial.report(float(metric_value), step=event.epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at epoch {event.epoch}")
        except Exception as exc:
            if "TrialPruned" in type(exc).__name__:
                raise
            logger.debug(f"Failed to report to Optuna: {exc}")
