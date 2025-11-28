"""RotatE Metrics Reporter Component.

Handles metrics computation, reporting, and persistence for RotatE training.
Extracted from RotatEManager for Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Observer Pattern:** Reports metrics to registered observers.
    - **Strategy Pattern:** Supports different output formats (JSON, MLflow).

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pff.utils import FileManager, logger
from pff.utils.performance.training_observer import TrainingObserver, TrainingEvent


class RotatEMetricsReporter:
    """Reporter for RotatE training metrics.

    Computes and reports link prediction metrics (MRR, Hits@K).
    Integrates with TrainingObserver for decoupled event handling.

    Attributes:
        file_manager: FileManager instance for I/O operations.
        observers: List of TrainingObserver instances.
        output_dir: Directory for saving metrics files.
    """

    def __init__(
        self,
        output_dir: Path | str,
        file_manager: FileManager | None = None,
        observers: list[TrainingObserver] | None = None,
    ) -> None:
        """Initialize metrics reporter.

        Args:
            output_dir: Directory for saving metrics files.
            file_manager: FileManager instance for I/O operations.
            observers: List of TrainingObserver instances.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_manager = file_manager or FileManager()
        self.observers: list[TrainingObserver] = observers or []

    def add_observer(self, observer: TrainingObserver) -> None:
        """Add an observer for training events.

        Args:
            observer: TrainingObserver to add.
        """
        self.observers.append(observer)

    def remove_observer(self, observer: TrainingObserver) -> None:
        """Remove an observer.

        Args:
            observer: TrainingObserver to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self, event: TrainingEvent) -> None:
        """Notify all observers of an event.

        Args:
            event: TrainingEvent to dispatch.
        """
        for observer in self.observers:
            try:
                observer.on_event(event)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")

    def report_epoch_start(self, epoch: int) -> None:
        """Report epoch start event.

        Args:
            epoch: Current epoch number.
        """
        event = TrainingEvent(event_type="epoch_start", epoch=epoch)
        self.notify_observers(event)

    def report_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Report epoch completion with metrics.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_metrics: Optional validation metrics.
            learning_rate: Current learning rate.
        """
        metrics = {"train_loss": train_loss}
        if val_metrics:
            metrics.update(val_metrics)

        metadata = {}
        if learning_rate is not None:
            metadata["learning_rate"] = learning_rate

        event = TrainingEvent(
            event_type="epoch_end",
            epoch=epoch,
            metrics=metrics,
            metadata=metadata,
        )
        self.notify_observers(event)

    def report_checkpoint(
        self,
        epoch: int,
        path: str | Path,
        is_best: bool = False,
    ) -> None:
        """Report checkpoint save event.

        Args:
            epoch: Epoch at which checkpoint was saved.
            path: Path to saved checkpoint.
            is_best: Whether this is the best model so far.
        """
        event = TrainingEvent(
            event_type="checkpoint",
            epoch=epoch,
            metadata={"path": str(path), "is_best": is_best},
        )
        self.notify_observers(event)

    def report_training_end(
        self,
        final_metrics: dict[str, float],
        epochs_trained: int,
        training_time: float,
    ) -> None:
        """Report training completion.

        Args:
            final_metrics: Final validation metrics.
            epochs_trained: Total epochs trained.
            training_time: Total training time in seconds.
        """
        event = TrainingEvent(
            event_type="training_end",
            metrics=final_metrics,
            metadata={
                "epochs_trained": epochs_trained,
                "training_time": training_time,
            },
        )
        self.notify_observers(event)

    def compute_link_prediction_metrics(
        self,
        model: torch.nn.Module,
        triples: np.ndarray,
        device: torch.device,
        batch_size: int = 128,
    ) -> dict[str, float]:
        """Compute link prediction metrics (MRR, Hits@K).

        Args:
            model: RotatE model to evaluate.
            triples: Triples array of shape [n_triples, 3].
            device: Computation device.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with 'mrr', 'hits@1', 'hits@10' metrics.
        """
        model.eval()
        num_samples = len(triples)
        num_entities = model.num_entities

        # Adaptive batch size to prevent OOM
        max_memory_bytes = 200 * 1024 * 1024  # 200MB
        max_batch_by_memory = max(1, max_memory_bytes // (num_entities * 4))
        eval_batch_size = min(batch_size, num_samples, max_batch_by_memory)

        all_mrr = []
        all_hits1 = []
        all_hits10 = []

        with torch.no_grad():
            all_entities = torch.arange(num_entities, device=device)

            for batch_start in range(0, num_samples, eval_batch_size):
                batch_end = min(batch_start + eval_batch_size, num_samples)
                batch_triples = triples[batch_start:batch_end]
                batch_len = len(batch_triples)

                heads = torch.tensor(batch_triples[:, 0], dtype=torch.long, device=device)
                rels = torch.tensor(batch_triples[:, 1], dtype=torch.long, device=device)
                tails = torch.tensor(batch_triples[:, 2], dtype=torch.long, device=device)

                # Expand for all-entity scoring
                heads_exp = heads.unsqueeze(1).expand(-1, num_entities)
                rels_exp = rels.unsqueeze(1).expand(-1, num_entities)
                all_tails = all_entities.unsqueeze(0).expand(batch_len, -1)

                # Score all tail candidates
                scores = model.forward(
                    heads_exp.reshape(-1),
                    rels_exp.reshape(-1),
                    all_tails.reshape(-1),
                ).reshape(batch_len, num_entities)

                # Get rank of true tail
                true_scores = scores[torch.arange(batch_len, device=device), tails]
                ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1

                all_mrr.append((1.0 / ranks.float()).cpu())
                all_hits1.append((ranks == 1).cpu())
                all_hits10.append((ranks <= 10).cpu())

        mrr_tensor = torch.cat(all_mrr)
        hits1_tensor = torch.cat(all_hits1)
        hits10_tensor = torch.cat(all_hits10)

        return {
            "mrr": mrr_tensor.mean().item(),
            "hits@1": hits1_tensor.float().mean().item(),
            "hits@10": hits10_tensor.float().mean().item(),
        }

    def save_metrics(
        self,
        metrics: dict[str, float],
        filename: str = "metrics.json",
    ) -> Path:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        self.file_manager.save(metrics, output_path)
        logger.info(f"Metricas salvas em: {output_path}")
        return output_path

    def save_training_history(
        self,
        history: list[dict[str, Any]],
        filename: str = "training_history.json",
    ) -> Path:
        """Save training history to JSON file.

        Args:
            history: List of epoch metrics dictionaries.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        self.file_manager.save(history, output_path)
        logger.info(f"Historico de treinamento salvo em: {output_path}")
        return output_path

    def format_metrics_string(self, metrics: dict[str, float]) -> str:
        """Format metrics dictionary as human-readable string.

        Args:
            metrics: Dictionary of metrics.

        Returns:
            Formatted string.
        """
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return " | ".join(parts)
