"""DSLFM Metrics Reporter Component.

Handles metrics computation, reporting, and persistence for DSLFM training.
Extracted from DSLFMManager for Single Responsibility Principle (SRP).

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

from pff.shared import FileManager, logger
from pff.domain.learning.ml.training_observer import TrainingObserver, TrainingEvent


class DSLFMMetricsReporter:
    """Reporter for DSLFM training metrics.

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
        max_eval_memory_bytes: int = 200 * 1024 * 1024,
    ) -> None:
        """Initialize metrics reporter.

        Args:
            output_dir: Directory for saving metrics files.
            file_manager: FileManager instance for I/O operations.
            observers: List of TrainingObserver instances.
            max_eval_memory_bytes: Max memory budget for evaluation batches.
        """
        self.file_manager = file_manager or FileManager()
        self.output_dir = Path(output_dir)
        self.file_manager.ensure_dir(self.output_dir)
        self.observers: list[TrainingObserver] = observers or []
        self.max_eval_memory_bytes = int(max_eval_memory_bytes)

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
            model: DSLFM backbone model to evaluate.
            triples: Triples array of shape [n_triples, 3].
            device: Computation device.
            batch_size: Batch size for evaluation.

        Returns:
            Dictionary with 'mrr', 'hits@1', 'hits@10' metrics.
        """
        if hasattr(model, "evaluate") and callable(model.evaluate):
            eval_triples = torch.as_tensor(triples, device=device, dtype=torch.long)
            metrics = model.evaluate(eval_triples, batch_size=batch_size)
            # Ensure return signature consistency
            return {
                "mrr": metrics.get("mrr", 0.0),
                "hits@1": metrics.get("hits@1", 0.0),
                "hits@10": metrics.get("hits@10", 0.0),
            }

        model.eval()
        num_samples = len(triples)
        num_entities = model.num_entities

        max_batch_by_memory = max(1, self.max_eval_memory_bytes // (num_entities * 4))
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

                heads_exp = heads.unsqueeze(1).expand(-1, num_entities)
                rels_exp = rels.unsqueeze(1).expand(-1, num_entities)
                all_tails = all_entities.unsqueeze(0).expand(batch_len, -1)

                scores = model.forward(
                    heads_exp.reshape(-1),
                    rels_exp.reshape(-1),
                    all_tails.reshape(-1),
                ).reshape(batch_len, num_entities)

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


def compute_structural_metrics(
    community_probs: torch.Tensor | None,
    num_entities: int,
    num_relations: int,
    num_triples: int,
) -> dict[str, float]:
    """Compute structural metrics from latent community probabilities.

    Based on:
    - Entropy calculation from VGAE-ECF (MDPI 2024)
    - Community overlap from soft modularity (Nature 2025)
    - Graph density formula from data_optimizer.py

    Args:
        community_probs: [num_entities, num_communities] soft assignments.
        num_entities: Total entity count.
        num_relations: Total relation count.
        num_triples: Total triple count.

    Returns:
        Dict with latentEntropy, communityOverlap, graphDensity, numClusters.
    """
    if community_probs is None or community_probs.numel() == 0:
        max_possible = num_entities * num_entities * max(1, num_relations)
        graph_density = float(num_triples / max_possible) if max_possible > 0 else 0.0
        return {
            "latentEntropy": 0.0,
            "communityOverlap": 0.0,
            "graphDensity": graph_density,
            "numClusters": 0,
        }

    probs = community_probs.detach().clamp(1e-8, 1.0 - 1e-8)

    entropy_per_entity = -torch.sum(probs * torch.log(probs), dim=-1)
    latent_entropy = float(entropy_per_entity.mean().item())

    # AGENTS.md: Config over hardcoding. Threshold loaded from config.
    from pff.shared.core.config import settings

    soft_threshold = settings.MODEL_CONFIG.get("dslfm", {}).get("community_overlap_threshold", 0.3)
    multi_member = (probs > soft_threshold).sum(dim=-1) > 1
    community_overlap = float(multi_member.float().mean().item())

    max_possible = num_entities * num_entities * max(1, num_relations)
    graph_density = float(num_triples / max_possible) if max_possible > 0 else 0.0

    hard_assign = probs > 0.5
    num_clusters = int((hard_assign.sum(dim=0) > 0).sum().item())

    return {
        "latentEntropy": latent_entropy,
        "communityOverlap": community_overlap,
        "graphDensity": graph_density,
        "numClusters": num_clusters,
    }
