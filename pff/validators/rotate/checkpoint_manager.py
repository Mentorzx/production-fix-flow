"""RotatE Checkpoint Manager.

Handles checkpoint save/load/cleanup operations for RotatE training.
Extracted from RotatEManager for Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Strategy Pattern:** Checkpoint format can be swapped (PyTorch, ONNX).
    - **Factory Pattern:** Creates checkpoint dictionaries.
    - **Memento Pattern:** Captures and restores model state.

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from pff.utils import FileManager, logger


class RotatECheckpointManager:
    """Manager for RotatE model checkpoints.

    Pattern: Manager/Facade
    
    Handles saving, loading, and cleaning up model checkpoints.
    Supports versioned checkpoints and best-model tracking.

    Attributes:
        checkpoint_dir: Directory for storing checkpoints.
        file_manager: FileManager instance for I/O operations.
        keep_top_k: Number of recent checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        file_manager: FileManager | None = None,
        keep_top_k: int = 3,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
            file_manager: FileManager instance for I/O operations.
            keep_top_k: Number of recent checkpoints to keep (default: 3).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.file_manager = file_manager or FileManager()
        self.keep_top_k = keep_top_k

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        filename: str | None = None,
    ) -> Path:
        """Save model checkpoint.

        Args:
            model: PyTorch model to save.
            optimizer: Optimizer state to save (optional).
            epoch: Current epoch number.
            metrics: Dictionary of metrics at checkpoint time.
            is_best: Whether this is the best model so far.
            filename: Custom filename (auto-generated if None).

        Returns:
            Path to saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint salvo: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.success(f"Melhor modelo salvo: {best_path}")

        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        path: Path | str | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into (optional).
            path: Path to checkpoint file (uses best_model.pt if None).
            device: Device to load tensors to.

        Returns:
            Dictionary with checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint carregado: {path} (epoca {checkpoint.get('epoch', '?')})")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "timestamp": checkpoint.get("timestamp", ""),
        }

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def get_best_checkpoint(self) -> Path | None:
        """Get path to the best model checkpoint.

        Returns:
            Path to best checkpoint, or None if it doesn't exist.
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None

    def has_completed_training(self, target_epochs: int) -> tuple[bool, dict[str, Any]]:
        """Check if training was previously completed.

        Args:
            target_epochs: Expected number of training epochs.

        Returns:
            Tuple of (completed: bool, completion_info: dict).
        """
        marker_path = self.checkpoint_dir / "training_completed.json"
        
        if not marker_path.exists():
            return False, {}

        try:
            completion_info = self.file_manager.read(marker_path)
            completed_epochs = completion_info.get("epochs_trained", 0)
            saved_target = completion_info.get("target_epochs", 0)
            
            is_complete = (
                completed_epochs >= saved_target 
                and saved_target == target_epochs
            )
            
            return is_complete, completion_info
        except Exception:
            return False, {}

    def mark_training_completed(
        self,
        epochs_trained: int,
        target_epochs: int,
        best_epoch: int,
        best_val_mrr: float,
        training_time: float,
        final_metrics: dict[str, float],
    ) -> None:
        """Mark training as completed.

        Args:
            epochs_trained: Number of epochs actually trained.
            target_epochs: Target number of epochs.
            best_epoch: Epoch with best validation score.
            best_val_mrr: Best validation MRR achieved.
            training_time: Total training time in seconds.
            final_metrics: Final validation metrics.
        """
        marker_path = self.checkpoint_dir / "training_completed.json"
        
        completion_info = {
            "epochs_trained": epochs_trained,
            "target_epochs": target_epochs,
            "best_epoch": best_epoch,
            "best_val_mrr": best_val_mrr,
            "training_time": training_time,
            "final_metrics": final_metrics,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        self.file_manager.save(completion_info, marker_path)
        logger.info(f"Marcador de conclusao salvo: {marker_path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep best_model.pt and training_completed.json always
        for old_checkpoint in checkpoints[self.keep_top_k:]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Checkpoint antigo removido: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

    def clear_all(self) -> None:
        """Remove all checkpoints and markers.

        Warning: This is destructive and cannot be undone.
        """
        for file in self.checkpoint_dir.glob("*.pt"):
            file.unlink()
        for file in self.checkpoint_dir.glob("*.json"):
            file.unlink()
        logger.info(f"Todos os checkpoints removidos de {self.checkpoint_dir}")
